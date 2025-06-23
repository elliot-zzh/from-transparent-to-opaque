import torch

torch.manual_seed(42)
import os

import torch.nn.functional as F
from tqdm import tqdm

from config import device
from data import data_train, verifier
from model import (
    accelerator,
    model,
    optimizers,
    tokenizer,
    writer,
)
from parameters import (
    acc_check_only,
    batch_size,
    clip_high,
    clip_low,
    concept_temperature,
    concept_temperature_increase_step,
    concept_temperature_max,
    corr_reward,
    enable_swapping,
    entropy_k,
    entropy_tao,
    gradient_accumulation_steps,
    l_cache_length,
    max_sample_length,
    max_train_length,
    num_epochs,
    sample_num,
    sample_problem_batch,
    sample_problem_sub_batch,
    sample_temperature,
    sample_topk,
    save_interval,
    self_distillation_factor,
    total_steps,
    train_gc_interval,
)
from sampler import sampler
from utils import cleanup, tokenize

rank = os.environ['CUDA_VISIBLE_DEVICES']


def save_model(steps):  # TODO: save checkpoints?
    accelerator.save_model(model, f'./model/rank-{rank}-model-{steps}')


def step_optimizer():
    for optim in optimizers:
        optim.step()
    # gater_scheduler.step()


def zero_grad_optimizer():
    for optim in optimizers:
        optim.zero_grad(set_to_none=True)


def linear_interpl(
    x: torch.Tensor, a: float, b: float, low: float, high: float
) -> torch.Tensor:  # only interpl between [a, b], linearly increase from low to high
    mask_low = x <= a
    mask_high = x >= b

    t = (x - a) / (b - a)
    mid_values = low + (high - low) * t

    return torch.where(mask_low, low, torch.where(mask_high, high, mid_values))


def norm(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean()
    return x / (x**2).mean() * 0.5


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10):
    return (p * (torch.log(p + eps) - q)).sum(dim=-1)


linear_interpl = torch.jit.script(linear_interpl)
norm = torch.jit.script(norm)
kl_divergence = torch.jit.script(kl_divergence)


def train():
    step = 0
    init_res = False

    while step <= total_steps:
        for problems, ans in data_train:
            input_ids, problem_attn_mask = tokenize(problems, direct=True)
            if input_ids.shape[1] >= max_train_length:
                continue  # skip too long problems

            input_ids = input_ids.to(device)
            problem_attn_mask = problem_attn_mask.to(device)
            cleanup()
            with torch.no_grad():
                res = res_probs = text_end_indices = mask = concept_token_probs = (
                    concept_token_indices
                ) = concept_mask = None
                for i in range(0, sample_problem_batch, sample_problem_sub_batch):
                    if init_res:
                        (
                            res_,
                            res_probs_,
                            text_end_indices_,
                            mask_,
                            concept_token_probs_,
                            concept_token_indices_,
                            concept_mask_,
                        ) = sampler(
                            input_ids[
                                i * sample_num : (i + sample_problem_sub_batch)
                                * sample_num
                            ],
                            problem_attn_mask[
                                i * sample_num : (i + sample_problem_sub_batch)
                                * sample_num
                            ],
                            topk=sample_topk,
                            max_length=max_sample_length,
                            temperature=sample_temperature,
                            concept_temperature=concept_temperature,
                            entropy_k=entropy_k,
                            entropy_tao=entropy_tao,
                        )
                        res = torch.cat([res, res_], dim=0)
                        res_probs = torch.cat([res_probs, res_probs_], dim=0)
                        concept_token_probs = torch.cat(
                            [concept_token_probs, concept_token_probs_], dim=0
                        )
                        concept_token_indices = torch.cat(
                            [concept_token_indices, concept_token_indices_], dim=0
                        )
                        text_end_indices = torch.cat(
                            [text_end_indices, text_end_indices_], dim=0
                        )
                        mask = torch.cat([mask, mask_], dim=0)
                        concept_mask = torch.cat([concept_mask, concept_mask_], dim=0)
                    else:
                        (
                            res,
                            res_probs,
                            text_end_indices,
                            mask,
                            concept_token_probs,
                            concept_token_indices,
                            concept_mask,
                        ) = sampler(
                            input_ids[: sample_problem_sub_batch * sample_num],
                            problem_attn_mask[: sample_problem_sub_batch * sample_num],
                            topk=sample_topk,
                            max_length=max_sample_length,
                            concept_temperature=concept_temperature,
                            entropy_k=entropy_k,
                            entropy_tao=entropy_tao,
                        )
                        init_res = True

                    cleanup()

                decoded = tokenizer.batch_decode(res, skip_special_tokens=True)
                correctness_rewards = torch.Tensor(
                    verifier(
                        decoded,
                        ans,
                        corr_score=corr_reward,
                    )
                ).to(device)

                len_rewards = text_end_indices.float() + 1
                l = (corr_filt := correctness_rewards == corr_reward).sum()

                # if l < 10:
                #     continue

                init_res = False

                correctness_rate = l.cpu().item() / res.shape[0]
                """
                if (
                    l < res.shape[0] / 3 and l != 0
                ):  # clip too many wrong answers, currently 1:2
                    incorr_filt = torch.ones_like(correctness_rewards).to(device)
                    incorr_filt[correctness_rewards == corr_reward] = 0
                    incorr_filt = torch.multinomial(incorr_filt, num_samples=l * 2)
                    filt = torch.cat(
                        [torch.nonzero(corr_filt, as_tuple=True)[0], incorr_filt], dim=0
                    )
                    filt = filt[torch.randperm(filt.size(0))]
                    correctness_rewards = correctness_rewards[filt]
                    len_rewards = len_rewards[filt]
                    mask = mask[filt]
                    input_ids = input_ids[filt]
                    hidden_cache = hidden_cache[filt]
                    res = res[filt]
                    text_end_indices = text_end_indices[filt]
                    res_probs = res_probs[filt]
                """

                shuffle_index = torch.randperm(res.shape[0])
                res = res[shuffle_index]
                mask = mask[shuffle_index]
                res_probs = res_probs[shuffle_index]
                text_end_indices = text_end_indices[shuffle_index]
                input_ids = input_ids[shuffle_index]
                len_rewards = len_rewards[shuffle_index]
                correctness_rewards = correctness_rewards[shuffle_index]
                concept_token_probs = concept_token_probs[shuffle_index]
                concept_token_indices = concept_token_indices[shuffle_index]

                print(rank, 'correctness rate: ', correctness_rate)
                writer.add_scalar('correctness_rate/train', correctness_rate, step)

                # reward normalization to get advantage
                max_len_mask = len_rewards >= max_sample_length
                if max_len_mask.any():
                    len_rewards[len_rewards >= max_sample_length] = -1
                cache_len_mask = len_rewards <= l_cache_length
                if cache_len_mask.any():
                    len_rewards[len_rewards <= l_cache_length] = 0
                len_interval_mask = torch.logical_not(
                    torch.logical_or(max_len_mask, cache_len_mask)
                )
                if len_interval_mask.any():
                    len_rewards[len_interval_mask] = (
                        l_cache_length - len_rewards[len_interval_mask]
                    ) / (max_sample_length - l_cache_length)
                # rewards = correctness_rewards + len_rewards # currently remove length penalty
                rewards = correctness_rewards
                rewards = norm(rewards)

                # truncate to max_train_length if the sampled result is too long
                if res.shape[1] + input_ids.shape[1] > max_train_length:
                    res = res[:, : max_train_length - input_ids.shape[1]]
                    concept_token_indices = concept_token_indices[
                        :, : max_train_length - input_ids.shape[1]
                    ]
                    concept_token_probs = concept_token_probs[
                        :, : max_train_length - input_ids.shape[1]
                    ]
                    mask = mask[:, :max_train_length]
                    res_probs = res_probs[:, : max_train_length - input_ids.shape[1]]
                    concept_mask = concept_mask[
                        :, : max_train_length - input_ids.shape[1]
                    ]

            if acc_check_only:
                continue

            # training
            print(rank, 'start training')
            model.train()

            accumulated_steps = 0

            for epoch in range(num_epochs):
                cleanup()
                for i in tqdm(
                    range(0, res.shape[0], batch_size),
                    desc=f'training epoch: {epoch + 1}',
                ):
                    if True:
                        if step % train_gc_interval == 0:
                            cleanup()
                        end = (
                            (i + batch_size)
                            if i + batch_size <= res.shape[0]
                            else res.shape[0]
                        )
                        with accelerator.autocast():
                            problem_embeds = model.model.model.embed_tokens(
                                input_ids[i:end]
                            )
                            soft_embeds = (
                                model.model.model.embed_tokens(
                                    concept_token_indices[i:end, :-1]
                                ).transpose(-2, -1)
                                * concept_token_probs[i:end, :-1].unsqueeze(-2)
                            ).sum(dim=-1)
                            original_embeds = model.model.model.embed_tokens(
                                res[i:end, :-1]
                            )
                            embeds = torch.cat(
                                [
                                    problem_embeds[:, : input_ids.shape[1]],
                                    soft_embeds * concept_mask[i:end, :-1].unsqueeze(-1)
                                    + original_embeds
                                    * (1 - concept_mask[i:end, :-1]).unsqueeze(-1),
                                ],
                                dim=1,
                            )
                            logits = model.model.forward(
                                inputs_embeds=embeds,
                                attention_mask=mask[i:end, :-1],
                                use_cache=False,
                                output_hidden_states=False,
                                return_dict=True,
                            ).logits
                            shrunk_logits, shrunk_indices = torch.topk(
                                logits[:, input_ids.shape[1] - 1 :],
                                k=128,
                                dim=-1,
                                largest=True,
                                sorted=False,
                            )
                            shrunk_logits = shrunk_logits.float()

                            # compute DAPO loss
                            target = res[i:end, :]
                            target[target >= logits.shape[-1]] = 0
                            new_probs = (
                                F.log_softmax(shrunk_logits, dim=-1)
                                * (shrunk_indices == target.unsqueeze(-1)).float()
                            ).sum(dim=-1)
                            loss = torch.exp(new_probs - res_probs[i:end, :])
                            clipped = torch.clamp(loss, 1 - clip_high, 1 + clip_low)
                            clipped *= rewards[i:end].unsqueeze(-1)
                            loss = torch.min(
                                loss * rewards[i:end].unsqueeze(-1), clipped
                            )
                            loss *= mask[i:end, input_ids.shape[1] :]
                            loss = (
                                (loss.sum(dim=-1))
                                / (text_end_indices[i:end] + 1).sum()
                                * (
                                    -1
                                )  # here we want to maximaize it, aligned with DAPO target
                            )
                            if self_distillation_factor > 0:
                                self_distillation_loss = kl_divergence(
                                    concept_token_probs[i:end, :],
                                    torch.log_softmax(
                                        logits[:, input_ids.shape[1] - 1 :]
                                        / concept_temperature,
                                        dim=-1,
                                    ).gather(-1, concept_token_indices[i:end, :]),
                                )
                                self_distillation_loss *= concept_mask[i:end, :]
                                self_distillation_loss = (
                                    self_distillation_loss.sum(dim=-1)
                                    / concept_mask[i:end, :].sum()
                                ) * rewards[i:end]
                                self_distillation_loss = self_distillation_loss.sum()
                                loss += (
                                    self_distillation_factor * self_distillation_loss
                                )

                        accelerator.backward(loss)
                        accumulated_steps += 1

                        if accumulated_steps % gradient_accumulation_steps == 0:
                            step_optimizer()
                            zero_grad_optimizer()
                            step += 1

                if accumulated_steps % gradient_accumulation_steps != 0:
                    step_optimizer()
                    zero_grad_optimizer()
                    step += 1

                print(rank, f'Step {step}, Loss: {loss.item():.8f}')
                writer.add_scalar(
                    'length/train', text_end_indices.float().mean().item() + 1, step
                )
                writer.add_text('sample_text/train', decoded[0], step)

                cleanup()

        # Save checkpoint
        if step % save_interval == 0:
            save_model(step)

    writer.close()
    print('all done')


if __name__ == '__main__':
    train()
