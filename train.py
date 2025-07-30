import torch

import os
import random


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
torch.set_float32_matmul_precision('high')

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
    entropy_k,
    entropy_tao,
    experiment_id,
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
    self_distillation_factor_pos,
    self_distillation_factor_neg,
    soft_embeds_train_start,
    total_steps,
    train_gc_interval,
)
from sampler import sampler
from utils import cleanup, tokenize

from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOLoss

from forward import model_forward

rank = os.environ['CUDA_VISIBLE_DEVICES']


dapo_lossf = LigerFusedLinearGRPOLoss(
    beta=0.0,
    use_ref_model=False,
    epsilon_low=clip_low,
    epsilon_high=clip_high,
    loss_type='bnpo',
)


def save_model(steps):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        f'./model/checkpoint_{experiment_id}',
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )


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


def norm(x: torch.Tensor, group_size: int) -> torch.Tensor:
    x = x.view(-1, group_size)
    x = x - x.mean()
    return (x / (x**2).mean() * 0.5).view(-1)


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
                ) = concept_mask = monitored_entropy = None
                concept_temperature_ = min(
                    concept_temperature_max,
                    concept_temperature
                    + (concept_temperature_max - concept_temperature)
                    * step
                    / concept_temperature_increase_step,
                )
                for i in range(
                    0, input_ids.shape[0] // sample_num, sample_problem_sub_batch
                ):
                    end = i + sample_problem_sub_batch
                    end = end if end <= input_ids.shape[0] else input_ids.shape[0]
                    if init_res:
                        (
                            res_,
                            res_probs_,
                            text_end_indices_,
                            mask_,
                            concept_token_probs_,
                            concept_token_indices_,
                            concept_mask_,
                            monitored_entropy_,
                        ) = sampler(
                            input_ids[i * sample_num : end * sample_num],
                            problem_attn_mask[i * sample_num : end * sample_num],
                            topk=sample_topk,
                            max_length=max_sample_length,
                            temperature=sample_temperature,
                            concept_temperature=concept_temperature_,
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
                        monitored_entropy += monitored_entropy_
                    else:
                        (
                            res,
                            res_probs,
                            text_end_indices,
                            mask,
                            concept_token_probs,
                            concept_token_indices,
                            concept_mask,
                            monitored_entropy,
                        ) = sampler(
                            input_ids[: sample_problem_sub_batch * sample_num],
                            problem_attn_mask[: sample_problem_sub_batch * sample_num],
                            topk=sample_topk,
                            max_length=max_sample_length,
                            concept_temperature=concept_temperature_,
                            entropy_k=entropy_k,
                            entropy_tao=entropy_tao,
                        )
                        init_res = True

                    cleanup()

                res = res.clone()
                res_probs = res_probs.clone()
                concept_token_probs = concept_token_probs.clone()
                concept_token_indices = concept_token_indices.clone()
                text_end_indices = text_end_indices.clone()
                mask = mask.clone()
                concept_mask = concept_mask.clone()
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

                print(rank, 'correctness rate: ', correctness_rate)
                print(
                    rank, 'average length: ', text_end_indices.float().mean().item() + 1
                )
                writer.add_scalar('correctness_rate/train', correctness_rate, step)
                writer.add_scalar(
                    'length/train', text_end_indices.float().mean().item() + 1, step
                )
                writer.add_scalar(
                    'correct_length/train',
                    (
                        (
                            text_end_indices[correctness_rewards == corr_reward]
                            .float()
                            .mean()
                            .item()
                            + 1.0
                        )
                        if (correctness_rewards == corr_reward).any()
                        else max_sample_length
                    ),
                    step,
                )
                writer.add_text('sampled_text/train', decoded[0], step)
                writer.add_scalar(
                    'entropy/train',
                    monitored_entropy * sample_problem_sub_batch / sample_problem_batch,
                    step,
                )
                writer.add_scalar(
                    'rewards/train', correctness_rewards.float().mean().item(), step
                )

                # normalize rewards
                correctness_rewards = norm(correctness_rewards, group_size=sample_num)

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
                            if soft_embeds_train_start <= step:
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
                                        soft_embeds
                                        * concept_mask[i:end, :-1].unsqueeze(-1)
                                        + original_embeds
                                        * (1 - concept_mask[i:end, :-1]).unsqueeze(-1),
                                    ],
                                    dim=1,
                                )
                            else:
                                embeds = problem_embeds

                            hidden = model_forward(
                                hidden_state=embeds,
                                attn_mask=mask[i:end, :-1],
                                apply_lm_head=False,
                                pos=torch.arange(0, embeds.shape[1])
                                .long()
                                .to(embeds.device),
                            )[:, input_ids.shape[1] - 1 :]

                            # compute DAPO loss
                            target = res[i:end, :]
                            target[target >= model.lm_head.weight.shape[-1]] = 0
                            loss = dapo_lossf(
                                hidden,
                                model.lm_head.weight,
                                target,
                                mask[i:end, input_ids.shape[1] :],
                                rewards[i:end],
                                ref_per_token_logps=res_probs[i:end],
                            )[0] * (-1)
                            if (
                                self_distillation_factor_pos > 0
                                and soft_embeds_train_start <= step
                            ):
                                """
                                matches = shrunk_indices.unsqueeze(
                                    -2
                                ) == concept_token_indices[i:end].unsqueeze(-1)
                                has_match = matches.any(dim=-1)
                                matched_indices = matches.short().argmax(dim=-1)
                                new_concept_probs = (
                                    torch.log_softmax(
                                        shrunk_logits / concept_temperature,
                                        dim=-1,
                                    ).gather(-1, matched_indices)
                                    * has_match.float()
                                )
                                new_concept_probs -= torch.log(
                                    torch.exp(new_concept_probs).sum(
                                        dim=-1, keepdim=True
                                    )
                                )
                                """
                                concept_temperature_ = min(
                                    concept_temperature_max,
                                    concept_temperature
                                    + (concept_temperature_max - concept_temperature)
                                    * step
                                    / concept_temperature_increase_step,
                                )
                                logits = model.lm_head(hidden)
                                new_concept_probs = torch.log_softmax(
                                    logits.gather(-1, concept_token_indices[i:end])
                                    / concept_temperature_,
                                    dim=-1,
                                )
                                self_distillation_loss = kl_divergence(
                                    concept_token_probs[i:end],
                                    new_concept_probs,
                                )
                                self_distillation_loss *= concept_mask[i:end]
                                self_distillation_factor = torch.abs(rewards[i:end])
                                self_distillation_factor[rewards[i:end] > 0] *= (
                                    self_distillation_factor_pos
                                )
                                self_distillation_factor[rewards[i:end] < 0] *= (
                                    self_distillation_factor_neg
                                )
                                self_distillation_loss = (
                                    self_distillation_loss.sum(dim=-1)
                                    / (concept_mask[i:end].float().sum() + 1e-10)
                                ) * self_distillation_factor
                                self_distillation_loss = self_distillation_loss.sum()
                                loss += self_distillation_loss

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

                print(rank, f'Step {step}, Loss: {loss.item():.5f}')

                cleanup()

            if step % save_interval == 0:
                print(rank, f'Saving model at step {step}')
                save_model(step)

            if step > total_steps:
                break

    # Save checkpoint
    if step % save_interval != 0:
        print(rank, f'Saving model at step {step}')
        save_model(step)

    writer.close()
    print('all done')


if __name__ == '__main__':
    train()
