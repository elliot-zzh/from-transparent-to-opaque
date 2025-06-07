import torch
from tqdm import tqdm
import torch.nn.functional as F
from config import device
from utils import tokenize
from model import (
    model,
    writer,
    accelerator,
    gater,
    optimizers,
    # gater_scheduler,
    tokenizer,
    hidden_regularizer,
)
from data import data_train, verifier
from parameters import (
    max_train_length,
    acc_check_only,
    max_sample_length,
    l_cache_length,
    sample_num,
    sample_problem_batch,
    corr_reward,
    looping_depth,
    concept_temperature,
    concept_temperature_increase_step,
    concept_temperature_max,
    concept_topk,
    sample_temperature,
    sample_topk,
    sample_problem_sub_batch,
    total_steps,
    batch_size,
    hidden_dropout_rate,
    depth_start_layer_num,
    hidden_layer_num,
    train_gc_interval,
    num_epochs,
    hidden_reg_len_bonus_a,
    hidden_reg_len_bonus_high,
    gating_value_lambda,
    hidden_regularization_rate,
    gating_value_bonus,
    gating_value_decay,
    gating_bonus_update_step,
    hidden_updating_rate,
    save_interval,
    clip_high,
    clip_low,
    gradient_accumulation_steps,
    enable_hidden_regularization,
    enable_length_reg_bonus,
    enable_gating_bonus,
    enable_hidden_updating,
    gating_bonus_mode,
    enable_gating,
)
from sampler import sampler
from utils import cleanup
import os

rank = os.environ['CUDA_VISIBLE_DEVICES']


def save_model(steps):
    accelerator.save_model(model, f'./model/rank-{rank}-model-{steps}')
    accelerator.save_model(vae, f'./model/rank-{rank}-vae-{steps}')
    accelerator.save_model(gater, f'./model/rank-{rank}-gater-{steps}')


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


linear_interpl = torch.jit.script(linear_interpl)
norm = torch.jit.script(norm)


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
                for i in range(0, sample_problem_batch, sample_problem_sub_batch):
                    concept_temperature_ = concept_temperature * min(concept_temperature_max, concept_temperature + (concept_temperature_max - concept_temperature) / concept_temperature_increase_step * step)
                    if init_res:
                        (
                            res_,
                            res_probs_,
                            text_end_indices_,
                            mask_,
                            concept_token_probs_,
                            concept_token_indices_,
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
                            concept_temperature=concept_temperature_,
                            concept_topk=concept_topk,
                        )
                        res = torch.cat([res, res_], dim=0)
                        res_probs = torch.cat([res, res_probs_], dim=0)
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
                    else:
                        (
                            res,
                            res_probs,
                            text_end_indices,
                            mask,
                            concept_token_probs,
                            concept_token_indices,
                        ) = sampler(
                            input_ids[: sample_problem_sub_batch * sample_num],
                            problem_attn_mask[: sample_problem_sub_batch * sample_num],
                            topk=sample_topk,
                            max_length=max_sample_length,
                            concept_temperature=concept_temperature_,
                            concept_topk=concept_topk,
                        )
                        init_res = True

                    cleanup()

                correctness_rewards = torch.Tensor(
                    verifier(
                        tokenizer.batch_decode(res, skip_special_tokens=True),
                        ans,
                        corr_score=corr_reward,
                    )
                ).to(device)
                print(rank, tokenizer.batch_decode(res, skip_special_tokens=True))
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
                rewards = correctness_rewards + len_rewards
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

                # attention: the end are all trauncated
                concept_token_probs = concept_token_probs[:, :-1]
                concept_token_indices = concept_token_indices[:, :-1]
                res_probs = res_probs[:, 0:]

            if acc_check_only:
                continue

            # training
            print(rank, 'start training')
            model.train()
            gater.train()

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
                            embeds = model.model.model.embed_tokens(input_ids[i:end])
                            soft_embeds = (
                                model.model.model.embed_tokens(
                                    concept_token_indices[i:end]
                                ).transpose(-2, -1)
                                * concept_token_probs[i:end].unsqueeze(-2)
                            ).sum(dim=-1)
                            if enable_gating:
                                soft_embeds = gater(
                                    soft_embeds, model.model.model.embed_tokens(res[i:end, :-1])
                                )
                            embeds = torch.cat(
                                [
                                    embeds[:, : input_ids.shape[1]],
                                    soft_embeds,
                                ],
                                dim=1,
                            )
                            logits = model.model.forward(
                                inputs_embeds=embeds,
                                attention_mask=mask[i:end, :-1],
                                use_cache=False,
                                output_hidden_states=False,
                                return_dict=True,
                            ).logits.float()

                            """
                            loss = lossf(
                                logits[:, input_ids.shape[1] - 1 :].transpose(1, 2),
                                seqs[i:end, input_ids.shape[1] :].masked_fill(
                                    mask[i:end, input_ids.shape[1] :] == 0, -100
                                ),
                            )
                            """

                            # compute loss
                            target = res[i:end, 0:]
                            target[target >= logits.shape[-1]] = 0
                            new_probs = (
                                F.log_softmax(
                                    logits[:, input_ids.shape[1] - 1 :], dim=-1
                                )
                                .gather(-1, target.unsqueeze(-1))
                                .squeeze(-1)
                            )
                            loss = torch.exp(new_probs - res_probs[i:end])
                            clipped = torch.clamp(loss, 1 - clip_high, 1 + clip_low)
                            clipped *= rewards[i:end].unsqueeze(-1)
                            loss = torch.min(
                                loss * rewards[i:end].unsqueeze(-1), clipped
                            )
                            loss *= mask[i:end, input_ids.shape[1] :]
                            dapo_loss = loss = (
                                (loss.sum(dim=-1))
                                / (text_end_indices[i:end] + 1).sum()
                                * (
                                    -1
                                )  # here we want to maximaize it, aligned with DAPO target
                            )

                            # compute hidden regularization
                            """
                            new_compressed_hidden = vae(
                                hidden[:, input_ids.shape[1] - 1 : -1], compressing=True
                            )
                            if enable_hidden_regularization:
                                new_processed_hidden = vae.uncompress(
                                    new_compressed_hidden
                                )
                                if looping_depth > 0:  # deep looping
                                    for depth_i in range(looping_depth):
                                        for layer_i in range(
                                            depth_start_layer_num, hidden_layer_num
                                        ):
                                            last_hidden = model_forward(
                                                hidden_state=new_processed_hidden,
                                                attn_mask=mask[
                                                    i:end, input_ids.shape[1] - 1 : -1
                                                ],
                                                pos=hidden_pos,
                                                start_layer=depth_start_layer_num,
                                                end_layer=hidden_layer_num,
                                            )

                                new_processed_hidden, gate = gater.forward_hidden(
                                    new_processed_hidden,
                                    embeds[:, input_ids.shape[1] :],
                                )
                                processed_hidden, _ = gater.forward_hidden(
                                    last_hidden, embeds[:, input_ids.shape[1] :]
                                )
                                hidden_loss = (
                                    hidden_regularizer(
                                        new_processed_hidden, processed_hidden
                                    ).mean(dim=-1)
                                    * mask[i:end, input_ids.shape[1] : -1]
                                )
                                if enable_length_reg_bonus:
                                    hidden_loss = hidden_loss * linear_interpl(
                                        (text_end_indices + 1)[i:end],
                                        hidden_reg_len_bonus_a,
                                        max_sample_length,
                                        1,
                                        hidden_reg_len_bonus_high,
                                    ).unsqueeze(1)
                                hidden_loss = (
                                    hidden_loss.mean() * hidden_regularization_rate
                                )
                                loss += hidden_loss
                                """

                            if False:
                                # apply gating value bonus
                                gate_bonus = (
                                    torch.exp(
                                        gating_value_lambda
                                        * (0.5 - torch.abs(gate)) ** 2
                                    ).mean()
                                    if gating_bonus_mode == 'exp'
                                    else -1 * (gate**2).mean()
                                )
                                gate_bonus = (
                                    gate_bonus
                                    * gating_value_bonus
                                    * gating_value_decay
                                    ** (step // gating_bonus_update_step)
                                )
                                loss += gate_bonus

                            loss *= (end - i) / gradient_accumulation_steps

                            # randomly update hidden cache
                            if False:
                                with torch.no_grad():
                                    update_index = torch.nonzero(
                                        torch.randn(
                                            hidden_cache.shape[1], device=device
                                        )
                                        < hidden_updating_rate
                                    )
                                    hidden_cache[i:end][:, update_index] = (
                                        new_compressed_hidden[:, update_index]
                                    )

                        accelerator.backward(loss)
                        accumulated_steps += 1

                        if accumulated_steps % gradient_accumulation_steps == 0:
                            step_optimizer()
                            zero_grad_optimizer()

                if accumulated_steps % gradient_accumulation_steps != 0:
                    step_optimizer()
                    zero_grad_optimizer()

                print(rank, f'Step {step}, Loss: {loss.item():.3f}')
                writer.add_scalar('loss/train', loss.item(), step)

                step += 1

                cleanup()

        # Save checkpoint
        if step % save_interval == 0:
            save_model(step)

    writer.close()
    print('all done')


if __name__ == '__main__':
    train()
