import torch
import torch.nn.functional as F
from config import device, ensure_tensor_type, tensor_concat
from data import data_train, verifier
from model import (
    model,
    writer,
    accelerator,
    vae,
    gater,
    lossf,
    optimizers,
    gater_scheduler,
    tokenizer,
    hidden_regularizer,
)
from parameters import (
    max_train_length,
    acc_check_only,
    max_sample_length,
    l_cache_length,
    sample_num,
    sample_problem_batch,
    corr_reward,
    looping_depth,
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
)
from forward import model_forward
from sampler import sampler
from utils import cleanup


def save_model(steps):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        f"./model/model-{steps}",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    accelerator.save_model(model, f"./model/model-{steps}")
    accelerator.save_model(vae, f"./model/vae-{steps}")
    accelerator.save_model(gater, f"./model/gater-{steps}")


def step_optimizer():
    for optim in optimizers:
        optim.step()
    gater_scheduler.step()


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

    while step <= total_steps:
        # Set epoch for distributed sampler
        if hasattr(data_train, "sampler") and hasattr(data_train.sampler, "set_epoch"):
            data_train.sampler.set_epoch(step // len(data_train))

        for input_ids, problem_attn_mask, ans in data_train:
            input_ids = input_ids.to(device)
            problem_attn_mask = problem_attn_mask.to(device)
            cleanup()
            with torch.no_grad():
                init_res = False

                # --- decentralized sampling ---

                for i in range(0, sample_problem_batch, sample_problem_sub_batch):
                    if init_res:
                        res_, hidden_cache_, text_end_indices_, mask_ = sampler(
                            input_ids[
                                i * sample_num : (i + sample_problem_sub_batch)
                                * sample_num
                            ],
                            problem_attn_mask[
                                i * sample_num : (i + sample_problem_sub_batch)
                                * sample_num
                            ],
                            num=sample_num,
                            topk=sample_topk,
                            max_length=max_sample_length,
                            temperature=sample_temperature,
                            depth=looping_depth,
                        )
                        res = tensor_concat(res, res_)
                        hidden_cache = tensor_concat(hidden_cache, hidden_cache_)
                        text_end_indices = tensor_concat(
                            text_end_indices, text_end_indices_
                        )
                        mask = tensor_concat(mask, mask_)
                    else:
                        res, hidden_cache, text_end_indices, mask = sampler(
                            input_ids[: sample_problem_sub_batch * sample_num],
                            problem_attn_mask[: sample_problem_sub_batch * sample_num],
                            num=sample_num,
                            topk=sample_topk,
                            max_length=max_sample_length,
                            depth=looping_depth,
                        )
                        init_res = True

                    cleanup()

                # --- end of decentralized sampling ---
                # --- centralized validating ---

                if res.shape[1] > max_train_length:
                    seqs = torch.cat([input_ids, res[:, :max_train_length]], dim=1)
                else:
                    seqs = torch.cat([input_ids, res], dim=1)

                res = accelerator.gather(res)
                hidden_cache = accelerator.gather(hidden_cache)
                text_end_indices = accelerator.gather(text_end_indices)
                mask = accelerator.gather(mask)
                seqs = accelerator.gather(seqs)

                if accelerator.is_main_process:
                    hidden_cache = hidden_cache[:, :-1]

                    correctness_rewards = torch.Tensor(
                        verifier(
                            tokenizer.batch_decode(res, skip_special_tokens=True),
                            ans,
                            corr_score=corr_reward,
                        )
                    ).to(device)
                    accelerator.print(tokenizer.decode(res[0]))
                    len_rewards = text_end_indices.float() + 1

                    filt = None
                    if (
                        l := (corr_filt := correctness_rewards == corr_reward).sum()
                    ) < res.shape[
                        0
                    ] / 3 and l != 0:  # clip too many wrong answers, currently 1:1
                        incorr_filt = torch.ones(sample_num * sample_problem_batch).to(
                            device
                        )
                        incorr_filt[correctness_rewards == 1] = 0
                        incorr_filt = torch.multinomial(
                            incorr_filt, num_samples=(l * 2).item()
                        )
                        filt = torch.cat(
                            [torch.nonzero(corr_filt, as_tuple=True)[0], incorr_filt],
                            dim=0,
                        )
                        filt = filt[torch.randperm(filt.size(0))]

                        torch.distributed.broadcast(filt, 0)

                        correctness_rewards = correctness_rewards[filt]
                        len_rewards = len_rewards[filt]
                        mask = mask[filt]
                        input_ids = input_ids[filt]
                        hidden_cache = hidden_cache[filt]
                        res = res[filt]
                        text_end_indices = text_end_indices[filt]

                    correctness = l.cpu().item()
                    if correctness == 0:
                        accelerator.print("NG. Re")
                        continue
                    accelerator.print("correctness: ", correctness)
                    writer.add_scalar("correctness/train", correctness, step)

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

                if acc_check_only:
                    continue

                with torch.no_grad():
                    if res.shape[1] > max_train_length:
                        hidden_cache = hidden_cache[:, : max_train_length - 1]

            # --- end of centralized validating ---

            # training
            accelerator.print("start training")
            model.train()
            vae.train()
            gater.train()
            for epoch in range(num_epochs):
                accelerator.print("training epoch: ", epoch + 1)
                cleanup()
                for i in range(0, res.shape[0], batch_size):
                    with accelerator.accumulate(model, vae, gater):
                        if step % train_gc_interval == 0:
                            cleanup()
                        end = (
                            (i + batch_size)
                            if i + batch_size <= res.shape[0]
                            else res.shape[0]
                        )
                        embeds = model.module.lm_head.weight[seqs[i:end]][:, :-1].to(
                            device
                        )
                        hidden_cache_slice = hidden_cache[i:end]
                        with accelerator.autocast():
                            last_hidden = vae.module.uncompress(
                                F.dropout(
                                    hidden_cache_slice,
                                    p=hidden_dropout_rate,
                                    training=True,
                                )
                            )
                            if looping_depth > 0:  # deep looping
                                hidden_pos = torch.arange(
                                    0,
                                    last_hidden.shape[1],
                                    dtype=torch.long,
                                    device=device,
                                )
                                for depth_i in range(looping_depth):
                                    for layer_i in range(
                                        depth_start_layer_num, hidden_layer_num
                                    ):
                                        last_hidden = model_forward(
                                            hidden_state=last_hidden,
                                            attn_mask=mask[
                                                i:end, input_ids.shape[1] - 1 : -1
                                            ],
                                            pos=hidden_pos,
                                            start_layer=depth_start_layer_num,
                                            end_layer=hidden_layer_num,
                                        )

                            # Get the parts we need to concatenate
                            embeds_first_part = embeds[:, : input_ids.shape[1]]
                            embeds_second_part = gater(
                                last_hidden, embeds[:, input_ids.shape[1] :]
                            )

                            # Ensure compatible tensor types for concatenation
                            embeds = tensor_concat(
                                embeds_first_part,
                                embeds_second_part,
                                dim=1,
                            )
                            final_hidden, hidden = model_forward(
                                hidden_state=embeds,
                                attn_mask=mask[i:end, :-1],
                                pos=torch.arange(
                                    seqs.shape[1] - 1, dtype=torch.long, device=device
                                ),
                                extract_specific=hidden_layer_num,
                            )
                            logits = model.module.lm_head(
                                model.module.model.model.norm(final_hidden)
                            ).float()
                            loss = lossf(
                                logits[:, input_ids.shape[1] - 1 :].transpose(1, 2),
                                seqs[i:end, input_ids.shape[1] :].masked_fill(
                                    mask[i:end, input_ids.shape[1] :] == 0, -100
                                ),
                            )

                            # compute loss
                            new_compressed_hidden = vae(
                                hidden[:, input_ids.shape[1] - 1 : -1], compressing=True
                            )
                            new_processed_hidden = vae.module.uncompress(
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

                            new_processed_hidden = gater.module.forward_hidden(
                                new_processed_hidden, embeds[:, input_ids.shape[1] :]
                            )
                            processed_hidden = gater.module.forward_hidden(
                                last_hidden, embeds[:, input_ids.shape[1] :]
                            )
                            hidden_loss = (
                                hidden_regularizer(
                                    new_processed_hidden, processed_hidden
                                ).mean(dim=-1)
                                * mask[i:end, input_ids.shape[1] : -1]
                            )
                            # apply hidden regularization bonus
                            hidden_loss = hidden_loss * linear_interpl(
                                (text_end_indices + 1)[i : i + batch_size],
                                hidden_reg_len_bonus_a,
                                max_sample_length,
                                1,
                                hidden_reg_len_bonus_high,
                            )
                            # apply gating value bonus
                            gate_bonus = torch.exp(
                                gating_value_lambda * (0.5 - gater.module.gate) ** 2
                            ).mean()
                            loss = (
                                (loss.sum(dim=-1) * rewards[i:end]).sum()
                                + hidden_loss.sum() * hidden_regularization_rate
                            ) / (text_end_indices[i : i + batch_size] + 1).sum()
                            loss += (
                                gate_bonus
                                * gating_value_bonus
                                * gating_value_decay
                                ** (step // gating_bonus_update_step)
                            )
                            # loss *= batch_size / res.shape[0]
                            loss *= batch_size

                            # randomly update hidden cache
                            with torch.no_grad():
                                update_index = torch.nonzero(
                                    torch.randn(hidden_cache.shape[1], device=device)
                                    < hidden_updating_rate
                                )
                                hidden_cache[i:end][:, update_index] = (
                                    new_compressed_hidden[:, update_index]
                                )

                        accelerator.backward(loss)

                step_optimizer()
                zero_grad_optimizer()

                accelerator.print(f"Step {step}, Loss: {loss.item():.3f}")
                gater.module.print_gates()

                step += 1

                cleanup()

        # Save checkpoint
        if step % save_interval == 0:
            save_model(step)

    writer.close()
    accelerator.print("all done")


if __name__ == "__main__":
    train()
