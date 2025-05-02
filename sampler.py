# A lot of hacking here. For details refer to
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/
from config import device
from model import model, vae, gater, accelerator
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache
from tqdm import tqdm

from parameters import hidden_layer_num, depth_start_layer_num, hidden_dropout_rate
from model import im_end, eot
from forward import model_forward
from utils import cleanup


def sampler(
    input_ids,
    attn_mask,
    temperature=0.7,
    topk=16,
    max_length=2048,
    num=16,
    gc_interval=64,
    depth=0,
):
    model.eval()
    vae.eval()
    gater.eval()

    # tokenize
    problem_batch_size = input_ids.shape[0]
    cache_pos = torch.arange(input_ids.shape[1], dtype=torch.long, device=device)
    kv_cache = DynamicCache()
    if depth > 0:
        deep_kv_cache = [DynamicCache() for _ in range(depth)]

    # prefill the problem
    with accelerator.autocast():
        embeds = model.lm_head.weight[input_ids]
        final_hidden, last_hidden = model_forward(
            hidden_state=embeds,
            attn_mask=attn_mask,
            pos=cache_pos,
            kv_cache=kv_cache,
            extract_specific=hidden_layer_num,
        )
        logits = model.lm_head(model.model.model.norm(final_hidden[:, -1, :])).float()

    hidden_cache = torch.zeros(problem_batch_size, 0, 256).to(device)

    # text_end_appeared = False # if the first <｜end▁of▁sentence｜>
    gen_all_done = False

    text_end_mask = torch.ones(problem_batch_size, dtype=torch.int8).to(
        device
    )  # 1 -> not ended
    text_end_indices = torch.ones(problem_batch_size, dtype=torch.long).to(device) * (
        max_length + input_ids.shape[1]
    )

    res = torch.zeros(problem_batch_size, 0, dtype=torch.long).to(device)

    for i in tqdm(range(max_length), desc="sampling progress"):
        with accelerator.autocast():
            hidden_cache = torch.cat(
                [
                    hidden_cache,
                    F.dropout(
                        vae(last_hidden[:, -1:, :], compressing=True),
                        p=hidden_dropout_rate,
                        training=True,
                    ),
                ],
                dim=1,
            )
            uncompressed_hidden = vae.uncompress(hidden_cache[:, -1:, :])

            # more depth -- model looping
            if depth > 0:
                last_hidden = uncompressed_hidden
                for depth_i in range(depth):
                    last_hidden = model_forward(
                        hidden_state=last_hidden,
                        attn_mask=attn_mask[:, input_ids.shape[1] + i - 1].unsqueeze(1),
                        pos=cache_pos[-1:] - input_ids.shape[1] + 1,
                        kv_cache=deep_kv_cache[depth_i],
                        start_layer=depth_start_layer_num,
                        end_layer=hidden_layer_num,
                    )
                uncompressed_hidden = last_hidden

        if i % gc_interval == 0:
            cleanup()

        values, indices = torch.topk(logits, topk, largest=True, sorted=False, dim=-1)
        probs = nn.functional.softmax(values / temperature, dim=-1)
        # probs = probs.masked_fill(probs < min_p * 1 / topk, 0)
        selected_choice = torch.multinomial(
            probs.view(problem_batch_size, -1), num_samples=1
        )
        selected_index = indices.gather(1, selected_choice)
        selected_index[(1 - text_end_mask).bool(), :] = eot
        res = torch.cat([res, selected_index], dim=1)
        selected_index = selected_index.view(problem_batch_size)

        if not gen_all_done and eot in selected_index:
            text_end_appeared = True
            text_end_mask.masked_fill_(selected_index == im_end, 0)
            text_end_indices.masked_fill_(selected_index == im_end, i)
            gen_all_done = not (1 in text_end_mask)
            # if text_end_mask.sum() < problem_batch_size * 0.2 and text_end_indices.max() + 128 < i:
            #     gen_all_done = True

        attn_mask = torch.cat(
            [attn_mask, text_end_indices.unsqueeze(1)], dim=1
        )  # update attention mask

        if gen_all_done:
            break

        # forward
        with accelerator.autocast():
            cache_pos = cache_pos[-1:] + 1
            embeds = model.lm_head.weight[selected_index.view(problem_batch_size, 1)]
            embeds = gater(uncompressed_hidden, embeds)
            final_hidden, last_hidden = model_forward(
                hidden_state=embeds,
                attn_mask=attn_mask,
                pos=cache_pos,
                kv_cache=kv_cache,
                extract_specific=hidden_layer_num,
            )
            logits = model.lm_head(
                model.model.model.norm(final_hidden[:, -1, :])
            ).float()

    cleanup()
    if depth > 0:
        return res, hidden_cache, text_end_indices, attn_mask
    return res, hidden_cache, text_end_indices, attn_mask
