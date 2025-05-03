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
        embeds = model.model.model.embed_tokens(input_ids)
        final_hidden, last_hidden = model_forward(
            hidden_state=embeds,
            attn_mask=attn_mask,
            pos=cache_pos,
            kv_cache=kv_cache,
            extract_specific=hidden_layer_num,
        )
        logits = model.lm_head(model.model.model.norm(final_hidden[:, -1, :])).float()

    # Create tensors using the same device as input_ids to ensure consistent tensor types
    hidden_cache = torch.zeros(problem_batch_size, 0, 256, device=input_ids.device, dtype=input_ids.dtype)

    # text_end_appeared = False # if the first <｜end▁of▁sentence｜>
    gen_all_done = False

    text_end_mask = torch.ones(problem_batch_size, dtype=torch.int8, device=input_ids.device)  # 1 -> not ended
    text_end_indices = torch.ones(problem_batch_size, dtype=torch.long, device=input_ids.device) * (
        max_length + input_ids.shape[1]
    )

    res = torch.zeros(problem_batch_size, 0, dtype=torch.long, device=input_ids.device)

    for i in tqdm(range(max_length), desc="sampling progress"):
        with accelerator.autocast():
            # Get the compressed hidden state and ensure it has the same type/device as hidden_cache
            compressed_hidden = F.dropout(
                vae(last_hidden[:, -1:, :], compressing=True),
                p=hidden_dropout_rate,
                training=True,
            )
            
            # Ensure both tensors have the same type before concatenation
            if hasattr(hidden_cache, "_local_tensor") and not hasattr(compressed_hidden, "_local_tensor"):
                # If hidden_cache is DTensor but compressed_hidden is regular tensor
                # This ensures consistent tensor types for concatenation
                compressed_hidden = accelerator.prepare(compressed_hidden)
            elif hasattr(compressed_hidden, "_local_tensor") and not hasattr(hidden_cache, "_local_tensor"):
                # If compressed_hidden is DTensor but hidden_cache is regular tensor
                # Move hidden_cache to the same type
                hidden_cache = accelerator.prepare(hidden_cache)
                
            # Now perform concatenation with compatible tensor types
            hidden_cache = torch.cat([hidden_cache, compressed_hidden], dim=1)
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
        
        # Handle tensor concatenation safely
        if hasattr(res, "_local_tensor") and not hasattr(selected_index, "_local_tensor"):
            selected_index = accelerator.prepare(selected_index)
        elif hasattr(selected_index, "_local_tensor") and not hasattr(res, "_local_tensor"):
            res = accelerator.prepare(res)
            
        res = torch.cat([res, selected_index], dim=1)
        selected_index = selected_index.view(problem_batch_size)

        if not gen_all_done and eot in selected_index:
            text_end_appeared = True
            text_end_mask.masked_fill_(selected_index == im_end, 0)
            text_end_indices.masked_fill_(selected_index == im_end, i)
            gen_all_done = not (1 in text_end_mask)
            # if text_end_mask.sum() < problem_batch_size * 0.2 and text_end_indices.max() + 128 < i:
            #     gen_all_done = True

        # Handle attn_mask concatenation safely
        text_end_indices_unsqueezed = text_end_indices.unsqueeze(1)
        if hasattr(attn_mask, "_local_tensor") and not hasattr(text_end_indices_unsqueezed, "_local_tensor"):
            text_end_indices_unsqueezed = accelerator.prepare(text_end_indices_unsqueezed)
        elif hasattr(text_end_indices_unsqueezed, "_local_tensor") and not hasattr(attn_mask, "_local_tensor"):
            attn_mask = accelerator.prepare(attn_mask)
            
        attn_mask = torch.cat(
            [attn_mask, text_end_indices_unsqueezed], dim=1
        )  # update attention mask

        if gen_all_done:
            break

        # forward
        with accelerator.autocast():
            cache_pos = cache_pos[-1:] + 1
            embeds = model.model.model.embed_tokens(
                selected_index.view(problem_batch_size, 1)
            )
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
