# A lot of hacking here. For details refer to
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/
from config import device
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache, SinkCache
from tqdm import tqdm
from parameters import hidden_layer_num, depth_start_layer_num, hidden_dropout_rate, enable_gating
from model import im_end, eot, gater, model, accelerator
from utils import cleanup
from forward import model_forward
import pandas as pd

def safe_entropy(logits, eps=1e-10):
    # Compute entropy: -sum(p * log(p))
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def pad_up(tensor: torch.Tensor, dim: int, target: int, filling=0) -> torch.Tensor:
    assert tensor.shape[dim] <= target, (
        'Target size must be greater than or equal to the current size.'
    )
    pad_size = target - tensor.shape[dim]
    if pad_size == 0:
        return tensor
    pad = [0] * (2 * tensor.dim())
    pad[-(2 * dim + 1)] = pad_size
    return F.pad(tensor, pad, mode='constant', value=filling)


def sampler(
    input_ids,
    attn_mask,
    temperature=0.7,
    concept_temperature=0.1,
    topk=16,
    concept_topk=10,
    max_length=2048,
    gc_interval=48,
):
    model.eval()
    gater.eval()

    # tokenize
    problem_batch_size = input_ids.shape[0]
    cache_pos = torch.arange(input_ids.shape[1], dtype=torch.int, device=device)
    # kv_cache = SinkCache(window_length=1280, num_sink_tokens=4)
    kv_cache = DynamicCache()

    # prefill the problem
    with accelerator.autocast():
        logits = model_forward(
            model.model.model.embed_tokens(input_ids),
            attn_mask=attn_mask,
            pos=cache_pos,
            kv_cache=kv_cache,
        )[:, -1:].float()

    concept_token_probs = torch.Tensor(problem_batch_size, 0, concept_topk).float().to(device)
    concept_token_indices = (
        torch.Tensor(problem_batch_size, 0, concept_topk).int().to(device)
    )
    entropy = torch.Tensor(problem_batch_size, 0).to(device)

    # text_end_appeared = False # if the first <｜end▁of▁sentence｜>
    gen_all_done = False

    text_end_mask = torch.ones(problem_batch_size, dtype=torch.int8).to(
        device
    )  # 1 -> not ended
    text_end_indices = torch.ones(problem_batch_size, dtype=torch.long).to(device) * (
        max_length + input_ids.shape[1]
    )

    res = torch.zeros(problem_batch_size, 0, dtype=torch.long).to(device)
    res_probs = torch.zeros(problem_batch_size, 0, dtype=torch.float32).to(device)

    for i in tqdm(range(max_length), desc='sampling progress'):
        if i % gc_interval == 0:
            cleanup()

        with accelerator.autocast():
            # concat concept tokens
            concept_probs = F.softmax(
                logits / concept_temperature, dim=-1
            )
            concept_probs, concept_indices = torch.topk(
                concept_probs, concept_topk, largest=True, sorted=False, dim=-1
            )
            concept_probs *= logits.gather(2, concept_indices)
            concept_probs /= concept_probs.sum(dim=-1, keepdim=True)
            concept_token_probs = torch.cat([concept_token_probs, concept_probs], dim=1)
            concept_token_indices = torch.cat(
                [concept_token_indices, concept_indices], dim=1
            )
            embeds = soft_embeds = (
                model.model.model.embed_tokens(concept_indices).transpose(-2, -1)
                * concept_probs.unsqueeze(-2)
            ).sum(dim=-1)

            probs_ = F.log_softmax(
                logits, dim=-1
            )  # without temperature, for training
            # entropy = torch.cat([entropy, safe_entropy(logits).view(problem_batch_size, -1)], dim=1) # compute entropy
            topk_probs, indices = torch.topk(
                logits, topk, largest=True, sorted=False, dim=-1
            )
            topk_probs = F.softmax(topk_probs / temperature, dim=-1)
            # probs = probs.masked_fill(probs < min_p * 1 / topk, 0)
            selected_choice = torch.multinomial(
                topk_probs.view(problem_batch_size, -1), num_samples=1
            )

            selected_index = indices.view(problem_batch_size, -1).gather(
                1, selected_choice
            )
            selected_probs = probs_.view(problem_batch_size, -1).gather(
                1, selected_index
            )
            selected_index[(1 - text_end_mask).bool(), :] = eot
            selected_probs[(1 - text_end_mask).bool(), :] = 1e6  # mask out
            res = torch.cat([res, selected_index], dim=1)
            res_probs = torch.cat([res_probs, selected_probs], dim=1)
            selected_index = selected_index.view(problem_batch_size)

        if not gen_all_done and im_end in selected_index:
            # text_end_appeared = True
            text_end_mask.masked_fill_(selected_index == im_end, 0)
            text_end_indices.masked_fill_(selected_index == im_end, i)
            gen_all_done = 1 not in text_end_mask
            # if text_end_mask.sum() < problem_batch_size * 0.2 and text_end_indices.max() + 128 < i:
            #     gen_all_done = True

        attn_mask = torch.cat(
            [attn_mask, text_end_indices.unsqueeze(1)], dim=1
        )  # update attention mask

        if gen_all_done:
            break

        # forward
        cache_pos = cache_pos[-1:] + 1
        with accelerator.autocast():
            if enable_gating:
                embeds = gater(soft_embeds, model.model.model.embed_tokens(selected_index.unsqueeze(-1)))
            logits = model_forward(
                embeds,
                attn_mask=attn_mask,
                pos=cache_pos,
                kv_cache=kv_cache,
            ).float()

    cleanup()

    res = pad_up(res, filling=eot, dim=1, target=max_length)
    res_probs = pad_up(res_probs, filling=eot, dim=1, target=max_length)
    concept_token_probs = pad_up(
        concept_token_probs, filling=0, dim=1, target=max_length
    )
    concept_token_indices = pad_up(
        concept_token_indices, filling=1, dim=1, target=max_length
    )
    attn_mask = pad_up(
        attn_mask, filling=0, dim=1, target=input_ids.shape[1] + max_length
    )

    # entropy = entropy.transpose(0, 1).cpu().numpy()
    # pd.DataFrame(entropy, columns=[f'Col{j+1}' for j in range(entropy.shape[1])]).to_csv('entropy.csv', index=False)

    return (
        res,
        res_probs,
        text_end_indices,
        attn_mask,
        concept_token_probs,
        concept_token_indices,
    )
