import gc
import torch
from config import device
from tokenizer import tokenizer


def cleanup():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.cuda.empty_cache()


def tokenize(text, direct=False, max_length=1024, pad=False, trn=True, device=device):
    if direct:
        res = tokenizer(text, return_tensors="pt", padding=True)
    else:
        res = tokenizer(
            text,
            return_tensors="pt",
            truncation=trn,
            max_length=max_length,
            padding="max_length",
        )
    input_ids = res.input_ids
    attn_mask = res.attention_mask
    return input_ids, attn_mask
