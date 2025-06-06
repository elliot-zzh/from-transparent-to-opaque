import shutil
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler
from config import model_name, accelerator, device
from data import data_train
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from gates import Gate
from tokenizer import tokenizer
from vae import VAE
from huggingface_hub import hf_hub_download
import os
from parameters import (
    lr,
    vae_lr,
    gater_lr_start_factor,
    gater_lr,
    gater_lr_min,
    gater_lr_decay_interval,
    gater_lr_warmup_interval,
    experiment_id,
    pissa_niter,
    lora_r,
    lora_alpha,
)


def print_num_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {total_params:,}')
    print(f'Trainable Parameters: {trainable_params:,}')


if not os.path.exists('./data/vae/vae_epoch15.pth'):
    os.makedirs('./data/vae')
    downloaded_path = hf_hub_download(
        repo_id='ethangoh7086cmd/gated-latent-reasoning-loop-vae',
        filename='vae_epoch15.pth',
        cache_dir='./data/vae',
        force_download=False,
    )
    shutil.copy(downloaded_path, './data/vae')
    print('VAE model downloaded and saved to ./data/vae/vae_epoch15.pth')
    os.remove(downloaded_path)

torch.manual_seed(42)

writer = SummaryWriter(f'runs/experiment-{experiment_id}')

# load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=torch.bfloat16,
    attn_implementation='sdpa',
)
model.config.use_sliding_window = True
model.sliding_window = 2048
model.gradient_checkpointing_enable()
torch.backends.cuda.enable_flash_sdp(True)

# inject LoRA
peft_config = LoraConfig(
    init_lora_weights='pissa_niter_' + str(pissa_niter),
    task_type='CAUSAL_LM',
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules='all-linear',
    lora_dropout=0.01,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Gater
gater = Gate(2048, 0.01)

print('gater: ', end=' ')
print_num_parameters(gater)

optimizers = [
    AdamW(model.parameters(), lr=lr),
    Adam(gater.parameters(), lr=gater_lr),
]

"""
warmup_scheduler = LinearLR(
    optimizers[2],
    start_factor=gater_lr_start_factor,
    total_iters=gater_lr_warmup_interval,
)
cosine_scheduler = CosineAnnealingLR(
    optimizers[2],
    T_max=gater_lr_decay_interval,
    eta_min=gater_lr_min,
)
gater_scheduler = ChainedScheduler(
    [
        warmup_scheduler,
        cosine_scheduler,
    ]
)
"""

(model, gater, optimizers[0], optimizers[1], data_train) = accelerator.prepare(
    model, gater, optimizers[0], optimizers[1], data_train
)

lossf = nn.CrossEntropyLoss(reduction='none')
hidden_regularizer = nn.MSELoss(reduction='none')

# end_of_text mark
# eot = tokenizer('<｜end▁of▁sentence｜>').input_ids[1:][0]
im_end, eot = tokenizer('<|im_end|><|endoftext|>').input_ids
