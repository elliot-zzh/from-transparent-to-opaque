import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM

from config import accelerator, device, model_name
from data import data_train
from parameters import (
    experiment_id,
    lora_alpha,
    lora_r,
    lr,
    pissa_niter,
)
from tokenizer import tokenizer

writer = SummaryWriter(f'runs/experiment-{experiment_id}')

# load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=torch.bfloat16,
    attn_implementation='sdpa',
)
# model.config.use_sliding_window = True
# model.config.sliding_window = 4096
model.gradient_checkpointing_enable()
torch.backends.cuda.enable_flash_sdp(True)

# inject LoRA
peft_config = LoraConfig(
    init_lora_weights='pissa_niter_' + str(pissa_niter),
    task_type='CAUSAL_LM',
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules='all-linear',
    lora_dropout=0.05,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print('gater: ', end=' ')

optimizers = [
    AdamW(model.parameters(), lr=lr),
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

(model, optimizers[0], data_train) = accelerator.prepare(
    model, optimizers[0], data_train
)

lossf = nn.CrossEntropyLoss(reduction='none')
hidden_regularizer = nn.MSELoss(reduction='none')

# end_of_text mark
# eot = tokenizer('<｜end▁of▁sentence｜>').input_ids[1:][0]
im_end, eot, eoth = tokenizer('<|im_end|><|endoftext|></think>').input_ids
