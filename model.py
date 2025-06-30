import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM

from config import accelerator, device, model_name, model_path, eot_token, eoth_token, im_end_token
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
if model_path:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
    )
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

(model, optimizers[0], data_train) = accelerator.prepare(
    model, optimizers[0], data_train
)

lossf = nn.CrossEntropyLoss(reduction='none')
hidden_regularizer = nn.MSELoss(reduction='none')

# end_of_text mark
im_end = tokenizer(im_end_token).input_ids[0]
eot = tokenizer(eot_token).input_ids[0]
eoth = tokenizer(eoth_token).input_ids[0]
