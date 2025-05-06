import torch
from accelerate import Accelerator

# Use accelerator for distributed training
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=6,
    device_placement=True,
    log_with="tensorboard",
    project_dir="./runs/vae_distributed",
)
device = accelerator.device

batch_size = 20
num_epochs = 30
gradient_accumulation_steps = 6
log_interval = 1
save_interval = 5
hidden_layer_num = 20
