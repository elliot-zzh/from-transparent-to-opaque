import torch
from accelerate import Accelerator
from parameters import gradient_accumulation_steps

model_name = "Qwen/Qwen3-1.7B"

accelerator = Accelerator(
    mixed_precision="bf16", 
    gradient_accumulation_steps=gradient_accumulation_steps,
    device_placement=True,
    log_with="tensorboard",
    project_dir="./runs/distributed"
)
device = accelerator.device
