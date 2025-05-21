from accelerate import Accelerator
from parameters import gradient_accumulation_steps

model_name = 'Qwen/Qwen3-1.7B'

accelerator = Accelerator(
    mixed_precision='bf16', gradient_accumulation_steps=gradient_accumulation_steps,
)
device = accelerator.device
