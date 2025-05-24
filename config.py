from accelerate import Accelerator

model_name = 'Qwen/Qwen3-1.7B'

accelerator = Accelerator(
    mixed_precision='bf16',
)
device = accelerator.device
