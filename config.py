import torch
from accelerate import Accelerator
from parameters import gradient_accumulation_steps

model_name = "Qwen/Qwen3-1.7B"

accelerator = Accelerator(
    mixed_precision="bf16",
    device_placement=True,
    log_with="tensorboard",
    project_dir="./runs/distributed",
)
device = accelerator.device


def ensure_tensor_type(tensor1, tensor2):
    if hasattr(tensor1, "_local_tensor") and not hasattr(tensor2, "_local_tensor"):
        # If tensor1 is DTensor, but target_type is a regular tensor,
        # This ensures consistent tensor types for concatenation
        tensor2 = accelerator.prepare(tensor2)
    elif hasattr(tensor2, "_local_tensor") and not hasattr(tensor1, "_local_tensor"):
        # If tensor2 is DTensor but tensor is a regular tensor,
        # move tensor to the same type
        tensor1 = accelerator.prepare(tensor1)
    return tensor1, tensor2


def tensor_concat(tensor1, tensor2, dim=0):
    tensor1, tensor2 = ensure_tensor_type(tensor1, tensor2)
    return torch.cat([tensor1, tensor2], dim=dim)
