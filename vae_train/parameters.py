import torch

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
batch_size = 20
num_epochs = 30
gradient_accumulation_steps = 6
log_interval = 1
save_interval = 5
hidden_layer_num = 20
