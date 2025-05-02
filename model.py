import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import model_name
from data import data_train
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator
from gates import Gate
from parameters import gradient_accumulation_steps
from vae import VAE

accelerator = Accelerator(
    mixed_precision="bf16", gradient_accumulation_steps=gradient_accumulation_steps
)
device = accelerator.device

torch.manual_seed(42)

writer = SummaryWriter("runs/demo")

# load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
torch.backends.cuda.enable_flash_sdp(True)

# inject LoRA
peft_config = LoraConfig(
    init_lora_weights="pissa_niter_4",
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.02,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Gater
gater = Gate(2048, 0.01)

# load VAE
vae = VAE(2048, 256, 2048 * 4)
vae = torch.jit.script(vae)
vae.load_state_dict(torch.load("/home/featurize/data/vae_epoch15.pth"))

optimizers = [
    AdamW(model.parameters(), lr=1e-5),
    AdamW(vae.parameters(), lr=5e-5),
    Adam(gater.parameters(), lr=3e-3),
]
gater_scheduler = CosineAnnealingLR(optimizers[2], T_max=500, eta_min=1e-3)
(model, vae, gater, optimizers[0], optimizers[1], optimizers[2], data_train) = (
    accelerator.prepare(
        model, vae, gater, optimizers[0], optimizers[1], optimizers[2], data_train
    )
)

lossf = nn.CrossEntropyLoss(reduction="none")
hidden_regularizer = nn.MSELoss(reduction="none")

# end_of_text mark
# eot = tokenizer('<｜end▁of▁sentence｜>').input_ids[1:][0]
im_end, eot = tokenizer("<|im_end|><|endoftext|>").input_ids
