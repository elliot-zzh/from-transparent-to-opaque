from accelerate import Accelerator
from parameters import config

model_name = config['general']['model_name']
model_path = config['general'].get('model_path', "")

accelerator = Accelerator(
    mixed_precision='bf16',
)
device = accelerator.device
