from accelerate import Accelerator
from parameters import config

model_name = config['general']['model_name']
model_path = config['general'].get('model_path', "")
im_end_token = config['general'].get('im_end_token', '<|im_end|>')
eot_token = config['general'].get('eot_token', '<|endoftext|>')
eoth_token = config['general'].get('eoth_token', '</think>')
soft_thinking = config['general'].get('soft_thinking', True)

accelerator = Accelerator(
    mixed_precision='bf16',
)
device = accelerator.device
