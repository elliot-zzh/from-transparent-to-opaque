from accelerate import Accelerator, FullyShardedDataParallelPlugin, DeepSpeedPlugin

from parameters import config

model_name = config['general']['model_name']
model_path = config['general'].get('model_path', '')
im_end_token = config['general'].get('im_end_token', '<|im_end|>')
eot_token = config['general'].get('eot_token', '<|endoftext|>')
eoth_token = config['general'].get('eoth_token', '</think>')
soft_thinking = config['general'].get('soft_thinking', True)

fsdp_plugin = FullyShardedDataParallelPlugin()

deepspeed_plugin = DeepSpeedPlugin()

accelerator = Accelerator(
    mixed_precision='bf16',
    fsdp_plugin=fsdp_plugin,
)
device = accelerator.device

def deepspeed_enabled():
    return accelerator.state.deepspeed_plugin is None\
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
