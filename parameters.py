import argparse

try:
    import tomllib as toml

    file_mode = 'rb'  # Binary mode for tomllib (Python 3.11+)
except ImportError:
    import toml

    file_mode = 'r'  # Text mode for toml (third-party library)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Load hyperparameters from a TOML file')
parser.add_argument(
    '--config',
    type=str,
    default='config.toml',
    help='Path to the TOML configuration file',
)

args = parser.parse_args()

# Load hyperparameters from TOML file
with open(args.config, file_mode) as f:
    config = toml.load(f)

# General
experiment_id = config['general']['id']
hf_dataset_name = config['general']['hf_dataset_name']
hf_train_split = config['general'].get('hf_train_split', 'train')

# hyperparameters
training = config['training']
num_epochs = training['num_epochs']
total_steps = training['total_steps']
log_interval = training['log_interval']
save_interval = training['save_interval']
batch_size = training['batch_size']
max_train_length = training['max_train_length']
max_sample_length = training['max_sample_length']
l_cache_length = training['l_cache_length']
concept_temperature = training['concept_temperature']
concept_temperature_increase_step = training['concept_temperature_increase_step']
concept_temperature_max = training['concept_temperature_max']
entropy_tao = training['entropy_tao']
entropy_k = training['entropy_k']
sample_num = training['sample_num']
sample_topk = training['sample_topk']
sample_temperature = training['sample_temperature']
sample_problem_batch = training['sample_problem_batch']
sample_problem_sub_batch = training['sample_problem_sub_batch']
acc_check_only = training['acc_check_only']
train_gc_interval = training['train_gc_interval']
corr_reward = training['corr_reward']
gradient_accumulation_steps = training['gradient_accumulation_steps']
step = training['step']
clip_high = training['clip_high']
clip_low = training['clip_low']
lr = training['lr']
self_distillation_factor = training['self_distillation_factor']
soft_embeds_train_start = training['soft_embeds_train_start']

# other parameters
model = config['model']
looping_depth = model['looping_depth']
hidden_layer_num = model['hidden_layer_num']
depth_start_layer_num = model['depth_start_layer_num']
hidden_injection_layer = model['hidden_injection_layer']
pissa_niter = model['pissa_niter']
lora_r = model['lora_r']
lora_alpha = model['lora_alpha']
double_gate = model['double_gate']
enbale_injection_scale = model['enable_injection_scale']
enable_thinking = model['enable_thinking']
enable_swapping = model['enable_swapping']
