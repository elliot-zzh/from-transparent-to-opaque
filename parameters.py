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
# also for passing the path of dataset files
parser.add_argument(
    '--traindataset',
    type=str,
    default='/home/featurize/data/train.jsonl',
    help='Path to the path of train dataset files',
)
parser.add_argument(
    '--testdataset',
    type=str,
    default='/home/featurize/data/test.jsonl',
    help='Path to the path of test dataset files',
)

args = parser.parse_args()

# Load hyperparameters from TOML file
with open(args.config, file_mode) as f:
    config = toml.load(f)

# General
experiment_id = config['general']['id']

# Dataset
train_dataset_path = args.traindataset
test_dataset_path = args.testdataset

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
concept_topk = training['concept_topk']
concept_temperature = training['concept_temperature']
concept_temperature_increase_step = training['concept_temperature_increase_step']
concept_temperature_max = training['concept_temperature_max']
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
vae_lr = training['vae_lr']
gater_lr = training['gater_lr']
gater_lr_start_factor = training['gater_lr_start_factor']
gater_lr_min = training['gater_lr_min']
gater_lr_decay_interval = training['gater_lr_decay_interval']
gater_lr_warmup_interval = training['gater_lr_warmup_interval']

# hidden regularization
hidden_reg = config['hidden_regularization']
enable_hidden_regularization = hidden_reg['enable_hidden_regularization']
enable_length_reg_bonus = hidden_reg['enable_length_reg_bonus']
hidden_regularization_rate = hidden_reg['hidden_regularization_rate']
hidden_dropout_rate = hidden_reg['hidden_dropout_rate']
hidden_reg_len_bonus_a = hidden_reg['hidden_reg_len_bonus_a']
hidden_reg_len_bonus_high = hidden_reg['hidden_reg_len_bonus_high']
enable_hidden_updating = hidden_reg['enable_hidden_updating']
hidden_updating_rate = hidden_reg['hidden_updating_rate']

# gating value bonus
gating = config['gating_value']
enable_gating_bonus = gating['enable_gating_bonus']
gating_bonus_mode = gating['gating_bonus_mode']
gating_value_bonus = gating['gating_value_bonus']
gating_value_decay = gating['gating_value_decay']
gating_value_lambda = gating['gating_value_lambda']
gating_bonus_update_step = gating['gating_bonus_update_step']

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
enable_gating = model['enable_gating']
enable_thinking = model['enable_thinking']
