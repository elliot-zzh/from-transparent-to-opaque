import argparse

try:
    import tomllib as toml

    file_mode = "rb"  # Binary mode for tomllib (Python 3.11+)
except ImportError:
    import toml

    file_mode = "r"  # Text mode for toml (third-party library)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Load hyperparameters from a TOML file")
parser.add_argument(
    "--config",
    type=str,
    default="config.toml",
    help="Path to the TOML configuration file",
)
args = parser.parse_args()

# Load hyperparameters from TOML file
with open(args.config, file_mode) as f:
    config = toml.load(f)

experiment_id = config['general']['id']

# hyperparameters
num_epochs = config["training"]["num_epochs"]
total_steps = config["training"]["total_steps"]
log_interval = config["training"]["log_interval"]
save_interval = config["training"]["save_interval"]
batch_size = config["training"]["batch_size"]
max_train_length = config["training"]["max_train_length"]
max_sample_length = config["training"]["max_sample_length"]
l_cache_length = config["training"]["l_cache_length"]
sample_num = config["training"]["sample_num"]
sample_topk = config["training"]["sample_topk"]
sample_temperature = config["training"]["sample_temperature"]
sample_problem_batch = config["training"]["sample_problem_batch"]
sample_problem_sub_batch = config["training"]["sample_problem_sub_batch"]
acc_check_only = config["training"]["acc_check_only"]
train_gc_interval = config["training"]["train_gc_interval"]
corr_reward = config["training"]["corr_reward"]

# hidden regularization
hidden_regularization_rate = config["hidden_regularization"][
    "hidden_regularization_rate"
]
hidden_dropout_rate = config["hidden_regularization"]["hidden_dropout_rate"]
hidden_reg_len_bonus_a = config["hidden_regularization"]["hidden_reg_len_bonus_a"]
hidden_reg_len_bonus_high = config["hidden_regularization"]["hidden_reg_len_bonus_high"]
hidden_updating_rate = config["hidden_regularization"]["hidden_updating_rate"]

# gating value bonus
gating_value_bonus = config["gating_value"]["gating_value_bonus"]
gating_value_decay = config["gating_value"]["gating_value_decay"]
gating_value_lambda = config["gating_value"]["gating_value_lambda"]
gating_bonus_update_step = config["gating_value"]["gating_bonus_update_step"]

# other parameters
looping_depth = config["model"]["looping_depth"]
gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
step = config["training"]["step"]
hidden_layer_num = config["model"]["hidden_layer_num"]
depth_start_layer_num = config["model"]["depth_start_layer_num"]
clip_high = config["training"]["clip_high"]
clip_low = config["training"]["clip_low"]
lr = config["training"]["lr"]
vae_lr = config['training']['vae_lr']
gater_lr = config['training']['gater_lr']
gater_lr_min = config['training']['gater_lr_min']
gater_lr_decay_interval = config['training']['gater_lr_decay_interval']