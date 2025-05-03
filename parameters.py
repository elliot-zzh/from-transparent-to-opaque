# hyperparameters
num_epochs = 4  # for each RL batch
total_steps = 1000  # on the whole data
log_interval = 1
save_interval = 256
batch_size = 1
max_train_length = 1024
max_sample_length = 512
l_cache_length = 400
sample_num = 12
sample_topk = 16
sample_temperature = 0.7
sample_problem_batch = 3
sample_problem_sub_batch = 3
acc_check_only = False
train_gc_interval = 15
corr_reward = 2

# hidden regularization
hidden_regularization_rate = 0.5
hidden_dropout_rate = 0.05
hidden_reg_len_bonus_a = 20
hidden_reg_len_bonus_high = 10
hidden_updating_rate = 0.05

# gating value bonus
gating_value_bonus = 0.2
gating_value_decay = 0.95
gating_value_lambda = 5
gating_bonus_update_step = 100

looping_depth = 0  # not ready for depth > 0 yet

gradient_accumulation_steps = 32  # yet to be decided
step = 1  # total step count

hidden_layer_num = 20
depth_start_layer_num = 10
