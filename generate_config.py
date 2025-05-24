import toml
import os
import sys


def generate_single_param_configs(original_config, param_name, param_values):
    configs = []
    for value in param_values:
        new_config = original_config.copy()
        current_dict = new_config
        key_parts = param_name.split('.')
        for part in key_parts[:-1]:
            current_dict = current_dict[part]
        current_dict[key_parts[-1]] = value
        configs.append(new_config)
    return configs


def write_configs_to_files(configs, base_filename, output_dir='configs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filenames = []
    for i, config in enumerate(configs):
        filename = os.path.join(output_dir, f'{base_filename}_{i}.toml')
        with open(filename, 'w') as f:
            toml.dump(config, f)
        print(f'Configuration written to {filename}')
        filenames.append(filename)
    return filenames


def get_best_param_score(score_file, param_name):
    if not os.path.exists(score_file):
        print(
            f'Score file {score_file} does not exist. Using default value for {param_name}.'
        )
        return None, None
    best_value = None
    best_score = float('-inf')
    with open(score_file, 'r') as f:
        for line in f:
            value, score = line.strip().split(',')
            value = float(value)
            score = float(score)
            if score > best_score:
                best_score = score
                best_value = value
    return best_value, best_score


def generate_training_scripts(filenames, gpu_num, script_name):
    script_lines = [
        '#!/bin/bash\n',
        '\n',
        f'gpu_queue=($(seq 0 {gpu_num - 1}))\n',
        'pending_tasks=()\n',
    ]

    for filename in filenames:
        script_lines.append(f'pending_tasks+=("--config {filename}")\n')

    script_lines.extend(
        [
            '\n',
            'pids=()\n',
            'launch_next_task() {\n',
            '  if [ ${#pending_tasks[@]} -eq 0 ]; then return; fi\n',
            '  if [ ${#gpu_queue[@]} -eq 0 ]; then return; fi\n',
            '  gpu=${gpu_queue[0]}\n',
            '  gpu_queue=("${gpu_queue[@]:1}")\n',
            '  task=${pending_tasks[0]}\n',
            '  pending_tasks=("${pending_tasks[@]:1}")\n',
            f'  accelerate launch --gpu_ids=$gpu train.py $task --traindataset dataset/train.jsonl --testdataset dataset/test.jsonl &\n',
            '  pids+=($!)\n',
            '}\n',
            '\n',
            'while [ ${#pending_tasks[@]} -gt 0 ] || [ ${#pids[@]} -gt 0 ]; do\n',
            '  while [ ${#pending_tasks[@]} -gt 0 ] && [ ${#gpu_queue[@]} -gt 0 ]; do launch_next_task; done\n',
            '  for i in $(seq 0 $((${#pids[@]} - 1)) ); do\n',
            '    if ! ps -p ${pids[$i]} > /dev/null; then\n',
            '      gpu_used=$(grep -oP "gpu_ids=\K[0-9]+" <(ps -p ${pids[$i]} -o cmd=) 2>/dev/null || echo -1)\n',
            '      if [ $gpu_used -ne -1 ]; then gpu_queue+=($gpu_used); fi\n',
            '      unset pids[$i]\n',
            '      pids=("${pids[@]}")\n',
            '    fi\n',
            '  done\n',
            '  sleep 1\n',
            'done\n',
            'wait\n',
        ]
    )

    with open(script_name, 'w') as f:
        f.writelines(script_lines)

    os.chmod(script_name, 0o755)
    print(f"Training script '{script_name}' generated")


def main():
    if len(sys.argv) < 4:
        print(
            'Usage: python script_name.py <original_config.toml> <gpu_num> <searching_step>'
        )
        sys.exit(1)

    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' does not exist.")
        sys.exit(1)

    gpu_num = int(sys.argv[2])
    searching_step = int(sys.argv[3])

    with open(config_file, 'r') as f:
        original_config = toml.load(f)

    # Update total_steps with searching_step
    original_config['training']['total_steps'] = searching_step

    # Define hyperparameters to tune one at a time
    hyperparameters_to_tune = [
        {
            'param_name': 'training.lr',
            'param_values': [1e-6, 5e-6, 1e-5],
            'base_filename': 'config_lr',
            'score_filename': 'score_lr.txt',
        },
        {
            'param_name': 'training.batch_size',
            'param_values': [1, 2],
            'base_filename': 'config_batch',
            'score_filename': 'score_batch.txt',
        },
    ]

    # First phase: Tune lr
    lr_files = write_configs_to_files(
        generate_single_param_configs(
            original_config,
            hyperparameters_to_tune[0]['param_name'],
            hyperparameters_to_tune[0]['param_values'],
        ),
        hyperparameters_to_tune[0]['base_filename'],
    )
    generate_training_scripts(lr_files, gpu_num, 'train_lr.sh')

    # Generate the main train.sh script
    with open('train.sh', 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write(f'./train_lr.sh\n')
        f.write('wait\n')

        # Check the best lr and update the config
        f.write(
            f"best_lr=$(awk 'NR==1{{print $1}}' {hyperparameters_to_tune[0]['score_filename']} 2>/dev/null)\n"
        )
        f.write(f'if [ -z "$best_lr" ]; then\n')
        f.write(f'  best_lr={hyperparameters_to_tune[0]["param_values"][0]}\n')
        f.write(f'fi\n')
        f.write(f'echo "Best lr: $best_lr"\n')

        # Update original_config with best_lr
        f.write(
            f"python -c \"import toml; config = toml.load('{config_file}'); config['training']['lr'] = float(os.environ.get('best_lr', {hyperparameters_to_tune[0]['param_values'][0]})); toml.dump(config, 'final_best_config.toml')\"\n"
        )

        # Tune other hyperparameters
        other_files = []
        for hp in hyperparameters_to_tune[1:]:
            configs = generate_single_param_configs(
                original_config, hp['param_name'], hp['param_values']
            )
            filenames = write_configs_to_files(configs, hp['base_filename'])
            other_files.extend(filenames)

        generate_training_scripts(other_files, gpu_num, 'train_other.sh')
        f.write('./train_other.sh\n')
        f.write('wait\n')

    os.chmod('train.sh', 0o755)
    print("Main training script 'train.sh' generated")


if __name__ == '__main__':
    main()
