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


def generate_training_scripts(filenames, gpu_num, script_name):
    script_lines = [
        '#!/bin/bash\n',
        '\n',
        f'gpu_queue=($(seq 0 {gpu_num - 1}))\n',
        'pending_tasks=()\n',
    ]

    for filename in filenames:
        script_lines.append(f'pending_tasks+=("--config {filename}\n")')

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
            f'  accelerate launch --gpu_ids=$gpu --num_processes=1 --mixed_precision=bf16 train.py $task --traindataset dataset/train.jsonl --testdataset dataset/test.jsonl &\n',
            '  pids+=($!)\n',
            '}\n',
            '\n',
            'while [ ${#pending_tasks[@]} -gt 0 ] || [ ${#pids[@]} -gt 0 ]; do\n',
            '  while [ ${#pending_tasks[@]} -gt 0 ] && [ ${#gpu_queue[@]} -gt 0 ]; do launch_next_task; done\n',
            '  for i in $(seq 0 $((${#pids[@]} - 1)) ); do\n',
            '    if ! ps -p ${pids[$i]} > /dev/null; then\n',
            '      gpu_used=$(grep -oP "gpu_ids=\K[0-9]+" <(ps -p ${pids[$i]} -o cmd=) 2>/dev/null || echo -1\n',
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
        },
        {
            'param_name': 'training.batch_size',
            'param_values': [1, 2],
            'base_filename': 'config_batch',
        },
    ]

    for hp in hyperparameters_to_tune:
        # Generate configurations for the current hyperparameter
        configs = generate_single_param_configs(
            original_config, hp['param_name'], hp['param_values']
        )
        filenames = write_configs_to_files(configs, hp['base_filename'])

        # Generate training script for the current hyperparameter
        script_name = f'train_{hp["base_filename"]}.sh'
        generate_training_scripts(filenames, gpu_num, script_name)

    # Generate the main train.sh script to run all training scripts sequentially
    with open('train.sh', 'w') as f:
        f.write('#!/bin/bash\n\n')
        for hp in hyperparameters_to_tune:
            script_name = f'train_{hp["base_filename"]}.sh'
            f.write(f'./{script_name}\n')
            f.write('wait\n')

    os.chmod('train.sh', 0o755)
    print("Main training script 'train.sh' generated")


if __name__ == '__main__':
    main()
