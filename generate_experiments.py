import toml
import os
import sys
import copy


def generate_single_param_configs(original_config, param_name, param_values):
    """Generate multiple configurations by varying a single parameter."""
    configs = []
    for value in param_values:
        # Deep copy to avoid modifying original
        new_config = copy.deepcopy(original_config)

        # Navigate to nested parameter
        current_dict = new_config
        key_parts = param_name.split('.')

        # Navigate to parent dict
        for part in key_parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]

        # Set the final value
        current_dict[key_parts[-1]] = value
        configs.append(new_config)
    return configs


def write_configs_to_files(configs, base_filename, output_dir='configs'):
    """Write configuration dictionaries to TOML files."""
    os.makedirs(output_dir, exist_ok=True)
    filenames = []

    for i, config in enumerate(configs):
        # Substitute rank for id = -1 in general section
        if 'general' in config and config['general'].get('id') == -1:
            config['general']['id'] = i

        filename = os.path.join(output_dir, f'{base_filename}_{i}.toml')
        with open(filename, 'w') as f:
            toml.dump(config, f)
        print(f'Configuration written to {filename}')
        filenames.append(filename)
    return filenames


def generate_sub_script(sub_script_name, assigned_configs, gpu_id):
    """Generate a Bash sub-script to run assigned training jobs sequentially on a specific GPU."""
    if not assigned_configs:
        return

    with open(sub_script_name, 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('set -e  # Exit on any error\n\n')

        for config in assigned_configs:
            f.write(
                f'echo "Starting training with config: {config}"\n'
                f'CUDA_VISIBLE_DEVICES={gpu_id} python '
                f'train.py --config {config} --traindataset dataset/train.jsonl --testdataset dataset/test.jsonl\n'
                f'echo "Completed training with config: {config}"\n\n'
            )

    os.chmod(sub_script_name, 0o755)
    print(f"Sub-script '{sub_script_name}' generated")


def generate_main_script(main_script_name, sub_script_names):
    """Generate a main Bash script to run all sub-scripts in parallel and wait for completion."""
    if not sub_script_names:
        return

    with open(main_script_name, 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('set -e  # Exit on any error\n\n')
        f.write('echo "Starting parallel GPU training..."\n\n')
        f.write('. ./log_vram_usage.sh vram_usage_log.csv 5\n\n')

        # Start all sub-scripts in background
        for sub_script in sub_script_names:
            f.write(f'echo "Launching {sub_script}"\n')
            f.write(f'./{sub_script} &\n')

        f.write('\necho "Waiting for all GPU processes to complete..."\n')
        f.write('wait\n')
        f.write('echo "All GPU processes completed"\n')

    os.chmod(main_script_name, 0o755)
    print(f"Main training script '{main_script_name}' generated")


def main():
    """Main function to process input and generate scripts."""
    if len(sys.argv) < 4:
        print(
            'Usage: python script_name.py <original_config.toml> <gpu_num> <searching_step>'
        )
        sys.exit(1)

    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' does not exist.")
        sys.exit(1)

    try:
        gpu_num = int(sys.argv[2])
        searching_step = int(sys.argv[3])
    except ValueError:
        print('Error: gpu_num and searching_step must be integers')
        sys.exit(1)

    if gpu_num <= 0:
        print('Error: gpu_num must be positive')
        sys.exit(1)

    # Load original config
    try:
        with open(config_file, 'r') as f:
            original_config = toml.load(f)
    except Exception as e:
        print(f'Error loading config file: {e}')
        sys.exit(1)

    # Update total_steps with searching_step
    if 'training' not in original_config:
        original_config['training'] = {}
    original_config['training']['total_steps'] = searching_step

    # Define hyperparameters to tune one at a time
    hyperparameters_to_tune = [
        {
            'param_name': 'training.lr',
            'param_values': [1e-6, 3e-6, 5e-6, 1e-5, 3e-5],
            'base_filename': 'config_lr',
        },
        {
            'param_name': 'training.batch_size',
            'param_values': [1, 2],
            'base_filename': 'config_batch',
        },
    ]

    main_script_names = []

    for hp in hyperparameters_to_tune:
        print(f'\nProcessing hyperparameter: {hp["param_name"]}')

        # Generate configurations for the current hyperparameter
        configs = generate_single_param_configs(
            original_config, hp['param_name'], hp['param_values']
        )
        filenames = write_configs_to_files(configs, hp['base_filename'])

        base_filename = hp['base_filename']
        sub_script_names = []

        # Generate sub-scripts for each GPU
        for gpu in range(gpu_num):
            # Assign configs to this GPU using round-robin distribution
            assigned_configs = [
                filenames[j] for j in range(len(filenames)) if j % gpu_num == gpu
            ]

            if (
                assigned_configs
            ):  # Only generate sub-script if there are configs assigned
                sub_script_name = f'train_{base_filename}_gpu{gpu}.sh'
                generate_sub_script(sub_script_name, assigned_configs, gpu)
                sub_script_names.append(sub_script_name)

        # Generate the main script for this hyperparameter
        if sub_script_names:
            main_script_name = f'train_{base_filename}.sh'
            generate_main_script(main_script_name, sub_script_names)
            main_script_names.append(main_script_name)

    # Generate the overall train.sh script to run all main scripts sequentially
    if main_script_names:
        with open('train.sh', 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write('set -e  # Exit on any error\n\n')
            f.write('echo "Starting hyperparameter search..."\n\n')

            for main_script in main_script_names:
                f.write(f'echo "Running hyperparameter sweep: {main_script}"\n')
                f.write(f'./{main_script}\n')
                f.write(f'echo "Completed hyperparameter sweep: {main_script}"\n\n')

            f.write('echo "All hyperparameter sweeps completed"\n')

        os.chmod('train.sh', 0o755)
        print("\nOverall training script 'train.sh' generated")
    else:
        print('No scripts generated')


if __name__ == '__main__':
    main()
