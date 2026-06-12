# ablation.py

import yaml
import subprocess
import argparse
import copy

def run_ablation_study(config_path):
    """
    Runs an ablation study by executing main.py with different configurations.
    """
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    if 'ablation' not in base_config or not base_config['ablation']['enabled']:
        print("Ablation study is not enabled in the config file.")
        return

    experiments = base_config['ablation']['experiments']
    print(f"--- Starting Ablation Study: {len(experiments)} experiments to run ---")

    for i, exp_config in enumerate(experiments):
        exp_name = exp_config.get('name', f'experiment_{i+1}')
        print(f"\n--- Running Experiment {i+1}/{len(experiments)}: {exp_name} ---")

        # Build the command
        cmd = ['/usr/local/bin/python3', 'main.py', '--config', config_path, '--experiment_name', exp_name]
        
        # Override base config with experiment-specific settings
        for key, value in exp_config.items():
            if key != 'name':
                if value is None:
                    # Handle null values, which might mean disabling a feature
                    # This is a simplification; a more robust way might be needed
                    pass
                else:
                    cmd.extend([f'--{key}', str(value)])

        print(f"Command: {' '.join(cmd)}")
        
        # Run the experiment
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"!!! Experiment '{exp_name}' failed with error: {e} !!!")
            # Optionally, continue to the next experiment
            # continue
            # Or stop the whole study
            break
        except FileNotFoundError:
            print("Error: 'python' or 'main.py' not found. Make sure you are in the correct directory.")
            break
            
    print("\n--- Ablation Study Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ablation studies for RAC optimization.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    # Enable ablation in the config for the run
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['ablation']['enabled'] = True
    
    # Create a temporary config for the ablation run
    temp_config_path = 'temp_ablation_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    run_ablation_study(temp_config_path)

    # Clean up the temporary config
    import os
    os.remove(temp_config_path)
