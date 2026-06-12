# main.py

import os
import torch
import yaml
import pandas as pd
import numpy as np
import copy
import random

from dataset import get_data_loaders
from tap_hybridnet import TAPCDP
from train import run_experiment
from visualization import plot_training_history, plot_accuracy_vs_complexity, plot_energy_consumption

def deep_update(base_dict, update_dict):
    """Recursively update a dictionary."""
    d = copy.deepcopy(base_dict)
    for k, v in update_dict.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k] = deep_update(d[k], v)
        else:
            # Handle dot notation for nested keys
            keys = k.split('.')
            sub_dict = d
            for i, key in enumerate(keys):
                if i == len(keys) - 1:
                    sub_dict[key] = v
                else:
                    sub_dict = sub_dict.setdefault(key, {})
    return d

def main():
    """
    Main driver script to run experiments across multiple seeds.
    """
    with open('config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    results_dir = base_config['evaluation']['results_dir']
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    summary_list = []

    if not base_config['ablation']['enabled']:
        print("Ablation study is disabled.")
        return

    experiments = base_config['ablation']['experiments']
    seeds = base_config['ablation'].get('seeds', [42])

    for exp_overrides in experiments:
        exp_name = exp_overrides.get('name', 'experiment')
        print(f"\n>>> Running Experiment Group: {exp_name}")
        
        seed_results = []
        
        for seed in seeds:
            print(f"  > Seed: {seed}")
            config = deep_update(base_config, exp_overrides)
            config['training']['seed'] = seed
            
            # Run the experiment
            model, history = run_experiment(config, seed)
            
            # Save individual run history plot
            plot_training_history(history, plots_dir, f"{exp_name}_seed{seed}")
            
            final_metrics = history[-1]
            seed_results.append(final_metrics)
            
        # Aggregate results across seeds
        avg_acc = np.mean([r['val_acc'] for r in seed_results])
        std_acc = np.std([r['val_acc'] for r in seed_results])
        avg_sparsity = np.mean([r['val_sparsity'] for r in seed_results])
        avg_latency = np.mean([r['val_latency'] for r in seed_results])
        avg_energy = np.mean([r['val_energy'] for r in seed_results])
        
        summary_list.append({
            'experiment': exp_name,
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'avg_sparsity': avg_sparsity,
            'avg_latency': avg_latency,
            'avg_energy': avg_energy
        })
        
        print(f"--- Summary for {exp_name} ---")
        print(f"Accuracy: {avg_acc:.2f}% (+/- {std_acc:.2f}%)")
        print(f"Sparsity: {avg_sparsity:.4f}")
        print(f"Latency: {avg_latency:.2f} ms | Energy: {avg_energy:.4f} mJ/image")

    # Save summary and plot aggregate results
    summary_csv_path = os.path.join(results_dir, 'experiment_summary.csv')
    df = pd.DataFrame(summary_list)
    df.to_csv(summary_csv_path, index=False)
    
    plot_accuracy_vs_complexity(summary_csv_path, plots_dir)
    plot_energy_consumption(summary_csv_path, plots_dir)
    
    print(f"\nSummary saved to {summary_csv_path}")

if __name__ == '__main__':
    main()
