# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_training_history(history, plots_dir, experiment_name):
    """Plots and saves the training history (Accuracy and Sparsity)."""
    df = pd.DataFrame(history)
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['val_acc'], label='Val Accuracy', marker='o')
    plt.title(f'Accuracy - {experiment_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    # Sparsity Plot
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['val_sparsity'], label='Sparsity Ratio', color='orange', marker='s')
    plt.title(f'Sparsity Ratio - {experiment_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Sparsity (Fraction)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, f'{experiment_name}_history.png')
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_vs_complexity(summary_csv_path, plots_dir):
    """Plots Accuracy vs. Latency and Accuracy vs. Sparsity from summary CSV."""
    if not os.path.exists(summary_csv_path):
        print(f"Summary CSV not found at {summary_csv_path}")
        return
        
    df = pd.read_csv(summary_csv_path)
    
    plt.figure(figsize=(15, 6))
    
    # Accuracy vs. Sparsity
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='avg_sparsity', y='avg_accuracy', hue='experiment', s=100)
    plt.errorbar(df['avg_sparsity'], df['avg_accuracy'], yerr=df['std_accuracy'], fmt='none', alpha=0.5)
    plt.title('Accuracy vs. Sparsity Ratio')
    plt.xlabel('Sparsity Ratio (Lower is better)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.grid(True)
    
    # Accuracy vs. Latency
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='avg_latency', y='avg_accuracy', hue='experiment', s=100, palette='viridis')
    plt.title('Accuracy vs. Latency (T4 Simulation)')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, 'pareto_frontier.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Pareto frontier plot saved to {save_path}")

def plot_energy_consumption(summary_csv_path, plots_dir):
    """Plots Energy Consumption across experiments."""
    if not os.path.exists(summary_csv_path):
        return
        
    df = pd.read_csv(summary_csv_path)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='experiment', y='avg_energy')
    plt.title('Estimated Energy Consumption per Image (mJ)')
    plt.ylabel('Energy (mJ)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, 'energy_comparison.png')
    plt.savefig(save_path)
    plt.close()
