# eval.py

import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
from tqdm import tqdm

def evaluate_model(model, data_loader, device, results_dir, config):
    """
    Evaluates the model on the test set and saves the results.
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_bpp = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating on Test Set")
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            
            logits, bpp = model(images)
            
            _, predicted = torch.max(logits.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total_bpp += bpp.item()

    # Calculate metrics
    accuracy = 100 * (np.array(all_preds) == np.array(all_targets)).sum() / len(all_targets)
    avg_bpp = total_bpp / len(data_loader)
    cm = confusion_matrix(all_targets, all_preds)
    
    print("\n--- Evaluation Summary ---")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Average BPP: {avg_bpp:.4f}")

    # Save results
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    results_data = {
        'model': config['task_model']['name'],
        'compression': 'learned' if model.compression_model else 'none',
        'jpeg_quality': config.get('jpeg_quality', 'N/A'),
        'accuracy': accuracy,
        'bpp': avg_bpp,
        'lambda_rate': config['training']['rac_weights']['lambda_rate'],
        'lambda_accuracy': config['training']['rac_weights']['lambda_accuracy'],
        'lambda_complexity': config['training']['rac_weights']['lambda_complexity'],
    }
    
    df = pd.DataFrame([results_data])
    results_path = os.path.join(results_dir, 'evaluation_summary.csv')
    
    if os.path.exists(results_path):
        df_existing = pd.read_csv(results_path)
        df = pd.concat([df_existing, df], ignore_index=True)
        
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    return accuracy, avg_bpp, cm

if __name__ == '__main__':
    import yaml
    from dataset import get_data_loaders
    from models import get_full_model
    from compression import CompressionModel
    from train import run_training

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Use smaller epochs for example run
    config['training']['epochs'] = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Get Data ---
    # We use the validation loader as a stand-in for the test loader in this example
    _, test_loader, num_classes = get_data_loaders(config)
    
    # --- Get Model ---
    # This assumes a model is already trained. For a quick test, we'll train for one epoch.
    print("--- Running a mock training for 1 epoch to get a model ---")
    train_loader, _, _ = get_data_loaders(config)
    compression_model = CompressionModel(config).to(device)
    full_model = get_full_model(config, num_classes, compression_model).to(device)
    trained_model, _ = run_training(config, full_model, train_loader, test_loader, device)
    
    # --- Evaluate ---
    results_dir = config['evaluation']['results_dir']
    evaluate_model(trained_model, test_loader, device, results_dir, config)
