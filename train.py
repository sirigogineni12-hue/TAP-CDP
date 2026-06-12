# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import time
import numpy as np
from tqdm import tqdm
import os
import random

from dataset import get_data_loaders
from tap_hybridnet import TAPCDP, block_dct
from models import calculate_flops

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, data_loader, optimizer, scheduler, device, epoch, config, scaler=None):
    model.train()
    model.tacs.current_epoch = epoch
    
    total_loss = 0
    total_task_loss = 0
    total_comp_loss = 0
    
    lambda_c = config['training']['rac_weights'].get('lambda_rate', 0.01)
    use_amp = config['training'].get('use_amp', False)

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    for images, targets in progress_bar:
        images, targets = images.to(device), targets.to(device)
        
        # Convert RGB to Luminance (Y) if needed for DCT
        if images.shape[1] == 3:
            y = 0.299*images[:,0:1] + 0.587*images[:,1:2] + 0.114*images[:,2:3]
        else:
            y = images

        optimizer.zero_grad()
        
        if use_amp and scaler:
            with autocast():
                logits, m = model(y)
                task_loss = nn.CrossEntropyLoss()(logits, targets)
                # Complexity/Rate loss as mean of mask (Sparsity Ratio)
                comp_loss = m.mean()
                loss = task_loss + lambda_c * comp_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, m = model(y)
            task_loss = nn.CrossEntropyLoss()(logits, targets)
            comp_loss = m.mean()
            loss = task_loss + lambda_c * comp_loss
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_comp_loss += comp_loss.item()
        
        progress_bar.set_postfix({
            'L': f"{loss.item():.3f}",
            'T': f"{task_loss.item():.3f}",
            'S': f"{comp_loss.item():.3f}",
            'tau': f"{model.tacs.get_temperature():.2f}"
        })

    avg_loss = total_loss / len(data_loader)
    return avg_loss, total_task_loss / len(data_loader), total_comp_loss / len(data_loader)

def validate(model, data_loader, device, config):
    model.eval()
    correct = 0
    total = 0
    total_comp = 0
    
    # For latency measurement (simulation for T4)
    latencies = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            if images.shape[1] == 3:
                y = 0.299*images[:,0:1] + 0.587*images[:,1:2] + 0.114*images[:,2:3]
            else:
                y = images
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            logits, m = model(y)
            end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
            
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            total_comp += m.mean().item()

    accuracy = 100 * correct / total
    avg_sparsity = total_comp / len(data_loader)
    
    # Latency: In a real T4, we'd scale this or just report measured.
    # Here we report measured on current device.
    avg_latency = np.mean(latencies) # in ms
    
    # Energy: E (mJ) = P (70W) * Latency (s)
    # 70W = 70 J/s. Latency in ms -> Latency / 1000 in s.
    # E = 70 * (Latency / 1000) * 1000 = 70 * Latency mJ.
    avg_energy = config['evaluation'].get('device_tdp', 70) * (avg_latency / 1000) * 1000 # mJ per batch
    avg_energy_per_image = avg_energy / data_loader.batch_size

    return accuracy, avg_sparsity, avg_latency, avg_energy_per_image

def run_experiment(config, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, num_classes = get_data_loaders(config)
    
    model = TAPCDP(
        backbone_name=config['task_model']['name'],
        num_classes=num_classes,
        pretrained=config['task_model']['pretrained'],
        epochs=config['training']['epochs']
    ).to(device)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=float(config['training']['weight_decay']),
        nesterov=config['training']['nesterov']
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    scaler = GradScaler() if config['training'].get('use_amp', False) else None
    
    history = []
    
    for epoch in range(config['training']['epochs']):
        train_loss, train_task, train_comp = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, config, scaler
        )
        scheduler.step()
        
        val_acc, val_sparsity, val_latency, val_energy = validate(model, val_loader, device, config)
        
        print(f"Epoch {epoch+1}: Loss {train_loss:.4f} | Acc {val_acc:.2f}% | Sparsity {val_sparsity:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_sparsity': val_sparsity,
            'val_latency': val_latency,
            'val_energy': val_energy
        })
        
    return model, history

if __name__ == '__main__':
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Example run with a single seed
    model, history = run_experiment(config, config['training']['seed'])
