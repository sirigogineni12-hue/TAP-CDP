# losses.py

import torch
import torch.nn as nn
from models import calculate_flops

class RACLoss(nn.Module):
    """
    Multi-objective RAC (Rate-Accuracy-Complexity) loss function.
    Loss = lambda_rate * Rate + lambda_accuracy * Accuracy + lambda_complexity * Complexity
    """
    def __init__(self, config, model, input_shape, device):
        super(RACLoss, self).__init__()
        self.lambda_rate = config['training']['rac_weights']['lambda_rate']
        self.lambda_accuracy = config['training']['rac_weights']['lambda_accuracy']
        self.lambda_complexity = config['training']['rac_weights']['lambda_complexity']
        
        self.accuracy_loss = nn.CrossEntropyLoss()
        
        self.model = model
        self.input_shape = input_shape
        self.device = device

    def forward(self, logits, targets, bpp):
        # Rate loss (BPP)
        rate_loss = bpp
        
        # Accuracy loss (Cross-Entropy)
        acc_loss = self.accuracy_loss(logits, targets)
        
        # Complexity loss (FLOPs)
        if self.lambda_complexity > 0:
            # Note: Calculating FLOPs at every step is very slow.
            # In a real scenario, you might calculate this once per epoch or use a proxy.
            # For this project, we demonstrate the principle.
            complexity_loss = calculate_flops(self.model, self.input_shape, self.device)
            # Normalize FLOPs (e.g., by dividing by 1e9 to get GFLOPs)
            complexity_loss = complexity_loss / 1e9 
        else:
            complexity_loss = torch.tensor(0.0).to(self.device)

        # Total loss
        total_loss = (self.lambda_rate * rate_loss +
                      self.lambda_accuracy * acc_loss +
                      self.lambda_complexity * complexity_loss)
        
        return total_loss, rate_loss, acc_loss, complexity_loss

if __name__ == '__main__':
    import yaml
    from models import get_full_model
    from compression import CompressionModel

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a dummy model
    compression_model = CompressionModel(config).to(device)
    full_model = get_full_model(config, num_classes=10, compression_model=compression_model).to(device)
    
    # Create the loss function
    input_shape = (1, 3, 32, 32)
    rac_loss_fn = RACLoss(config, full_model, input_shape, device)
    
    # Create dummy inputs
    dummy_images = torch.rand(4, 3, 32, 32).to(device)
    dummy_labels = torch.randint(0, 10, (4,)).to(device)
    
    # Forward pass
    logits, bpp = full_model(dummy_images)
    
    # Calculate loss
    total_loss, rate_loss, acc_loss, complexity_loss = rac_loss_fn(logits, dummy_labels, bpp)
    
    print(f"--- RAC Loss Test ---")
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Rate Loss (BPP): {rate_loss.item():.4f}")
    print(f"Accuracy Loss: {acc_loss.item():.4f}")
    print(f"Complexity Loss (GFLOPs): {complexity_loss.item():.4f}")

    # Test with complexity weight
    config['training']['rac_weights']['lambda_complexity'] = 1e-5
    rac_loss_fn_with_complexity = RACLoss(config, full_model, input_shape, device)
    total_loss, _, _, _ = rac_loss_fn_with_complexity(logits, dummy_labels, bpp)
    print(f"\n--- RAC Loss with Complexity Weight ---")
    print(f"Total Loss: {total_loss.item():.4f}")
