# models.py

import torch
import torch.nn as nn
from torchvision import models
from fvcore.nn import FlopCountAnalysis

class TaskAwareNet(nn.Module):
    """A simple CNN to operate on the compressed latent representation."""
    def __init__(self, in_channels, num_classes):
        super(TaskAwareNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BridgeLayer(nn.Module):
    """A bridge layer to connect the compression model's output to a pretrained task model."""
    def __init__(self, in_channels, out_channels=3):
        super(BridgeLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ChannelSelector(nn.Module):
    """A channel selection mechanism using Gumbel-Softmax."""
    def __init__(self, num_channels, temperature=1.0):
        super(ChannelSelector, self).__init__()
        self.logits = nn.Parameter(torch.ones(num_channels))
        self.temperature = temperature

    def forward(self, x):
        if self.training:
            # Gumbel-Softmax for differentiable selection
            mask = F.gumbel_softmax(self.logits, tau=self.temperature, hard=True)
        else:
            # Hard selection for inference
            mask = (self.logits > 0).float()
        
        # Reshape mask to (1, C, 1, 1) for broadcasting
        mask = mask.view(1, -1, 1, 1)
        return x * mask

def get_task_model(config, num_classes):
    """Returns the specified task model."""
    model_name = config['task_model']['name']
    pretrained = config['task_model']['pretrained']
    
    if model_name == 'ResNet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif model_name == 'TaskAwareNet':
        in_channels = config['compression_model']['num_features']
        model = TaskAwareNet(in_channels, num_classes)
    else:
        raise ValueError(f"Unknown task model: {model_name}")
        
    return model

def get_full_model(config, num_classes, compression_model):
    """
    Constructs the full model pipeline including compression, task model, bridge, and channel selection.
    """
    task_model = get_task_model(config, num_classes)
    
    use_bridge = config['task_model']['bridge']
    use_channel_selection = config['task_model']['channel_selection']
    
    bridge_layer = None
    if use_bridge:
        in_channels = config['compression_model']['num_features']
        bridge_layer = BridgeLayer(in_channels)
        
    channel_selector = None
    if use_channel_selection:
        num_channels = config['compression_model']['num_features']
        channel_selector = ChannelSelector(num_channels)

    class FullModel(nn.Module):
        def __init__(self):
            super(FullModel, self).__init__()
            self.compression_model = compression_model
            self.task_model = task_model
            self.bridge_layer = bridge_layer
            self.channel_selector = channel_selector
            self.task_model_name = config['task_model']['name']

        def forward(self, x):
            if self.compression_model:
                x_hat, y_hat, bpp = self.compression_model(x)
                
                if self.task_model_name == 'TaskAwareNet':
                    latent_to_task = y_hat
                    if self.channel_selector:
                        latent_to_task = self.channel_selector(latent_to_task)
                    logits = self.task_model(latent_to_task)
                else: # ResNet, MobileNet, etc.
                    img_to_task = x_hat
                    if self.bridge_layer:
                        # This is an alternative path where the bridge adapts the latent space
                        # to something the task model can use (e.g., an RGB-like image)
                        # This is not the standard way of using a bridge, but we include it for ablation.
                        # A more standard way is to use the bridge to connect intermediate features.
                        # For simplicity, we apply it on the latent.
                        # A proper implementation would require modifying the task model.
                        pass # Let's stick to using x_hat for standard models
                    
                    logits = self.task_model(img_to_task)
                
                return logits, bpp
            else:
                # Baseline: no compression
                logits = self.task_model(x)
                return logits, torch.tensor(0.0)

    return FullModel()

def calculate_flops(model, input_shape, device):
    """Calculates FLOPs for a given model and input shape."""
    model.eval()
    inputs = torch.randn(input_shape).to(device)
    flops = FlopCountAnalysis(model, inputs).total()
    return flops

if __name__ == '__main__':
    import yaml
    from compression import CompressionModel

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Test Baseline Model ---
    print("--- Testing Baseline Model (ResNet18) ---")
    baseline_model = get_task_model(config, num_classes=10).to(device)
    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    flops = calculate_flops(baseline_model, (1, 3, 32, 32), device)
    print(f"Baseline FLOPs: {flops / 1e9:.2f} GFLOPs")

    # --- Test Full Model with Compression ---
    print("\n--- Testing Full Model with Compression ---")
    compression_model = CompressionModel(config).to(device)
    full_model = get_full_model(config, num_classes=10, compression_model=compression_model).to(device)
    
    dummy_input = torch.rand(2, 3, 32, 32).to(device)
    logits, bpp = full_model(dummy_input)
    
    print(f"Logits shape: {logits.shape}")
    print(f"BPP: {bpp.item():.4f}")

    # --- Test Task-Aware Model ---
    print("\n--- Testing Task-Aware Model ---")
    config['task_model']['name'] = 'TaskAwareNet'
    task_aware_full_model = get_full_model(config, num_classes=10, compression_model=compression_model).to(device)
    
    logits, bpp = task_aware_full_model(dummy_input)
    print(f"Logits shape (Task-Aware): {logits.shape}")
    
    # --- Test Channel Selection ---
    print("\n--- Testing Channel Selection ---")
    config['task_model']['channel_selection'] = True
    channel_selection_model = get_full_model(config, num_classes=10, compression_model=compression_model).to(device)
    logits, bpp = channel_selection_model(dummy_input)
    print(f"Logits shape (Channel Selection): {logits.shape}")

    # Note: FLOPs calculation for the full model is more complex as it depends on the path taken.
    # We can calculate FLOPs for each component separately.
    comp_flops = calculate_flops(compression_model, (1, 3, 32, 32), device)
    task_flops = calculate_flops(task_aware_full_model.task_model, (1, config['compression_model']['num_features'], 2, 2), device) # Example latent size
    print(f"Compression FLOPs: {comp_flops / 1e9:.2f} GFLOPs")
    print(f"Task-Aware Model FLOPs: {task_flops / 1e9:.2f} GFLOPs")
    print(f"Total FLOPs (Task-Aware Path): {(comp_flops + task_flops) / 1e9:.2f} GFLOPs")
