import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

# -----------------------------
# Utils: JPEG DCT Extraction
# -----------------------------

def block_dct(x):
    """
    Computes the 8x8 Block DCT of an image tensor.
    x: (B, 1, H, W) luminance channel
    Returns: (B, 64, H/8, W/8) spectral coefficients
    """
    B, C, H, W = x.shape
    # Unfold into 8x8 blocks
    x = x.unfold(2, 8, 8).unfold(3, 8, 8)  # B, C, H/8, W/8, 8, 8
    x = x.contiguous()

    # Precompute DCT matrix
    N = 8
    dct_mat = torch.zeros((N, N), device=x.device)
    for k in range(N):
        for n in range(N):
            alpha = math.sqrt(1/N) if k == 0 else math.sqrt(2/N)
            dct_mat[k, n] = alpha * math.cos((math.pi * (2*n+1) * k) / (2*N))

    # Apply DCT: T * X * T'
    x = torch.einsum('ij,bcxyjk->bcxyik', dct_mat, x)
    x = torch.einsum('bcxyik,kj->bcxyij', x, dct_mat.t())

    # Reshape to (B, 64, H/8, W/8)
    x = x.reshape(B, C, x.shape[2], x.shape[3], 64)
    x = x.permute(0, 1, 4, 2, 3).reshape(B, 64, x.shape[3], x.shape[4]).contiguous()
    return x

# -----------------------------
# TACS Module
# -----------------------------

class TACS(nn.Module):
    """
    Task-Aware Channel Selection (TACS) module.
    Learns a binary mask for spectral coefficients.
    """
    def __init__(self, channels=64, tau_start=5.0, tau_end=0.1, total_epochs=100):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(channels))
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def get_temperature(self):
        """Linearly anneal temperature from tau_start to tau_end."""
        if self.total_epochs <= 1:
            return self.tau_end
        progress = min(self.current_epoch / (self.total_epochs - 1), 1.0)
        return self.tau_start + progress * (self.tau_end - self.tau_start)

    def forward(self, x):
        tau = self.get_temperature()
        m = torch.sigmoid(self.w / tau)

        if self.training:
            # Straight-Through Estimator (STE)
            mask = (m > 0.5).float()
            mask = mask + m - m.detach()
        else:
            mask = (m > 0.5).float()

        return x * mask.view(1, -1, 1, 1), m

# -----------------------------
# Modified ResNet Backbones
# -----------------------------

class ModifiedResNet(nn.Module):
    """
    ResNet backbone modified to accept 64-channel DCT coefficients.
    The initial 7x7 conv is replaced by a 1x1 conv.
    """
    def __init__(self, name='resnet18', num_classes=10, pretrained=True):
        super().__init__()
        if name == 'resnet18':
            base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        elif name == 'resnet50':
            base_model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f"Unsupported ResNet: {name}")

        # Replace initial conv (7x7, stride 2, pad 3) with 1x1 conv
        # The input DCT has 64 channels. We map it to 64 output channels to match ResNet stem.
        self.stem = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
            base_model.bn1,
            base_model.relu,
            # Note: We remove maxpool if spatial resolution is already small (CIFAR), 
            # but keep it for ImageNet to maintain architecture consistency.
            base_model.maxpool 
        )

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -----------------------------
# TAP-CDP Framework
# -----------------------------

class TAPCDP(nn.Module):
    def __init__(self, backbone_name='resnet18', num_classes=10, pretrained=True, epochs=100):
        super().__init__()
        self.tacs = TACS(channels=64, total_epochs=epochs)
        self.backbone = ModifiedResNet(name=backbone_name, num_classes=num_classes, pretrained=pretrained)

    def forward(self, x, mask_type='learned'):
        """
        Input x is assumed to be luminance in spatial domain (B, 1, H, W)
        or DCT coefficients if pre-computed.
        mask_type: 'learned' (TACS), 'none' (CI-DCT), or a specific mask tensor.
        """
        if x.shape[1] == 1:
            x = block_dct(x)
        
        if mask_type == 'learned':
            z, m = self.tacs(x)
        elif mask_type == 'none':
            z = x
            m = torch.ones(x.shape[1], device=x.device)
        else:
            z = x * mask_type.view(1, -1, 1, 1)
            m = mask_type

        logits = self.backbone(z)
        return logits, m
