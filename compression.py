# compression.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# Quantization
class STEQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def ste_quantize(x):
    return STEQuantize.apply(x)

def additive_uniform_noise(x):
    if x.is_cuda:
        noise = torch.rand_like(x).cuda() - 0.5
    else:
        noise = torch.rand_like(x) - 0.5
    return x + noise

# GDN
class GDN(nn.Module):
    """Generalized Divisive Normalization."""
    def __init__(self, num_features, inverse=False, gamma_init=0.1, beta_min=1e-6):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.gamma = nn.Parameter(torch.eye(num_features) * gamma_init)
        self.beta = nn.Parameter(torch.ones(num_features) * beta_min)
        self.beta_min = beta_min

    def forward(self, x):
        # x: (B, C, H, W)
        x_flat = x.view(x.size(0), x.size(1), -1)
        norm_pool = torch.einsum('bcn,co->bon', x_flat**2, self.gamma)
        norm_pool = norm_pool.view(x.size(0), x.size(1), x.size(2), x.size(3))
        norm_pool = torch.sqrt(norm_pool + self.beta.view(1, -1, 1, 1))
        
        if self.inverse:
            return x * norm_pool
        else:
            return x / norm_pool

# Main Compression Model
class AnalysisTransform(nn.Module):
    def __init__(self, in_channels, num_filters, num_features):
        super(AnalysisTransform, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=9, stride=4, padding=4)
        self.gdn1 = GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2)
        self.gdn2 = GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_features, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.conv3(x)
        return x

class SynthesisTransform(nn.Module):
    def __init__(self, out_channels, num_features, num_filters):
        super(SynthesisTransform, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(num_features, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn1 = GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn2 = GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, out_channels, kernel_size=9, stride=4, padding=4, output_padding=3)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.deconv3(x)
        return x

# Hyperprior
class HyperAnalysisTransform(nn.Module):
    def __init__(self, num_features, num_hyperpriors):
        super(HyperAnalysisTransform, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_hyperpriors, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_hyperpriors, num_hyperpriors, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(num_hyperpriors, num_hyperpriors, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = torch.abs(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

class HyperSynthesisTransform(nn.Module):
    def __init__(self, num_hyperpriors, num_features):
        super(HyperSynthesisTransform, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(num_hyperpriors, num_hyperpriors, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(num_hyperpriors, num_hyperpriors, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(num_hyperpriors, num_features * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.deconv3(x)
        return x

# Full Model
class CompressionModel(nn.Module):
    def __init__(self, config):
        super(CompressionModel, self).__init__()
        num_filters = config['compression_model']['num_filters']
        num_features = config['compression_model']['num_features']
        num_hyperpriors = config['compression_model']['num_hyperpriors']
        quantization = config['compression_model']['quantization']

        self.analysis_transform = AnalysisTransform(3, num_filters, num_features)
        self.synthesis_transform = SynthesisTransform(3, num_features, num_filters)
        
        self.hyper_analysis_transform = HyperAnalysisTransform(num_features, num_hyperpriors)
        self.hyper_synthesis_transform = HyperSynthesisTransform(num_hyperpriors, num_features)

        if quantization == 'ste':
            self.quantize = ste_quantize
        elif quantization == 'noise':
            self.quantize = additive_uniform_noise
        else:
            raise ValueError(f"Unknown quantization method: {quantization}")

    def forward(self, x):
        # Main transform
        y = self.analysis_transform(x)
        
        # Hyperprior transform
        z = self.hyper_analysis_transform(y)
        z_hat = self.quantize(z)
        
        # Estimate sigma from hyperprior
        sigma = self.hyper_synthesis_transform(z_hat)
        sigma_mean, sigma_log_scale = sigma.chunk(2, 1)
        sigma_scale = torch.exp(sigma_log_scale)
        
        # Quantize latent representation y
        y_hat = self.quantize(y)
        
        # Reconstruct image
        x_hat = self.synthesis_transform(y_hat)

        # BPP calculation
        # BPP of y
        # Use Gaussian model for entropy estimation
        likelihood_y = torch.distributions.laplace.Laplace(y, torch.ones_like(y))
        bpp_y = -torch.log2(likelihood_y.cdf(y_hat + 0.5) - likelihood_y.cdf(y_hat - 0.5) + 1e-6).sum() / (x.shape[0] * x.shape[2] * x.shape[3])

        # BPP of z
        likelihood_z = torch.distributions.laplace.Laplace(z, torch.ones_like(z))
        bpp_z = -torch.log2(likelihood_z.cdf(z_hat + 0.5) - likelihood_z.cdf(z_hat - 0.5) + 1e-6).sum() / (x.shape[0] * x.shape[2] * x.shape[3])
        
        bpp = bpp_y + bpp_z

        return x_hat, y_hat, bpp

if __name__ == '__main__':
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = CompressionModel(config).cuda()
    dummy_input = torch.rand(2, 3, 32, 32).cuda()
    x_hat, y_hat, bpp = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Reconstructed image shape: {x_hat.shape}")
    print(f"Latent representation shape: {y_hat.shape}")
    print(f"Estimated BPP: {bpp.item():.4f}")
