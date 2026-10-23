"""Shared helpers for the TAP-CDP revision experiments.

TACS and ModifiedResNet are imported unchanged from the repository's tap_hybridnet.py (block_dct is
reimplemented because the original crashes on 32x32 and 224x224). Design options added for the revision
(colour, stem, coefficient normalization, the Xu et al. gate) default to the repository behaviour.
"""
import hashlib
import io
import json
import math
import os
import platform
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_DIR)  # append, so repo files never shadow standard-library modules
from tap_hybridnet import TACS, ModifiedResNet  # noqa: E402
from tap_hybridnet import block_dct as repo_block_dct  # noqa: E402,F401  (kept for the equivalence test)


def _dct_matrix(device, dtype):
    n = torch.arange(8, dtype=torch.float64)
    k = n.view(-1, 1)
    t = torch.cos(math.pi * (2 * n + 1) * k / 16) * math.sqrt(2.0 / 8)
    t[0] = t[0] / math.sqrt(2.0)
    return t.to(device=device, dtype=dtype)


def block_dct(x):
    """8x8 orthonormal block DCT-II: (B, C, H, W) -> (B, C*64, H/8, W/8), band index = vertical*8 + horizontal.

    Same computation as tap_hybridnet.block_dct, which crashes unless H = 512 (its final reshape reads the
    shape of the tensor before the permute). tests_block_dct() checks both agree where the original runs.
    """
    b, c, h, w = x.shape
    assert h % 8 == 0 and w % 8 == 0, "image size must be a multiple of 8"
    blocks = x.unfold(2, 8, 8).unfold(3, 8, 8)            # B, C, H/8, W/8, 8, 8
    t = _dct_matrix(x.device, x.dtype)
    coeffs = t @ blocks @ t.T                              # T X T^T per block
    hb, wb = coeffs.shape[2], coeffs.shape[3]
    return coeffs.reshape(b, c, hb, wb, 64).permute(0, 1, 4, 2, 3).reshape(b, c * 64, hb, wb)

# Imagenette (fast.ai): 10 ImageNet classes; its val images come from the ImageNet validation set, so the
# ImageNet-pretrained weights have not seen them. Same normalization and input size as ImageNet.
MEAN = {"cifar10": (0.4914, 0.4822, 0.4465), "imagenet": (0.485, 0.456, 0.406), "imagenette": (0.485, 0.456, 0.406)}
STD = {"cifar10": (0.2023, 0.1994, 0.2010), "imagenet": (0.229, 0.224, 0.225), "imagenette": (0.229, 0.224, 0.225)}
NUM_CLASSES = {"cifar10": 10, "imagenet": 1000, "imagenette": 10}
IMAGE_SIZE = {"cifar10": 32, "imagenet": 224, "imagenette": 224}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env_info():
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        info["nvidia_smi"] = None
    return info


def code_fingerprint():
    """SHA-256 of the scripts and the method file, stored with every result for provenance."""
    files = [os.path.join(REPO_DIR, "tap_hybridnet.py")]
    here = os.path.dirname(os.path.abspath(__file__))
    files += sorted(os.path.join(here, f) for f in os.listdir(here) if f.endswith(".py"))
    h = hashlib.sha256()
    for f in files:
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# DCT coefficient ordering (block_dct returns index = vertical*8 + horizontal)
# ----------------------------------------------------------------------------

def zigzag_order():
    """Raster indices (row*8+col) of the 64 DCT bands in standard JPEG zigzag order."""
    cells = [(r, c) for r in range(8) for c in range(8)]
    cells.sort(key=lambda rc: (rc[0] + rc[1], rc[0] if (rc[0] + rc[1]) % 2 else -rc[0]))
    return [r * 8 + c for r, c in cells]


def luma_weighted_split(k):
    """Bands per component (Y, Cb, Cr) for a k-band low-frequency mask on YCbCr.
    About 2/3 to luma, as in the static selections of Xu et al. (CVPR 2020): 24 -> (16, 4, 4), 48 -> (32, 8, 8)."""
    ny = min(64, round(2 * k / 3))
    ncb = (k - ny) // 2
    return ny, ncb, k - ny - ncb


def fixed_mask(mode, k, mask_seed, components=1):
    """Channel index = component*64 + band (component 0 = Y, then Cb, Cr)."""
    n = 64 * components
    mask = torch.zeros(n)
    if mode == "lowfreq":
        counts = [k] if components == 1 else luma_weighted_split(k)
        for comp, kc in enumerate(counts):
            mask[[comp * 64 + i for i in zigzag_order()[:kc]]] = 1.0
    elif mode == "random":
        g = torch.Generator().manual_seed(mask_seed)
        mask[torch.randperm(n, generator=g)[:k]] = 1.0
    elif mode == "none":
        mask[:] = 1.0
    else:
        raise ValueError(mode)
    return mask


def rgb_to_y(x):
    """Same luminance conversion as the authors' train.py (applied to normalized RGB)."""
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def rgb_to_ycbcr(x):
    """Full-resolution (4:4:4) YCbCr with the JPEG (BT.601) matrix, applied to normalized RGB like rgb_to_y."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.cat([y, cb, cr], 1)


COMPONENTS = {"y": 1, "ycbcr": 3}


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

def apply_stem(backbone, stem):
    """Spatial design of the DCT backbone. 'pool' = repository (max-pool after the 1x1 stem);
    'nopool' drops that max-pool, as the repository comment intends for CIFAR-10 and as Xu et al. do;
    'nopool_s1' also makes the first block of layer2 stride 1, keeping one more resolution level."""
    if stem in ("nopool", "nopool_s1"):
        backbone.stem[3] = nn.Identity()
    if stem == "nopool_s1":
        for m in backbone.layer2[0].modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride = (1, 1)
    elif stem not in ("pool", "nopool"):
        raise ValueError(stem)


class TopKMask(nn.Module):
    """Budgeted TACS: keeps exactly k channels. Learned scores s; the k highest are kept (hard top-k), and the
    gradient reaches s through sigmoid(s) (straight-through), so channels that are off still receive the signal
    of how much they would help. Replaces the lambda penalty, which does not control the number of channels.
    Scores start at N(0, 1e-3) with a seeded generator, so the initial selection is random and is then learned."""

    def __init__(self, channels, k, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.k = k
        self.s = nn.Parameter(torch.randn(channels, generator=g) * 1e-3)

    def hard(self):
        m = torch.zeros_like(self.s)
        m[torch.topk(self.s.detach(), self.k).indices] = 1.0
        return m

    def forward(self, x):
        hard = self.hard()
        mask = hard + torch.sigmoid(self.s) - torch.sigmoid(self.s).detach() if self.training else hard
        return x * mask.view(1, -1, 1, 1), hard


class XuGate(nn.Module):
    """Input-dependent channel gate after Xu et al., "Learning in the Frequency Domain" (CVPR 2020), Sec. 3.2
    (our re-implementation): average pooling -> 1x1 conv (C -> C) -> two numbers per channel from per-channel
    trainable parameters -> hard Gumbel-softmax during training. At evaluation the more probable state is used
    (deterministic). The bias starts every gate on (P(on) ~ 0.95), matching the all-on start used for TACS."""

    def __init__(self, channels, init_on=3.0):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.ones(2, channels))
        self.bias = nn.Parameter(torch.tensor([[init_on], [0.0]]).repeat(1, channels))   # row 0 = on, row 1 = off
        self.register_buffer("on_sum", torch.zeros(channels))
        self.register_buffer("n_seen", torch.zeros(()))

    def reset_stats(self):
        self.on_sum.zero_(); self.n_seen.zero_()

    def forward(self, x):
        t3 = self.fc(x.mean((2, 3), keepdim=True)).flatten(1)                      # B x C
        logits = torch.stack([t3 * self.scale[0] + self.bias[0], t3 * self.scale[1] + self.bias[1]], -1)
        if self.training:
            gate = nn.functional.gumbel_softmax(logits, tau=1.0, hard=True)[..., 0]
        else:
            gate = (logits[..., 0] > logits[..., 1]).float()
            self.on_sum += gate.sum(0).detach(); self.n_seen += gate.shape[0]
        return x * gate[:, :, None, None], gate


class DctClassifier(nn.Module):
    """RGB -> Y (or YCbCr) -> 8x8 block DCT (64 channels per component) -> [normalize] -> mask -> ModifiedResNet.

    mode='learned' uses the authors' TACS module; 'topk' budgeted TACS (exactly k channels); 'xugate' the
    Xu et al. gate; 'none' is CI-DCT (all bands);
    'random' and 'lowfreq' use a fixed mask with k bands. color, stem and dct_norm default to the repository
    design (luminance only, max-pool kept, no coefficient normalization).
    """

    def __init__(self, backbone, num_classes, mode, k=None, mask_seed=0, epochs=100,
                 pretrained=True, tau_start=5.0, tau_end=0.1, tacs_init=0.0, color="y", stem="pool", dct_norm=False):
        super().__init__()
        self.mode, self.color, self.dct_norm = mode, color, dct_norm
        c = 64 * COMPONENTS[color]
        self.backbone = ModifiedResNet(name=backbone, num_classes=num_classes, pretrained=pretrained)
        if c != 64:
            self.backbone.stem[0] = nn.Conv2d(c, 64, kernel_size=1, stride=1, bias=False)
        apply_stem(self.backbone, stem)
        # Per-channel mean/std of the DCT coefficients on training images (Xu et al.); set by set_dct_stats().
        self.register_buffer("dct_mean", torch.zeros(c))
        self.register_buffer("dct_std", torch.ones(c))
        if mode == "learned":
            self.tacs = TACS(channels=c, tau_start=tau_start, tau_end=tau_end, total_epochs=epochs)
            # The repository initializes w = 0, so sigmoid(0) = 0.5 fails the strict "> 0.5" test and every
            # band starts switched off. tacs_init > 0 starts with all bands on; 0.0 reproduces the repository.
            nn.init.constant_(self.tacs.w, tacs_init)
        elif mode == "xugate":
            self.gate = XuGate(c)
        elif mode == "topk":
            self.topk = TopKMask(c, k, mask_seed)
        else:
            self.register_buffer("fixed", fixed_mask(mode, c if mode == "none" else k, mask_seed, COMPONENTS[color]))

    def dct_features(self, rgb):
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            x = rgb.float()
            return block_dct(rgb_to_y(x) if self.color == "y" else rgb_to_ycbcr(x))

    def set_dct_stats(self, mean, std):
        self.dct_mean.copy_(mean); self.dct_std.copy_(std.clamp_min(1e-6))

    def forward(self, rgb):
        coeffs = self.dct_features(rgb)
        if self.dct_norm:
            coeffs = (coeffs - self.dct_mean.view(1, -1, 1, 1)) / self.dct_std.view(1, -1, 1, 1)
        if self.mode == "learned":
            z, m = self.tacs(coeffs)
        elif self.mode == "xugate":
            z, m = self.gate(coeffs)
        elif self.mode == "topk":
            z, m = self.topk(coeffs)
        else:
            z, m = coeffs * self.fixed.view(1, -1, 1, 1), self.fixed
        return self.backbone(z), m

    def mask_state(self):
        """Return (probabilities, binary mask, tau) as used at evaluation time. For the input-dependent gate,
        the probability is the fraction of images of the last evaluation for which each channel was on."""
        if self.mode == "learned":
            tau = self.tacs.get_temperature()
            probs = torch.sigmoid(self.tacs.w.detach().float().cpu() / tau)
            return probs, (probs > 0.5).float(), tau
        if self.mode == "xugate":
            freq = (self.gate.on_sum / self.gate.n_seen.clamp_min(1)).cpu()
            return freq, (freq > 0.5).float(), None
        if self.mode == "topk":
            return torch.sigmoid(self.topk.s.detach().float().cpu()), self.topk.hard().cpu(), None
        return self.fixed.cpu(), self.fixed.cpu(), None


@torch.no_grad()
def estimate_dct_stats(model, loader, device, max_images=10000):
    """Per-channel mean and std of the (unnormalized) DCT coefficients over training images."""
    s = s2 = None
    n = 0
    for x, _ in loader:
        f = model.dct_features(x.to(device)).double()
        cur = f.sum((0, 2, 3)), (f * f).sum((0, 2, 3))
        s, s2 = (cur[0], cur[1]) if s is None else (s + cur[0], s2 + cur[1])
        n += f.shape[0] * f.shape[2] * f.shape[3]
        if n >= max_images * f.shape[2] * f.shape[3]:
            break
    mean = s / n
    return mean.float(), (s2 / n - mean * mean).clamp_min(0).sqrt().float()


class PixelClassifier(nn.Module):
    """Standard torchvision ResNet on normalized RGB (pixel-domain baseline).
    stem='cifar' is the usual CIFAR adaptation: 3x3 stride-1 first convolution and no max-pool."""

    def __init__(self, backbone, num_classes, pretrained=True, stem="pool"):
        super().__init__()
        weights = {"resnet18": "IMAGENET1K_V1", "resnet50": "IMAGENET1K_V1"}[backbone] if pretrained else None
        self.net = getattr(torchvision.models, backbone)(weights=weights)
        if num_classes != 1000:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        if stem == "cifar":
            self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.net.maxpool = nn.Identity()
        elif stem != "pool":
            raise ValueError(stem)

    def forward(self, rgb):
        return self.net(rgb), None


def build_model(args, num_classes):
    stem = getattr(args, "stem", "pool")
    if args.mode == "pixel":
        return PixelClassifier(args.backbone, num_classes, pretrained=not args.no_pretrained, stem=stem)
    return DctClassifier(args.backbone, num_classes, args.mode, k=args.k, mask_seed=args.seed,
                         epochs=args.epochs, pretrained=not args.no_pretrained,
                         tacs_init=getattr(args, "tacs_init", 0.0), color=getattr(args, "color", "y"),
                         stem=stem, dct_norm=getattr(args, "dct_norm", False))


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

class JpegReencode:
    """Encode a PIL image as JPEG at a given quality and decode it again."""

    def __init__(self, quality):
        self.quality = quality

    def __call__(self, img):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class Subset(Dataset):
    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, target = self.base[self.indices[i]]
        return self.transform(img), target


def eval_transform(dataset, jpeg_quality=None, to_tensor=True):
    ops = [] if dataset == "cifar10" else [transforms.Resize(256), transforms.CenterCrop(224)]
    if jpeg_quality is not None:
        ops.append(JpegReencode(jpeg_quality))
    if to_tensor:
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN[dataset], STD[dataset])]
    return transforms.Compose(ops)


def train_transform(dataset):
    # Identical to the authors' dataset.py.
    if dataset == "cifar10":
        ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    elif dataset == "imagenette":   # torchvision reference recipe without colour jitter (8 CPUs decode the images)
        ops = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    else:
        ops = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
               transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN[dataset], STD[dataset])])


def heldout_split(targets, val_per_class, split_seed=0):
    """Per-class held-out split of the training set, fixed independently of the run seed."""
    rng = np.random.RandomState(split_seed)
    targets = np.asarray(targets)
    val = []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        val.extend(rng.choice(idx, size=min(val_per_class, len(idx)), replace=False).tolist())
    val_set = set(val)
    train = [i for i in range(len(targets)) if i not in val_set]
    return train, sorted(val)


def build_datasets(dataset, root, val_per_class, test_jpeg_quality=None):
    """Returns (train, heldout_val, test). The test set is never used for model selection."""
    if dataset == "cifar10":
        base_train = datasets.CIFAR10(root=root, train=True, download=True)
        base_test = datasets.CIFAR10(root=root, train=False, download=True)
        targets = base_train.targets
    elif dataset in ("imagenet", "imagenette"):
        base_train = datasets.ImageFolder(os.path.join(root, "train"))
        base_test = datasets.ImageFolder(os.path.join(root, "val"))
        targets = base_train.targets
    else:
        raise ValueError(dataset)
    train_idx, val_idx = heldout_split(targets, val_per_class)
    train = Subset(base_train, train_idx, train_transform(dataset))
    val = Subset(base_train, val_idx, eval_transform(dataset))
    test = Subset(base_test, range(len(base_test)), eval_transform(dataset, test_jpeg_quality))
    return train, val, test


def load_run(run_dir, device="cpu"):
    """Rebuild a trained model from runs/<id>. TACS stores its epoch outside the state dict,
    so it is restored here; otherwise the mask would be evaluated at tau_start instead of tau_end."""
    import argparse
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "result.json")) as f:
        res = json.load(f)
    args = argparse.Namespace(**cfg)
    model = build_model(args, NUM_CLASSES[cfg["dataset"]])
    ck = torch.load(os.path.join(run_dir, "last.pt"), map_location="cpu")
    # Runs of the first batch were saved before the normalization buffers existed; they never use them
    # (dct_norm off), so the defaults (mean 0, std 1) are exact. Any other mismatch is an error.
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    allowed = set() if cfg.get("dct_norm") else {"dct_mean", "dct_std"}
    if unexpected or set(missing) - allowed:
        raise RuntimeError(f"{run_dir}: checkpoint does not match the model (missing {missing}, unexpected {unexpected})")
    if cfg["mode"] == "learned":
        model.tacs.current_epoch = max(cfg["epochs"] - 1, 0)
    if cfg["mode"] not in ("pixel", "xugate"):
        assert model.mask_state()[1].tolist() == res["mask_binary"], "reloaded mask differs from saved mask"
    return model.to(device).eval(), cfg, res
