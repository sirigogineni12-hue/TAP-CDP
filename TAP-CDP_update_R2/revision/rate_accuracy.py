"""Real bit-rate versus accuracy, measured on actual JPEG files.

For each test image (center-cropped to the network input size) and each JPEG quality Q:
  * pixel baseline: standard libjpeg tables at quality Q,
  * luminance-only DCT models: the luma table at quality Q with every band the model discards set to 255
    (the coarsest 8-bit step), chroma at quality Q, 4:2:0 subsampling (as the pixel baseline);
  * YCbCr DCT models (192 channels): the same for luma, and the chroma table masked by the union of the
    Cb and Cr masks (baseline JPEG shares one table between Cb and Cr); 4:4:4, as in training.
For the input-dependent Xu et al. gate, the static mask is the set of channels kept for more than half of
the test images (Xu et al. likewise derive a static selection from the gate statistics).
The encoded bytes are measured directly; bpp = 8 * bytes / (H * W) is reported both for the
whole file (including headers) and for the entropy-coded scan only. Accuracy is computed on
the decoded files, using each model's normal preprocessing.
Output: rate_accuracy.json
"""
import argparse
import io
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from common import MEAN, STD, IMAGE_SIZE, PixelClassifier, block_dct, env_info, load_run, rgb_to_y, write_json

LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
CHROMA = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
          47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32


def scaled(table, q):
    """libjpeg quality scaling (verified identical to Pillow's own tables); natural/raster order."""
    s = 5000 // q if q < 50 else 200 - 2 * q
    return [min(255, max(1, (t * s + 50) // 100)) for t in table]


def scan_bytes(data):
    """Bytes of entropy-coded data in a baseline single-scan JPEG (SOS payload up to EOI)."""
    pos = data.find(b"\xff\xda")
    seg_len = int.from_bytes(data[pos + 2:pos + 4], "big")
    return len(data) - 2 - (pos + 2 + seg_len)


class Identity:
    def __call__(self, x):
        return x


class EncodedSet(Dataset):
    def __init__(self, base, crop, qtables, mean, std, subsampling=2):
        self.base, self.crop, self.qtables, self.subsampling = base, crop, qtables, subsampling
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, target = self.base[i]
        img = self.crop(img.convert("RGB"))
        buf = io.BytesIO()
        img.save(buf, "JPEG", qtables=self.qtables, subsampling=self.subsampling)
        data = buf.getvalue()
        dec = Image.open(io.BytesIO(data)).convert("RGB")
        return self.to_tensor(dec), target, len(data), scan_bytes(data)


def design_label(cfg):
    if cfg["mode"] == "pixel":
        return f"pixel-{cfg.get('stem', 'pool')}"
    return f"{cfg.get('color', 'y')}/{cfg.get('stem', 'pool')}" + ("/norm" if cfg.get("dct_norm") else "")


def masked_luma(q, mask_binary):
    t = scaled(LUMA, q)
    return [v if keep > 0.5 else 255 for v, keep in zip(t, mask_binary[:64])]


def encoder_settings(q, mask):
    """(quantization tables, subsampling) for a model; mask None = pixel baseline (standard JPEG)."""
    if mask is None:
        return [scaled(LUMA, q), scaled(CHROMA, q)], 2
    if len(mask) == 64:
        return [masked_luma(q, mask), scaled(CHROMA, q)], 2
    chroma_keep = [max(cb, cr) for cb, cr in zip(mask[64:128], mask[128:192])]
    chroma = [v if keep > 0.5 else 255 for v, keep in zip(scaled(CHROMA, q), chroma_keep)]
    return [masked_luma(q, mask), chroma], 0


@torch.no_grad()
def masked_band_leakage(base, crop, mask_binary, q, n=64):
    """Fraction of luma DCT energy left in discarded bands after encoding (sanity check that
    the 255 quantizer effectively removes them; computed on unnormalized Y of the decoded image)."""
    disc = torch.tensor([b < 0.5 for b in mask_binary[:64]])
    if not disc.any():
        return 0.0
    num = den = 0.0
    for i in range(min(n, len(base))):
        img = crop(base[i][0].convert("RGB"))
        buf = io.BytesIO()
        qt, sub = encoder_settings(q, mask_binary)
        img.save(buf, "JPEG", qtables=qt, subsampling=sub)
        y = torch.from_numpy(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("YCbCr"))[..., 0].copy()).float()
        c = block_dct((y - 128.0)[None, None]).squeeze(0) ** 2
        num += c[disc].sum().item(); den += c[1:].sum().item()
    return num / max(den, 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "imagenet", "imagenette"], required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--runs", nargs="*", default=[], help="run directories of trained models (DCT or pixel)")
    p.add_argument("--pretrained_pixel", default=None, help="also evaluate torchvision pretrained resnet18/resnet50 (ImageNet only)")
    p.add_argument("--qualities", type=int, nargs="+", default=[10, 30, 50, 70, 90])
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--designs", nargs="*", default=None,
                   help="only these designs, e.g. y/pool ycbcr/nopool_s1/norm pixel-pool pixel-cifar (default: all)")
    p.add_argument("--out", default="rate_accuracy.json")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.dataset == "cifar10":
        base = datasets.CIFAR10(root=a.data_root, train=False, download=True)
        crop = Identity()
    else:   # imagenet / imagenette: the val folder is the test set
        base = datasets.ImageFolder(f"{a.data_root}/val")
        crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    if a.max_images:
        base = torch.utils.data.Subset(base, range(min(a.max_images, len(base))))
    size = IMAGE_SIZE[a.dataset]

    models = []
    for r in a.runs:
        with open(f"{r}/config.json") as f:
            cfg = json.load(f)
        if cfg["mode"] == "learned" and cfg.get("tacs_init", 0.0) == 0.0:
            continue   # repository initialization: diverged in every run (chance accuracy), nothing to measure
        if a.designs is not None and design_label(cfg) not in a.designs:
            continue
        model, cfg, res = load_run(r, dev)
        models.append((res["run_id"], cfg["mode"], model, res.get("mask_binary")))
    print(f"{len(models)} models to measure", flush=True)
    if a.pretrained_pixel:
        assert a.dataset == "imagenet"
        models.append((f"torchvision_{a.pretrained_pixel}_pretrained", "pixel",
                       PixelClassifier(a.pretrained_pixel, 1000).to(dev).eval(), None))

    out = {"method": __doc__, "env": env_info(), "image_size": size, "results": []}
    for name, mode, model, mask in models:
        for q in a.qualities:
            qt, sub = encoder_settings(q, None if mode == "pixel" else mask)
            # Workers start from a clean forkserver process: forking this process after it has used the CPU
            # thread pool (masked_band_leakage) can deadlock the workers on Linux (job 61875 hung this way).
            loader = DataLoader(EncodedSet(base, crop, qt, MEAN[a.dataset], STD[a.dataset], sub),
                                batch_size=a.batch_size, num_workers=a.workers,
                                multiprocessing_context="forkserver" if a.workers > 0 else None)
            correct = n = tot_bytes = tot_scan = 0
            for x, y, nb, ns in loader:
                logits, _ = model(x.to(dev))
                correct += (logits.argmax(1).cpu() == y).sum().item()
                n += y.numel(); tot_bytes += nb.sum().item(); tot_scan += ns.sum().item()
            rec = {"model": name, "mode": mode, "quality": q, "n_images": n, "accuracy": 100.0 * correct / n,
                   "bytes_mean": tot_bytes / n, "header_bytes_mean": (tot_bytes - tot_scan) / n,
                   "bpp_total": 8.0 * tot_bytes / n / (size * size), "bpp_scan_only": 8.0 * tot_scan / n / (size * size),
                   "chroma_subsampling": {2: "4:2:0", 0: "4:4:4"}[sub]}
            if mode != "pixel":
                rec["active_bands"] = int(sum(mask))
                rec["discarded_band_energy_fraction"] = masked_band_leakage(base, crop, mask, q)
            out["results"].append(rec)
            print(f"{name} Q={q}: acc {rec['accuracy']:.2f}% | {rec['bytes_mean']:.0f} B | "
                  f"{rec['bpp_total']:.3f} bpp (scan only {rec['bpp_scan_only']:.3f})", flush=True)
    write_json(a.out, out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
