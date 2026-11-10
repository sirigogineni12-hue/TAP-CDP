"""Count multiply-accumulates (fvcore convention, the one behind "ResNet-50 = 4.1 GFLOPs").

Reports, per backbone and input size:
  * pixel ResNet on 3 x H x W,
  * the DCT backbone on 64 x H/8 x W/8 (CI-DCT, and TAP-CDP as implemented: zero-masking
    keeps all 64 input channels, so its cost is identical),
  * the DCT backbone with the stem physically pruned to k input channels, for every k,
  * every input/stem design of the revision (colour x stem; coefficient normalization adds no MACs),
    with all channels and with the stem pruned to k channels, and the CIFAR-stem pixel ResNet (CIFAR-10 only).
Output: flops.json
"""
import argparse
import json

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from common import COMPONENTS, DctClassifier, ModifiedResNet, PixelClassifier, write_json

DESIGNS = [(c, s) for c in ("y", "ycbcr") for s in ("pool", "nopool", "nopool_s1")]
KS = {64: [1, 4, 8, 12, 16, 24, 32, 48, 64], 192: [1, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192]}


def count(model, shape):
    model.eval()
    with torch.no_grad():
        fca = FlopCountAnalysis(model, torch.zeros(1, *shape))
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return int(fca.total())


def dct_backbone(name, num_classes, k=64):
    m = ModifiedResNet(name=name, num_classes=num_classes, pretrained=False)
    if k != 64:
        m.stem[0] = nn.Conv2d(k, 64, kernel_size=1, stride=1, bias=False)
    return m


def design_backbone(name, num_classes, color, stem, k=None):
    m = DctClassifier(name, num_classes, "none", pretrained=False, color=color, stem=stem).backbone
    if k is not None:
        m.stem[0] = nn.Conv2d(k, 64, kernel_size=1, stride=1, bias=False)
    return m


def block_dct_macs(h, w):
    # Two 8x8 matrix products per block: 2 * 8*8*8 MACs.
    return (h // 8) * (w // 8) * 2 * 8 * 8 * 8


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="flops.json")
    a = p.parse_args()
    settings = [("cifar10", 10, 32), ("imagenet", 1000, 224), ("imagenette", 10, 224)]
    out = {"convention": "fvcore count; one multiply-accumulate = 1 FLOP", "results": []}
    for dataset, ncls, size in settings:
        for backbone in ["resnet18", "resnet50"]:
            pixel = count(PixelClassifier(backbone, ncls, pretrained=False), (3, size, size))
            g = size // 8
            dct64 = count(dct_backbone(backbone, ncls), (64, g, g))
            pruned = {k: count(dct_backbone(backbone, ncls, k), (k, g, g)) for k in [1, 4, 8, 12, 16, 24, 32, 48, 64]}
            rec = {"dataset": dataset, "backbone": backbone, "input_size": size,
                   "pixel_macs": pixel, "dct_backbone_macs_64ch": dct64,
                   "block_dct_macs": block_dct_macs(size, size),
                   "dct_backbone_macs_pruned_k": pruned,
                   "note": "CI-DCT and TAP-CDP-as-implemented (zero-masking) both equal dct_backbone_macs_64ch"}
            if dataset == "cifar10":
                rec["pixel_cifar_stem_macs"] = count(PixelClassifier(backbone, ncls, pretrained=False, stem="cifar"),
                                                     (3, size, size))
            rec["designs"] = {}
            for color, stem in DESIGNS:
                c = 64 * COMPONENTS[color]
                rec["designs"][f"{color}/{stem}"] = {
                    "channels": c,
                    "macs_all_channels": count(design_backbone(backbone, ncls, color, stem), (c, g, g)),
                    "macs_pruned_k": {k: count(design_backbone(backbone, ncls, color, stem, k), (k, g, g)) for k in KS[c]},
                    "block_dct_macs": block_dct_macs(size, size) * COMPONENTS[color]}
            out["results"].append(rec)
            print(f"{dataset:8s} {backbone}: pixel {pixel / 1e9:.3f} G | DCT 64ch {dct64 / 1e9:.4f} G | "
                  f"pruned k=16 {pruned[16] / 1e9:.4f} G | block DCT {rec['block_dct_macs'] / 1e9:.5f} G")
            print("    designs (all channels): " + " | ".join(f"{d} {v['macs_all_channels'] / 1e9:.4f} G"
                                                          for d, v in rec["designs"].items()))
    write_json(a.out, out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
