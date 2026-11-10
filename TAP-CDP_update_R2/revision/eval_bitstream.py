"""Accuracy of trained models when the input is read directly from JPEG bitstreams.

Every test image (resized to 256, centre-cropped to 224x224) is encoded at each quality Q with the model's
encoder settings (rate_accuracy.encoder_settings: masked tables and 4:4:4 for DCT models, standard tables and
4:2:0 for pixel models). For DCT-domain models trained on JPEG-convention YCbCr (--color ycbcr_jpeg), the
same file is classified in two ways:
  pixel path     : full decode -> RGB -> model (the DCT is computed from the decoded pixels on the GPU),
  bitstream path : entropy decoding only -> dequantized coefficients (bitstream.coefficient_tensor) -> model.
Pixel-domain models use the pixel path only. Also reported: bits per pixel of the files.
Output: bitstream_eval_<dataset>.json
"""
import argparse
import io
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import bitstream as bs
from common import IMAGE_SIZE, MEAN, NUM_CLASSES, STD, PixelClassifier, env_info, load_run
from rate_accuracy import encoder_settings


class EncodedPair(Dataset):
    def __init__(self, base, qtables, subsampling, mean, std, with_coefficients):
        self.base, self.qtables, self.subsampling, self.with_coefficients = base, qtables, subsampling, with_coefficients
        self.crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, target = self.base[i]
        b = io.BytesIO()
        self.crop(img.convert("RGB")).save(b, "JPEG", qtables=self.qtables, subsampling=self.subsampling)
        data = b.getvalue()
        rgb = self.to_tensor(Image.open(io.BytesIO(data)).convert("RGB"))
        coef = torch.from_numpy(bs.coefficient_tensor(data)) if self.with_coefficients else torch.zeros(1)
        return rgb, coef, target, len(data)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["imagenet", "imagenette"], required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--qualities", type=int, nargs="+", default=[50, 90])
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = datasets.ImageFolder(f"{a.data_root}/val")
    if a.max_images:
        idx = np.linspace(0, len(base) - 1, a.max_images).astype(int).tolist()
        base = torch.utils.data.Subset(base, idx)
    size = IMAGE_SIZE[a.dataset]
    out = {"method": __doc__, "env": env_info(), "n_images": len(base), "results": []}
    path = a.out or f"bitstream_eval_{a.dataset}.json"
    for r in a.runs:
        with open(os.path.join(r, "config.json")) as f:
            cfg0 = json.load(f)
        if os.path.exists(os.path.join(r, "last.pt")):
            model, cfg, res = load_run(r, dev)
        elif cfg0["mode"] == "pixel" and cfg0["epochs"] == 0:   # published torchvision weights, never trained here
            cfg = cfg0
            with open(os.path.join(r, "result.json")) as f:
                res = json.load(f)
            model = PixelClassifier(cfg["backbone"], NUM_CLASSES[cfg["dataset"]], pretrained=True).to(dev).eval()
        else:
            print(f"skipping {r}: no checkpoint (last.pt)", flush=True)
            continue
        jpeg_input = cfg["mode"] != "pixel" and cfg.get("color") == "ycbcr_jpeg"
        for q in a.qualities:
            qt, sub = encoder_settings(q, None if cfg["mode"] == "pixel" else res.get("mask_binary"))
            ds = EncodedPair(base, qt, sub, MEAN[a.dataset], STD[a.dataset], jpeg_input)
            loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers,
                                multiprocessing_context="forkserver" if a.workers > 0 else None)
            c_pix = c_bit = n = nbytes = 0
            max_diff = 0.0
            for rgb, coef, y, nb in loader:
                y = y.to(dev)
                logits_pix, _ = model(rgb.to(dev))
                c_pix += (logits_pix.argmax(1) == y).sum().item()
                if jpeg_input:
                    logits_bit, _ = model.classify_coefficients(coef.to(dev))
                    c_bit += (logits_bit.argmax(1) == y).sum().item()
                    max_diff = max(max_diff, (logits_bit.softmax(1) - logits_pix.softmax(1)).abs().max().item())
                n += y.numel()
                nbytes += nb.sum().item()
            rec = {"run_id": res["run_id"], "mode": cfg["mode"], "quality": q, "n_images": n,
                   "chroma_subsampling": {2: "4:2:0", 0: "4:4:4"}[sub],
                   "bpp_total": 8.0 * nbytes / n / (size * size), "acc_pixel_path": 100.0 * c_pix / n}
            if jpeg_input:
                rec.update(acc_bitstream_path=100.0 * c_bit / n, max_softmax_diff_paths=max_diff)
            out["results"].append(rec)
            print(f"{res['run_id']} Q={q}: pixel path {rec['acc_pixel_path']:.2f}%"
                  + (f" | bitstream path {rec['acc_bitstream_path']:.2f}%" if jpeg_input else "")
                  + f" | {rec['bpp_total']:.3f} bpp", flush=True)
            with open(path, "w") as f:     # after every model, so a later failure loses nothing
                json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
