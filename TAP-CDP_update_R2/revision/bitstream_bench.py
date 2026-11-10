"""Decoder-side cost of the pixel path and of the bitstream (coefficient) path, measured on real JPEG files.

For N test photos (resized to 256, centre-cropped to 224x224) and each quality Q, files are encoded as
  * pixel_std420   : standard libjpeg tables, 4:2:0 (input of a pixel-domain network),
  * dct_std444     : standard tables, 4:4:4 (input of a DCT-domain network using all channels),
  * dct_lowfreq{k} : tables with the bands outside the k-channel low-frequency mask set to step 255, 4:4:4.
Timed on one CPU thread (median of --repeats passes over all files, ms per image):
  pillow_rgb     Pillow decode to RGB
  libjpeg_rgb    libjpeg full decode (entropy decoding, dequantization, inverse DCT, upsampling, colour)
  pixel_input    libjpeg_rgb + float conversion and normalization (CPU input of the pixel network)
  dct_from_pix   libjpeg decode to YCbCr + forward block DCT in numpy (how the DCT input is computed today)
  coef_read      entropy decoding only (jpeg_read_coefficients)
  coef_input     coef_read + dequantization + channel layout (CPU input of a DCT network, from the bitstream)
Also reported: bytes per file, nonzero coefficients per image, and for 4:4:4 files the agreement between
bitstream coefficients and the DCT of the decoded image. Output: bitstream_bench.json
"""
import argparse
import io
import json
import os
import statistics
import time

import numpy as np
from PIL import Image

import bitstream as bs
from common import IMAGE_SIZE, MEAN, STD, env_info, fixed_mask
from rate_accuracy import encoder_settings


def load_images(folder, n):
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(folder) for f in fs
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    step = max(1, len(files) // n)
    out = []
    for f in files[::step][:n]:
        img = Image.open(f).convert("RGB")
        w, h = img.size
        s = 256 / min(w, h)
        img = img.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
        w, h = img.size
        l, t = (w - 224) // 2, (h - 224) // 2
        out.append(img.crop((l, t, l + 224, t + 224)))
    return out


def encode(img, qtables, subsampling):
    b = io.BytesIO()
    img.save(b, "JPEG", qtables=qtables, subsampling=subsampling)
    return b.getvalue()


def timed(fn, blobs, repeats):
    per_image = []
    for _ in range(repeats):
        t = time.perf_counter()
        for d in blobs:
            fn(d)
        per_image.append((time.perf_counter() - t) * 1e3 / len(blobs))
    return {"median_ms": statistics.median(per_image), "min_ms": min(per_image), "max_ms": max(per_image)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--qualities", type=int, nargs="+", default=[50, 90])
    p.add_argument("--ks", type=int, nargs="+", default=[24, 48])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", default="bitstream_bench.json")
    a = p.parse_args()

    imgs = load_images(a.images, a.n)
    mean = np.array(MEAN["imagenet"], np.float32) * 255
    std = np.array(STD["imagenet"], np.float32) * 255

    def pixel_input(d):
        x = bs.decode(d).astype(np.float32)
        return ((x - mean) / std).transpose(2, 0, 1)

    def dct_from_pix(d):
        return bs.block_dct_numpy(bs.decode(d, ycbcr=True).astype(np.float32).transpose(2, 0, 1) - 128.0)

    methods = {"pillow_rgb": lambda d: np.asarray(Image.open(io.BytesIO(d)).convert("RGB")),
               "libjpeg_rgb": bs.decode, "pixel_input": pixel_input, "dct_from_pix": dct_from_pix,
               "coef_read": bs.read_coefficients, "coef_input": bs.coefficient_tensor}
    variants = [("pixel_std420", None, 2), ("dct_std444", None, 0)]
    variants += [(f"dct_lowfreq{k}", fixed_mask("lowfreq", k, 0, components=3).tolist(), 0) for k in a.ks]

    out = {"method": __doc__, "env": env_info(), "n_images": len(imgs), "image_size": IMAGE_SIZE["imagenet"],
           "threads": os.environ.get("OMP_NUM_THREADS"), "results": []}
    for q in a.qualities:
        for name, mask, sub in variants:
            qt, _ = encoder_settings(q, mask)
            blobs = [encode(im, qt, sub) for im in imgs]
            nz = [sum(int(np.count_nonzero(c)) for c in bs.read_coefficients(d)[0]) for d in blobs]
            rec = {"variant": name, "quality": q, "subsampling": {2: "4:2:0", 0: "4:4:4"}[sub],
                   "bytes_mean": float(np.mean([len(d) for d in blobs])),
                   "nonzero_coefficients_mean": float(np.mean(nz)), "timing": {}}
            if sub == 0:
                diffs = []
                for d in blobs[:64]:
                    c, r = bs.coefficient_tensor(d), dct_from_pix(d)
                    diffs.append(float(np.abs(c - r).mean() / max(np.abs(r).mean(), 1e-9)))
                rec["coef_vs_dct_of_decoded_rel_diff"] = float(np.mean(diffs))
            for m, fn in methods.items():
                if m in ("coef_input",) and sub != 0:
                    continue
                rec["timing"][m] = timed(fn, blobs, a.repeats)
            out["results"].append(rec)
            t = rec["timing"]
            print(f"Q={q:3d} {name:15s} {rec['bytes_mean']:8.0f} B  nz={rec['nonzero_coefficients_mean']:7.0f}  "
                  + "  ".join(f"{m}={v['median_ms']:.3f}" for m, v in t.items()), flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
