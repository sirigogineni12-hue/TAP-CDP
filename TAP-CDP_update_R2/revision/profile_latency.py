"""Measured per-stage latency (and optionally GPU board energy) on real JPEG files.

Protocol (reported in latency.json so it can be quoted in the paper):
  * input: N real JPEGs, center-cropped to 224x224 and re-encoded at --quality,
  * batch sizes 1 and 64, --warmup untimed iterations, then --iters timed iterations,
  * torch.cuda.synchronize() before and after every timed iteration, wall-clock timer,
  * statistics: median (reported), mean, p10, p90 in milliseconds per batch and per image,
  * CPU decode = Pillow (libjpeg-turbo) on one CPU thread; GPU decode = nvJPEG via
    torchvision.io.decode_jpeg(device="cuda"); host-to-device copy timed separately.
  * DCT designs (--designs, colour/stem[/norm]) are timed with all channels; their YCbCr conversion,
    3-component DCT and per-channel normalization are separate stages. The weights are random (timing only),
    so any folder of real photos works (e.g. COCO val2017 when ImageNet is not available).
Note: the TAP-CDP code computes the DCT from decoded pixels, so its pipeline still
performs the full JPEG decode (including the IDCT). Both pipelines are timed as implemented.
Output: latency.json
"""
import argparse
import io
import os
import statistics
import threading
import time

import numpy as np
import torch
from PIL import Image
from torchvision.io import decode_jpeg
from torchvision.transforms import functional as TF

from common import (COMPONENTS, MEAN, STD, DctClassifier, PixelClassifier, block_dct, env_info, rgb_to_y,
                    rgb_to_ycbcr, write_json)
from measure_flops import dct_backbone


def stats(samples, batch):
    s = sorted(samples)
    pct = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    med = statistics.median(s)
    return {"median_ms": med, "mean_ms": statistics.fmean(s), "p10_ms": pct(0.10), "p90_ms": pct(0.90),
            "median_ms_per_image": med / batch, "n_timed": len(s), "batch": batch}


def timed(fn, warmup, iters, cuda):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1e3)
    return out


def load_jpegs(folder, n, quality):
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(folder) for f in fs
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))[:n]
    if not files:
        raise SystemExit(f"no images found under {folder}")
    blobs = []
    for f in files:
        img = Image.open(f).convert("RGB")
        img = TF.center_crop(TF.resize(img, 256), 224)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        blobs.append(buf.getvalue())
    return blobs


class PowerSampler(threading.Thread):
    def __init__(self, handle, period_s=0.01):
        super().__init__(daemon=True)
        self.h, self.period, self.samples, self.running = handle, period_s, [], True

    def run(self):
        import pynvml
        while self.running:
            self.samples.append(pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0)  # watts
            time.sleep(self.period)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="folder with real JPEG images (ImageNet val or COCO val2017)")
    p.add_argument("--n_images", type=int, default=512)
    p.add_argument("--quality", type=int, default=90)
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--ks", type=int, nargs="+", default=[64, 32, 16, 8])
    p.add_argument("--designs", nargs="*", default=["ycbcr/pool/norm", "ycbcr/nopool/norm", "ycbcr/nopool_s1/norm"],
                   help="DCT designs colour/stem[/norm] timed with all channels (y/pool is dct_k64)")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 64])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--energy_seconds", type=float, default=10.0, help="0 disables NVML energy measurement")
    p.add_argument("--allow_cpu", action="store_true", help="smoke test only; CPU timings are not GPU latency")
    p.add_argument("--out", default="latency.json")
    a = p.parse_args()

    cuda = torch.cuda.is_available()
    assert cuda or a.allow_cpu, "latency profiling needs a CUDA GPU"
    dev = torch.device("cuda" if cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    blobs = load_jpegs(a.images, a.n_images, a.quality)
    mean = torch.tensor(MEAN["imagenet"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD["imagenet"], device=dev).view(1, 3, 1, 1)

    models = {"pixel": PixelClassifier(a.backbone, 1000, pretrained=False).net}
    in_ch = {"pixel": 3}
    for k in a.ks:
        models[f"dct_k{k}"] = dct_backbone(a.backbone, 1000, k)
        in_ch[f"dct_k{k}"] = k
    for d in a.designs:
        color, stem = d.split("/")[:2]
        models[f"dct_{d.replace('/', '-')}"] = DctClassifier(a.backbone, 1000, "none", pretrained=False,
                                                              color=color, stem=stem).backbone
        in_ch[f"dct_{d.replace('/', '-')}"] = 64 * COMPONENTS[color]
    for m in models.values():
        m.to(dev, dtype=dtype).eval()

    res = {"protocol": {k: v for k, v in vars(a).items() if k != "images"}, "env": env_info(), "stages": {}}
    with torch.inference_mode():
        for b in a.batches:
            batch_blobs = (blobs * ((b // len(blobs)) + 1))[:b]
            st = {}
            cpu = lambda: [np.asarray(Image.open(io.BytesIO(x)).convert("RGB")) for x in batch_blobs]
            st["cpu_decode_pillow"] = stats(timed(cpu, min(a.warmup, 10), min(a.iters, 50), False), b)
            arrays = cpu()
            host = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
            host = host.pin_memory() if cuda else host
            st["h2d_copy"] = stats(timed(lambda: host.to(dev, non_blocking=True), a.warmup, a.iters, cuda), b)
            datas = [torch.frombuffer(bytearray(x), dtype=torch.uint8) for x in batch_blobs]
            try:
                st["gpu_decode_nvjpeg"] = stats(timed(lambda: decode_jpeg(datas, device=dev), a.warmup, a.iters, cuda), b)
            except Exception as e:  # nvJPEG unavailable in this torchvision build
                st["gpu_decode_nvjpeg"] = {"error": str(e)}
            u8 = host.to(dev)
            norm = lambda: ((u8.float() / 255.0) - mean) / std
            st["preprocess_rgb_normalize"] = stats(timed(norm, a.warmup, a.iters, cuda), b)
            rgb = norm()
            st["preprocess_rgb_to_y"] = stats(timed(lambda: rgb_to_y(rgb), a.warmup, a.iters, cuda), b)
            y = rgb_to_y(rgb)
            st["forward_block_dct"] = stats(timed(lambda: block_dct(y), a.warmup, a.iters, cuda), b)
            coeffs = block_dct(y)
            st["backbone_pixel"] = stats(timed(lambda: models["pixel"](rgb.to(dtype)), a.warmup, a.iters, cuda), b)
            for k in a.ks:
                z = coeffs[:, :k].contiguous().to(dtype)
                st[f"backbone_dct_k{k}"] = stats(timed(lambda: models[f"dct_k{k}"](z), a.warmup, a.iters, cuda), b)
            if a.designs:
                st["preprocess_rgb_to_ycbcr"] = stats(timed(lambda: rgb_to_ycbcr(rgb), a.warmup, a.iters, cuda), b)
                ycc = rgb_to_ycbcr(rgb)
                st["forward_block_dct_ycbcr"] = stats(timed(lambda: block_dct(ycc), a.warmup, a.iters, cuda), b)
                coeffs3 = block_dct(ycc)
                for c, src in [(64, coeffs), (192, coeffs3)]:
                    mu, sd = torch.zeros(1, c, 1, 1, device=dev), torch.ones(1, c, 1, 1, device=dev)
                    st[f"dct_normalize_{c}ch"] = stats(timed(lambda: (src - mu) / sd, a.warmup, a.iters, cuda), b)
                for d in a.designs:
                    name = f"dct_{d.replace('/', '-')}"
                    z = (coeffs3 if in_ch[name] == 192 else coeffs).contiguous().to(dtype)
                    st[f"backbone_{name}"] = stats(timed(lambda: models[name](z), a.warmup, a.iters, cuda), b)
            res["stages"][f"batch{b}"] = st
            print(f"batch {b}: " + ", ".join(f"{n}={v['median_ms_per_image']:.3f}ms/img"
                                             for n, v in st.items() if "median_ms_per_image" in v))

        if a.energy_seconds > 0 and cuda:
            try:
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                idle = PowerSampler(h); idle.start(); time.sleep(3); idle.running = False; idle.join()
                b = max(a.batches)
                rgb = torch.randn(b, 3, 224, 224, device=dev, dtype=dtype)
                energy = {"idle_power_w": statistics.fmean(idle.samples), "batch": b}
                for name, model in models.items():
                    x = rgb if name == "pixel" else torch.randn(b, in_ch[name], 28, 28, device=dev, dtype=dtype)
                    for _ in range(20):
                        model(x)
                    torch.cuda.synchronize()
                    s = PowerSampler(h); s.start(); t0 = time.perf_counter(); n = 0
                    while time.perf_counter() - t0 < a.energy_seconds:
                        model(x); n += b
                    torch.cuda.synchronize(); el = time.perf_counter() - t0; s.running = False; s.join()
                    pw = statistics.fmean(s.samples)
                    energy[name] = {"mean_board_power_w": pw, "images": n, "seconds": el,
                                    "energy_mj_per_image_total": pw * el / n * 1e3,
                                    "energy_mj_per_image_above_idle": (pw - energy["idle_power_w"]) * el / n * 1e3}
                res["energy_backbone_only"] = energy
                print("energy (mJ/image, above idle):",
                      {k: round(v["energy_mj_per_image_above_idle"], 4) for k, v in energy.items() if isinstance(v, dict)})
            except Exception as e:
                res["energy_backbone_only"] = {"error": f"NVML measurement unavailable: {e}"}
    write_json(a.out, res)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
