"""Figure 5 of the resubmitted manuscript: ImageNet-1k top-1 accuracy against the bit-rate of the JPEG files.
DCT-domain models are evaluated through the bitstream path, the pixel model through the pixel path.
  python revision/plot_rate_imagenet.py --json results/bitstream_eval_imagenet.json --out results/report/rate_imagenet.pdf
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CURVES = [  # run id, label, colour, marker, accuracy field
    ("imagenet_resnet50_pixel_e0_seed42", "Pixel ResNet-50, standard JPEG 4:2:0", "k", "o", "acc_pixel_path"),
    ("imagenet_resnet50_none_ycbcr_jpeg_stemnopool_dctnorm_e16_seed42", "CI-DCT, all 192 channels", "tab:red", "s", "acc_bitstream_path"),
    ("imagenet_resnet50_none_ycbcr_jpeg_dctnorm_e16_seed42", "CI-DCT, all channels, max-pool", "tab:pink", "s", "acc_bitstream_path"),
    ("imagenet_resnet50_lowfreq_k24_ycbcr_jpeg_stemnopool_dctnorm_e16_seed42", "Low-frequency, k=24", "tab:blue", "^", "acc_bitstream_path"),
    ("imagenet_resnet50_lowfreq_k48_ycbcr_jpeg_stemnopool_dctnorm_e16_seed42", "Low-frequency, k=48", "tab:cyan", "^", "acc_bitstream_path"),
    ("imagenet_resnet50_topk_k24_ycbcr_jpeg_stemnopool_dctnorm_e16_seed42", "Budgeted TACS, k=24", "tab:green", "P", "acc_bitstream_path"),
    ("imagenet_resnet50_random_k24_ycbcr_jpeg_stemnopool_dctnorm_e16_seed42", "Random, k=24", "tab:brown", "x", "acc_bitstream_path"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="results/bitstream_eval_imagenet.json")
    p.add_argument("--out", default="results/report/rate_imagenet.pdf")
    a = p.parse_args()
    r = json.load(open(a.json))["results"]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for rid, label, colour, marker, key in CURVES:
        pts = sorted((x["bpp_total"], x[key]) for x in r if x["run_id"] == rid)
        ax.plot([q[0] for q in pts], [q[1] for q in pts], marker=marker, color=colour, label=label,
                markersize=4, linewidth=1.2)
    ax.set_xlabel("Bits per pixel, whole JPEG file (224$\\times$224)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("ImageNet-1k validation, ResNet-50 (DCT models: coefficients read from the files)", fontsize=8.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0, 80)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
