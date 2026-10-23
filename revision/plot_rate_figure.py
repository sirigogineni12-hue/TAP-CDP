"""Figure 4 of the revised manuscript: Imagenette accuracy against the entropy-coded bit-rate (seed 42).

Reads results/report/rate_accuracy.csv (written by make_report.py) and writes the figure.
  python revision/plot_rate_figure.py --csv results/report/rate_accuracy.csv --out results/report/rate_imagenette.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CURVES = {  # run-id part -> (label, colour, marker)
    "pixel": ("Pixel ResNet, standard JPEG 4:2:0", "k", "o"),
    "none_ycbcr_stemnopool_dctnorm": ("CI-DCT, all 192 channels", "tab:red", "s"),
    "none_ycbcr_dctnorm": ("CI-DCT, all channels, max-pool", "tab:pink", "s"),
    "lowfreq_k24_ycbcr_stemnopool_dctnorm": ("Low-frequency, k=24", "tab:blue", "^"),
    "lowfreq_k48_ycbcr_stemnopool_dctnorm": ("Low-frequency, k=48", "tab:cyan", "^"),
    "topk_k24_ycbcr_stemnopool_dctnorm": ("Budgeted TACS, k=24", "tab:green", "P"),
    "topk_k48_ycbcr_stemnopool_dctnorm": ("Budgeted TACS, k=48", "tab:olive", "P"),
    "random_k24_ycbcr_stemnopool_dctnorm": ("Random, k=24", "tab:brown", "x"),
    "random_k48_ycbcr_stemnopool_dctnorm": ("Random, k=48", "tab:gray", "x"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/report/rate_accuracy.csv")
    p.add_argument("--out", default="results/report/rate_imagenette.pdf")
    a = p.parse_args()
    r = pd.read_csv(a.csv)
    r = r[r["dataset"] == "imagenette"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, bb in zip(axes, ["resnet18", "resnet50"]):
        g = r[r["backbone"] == bb]
        for key, (label, colour, marker) in CURVES.items():
            x = g[g["model"] == f"imagenette_{bb}_{key}_e20_seed42"].sort_values("bpp_scan_only")
            if not x.empty:
                ax.plot(x["bpp_scan_only"], x["accuracy"], marker=marker, color=colour, label=label,
                        markersize=4, linewidth=1.2)
        ax.set_title(f"Imagenette, {'ResNet-18' if bb == 'resnet18' else 'ResNet-50'} (seed 42)", fontsize=10)
        ax.set_xlabel("Bits per pixel, entropy-coded data only (224$\\times$224)")
        ax.grid(alpha=0.3)
        ax.set_ylim(10, 100)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
