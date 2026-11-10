"""Choose the CIFAR-10 DCT input/stem design from the pilot, using held-out validation accuracy only.

Rule (fixed before the pilot is run):
  1. Candidates are the CI-DCT runs (all bands, no mask) of every design (colour x stem x normalization),
     each trained with the same seeds; the score is the mean validation accuracy over those seeds.
  2. Take the design with the highest score. If a design with fewer MACs scores within --tolerance points
     of it, take the cheapest such design instead.
  3. The choice is made before any mask method (TACS, Xu et al. gate, fixed masks) runs on that design.
Test accuracy is listed for completeness and is not used. Every pilot result is reported in the paper.
Writes cifar_arch.env (sourced by stage 1), cifar_arch_reason.txt and report/cifar_arch_pilot.csv.
"""
import argparse
import glob
import json
import os

import pandas as pd
import torch
from fvcore.nn import FlopCountAnalysis

from common import DctClassifier, PixelClassifier


def macs(model):
    model.eval()
    with torch.no_grad():
        f = FlopCountAnalysis(model, torch.zeros(1, 3, 32, 32))
        f.unsupported_ops_warnings(False)
        f.uncalled_modules_warnings(False)
        return int(f.total())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--seeds", default="42 123")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--tolerance", type=float, default=0.5)
    p.add_argument("--out", default="cifar_arch.env")
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split()]

    rows = []
    for f in glob.glob(os.path.join(a.runs_dir, "cifar10_resnet18_*", "result.json")):
        r = json.load(open(f))
        c = r["config"]
        if c["mode"] not in ("none", "pixel") or c["epochs"] != a.epochs or c.get("test_jpeg_quality") \
                or c["seed"] not in seeds:
            continue
        rows.append({"mode": c["mode"], "color": c.get("color", "y"), "stem": c.get("stem", "pool"),
                     "dct_norm": bool(c.get("dct_norm", False)), "seed": c["seed"],
                     "val_acc": r["val_acc"], "test_acc": r["test_acc"], "train_s": r["train_time_s"]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no pilot runs found")
    g = df.groupby(["mode", "color", "stem", "dct_norm"]).agg(
        n_seeds=("seed", "nunique"), val_mean=("val_acc", "mean"), val_std=("val_acc", "std"),
        test_mean=("test_acc", "mean"), test_std=("test_acc", "std"), train_min=("train_s", "mean")).reset_index()
    g["train_min"] = g["train_min"] / 60
    g["gmacs"] = [macs(PixelClassifier("resnet18", 10, pretrained=False, stem=r.stem)) if r.mode == "pixel" else
                  macs(DctClassifier("resnet18", 10, "none", pretrained=False, color=r.color, stem=r.stem))
                  for r in g.itertuples()]
    g["gmacs"] = g["gmacs"] / 1e9
    os.makedirs("report", exist_ok=True)
    g.sort_values(["mode", "val_mean"], ascending=[True, False]).to_csv("report/cifar_arch_pilot.csv", index=False)

    cand = g[(g["mode"] == "none") & (g["n_seeds"] == len(seeds))].sort_values("val_mean", ascending=False)
    if cand.empty:
        raise SystemExit("no CI-DCT design has all pilot seeds finished")
    best = cand.iloc[0]
    close = cand[(cand["val_mean"] >= best["val_mean"] - a.tolerance) & (cand["gmacs"] < best["gmacs"])]
    choice = close.sort_values("gmacs").iloc[0] if not close.empty else best

    lines = [f"Rule: highest mean held-out validation accuracy of CI-DCT over seeds {a.seeds}; a cheaper design "
             f"within {a.tolerance} points is preferred. Test accuracy is not used.", ""]
    for r in g.sort_values(["mode", "val_mean"], ascending=[True, False]).itertuples():
        name = f"pixel stem={r.stem}" if r.mode == "pixel" else f"CI-DCT color={r.color} stem={r.stem} dct_norm={r.dct_norm}"
        lines.append(f"  {name:55s} val {r.val_mean:6.2f} +- {r.val_std:4.2f} | test {r.test_mean:6.2f} | "
                     f"{r.gmacs:.4f} GMACs | {r.n_seeds} seeds")
    args = f"--color {choice['color']} --stem {choice['stem']}" + (" --dct_norm" if choice["dct_norm"] else "")
    lines += ["", f"Selected design: {args}  (val {choice['val_mean']:.2f}%, {choice['gmacs']:.4f} GMACs)"]
    label = f"{choice['color']}/{choice['stem']}" + ("/norm" if choice["dct_norm"] else "")
    with open(a.out, "w") as f:
        f.write(f'CIFAR_ARCH_ARGS="{args}"\nCIFAR_ARCH_LABEL="{label}"\n')
    with open(a.out.replace(".env", "_reason.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
