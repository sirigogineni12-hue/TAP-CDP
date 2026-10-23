"""Choose lambda and the TACS initialization for ImageNet from CIFAR-10 held-out validation results.

Rule (fixed in advance, uses validation accuracy only):
  1. For each TACS initialization, take the largest lambda whose mean validation accuracy is
     within --tolerance points of CI-DCT and whose mean number of active bands is at least 1.
  2. Prefer the repository initialization (0) if it yields such a lambda; otherwise use 2.
  3. Run ImageNet with the selected lambda (and, with --num_lambdas 2, the next smaller one).
If no configuration qualifies, fall back to the defaults in --default_lambdas / init 2.
Writes imagenet_settings.env (sourced by the ImageNet stage) and imagenet_settings_reason.txt.
"""
import argparse

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grouped", default="report/grouped_results.csv")
    p.add_argument("--tolerance", type=float, default=1.0)
    p.add_argument("--default_lambdas", default="0.1")
    p.add_argument("--num_lambdas", type=int, default=1, help="1 = selected lambda only; 2 = also the next smaller one")
    p.add_argument("--arch", default="y/pool", help="CIFAR-10 design to read (CIFAR_ARCH_LABEL from cifar_arch.env)")
    p.add_argument("--out", default="imagenet_settings.env")
    a = p.parse_args()

    g = pd.read_csv(a.grouped)
    g = g[(g["dataset"] == "cifar10") & g["test_jpeg_q"].isna()]
    if "arch" in g.columns:
        g = g[g["arch"] == a.arch]
    reason = [f"Rule: largest lambda with mean val accuracy within {a.tolerance} points of CI-DCT and >= 1 active band."]
    ref = g[g["mode"] == "none"]
    choice = None
    if not ref.empty:
        ref_val = ref["val_mean"].iloc[0]
        reason.append(f"CI-DCT mean val accuracy: {ref_val:.2f}%")
        learned = g[g["mode"] == "learned"]
        for init in [0.0, 2.0]:
            cand = learned[learned["tacs_init"] == init].sort_values("lam")
            ok = cand[(cand["val_mean"] >= ref_val - a.tolerance) & (cand["bands_mean"] >= 1)]
            for _, r in cand.iterrows():
                reason.append(f"  init {init:g}, lambda {r['lam']:g}: val {r['val_mean']:.2f}%, bands {r['bands_mean']:.1f}")
            if choice is None and not ok.empty:
                lam = ok["lam"].max()
                grid = sorted(cand["lam"].unique())
                smaller = [x for x in grid if x < lam]
                lams = [lam] + ([smaller[-1]] if (smaller and a.num_lambdas >= 2) else [])
                choice = (init, lams)
    if choice is None:
        choice = (2.0, [float(x) for x in a.default_lambdas.split()])
        reason.append("No CIFAR-10 configuration met the rule; using defaults with TACS init 2.")
    init, lams = choice
    reason.append(f"Selected for ImageNet: TACS_INIT={init:g}, LAMBDAS={' '.join(f'{x:g}' for x in lams)}")
    with open(a.out, "w") as f:
        f.write(f'TACS_INIT={init:g}\nLAMBDAS="{" ".join(f"{x:g}" for x in lams)}"\n')
    with open(a.out.replace(".env", "_reason.txt"), "w") as f:
        f.write("\n".join(reason) + "\n")
    print("\n".join(reason))


if __name__ == "__main__":
    main()
