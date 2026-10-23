"""Build every table and figure for the revision from saved result files only.

Inputs: runs/*/result.json (+ log.csv), flops.json, latency.json, rate_accuracy*.json.
Lambda is selected on the held-out validation split with a rule fixed in advance:
the largest lambda whose mean validation accuracy is within --tolerance points of CI-DCT.
Test accuracy is reported but never used for selection.
"""
import argparse
import glob
import itertools
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_runs(runs_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(runs_dir, "*", "result.json"))):
        r = json.load(open(f))
        c = r["config"]
        color, stem, norm = c.get("color", "y"), c.get("stem", "pool"), bool(c.get("dct_norm", False))
        arch = f"stem {stem}" if c["mode"] == "pixel" else f"{color}/{stem}{'/norm' if norm else ''}"
        rows.append({"run_id": r["run_id"], "dataset": c["dataset"], "backbone": c["backbone"], "mode": c["mode"],
                     "arch": arch, "default_arch": arch in ("stem pool", "y/pool"),
                     "lam": c["lam"] if c["mode"] in ("learned", "xugate") else np.nan, "k": c.get("k"),
                     "tacs_init": c.get("tacs_init", 0.0) if c["mode"] == "learned" else np.nan,
                     "epochs": c["epochs"], "test_q": c.get("test_jpeg_quality"), "seed": c["seed"],
                     "val_acc": r["val_acc"], "test_acc": r["test_acc"], "active_bands": r.get("active_bands"),
                     "train_hours": r["train_time_s"] / 3600, "mask_binary": r.get("mask_binary"),
                     "mask_probs": r.get("mask_probs"), "gpus": ", ".join(r["env"]["gpus"]),
                     "dir": os.path.dirname(f)})
    return pd.DataFrame(rows)


def config_key(r):
    arch = "" if r["default_arch"] else f" [{r['arch']}]"
    if r["mode"] == "learned":
        init = "" if r["tacs_init"] == 0 else f", init={r['tacs_init']:g}"
        return f"TAP-CDP (lambda={r['lam']:g}{init}){arch}"
    if r["mode"] == "xugate":
        return f"Xu et al. gate (lambda={r['lam']:g}){arch}"
    if r["mode"] == "topk":
        return f"Budgeted TACS (k={int(r['k'])}){arch}"
    if r["mode"] in ("random", "lowfreq"):
        return f"{'Random' if r['mode'] == 'random' else 'Low-freq zigzag'} mask (k={int(r['k'])}){arch}"
    return {"none": "CI-DCT (all bands)", "pixel": "Pixel-domain ResNet"}[r["mode"]] + arch


def msd(x, fmt="{:.2f}"):
    x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not x:
        return ""
    if len(x) == 1:
        return fmt.format(x[0])
    return f"{fmt.format(statistics.fmean(x))} $\\pm$ {fmt.format(statistics.stdev(x))}"


def macs_for(flops, dataset, backbone, mode, bands, arch="y/pool"):
    rec = next((r for r in flops["results"] if r["dataset"] == dataset and r["backbone"] == backbone), None)
    if rec is None:
        return None, None
    if mode == "pixel":
        v = rec["pixel_macs"] if arch == "stem pool" else rec.get("pixel_cifar_stem_macs")
        return v, v
    if arch == "y/pool":
        as_impl, pruned = rec["dct_backbone_macs_64ch"], rec["dct_backbone_macs_pruned_k"]
    else:   # coefficient normalization adds no MACs
        d = rec.get("designs", {}).get(arch.replace("/norm", ""))
        if d is None:
            return None, None
        as_impl, pruned = d["macs_all_channels"], d["macs_pruned_k"]
    ks = sorted(int(k) for k in pruned)
    if bands is None:
        return as_impl, None
    # Only the stem's input channels change with k, so cost is linear in k between measured points.
    b = int(round(bands))
    lo = max(k for k in ks if k <= b) if any(k <= b for k in ks) else ks[0]
    hi = min(k for k in ks if k >= b) if any(k >= b for k in ks) else ks[-1]
    if lo == hi:
        return as_impl, pruned[str(lo)]
    w = (b - lo) / (hi - lo)
    return as_impl, pruned[str(lo)] + w * (pruned[str(hi)] - pruned[str(lo)])


def grouped_table(df, flops):
    out = []
    for (ds, bb, ep, tq, key), g in df.groupby(["dataset", "backbone", "epochs", df["test_q"].fillna(-1), "key"], sort=True):
        r0 = g.iloc[0]
        bands = g["active_bands"].dropna().tolist()
        as_impl, pruned = macs_for(flops, ds, bb, r0["mode"], statistics.fmean(bands) if bands else None,
                                   r0["arch"]) if flops else (None, None)
        out.append({"dataset": ds, "backbone": bb, "epochs": ep, "test_jpeg_q": None if tq == -1 else int(tq),
                    "config": key, "mode": r0["mode"], "arch": r0["arch"], "lam": r0["lam"], "tacs_init": r0["tacs_init"], "k": r0["k"], "n_seeds": len(g),
                    "seeds": ",".join(str(s) for s in sorted(g["seed"])),
                    "val_acc": msd(g["val_acc"]), "test_acc": msd(g["test_acc"]),
                    "val_mean": g["val_acc"].mean(), "test_mean": g["test_acc"].mean(),
                    "active_bands": msd(bands, "{:.1f}") if bands else "",
                    "bands_mean": statistics.fmean(bands) if bands else None,
                    "gmacs_as_implemented": None if as_impl is None else as_impl / 1e9,
                    "gmacs_if_stem_pruned": None if pruned is None else pruned / 1e9,
                    "train_hours_per_run": g["train_hours"].mean()})
    return pd.DataFrame(out)


def select_lambda(gt, tolerance):
    lines = [f"Rule (fixed in advance): largest lambda whose mean held-out validation accuracy is within "
             f"{tolerance} points of CI-DCT. Test accuracy is not used."]
    for (ds, bb, ep, arch), g in gt[gt["test_jpeg_q"].isna()].groupby(["dataset", "backbone", "epochs", "arch"]):
        ref = g[g["mode"] == "none"]
        learned = g[g["mode"].isin(["learned", "xugate"])]
        if ref.empty or learned.empty:
            continue
        lines.append(f"{ds}/{bb}/{ep} epochs, design {arch}: CI-DCT val {ref['val_mean'].iloc[0]:.2f}%")
        for (mode, ti), cand in learned.groupby(["mode", learned["tacs_init"].fillna(-1)]):
            cand = cand.sort_values("lam")
            ok = cand[cand["val_mean"] >= ref["val_mean"].iloc[0] - tolerance]
            lines.append("  Xu et al. gate:" if mode == "xugate" else
                         f"  TACS init {ti:g}{' (repository default)' if ti == 0 else ''}:")
            for _, r in cand.iterrows():
                lines.append(f"     lambda={r['lam']:g}: val {r['val_acc']}, bands {r['active_bands']}")
            lines.append(f"     -> selected lambda = {ok['lam'].max():g}" if not ok.empty else "     -> no lambda within tolerance")
    return "\n".join(lines)


def mask_stability(df, out_dir):
    lines, rows = [], []
    sel = df[df["mode"].isin(["learned", "xugate", "topk"])].copy()
    sel["param"] = np.where(sel["mode"] == "topk", sel["k"], sel["lam"])   # topk has a budget k instead of lambda
    for (ds, bb, ep, arch, mode, lam, ti), g in sel.groupby(["dataset", "backbone", "epochs", "arch", "mode", "param",
                                                             sel["tacs_init"].fillna(-1)]):
        masks = [np.array(m) for m in g["mask_binary"]]
        jac = [float((a * b).sum() / max(((a + b) > 0).sum(), 1)) for a, b in itertools.combinations(masks, 2)]
        n_ch = len(masks[0])
        agree = np.mean([np.all(np.stack(masks)[:, i] == masks[0][i]) for i in range(n_ch)]) if len(masks) > 1 else 1.0
        rows.append({"dataset": ds, "backbone": bb, "epochs": ep, "arch": arch, "mode": mode,
                     "lam_or_k": lam,
                     "tacs_init": None if ti == -1 else ti, "n_seeds": len(g),
                     "active_bands": msd(g["active_bands"].tolist(), "{:.1f}"),
                     "pairwise_jaccard_mean": float(np.mean(jac)) if jac else None,
                     "pairwise_jaccard_min": float(np.min(jac)) if jac else None,
                     "fraction_of_bands_identical_across_seeds": float(agree)})
        comps = n_ch // 64   # Y, then Cb, Cr: shown side by side
        fig, axes = plt.subplots(2, len(g), figsize=(2.6 * comps * len(g), 5.0), squeeze=False)
        for j, (_, r) in enumerate(g.sort_values("seed").iterrows()):
            for i, (vals, title) in enumerate([(r["mask_probs"], "probability"), (r["mask_binary"], "binary mask")]):
                ax = axes[i, j]
                img = np.concatenate(list(np.array(vals).reshape(comps, 8, 8)), axis=1)
                im = ax.imshow(img, vmin=0, vmax=1, cmap="viridis")
                ax.set_title(f"seed {r['seed']}: {title}" + (" (Y | Cb | Cr)" if comps == 3 else ""), fontsize=8)
                ax.set_xticks(range(8 * comps)); ax.set_xticklabels([i % 8 for i in range(8 * comps)])
                ax.set_yticks(range(8)); ax.tick_params(labelsize=6)
                ax.set_xlabel("horizontal frequency", fontsize=7); ax.set_ylabel("vertical frequency", fontsize=7)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        tag = {"xugate": f"xugate_lam{lam:g}", "topk": f"topk_k{int(lam)}"}.get(mode, f"lam{lam:g}_init{ti:g}")
        arch_tag = "" if g["default_arch"].iloc[0] else "_" + arch.replace("/", "-")
        fig.savefig(os.path.join(out_dir, f"masks_{ds}_{bb}{arch_tag}_{tag}.pdf"), bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "mask_stability.csv"), index=False)


def plot_acc_vs_bands(gt, df, out_dir):
    dct = gt[gt["test_jpeg_q"].isna() & (gt["mode"] != "pixel")]
    for (ds, bb, ep, arch), g in dct.groupby(["dataset", "backbone", "epochs", "arch"]):
        fig, ax = plt.subplots(figsize=(5, 3.6))
        series = [("random", None, "Random mask", "s"), ("lowfreq", None, "Low-frequency zigzag mask", "^")]
        inits = sorted(df.loc[(df["mode"] == "learned") & (df["arch"] == arch), "tacs_init"].dropna().unique())
        series += [("learned", ti, f"TAP-CDP (learned TACS, init {ti:g})", "o") for ti in inits]
        series += [("topk", None, "Budgeted TACS (exactly k)", "P"), ("xugate", None, "Xu et al. gate (mean per image)", "v"),
                   ("none", None, "CI-DCT (all bands)", "D")]
        for mode, ti, label, mk in series:
            sub = df[(df.dataset == ds) & (df.backbone == bb) & (df.epochs == ep) & (df["mode"] == mode) &
                     (df["arch"] == arch) & df.test_q.isna()]
            if ti is not None:
                sub = sub[sub["tacs_init"] == ti]
            if sub.empty:
                continue
            key = "lam" if mode in ("learned", "xugate") else "k"
            pts = sub.groupby(key if mode != "none" else "mode").agg(b=("active_bands", "mean"), m=("test_acc", "mean"),
                                                                     s=("test_acc", "std")).reset_index().sort_values("b")
            ax.errorbar(pts["b"], pts["m"], yerr=pts["s"].fillna(0), marker=mk, capsize=3, label=label,
                        linestyle="-" if mode in ("random", "lowfreq") else "none")
        pix = df[(df.dataset == ds) & (df.backbone == bb) & (df["mode"] == "pixel") & (df.epochs == ep) & df.test_q.isna()]
        for (parch, pg), ls in zip(pix.groupby("arch"), [":", "--"]):
            ax.axhline(pg["test_acc"].mean(), color="gray", linestyle=ls, label=f"Pixel-domain ResNet ({parch})")
        n_ch = 192 if arch.startswith("ycbcr") else 64
        ax.set_xlabel(f"Active DCT channels (of {n_ch})"); ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"{ds}, {bb}, {ep} epochs, design {arch} (mean $\\pm$ std over seeds)", fontsize=8)
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
        arch_tag = "" if arch == "y/pool" else "_" + arch.replace("/", "-")
        fig.savefig(os.path.join(out_dir, f"acc_vs_bands_{ds}_{bb}{arch_tag}_e{ep}.pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_training_curves(df, out_dir):
    for (ds, bb, ep), g in df.groupby(["dataset", "backbone", "epochs"]):
        if ep == 0:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for key, gk in g.groupby("key"):
            curves = [pd.read_csv(os.path.join(d, "log.csv"))["val_acc"].values for d in gk["dir"]]
            n = min(len(c) for c in curves)
            arr = np.stack([c[:n] for c in curves])
            ax.plot(range(1, n + 1), arr.mean(0), label=key)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Held-out validation accuracy (%)"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.savefig(os.path.join(out_dir, f"val_curves_{ds}_{bb}_e{ep}.pdf"), bbox_inches="tight")
        plt.close(fig)


def latency_tables(lat, out_dir, suffix=""):
    rows = []
    for bname, st in lat["stages"].items():
        med = {k: v["median_ms_per_image"] for k, v in st.items() if "median_ms_per_image" in v}
        for k, v in med.items():
            rows.append({"batch": bname, "stage": k, "median_ms_per_image": v, "p10_ms_batch": st[k]["p10_ms"],
                         "p90_ms_batch": st[k]["p90_ms"], "n_timed": st[k]["n_timed"]})
        pre_y = med["preprocess_rgb_normalize"] + med["preprocess_rgb_to_y"] + med["forward_block_dct"]
        for dec, extra in [("cpu_decode_pillow", med["h2d_copy"]), ("gpu_decode_nvjpeg", 0.0)]:
            if dec not in med:
                continue
            rows.append({"batch": bname, "stage": f"PIPELINE pixel ({dec})",
                         "median_ms_per_image": med[dec] + extra + med["preprocess_rgb_normalize"] + med["backbone_pixel"]})
            for k in [s.split("_k")[1] for s in med if s.startswith("backbone_dct_k")]:
                rows.append({"batch": bname, "stage": f"PIPELINE DCT k={k} ({dec})",
                             "median_ms_per_image": med[dec] + extra + pre_y + med[f"backbone_dct_k{k}"]})
            for d in lat["protocol"].get("designs", []):   # all channels; the mask multiply is not timed (negligible)
                color, norm = d.split("/")[0], d.endswith("/norm")
                c = 192 if color == "ycbcr" else 64
                pre = med["preprocess_rgb_normalize"] + (
                    med["preprocess_rgb_to_ycbcr"] + med["forward_block_dct_ycbcr"] if color == "ycbcr"
                    else med["preprocess_rgb_to_y"] + med["forward_block_dct"])
                pre += med[f"dct_normalize_{c}ch"] if norm else 0.0
                rows.append({"batch": bname, "stage": f"PIPELINE DCT design {d} ({dec})",
                             "median_ms_per_image": med[dec] + extra + pre + med[f"backbone_dct_{d.replace('/', '-')}"]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"latency{suffix}.csv"), index=False)
    if isinstance(lat.get("energy_backbone_only"), dict) and "idle_power_w" in lat["energy_backbone_only"]:
        e = lat["energy_backbone_only"]
        pd.DataFrame([{"model": k, **v} for k, v in e.items() if isinstance(v, dict)]).to_csv(
            os.path.join(out_dir, f"energy{suffix}.csv"), index=False)


def plot_rate(rate_files, out_dir):
    rows = []
    for f in rate_files:
        rows += json.load(open(f))["results"]
    if not rows:
        return
    r = pd.DataFrame(rows).drop_duplicates(["model", "quality"])   # a quality may appear in two files
    r["dataset"] = r["model"].str.split("_").str[0]
    r["backbone"] = r["model"].str.split("_").str[1]
    r.to_csv(os.path.join(out_dir, "rate_accuracy.csv"), index=False)
    for (ds, bb), g in r.groupby(["dataset", "backbone"]):
        for xcol, xlabel, tag in [("bpp_total", "Bits per pixel (whole JPEG file, incl. headers)", ""),
                                  ("bpp_scan_only", "Bits per pixel (entropy-coded data only)", "_scan")]:
            fig, ax = plt.subplots(figsize=(6, 4))
            for name, gm in g.groupby("model"):
                gm = gm.sort_values(xcol)
                label = name.replace(f"{ds}_{bb}_", "").split("_e")[0]
                ax.plot(gm[xcol], gm["accuracy"], marker="o", markersize=3, label=label)
            ax.set_xlabel(xlabel); ax.set_ylabel("Test accuracy (%)"); ax.set_title(f"{ds}, {bb}", fontsize=9)
            ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.3)
            fig.savefig(os.path.join(out_dir, f"rate_accuracy_{ds}_{bb}{tag}.pdf"), bbox_inches="tight")
            plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--flops", default="flops.json")
    p.add_argument("--latency", default="latency.json")
    p.add_argument("--rate", nargs="*", default=sorted(glob.glob("rate_accuracy*.json")))
    p.add_argument("--tolerance", type=float, default=1.0)
    p.add_argument("--out", default="report")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_runs(a.runs_dir)
    if df.empty:
        raise SystemExit("no finished runs found")
    df["key"] = df.apply(config_key, axis=1)
    flops = json.load(open(a.flops)) if os.path.exists(a.flops) else None
    df.drop(columns=["mask_binary", "mask_probs"]).to_csv(os.path.join(a.out, "runs_summary.csv"), index=False)
    gt = grouped_table(df, flops)
    gt.to_csv(os.path.join(a.out, "grouped_results.csv"), index=False)
    cols = ["config", "n_seeds", "val_acc", "test_acc", "active_bands", "gmacs_as_implemented", "gmacs_if_stem_pruned"]
    with open(os.path.join(a.out, "grouped_results.tex"), "w") as f:
        for (ds, bb, ep, tq), g in gt.groupby(["dataset", "backbone", "epochs", gt["test_jpeg_q"].fillna(-1)]):
            f.write(f"% {ds} {bb} {ep} epochs test JPEG Q={'original' if tq == -1 else int(tq)}\n")
            f.write(g[cols].to_latex(index=False, escape=False, na_rep="--", float_format="%.3f") + "\n")
    with open(os.path.join(a.out, "lambda_selection.txt"), "w") as f:
        f.write(select_lambda(gt, a.tolerance) + "\n")
    mask_stability(df, a.out)
    plot_acc_vs_bands(gt, df, a.out)
    plot_training_curves(df, a.out)
    if flops:
        pd.DataFrame([{"dataset": r["dataset"], "backbone": r["backbone"], "design": d, "channels": v["channels"],
                       "GMACs_all_channels": v["macs_all_channels"] / 1e9,
                       **{f"GMACs_stem_k{k}": m / 1e9 for k, m in v["macs_pruned_k"].items()}}
                      for r in flops["results"] for d, v in r.get("designs", {}).items()]).to_csv(
            os.path.join(a.out, "flops_designs.csv"), index=False)
        pd.DataFrame([{"dataset": r["dataset"], "backbone": r["backbone"], "pixel_GMACs": r["pixel_macs"] / 1e9,
                       "dct_64ch_GMACs": r["dct_backbone_macs_64ch"] / 1e9,
                       **{f"dct_k{k}_GMACs": v / 1e9 for k, v in r["dct_backbone_macs_pruned_k"].items()}}
                      for r in flops["results"]]).to_csv(os.path.join(a.out, "flops.csv"), index=False)
    if os.path.exists(a.latency):
        latency_tables(json.load(open(a.latency)), a.out)
    for f in sorted(glob.glob(os.path.join(os.path.dirname(a.latency) or ".", "latency_*.json"))):   # other backbones
        latency_tables(json.load(open(f)), a.out, "_" + os.path.basename(f)[len("latency_"):-len(".json")])
    plot_rate(a.rate, a.out)
    print(open(os.path.join(a.out, "lambda_selection.txt")).read())
    print(f"report written to {a.out}/")


if __name__ == "__main__":
    main()
