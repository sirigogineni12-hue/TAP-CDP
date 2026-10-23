"""Train one configuration with one seed, then evaluate it once on the test set.

Every run writes runs/<run_id>/{config.json, log.csv, result.json, last.pt}.
Model selection (e.g. choosing lambda) must use val_acc from the held-out split,
never test_acc. The script is resumable and skips runs that already finished.
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (NUM_CLASSES, build_datasets, build_model, code_fingerprint, env_info,
                    estimate_dct_stats, set_seed, write_json)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "imagenet", "imagenette"], required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--backbone", choices=["resnet18", "resnet50"], required=True)
    p.add_argument("--mode", choices=["learned", "topk", "xugate", "none", "random", "lowfreq", "pixel"], required=True,
                   help="learned=TACS, topk=budgeted TACS (exactly k), xugate=Xu et al. gate, none=CI-DCT (all bands), "
                        "random/lowfreq=fixed k-band mask, pixel=RGB baseline")
    p.add_argument("--lam", type=float, default=0.0,
                   help="penalty weight: TACS uses lambda*mean(mask) (repository), xugate lambda*sum(gates) (Xu et al. Eq. 2)")
    p.add_argument("--color", choices=["y", "ycbcr"], default="y", help="y = repository (luminance only); ycbcr = 4:4:4")
    p.add_argument("--stem", choices=["pool", "nopool", "nopool_s1", "cifar"], default="pool",
                   help="pool = repository; nopool/nopool_s1 for DCT models; cifar = 3x3 stem without max-pool (pixel only)")
    p.add_argument("--dct_norm", action="store_true", help="normalize each DCT channel by training-set mean/std")
    p.add_argument("--tacs_init", type=float, default=0.0,
                   help="initial TACS logit; 0.0 = repository behaviour (all bands start off), e.g. 2.0 = all start on")
    p.add_argument("--k", type=int, default=None, help="number of kept channels for topk/random/lowfreq")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=100, help="0 = evaluate the pretrained model without training (pixel baselines)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val_per_class", type=int, default=None,
                   help="held-out images per class (default 500 for CIFAR-10, 10 for ImageNet)")
    p.add_argument("--test_jpeg_quality", type=int, default=None,
                   help="re-encode test images at this JPEG quality before evaluation")
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--max_steps_per_epoch", type=int, default=None, help="smoke tests only")
    p.add_argument("--max_eval_batches", type=int, default=None, help="smoke tests only")
    p.add_argument("--out", default="runs")
    args = p.parse_args()
    if args.mode in ("random", "lowfreq", "topk") and args.k is None:
        p.error("--k is required for topk/random/lowfreq")
    if args.mode == "pixel" and (args.stem not in ("pool", "cifar") or args.color != "y" or args.dct_norm):
        p.error("pixel mode takes only --stem pool|cifar")
    if args.mode != "pixel" and args.stem == "cifar":
        p.error("--stem cifar is for the pixel baseline")
    if args.val_per_class is None:
        args.val_per_class = {"cifar10": 500, "imagenette": 50}.get(args.dataset, 10)
    return args


def run_id(a):
    init = f"_init{a.tacs_init:g}" if (a.mode == "learned" and a.tacs_init != 0.0) else ""
    tag = {"learned": f"learned_lam{a.lam:g}{init}", "xugate": f"xugate_lam{a.lam:g}", "random": f"random_k{a.k}",
           "lowfreq": f"lowfreq_k{a.k}", "topk": f"topk_k{a.k}"}.get(a.mode, a.mode)
    # Repository defaults add nothing, so the run ids of the first CIFAR-10 batch stay valid.
    arch = (f"_{a.color}" if a.color != "y" else "") + (f"_stem{a.stem}" if a.stem != "pool" else "") + \
           ("_dctnorm" if a.dct_norm else "")
    q = f"_testq{a.test_jpeg_quality}" if a.test_jpeg_quality else ""
    return f"{a.dataset}_{a.backbone}_{tag}{arch}_e{a.epochs}{q}_seed{a.seed}"


def autocast_ctx(device, amp):
    if device.type != "cuda" or amp == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


@torch.no_grad()
def evaluate(model, loader, device, amp, max_batches=None):
    model.eval()
    if hasattr(model, "gate"):
        model.gate.reset_stats()   # gate statistics then describe exactly this evaluation set
    correct = total = 0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx(device, amp):
            logits, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def main():
    a = parse_args()
    rid = run_id(a)
    out_dir = os.path.join(a.out, rid)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "result.json")):
        print(f"[skip] {rid} already finished")
        return

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_set, val_set, test_set = build_datasets(a.dataset, a.data_root, a.val_per_class, a.test_jpeg_quality)
    g = torch.Generator().manual_seed(a.seed)
    train_loader = DataLoader(train_set, a.batch_size, shuffle=True, num_workers=a.workers,
                              pin_memory=True, drop_last=True, generator=g, persistent_workers=a.workers > 0)
    val_loader = DataLoader(val_set, a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True)
    test_loader = DataLoader(test_set, a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True)

    if a.epochs > 0 and len(train_loader) == 0:
        raise SystemExit(f"training set ({len(train_set)} images) is smaller than one batch ({a.batch_size})")
    model = build_model(a, NUM_CLASSES[a.dataset]).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, nesterov=True, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs))
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and a.amp == "fp16") else None
    ce = nn.CrossEntropyLoss()

    config = dict(vars(a), run_id=rid, n_train=len(train_set), n_val=len(val_set), n_test=len(test_set))
    write_json(os.path.join(out_dir, "config.json"), config)

    ckpt_path = os.path.join(out_dir, "last.pt")
    start_epoch, train_time = 0, 0.0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        if scaler and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch, train_time = ck["epoch"] + 1, ck["train_time"]
        print(f"[resume] {rid} from epoch {start_epoch}")
    elif a.dct_norm:
        mean, std = estimate_dct_stats(model, train_loader, device)
        model.set_dct_stats(mean, std)
        print(f"{rid} | DCT statistics from training images: mean |mean| {mean.abs().mean():.3f}, "
              f"std range {std.min():.3f}-{std.max():.3f}")

    log_path = os.path.join(out_dir, "log.csv")
    fields = ["epoch", "lr", "tau", "train_loss", "task_loss", "mask_mean", "active_bands", "val_acc", "epoch_time_s"]
    if start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    val_acc = None
    for epoch in range(start_epoch, a.epochs):
        model.train()
        if a.mode == "learned":
            model.tacs.current_epoch = epoch
        t0 = time.time()
        sums = {"loss": 0.0, "task": 0.0, "mask": 0.0}
        steps = 0
        for x, y in train_loader:
            if a.max_steps_per_epoch is not None and steps >= a.max_steps_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device, a.amp):
                logits, m = model(x)
                task = ce(logits, y)
                if a.mode == "learned":      # repository objective: lambda * mean of the mask probabilities
                    mask_mean = m.float().mean()
                elif a.mode == "xugate":     # Xu et al. Eq. (2): lambda * number of selected channels (per image)
                    mask_mean = m.float().sum(1).mean()
                else:
                    mask_mean = torch.zeros((), device=device)
                loss = task + a.lam * mask_mean
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            sums["loss"] += loss.item(); sums["task"] += task.item(); sums["mask"] += mask_mean.item()
            steps += 1
        lr_now = opt.param_groups[0]["lr"]
        sched.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0
        train_time += epoch_time

        val_acc = evaluate(model, val_loader, device, a.amp, a.max_eval_batches)
        tau = model.tacs.get_temperature() if a.mode == "learned" else ""
        if a.mode == "xugate":    # mean number of channels kept per validation image
            active = round(float(model.mask_state()[0].sum()), 2)
        else:
            active = int(model.mask_state()[1].sum().item()) if a.mode != "pixel" else ""
        row = [epoch + 1, f"{lr_now:.6g}", tau, sums["loss"] / steps, sums["task"] / steps,
               sums["mask"] / steps, active, f"{val_acc:.4f}", f"{epoch_time:.1f}"]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"{rid} | epoch {epoch + 1}/{a.epochs} | loss {sums['loss'] / steps:.4f} | val {val_acc:.2f}% "
              f"| bands {active} | {epoch_time:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None, "epoch": epoch,
                    "train_time": train_time}, ckpt_path)

    # Final model (last epoch) is evaluated once on the test set; no checkpoint is chosen by test accuracy.
    if val_acc is None:
        val_acc = evaluate(model, val_loader, device, a.amp, a.max_eval_batches)
    test_acc = evaluate(model, test_loader, device, a.amp, a.max_eval_batches)
    result = {"run_id": rid, "config": config, "val_acc": val_acc, "test_acc": test_acc,
              "train_time_s": train_time, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
              "env": env_info(), "code_sha256": code_fingerprint()}
    if a.mode != "pixel":
        probs, binary, tau = model.mask_state()
        # For the input-dependent gate, active_bands is the mean number of channels kept per test image.
        active = float(probs.sum()) if a.mode == "xugate" else int(binary.sum().item())
        result.update(mask_probs=probs.tolist(), mask_binary=binary.tolist(), active_bands=active, tau_final=tau)
    write_json(os.path.join(out_dir, "result.json"), result)
    print(f"[done] {rid} | val {val_acc:.2f}% | test {test_acc:.2f}%")


if __name__ == "__main__":
    main()
