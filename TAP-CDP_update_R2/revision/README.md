# TAP-CDP revision experiments (electronics-4595630)

These scripts produce every number and figure needed to answer the three reviewers, directly
from real runs, on a SLURM GPU cluster (job files in this folder; set the account, partition and QoS lines for your cluster). They use `TACS` and `ModifiedResNet` unchanged from
the repository's `tap_hybridnet.py`, plus a corrected `block_dct` (see "Facts from the code").

**Rule for the revision:** every number in the revised paper and the response letter must come
from the files these scripts write. Do not edit result files or type numbers into plotting code.
If a run fails or gives a surprising result, report it as it is.

## Layout

```
~/TAP-CDP/            the authors' repository (unchanged)
├── revision/         this kit (all scripts in one folder)
└── results/          everything produced:
    ├── data/         CIFAR-10, pretrained weights cache (and ImageNet, if placed here)
    ├── runs/         one folder per training run: config.json, log.csv, result.json, last.pt
    ├── logs/         SLURM .out/.err files and per-run logs
    ├── report/       tables (.csv, .tex) and figures (.pdf) for the revision
    ├── flops.json, latency.json, rate_accuracy_*.json, imagenet_settings*.txt/.env
    └── revision_results.tar.gz   archive to send back (no datasets, no checkpoints)
```

## Steps (on the login node, from ~/TAP-CDP)

```bash
bash revision/setup_env.sh             # once: conda env "tapcdp", packages, CIFAR-10, weights
bash revision/get_coco.sh              # once, before job 4 if ImageNet is missing: COCO val2017 photos for latency
sbatch revision/job0_checks.sh         # 1. checks (minutes)
sbatch revision/job1p_pilot.sh         #    CIFAR-10 architecture pilot (see below), before job 1
sbatch revision/job1_cifar.sh          # 2. after job 0 finished: CIFAR-10 (several hours)
sbatch revision/job2_pick.sh           # 3. after job 1 finished: report + ImageNet settings (minutes)
sbatch revision/job3_imagenet.sh       # 4. after job 2 finished: ImageNet (resubmit if it hits 24 h)
sbatch revision/job4_measure.sh        # 5. after job 3 finished: measurements, report, archive
```

Before each submission, check the previous job finished cleanly: `squeue -u $USER` shows
nothing running, and the end of `results/logs/<job name>_<id>.out` says "stage N finished".
`setup_env.sh` needs internet, so it runs on the login node; the jobs then work offline.

**Latency without ImageNet.** Timing needs real photos but no labels, so `job4_measure.sh` uses COCO
val2017 (`get_coco.sh`, about 0.8 GB) when ImageNet is missing. State this in the paper: latency and energy
were measured on 512 COCO val2017 images centre-cropped to 224x224 and re-encoded at JPEG quality 90, with
random weights (timing does not depend on them). Besides the repository design, the colour designs
(YCbCr; max-pool kept, removed, removed + stride-1 layer2; normalized) are timed at ImageNet size.

**ImageNet** is not downloaded automatically (it needs an ImageNet account). Point
`IMAGENET_ROOT` in `config.sh` at a copy with `train/` and `val/`, each holding one folder per
class. Without it, jobs 3 and 4 skip the ImageNet parts and only CIFAR-10 results are produced.

## Second CIFAR-10 batch: architecture pilot (job1p), then the full matrix again

The first batch (repository design: luminance only, max-pool after the 1x1 stem, raw coefficients) gave
CI-DCT 50.5% against 83.0% for the pixel ResNet-18. The repository's TACS initialization (w = 0) diverges:
the zero input gives batch-norm a zero variance and a first-step gradient of about 1e12 on the mask logits.
Those results stay in `results/runs/` and are reported as they are.

`job1p_pilot.sh` (about 1-1.5 h) trains CI-DCT, all bands with no mask, for every design:
colour {Y, YCbCr 4:4:4} x stem {max-pool, no max-pool, no max-pool + stride-1 layer2} x coefficient
normalization {off, on}, seeds 42 and 123. It also trains the CIFAR-adapted pixel ResNet-18 (3x3 stem, no max-pool) as
the fair pixel baseline. `pick_cifar_arch.py` then picks the design on held-out validation accuracy (rule in
its docstring, fixed in advance; test accuracy unused) and writes `results/cifar_arch.env`. Because the
choice is made with CI-DCT only, it cannot favour TACS or any other mask. Report every pilot row
(`report/cifar_arch_pilot.csv`) in the paper.

After checking the pilot, `sbatch revision/job1_cifar.sh` runs the full matrix on the chosen design:
TACS (init 2), the Xu et al. (CVPR 2020) gate re-implemented as the baseline Reviewer 2 asked for, low-frequency
masks (luma-weighted for YCbCr, as in Xu et al.'s static selection), random masks, CI-DCT, and both pixel
baselines. The finished first-batch runs are skipped automatically (run ids differ by design).

## Additional experiments without full ImageNet (jobs 5 and 6)

ImageNet-1k (about 150 GB, 1.28 M training images) could not be obtained and trained within the revision period.
Instead:
- `job5_cifar_extra.sh`: budgeted TACS (`--mode topk`, exactly k = 8/16/24/48 learned channels, 3 seeds) on the
  CIFAR-10 design chosen by the pilot, so learned and fixed masks are compared at exactly the same channel
  count; then the low-rate JPEG curve (quality 1-7) for that design and both pixel baselines.
- `job6_imagenette.sh` (run `bash revision/get_imagenette.sh` once first): Imagenette, a 10-class ImageNet
  subset at 224x224, with ResNet-50 and ResNet-18, 3 seeds. Pixel baseline, CI-DCT (YCbCr, normalized; max-pool kept
  and removed), and budgeted TACS / low-frequency / random masks at k = 24 and 48. Same fine-tuning recipe for
  every model, fixed before any run (`config.sh`). Then real JPEG bit-rate at 224x224, ResNet-18 latency/energy.

## Jobs

| Job | GPUs / memory | What it does |
|---|---|---|
| `job0_checks.sh` | 1 / 94G | GPU test, corrected-DCT test, FLOPs |
| `job1_cifar.sh` | 2 / 188G | full CIFAR-10 matrix (3 seeds), split across both GPUs |
| `job2_pick.sh` | 1 / 94G | CIFAR-10 report; picks λ and the TACS initialization for ImageNet from validation accuracy (reason written to `results/imagenet_settings_reason.txt`) |
| `job3_imagenet.sh` | 2 / 188G | ImageNet runs, one per GPU at a time; if the 24 h limit is reached, submit it again and it continues where it stopped |
| `job4_measure.sh` | 1 / 94G | latency and energy (idle GPU), real JPEG bit-rate, final report, `results/revision_results.tar.gz` |

Each job file only sets the SLURM options and calls the matching `stage*.sh` script with `srun`.
If a job fails, read its `.err` file in `results/logs/`, fix the problem and submit the same job
again: finished runs are skipped and interrupted runs resume.

**Timing.** CIFAR-10 should finish within a day. ImageNet is limited by JPEG decoding on the
8 CPUs allowed per job: check `epoch_time_s` in `results/runs/imagenet_*/log.csv` after the first
epoch. Defaults are 10 fine-tuning epochs and one λ for ImageNet; change `EPOCHS_IMAGENET`,
`IMAGENET_NUM_LAMBDAS` or `SEEDS` in `config.sh` if needed. The paper must
state exactly the schedule that was used.

## How λ and the TACS initialization are chosen for ImageNet

The largest λ whose mean CIFAR-10 held-out validation accuracy is within 1 point of CI-DCT and
that keeps at least one band active; the repository initialization is preferred if it
qualifies, otherwise all bands start on (init 2). Test accuracy is never used. State this rule
in the paper.

## What each output answers

| Output (in results/) | Reviewer question |
|---|---|
| `flops.json`, `report/flops.csv` | R1-8, R2-2, R3: real MACs for pixel ResNet, CI-DCT, TAP-CDP as implemented, and with the stem pruned to k bands |
| `latency.json`, `report/latency.csv` | R1-7, R2-4, R3: measured per-stage latency with the full protocol (batch size, warm-up, iterations, synchronization, CPU vs GPU decode, transfer, median) |
| `report/energy.csv` | R2-4: GPU board energy from NVML power sampling |
| `rate_accuracy_*.json`, `report/rate_accuracy.pdf` | R2-3, R2-5: real JPEG file sizes (bpp with and without headers, stated image size) and accuracy on the decoded files; JPEG pixel baseline as a curve over quality factors |
| `report/grouped_results.*` | R2-5, R3: mean ± std over three distinct seeds (42, 123, 456), CI-DCT, λ sweep, pixel baseline |
| `report/acc_vs_bands_*.pdf` | R2-5, R3: learned TACS vs random and low-frequency masks at several matched sparsity levels |
| `report/lambda_selection.txt` | R2-7, R3: one λ (the mask-mean penalty), chosen on a held-out split; test set evaluated once |
| `report/masks_*.pdf`, `report/mask_stability.csv` | R1-9, R2-6: the real learned masks per seed and their agreement across seeds |
| `report/val_curves_*.pdf` | reproducibility: validation accuracy per epoch |

## Facts from the code (no GPU needed)

Problems in the repository code:
- **`block_dct` crashes on the paper's input sizes.** Its last reshape reads the shape of the
  tensor before the permute, so it only runs when the image height is 512; it fails on 32x32
  (CIFAR-10) and 224x224 (ImageNet). The kit uses a corrected version, and `test_block_dct.py`
  shows it matches the original exactly where the original runs. The original `tap_hybridnet.py`
  is kept unchanged for transparency; all revision experiments use the corrected function.
- **TACS starts with every band switched off** (`w = 0`, and `sigmoid(0) = 0.5` fails the strict
  `> 0.5` test). In a short test on real CIFAR-10 images bands did switch on after a few steps;
  to be safe, CIFAR-10 runs both the repository initialization and an all-on start (init 2).

Statements in the paper that must change:
- **Colour:** only luminance is used, `Y = 0.299 R + 0.587 G + 0.114 B` on normalized RGB, then an
  8x8 block DCT gives 64 channels at H/8 x W/8. Cb and Cr are not used ("4:4:4 YUV" is wrong).
- **Tensor shapes (ImageNet):** 3x224x224 RGB -> 1x224x224 Y -> 64x28x28 DCT -> mask -> 1x1 conv
  stem 64->64 -> BN, ReLU, max-pool (64x14x14) -> ResNet layers 1-4 -> 2048 -> classifier.
  CIFAR-10: 3x32x32 -> 64x4x4 -> max-pool 2x2 -> layers 1-4.
- **Masking** multiplies channels by 0 or 1; all 64 channels still enter the stem, so the mask
  does not change the computation as implemented.
- **Objective:** cross-entropy + λ · mean(mask probabilities). There is no bit-rate term and only
  one λ; the λ_R / λ_C formulation is not what was trained.
- **Pipeline:** images are decoded to RGB by the data loader and the DCT is computed on the GPU;
  skipping entropy decoding or the IDCT is not implemented.
- **Temperature:** τ is annealed linearly from 5.0 to 0.1 over all training epochs.
- **Measured MACs (ImageNet 224x224):** pixel ResNet-50 4.112 G; DCT ResNet-50 (CI-DCT and
  TAP-CDP as implemented) 0.296 G; stem pruned to 16 bands 0.294 G. The saving comes from running
  on the 28x28 DCT grid, not from TACS. Quote `results/flops.json` from the cluster run.
