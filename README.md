# TAP-CDP: Task-Aware DCT Coefficient Selection for Compressed-Domain Classification

Code, experiment scripts, logs and result files for the revised manuscript
*"Task-Aware DCT Coefficient Selection for Compressed-Domain Image Classification: A Measured Study of
Accuracy, Real Bit-Rate and Latency"* (MDPI *Electronics*, manuscript electronics-4595630, revision 1).

> **Note on the first submission.** Several values in the first submission (the per-stage latency
> decomposition, energy values, the FLOPs of the pruned models, the bit-rate table, and the curves and
> mask figures) were not produced by measured runs of this code; an earlier version of this README listed
> the hardware metrics as *simulated*. These results have been withdrawn. Every number in the revised
> manuscript is produced by the scripts in `revision/` and is stored in `results/`.

## Repository layout

| Path | Content |
|---|---|
| `tap_hybridnet.py` | Original TACS module and `ModifiedResNet` (kept unchanged; the revision imports them) |
| `train.py`, `main.py`, `dataset.py`, `models.py`, `losses.py`, `compression.py`, `eval.py`, `config.yaml`, `scripts/` | Original training code of the first submission |
| `revision/` | All revision experiments: training/evaluation, design pilot, FLOPs, latency/energy, real JPEG bit-rate, report and SLURM job files (see `revision/README.md`) |
| `results/` | Everything produced by the revision runs (no datasets, no model checkpoints) |

Two problems of the original code are documented and handled in `revision/`:
- `tap_hybridnet.block_dct` fails unless the image height is 512 pixels. `revision/common.py` contains a
  corrected version; `revision/test_block_dct.py` shows that both agree wherever the original runs.
- TACS starts with every coefficient switched off (`w = 0`), which makes training diverge. The revision
  runs use an all-on start (`--tacs_init 2`); the diverging runs are kept in `results/` and reported.

## Where each table and figure of the revised manuscript comes from

| Manuscript | File(s) in `results/` |
|---|---|
| Table 3 (MACs) | `flops.json`, `report/flops.csv`, `report/flops_designs.csv` |
| Table 4 (CIFAR-10 design study) | `report/cifar_arch_pilot.csv`, `cifar_arch_reason.txt`, `cifar_arch.env` |
| Table 5, Figure 2 (CIFAR-10 selection methods) | `report/grouped_results.csv` (dataset `cifar10`, arch `ycbcr/nopool_s1/norm`), `report/acc_vs_bands_cifar10_resnet18_ycbcr-nopool_s1-norm_e100.pdf` |
| Figure 3, mask stability | `report/masks_cifar10_resnet18_ycbcr-nopool_s1-norm_lam0.5_init2.pdf`, `report/mask_stability.csv` |
| Table 6 (Imagenette) | `report/grouped_results.csv` (dataset `imagenette`) |
| Table 7, Figure 4 (real bit-rate) | `rate_accuracy_imagenette.json`, `rate_accuracy_cifar10*.json`, `report/rate_accuracy.csv`; Figure 4: `python revision/plot_rate_figure.py` |
| Table 8 (latency, energy) | `latency.json` (ResNet-50), `latency_resnet18.json`, `report/latency*.csv`, `report/energy*.csv` |
| Table 9 (original design) | `report/grouped_results.csv` (arch `y/pool`) |
| Validation curves | `report/val_curves_*.pdf` |
| Choice of lambda | `report/lambda_selection.txt`, `imagenet_settings_reason.txt` |

Every training run has its own folder `results/runs/<run id>/` with `config.json` (all arguments),
`log.csv` (per-epoch loss, temperature, kept channels, validation accuracy, epoch time) and `result.json`
(validation and test accuracy, the learned mask, software/GPU environment and a SHA-256 fingerprint of the
code). SLURM outputs and per-run logs are in `results/logs/`. Absolute paths of the machine that ran the
experiments were replaced by `<repo>` and `<home>`.

## Reproducing

```bash
bash revision/setup_env.sh          # conda environment, CIFAR-10, pretrained weights
bash revision/get_imagenette.sh     # Imagenette (about 0.33 GB)
bash revision/get_coco.sh           # COCO val2017 photos for latency timing (about 0.8 GB)
sbatch revision/job0_checks.sh      # then job1p_pilot, job1_cifar, job2_pick, job4_measure, job5_cifar_extra, job6_imagenette
```

Set `--account`, `--partition` and `--qos` in the `revision/job*.sh` files for your cluster. The order of the
jobs and what each produces are described in `revision/README.md`. Experiments ran on NVIDIA RTX PRO 6000
Blackwell GPUs with PyTorch 2.11 (CUDA 12.8); seeds 42, 123 and 456.

## Datasets

CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html), Imagenette (https://github.com/fastai/imagenette)
and COCO val2017 (https://cocodataset.org) are downloaded by the scripts above and are not included here.
