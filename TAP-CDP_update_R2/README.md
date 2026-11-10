# TAP-CDP: Task-Aware DCT Coefficient Selection for JPEG Bitstream-Domain Classification

Code, experiment scripts, logs and result files for the manuscript
*"Task-Aware DCT Coefficient Selection for JPEG Bitstream-Domain Image Classification: A Measured Study of
Accuracy, Bit-Rate, Decoding Cost and Latency"* (MDPI *Electronics*, resubmission of manuscript electronics-4595630).

> **Note on the first submission.** Several values in the first submission (the per-stage latency
> decomposition, energy values, the FLOPs of the pruned models, the bit-rate table, and the curves and
> mask figures) were not produced by measured runs of this code; an earlier version of this README listed
> the hardware metrics as *simulated*. These results have been withdrawn. Every number in the current
> manuscript is produced by the scripts in `revision/` and is stored in `results/`.

## Repository layout

| Path | Content |
|---|---|
| `tap_hybridnet.py` | Original TACS module and `ModifiedResNet` (kept unchanged; the revision imports them) |
| `train.py`, `main.py`, `dataset.py`, `models.py`, `losses.py`, `compression.py`, `eval.py`, `config.yaml`, `scripts/` | Original training code of the first submission |
| `revision/` | All experiments: training/evaluation, design pilot, FLOPs, latency/energy, real JPEG bit-rate, the JPEG bitstream reader, bitstream evaluation, decoder timing, report and SLURM job files (see `revision/README.md`) |
| `results/` | Everything produced by the runs (no datasets, no model checkpoints) |
| `legacy_first_submission/` | Figures of the first submission (withdrawn) |

Two problems of the original code are documented and handled in `revision/`:
- `tap_hybridnet.block_dct` fails unless the image height is 512 pixels. `revision/common.py` contains a
  corrected version; `revision/test_block_dct.py` shows that both agree wherever the original runs.
- TACS starts with every coefficient switched off (`w = 0`), which makes training diverge. The revision
  runs use an all-on start (`--tacs_init 2`); the diverging runs are kept in `results/` and reported.

## Bitstream pipeline (added for the resubmission)

- `revision/jpegcoef.c`, `revision/build_jpegcoef.sh`: a small libjpeg-turbo library that returns the
  quantized DCT coefficients of a JPEG file (`jpeg_read_coefficients`, entropy decoding only) and, for
  comparison, the fully decoded image.
- `revision/bitstream.py`: Python wrapper; `coefficient_tensor()` gives the dequantized coefficients of a
  4:4:4 JPEG file in the channel layout of the network (`python revision/bitstream.py --selftest`).
- Models trained with `--color ycbcr_jpeg` use the JPEG colour convention, so they classify these
  coefficients directly (`DctClassifier.classify_coefficients`).
- `revision/eval_bitstream.py`: accuracy of the pixel path and of the bitstream path on the same files.
- `revision/bitstream_bench.py`: single-thread decoder cost (full decoding against coefficient reading).

## Where each table and figure of the manuscript comes from

| Manuscript | File(s) in `results/` |
|---|---|
| Table 3 (MACs) | `flops.json`, `report/flops.csv`, `report/flops_designs.csv` |
| Table 4 (CIFAR-10 design study) | `report/cifar_arch_pilot.csv`, `cifar_arch_reason.txt`, `cifar_arch.env` |
| Table 5, Figure 2 (CIFAR-10 selection methods) | `report/grouped_results.csv` (dataset `cifar10`, arch `ycbcr/nopool_s1/norm`), `report/acc_vs_bands_cifar10_resnet18_ycbcr-nopool_s1-norm_e100.pdf` |
| Figure 3, mask stability | `report/masks_cifar10_resnet18_ycbcr-nopool_s1-norm_lam0.5_init2.pdf`, `report/mask_stability.csv` |
| Table 6 (Imagenette, both input conventions) | `report/grouped_results.csv` (dataset `imagenette`; arch `ycbcr/...` and `ycbcr_jpeg/...`) |
| Table 7 (ImageNet-1k) | `runs/imagenet_resnet50_*/result.json`, `report/grouped_results.csv` (dataset `imagenet`) |
| Table 8 (pixel path vs bitstream path) | `bitstream_eval_imagenet.json`, `bitstream_eval_imagenette.json` |
| Table 9, Figure 4 (Imagenette bit-rate) | `rate_accuracy_imagenette.json`, `report/rate_accuracy.csv`; Figure 4: `python revision/plot_rate_figure.py` |
| Table 10, Figure 5 (ImageNet-1k bit-rate) | `bitstream_eval_imagenet.json`; Figure 5: `python revision/plot_rate_imagenet.py` |
| Table 11 (decoding cost) | `bitstream_bench_imagenet.json` (ImageNet-1k images), `bitstream_bench.json` (COCO images) |
| Table 12 (latency, energy) | `latency.json` (ResNet-50), `latency_resnet18.json`, `report/latency*.csv`, `report/energy*.csv` |
| Table 13 (original design) | `report/grouped_results.csv` (arch `y/pool`) |
| CIFAR-10 bit-rate (text) | `rate_accuracy_cifar10.json`, `rate_accuracy_cifar10_lowq.json` |
| Choice of lambda | `report/lambda_selection.txt` |

Every training run has its own folder `results/runs/<run id>/` with `config.json` (all arguments),
`log.csv` (per-epoch loss, temperature, kept channels, validation accuracy, epoch time) and `result.json`
(validation and test accuracy, the learned mask, software/GPU environment and a SHA-256 fingerprint of the
code). SLURM outputs and per-run logs are in `results/logs/`. Machine-specific paths were replaced by
`<repo>`, `<home>`, `<private-storage>` and `<shared-datasets>`.

## Reproducing

```bash
bash revision/setup_env.sh          # Python environment, CIFAR-10, pretrained weights
bash revision/get_imagenette.sh     # Imagenette (about 0.33 GB)
bash revision/build_jpegcoef.sh     # JPEG coefficient reader (needs libjpeg-turbo headers)
```

ImageNet-1k is not downloaded by the scripts (it needs an ImageNet account): set `IMAGENET_ROOT` in
`revision/config.sh` to a folder with `train/` and `val/` (one subfolder per class);
`revision/get_imagenet_val_hf.py` can build `val/` from the Hugging Face copy of the validation set, and
`revision/job9_prep_imagenet256.sh` makes the 256-pixel training copy. The jobs (`revision/job*.sh`) are
SLURM scripts; set `--account`, `--partition` and `--qos` for your cluster. Order and outputs are described
in `revision/README.md`. Experiments ran on NVIDIA RTX PRO 6000 Blackwell GPUs.

## Datasets

CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html), Imagenette (https://github.com/fastai/imagenette),
ImageNet-1k (https://www.image-net.org) and COCO val2017 (https://cocodataset.org) are not included here.
