#!/bin/bash
# FLOPs, latency/energy, real bit-rate, and the final report. Normally started by stage4_measure.sh
# after training has finished (latency must run on an otherwise idle GPU).
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=${GPU:-0}
IMAGENET_ROOT=${IMAGENET_ROOT:-}
LATENCY_IMAGES=${LATENCY_IMAGES:-}        # used for timing when ImageNet is missing (labels are not needed)
RATE_DESIGNS=${RATE_DESIGNS:-}            # e.g. "y/pool ycbcr/nopool_s1/norm pixel-pool pixel-cifar"; empty = all
CIFAR_ROOT=${CIFAR_ROOT:-./data}
WORKERS=${WORKERS:-16}

python "$KIT/measure_flops.py" --out flops.json

HAS_GPU=$(python "$KIT/prefetch.py" --has_gpu)
LAT_SRC=""
if [ -d "$IMAGENET_ROOT/val" ]; then LAT_SRC="$IMAGENET_ROOT/val"; elif [ -d "$LATENCY_IMAGES" ]; then LAT_SRC="$LATENCY_IMAGES"; fi
if [ -n "$LAT_SRC" ] && [ "$HAS_GPU" = "1" ]; then
  echo "latency images: $LAT_SRC"
  CUDA_VISIBLE_DEVICES=$GPU python "$KIT/profile_latency.py" --images "$LAT_SRC" --n_images 512 \
    --quality 90 --backbone resnet50 --batches 1 64 --dtype fp32 --out latency.json
else
  echo "skipping latency profiling (needs ImageNet val or COCO val2017 images, and a CUDA GPU)"
fi

# Real JPEG bit-rate vs accuracy, for every finished run with seed 42 (all seeds is fine too, just slower).
finished() { for d in "$@"; do [ -f "$d/result.json" ] && echo "$d"; done; return 0; }
CIFAR_RUNS=$(finished runs/cifar10_*_seed42 | grep -v testq || true)
if [ -n "$CIFAR_RUNS" ]; then
  CUDA_VISIBLE_DEVICES=$GPU python "$KIT/rate_accuracy.py" --dataset cifar10 --data_root "$CIFAR_ROOT" \
    --runs $CIFAR_RUNS --workers "$WORKERS" ${RATE_DESIGNS:+--designs $RATE_DESIGNS} ${RATE_EXTRA:-} \
    --out rate_accuracy_cifar10.json
fi
if [ -d "$IMAGENET_ROOT/val" ]; then
  IN_RUNS=$(finished runs/imagenet_*_seed42 | grep -v -e testq -e pixel || true)
  CUDA_VISIBLE_DEVICES=$GPU python "$KIT/rate_accuracy.py" --dataset imagenet --data_root "$IMAGENET_ROOT" \
    --runs $IN_RUNS --pretrained_pixel resnet50 --workers "$WORKERS" --qualities 10 30 50 70 90 ${RATE_EXTRA:-} \
    --out rate_accuracy_imagenet.json
fi

python "$KIT/make_report.py" --runs_dir runs --flops flops.json --latency latency.json --out report
