#!/bin/bash
# ImageNet / ResNet-50 experiments, sharded over GPUs. Normally started by stage3_imagenet.sh; writes to the current folder (results/).
# IMAGENET_ROOT must contain train/ and val/, each with one folder per class.
# Example (two GPUs):
#   GPU=0 SHARD=0 NSHARDS=2 bash revision/run_imagenet.sh &
#   GPU=1 SHARD=1 NSHARDS=2 bash revision/run_imagenet.sh &
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=${GPU:-0}
SHARD=${SHARD:-0}
NSHARDS=${NSHARDS:-1}
EPOCHS=${EPOCHS_IMAGENET:-30}
ROOT=${IMAGENET_ROOT:?set IMAGENET_ROOT to the folder containing train/ and val/}
WORKERS=${WORKERS:-24}
BATCH=${BATCH:-256}
SEEDS=${SEEDS:-"42 123 456"}
LAMBDAS=${LAMBDAS:-"0.1 1.0"}     # choose after reading the CIFAR-10 lambda_selection.txt if possible
KS=${KS:-"16"}
EXTRA=${EXTRA_ARGS:-}            # extra train_eval.py arguments (smoke tests only)
TACS_INIT=${TACS_INIT:-0}       # 0 = repository behaviour; set from the CIFAR-10 results (see README)
mkdir -p logs runs
[ -z "${SKIP_DOWNLOAD:-}" ] && python "$KIT/prefetch.py" --weights resnet50

JOBS=()
# Pixel baseline: published torchvision ResNet-50 weights evaluated by us (no training), original and Q=90 test images.
JOBS+=("--mode pixel --seed 42 --epochs 0")
JOBS+=("--mode pixel --seed 42 --epochs 0 --test_jpeg_quality 90")
for s in $SEEDS; do
  JOBS+=("--mode none --seed $s --epochs $EPOCHS")
  for l in $LAMBDAS; do JOBS+=("--mode learned --lam $l --tacs_init $TACS_INIT --seed $s --epochs $EPOCHS"); done
done
for k in $KS; do
  JOBS+=("--mode lowfreq --k $k --seed 42 --epochs $EPOCHS" "--mode random --k $k --seed 42 --epochs $EPOCHS")
done

i=0
for j in "${JOBS[@]}"; do
  if [ $((i % NSHARDS)) -eq "$SHARD" ]; then
    name=$(echo "$j" | tr -s ' -' '_')
    echo "[GPU $GPU] start: $j"
    if CUDA_VISIBLE_DEVICES=$GPU python "$KIT/train_eval.py" --dataset imagenet --data_root "$ROOT" --backbone resnet50 \
         --batch_size "$BATCH" --workers "$WORKERS" $EXTRA $j > "logs/imagenet_${name}.log" 2>&1; then
      echo "[GPU $GPU] done: $j"
    else
      echo "[GPU $GPU] FAILED: $j (see logs/imagenet_${name}.log)"
    fi
  fi
  i=$((i + 1))
done
