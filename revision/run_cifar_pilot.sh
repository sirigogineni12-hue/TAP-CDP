#!/bin/bash
# CIFAR-10 architecture pilot: CI-DCT (all bands, no mask) under each input/stem design, plus the
# CIFAR-adapted pixel baseline. Normally started by stage1p_pilot.sh; writes to the current folder (results/).
# The design is chosen on held-out validation accuracy by pick_cifar_arch.py, before any mask method is run,
# so the choice cannot favour TACS or any other mask.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=${GPU:-0}
SHARD=${SHARD:-0}
NSHARDS=${NSHARDS:-1}
EXTRA=${EXTRA_ARGS:-}            # extra train_eval.py arguments (smoke tests only)
PARALLEL=${PARALLEL:-2}
EPOCHS=${EPOCHS_CIFAR:-100}
ROOT=${CIFAR_ROOT:-./data}
WORKERS=${WORKERS:-1}
SEEDS=${PILOT_SEEDS:-"42 123"}
COLORS=${PILOT_COLORS:-"y ycbcr"}
STEMS=${PILOT_STEMS:-"pool nopool nopool_s1"}
NORMS=${PILOT_NORMS:-"off on"}
mkdir -p logs runs

JOBS=()
for s in $SEEDS; do
  for c in $COLORS; do for st in $STEMS; do for n in $NORMS; do
    norm=""; [ "$n" = "on" ] && norm="--dct_norm"
    JOBS+=("--mode none --color $c --stem $st $norm --seed $s")
  done; done; done
  JOBS+=("--mode pixel --stem cifar --seed $s")
done
MINE=()
for i in "${!JOBS[@]}"; do [ $((i % NSHARDS)) -eq "$SHARD" ] && MINE+=("${JOBS[$i]}"); done
echo "CIFAR-10 pilot: ${#MINE[@]} of ${#JOBS[@]} runs (shard $SHARD/$NSHARDS), $PARALLEL at a time on GPU $GPU"
run_one() {
  local name; name=$(echo "$1" | tr -s ' -' '_')
  if CUDA_VISIBLE_DEVICES=$GPU python "$KIT/train_eval.py" --dataset cifar10 --data_root "$ROOT" --backbone resnet18 \
       --epochs "$EPOCHS" --workers "$WORKERS" $EXTRA $1 > "logs/cifar10_${name}.log" 2>&1; then
    echo "done: $1"
  else
    echo "FAILED: $1 (see logs/cifar10_${name}.log)"
  fi
}
for j in "${MINE[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 2; done
  run_one "$j" &
done
wait
