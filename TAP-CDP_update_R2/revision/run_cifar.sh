#!/bin/bash
# CIFAR-10 / ResNet-18 experiment matrix. Normally started by stage1_cifar.sh; writes to the current folder (results/).
# Example: GPU=1 PARALLEL=3 bash revision/run_cifar.sh
# Two GPUs: GPU=0 SHARD=0 NSHARDS=2 bash revision/run_cifar.sh & GPU=1 SHARD=1 NSHARDS=2 bash revision/run_cifar.sh &
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=${GPU:-0}
SHARD=${SHARD:-0}
NSHARDS=${NSHARDS:-1}
EXTRA=${EXTRA_ARGS:-}            # extra train_eval.py arguments (smoke tests only)
PARALLEL=${PARALLEL:-3}          # CIFAR runs are small; several fit on one GPU
EPOCHS=${EPOCHS_CIFAR:-100}
ROOT=${CIFAR_ROOT:-./data}
WORKERS=${WORKERS:-6}
SEEDS=${SEEDS:-"42 123 456"}
LAMBDAS=${LAMBDAS:-"0.01 0.05 0.1 0.5 1.0"}
KS=${KS:-"8 16 32"}
TACS_INITS=${TACS_INITS:-"2"}    # 2 = all bands start on; 0 (repository) diverges and is already reported from the first batch
ARCH=${CIFAR_ARCH_ARGS:-}        # design chosen by the pilot (cifar_arch.env); empty = repository design
PIXEL_STEMS=${PIXEL_STEMS:-"pool cifar"}
XU_LAMBDAS=${XU_LAMBDAS:-"0.0005 0.001 0.005 0.01 0.05"}   # Xu et al. penalize the channel count (sum), so a smaller scale
mkdir -p logs runs

# Download data and weights once, before parallel jobs start (stage1_cifar.sh does this itself).
if [ -z "${SKIP_DOWNLOAD:-}" ]; then
  python "$KIT/prefetch.py" --cifar_root "$ROOT" --weights resnet18
fi

JOBS=()
for s in $SEEDS; do
  JOBS+=("--mode none $ARCH --seed $s")
  for ps in $PIXEL_STEMS; do JOBS+=("--mode pixel --stem $ps --seed $s"); done
  for l in $LAMBDAS; do for ti in $TACS_INITS; do JOBS+=("--mode learned --lam $l --tacs_init $ti $ARCH --seed $s"); done; done
  for l in $XU_LAMBDAS; do JOBS+=("--mode xugate --lam $l $ARCH --seed $s"); done
  for k in $KS; do JOBS+=("--mode lowfreq --k $k $ARCH --seed $s" "--mode random --k $k $ARCH --seed $s"); done
done
MINE=()
for i in "${!JOBS[@]}"; do [ $((i % NSHARDS)) -eq "$SHARD" ] && MINE+=("${JOBS[$i]}"); done
echo "CIFAR-10: ${#MINE[@]} of ${#JOBS[@]} runs (shard $SHARD/$NSHARDS), $PARALLEL at a time on GPU $GPU"
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
