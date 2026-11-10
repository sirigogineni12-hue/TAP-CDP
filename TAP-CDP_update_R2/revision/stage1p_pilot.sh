#!/bin/bash
# Stage 1p: CIFAR-10 architecture pilot (CI-DCT designs + CIFAR pixel baseline), then pick the design.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
activate_env; enter_results
banner "stage 1p: CIFAR-10 architecture pilot"
python "$KIT_DIR/prefetch.py" --cifar_root "$CIFAR_ROOT" --weights resnet18   # already cached by setup_env.sh
GPUS=($(gpu_list)); N=${#GPUS[@]}
for s in $(seq 0 $((N - 1))); do
  GPU=${GPUS[$s]} SHARD=$s NSHARDS=$N PARALLEL=$CIFAR_PARALLEL EPOCHS_CIFAR=$EPOCHS_CIFAR \
    CIFAR_ROOT=$CIFAR_ROOT WORKERS=$WORKERS_CIFAR PILOT_SEEDS="$PILOT_SEEDS" \
    bash "$KIT_DIR/run_cifar_pilot.sh" > "logs/stage1p_pilot_shard$s.txt" 2>&1 &
done
wait
grep -h "FAILED" logs/stage1p_pilot_shard*.txt || true
python "$KIT_DIR/pick_cifar_arch.py" --runs_dir runs --seeds "$PILOT_SEEDS" --epochs "$EPOCHS_CIFAR" --out cifar_arch.env
banner "stage 1p finished"
