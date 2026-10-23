#!/bin/bash
# Stage 1: the full CIFAR-10 matrix, split across the GPUs of this job.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
activate_env; enter_results
banner "stage 1: CIFAR-10"
python "$KIT_DIR/prefetch.py" --cifar_root "$CIFAR_ROOT" --weights resnet18   # already cached by setup_env.sh
CIFAR_ARCH_ARGS=""
if [ -f cifar_arch.env ]; then source cifar_arch.env; echo "design from the pilot: $CIFAR_ARCH_ARGS"; fi
KS_DEFAULT="8 16 32"; [[ "$CIFAR_ARCH_ARGS" == *ycbcr* ]] && KS_DEFAULT="8 16 24 48 96"
GPUS=($(gpu_list)); N=${#GPUS[@]}
for s in $(seq 0 $((N - 1))); do
  SKIP_DOWNLOAD=1 GPU=${GPUS[$s]} SHARD=$s NSHARDS=$N PARALLEL=$CIFAR_PARALLEL EPOCHS_CIFAR=$EPOCHS_CIFAR \
    CIFAR_ROOT=$CIFAR_ROOT WORKERS=$WORKERS_CIFAR SEEDS="$SEEDS" CIFAR_ARCH_ARGS="$CIFAR_ARCH_ARGS" KS="${KS:-$KS_DEFAULT}" \
    bash "$KIT_DIR/run_cifar.sh" > "logs/stage1_cifar_shard$s.txt" 2>&1 &
done
wait
grep -h "FAILED" logs/stage1_cifar_shard*.txt || true
echo "finished CIFAR-10 runs: $(ls runs/cifar10_*/result.json 2>/dev/null | wc -l)"
banner "stage 1 finished"
