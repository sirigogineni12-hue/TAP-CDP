#!/bin/bash
# Stage 3: ImageNet runs, one shard per GPU, with the settings chosen in stage 2.
# Resumable: finished runs are skipped and interrupted runs continue from their last epoch.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
activate_env; enter_results
if [ ! -d "$IMAGENET_ROOT/train" ] || [ ! -d "$IMAGENET_ROOT/val" ]; then
  echo "ImageNet not found at $IMAGENET_ROOT: skipping stage 3"; exit 0
fi
banner "stage 3: ImageNet"
[ -f imagenet_settings.env ] && source imagenet_settings.env
echo "TACS_INIT=${TACS_INIT:-0} LAMBDAS=${LAMBDAS:-0.1}"
python "$KIT_DIR/prefetch.py" --weights resnet50   # already cached by setup_env.sh
GPUS=($(gpu_list)); N=${#GPUS[@]}
for s in $(seq 0 $((N - 1))); do
  SKIP_DOWNLOAD=1 GPU=${GPUS[$s]} SHARD=$s NSHARDS=$N EPOCHS_IMAGENET=$EPOCHS_IMAGENET IMAGENET_ROOT=$IMAGENET_ROOT \
    WORKERS=$WORKERS_IMAGENET SEEDS="$SEEDS" TACS_INIT=${TACS_INIT:-0} LAMBDAS="${LAMBDAS:-0.1}" \
    bash "$KIT_DIR/run_imagenet.sh" >> "logs/stage3_imagenet_shard$s.txt" 2>&1 &
done
wait
grep -h "FAILED" logs/stage3_imagenet_shard*.txt || true
echo "finished ImageNet runs: $(ls runs/imagenet_*/result.json 2>/dev/null | wc -l)"
banner "stage 3 finished"
