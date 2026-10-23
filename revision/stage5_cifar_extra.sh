#!/bin/bash
# Stage 5: budgeted TACS (exactly k channels) on CIFAR-10 with the pilot's design, then the low-rate
# JPEG curve (quality 1-7) for that design and both pixel baselines, then the report.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
activate_env; enter_results
banner "stage 5: budgeted TACS on CIFAR-10 and low-rate JPEG curve"
CIFAR_ARCH_ARGS=""; CIFAR_ARCH_LABEL="y/pool"; [ -f cifar_arch.env ] && source cifar_arch.env
LIST=logs/stage5_runs.txt; : > "$LIST"
for s in $SEEDS; do for k in ${TOPK_KS:-8 16 24 48}; do
  echo "--dataset cifar10 --data_root $CIFAR_ROOT --backbone resnet18 --epochs $EPOCHS_CIFAR --workers $WORKERS_CIFAR --mode topk --k $k $CIFAR_ARCH_ARGS --seed $s" >> "$LIST"
done; done
GPUS=($(gpu_list)); N=${#GPUS[@]}
for i in $(seq 0 $((N - 1))); do
  LIST=$LIST GPU=${GPUS[$i]} SHARD=$i NSHARDS=$N PARALLEL=$CIFAR_PARALLEL bash "$KIT_DIR/run_list.sh" > "logs/stage5_shard$i.txt" 2>&1 &
done
wait
grep -h "FAILED" logs/stage5_shard*.txt || true
finished() { for d in "$@"; do [ -f "$d/result.json" ] && echo "$d"; done; return 0; }
RUNS=$(finished runs/cifar10_*_seed42 | grep -v testq || true)
CUDA_VISIBLE_DEVICES=${GPUS[0]} python "$KIT_DIR/rate_accuracy.py" --dataset cifar10 --data_root "$CIFAR_ROOT" --runs $RUNS \
  --designs "$CIFAR_ARCH_LABEL" pixel-pool pixel-cifar --qualities 1 2 3 5 7 --workers $((SLURM_CPUS - 1)) \
  --out rate_accuracy_cifar10_lowq.json
python "$KIT_DIR/make_report.py" --runs_dir runs --flops flops.json --latency latency.json --out report
banner "stage 5 finished"
