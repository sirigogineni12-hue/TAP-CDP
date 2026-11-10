#!/bin/bash
# Stage 4: latency/energy on an idle GPU, real bit-rate, final report, and an archive to send back.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
activate_env; enter_results
banner "stage 4: measurements and report"
GPUS=($(gpu_list))
CIFAR_ARCH_LABEL=""; [ -f cifar_arch.env ] && source cifar_arch.env
# Bit-rate for the repository design, the design chosen by the pilot, and both pixel baselines.
GPU=${GPUS[0]} IMAGENET_ROOT=$IMAGENET_ROOT LATENCY_IMAGES=$LATENCY_IMAGES CIFAR_ROOT=$CIFAR_ROOT \
  WORKERS=$((SLURM_CPUS - 1)) RATE_DESIGNS="y/pool $CIFAR_ARCH_LABEL pixel-pool pixel-cifar" \
  bash "$KIT_DIR/run_measurements.sh"
# Everything except the datasets and model checkpoints.
tar -czf revision_results.tar.gz --exclude='last.pt' --exclude='logs/tapcdp_4_measure_*' \
  runs report logs flops.json \
  $(ls latency.json rate_accuracy_*.json imagenet_settings.env imagenet_settings_reason.txt \
       cifar_arch.env cifar_arch_reason.txt 2>/dev/null)
echo "Send back: $RESULTS_DIR/revision_results.tar.gz"
banner "stage 4 finished"
