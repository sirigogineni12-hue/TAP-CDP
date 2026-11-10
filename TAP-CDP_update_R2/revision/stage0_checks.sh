#!/bin/bash
# Stage 0: environment and GPU check, corrected-DCT test, FLOPs.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
activate_env; enter_results
banner "stage 0: checks"
nvidia-smi || echo "nvidia-smi not available"
python "$KIT_DIR/prefetch.py" --env --gpu_test
python "$KIT_DIR/test_block_dct.py"
python "$KIT_DIR/measure_flops.py" --out flops.json
if [ -d "$IMAGENET_ROOT/train" ] && [ -d "$IMAGENET_ROOT/val" ]; then
  echo "ImageNet found at $IMAGENET_ROOT"
else
  echo "WARNING: ImageNet not found at $IMAGENET_ROOT; ImageNet runs and latency profiling will be skipped."
fi
banner "stage 0 finished"
