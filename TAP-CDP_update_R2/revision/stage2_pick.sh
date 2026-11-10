#!/bin/bash
# Stage 2: CIFAR-10 report, then choose lambda and the TACS initialization for ImageNet
# from held-out validation accuracy (rule in pick_imagenet_settings.py).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
activate_env; enter_results
banner "stage 2: CIFAR-10 report and ImageNet settings"
python "$KIT_DIR/make_report.py" --runs_dir runs --flops flops.json --out report
CIFAR_ARCH_LABEL="y/pool"; [ -f cifar_arch.env ] && source cifar_arch.env
python "$KIT_DIR/pick_imagenet_settings.py" --grouped report/grouped_results.csv --arch "$CIFAR_ARCH_LABEL" \
  --num_lambdas "$IMAGENET_NUM_LAMBDAS" --out imagenet_settings.env
banner "stage 2 finished"
