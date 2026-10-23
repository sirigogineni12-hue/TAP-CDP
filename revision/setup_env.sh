#!/bin/bash
# Run ONCE on the login node (it needs internet): creates the conda environment, installs the
# packages, and downloads CIFAR-10 and the pretrained ResNet weights into results/data/.
#   bash revision/setup_env.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

set +u
source "$CONDA_ACTIVATE"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
fi
conda activate "$CONDA_ENV"
set -u

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
python -m pip install fvcore pandas matplotlib nvidia-ml-py pyyaml pillow

mkdir -p "$RESULTS_DIR/data" "$RESULTS_DIR/logs"
python "$KIT_DIR/prefetch.py" --env --cifar_root "$CIFAR_ROOT" --weights resnet18 resnet50
python "$KIT_DIR/test_block_dct.py"

if [ -d "$IMAGENET_ROOT/train" ] && [ -d "$IMAGENET_ROOT/val" ]; then
  echo "ImageNet found at $IMAGENET_ROOT"
else
  echo "NOTE: ImageNet not found at $IMAGENET_ROOT (needs train/ and val/ with one folder per class)."
  echo "      The ImageNet stages will be skipped unless IMAGENET_ROOT in revision/config.sh points to it."
fi
echo "Setup finished. Next: sbatch revision/job0_checks.sh"
