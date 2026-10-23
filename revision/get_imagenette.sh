#!/bin/bash
# Run ONCE on the login node (it needs internet): downloads Imagenette (fast.ai; 10 ImageNet classes,
# 9,469 train / 3,925 val images, 320-px version, about 0.33 GB) into results/data/imagenette2-320.
#   bash revision/get_imagenette.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
if [ -d "$IMAGENETTE_ROOT/train" ] && [ -d "$IMAGENETTE_ROOT/val" ]; then
  echo "Imagenette already in $IMAGENETTE_ROOT"; exit 0
fi
mkdir -p "$(dirname "$IMAGENETTE_ROOT")"
cd "$(dirname "$IMAGENETTE_ROOT")"
wget -c https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf imagenette2-320.tgz && rm imagenette2-320.tgz
echo "Imagenette: $(find "$IMAGENETTE_ROOT/train" -name '*.JPEG' | wc -l) train, $(find "$IMAGENETTE_ROOT/val" -name '*.JPEG' | wc -l) val images"
