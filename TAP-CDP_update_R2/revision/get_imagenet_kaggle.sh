#!/bin/bash
# Downloads ImageNet-1k (ILSVRC-2012) from Kaggle on the LOGIN node (needs internet). About 160 GB.
# Before running: accept the rules of https://www.kaggle.com/competitions/imagenet-object-localization-challenge
# with the Kaggle account that owns ~/.kaggle/kaggle.json, and check free space (about 330 GB needed at the
# peak: zip + extracted train/val). Run detached so it survives logout:
#   nohup bash revision/get_imagenet_kaggle.sh > results/logs/get_imagenet.log 2>&1 &
# Then extract with:  sbatch revision/job_prep_imagenet.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
activate_env
ZIPDIR="$RESULTS_DIR/data/imagenet_kaggle"
mkdir -p "$ZIPDIR"
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "missing $HOME/.kaggle/kaggle.json"; exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json"
python -m pip install --quiet kaggle
df -h "$ZIPDIR"
cd "$ZIPDIR"
date
kaggle competitions download -c imagenet-object-localization-challenge
ls -la "$ZIPDIR"
date
echo "Download finished. Next: sbatch revision/job_prep_imagenet.sh"
