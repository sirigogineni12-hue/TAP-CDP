#!/bin/bash
#SBATCH --job-name=tapcdp_prep_imagenet
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=32G                 ## <= 90G per GPU
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# Extracts train/ and val/ from the Kaggle ImageNet zip into results/data/imagenet/{train,val}/<wnid>/
# and checks the counts (1000 classes; 1,281,167 train and 50,000 val images). The zip is kept; delete it
# yourself after checking the counts. Submit from ~/TAP-CDP after the download:
#   sbatch revision/job_prep_imagenet.sh

cd "$SLURM_SUBMIT_DIR"
source revision/config.sh
activate_env
set -euo pipefail
ZIP=$(ls "$RESULTS_DIR"/data/imagenet_kaggle/*.zip | head -1)
STAGE="$RESULTS_DIR/data/imagenet_kaggle/extract"
echo "Job started at: $(date) on $(hostname); zip: $ZIP"
mkdir -p "$STAGE" "$IMAGENET_ROOT"
cd "$STAGE"
if command -v unzip >/dev/null; then
  unzip -q -n "$ZIP" 'ILSVRC/Data/CLS-LOC/train/*' 'ILSVRC/Data/CLS-LOC/val/*' LOC_val_solution.csv LOC_synset_mapping.txt
else
  python - "$ZIP" <<'EOF'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
keep = [n for n in z.namelist() if n.startswith(("ILSVRC/Data/CLS-LOC/train/", "ILSVRC/Data/CLS-LOC/val/"))
        or n in ("LOC_val_solution.csv", "LOC_synset_mapping.txt")]
z.extractall(".", keep)
EOF
fi
echo "extracted at $(date)"
[ -d "$IMAGENET_ROOT/train" ] || mv ILSVRC/Data/CLS-LOC/train "$IMAGENET_ROOT/train"
python "$KIT_DIR/prep_imagenet_val.py" --solution LOC_val_solution.csv --src ILSVRC/Data/CLS-LOC/val --dst "$IMAGENET_ROOT/val"
echo "train classes: $(ls "$IMAGENET_ROOT/train" | wc -l), train images: $(find "$IMAGENET_ROOT/train" -name '*.JPEG' | wc -l)"
echo "val classes: $(ls "$IMAGENET_ROOT/val" | wc -l), val images: $(find "$IMAGENET_ROOT/val" -name '*.JPEG' | wc -l)"
du -sh "$IMAGENET_ROOT"
echo "Job ended at: $(date)"
