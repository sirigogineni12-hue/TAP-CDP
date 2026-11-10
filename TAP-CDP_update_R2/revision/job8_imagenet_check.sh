#!/bin/bash
#SBATCH --job-name=tapcdp_8_imagenet_check
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=64G                 ## <= 90G per GPU
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# Checks the ImageNet folders: torchvision's pretrained ResNet-50 (IMAGENET1K_V1) evaluated on val/ must give
# about 76.1% top-1 (its published accuracy); this run is also the pixel baseline of the paper.
# Run after: python revision/get_imagenet_val_hf.py ...   Submit from ~/TAP-CDP: sbatch revision/job8_imagenet_check.sh

cd "$SLURM_SUBMIT_DIR"
source revision/config.sh
activate_env
export PYTHONUNBUFFERED=1
echo "Job started at: $(date) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
cd "$RESULTS_DIR"
srun python "$KIT_DIR/train_eval.py" --dataset imagenet --data_root "$IMAGENET_ROOT" --backbone resnet50 \
  --mode pixel --epochs 0 --seed 42 --batch_size 256 --workers 7
python - <<'PY'
import json
r = json.load(open("runs/imagenet_resnet50_pixel_e0_seed42/result.json"))
print(f"ImageNet val top-1 of torchvision ResNet-50: {r['test_acc']:.2f}% (published 76.13%)")
PY
echo "Job ended at: $(date)"
