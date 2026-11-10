#!/bin/bash
#SBATCH --job-name=tapcdp_9_prep256
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=32G                 ## <= 90G per GPU
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# One-time ImageNet training copy with shorter side 256 (about 30 GB, 30-90 min). CPU only.
# Submit from ~/TAP-CDP:  sbatch revision/job9_prep_imagenet256.sh
cd "$SLURM_SUBMIT_DIR"
source revision/config.sh
activate_env
export PYTHONUNBUFFERED=1
echo "Job started at: $(date) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
cd "$RESULTS_DIR"
echo "ImageNet copy goes to: $IMAGENET_FT_ROOT"
case "$IMAGENET_FT_ROOT" in
  "$HOME"/*) echo "Refusing to write the ~50 GB copy into the 200 GB home quota. JANAKI_PRIVATE=$JANAKI_PRIVATE is not visible here."; exit 1;;
esac
set -e
srun python "$KIT_DIR/prep_imagenet_resized.py" --src "$IMAGENET_ROOT" --out "$IMAGENET_FT_ROOT" --workers 8
echo "Job ended at: $(date)"
