#!/bin/bash
#SBATCH --job-name=tapcdp_3_imagenet
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=188G                ## adjust to your cluster
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# ImageNet runs on both GPUs. Submit after job 2. If it hits the 24 h limit, submit it again: it continues where it stopped.
# Submit from ~/TAP-CDP:  sbatch revision/job3_imagenet.sh

cd "$SLURM_SUBMIT_DIR"
echo "Job started at: $(date) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
srun bash revision/stage3_imagenet.sh
echo "Job ended at: $(date)"
