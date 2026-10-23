#!/bin/bash
#SBATCH --job-name=tapcdp_5_cifar_extra
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=94G                 ## adjust to your cluster
#SBATCH --time=03:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# Budgeted TACS on CIFAR-10 (12 runs, about 30 min) + low-rate JPEG curve (about 30 min).
# Submit from ~/TAP-CDP:  sbatch revision/job5_cifar_extra.sh

cd "$SLURM_SUBMIT_DIR"
export CIFAR_PARALLEL=${CIFAR_PARALLEL:-4}     # one GPU: 4 runs at a time (CPU-bound)
echo "Job started at: $(date) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
srun bash revision/stage5_cifar_extra.sh
echo "Job ended at: $(date)"
