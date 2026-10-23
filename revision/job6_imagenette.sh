#!/bin/bash
#SBATCH --job-name=tapcdp_6_imagenette
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=94G                 ## adjust to your cluster
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# Imagenette, ResNet-50 and ResNet-18, 3 seeds (54 runs, about 3-4 h on 1 GPU) + bit-rate + latency.
# Submit from ~/TAP-CDP:  sbatch revision/job6_imagenette.sh

cd "$SLURM_SUBMIT_DIR"
export CIFAR_PARALLEL=${CIFAR_PARALLEL:-4}     # one GPU: 4 runs at a time (CPU-bound)
echo "Job started at: $(date) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
srun bash revision/stage6_imagenette.sh
echo "Job ended at: $(date)"
