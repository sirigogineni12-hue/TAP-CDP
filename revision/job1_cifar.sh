#!/bin/bash
#SBATCH --job-name=tapcdp_1_cifar
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

# All CIFAR-10 runs (3 seeds) on both GPUs. Submit after job 0 has finished.
# Submit from ~/TAP-CDP:  sbatch revision/job1_cifar.sh

cd "$SLURM_SUBMIT_DIR"
echo "Job started at: $(date) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
srun bash revision/stage1_cifar.sh
echo "Job ended at: $(date)"
