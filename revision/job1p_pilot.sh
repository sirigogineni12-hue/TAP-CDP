#!/bin/bash
#SBATCH --job-name=tapcdp_1p_pilot
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=188G                ## adjust to your cluster
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# CIFAR-10 architecture pilot (about 1-1.5 h): 12 CI-DCT designs x 2 seeds + CIFAR pixel baseline.
# Submit from ~/TAP-CDP:  sbatch revision/job1p_pilot.sh

cd "$SLURM_SUBMIT_DIR"
echo "Job started at: $(date) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
srun bash revision/stage1p_pilot.sh
echo "Job ended at: $(date)"
