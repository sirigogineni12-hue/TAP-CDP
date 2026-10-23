#!/bin/bash
#SBATCH --job-name=tapcdp_0_checks
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=94G                ## adjust to your cluster
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# GPU test, corrected-DCT test, FLOPs. A few minutes.
# Submit from ~/TAP-CDP:  sbatch revision/job0_checks.sh

cd "$SLURM_SUBMIT_DIR"
echo "Job started at: $(date) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
srun bash revision/stage0_checks.sh
echo "Job ended at: $(date)"
