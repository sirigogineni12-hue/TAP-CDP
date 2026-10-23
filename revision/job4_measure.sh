#!/bin/bash
#SBATCH --job-name=tapcdp_4_measure
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=94G                ## adjust to your cluster
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# Latency/energy, real bit-rate, final report, results archive. Submit after job 3 has finished.
# Submit from ~/TAP-CDP:  sbatch revision/job4_measure.sh

cd "$SLURM_SUBMIT_DIR"
echo "Job started at: $(date) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
srun bash revision/stage4_measure.sh
echo "Job ended at: $(date)"
