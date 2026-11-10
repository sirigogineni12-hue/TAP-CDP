#!/bin/bash
#SBATCH --job-name=tapcdp_7_bitstream
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=32G                 ## <= 90G per GPU
#SBATCH --time=03:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# Feasibility check of the bitstream path (CPU only, one thread; about 30-60 min).
# Build the reader on the login node first:  bash revision/build_jpegcoef.sh
# Submit from ~/TAP-CDP:                       sbatch revision/job7_bitstream_bench.sh

cd "$SLURM_SUBMIT_DIR"
source revision/config.sh
activate_env
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "Job started at: $(date) on $(hostname)"
cd "$RESULTS_DIR"
srun python "$KIT_DIR/bitstream.py" --selftest
srun python "$KIT_DIR/bitstream_bench.py" --images "$LATENCY_IMAGES" --n 512 --qualities 50 90 \
  --ks 24 48 --repeats 5 --out bitstream_bench.json
echo "Job ended at: $(date)"
