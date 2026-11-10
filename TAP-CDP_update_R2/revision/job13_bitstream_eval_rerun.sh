#!/bin/bash
#SBATCH --job-name=tapcdp_13_bitstream_rerun
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=64G                 ## <= 90G per GPU
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# Re-run of the bitstream evaluation of job 12 with the fixed script (results saved after every model; the
# ImageNet pixel baseline uses the published torchvision weights). About 30-45 min.
# Submit from ~/TAP-CDP:  sbatch revision/job13_bitstream_eval_rerun.sh

cd "$SLURM_SUBMIT_DIR"
source revision/config.sh
activate_env
export PYTHONUNBUFFERED=1
echo "Job started at: $(date) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
cd "$RESULTS_DIR"
finished() { for d in "$@"; do [ -f "$d/result.json" ] && echo "$d"; done; return 0; }
IN_RUNS=$(finished runs/imagenet_resnet50_pixel_e0_seed42 runs/imagenet_resnet50_*ycbcr_jpeg*_seed42)
NE_RUNS=$(finished runs/imagenette_resnet*_*ycbcr_jpeg*_seed42)
srun python "$KIT_DIR/eval_bitstream.py" --dataset imagenet --data_root "$IMAGENET_ROOT" --runs $IN_RUNS --qualities 10 30 50 70 90 --workers 7
srun python "$KIT_DIR/eval_bitstream.py" --dataset imagenette --data_root "$IMAGENETTE_ROOT" --runs $NE_RUNS --qualities 50 90 --workers 7
python "$KIT_DIR/make_report.py" --runs_dir runs --flops flops.json --latency latency.json --out report
echo "Job ended at: $(date)"
