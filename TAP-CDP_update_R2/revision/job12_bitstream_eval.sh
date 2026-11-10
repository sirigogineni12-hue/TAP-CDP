#!/bin/bash
#SBATCH --job-name=tapcdp_12_bitstream_eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=64G                 ## <= 90G per GPU
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# After jobs 10 and 11: accuracy through the pixel path and the bitstream path (coefficients read from the
# JPEG files) on the ImageNet and Imagenette test sets, decoder timing on ImageNet validation images,
# ResNet-50/18 latency on GPU, and the report.
# Submit from ~/TAP-CDP:  sbatch revision/job12_bitstream_eval.sh
cd "$SLURM_SUBMIT_DIR"
source revision/config.sh
activate_env
export PYTHONUNBUFFERED=1
echo "Job started at: $(date) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
cd "$RESULTS_DIR"
finished() { for d in "$@"; do [ -f "$d/result.json" ] && echo "$d"; done; return 0; }
IN_RUNS=$(finished runs/imagenet_resnet50_*ycbcr_jpeg*_seed42 runs/imagenet_resnet50_pixel_e0_seed42)
NE_RUNS=$(finished runs/imagenette_resnet*_*ycbcr_jpeg*_seed42 runs/imagenette_resnet*_pixel_e20_seed42)
srun python "$KIT_DIR/eval_bitstream.py" --dataset imagenet --data_root "$IMAGENET_ROOT" --runs $IN_RUNS --qualities 50 90 --workers 7
srun python "$KIT_DIR/eval_bitstream.py" --dataset imagenette --data_root "$IMAGENETTE_ROOT" --runs $NE_RUNS --qualities 50 90 --workers 7
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 srun python "$KIT_DIR/bitstream_bench.py" --images "$IMAGENET_ROOT/val" --n 512 \
  --qualities 50 90 --ks 24 48 --repeats 5 --out bitstream_bench_imagenet.json
python "$KIT_DIR/measure_flops.py" --out flops.json
python "$KIT_DIR/make_report.py" --runs_dir runs --flops flops.json --latency latency.json --out report
echo "Job ended at: $(date)"
