#!/bin/bash
#SBATCH --job-name=tapcdp_10_imagenet
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=90G                 ## <= 90G per GPU
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# ImageNet-1k, ResNet-50, JPEG-convention coefficients, seed 42, 16 epochs of fine-tuning from ImageNet
# weights (SGD, lr 0.05 cosine, batch 256), on the 256-pixel training copy (job 9). Six runs split over two
# jobs; submit both after job 9 (SHARD=0 and SHARD=1). Resubmit the same job if it hits 24 h: finished runs
# are skipped and interrupted runs resume.
#   sbatch --export=ALL,SHARD=0 revision/job10_imagenet_train.sh
#   sbatch --export=ALL,SHARD=1 revision/job10_imagenet_train.sh
cd "$SLURM_SUBMIT_DIR"
source revision/config.sh
activate_env
export PYTHONUNBUFFERED=1
echo "Job started at: $(date) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
cd "$RESULTS_DIR"
ROOT="$IMAGENET_FT_ROOT"
DESIGN="--color ycbcr_jpeg --stem nopool --dct_norm"
C="--dataset imagenet --data_root $ROOT --backbone resnet50 --epochs ${EPOCHS_IMAGENET_FT:-16} --lr 0.05 --batch_size 256 --workers 7 --seed 42"
LIST=logs/stage10_runs.txt; : > "$LIST"
echo "$C --mode none $DESIGN" >> "$LIST"
echo "$C --mode none --color ycbcr_jpeg --stem pool --dct_norm" >> "$LIST"
echo "$C --mode lowfreq --k 24 $DESIGN" >> "$LIST"
echo "$C --mode lowfreq --k 48 $DESIGN" >> "$LIST"
echo "$C --mode random --k 24 $DESIGN" >> "$LIST"
echo "$C --mode topk --k 24 $DESIGN" >> "$LIST"
LIST=$LIST GPU=0 SHARD=${SHARD:-0} NSHARDS=2 PARALLEL=1 bash "$KIT_DIR/run_list.sh"
echo "Job ended at: $(date)"
