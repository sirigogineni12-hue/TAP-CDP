#!/bin/bash
#SBATCH --job-name=tapcdp_11_nette_jpeg
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          ## adjust to your cluster
#SBATCH --mem=64G                 ## <= 90G per GPU
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# Imagenette retrained on JPEG-convention coefficients (--color ycbcr_jpeg), so the models can classify
# coefficients read from the JPEG files: ResNet-18/50, 3 seeds, CI-DCT (pool, no pool), budgeted TACS,
# low-frequency and random masks at k = 24 and 48 (48 runs, about 3 h). Same recipe as job 6.
# Submit from ~/TAP-CDP:  sbatch revision/job11_imagenette_jpeg.sh
cd "$SLURM_SUBMIT_DIR"
source revision/config.sh
activate_env
export PYTHONUNBUFFERED=1
echo "Job started at: $(date) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
cd "$RESULTS_DIR"
DESIGN="--color ycbcr_jpeg --stem nopool --dct_norm"
LIST=logs/stage11_runs.txt; : > "$LIST"
for s in $SEEDS; do for bb in resnet50 resnet18; do
  C="--dataset imagenette --data_root $IMAGENETTE_ROOT --backbone $bb --epochs $EPOCHS_IMAGENETTE --lr $LR_IMAGENETTE --batch_size $BATCH_IMAGENETTE --workers $WORKERS_IMAGENETTE --seed $s"
  echo "$C --mode pixel" >> "$LIST"
  echo "$C --mode none --color ycbcr_jpeg --stem pool --dct_norm" >> "$LIST"
  echo "$C --mode none $DESIGN" >> "$LIST"
  for k in 24 48; do for m in topk lowfreq random; do echo "$C --mode $m --k $k $DESIGN" >> "$LIST"; done; done
done; done
LIST=$LIST GPU=0 SHARD=0 NSHARDS=1 PARALLEL=$IMAGENETTE_PARALLEL bash "$KIT_DIR/run_list.sh"
echo "Job ended at: $(date)"
