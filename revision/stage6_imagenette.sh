#!/bin/bash
# Stage 6: Imagenette (10 ImageNet classes, 224x224) with ResNet-50 and ResNet-18, three seeds:
# pixel baseline, CI-DCT (YCbCr, normalized; max-pool kept and removed), and at k = 24 and 48 channels
# budgeted TACS, the luma-weighted low-frequency mask (Xu et al.'s static selection) and a random mask.
# Then real JPEG bit-rate (seed 42), ResNet-18 latency/energy, the report and the archive.
# Recipe (fixed before any run, identical for every model): ImageNet-pretrained weights, SGD momentum 0.9,
# lr 0.01 cosine, batch 64, 20 epochs, weight decay 1e-4; 50 training images per class held out for validation.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
activate_env; enter_results
banner "stage 6: Imagenette"
if [ ! -d "$IMAGENETTE_ROOT/train" ]; then echo "Imagenette not found at $IMAGENETTE_ROOT: run bash revision/get_imagenette.sh"; exit 1; fi
python "$KIT_DIR/prefetch.py" --weights resnet18 resnet50   # cached by setup_env.sh
DESIGN="--color ycbcr --stem nopool --dct_norm"   # max-pool removed (pilot); stride-1 layer2 dropped: 3.4 GMACs at 224
LIST=logs/stage6_runs.txt; : > "$LIST"
for s in $SEEDS; do for bb in resnet50 resnet18; do
  C="--dataset imagenette --data_root $IMAGENETTE_ROOT --backbone $bb --epochs $EPOCHS_IMAGENETTE --lr $LR_IMAGENETTE --batch_size $BATCH_IMAGENETTE --workers $WORKERS_IMAGENETTE --seed $s"
  echo "$C --mode pixel" >> "$LIST"
  echo "$C --mode none --color ycbcr --stem pool --dct_norm" >> "$LIST"
  echo "$C --mode none $DESIGN" >> "$LIST"
  for k in 24 48; do for m in topk lowfreq random; do echo "$C --mode $m --k $k $DESIGN" >> "$LIST"; done; done
done; done
GPUS=($(gpu_list)); N=${#GPUS[@]}
for i in $(seq 0 $((N - 1))); do
  LIST=$LIST GPU=${GPUS[$i]} SHARD=$i NSHARDS=$N PARALLEL=$IMAGENETTE_PARALLEL bash "$KIT_DIR/run_list.sh" > "logs/stage6_shard$i.txt" 2>&1 &
done
wait
grep -h "FAILED" logs/stage6_shard*.txt || true
echo "finished Imagenette runs: $(ls runs/imagenette_*/result.json 2>/dev/null | wc -l) of $(wc -l < "$LIST")"
finished() { for d in "$@"; do [ -f "$d/result.json" ] && echo "$d"; done; return 0; }
RUNS=$(finished runs/imagenette_*_seed42 | grep -v testq || true)
[ -n "$RUNS" ] && CUDA_VISIBLE_DEVICES=${GPUS[0]} python "$KIT_DIR/rate_accuracy.py" --dataset imagenette \
  --data_root "$IMAGENETTE_ROOT" --runs $RUNS --qualities 2 5 10 30 50 70 90 --workers $((SLURM_CPUS - 1)) \
  --out rate_accuracy_imagenette.json
if [ -d "$LATENCY_IMAGES" ]; then
  CUDA_VISIBLE_DEVICES=${GPUS[0]} python "$KIT_DIR/profile_latency.py" --images "$LATENCY_IMAGES" --n_images 512 \
    --quality 90 --backbone resnet18 --batches 1 64 --dtype fp32 --out latency_resnet18.json
fi
python "$KIT_DIR/measure_flops.py" --out flops.json
python "$KIT_DIR/make_report.py" --runs_dir runs --flops flops.json --latency latency.json --out report
tar -czf revision_results.tar.gz --exclude='last.pt' runs report logs flops.json \
  $(ls latency*.json rate_accuracy_*.json cifar_arch.env cifar_arch_reason.txt imagenet_settings* 2>/dev/null)
banner "stage 6 finished"
