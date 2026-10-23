#!/bin/bash
# Train every run listed in $LIST (one line of train_eval.py arguments per run) on one GPU, $PARALLEL at a time.
# Runs are split over GPUs by line number (SHARD of NSHARDS). Finished runs are skipped, interrupted ones resume.
#   LIST=logs/stage5_runs.txt GPU=0 SHARD=0 NSHARDS=1 PARALLEL=4 bash revision/run_list.sh
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=${GPU:-0}
SHARD=${SHARD:-0}
NSHARDS=${NSHARDS:-1}
PARALLEL=${PARALLEL:-2}
EXTRA=${EXTRA_ARGS:-}            # extra train_eval.py arguments (smoke tests only)
mkdir -p logs runs
JOBS=()
while IFS= read -r line; do [ -n "$line" ] && JOBS+=("$line"); done < "$LIST"
MINE=()
for i in "${!JOBS[@]}"; do [ $((i % NSHARDS)) -eq "$SHARD" ] && MINE+=("${JOBS[$i]}"); done
echo "$(basename "$LIST"): ${#MINE[@]} of ${#JOBS[@]} runs (shard $SHARD/$NSHARDS), $PARALLEL at a time on GPU $GPU"
run_one() {
  local name; name=$(echo "$1" | sed 's|--data_root [^ ]*||' | tr -s ' -' '_')
  if CUDA_VISIBLE_DEVICES=$GPU python "$KIT/train_eval.py" $1 $EXTRA > "logs/run_${name}.log" 2>&1; then
    echo "done: $1"
  else
    echo "FAILED: $1 (see logs/run_${name}.log)"
  fi
}
for j in "${MINE[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 2; done
  run_one "$j" &
done
wait
