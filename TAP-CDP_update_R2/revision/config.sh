# Settings read by every script. Edit here once. Values can also be overridden from the environment.

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # .../TAP-CDP/revision
REPO_DIR="$(dirname "$KIT_DIR")"                              # .../TAP-CDP
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"               # every output goes here

# ---- Conda environment (created by setup_env.sh) ----
CONDA_ACTIVATE="${CONDA_ACTIVATE:-$HOME/miniconda3/bin/activate}"
CONDA_ENV="${CONDA_ENV-base}"                                 # base env (torch 2.14); "tapcdp" = env made by setup_env.sh
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"   # CUDA 12.8 builds support Blackwell GPUs

# ---- Data (downloaded on the login node by setup_env.sh; compute nodes need no internet) ----
SHARED=/janaki/common/Datasets                                # shared datasets on the cluster NAS (read-only use)
if [ -d "$SHARED/cifar10/cifar-10-batches-py" ]; then CIFAR_DEFAULT="$SHARED/cifar10"; else CIFAR_DEFAULT="$RESULTS_DIR/data/cifar10"; fi
if [ -d "$SHARED/COCO/val2017" ]; then COCO_DEFAULT="$SHARED/COCO/val2017"; else COCO_DEFAULT="$RESULTS_DIR/data/coco/val2017"; fi
CIFAR_ROOT="${CIFAR_ROOT:-$CIFAR_DEFAULT}"
IMAGENET_ROOT="${IMAGENET_ROOT:-$RESULTS_DIR/data/imagenet}"  # needs train/ and val/ (one folder per class); missing = ImageNet stages are skipped
JANAKI_PRIVATE="/janaki/backup$HOME"                          # private 500 GB NAS folder (mirrors the home path)
JANAKI_SHARED="/janaki/common/$USER"                          # own folder in the shared pool (used until the private one exists)
if [ -d "$JANAKI_PRIVATE" ]; then FT_DEFAULT="$JANAKI_PRIVATE/tapcdp_data/imagenet256"
elif [ -d "$JANAKI_SHARED" ]; then FT_DEFAULT="$JANAKI_SHARED/tapcdp_data/imagenet256"
else FT_DEFAULT="$RESULTS_DIR/data/imagenet256"; fi
IMAGENET_FT_ROOT="${IMAGENET_FT_ROOT:-$FT_DEFAULT}"            # 256-px ImageNet training copy (~50 GB, job 9): on Janaki, not in the 200 GB home
IMAGENETTE_ROOT="${IMAGENETTE_ROOT:-$RESULTS_DIR/data/imagenette2-320}"   # train/ and val/ (get_imagenette.sh)
LATENCY_IMAGES="${LATENCY_IMAGES:-$COCO_DEFAULT}"   # photos for latency timing if ImageNet is missing (get_coco.sh)
export TORCH_HOME="${TORCH_HOME:-$RESULTS_DIR/data/torch_cache}"   # pretrained ResNet weights

# ---- Hardware (the SLURM settings themselves are in the #SBATCH lines of job*.sh) ----
SLURM_CPUS="${SLURM_CPUS:-8}"                 # CPUs per job (max on the cluster partition)
NUM_GPUS="${NUM_GPUS:-2}"

# ---- Experiment size ----
SEEDS="${SEEDS:-42 123 456}"
PILOT_SEEDS="${PILOT_SEEDS:-42 123}"              # seeds of the CIFAR-10 architecture pilot (job1p)
EPOCHS_CIFAR="${EPOCHS_CIFAR:-100}"
EPOCHS_IMAGENET="${EPOCHS_IMAGENET:-10}"       # fine-tuning from pretrained weights; 8 CPUs make ImageNet decoding slow
IMAGENET_NUM_LAMBDAS="${IMAGENET_NUM_LAMBDAS:-1}"   # 1 = only the lambda selected on CIFAR-10; 2 = also the next smaller one
CIFAR_PARALLEL="${CIFAR_PARALLEL:-2}"          # concurrent CIFAR-10 runs per GPU
WORKERS_CIFAR="${WORKERS_CIFAR:-1}"            # data-loader workers per CIFAR-10 run (2 GPUs x 2 runs x 2 processes = 8 CPUs)
WORKERS_IMAGENET="${WORKERS_IMAGENET:-3}"      # per ImageNet run (2 runs x 4 processes = 8 CPUs)
# Imagenette (job 6): fine-tuning from ImageNet weights, same recipe for every model, fixed before any run.
EPOCHS_IMAGENETTE="${EPOCHS_IMAGENETTE:-20}"
LR_IMAGENETTE="${LR_IMAGENETTE:-0.01}"
BATCH_IMAGENETTE="${BATCH_IMAGENETTE:-64}"
IMAGENETTE_PARALLEL="${IMAGENETTE_PARALLEL:-2}"   # concurrent runs per GPU
WORKERS_IMAGENETTE="${WORKERS_IMAGENETTE:-3}"     # per run (2 runs x 4 processes = 8 CPUs)

# ---- Helpers ----
activate_env() {
  if [ -n "$CONDA_ENV" ]; then
    set +u
    source "$CONDA_ACTIVATE" "$CONDA_ENV"
    set -u
  fi
}

gpu_list() {   # GPUs allocated by SLURM, else 0 .. NUM_GPUS-1
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "${CUDA_VISIBLE_DEVICES//,/ }"; else seq 0 $((NUM_GPUS - 1)) | tr '\n' ' '; fi
}

banner() { echo "==== $(date '+%Y-%m-%d %H:%M:%S') | $* ===="; }

enter_results() { mkdir -p "$RESULTS_DIR/logs"; cd "$RESULTS_DIR"; }
