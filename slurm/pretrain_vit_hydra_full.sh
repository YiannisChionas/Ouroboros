#!/bin/bash
#SBATCH --job-name=pretrain_vit_hydra_full
#SBATCH --partition=yoda
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.out
#SBATCH --error=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.err

# vit-hydra pretraining, 1 GPU. Each job resumes from the last checkpoint.
# SLURM sends SIGUSR1 5 min before the time limit -> the script checkpoints and exits.
#
# Env vars (sbatch --export=ALL,...):
#   RUN_EPOCHS  epochs to finish in this job chain step (default: run until time limit / 300)
#   EVAL_EVERY  eval interval in epochs (default 10)
#   BENCHMARK=1 run GPU / dataloader throughput benchmarks before training

set -euo pipefail

BASE_DIR="/home/it219111"
PROJECT_DIR="${BASE_DIR}/git/Ouroboros"
CONDA_ENV="test_env"
DATA_DIR="/var/local/storage/${USER}/imagenet1k"
OUT_DIR="${PROJECT_DIR}/results/pretraining/vit_hydra_full"

source "${BASE_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export TMPDIR=/tmp

echo "Node: $(hostname)"
if [ ! -f "${DATA_DIR}/EXTRACTION_COMPLETE" ]; then
  echo "ERROR: ImageNet not found at ${DATA_DIR} (run slurm/extract_imagenet.sh first)"
  exit 1
fi

EXTRA=()
[ -n "${RUN_EPOCHS:-}" ] && EXTRA+=(--run-epochs "$RUN_EPOCHS")
[ "${BENCHMARK:-0}" = "1" ] && EXTRA+=(--benchmark)

# exec: python replaces the batch shell, so it receives SIGUSR1 directly
exec python -u "${PROJECT_DIR}/src/pretrain_vit_hydra_full.py" \
  --data          "$DATA_DIR" \
  --output-dir    "$OUT_DIR" \
  --epochs        300 \
  --batch-size    1024 \
  --micro-batch   128 \
  --num-workers   3 \
  --T             1.0 \
  --eval-every    "${EVAL_EVERY:-10}" \
  --resume \
  ${EXTRA[@]+"${EXTRA[@]}"}
