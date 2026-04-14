#!/bin/bash
#SBATCH --job-name=finetuning_cifar100_vit_small_in1k
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/it219111/experiments/FACILCUSTOM/logs/%x/%x-%j.out
#SBATCH --error=/home/it219111/experiments/FACILCUSTOM/logs/%x/%x-%j.err

set -euo pipefail

BASE_DIR="/home/it219111"
PROJECT_DIR="${BASE_DIR}/git/FACILCUSTOM"
CONFIG="${PROJECT_DIR}/configs/vit_small_in1k/cifar100/finetuning.json"

source "${BASE_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate test_env

echo "START_TASK=$START_TASK STOP_TASK=$STOP_TASK"

python -u "${PROJECT_DIR}/src/main_incremental.py" \
  --config "$CONFIG" \
  --start-at-task "$START_TASK" \
  --stop-at-task  "$STOP_TASK"
