#!/bin/bash
#SBATCH --job-name=joint_food101_vit_small_in1k
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=/home/it219111/experiments/FACILCUSTOM/logs/%x/%x-%j.out
#SBATCH --error=/home/it219111/experiments/FACILCUSTOM/logs/%x/%x-%j.err

set -eo pipefail

# Defaults for epoch-splitting (0 = run full nepochs)
START_EPOCH="${START_EPOCH:-0}"
STOP_EPOCH="${STOP_EPOCH:-0}"

BASE_DIR="/home/it219111"
PROJECT_DIR="${BASE_DIR}/git/FACILCUSTOM"
CONDA_ENV="test_env"
CONFIG="${PROJECT_DIR}/configs/vit_small_in1k/food101/joint.json"

source "${BASE_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "START_TASK=$START_TASK STOP_TASK=$STOP_TASK START_EPOCH=$START_EPOCH STOP_EPOCH=$STOP_EPOCH"

python -u "${PROJECT_DIR}/src/main_incremental.py" \
  --config "$CONFIG" \
  --start-at-task "$START_TASK" \
  --stop-at-task  "$STOP_TASK" \
  --start-epoch   "$START_EPOCH" \
  --stop-epoch    "$STOP_EPOCH"
