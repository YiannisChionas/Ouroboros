#!/bin/bash
#SBATCH --job-name=joint_cifar100_vit_small_in1k
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.out
#SBATCH --error=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.err

set -eo pipefail

# Defaults for epoch-splitting (0 = run full nepochs)
START_EPOCH="${START_EPOCH:-0}"
STOP_EPOCH="${STOP_EPOCH:-0}"

BASE_DIR="/home/it219111"
PROJECT_DIR="${BASE_DIR}/git/Ouroboros"
PYTHON="${BASE_DIR}/miniconda3/envs/test_env/bin/python"
CONFIG="${PROJECT_DIR}/configs/vit_small_in1k/cifar100/joint.json"

echo "START_TASK=$START_TASK STOP_TASK=$STOP_TASK START_EPOCH=$START_EPOCH STOP_EPOCH=$STOP_EPOCH"

"$PYTHON" -u "${PROJECT_DIR}/src/main_incremental.py" \
  --config "$CONFIG" \
  --start-at-task "$START_TASK" \
  --stop-at-task  "$STOP_TASK" \
  --start-epoch   "$START_EPOCH" \
  --stop-epoch    "$STOP_EPOCH"
