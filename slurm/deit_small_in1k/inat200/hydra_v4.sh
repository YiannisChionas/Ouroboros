#!/bin/bash
#SBATCH --job-name=hydra_v4_inat200_deit_small_in1k
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.out
#SBATCH --error=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.err

set -euo pipefail

BASE_DIR="/home/it219111"
PROJECT_DIR="${BASE_DIR}/git/Ouroboros"
CONDA_ENV="test_env"
CONFIG="${PROJECT_DIR}/configs/deit_small_in1k/inat200/hydra_v4.json"

source "${BASE_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "START_TASK=$START_TASK STOP_TASK=$STOP_TASK"

python -u "${PROJECT_DIR}/src/main_incremental.py" \
  --config "$CONFIG" \
  --start-at-task "$START_TASK" \
  --stop-at-task  "$STOP_TASK"
