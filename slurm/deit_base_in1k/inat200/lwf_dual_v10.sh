#!/bin/bash
#SBATCH --job-name=lwf_dual_v10_inat200_deit_base_in1k
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.out
#SBATCH --error=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.err

set -euo pipefail

BASE_DIR="/home/it219111"
PROJECT_DIR="${BASE_DIR}/git/Ouroboros"
CONDA_ENV="test_env"
CONFIG="${PROJECT_DIR}/configs/deit_base_in1k/inat200/lwf_dual_v10.json"

# Copy inat200 to the node's fast local disk once (cluster guide); the lock keeps parallel jobs from copying at the same time.
SRC_DATA="${BASE_DIR}/git/data/inat200"
LOCAL_DATA="/var/local/storage/${USER}/inat200"
mkdir -p "/var/local/storage/${USER}"
flock "/var/local/storage/${USER}/inat200.lock" bash -c "
  if [ ! -f '${LOCAL_DATA}/.complete' ]; then
    mkdir -p '${LOCAL_DATA}' && rsync -a '${SRC_DATA}/' '${LOCAL_DATA}/' && touch '${LOCAL_DATA}/.complete'
  fi"

source "${BASE_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "START_TASK=$START_TASK STOP_TASK=$STOP_TASK"

python -u "${PROJECT_DIR}/src/main_incremental.py" \
  --config "$CONFIG" \
  --start-at-task "$START_TASK" \
  --stop-at-task  "$STOP_TASK"
