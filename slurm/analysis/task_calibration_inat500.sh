#!/bin/bash
#SBATCH --job-name=task_calibration_inat500
#SBATCH --partition=obiwan
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.out
#SBATCH --error=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.err

# Eval only, on one MIG slice: task calibration of the final checkpoints of ViT-B in21k runs on iNat-500 (50 tasks)
# (lwf_v2 lamb 6 vs 10).
# Test images are read directly from the source on /opt/storage (no copy to the node's disk, which counts in the quota).

set -euo pipefail

BASE_DIR="/home/it219111"
PROJECT_DIR="${BASE_DIR}/git/Ouroboros"
CONDA_ENV="test_env"
RESULTS="${PROJECT_DIR}/results/vit_base_patch16_224.orig_in21k/inat500"
OUT="/opt/storage/users/${USER}/analysis/task_calibration/inat500"

source "${BASE_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export TMPDIR=/tmp
echo "Node: $(hostname)"

for run in lwf_v2_l6 lwf_v2_l10; do
  echo "=== ${run}"
  python -u "${PROJECT_DIR}/src/analyze_task_calibration.py" \
    --config "${PROJECT_DIR}/configs/vit_base_in21k/inat500/${run}.json" \
    --ckpt "${RESULTS}/${run}/10/models/task49.ckpt" \
    --out "${OUT}/${run}" \
    --data-path "/opt/storage/users/${USER}/data/inat500"
done
