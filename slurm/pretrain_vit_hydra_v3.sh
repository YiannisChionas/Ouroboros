#!/bin/bash
#SBATCH --job-name=pretrain_vit_hydra_v3
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
RESULTS_DIR="${PROJECT_DIR}/results/pretraining"

source "${BASE_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$RESULTS_DIR"

python -u "${PROJECT_DIR}/src/pretrain_vit_hydra_v3.py" \
  --data        /opt/storage/datasets/imagenet1k \
  --epochs      10 \
  --run-epochs  1 \
  --lr          1e-3 \
  --batch-size  512 \
  --num-workers 2 \
  --T           3.0 \
  --output      "${RESULTS_DIR}/mlp_weights_v3.pth" \
  --resume
