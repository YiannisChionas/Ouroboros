#!/bin/bash
#SBATCH --job-name=token_diversity
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.out
#SBATCH --error=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.err

set -euo pipefail

BASE_DIR="/home/it219111"
PROJECT_DIR="${BASE_DIR}/git/Ouroboros"
CONDA_ENV="test_env"

source "${BASE_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export TMPDIR=/tmp

cd "$PROJECT_DIR"
python -u src/analyze_token_diversity.py \
  --data-path ./data \
  --vit-hydra-weights "${PROJECT_DIR}/results/pretraining/vit_hydra_full/vit_hydra_final.pth" \
  --out ./results/analysis/token_diversity.json
