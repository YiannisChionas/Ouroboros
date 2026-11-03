#!/bin/bash
#SBATCH --job-name=token_diversity_mlp
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
python -u src/analyze_token_diversity_mlp.py \
  --data-path ./data \
  --mlp-ckpt "${PROJECT_DIR}/results/pretraining/mlp_weights_v2.pth.ckpt" \
  --out ./results/analysis/token_diversity_mlp.json
