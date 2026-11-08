#!/bin/bash
#SBATCH --job-name=weight_drift
#SBATCH --partition=yoda
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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
python -u src/analyze_weight_drift.py \
  --results ./results/deit_base_distilled_patch16_224_cil/cifar100 \
  --approaches lwf_dual lwf_dual_v3 lwf_dual_v5 lwf_dual_v6 lwf_dual_v7 \
  --out ./results/analysis/weight_drift.json
