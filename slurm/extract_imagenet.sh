#!/bin/bash
#SBATCH --job-name=extract_imagenet
#SBATCH --partition=yoda
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.out
#SBATCH --error=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.err

# CPU-only job (no --gres): one-time extraction of ImageNet-1k parquet -> ImageFolder
# on the node's fast local storage (cluster guide: jobs read data from /var/local/storage).
# Resumable: re-submit if it hits the time limit, finished parquet files are skipped.

set -euo pipefail

BASE_DIR="/home/it219111"
PROJECT_DIR="${BASE_DIR}/git/Ouroboros"
CONDA_ENV="test_env"
SRC="/opt/storage/datasets/imagenet1k/data"
DST="/var/local/storage/${USER}/imagenet1k"

source "${BASE_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export TMPDIR=/tmp

echo "Node: $(hostname)"
mkdir -p "$DST"
df -h "/var/local/storage/${USER}"

if [ -f "${DST}/EXTRACTION_COMPLETE" ]; then
  echo "Already extracted: ${DST}"
  exit 0
fi

python -u "${PROJECT_DIR}/src/extract_imagenet_parquet.py" \
  --src     "$SRC" \
  --dst     "$DST" \
  --workers 4

df -h "/var/local/storage/${USER}"
