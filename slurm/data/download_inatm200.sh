#!/bin/bash
#SBATCH --job-name=download_inatm200
#SBATCH --partition=obiwan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --output=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.out
#SBATCH --error=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.err

# CPU-only job (no --gres): download iNatM-200 from the Hugging Face Hub
# (https://huggingface.co/datasets/YiannisChionas/iNatM-200) to /home/it219111/git/data/inatm200.
# Resumable: files already downloaded are skipped, so re-submit if it hits the time limit or a rate limit.
# Training jobs copy this folder to the node's local disk before reading it.

set -euo pipefail

BASE_DIR="/home/it219111"
CONDA_ENV="test_env"
DST="${BASE_DIR}/git/data/inatm200"

source "${BASE_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Node: $(hostname)"
mkdir -p "$DST"
hf download YiannisChionas/iNatM-200 --repo-type dataset --local-dir "$DST" \
  --exclude "__pycache__/*" --exclude "_hf_upload.log" --max-workers 4

# Check: every image listed in train.txt / test.txt exists on disk
n_list=$(cat "$DST/train.txt" "$DST/test.txt" | wc -l)
n_missing=$(cat "$DST/train.txt" "$DST/test.txt" | awk -v d="$DST" '{print d"/"$1}' | xargs -d '\n' ls 2>&1 >/dev/null | wc -l)
echo "listed=${n_list} missing=${n_missing} classes=$(wc -l < "$DST/classes.txt")"
if [ "$n_missing" -eq 0 ]; then echo "DOWNLOAD COMPLETE"; else echo "DOWNLOAD INCOMPLETE"; exit 1; fi
