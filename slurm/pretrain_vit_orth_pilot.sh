#!/bin/bash
#SBATCH --job-name=pretrain_vit_orth_pilot
#SBATCH --partition=yoda
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.out
#SBATCH --error=/home/it219111/experiments/Ouroboros/logs/%x/%x-%j.err

# vit-orth pilot: ViT-S/16 (cls + dist token, both on GT) from scratch, 30 epochs (own cosine schedule),
# on the 200 mammal classes of iNat-500. SimpleCIL on classes 200-499 at the end.
# Submit with --job-name=pretrain_vit_orth_pilot_<RUN> so each run has its own log dir.
#
# Env vars (sbatch --export=ALL,...):
#   RUN       A = no constraint · B = fixed penalty · C = augmented Lagrangian
#   VAR_MIN   within-class std floor for B / C (from run A)
#   BENCHMARK=1 dataloader throughput before training
# Each job resumes from the last checkpoint; SIGUSR1 5 min before the limit -> checkpoint and exit.

set -euo pipefail

BASE_DIR="/home/it219111"
PROJECT_DIR="${BASE_DIR}/git/Ouroboros"
CONDA_ENV="test_env"
OUT_DIR="${PROJECT_DIR}/results/pretraining/vit_orth_pilot_${RUN}"

case "$RUN" in
  A) MODE_ARGS=(--mode none) ;;
  B) MODE_ARGS=(--mode penalty --w-cc 1.0 --w-var 1.0 --var-min "$VAR_MIN") ;;
  C) MODE_ARGS=(--mode lagrangian --var-min "$VAR_MIN") ;;
  *) echo "unknown RUN=$RUN"; exit 1 ;;
esac

# Copy inat500 to the node's fast local disk once (cluster guide); the lock keeps parallel jobs from copying at the same time.
SRC_DATA="/opt/storage/users/${USER}/data/inat500"
LOCAL_DATA="/var/local/storage/${USER}/inat500"
mkdir -p "/var/local/storage/${USER}"
flock "/var/local/storage/${USER}/inat500.lock" bash -c "
  if [ ! -f '${LOCAL_DATA}/.complete' ]; then
    mkdir -p '${LOCAL_DATA}' && rsync -a '${SRC_DATA}/' '${LOCAL_DATA}/' && touch '${LOCAL_DATA}/.complete'
  fi"

source "${BASE_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export TMPDIR=/tmp
echo "Node: $(hostname)  RUN=$RUN"

EXTRA=()
[ "${BENCHMARK:-0}" = "1" ] && EXTRA+=(--benchmark)

# exec: python replaces the batch shell, so it receives SIGUSR1 directly
exec python -u "${PROJECT_DIR}/src/pretrain_vit_orth.py" \
  --data          "$LOCAL_DATA" \
  --num-classes   200 \
  --output-dir    "$OUT_DIR" \
  "${MODE_ARGS[@]}" \
  --epochs        30 \
  --warmup-epochs 3 \
  --constraint-warmup 3 \
  --batch-size    1024 \
  --micro-batch   256 \
  --num-workers   3 \
  --eval-every    10 \
  --resume \
  ${EXTRA[@]+"${EXTRA[@]}"}
