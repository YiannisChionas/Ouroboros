#!/bin/bash
#SBATCH --job-name=copy_data_to_opt
#SBATCH --partition=obiwan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=05:00:00
#SBATCH --output=/opt/storage/users/it219111/%x-%j.out
#SBATCH --error=/opt/storage/users/it219111/%x-%j.err

# CPU-only job (no --gres): copy the datasets from /home (200G quota, counts /var/local too) to
# /opt/storage/users/it219111/data (not in the quota, slower NFS). Training jobs only read it to fill the node's
# local copy (/var/local/storage) when that copy is missing. Resumable: re-submit if it hits the time limit.
# The /home copies are deleted separately, after this job reports COPY COMPLETE for every dataset.

set -euo pipefail

SRC="/home/it219111/git/data"
DST="/opt/storage/users/it219111/data"
mkdir -p "$DST"
echo "Node: $(hostname)"

for name in "$@"; do
  rsync -a --exclude '.cache' "${SRC}/${name}/" "${DST}/${name}/"
  # Check: every image listed in train.txt / test.txt exists in the copy
  n_list=$(cat "$DST/$name/train.txt" "$DST/$name/test.txt" | wc -l)
  n_missing=$(cat "$DST/$name/train.txt" "$DST/$name/test.txt" | awk -v d="$DST/$name" '{print d"/"$1}' | xargs -d '\n' ls 2>&1 >/dev/null | wc -l)
  echo "${name}: listed=${n_list} missing=${n_missing} classes=$(wc -l < "$DST/$name/classes.txt")"
  if [ "$n_missing" -eq 0 ]; then echo "${name}: COPY COMPLETE"; else echo "${name}: COPY INCOMPLETE"; exit 1; fi
done
