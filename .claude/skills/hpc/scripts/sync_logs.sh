#!/bin/bash
# Sync SLURM logs HPC (flat) -> local (structured). Runs LOCALLY only.
#
# Usage:
#   sync_logs.sh <regex>     sync the HPC log dirs whose name matches <regex> (grep -E)
#   sync_logs.sh --all       sync every HPC log dir
#
# Mapping:
#   {approach}_{dataset}_{backbone}  ->  logs/{backbone}/{dataset}/{approach}/
#   (dataset anchors: cifar100, food101, inatm200, inat500, inat200, five_datasets)
#   no dataset in the name           ->  logs/analysis/{name}/ if it exists locally, else logs/_misc/{name}/
#
# Existing local files are not overwritten (--ignore-existing). Afterwards every local .out without a
# "TAg Acc" section (copied while its job was running) is fetched again.

set -euo pipefail

REPO="/Users/ioannischionas/Code/cl/Ouroboros"
HPC_LOGS="/home/it219111/experiments/Ouroboros/logs"
PATTERN="${1:-}"

if [ -z "$PATTERN" ]; then
  echo "Usage: $0 <regex> | --all"; exit 1
fi
[ "$PATTERN" = "--all" ] && PATTERN="."

names=$(ssh hpc "ls ${HPC_LOGS}" | grep -E -- "$PATTERN" || true)
if [ -z "$names" ]; then
  echo "No HPC log dir matches '$PATTERN'"; exit 0
fi

local_dir() {
  local name="$1"
  if [[ "$name" =~ ^(.+)_(cifar100|food101|inatm200|inat500|inat200|five_datasets)_(.+)$ ]]; then
    echo "${REPO}/logs/${BASH_REMATCH[3]}/${BASH_REMATCH[2]}/${BASH_REMATCH[1]}"
  elif [ -d "${REPO}/logs/analysis/${name}" ]; then
    echo "${REPO}/logs/analysis/${name}"
  else
    echo "${REPO}/logs/_misc/${name}"
  fi
}

for name in $names; do
  dst=$(local_dir "$name")
  mkdir -p "$dst"
  rsync -az --ignore-existing "hpc:${HPC_LOGS}/${name}/" "${dst}/"

  # Re-fetch .out files that were copied before their job finished
  for f in "$dst"/*.out; do
    [ -e "$f" ] || continue
    if ! grep -q 'TAg Acc' "$f"; then
      rsync -az "hpc:${HPC_LOGS}/${name}/$(basename "$f")" "${dst}/"
      rsync -az "hpc:${HPC_LOGS}/${name}/$(basename "${f%.out}.err")" "${dst}/" 2>/dev/null || true
    fi
  done

  n_out=$(find "$dst" -maxdepth 1 -name '*.out' | wc -l | tr -d ' ')
  n_inc=0
  for f in "$dst"/*.out; do [ -e "$f" ] && ! grep -q 'TAg Acc' "$f" && n_inc=$((n_inc + 1)); done
  echo "${name} -> ${dst#${REPO}/}  (.out: ${n_out}, without TAg Acc: ${n_inc})"
done
