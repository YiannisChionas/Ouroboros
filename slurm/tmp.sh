#!/bin/bash
# One-off: joint food101 vit_base_in21k from task 6 (tasks 6-9)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/vit_base_in21k/food101/joint.sh"

# Task 6: 2 jobs (35 epochs each)
JID=$(sbatch --export=ALL,START_TASK=6,STOP_TASK=7,START_EPOCH=0,STOP_EPOCH=35 "$SCRIPT" | awk '{print $NF}')
JID=$(sbatch --dependency=afterok:$JID --export=ALL,START_TASK=6,STOP_TASK=7,START_EPOCH=35,STOP_EPOCH=70 "$SCRIPT" | awk '{print $NF}')

# Task 7: 2 jobs
JID=$(sbatch --dependency=afterok:$JID --export=ALL,START_TASK=7,STOP_TASK=8,START_EPOCH=0,STOP_EPOCH=35 "$SCRIPT" | awk '{print $NF}')
JID=$(sbatch --dependency=afterok:$JID --export=ALL,START_TASK=7,STOP_TASK=8,START_EPOCH=35,STOP_EPOCH=70 "$SCRIPT" | awk '{print $NF}')

# Task 8: 4 jobs (17 epochs each)
JID=$(sbatch --dependency=afterok:$JID --export=ALL,START_TASK=8,STOP_TASK=9,START_EPOCH=0,STOP_EPOCH=17 "$SCRIPT" | awk '{print $NF}')
JID=$(sbatch --dependency=afterok:$JID --export=ALL,START_TASK=8,STOP_TASK=9,START_EPOCH=17,STOP_EPOCH=35 "$SCRIPT" | awk '{print $NF}')
JID=$(sbatch --dependency=afterok:$JID --export=ALL,START_TASK=8,STOP_TASK=9,START_EPOCH=35,STOP_EPOCH=52 "$SCRIPT" | awk '{print $NF}')
JID=$(sbatch --dependency=afterok:$JID --export=ALL,START_TASK=8,STOP_TASK=9,START_EPOCH=52,STOP_EPOCH=70 "$SCRIPT" | awk '{print $NF}')

# Task 9: 4 jobs
JID=$(sbatch --dependency=afterok:$JID --export=ALL,START_TASK=9,STOP_TASK=10,START_EPOCH=0,STOP_EPOCH=17 "$SCRIPT" | awk '{print $NF}')
JID=$(sbatch --dependency=afterok:$JID --export=ALL,START_TASK=9,STOP_TASK=10,START_EPOCH=17,STOP_EPOCH=35 "$SCRIPT" | awk '{print $NF}')
JID=$(sbatch --dependency=afterok:$JID --export=ALL,START_TASK=9,STOP_TASK=10,START_EPOCH=35,STOP_EPOCH=52 "$SCRIPT" | awk '{print $NF}')
JID=$(sbatch --dependency=afterok:$JID --export=ALL,START_TASK=9,STOP_TASK=10,START_EPOCH=52,STOP_EPOCH=70 "$SCRIPT" | awk '{print $NF}')

echo "Done. Last job: $JID"
