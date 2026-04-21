#!/bin/bash
# Launch a SLURM job chain for a single approach, one task at a time.
#
# Usage:
#   bash general.sh <job_script.sh> [NUM_TASKS] [START_FROM]
#
# Examples:
#   bash general.sh vit_small_in1k/cifar100/finetuning.sh 10
#   bash general.sh vit_small_in1k/inat200/lwf.sh 20
#   bash general.sh deit_small_in1k/inat200/hydra_v4.sh 20 10   # resume from task 10

set -euo pipefail

JOB_SCRIPT="${1:-}"
NUM_TASKS="${2:-10}"
START_FROM="${3:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL_SCRIPT="$SCRIPT_DIR/$JOB_SCRIPT"

if [ -z "$JOB_SCRIPT" ]; then
  echo "ERROR: Missing job script."
  echo "Usage: $0 <job_script.sh> [NUM_TASKS]"
  exit 1
fi

if [ ! -f "$FULL_SCRIPT" ]; then
  echo "ERROR: Script not found: $FULL_SCRIPT"
  exit 1
fi

prev_jobid=""
start="$START_FROM"

while [ "$start" -lt "$NUM_TASKS" ]; do
  stop=$((start + 1))

  export START_TASK="$start"
  export STOP_TASK="$stop"
  export NUM_TASKS="$NUM_TASKS"

  if [ -z "$prev_jobid" ]; then
    jobid=$(sbatch --export=ALL --parsable "$FULL_SCRIPT")
  else
    jobid=$(sbatch --export=ALL --parsable --dependency=afterok:"$prev_jobid" "$FULL_SCRIPT")
  fi

  echo "Submitted task $start→$stop as job $jobid"
  prev_jobid="$jobid"
  start="$stop"
done

echo "All tasks $START_FROM→$NUM_TASKS submitted."
