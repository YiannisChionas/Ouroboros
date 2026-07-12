#!/bin/bash
# Like general.sh but serializes the first job of each approach via an optional initial dependency.
# Use when submitting multiple independent approaches simultaneously to avoid AssocMaxJobsLimit.
#
# Usage:
#   bash general_serial.sh <job_script.sh> [NUM_TASKS] [START_FROM] [INITIAL_DEP_JOBID]
#
# Examples:
#   # Submit 4 approaches in sequence — each waits for the previous approach's task-0 to finish:
#   ewc_first=$(bash general_serial.sh vit_base_in21k/inat200/ewc.sh 20 | grep 'first_job' | cut -d= -f2)
#   bash general_serial.sh vit_base_in21k/inat200/l2p.sh 20 0 $ewc_first
#
# The first job uses --dependency=afterany:INITIAL_DEP_JOBID (afterany = regardless of success/failure).
# Subsequent jobs in the chain still use afterok (normal behavior).
# Prints "first_job=JOBID" on the last line for easy capture.

set -euo pipefail

JOB_SCRIPT="${1:-}"
NUM_TASKS="${2:-10}"
START_FROM="${3:-0}"
INITIAL_DEP="${4:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL_SCRIPT="$SCRIPT_DIR/$JOB_SCRIPT"

if [ -z "$JOB_SCRIPT" ]; then
  echo "ERROR: Missing job script."
  echo "Usage: $0 <job_script.sh> [NUM_TASKS] [START_FROM] [INITIAL_DEP_JOBID]"
  exit 1
fi

if [ ! -f "$FULL_SCRIPT" ]; then
  echo "ERROR: Script not found: $FULL_SCRIPT"
  exit 1
fi

prev_jobid=""
first_jobid=""
start="$START_FROM"

while [ "$start" -lt "$NUM_TASKS" ]; do
  stop=$((start + 1))

  export START_TASK="$start"
  export STOP_TASK="$stop"
  export NUM_TASKS="$NUM_TASKS"

  if [ -z "$prev_jobid" ]; then
    if [ -n "$INITIAL_DEP" ]; then
      jobid=$(sbatch --export=ALL --parsable --dependency=afterany:"$INITIAL_DEP" "$FULL_SCRIPT")
    else
      jobid=$(sbatch --export=ALL --parsable "$FULL_SCRIPT")
    fi
    first_jobid="$jobid"
  else
    jobid=$(sbatch --export=ALL --parsable --dependency=afterok:"$prev_jobid" "$FULL_SCRIPT")
  fi

  echo "Submitted task $start→$stop as job $jobid"
  prev_jobid="$jobid"
  start="$stop"
done

echo "All tasks $START_FROM→$NUM_TASKS submitted."
echo "first_job=$first_jobid"
echo "last_job=$prev_jobid"
