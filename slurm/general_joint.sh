#!/bin/bash
# Usage: bash general_joint.sh <job_script> <num_tasks> <nepochs> [SPLIT1] [SPLIT2] [INITIAL_DEP] [START_FROM]
#
# Submits joint training jobs with epoch-based splitting per task.
# Split scheme (based on dataset size growth):
#   tasks 0       to SPLIT1-1 : 1 job  per task (full nepochs in one job)
#   tasks SPLIT1  to SPLIT2-1 : 2 jobs per task (nepochs/2 each)
#   tasks SPLIT2  to end      : 4 jobs per task (nepochs/4 each)
#
# SPLIT1 and SPLIT2 default to 5 and 8 (for 10-task datasets).
# For inat200 (20 tasks) use SPLIT1=10 SPLIT2=15.
# START_FROM: first task to submit (default 0) — use to resume after partial completion.
#
# Example:
#   bash general_joint.sh vit_small_in1k/cifar100/joint.sh 10 50
#   bash general_joint.sh vit_small_in1k/food101/joint.sh  10 70
#   bash general_joint.sh resnet50_in1k/inat200/joint.sh   20 90 10 15
#   bash general_joint.sh resnetv2_101_in21k/food101/joint.sh 10 70 5 8 "" 3  # resume from task 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/$1"
NUM_TASKS="$2"
NEPOCHS="$3"
SPLIT1="${4:-5}"
SPLIT2="${5:-8}"
INITIAL_DEP="${6:-}"
START_FROM="${7:-0}"

if [ ! -f "$JOB_SCRIPT" ]; then
    echo "ERROR: Job script not found: $JOB_SCRIPT"
    exit 1
fi

echo "Submitting joint jobs: script=$1 tasks=${START_FROM}-$((NUM_TASKS-1)) nepochs=$NEPOCHS"
echo "Split: tasks 0-$((SPLIT1-1))=1job, ${SPLIT1}-$((SPLIT2-1))=2jobs, ${SPLIT2}+=4jobs"
echo ""

prev_jid=""

for task in $(seq $START_FROM $((NUM_TASKS - 1))); do

    # Determine number of epoch-jobs for this task
    if [ "$task" -lt "$SPLIT1" ]; then
        N_EPOCH_JOBS=1
    elif [ "$task" -lt "$SPLIT2" ]; then
        N_EPOCH_JOBS=2
    else
        N_EPOCH_JOBS=4
    fi

    EPOCHS_PER_JOB=$(( NEPOCHS / N_EPOCH_JOBS ))

    task_prev_jid=""

    for ej in $(seq 0 $((N_EPOCH_JOBS - 1))); do
        START_EPOCH=$(( ej * EPOCHS_PER_JOB ))
        STOP_EPOCH=$(( START_EPOCH + EPOCHS_PER_JOB ))
        # Last epoch-job of last task gets full nepochs to avoid rounding issues
        if [ "$ej" -eq "$((N_EPOCH_JOBS - 1))" ]; then
            STOP_EPOCH=$NEPOCHS
        fi

        # Build dependency string
        if [ -n "$task_prev_jid" ]; then
            # Wait for previous epoch-job of same task
            DEP="--dependency=afterok:${task_prev_jid}"
        elif [ -n "$prev_jid" ]; then
            # First epoch-job of this task: wait for last job of previous task
            DEP="--dependency=afterok:${prev_jid}"
        elif [ -n "$INITIAL_DEP" ]; then
            # Very first job: wait for external dependency (chaining approaches)
            DEP="--dependency=afterany:${INITIAL_DEP}"
        else
            DEP=""
        fi

        JID=$(sbatch $DEP \
            --export=ALL,START_TASK=$task,STOP_TASK=$((task+1)),START_EPOCH=$START_EPOCH,STOP_EPOCH=$STOP_EPOCH \
            "$JOB_SCRIPT" | awk '{print $NF}')

        echo "  Task $task | epochs $START_EPOCH-$STOP_EPOCH | job $JID $([ -n "$DEP" ] && echo "dep=$DEP" || echo "")"

        task_prev_jid="$JID"
    done

    prev_jid="$task_prev_jid"
done

echo ""
echo "Done. All jobs submitted."
echo "last_job=$prev_jid"
