#!/bin/bash
# Submits all missing approaches for resnet50_in1k/cifar100 in series.
# Each approach waits for the entire previous approach to finish.
# Usage: bash slurm/submit_resnet50_cifar100_chain.sh
# Run from: /home/it219111/git/Ouroboros

set -euo pipefail
cd /home/it219111/git/Ouroboros

BASE_LOG="/home/it219111/experiments/Ouroboros/logs"

# Create log dirs
for app in finetuning freezing simplecil ewc lwf icarl joint; do
    mkdir -p "${BASE_LOG}/${app}_cifar100_resnet50_in1k"
    echo "Created log dir: ${app}_cifar100_resnet50_in1k"
done

extract_last() { grep 'last_job' | tail -1 | cut -d= -f2; }

# 1. finetuning
echo "=== Submitting finetuning ==="
out=$(bash slurm/general_serial.sh resnet50_in1k/cifar100/finetuning.sh 10)
echo "$out"
last=$(echo "$out" | extract_last)

# 2. freezing
echo "=== Submitting freezing (dep: $last) ==="
out=$(bash slurm/general_serial.sh resnet50_in1k/cifar100/freezing.sh 10 0 "$last")
echo "$out"
last=$(echo "$out" | extract_last)

# 3. simplecil
echo "=== Submitting simplecil (dep: $last) ==="
out=$(bash slurm/general_serial.sh resnet50_in1k/cifar100/simplecil.sh 10 0 "$last")
echo "$out"
last=$(echo "$out" | extract_last)

# 4. ewc
echo "=== Submitting ewc (dep: $last) ==="
out=$(bash slurm/general_serial.sh resnet50_in1k/cifar100/ewc.sh 10 0 "$last")
echo "$out"
last=$(echo "$out" | extract_last)

# 5. lwf
echo "=== Submitting lwf (dep: $last) ==="
out=$(bash slurm/general_serial.sh resnet50_in1k/cifar100/lwf.sh 10 0 "$last")
echo "$out"
last=$(echo "$out" | extract_last)

# 6. icarl
echo "=== Submitting icarl (dep: $last) ==="
out=$(bash slurm/general_serial.sh resnet50_in1k/cifar100/icarl.sh 10 0 "$last")
echo "$out"
last=$(echo "$out" | extract_last)

# 7. joint (cifar100=10 tasks, 90 epochs, default splits 5/8)
echo "=== Submitting joint (dep: $last) ==="
out=$(bash slurm/general_joint.sh resnet50_in1k/cifar100/joint.sh 10 90 5 8 "$last")
echo "$out"

echo ""
echo "All resnet50_in1k/cifar100 approaches submitted in series."
