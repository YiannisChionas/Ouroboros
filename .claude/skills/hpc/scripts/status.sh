#!/bin/bash
# One-shot cluster status for it219111 (a single ssh call, read-only). Runs LOCALLY only.
#
# Usage:
#   status.sh [SINCE]     SINCE = start date for the failure list (default: 2 days ago), e.g. 2026-11-10
#
# Prints: running jobs, pending jobs that are not just waiting on a dependency, number of queued jobs,
# jobs since SINCE that did not complete (FAILED / TIMEOUT / OOM / CANCELLED ...), node load of the
# partitions we can use, and our position in the yoda priority queue.

set -euo pipefail

SINCE="${1:-$(date -v-2d +%Y-%m-%d 2>/dev/null || date -d '2 days ago' +%Y-%m-%d)}"

ssh hpc "
echo '== running'
squeue -u it219111 -t R -o '%.8i %.42j %.8P %.10M %.10l %.10R'
echo '== pending, not waiting on a dependency'
squeue -u it219111 -t PD -o '%.8i %.42j %.8P %.20R' | grep -v Dependency
echo \"== queued in total: \$(squeue -u it219111 -h | wc -l)\"
echo '== not completed since ${SINCE}'
sacct -u it219111 -S ${SINCE} -X -n --format=JobID,JobName%42,Partition%8,State%14,ExitCode,Elapsed \
  | grep -v -E 'COMPLETED|PENDING|RUNNING' || echo '(none)'
echo '== node load (yoda, obiwan, leia)'
sinfo -N -p yoda,obiwan,leia -o '%N %P %t %C %G'
echo '== our position in the yoda priority queue'
squeue -p yoda -t PD -h -o '%Q %u %r' | sort -rn | awk '\$3==\"Priority\" || \$3==\"Resources\" {n++; if (\$2==\"it219111\") print \"rank \" n \" (priority \" \$1 \")\"}' | head -3 | grep . || echo '(none of our jobs is waiting on priority)'
"
