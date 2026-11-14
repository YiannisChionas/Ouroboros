"""List the redundant SLURM logs of finished runs (failed / repeated jobs). Runs LOCALLY only.

Usage:
    python3 cleanup_list.py <local_log_dir> [<local_log_dir> ...] [--apply-local]

Rule: for every task keep only the latest (highest job id) .out that has a "TAg Acc" section; every other
.out of that task, and its .err, is redundant.
Prints the redundant files per dir and the matching `ssh hpc "cd <flat dir> && rm ..."` command for the
HPC copy (not executed). --apply-local also deletes them locally.

Never use it on `joint` runs (several .out per task by design) or on runs that still have queued jobs.
"""
import os
import re
import sys

HPC_LOGS = '/home/it219111/experiments/Ouroboros/logs'


def redundant(dirpath):
    by_task = {}
    for fname in os.listdir(dirpath):
        if not fname.endswith('.out'):
            continue
        with open(os.path.join(dirpath, fname)) as f:
            first = f.readline()
        m = re.match(r'START_TASK=(\d+)', first)
        jm = re.search(r'-(\d+)\.out$', fname)
        if not m or not jm:
            continue
        by_task.setdefault(int(m.group(1)), []).append((int(jm.group(1)), fname))
    out = []
    for task, jobs in by_task.items():
        kept = False
        for jobid, fname in sorted(jobs, reverse=True):
            with open(os.path.join(dirpath, fname)) as f:
                complete = 'TAg Acc' in f.read()
            if not kept and complete:
                kept = True
            else:
                out.append(fname)
                err = fname[:-4] + '.err'
                if os.path.exists(os.path.join(dirpath, err)):
                    out.append(err)
    return sorted(out)


def flat_name(dirpath):
    """logs/{backbone}/{dataset}/{approach} -> {approach}_{dataset}_{backbone}"""
    parts = os.path.normpath(os.path.abspath(dirpath)).split(os.sep)
    return f'{parts[-1]}_{parts[-2]}_{parts[-3]}'


def main():
    dirs = [a for a in sys.argv[1:] if a != '--apply-local']
    if not dirs:
        print(__doc__)
        sys.exit(1)
    for d in dirs:
        if os.path.basename(os.path.normpath(d)) == 'joint':
            print(f'{d}: skipped (joint)')
            continue
        files = redundant(d)
        n_left = len([f for f in os.listdir(d) if f.endswith('.out')]) - len([f for f in files if f.endswith('.out')])
        print(f'{d}: {len(files)} redundant files, {n_left} .out left')
        if not files:
            continue
        for f in files:
            print(f'  {f}')
        print(f'  HPC: ssh hpc "cd {HPC_LOGS}/{flat_name(d)} && rm {" ".join(files)}"')
        if '--apply-local' in sys.argv:
            for f in files:
                os.remove(os.path.join(d, f))
            print('  deleted locally')


if __name__ == '__main__':
    main()
