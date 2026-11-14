"""Final results of runs from their local SLURM .out logs. Runs LOCALLY only.

Usage:
    python3 extract_results.py <log_dir> [<log_dir> ...] [--mean]

Each .out prints the TAw Acc / TAg Acc / TAw Forg / TAg Forg matrices up to its STOP_TASK; rows of tasks
not yet trained are 0. The result of a run is read from the .out with the highest STOP_TASK
(ties: highest job id) that has a "TAg Acc" section; the value is the last "Avg.:" of each section.

--mean  also prints mean and sample std over the given dirs (e.g. seeds 0/1/2 of one run).
"""
import os
import re
import statistics
import sys

SECTIONS = ['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg']


def last_avg(lines, name):
    """Last 'Avg.:' value of section `name`, or None."""
    for i, line in enumerate(lines):
        if line.strip() == name:
            avgs = []
            for line2 in lines[i + 1:]:
                if line2.startswith('*' * 10):
                    break
                m = re.search(r'Avg\.: *([0-9]+\.[0-9]+)%', line2)
                if m:
                    avgs.append(float(m.group(1)))
            return avgs[-1] if avgs else None
    return None


def run_result(log_dir):
    """(stop_task, num_tasks, {section: value}) of the latest complete .out, or None."""
    best = None
    for fname in os.listdir(log_dir):
        if not fname.endswith('.out'):
            continue
        with open(os.path.join(log_dir, fname)) as f:
            content = f.read()
        if 'TAg Acc' not in content:
            continue
        m = re.match(r'START_TASK=(\d+) STOP_TASK=(\d+)', content)
        stop = int(m.group(2)) if m else -1
        jobid = int(re.search(r'-(\d+)\.out$', fname).group(1)) if re.search(r'-(\d+)\.out$', fname) else -1
        if best is None or (stop, jobid) > best[0]:
            best = ((stop, jobid), content)
    if best is None:
        return None
    lines = best[1].splitlines()
    # number of tasks = number of rows of the TAg Acc matrix
    i = next(k for k, l in enumerate(lines) if l.strip() == 'TAg Acc')
    n_tasks = 0
    for l in lines[i + 1:]:
        if l.startswith('*' * 10):
            break
        n_tasks += 'Avg.:' in l
    return best[0][0], n_tasks, {s: last_avg(lines, s) for s in SECTIONS}


def main():
    args = [a for a in sys.argv[1:] if a != '--mean']
    if not args:
        print(__doc__)
        sys.exit(1)
    rows = []
    width = max(len(a.rstrip('/')) for a in args)
    print(f"{'run':<{width}}  {'tasks':>7}  {'TAw':>6}  {'TAg':>6}  {'TAwF':>6}  {'TAgF':>6}")
    for d in args:
        d = d.rstrip('/')
        if not os.path.isdir(d):
            print(f'{d:<{width}}  missing dir')
            continue
        r = run_result(d)
        if r is None:
            print(f'{d:<{width}}  no complete .out')
            continue
        stop, n, v = r
        done = f'{stop}/{n}' + ('' if stop == n else ' !')
        fmt = lambda x: '   n/a' if x is None else f'{x:6.1f}'
        print(f'{d:<{width}}  {done:>7}  ' + '  '.join(fmt(v[s]) for s in SECTIONS))
        if stop == n:
            rows.append(v)
    if '--mean' in sys.argv and len(rows) > 1:
        line = []
        for s in SECTIONS:
            xs = [r[s] for r in rows if r[s] is not None]
            line.append(f'{statistics.mean(xs):.2f}±{statistics.stdev(xs):.2f}' if len(xs) > 1 else 'n/a')
        print(f"{'mean±std (' + str(len(rows)) + ' runs)':<{width}}  {'':>7}  " + '  '.join(line))


if __name__ == '__main__':
    main()
