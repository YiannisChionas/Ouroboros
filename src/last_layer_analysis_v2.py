"""last_layer_analysis + task calibration analysis.

task_calibration() runs after every task (config key task_calibration_analysis) and adds that step to
<exp_path>/task_calibration.npz. Arrays are (total_tasks, total_tasks, total_tasks), indexed [step, i, j]:
  confusion[step, i, j]  fraction of task-i test samples whose task-agnostic prediction falls in task j
  max_logit[step, i, h]  mean max logit of head h on task-i test samples
The diagonal of confusion[step] is the task-ID accuracy. calibration_figures() plots the last step.
"""
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from last_layer_analysis import last_layer_analysis  # noqa: F401  (old weight/bias plots, re-exported)

CALIBRATION_FILE = 'task_calibration.npz'


def _head_logits(outputs):
    """Per-head logits. Distilled networks return a dict: use cls_logits, as their eval does for TAg."""
    return outputs['cls_logits'] if isinstance(outputs, dict) else outputs


def task_calibration(approach, test_loader, task, total_tasks, exp_path):
    """Measure task confusion and per-head max logits after `task` and store them in the npz file."""
    path = os.path.join(exp_path, CALIBRATION_FILE)
    if os.path.isfile(path):
        data = dict(np.load(path))
    else:
        shape = (total_tasks, total_tasks, total_tasks)
        data = {'confusion': np.zeros(shape, dtype=np.float32), 'max_logit': np.zeros(shape, dtype=np.float32)}

    n_heads = task + 1
    model = approach.model
    task_offset = model.task_offset[:n_heads].to(approach.device)
    model.eval()
    with torch.no_grad():
        for i in range(n_heads):
            counts = torch.zeros(n_heads, device=approach.device)
            logit_sum = torch.zeros(n_heads, device=approach.device)
            num = 0
            for images, _ in test_loader[i]:
                heads = _head_logits(model(images.to(approach.device)))[:n_heads]
                if approach.multi_softmax:
                    heads_tag = [torch.nn.functional.log_softmax(h, dim=1) for h in heads]
                else:
                    heads_tag = heads
                pred = torch.cat(heads_tag, dim=1).argmax(1)
                pred_task = (task_offset.unsqueeze(0) <= pred.unsqueeze(1)).sum(1) - 1
                counts += torch.bincount(pred_task, minlength=n_heads).float()
                logit_sum += torch.stack([h.max(1).values.sum() for h in heads])
                num += len(images)
            data['confusion'][task, i, :n_heads] = (counts / num).cpu().numpy()
            data['max_logit'][task, i, :n_heads] = (logit_sum / num).cpu().numpy()

    np.savez(path, **data)
    conf = data['confusion'][task, :n_heads, :n_heads]
    print('Task calibration after task {}: task-ID acc avg {:.1f}% | per task: {}'.format(
        task, 100 * np.diag(conf).mean(), ' '.join('{:.1f}'.format(100 * x) for x in np.diag(conf))))


def calibration_figures(exp_path, task):
    """Figures for the stored calibration up to `task`: confusion and max-logit heatmaps, task-ID accuracy per step."""
    data = np.load(os.path.join(exp_path, CALIBRATION_FILE))
    n = task + 1
    figs = {}

    for name, label in [('confusion', 'Fraction of predictions'), ('max_logit', 'Mean max logit')]:
        f = plt.figure(dpi=300)
        ax = f.subplots(nrows=1, ncols=1)
        im = ax.imshow(data[name][task, :n, :n], cmap='viridis')
        ax.set_xlabel('Predicted task' if name == 'confusion' else 'Head', fontsize=11, fontfamily='serif')
        ax.set_ylabel('True task', fontsize=11, fontfamily='serif')
        f.colorbar(im, ax=ax, label=label)
        figs['calib_' + name] = f

    f = plt.figure(dpi=300)
    ax = f.subplots(nrows=1, ncols=1)
    steps = [s for s in range(n) if data['confusion'][s].any()]
    ax.plot(steps, [100 * np.diag(data['confusion'][s, :s + 1, :s + 1]).mean() for s in steps], marker='o')
    ax.set_xlabel('Task', fontsize=11, fontfamily='serif')
    ax.set_ylabel('Task-ID accuracy (%)', fontsize=11, fontfamily='serif')
    figs['calib_taskid_acc'] = f
    return figs
