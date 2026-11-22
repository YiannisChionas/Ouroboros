"""Task calibration of a finished run, from its final checkpoint (offline, eval only).

Usage:
    python src/analyze_task_calibration.py --config <config.json> --ckpt <models/taskT.ckpt> --out <dir>
        [--data-path <dir>] [--max-batches N]

Builds the network as cil_trainer.train() does, adds one head per task up to the checkpoint's task, loads the
checkpoint and runs last_layer_analysis_v2.task_calibration() on the test sets of all seen tasks:
<out>/task_calibration.npz with confusion[T, i, j] (task-agnostic predictions of task-i samples falling in task j)
and max_logit[T, i, h] (mean max logit of head h on task-i samples). Also prints task-ID accuracy and TAg / TAw per
task as a check against the run's own log.
--data-path overrides the config (e.g. read the test images from /opt/storage instead of the node's local copy).
--max-batches limits the batches per task (local smoke tests only).
"""
import argparse
import importlib
import json
import os
import re
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.data_loader import get_loaders  # noqa: E402
from last_layer_analysis_v2 import task_calibration, _head_logits  # noqa: E402
from networks import tvmodels, timmmodels, set_model_head_var  # noqa: E402
from networks.network import LLL_Net  # noqa: E402
from networks.distil_network import LLL_Net_Distilled  # noqa: E402
from networks.distil_network_hydra_v2 import LLL_Net_Hydra_v2  # noqa: E402
from networks.distil_network_vit_hydra import LLL_Net_VitHydra  # noqa: E402
from networks.distil_network_cls_only import LLL_Net_Cls_Only  # noqa: E402
from trainers.cil_trainer import _set_defaults  # noqa: E402


def build_network(args):
    """Same selection as cil_trainer.train(); weights come from the checkpoint, so no pretrained download."""
    name = args['network']
    if name in tvmodels:
        init = getattr(importlib.import_module('torchvision.models'), name)(pretrained=False)
    elif name in timmmodels:
        import timm
        init = timm.create_model(name, pretrained=False, num_classes=0)
    else:
        init = getattr(importlib.import_module('networks'), name)(pretrained=False)
    set_model_head_var(init)
    if args.get('vit_hydra'):
        return LLL_Net_VitHydra(init, None)
    if args.get('hydra_v2'):
        return LLL_Net_Hydra_v2(init, None)
    if args.get('cls_only'):
        return LLL_Net_Cls_Only(init, remove_existing_head=False)
    if args.get('distilled'):
        return LLL_Net_Distilled(init, remove_existing_head=False)
    return LLL_Net(init, remove_existing_head=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--data-path', default=None)
    p.add_argument('--max-batches', type=int, default=None)
    a = p.parse_args()

    args = json.load(open(a.config))
    _set_defaults(args)
    if a.data_path:
        args['data_path'] = a.data_path
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    task = int(re.search(r'task(\d+)\.ckpt$', a.ckpt).group(1))

    _, _, test_loader, classes_per_task = get_loaders(args)
    total_tasks = len(classes_per_task)
    if a.max_batches:
        test_loader = [[b for _, b in zip(range(a.max_batches), ld)] for ld in test_loader]

    net = build_network(args)
    for t in range(task + 1):
        net.add_head(classes_per_task[t][1])
    net.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
    net.to(device).eval()
    print(f'loaded {a.ckpt} (task {task}), {total_tasks} tasks, device {device}')

    # task_calibration only needs model, device and multi_softmax from the approach
    appr = types.SimpleNamespace(model=net, device=device, multi_softmax=args['multi_softmax'])
    os.makedirs(a.out, exist_ok=True)
    task_calibration(appr, test_loader, task, total_tasks, a.out)

    # Check against the run's log: TAw / TAg per task from the same checkpoint
    offset = net.task_offset
    taw, tag = [], []
    with torch.no_grad():
        for i in range(task + 1):
            hits_taw = hits_tag = num = 0
            for images, targets in test_loader[i]:
                heads = _head_logits(net(images.to(device)))[:task + 1]
                targets = targets.to(device)
                hits_taw += (heads[i].argmax(1) + offset[i].to(device) == targets).sum().item()
                if args['multi_softmax']:
                    heads = [torch.nn.functional.log_softmax(h, dim=1) for h in heads]
                hits_tag += (torch.cat(heads, dim=1).argmax(1) == targets).sum().item()
                num += len(targets)
            taw.append(hits_taw / num)
            tag.append(hits_tag / num)
    print('TAw per task: ' + ' '.join(f'{100 * x:.1f}' for x in taw) + f' | avg {100 * np.mean(taw):.1f}')
    print('TAg per task: ' + ' '.join(f'{100 * x:.1f}' for x in tag) + f' | avg {100 * np.mean(tag):.1f}')


if __name__ == '__main__':
    main()
