"""Smoke test of an approach with its real config, real network and random data. Runs LOCALLY (env ouroboros-test).

Usage (from the Ouroboros root):
    python .claude/skills/localtesting/scripts/smoke_approach.py <config.json> [--tasks 2] [--classes 3]

Builds the network like cil_trainer.train() (pretrained=False, weight files -> None), then for each task runs
approach.train() with 1 epoch and 1 warm-up epoch on 2 random images per class, and approach.eval() on every seen
task. Prints optimizer, trainable parameters, whether the warm-up changed the backbone, and the eval results.
Not covered: approaches with exemplars, data loading, checkpoint save/resume.
"""
import argparse
import importlib
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from trainers.cil_trainer import _set_defaults  # noqa: E402
from networks import tvmodels, timmmodels, set_model_head_var  # noqa: E402
from networks.network import LLL_Net  # noqa: E402
from networks.distil_network import LLL_Net_Distilled  # noqa: E402
from networks.distil_network_hydra import LLL_Net_Hydra  # noqa: E402
from networks.distil_network_hydra_v2 import LLL_Net_Hydra_v2  # noqa: E402
from networks.distil_network_hydra_v3 import LLL_Net_Hydra_v3  # noqa: E402
from networks.distil_network_vit_hydra import LLL_Net_VitHydra  # noqa: E402
from networks.distil_network_cls_only import LLL_Net_Cls_Only  # noqa: E402


class RandomTaskData(TensorDataset):
    """Random images with a .labels list, as the base warm-up reads train_loader.dataset.labels."""
    def __init__(self, images, labels):
        super().__init__(images, labels)
        self.labels = labels.tolist()


class StubLogger:
    def log_scalar(self, **kwargs):
        pass


def build_network(args):
    """Same selection as cil_trainer.train(), without pretrained weights."""
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
    for flag, cls in (('hydra_v3', LLL_Net_Hydra_v3), ('hydra_v2', LLL_Net_Hydra_v2), ('hydra', LLL_Net_Hydra)):
        if args.get(flag):
            return cls(init, None)
    if args.get('cls_only'):
        return LLL_Net_Cls_Only(init, remove_existing_head=False)
    if args.get('distilled'):
        return LLL_Net_Distilled(init, remove_existing_head=False)
    return LLL_Net(init, remove_existing_head=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--tasks', type=int, default=2)
    p.add_argument('--classes', type=int, default=3)
    a = p.parse_args()

    args = json.load(open(a.config))
    _set_defaults(args)
    args.update(device='cpu', nepochs=1, warmup_nepochs=min(args['warmup_nepochs'], 1))
    torch.manual_seed(0)

    net = build_network(args)
    Appr = getattr(importlib.import_module('approach.' + args['approach']), 'Appr')
    appr = Appr(args, model=net, logger=StubLogger(), exemplars_dataset=None)
    print(f"approach={args['approach']} network={args['network']} ({type(net).__name__}) "
          f"optimizer={args['optimizer_name']} lr={args['lr']} warmup_epochs={args['warmup_nepochs']}")

    size = args.get('crop', 224)
    loaders = []
    for t in range(a.tasks):
        y = torch.arange(t * a.classes, (t + 1) * a.classes).repeat(2)
        loaders.append(DataLoader(RandomTaskData(torch.randn(len(y), 3, size, size), y), batch_size=2))

    pre_train = appr.pre_train_process

    def checked_pre_train(task, loader):
        before = [q.detach().clone() for q in net.model.parameters()]
        pre_train(task, loader)
        changed = any(not torch.equal(b, q) for b, q in zip(before, net.model.parameters()))
        print(f'  warm-up changed backbone: {changed}')
    appr.pre_train_process = checked_pre_train

    for t in range(a.tasks):
        net.add_head(a.classes)
        print(f'task {t}')
        appr.train(t, loaders[t], loaders[t])
        n_train = sum(q.numel() for q in net.parameters() if q.requires_grad)
        n_all = sum(q.numel() for q in net.parameters())
        print(f'  optimizer: {type(appr.optimizer).__name__}, trainable params: {n_train:,} / {n_all:,}')
        for e in range(t + 1):
            loss, taw, tag = appr.eval(e, loaders[e])
            print(f'  eval task {e}: loss={loss:.3f} TAw={100 * taw:.1f} TAg={100 * tag:.1f}')
    print('OK')


if __name__ == '__main__':
    main()
