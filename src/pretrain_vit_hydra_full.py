#!/usr/bin/env python3
"""vit-hydra pretraining on ImageNet-1k (single GPU).

ViT-B/16 from scratch with 1 cls token + 5 distillation tokens (vit_hydra_tokens.py),
DeiT-B recipe, online distillation from 5 frozen in1k teachers:

    loss = w_cls * CE(cls, mixup targets) + sum_i w_kd * KD(dist_i || teacher_i)

KD: Hinton soft distillation, KL(batchmean) * T^2 (NOT the DeiT `/numel` variant).

Single process, global batch via gradient accumulation. Resumable mid-epoch:
  - checkpoint every --ckpt-minutes and on SIGUSR1 (sbatch --signal=B:USR1@300)
  - deterministic per-epoch sample order (seed + epoch), resumed at the saved step

Usage:
  python pretrain_vit_hydra_full.py --data /var/local/storage/$USER/imagenet1k \
      --output-dir results/pretraining/vit_hydra_full --run-epochs 1 --resume
"""

import argparse
import os
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import timm
from timm.data import create_transform, Mixup, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.dirname(__file__))
from networks.vit_hydra_tokens import vit_hydra_base_patch16_224

# (short name, timm model) — order = order of dist tokens 1..5
TEACHERS = [
    ('regnety160', 'regnety_160.deit_in1k'),
    ('convnext_b', 'convnext_base.fb_in1k'),
    ('swin_b',     'swin_base_patch4_window7_224.ms_in1k'),
    # torchvision v2 recipe (softmax CE); the timm a1 weights are BCE-trained -> logits not softmax-calibrated
    ('resnet101',  'resnet101.tv2_in1k'),
    ('resnet50',   'resnet50.tv2_in1k'),
]

STOP = {'flag': False}


def _on_sigusr1(signum, frame):
    STOP['flag'] = True


def kd_loss(student_logits, teacher_logits, T):
    """Hinton soft distillation: KL(teacher || student) at temperature T, batchmean, * T^2."""
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.log_softmax(teacher_logits / T, dim=1),
        reduction='batchmean',
        log_target=True,
    ) * (T * T)


class RepeatAugSampler(Sampler):
    """DeiT repeated augmentation for a single process.

    Each epoch: permutation(seed + epoch), every index repeated `num_repeats` times,
    truncated to a multiple of the global batch (len(dataset) samples per epoch, as in
    DeiT's RASampler). `start` skips already-seen samples when resuming mid-epoch.
    """

    def __init__(self, num_items, num_repeats, global_batch, seed):
        self.num_items = num_items
        self.num_repeats = num_repeats
        self.seed = seed
        self.num_samples = num_items // global_batch * global_batch
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch, start=0):
        self.epoch, self.start = epoch, start

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.randperm(self.num_items, generator=g).repeat_interleave(self.num_repeats)
        return iter(idx[:self.num_samples][self.start:].tolist())

    def __len__(self):
        return self.num_samples - self.start


def build_datasets(data_dir):
    # DeiT build_transform (is_train=True / False)
    train_tf = create_transform(
        224, is_training=True, color_jitter=0.3, auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic', re_prob=0.25, re_mode='pixel', re_count=1)
    val_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return (ImageFolder(os.path.join(data_dir, 'train'), train_tf),
            ImageFolder(os.path.join(data_dir, 'val'), val_tf))


def build_teachers(device):
    teachers = []
    for short, name in TEACHERS:
        t = timm.create_model(name, pretrained=True)
        cfg = t.pretrained_cfg
        # all teachers must see the student's input as-is (same normalization, 224px)
        assert tuple(cfg['mean']) == IMAGENET_DEFAULT_MEAN and tuple(cfg['std']) == IMAGENET_DEFAULT_STD, name
        t = t.to(device).eval().to(memory_format=torch.channels_last)
        for p in t.parameters():
            p.requires_grad = False
        teachers.append(t)
        print(f'  teacher {short:<11} {name:<40} train input={cfg["input_size"]} '
              f'test input={cfg.get("test_input_size", cfg["input_size"])}', flush=True)
    return teachers


def save_ckpt(path, model, optimizer, epoch, step, global_step):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,              # current epoch
        'step': step,                # optimizer steps already done inside `epoch`
        'global_step': global_step,
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'rng_numpy': np.random.get_state(),
        'rng_python': random.getstate(),
    }
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: never leaves a half-written checkpoint


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    names = ['cls'] + [s for s, _ in TEACHERS] + ['avg']
    correct = torch.zeros(len(names))
    total = 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype):
            cls_logits, dist_logits = model(images)
        logits = [cls_logits.float()] + [d.float() for d in dist_logits]
        logits.append(torch.stack(logits).mean(0))
        for i, lg in enumerate(logits):
            correct[i] += (lg.argmax(1) == targets).sum().item()
        total += targets.numel()
    model.train()
    return {n: 100.0 * c.item() / total for n, c in zip(names, correct)}


def benchmark(teachers, device, amp_dtype, micro_batches, n_steps=20):
    """Throughput with synthetic data on the GPU (no dataloader)."""
    print('\n=== Benchmark: GPU only (synthetic data) ===', flush=True)
    for mb in micro_batches:
        model = vit_hydra_base_patch16_224().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(mb, 3, 224, 224, device=device).contiguous(memory_format=torch.channels_last)
        y = torch.randint(0, 1000, (mb,), device=device)
        torch.cuda.reset_peak_memory_stats()
        try:
            for i in range(n_steps + 5):
                if i == 5:
                    torch.cuda.synchronize(); t0 = time.time()
                with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype):
                    t_logits = [t(x) for t in teachers]
                with torch.autocast(device.type, dtype=amp_dtype):
                    cls_logits, dist_logits = model(x)
                loss = F.cross_entropy(cls_logits.float(), y) + sum(
                    kd_loss(d.float(), tl.float(), 3.0) for d, tl in zip(dist_logits, t_logits))
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            ips = mb * n_steps / (time.time() - t0)
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f'  micro-batch {mb:4d}: {ips:7.1f} img/s  peak mem {mem:5.1f} GB  '
                  f'-> 1 epoch {1281167 / ips / 3600:.2f} h, 300 epochs {300 * 1281167 / ips / 86400:.1f} days',
                  flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f'  micro-batch {mb:4d}: OOM', flush=True)
        del model, opt, x, y
        torch.cuda.empty_cache()

    mb = micro_batches[0]
    x = torch.randn(mb, 3, 224, 224, device=device).contiguous(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype):
        for (short, _), t in zip(TEACHERS, teachers):
            for _ in range(3):
                t(x)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(n_steps):
                t(x)
            torch.cuda.synchronize()
            print(f'  teacher {short:<11} forward: {mb * n_steps / (time.time() - t0):7.1f} img/s', flush=True)


@torch.no_grad()
def teacher_stats(teachers, dataset, device, amp_dtype, T, micro_batch, num_workers, n_batches=20):
    """Per-teacher softmax statistics on real val images (are the 5 KD targets comparable?).

    entropy@T     mean entropy of softmax(logits / T), max = ln(1000) = 6.91
    KD(uniform)   T^2 * KL(teacher_T || uniform) = T^2 * (ln 1000 - entropy@T): the KD loss
                  of an untrained student with ~uniform output — starting point of each KD term
    """
    print(f'\n=== Teacher stats ({n_batches * micro_batch} val images, T={T}) ===', flush=True)
    g = torch.Generator()
    g.manual_seed(0)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=num_workers, pin_memory=True,
                        sampler=torch.randperm(len(dataset), generator=g)[:n_batches * micro_batch].tolist())
    stats = torch.zeros(len(teachers), 3)  # correct, entropy sum, max-logit sum
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype):
            logits = [t(images).float() for t in teachers]
        for i, lg in enumerate(logits):
            logp = F.log_softmax(lg / T, dim=1)
            stats[i, 0] += (lg.argmax(1) == targets).sum().item()
            stats[i, 1] += -(logp.exp() * logp).sum(1).sum().item()
            stats[i, 2] += lg.max(1).values.sum().item()
        total += targets.numel()
    for (short, _), (correct, ent, maxlog) in zip(TEACHERS, stats.tolist()):
        ent /= total
        print(f'  {short:<11} top1={100 * correct / total:5.1f}%  entropy@T={ent:.3f}  '
              f'KD(uniform)={T * T * (np.log(1000) - ent):6.3f}  mean max-logit={maxlog / total:6.2f}', flush=True)


def benchmark_loader(dataset, micro_batch, num_workers, n_batches=40):
    print(f'\n=== Benchmark: dataloader ({num_workers} workers, train transforms) ===', flush=True)
    loader = DataLoader(dataset, batch_size=micro_batch, shuffle=True, num_workers=num_workers,
                        pin_memory=True, drop_last=True)
    it = iter(loader)
    for _ in range(3):
        next(it)
    t0 = time.time()
    for _ in range(n_batches):
        next(it)
    ips = micro_batch * n_batches / (time.time() - t0)
    print(f'  {ips:.1f} img/s ({ips / max(num_workers, 1):.1f} img/s per worker)\n', flush=True)
    del it, loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',          default=f'/var/local/storage/{os.environ.get("USER", "")}/imagenet1k')
    parser.add_argument('--output-dir',    default='results/pretraining/vit_hydra_full')
    parser.add_argument('--epochs',        type=int,   default=300, help='total epochs (defines the LR schedule)')
    parser.add_argument('--run-epochs',    type=int,   default=None, help='epochs to finish in this session')
    parser.add_argument('--batch-size',    type=int,   default=1024, help='global batch (grad accumulation)')
    parser.add_argument('--micro-batch',   type=int,   default=128)
    parser.add_argument('--lr',            type=float, default=None, help='default: 5e-4 * batch / 512 (DeiT)')
    parser.add_argument('--min-lr',        type=float, default=1e-5)
    parser.add_argument('--warmup-epochs', type=int,   default=5)
    parser.add_argument('--warmup-lr',     type=float, default=1e-6)
    parser.add_argument('--weight-decay',  type=float, default=0.05)
    parser.add_argument('--drop-path',     type=float, default=0.1)
    parser.add_argument('--repeats',       type=int,   default=3, help='repeated augmentation')
    parser.add_argument('--T',             type=float, default=1.0)
    parser.add_argument('--w-cls',         type=float, default=0.5)
    parser.add_argument('--w-kd',          type=float, default=0.1, help='weight of each teacher KD term')
    parser.add_argument('--num-workers',   type=int,   default=2)
    parser.add_argument('--eval-every',    type=int,   default=10, help='epochs (always at the last epoch)')
    parser.add_argument('--ckpt-minutes',  type=float, default=30)
    parser.add_argument('--log-every',     type=int,   default=50, help='optimizer steps')
    parser.add_argument('--seed',          type=int,   default=0)
    parser.add_argument('--resume',        action='store_true')
    parser.add_argument('--benchmark',     action='store_true', help='run throughput benchmarks first')
    args = parser.parse_args()

    assert args.batch_size % args.micro_batch == 0
    accum = args.batch_size // args.micro_batch
    lr = args.lr or 5e-4 * args.batch_size / 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, 'checkpoint.pth')
    signal.signal(signal.SIGUSR1, _on_sigusr1)

    print(f'torch {torch.__version__}  timm {timm.__version__}  device {device}'
          f'{" (" + torch.cuda.get_device_name() + ")" if device.type == "cuda" else ""}', flush=True)
    print(f'args: {vars(args)}', flush=True)
    print(f'lr {lr:.2e}, global batch {args.batch_size} = {accum} x {args.micro_batch}', flush=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print('Loading teachers...', flush=True)
    teachers = build_teachers(device)
    train_ds, val_ds = build_datasets(args.data)
    print(f'train {len(train_ds)} images, val {len(val_ds)} images, {len(train_ds.classes)} classes', flush=True)

    if args.benchmark:
        teacher_stats(teachers, val_ds, device, amp_dtype, args.T, args.micro_batch, args.num_workers)
    if args.benchmark and device.type == 'cuda':
        benchmark(teachers, device, amp_dtype, [args.micro_batch, 2 * args.micro_batch])
        benchmark_loader(train_ds, args.micro_batch, args.num_workers)

    model = vit_hydra_base_patch16_224(drop_path_rate=args.drop_path).to(device)
    optimizer = create_optimizer_v2(model, 'adamw', lr=lr, weight_decay=args.weight_decay)

    sampler = RepeatAugSampler(len(train_ds), args.repeats, args.batch_size, args.seed)
    steps_per_epoch = sampler.num_samples // args.batch_size
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs * steps_per_epoch, lr_min=args.min_lr,
        warmup_t=args.warmup_epochs * steps_per_epoch, warmup_lr_init=args.warmup_lr,
        t_in_epochs=False)
    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch',
                  label_smoothing=0.1, num_classes=1000)
    soft_ce = SoftTargetCrossEntropy()

    start_epoch, start_step, global_step = 0, 0, 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch, start_step, global_step = ckpt['epoch'], ckpt['step'], ckpt['global_step']
        torch.set_rng_state(ckpt['rng_torch'])
        if ckpt['rng_cuda'] is not None and device.type == 'cuda':
            torch.cuda.set_rng_state(ckpt['rng_cuda'])
        np.random.set_state(ckpt['rng_numpy'])
        random.setstate(ckpt['rng_python'])
        print(f'Resumed: epoch {start_epoch}, step {start_step}/{steps_per_epoch}, global step {global_step}',
              flush=True)

    end_epoch = min(start_epoch + args.run_epochs, args.epochs) if args.run_epochs else args.epochs
    if start_epoch >= end_epoch:
        print('Nothing to do: target epochs already reached.', flush=True)
        return
    print(f'Training epochs {start_epoch}->{end_epoch - 1} (total {args.epochs}), '
          f'{steps_per_epoch} steps/epoch', flush=True)

    val_loader = DataLoader(val_ds, batch_size=args.micro_batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    head_names = ['cls'] + [s for s, _ in TEACHERS]
    model.train()
    last_ckpt = time.time()

    for epoch in range(start_epoch, end_epoch):
        sampler.set_epoch(epoch, start=start_step * args.batch_size)
        loader = DataLoader(train_ds, batch_size=args.micro_batch, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        it = iter(loader)
        step = start_step
        scheduler.step_update(global_step)
        meter = torch.zeros(len(head_names))
        n_meter, data_time, t_log = 0, 0.0, time.time()

        while step < steps_per_epoch:
            for _ in range(accum):
                t0 = time.time()
                images, targets = next(it)
                data_time += time.time() - t0
                images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
                images, soft_targets = mixup(images, targets)

                with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype):
                    t_logits = [t(images) for t in teachers]
                with torch.autocast(device.type, dtype=amp_dtype):
                    cls_logits, dist_logits = model(images)

                ce = soft_ce(cls_logits.float(), soft_targets)
                kds = [kd_loss(d.float(), tl.float(), args.T) for d, tl in zip(dist_logits, t_logits)]
                loss = args.w_cls * ce + args.w_kd * sum(kds)
                (loss / accum).backward()

                meter += torch.tensor([ce.item()] + [k.item() for k in kds])
                n_meter += 1

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            global_step += 1
            scheduler.step_update(global_step)

            if step % args.log_every == 0 or step == steps_per_epoch:
                dt = time.time() - t_log
                ips = n_meter * args.micro_batch / dt
                losses = ' '.join(f'{n}={v:.4f}' for n, v in zip(head_names, (meter / n_meter).tolist()))
                print(f'[ep {epoch} step {step}/{steps_per_epoch}] {losses} '
                      f'lr={optimizer.param_groups[0]["lr"]:.2e} {ips:.1f} img/s data_wait={100 * data_time / dt:.0f}%',
                      flush=True)
                meter.zero_(); n_meter, data_time, t_log = 0, 0.0, time.time()

            if STOP['flag'] or time.time() - last_ckpt > 60 * args.ckpt_minutes:
                save_ckpt(ckpt_path, model, optimizer, epoch, step, global_step)
                last_ckpt = time.time()
                print(f'Checkpoint saved: epoch {epoch} step {step}', flush=True)
                if STOP['flag']:
                    print('SIGUSR1 received (time limit) — exiting, resume in next job.', flush=True)
                    return

        del it, loader
        start_step = 0
        save_ckpt(ckpt_path, model, optimizer, epoch + 1, 0, global_step)
        last_ckpt = time.time()
        print(f'Epoch {epoch} done — checkpoint saved', flush=True)

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            acc = evaluate(model, val_loader, device, amp_dtype)
            print(f'[ep {epoch} val] ' + ' '.join(f'{n}={v:.2f}' for n, v in acc.items()), flush=True)

        if STOP['flag']:
            print('SIGUSR1 received (time limit) — exiting, resume in next job.', flush=True)
            return

    if end_epoch == args.epochs:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'vit_hydra_final.pth'))
        print('Saved final weights.', flush=True)


if __name__ == '__main__':
    main()
