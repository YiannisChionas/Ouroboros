#!/usr/bin/env python3
"""vit-orth pretraining on the 200 mammal classes of iNat-500 (= iNatM-200), single GPU.

ViT-S/16 from scratch with 1 classification token + 1 distillation token (networks/vit_orth.py),
DeiT-S recipe, both heads trained on ground truth, no teacher:

    min  CE(cls) + CE(dist)
    s.t. g_cc  = within-class cross-correlation(z_cls, z_dist) <= eps_cc
         g_var = within-class std floor hinge of each token       <= 0

Within-class residuals r_k = z_k - M_k[y]: M_k are running class means (EMA) of each token's
final-norm features. The constraints are computed only on clean (not mixed) batches, so that each
image has a single class (Mixup/CutMix with probability --mix-prob per batch, same in every run).

    g_cc  = mean_ij C_ij^2,   C = std(r_cls)^T std(r_dist) / B   (D x D, standardized over the batch)
    g_var = mean_d relu(var_min - std_d(r_k)), summed over the two tokens

Modes (--mode):
    none        CE only (run A). g_cc / g_var are only measured.
    penalty     CE + w_cc * g_cc + w_var * g_var (run B, fixed weights)
    lagrangian  augmented Lagrangian (run C):
                    L = CE + mu_cc * g_cc + rho/2 relu(g_cc - eps_cc)^2 + mu_var * g_var + rho/2 g_var^2
                dual ascent once per epoch on the epoch mean of g (clean batches):
                    mu <- clip(mu + dual_lr * (lr_t / lr_max) * (mean g - eps), 0, mu_max)
In penalty / lagrangian the constraint terms are off for the first --constraint-warmup epochs.

Evaluation (every --eval-every epochs, on the 200-class test set): top-1 of each head and of their
concatenation, within-class cc, within-class std, cos, linear / RBF CKA, linear and MLP probe R^2.
At the end (or with --eval-only): SimpleCIL on classes 200-499 (prototypes from their train images,
nearest prototype on their test images) with z_cls, z_dist and [z_cls; z_dist].

Single process, global batch via gradient accumulation. Resumable mid-epoch:
  - checkpoint every --ckpt-minutes and on SIGUSR1 (sbatch --signal=B:USR1@300)
  - deterministic per-epoch sample order (seed + epoch), resumed at the saved step
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
from PIL import Image
from timm.data import create_transform, Mixup, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.mixup import mixup_target
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from networks.vit_orth import vit_orth_small_patch16_224
from pretrain_vit_hydra_full import RepeatAugSampler

STOP = {'flag': False}


def _on_sigusr1(signum, frame):
    STOP['flag'] = True


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

class ListDataset(Dataset):
    """Images from a `<relative_path> <class_id>` list with lo <= class_id < hi, labels shifted by -lo."""

    def __init__(self, root, list_file, lo, hi, transform):
        self.samples = []
        with open(os.path.join(root, list_file)) as f:
            for line in f:
                path, label = line.split()
                if lo <= int(label) < hi:
                    self.samples.append((os.path.join(root, path), int(label) - lo))
        self.transform = transform
        self.num_classes = len({y for _, y in self.samples})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        return self.transform(Image.open(path).convert('RGB')), y


def train_transform():
    # DeiT build_transform (is_train=True)
    return create_transform(
        224, is_training=True, color_jitter=0.3, auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic', re_prob=0.25, re_mode='pixel', re_count=1)


def test_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


# ----------------------------------------------------------------------------
# Constraints
# ----------------------------------------------------------------------------

def standardize(z, eps=1e-5):
    return (z - z.mean(0)) / (z.std(0) + eps)


def cross_corr(r1, r2):
    """Mean squared cross-correlation between two [B, D] feature sets (float32)."""
    r1, r2 = standardize(r1.float()), standardize(r2.float())
    return (r1.T @ r2 / r1.shape[0]).pow(2).mean()


def std_hinge(r, var_min, eps=1e-4):
    """mean_d relu(var_min - std_d(r)) for a [B, D] residual (float32)."""
    return F.relu(var_min - (r.float().var(0) + eps).sqrt()).mean()


class ClassMeans:
    """Running (EMA) class means of the two tokens' features, [2, C, D] float32 on the device."""

    def __init__(self, num_classes, dim, momentum, device):
        self.m = momentum
        self.means = torch.zeros(2, num_classes, dim, device=device)
        self.seen = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, z, y):
        """z: [2, B, D] detached features, y: [B] hard labels."""
        z = z.float()
        counts = torch.bincount(y, minlength=self.seen.numel())
        present = counts > 0
        sums = torch.zeros_like(self.means).index_add_(1, y, z)
        batch_mean = sums[:, present] / counts[present].unsqueeze(1)
        old = self.means[:, present]
        seen = self.seen[present].view(1, -1, 1)
        self.means[:, present] = torch.where(seen, self.m * old + (1 - self.m) * batch_mean, batch_mean)
        self.seen |= present

    def residuals(self, z, y):
        return z.float() - self.means[:, y]

    def state_dict(self):
        return {'means': self.means.cpu(), 'seen': self.seen.cpu()}

    def load_state_dict(self, s):
        self.means.copy_(s['means']); self.seen.copy_(s['seen'])


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def extract(model, loader, device, amp_dtype):
    """Final-norm token features [N, D] x 2, logits [N, C] x 2, labels [N]."""
    model.eval()
    zc, zd, lc, ld, ys = [], [], [], [], []
    for images, targets in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            tokens = model.forward_tokens(images.to(device, non_blocking=True))
            cls_logits, dist_logits = model.forward_head(tokens)
        zc.append(tokens[:, 0].float().cpu()); zd.append(tokens[:, 1].float().cpu())
        lc.append(cls_logits.float().cpu()); ld.append(dist_logits[0].float().cpu())
        ys.append(targets)
    model.train()
    return torch.cat(zc), torch.cat(zd), torch.cat(lc), torch.cat(ld), torch.cat(ys)


def class_residuals(z, y):
    """z - (exact) class mean of z, per class."""
    r = z.clone()
    for c in y.unique():
        r[y == c] -= z[y == c].mean(0)
    return r


def linear_cka(x, y):
    x = x - x.mean(0); y = y - y.mean(0)
    return ((x.T @ y).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm())).item()


def rbf_cka(x, y, n=2000, seed=0):
    """RBF-kernel CKA on n random samples, bandwidth = median pairwise distance (float64)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:n]
    x, y = x[idx].double(), y[idx].double()

    def gram(a):
        d2 = torch.cdist(a, a).pow(2)
        sigma2 = d2[d2 > 0].median()
        k = torch.exp(-d2 / (2 * sigma2))
        h = torch.eye(k.shape[0], dtype=k.dtype) - 1.0 / k.shape[0]
        return h @ k @ h

    kx, ky = gram(x), gram(y)
    return ((kx * ky).sum() / (kx.norm() * ky.norm())).item()


def r2(pred, target):
    return (1 - (pred - target).pow(2).sum() / (target - target.mean(0)).pow(2).sum()).item()


def probe_r2(x, y, device, seed=0, mlp_steps=500):
    """R^2 of predicting y from x on a held-out half: ridge (linear) and a 2-layer ReLU MLP."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)
    tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
    mu_x, sd_x = x[tr].mean(0), x[tr].std(0) + 1e-5
    xs = (x - mu_x) / sd_x
    # ridge, closed form
    xtr = torch.cat([xs[tr], torch.ones(len(tr), 1)], 1).double()
    xte = torch.cat([xs[te], torch.ones(len(te), 1)], 1).double()
    w = torch.linalg.solve(xtr.T @ xtr + 1e-2 * torch.eye(xtr.shape[1], dtype=xtr.dtype), xtr.T @ y[tr].double())
    lin = r2((xte @ w).float(), y[te])
    # MLP, full-batch Adam
    torch.manual_seed(seed)
    d = x.shape[1]
    mlp = torch.nn.Sequential(torch.nn.Linear(d, 2 * d), torch.nn.ReLU(), torch.nn.Linear(2 * d, y.shape[1])).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    a, b = xs[tr].to(device), y[tr].to(device)
    with torch.enable_grad():
        for _ in range(mlp_steps):
            loss = F.mse_loss(mlp(a), b)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mlp_r2 = r2(mlp(xs[te].to(device)).cpu(), y[te])
    return lin, mlp_r2


def evaluate(model, loader, device, amp_dtype):
    zc, zd, lc, ld, y = extract(model, loader, device, amp_dtype)
    acc = {n: 100.0 * (lg.argmax(1) == y).float().mean().item()
           for n, lg in [('cls', lc), ('dist', ld), ('concat', torch.maximum(lc, ld))]}
    rc, rd = class_residuals(zc, y), class_residuals(zd, y)
    lin_cd, mlp_cd = probe_r2(zc, zd, device)
    lin_dc, mlp_dc = probe_r2(zd, zc, device)
    div = {
        'wcc': cross_corr(rc, rd).item(),                       # within-class cross-correlation
        'cc': cross_corr(zc, zd).item(),                        # total cross-correlation
        'wstd_cls': rc.std(0).mean().item(), 'wstd_dist': rd.std(0).mean().item(),
        'wstd_min_cls': rc.std(0).min().item(), 'wstd_min_dist': rd.std(0).min().item(),
        'cos': F.cosine_similarity(zc, zd, dim=1).mean().item(),
        'cka_lin': linear_cka(zc.double(), zd.double()), 'cka_rbf': rbf_cka(zc, zd),
        'wcka_lin': linear_cka(rc.double(), rd.double()), 'wcka_rbf': rbf_cka(rc, rd),
        'r2_lin_c2d': lin_cd, 'r2_mlp_c2d': mlp_cd, 'r2_lin_d2c': lin_dc, 'r2_mlp_d2c': mlp_dc,
    }
    return acc, div


def simplecil(model, data_dir, lo, hi, micro_batch, num_workers, device, amp_dtype):
    """Nearest-prototype (cosine) top-1 on classes [lo, hi): z_cls, z_dist, [z_cls; z_dist]."""
    def loader(list_file):
        return DataLoader(ListDataset(data_dir, list_file, lo, hi, test_transform()), batch_size=micro_batch,
                          shuffle=False, num_workers=num_workers, pin_memory=True)
    trc, trd, _, _, ytr = extract(model, loader('train.txt'), device, amp_dtype)
    tec, ted, _, _, yte = extract(model, loader('test.txt'), device, amp_dtype)
    out = {}
    for name, ftr, fte in [('cls', trc, tec), ('dist', trd, ted),
                           ('concat', torch.cat([F.normalize(trc, dim=1), F.normalize(trd, dim=1)], 1),
                            torch.cat([F.normalize(tec, dim=1), F.normalize(ted, dim=1)], 1))]:
        ftr, fte = F.normalize(ftr, dim=1), F.normalize(fte, dim=1)
        protos = F.normalize(torch.stack([ftr[ytr == c].mean(0) for c in range(hi - lo)]), dim=1)
        out[name] = 100.0 * ((fte @ protos.T).argmax(1) == yte).float().mean().item()
    return out


def fmt(d):
    return ' '.join(f'{k}={v:.4f}' if abs(v) < 10 else f'{k}={v:.2f}' for k, v in d.items())


def run_eval(tag, model, val_loader, args, device, amp_dtype, with_simplecil=False):
    acc, div = evaluate(model, val_loader, device, amp_dtype)
    print(f'[{tag} val] {fmt(acc)}', flush=True)
    print(f'[{tag} div] {fmt(div)}', flush=True)
    if with_simplecil:
        sc = simplecil(model, args.data, args.num_classes, args.simplecil_hi, args.micro_batch,
                       args.num_workers, device, amp_dtype)
        print(f'[{tag} simplecil {args.num_classes}-{args.simplecil_hi - 1}] {fmt(sc)}', flush=True)


# ----------------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------------

def save_ckpt(path, model, optimizer, epoch, step, global_step, extra):
    state = {
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'epoch': epoch, 'step': step, 'global_step': global_step, 'extra': extra,
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'rng_numpy': np.random.get_state(), 'rng_python': random.getstate(),
    }
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',          default=f'/var/local/storage/{os.environ.get("USER", "")}/inat500')
    p.add_argument('--num-classes',   type=int,   default=200, help='class_id < num_classes are pretrained on')
    p.add_argument('--simplecil-hi',  type=int,   default=500, help='SimpleCIL on classes [num_classes, hi)')
    p.add_argument('--output-dir',    required=True)
    p.add_argument('--mode',          required=True, choices=['none', 'penalty', 'lagrangian'])
    # constraints
    p.add_argument('--eps-cc',        type=float, default=None, help='default 2 / micro-batch (noise floor x 2)')
    p.add_argument('--var-min',       type=float, default=0.0, help='within-class std floor (0 = no variance constraint)')
    p.add_argument('--w-cc',          type=float, default=1.0, help='penalty mode')
    p.add_argument('--w-var',         type=float, default=1.0, help='penalty mode')
    p.add_argument('--dual-lr',       type=float, default=10.0)
    p.add_argument('--rho',           type=float, default=10.0)
    p.add_argument('--mu-max',        type=float, default=100.0, help='safety bound; reaching it = constraint failed')
    p.add_argument('--constraint-warmup', type=int, default=3, help='epochs with the constraint terms off')
    p.add_argument('--ema',           type=float, default=0.99, help='class-mean EMA momentum')
    p.add_argument('--mix-prob',      type=float, default=0.5, help='probability of Mixup/CutMix per batch')
    # recipe
    p.add_argument('--w-ce',          type=float, default=0.5, help='weight of each head CE')
    p.add_argument('--epochs',        type=int,   default=300, help='total epochs (defines the LR schedule)')
    p.add_argument('--run-epochs',    type=int,   default=None, help='epochs to finish in this session')
    p.add_argument('--batch-size',    type=int,   default=1024, help='global batch (grad accumulation)')
    p.add_argument('--micro-batch',   type=int,   default=256)
    p.add_argument('--lr',            type=float, default=None, help='default: 5e-4 * batch / 512 (DeiT)')
    p.add_argument('--min-lr',        type=float, default=1e-5)
    p.add_argument('--warmup-epochs', type=int,   default=5)
    p.add_argument('--warmup-lr',     type=float, default=1e-6)
    p.add_argument('--weight-decay',  type=float, default=0.05)
    p.add_argument('--drop-path',     type=float, default=0.1)
    p.add_argument('--repeats',       type=int,   default=3, help='repeated augmentation')
    p.add_argument('--num-workers',   type=int,   default=3)
    p.add_argument('--eval-every',    type=int,   default=10, help='epochs (always at the last epoch)')
    p.add_argument('--ckpt-minutes',  type=float, default=30)
    p.add_argument('--log-every',     type=int,   default=20, help='optimizer steps')
    p.add_argument('--max-steps',     type=int,   default=None, help='stop after this many optimizer steps (smoke test)')
    p.add_argument('--seed',          type=int,   default=0)
    p.add_argument('--resume',        action='store_true')
    p.add_argument('--eval-only',     action='store_true', help='load checkpoint, run evaluation + SimpleCIL, exit')
    p.add_argument('--benchmark',     action='store_true', help='dataloader throughput before training')
    args = p.parse_args()

    assert args.batch_size % args.micro_batch == 0
    accum = args.batch_size // args.micro_batch
    lr = args.lr or 5e-4 * args.batch_size / 512
    eps_cc = args.eps_cc if args.eps_cc is not None else 2.0 / args.micro_batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, 'checkpoint.pth')
    signal.signal(signal.SIGUSR1, _on_sigusr1)

    print(f'torch {torch.__version__}  device {device}'
          f'{" (" + torch.cuda.get_device_name() + ")" if device.type == "cuda" else ""}', flush=True)
    print(f'args: {vars(args)}', flush=True)
    print(f'lr {lr:.2e}, global batch {args.batch_size} = {accum} x {args.micro_batch}, eps_cc {eps_cc:.4f}', flush=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    train_ds = ListDataset(args.data, 'train.txt', 0, args.num_classes, train_transform())
    val_ds = ListDataset(args.data, 'test.txt', 0, args.num_classes, test_transform())
    assert train_ds.num_classes == val_ds.num_classes == args.num_classes
    print(f'train {len(train_ds)} images, test {len(val_ds)} images, {train_ds.num_classes} classes', flush=True)
    val_loader = DataLoader(val_ds, batch_size=args.micro_batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    if args.benchmark:
        loader = DataLoader(train_ds, batch_size=args.micro_batch, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)
        it = iter(loader); next(it); next(it)
        t0 = time.time()
        for _ in range(10):
            next(it)
        print(f'[benchmark] dataloader {10 * args.micro_batch / (time.time() - t0):.1f} img/s '
              f'({args.num_workers} workers)', flush=True)
        del it, loader

    model = vit_orth_small_patch16_224(num_classes=args.num_classes, drop_path_rate=args.drop_path).to(device)
    optimizer = create_optimizer_v2(model, 'adamw', lr=lr, weight_decay=args.weight_decay)
    means = ClassMeans(args.num_classes, model.embed_dim, args.ema, device)

    sampler = RepeatAugSampler(len(train_ds), args.repeats, args.batch_size, args.seed)
    steps_per_epoch = sampler.num_samples // args.batch_size
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs * steps_per_epoch, lr_min=args.min_lr,
        warmup_t=args.warmup_epochs * steps_per_epoch, warmup_lr_init=args.warmup_lr,
        t_in_epochs=False)
    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch',
                  label_smoothing=0.1, num_classes=args.num_classes)
    soft_ce = SoftTargetCrossEntropy()

    # dual state: multipliers + running sums of g over the clean micro-batches of the current epoch
    extra = {'mu_cc': 0.0, 'mu_var': 0.0, 'sum_cc': 0.0, 'sum_var': 0.0, 'n_clean': 0, 'means': None}

    start_epoch, start_step, global_step = 0, 0, 0
    if (args.resume or args.eval_only) and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch, start_step, global_step = ckpt['epoch'], ckpt['step'], ckpt['global_step']
        extra = ckpt['extra']
        means.load_state_dict(extra['means'])
        torch.set_rng_state(ckpt['rng_torch'])
        if ckpt['rng_cuda'] is not None and device.type == 'cuda':
            torch.cuda.set_rng_state(ckpt['rng_cuda'])
        np.random.set_state(ckpt['rng_numpy'])
        random.setstate(ckpt['rng_python'])
        print(f'Resumed: epoch {start_epoch}, step {start_step}/{steps_per_epoch}, global step {global_step}, '
              f'mu_cc {extra["mu_cc"]:.4f} mu_var {extra["mu_var"]:.4f}', flush=True)

    if args.eval_only:
        run_eval(f'ep {start_epoch - 1}', model, val_loader, args, device, amp_dtype, with_simplecil=True)
        return

    end_epoch = min(start_epoch + args.run_epochs, args.epochs) if args.run_epochs else args.epochs
    if start_epoch >= end_epoch:
        print('Nothing to do: target epochs already reached.', flush=True)
        return
    print(f'Training epochs {start_epoch}->{end_epoch - 1} (total {args.epochs}), '
          f'{steps_per_epoch} steps/epoch, mode {args.mode}', flush=True)

    def checkpoint(epoch, step):
        extra['means'] = means.state_dict()
        save_ckpt(ckpt_path, model, optimizer, epoch, step, global_step, extra)

    model.train()
    last_ckpt = time.time()

    for epoch in range(start_epoch, end_epoch):
        sampler.set_epoch(epoch, start=start_step * args.batch_size)
        loader = DataLoader(train_ds, batch_size=args.micro_batch, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        it = iter(loader)
        step = start_step
        scheduler.step_update(global_step)
        constrained = args.mode != 'none' and epoch >= args.constraint_warmup
        meter = torch.zeros(4)  # ce_cls, ce_dist, g_cc, g_var (g on clean batches only)
        n_meter, n_clean_log, data_time, t_log = 0, 0, 0.0, time.time()

        while step < steps_per_epoch:
            for _ in range(accum):
                t0 = time.time()
                images, targets = next(it)
                data_time += time.time() - t0
                images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
                clean = np.random.rand() >= args.mix_prob
                if clean:
                    soft_targets = mixup_target(targets, args.num_classes, lam=1.0, smoothing=0.1)
                else:
                    images, soft_targets = mixup(images, targets)

                with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    tokens = model.forward_tokens(images)
                    cls_logits, dist_logits = model.forward_head(tokens)

                ce_c = soft_ce(cls_logits.float(), soft_targets)
                ce_d = soft_ce(dist_logits[0].float(), soft_targets)
                loss = args.w_ce * (ce_c + ce_d)
                g_cc = g_var = torch.zeros((), device=device)

                if clean:
                    z = torch.stack([tokens[:, 0], tokens[:, 1]])          # [2, B, D]
                    r = means.residuals(z, targets)
                    g_cc = cross_corr(r[0], r[1])
                    if args.var_min > 0:
                        g_var = std_hinge(r[0], args.var_min) + std_hinge(r[1], args.var_min)
                    if constrained and args.mode == 'penalty':
                        loss = loss + args.w_cc * g_cc + args.w_var * g_var
                    elif constrained and args.mode == 'lagrangian':
                        loss = (loss + extra['mu_cc'] * g_cc + args.rho / 2 * F.relu(g_cc - eps_cc) ** 2
                                + extra['mu_var'] * g_var + args.rho / 2 * g_var ** 2)
                    means.update(z.detach(), targets)
                    extra['sum_cc'] += g_cc.item(); extra['sum_var'] += g_var.item(); extra['n_clean'] += 1
                    n_clean_log += 1

                (loss / accum).backward()
                meter += torch.tensor([ce_c.item(), ce_d.item(), g_cc.item(), g_var.item()])
                n_meter += 1

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            global_step += 1
            scheduler.step_update(global_step)

            if step % args.log_every == 0 or step == steps_per_epoch:
                dt = time.time() - t_log
                ips = n_meter * args.micro_batch / dt
                m = meter.tolist()
                print(f'[ep {epoch} step {step}/{steps_per_epoch}] ce_cls={m[0] / n_meter:.4f} '
                      f'ce_dist={m[1] / n_meter:.4f} g_cc={m[2] / max(n_clean_log, 1):.4f} '
                      f'g_var={m[3] / max(n_clean_log, 1):.4f} mu_cc={extra["mu_cc"]:.3f} '
                      f'mu_var={extra["mu_var"]:.3f} lr={optimizer.param_groups[0]["lr"]:.2e} '
                      f'{ips:.1f} img/s data_wait={100 * data_time / dt:.0f}%', flush=True)
                meter.zero_(); n_meter, n_clean_log, data_time, t_log = 0, 0, 0.0, time.time()

            if args.max_steps is not None and global_step >= args.max_steps:
                run_eval('max-steps', model, val_loader, args, device, amp_dtype, with_simplecil=True)
                checkpoint(epoch, step)
                print('max-steps reached — checkpoint saved, exiting.', flush=True)
                return

            if STOP['flag'] or time.time() - last_ckpt > 60 * args.ckpt_minutes:
                checkpoint(epoch, step)
                last_ckpt = time.time()
                print(f'Checkpoint saved: epoch {epoch} step {step}', flush=True)
                if STOP['flag']:
                    print('SIGUSR1 received (time limit) — exiting, resume in next job.', flush=True)
                    return

        del it, loader
        start_step = 0

        # dual ascent on the epoch means of g (clean batches)
        n = max(extra['n_clean'], 1)
        mean_cc, mean_var = extra['sum_cc'] / n, extra['sum_var'] / n
        if args.mode == 'lagrangian' and constrained:
            scale = args.dual_lr * optimizer.param_groups[0]['lr'] / lr
            extra['mu_cc'] = float(np.clip(extra['mu_cc'] + scale * (mean_cc - eps_cc), 0, args.mu_max))
            if args.var_min > 0:
                extra['mu_var'] = float(np.clip(extra['mu_var'] + scale * mean_var, 0, args.mu_max))
        print(f'[ep {epoch} dual] mean g_cc={mean_cc:.4f} (eps {eps_cc:.4f}) mean g_var={mean_var:.4f} '
              f'clean batches={extra["n_clean"]} -> mu_cc={extra["mu_cc"]:.4f} mu_var={extra["mu_var"]:.4f}'
              f'{"  [mu_max reached]" if max(extra["mu_cc"], extra["mu_var"]) >= args.mu_max else ""}', flush=True)
        extra.update(sum_cc=0.0, sum_var=0.0, n_clean=0)

        checkpoint(epoch + 1, 0)
        last_ckpt = time.time()
        print(f'Epoch {epoch} done — checkpoint saved', flush=True)

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            run_eval(f'ep {epoch}', model, val_loader, args, device, amp_dtype,
                     with_simplecil=epoch + 1 == args.epochs)

        if STOP['flag']:
            print('SIGUSR1 received (time limit) — exiting, resume in next job.', flush=True)
            return

    if end_epoch == args.epochs:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'vit_orth_final.pth'))
        print('Saved final weights.', flush=True)


if __name__ == '__main__':
    main()
