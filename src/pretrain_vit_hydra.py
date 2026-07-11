#!/usr/bin/env python3
"""Mini-pretraining of VitHydra MLP heads using two teacher networks.

Freezes DeiT backbone and trains:
  mlp_cls  + fc_cls  <- soft KD from Teacher A (convnext_base)
  mlp_dist + fc_dist <- soft KD from Teacher B (swin_base)

Loss: DeiT soft distillation (KL with log_target=True, T^2 scaling). No CE term.

Usage:
  python pretrain_vit_hydra.py --data /opt/storage/datasets/imagenet1k --output mlp_weights.pth
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
import networks.deit_original  # registers deit_base_distilled_patch16_224_cil
from networks.vit_hydra import VitHydra


TEACHER_A = 'convnext_base.fb_in1k'
TEACHER_B = 'swin_base_patch4_window7_224.ms_in1k'
BACKBONE  = 'deit_base_distilled_patch16_224_cil'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def kd_loss(student_logits, teacher_logits, T):
    """DeiT soft distillation loss (facebookresearch/deit, losses.py)."""
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.log_softmax(teacher_logits / T, dim=1),
        reduction='sum',
        log_target=True,
    ) * (T * T) / student_logits.numel()


def build_loader(data_path, batch_size, num_workers):
    from datasets import load_dataset

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    ds = load_dataset(
        'parquet',
        data_files=os.path.join(data_path, 'data', 'train-*.parquet'),
        split='train',
    )

    def apply_transform(batch):
        batch['pixel_values'] = [transform(img.convert('RGB')) for img in batch['image']]
        return batch

    ds.set_transform(apply_transform)

    def collate_fn(samples):
        images = torch.stack([s['pixel_values'] for s in samples])
        labels = torch.tensor([s['label'] for s in samples])
        return images, labels

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',        default='/opt/storage/datasets/imagenet1k')
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--batch-size',  type=int,   default=256)
    parser.add_argument('--num-workers', type=int,   default=2)
    parser.add_argument('--T',           type=float, default=3.0)
    parser.add_argument('--output',      default='mlp_weights.pth')
    parser.add_argument('--resume',      action='store_true')
    parser.add_argument('--run-epochs',  type=int, default=None,
                        help='epochs to run this session (default: run to --epochs)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Teachers (frozen)
    print('Loading teachers...')
    teacher_a = timm.create_model(TEACHER_A, pretrained=True).to(device).eval()
    teacher_b = timm.create_model(TEACHER_B, pretrained=True).to(device).eval()
    for p in teacher_a.parameters(): p.requires_grad = False
    for p in teacher_b.parameters(): p.requires_grad = False

    # Backbone + VitHydra
    print('Loading backbone...')
    backbone = timm.create_model(BACKBONE, pretrained=True)
    model = VitHydra(backbone, teacher_out_dim=1000).to(device)
    model.freeze_backbone()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable / 1e6:.1f}M (mlp_cls + mlp_dist + fc_cls + fc_dist)')

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.05,
    )

    ckpt_path = args.output + '.ckpt'
    start_epoch = 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.mlp_cls.load_state_dict(ckpt['mlp_cls'])
        model.mlp_dist.load_state_dict(ckpt['mlp_dist'])
        model.fc_cls.load_state_dict(ckpt['fc_cls'])
        model.fc_dist.load_state_dict(ckpt['fc_dist'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')

    # Dataset
    print('Building dataloader...')
    loader = build_loader(args.data, args.batch_size, args.num_workers)

    T = args.T
    end_epoch = min(start_epoch + args.run_epochs, args.epochs) if args.run_epochs else args.epochs
    print(f'Starting pretraining — epochs {start_epoch}→{end_epoch-1} (total={args.epochs}), T={T}, lr={args.lr}')

    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss = 0.0

        for i, (images, _) in enumerate(loader):
            images = images.to(device)

            with torch.no_grad():
                logits_a = teacher_a(images)
                logits_b = teacher_b(images)

            out  = model(images)
            loss = kd_loss(out['fc_cls_logits'],  logits_a, T) + \
                   kd_loss(out['fc_dist_logits'], logits_b, T)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 200 == 0:
                print(f'  [{epoch}/{args.epochs}][{i}/{len(loader)}] loss: {loss.item():.4f}')

        print(f'Epoch {epoch} — avg loss: {total_loss / len(loader):.4f}')
        torch.save({
            'epoch':      epoch,
            'mlp_cls':    model.mlp_cls.state_dict(),
            'mlp_dist':   model.mlp_dist.state_dict(),
            'fc_cls':     model.fc_cls.state_dict(),
            'fc_dist':    model.fc_dist.state_dict(),
            'optimizer':  optimizer.state_dict(),
        }, ckpt_path)

    model.save_mlps(args.output)
    print(f'Saved MLP weights → {args.output}')


if __name__ == '__main__':
    main()
