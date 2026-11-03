"""Head diversity of DeiT-B⚗ + 4 pretrained MLP pseudo-heads (lwf_hydra_v2 setup), no CIL training.

Same metrics / data / transform as analyze_token_diversity.py, on the 6 head INPUT features
the CIL linear heads see in LLL_Net_Hydra_v2:
  cls       = cls token feature                      (IN-1k head: DeiT head)
  dist      = dist token feature                     (IN-1k head: DeiT head_dist)
  mlp_cls   = mlp_cls(cls)    <- ConvNeXt-B teacher   (IN-1k head: fc_cls)
  mlp_dist  = mlp_dist(dist)  <- Swin-B teacher       (IN-1k head: fc_dist)
  mlp_res   = mlp_res(cls)    <- ResNet101 a1 teacher (IN-1k head: fc_res)  [a1 = BCE-trained teacher]
  mlp_eff   = mlp_eff(dist)   <- EfficientNet-B4      (IN-1k head: fc_eff)
MLPs and fc heads from mlp_weights_v2.pth.ckpt (the fc heads are discarded in CIL, used here only for
agreement / JS).
"""
import argparse
import json
import os

import timm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import InterpolationMode

from analyze_token_diversity import pairwise, print_matrix, off_diag_mean
from networks.mlp import MLP

NAMES = ['cls', 'dist', 'mlp_cls', 'mlp_dist', 'mlp_res', 'mlp_eff']


@torch.no_grad()
def collect(deit, mlps, fcs, loader, device):
    feats, logits = [], []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            tok = deit.forward_features(x)[:, :2]
            c, d = tok[:, 0], tok[:, 1]
            f = [c, d, mlps['mlp_cls'](c), mlps['mlp_dist'](d), mlps['mlp_res'](c), mlps['mlp_eff'](d)]
            lg = [deit.head(c), deit.head_dist(d)] + [fcs[k](h) for k, h in zip(
                ['fc_cls', 'fc_dist', 'fc_res', 'fc_eff'], f[2:])]
        feats.append(torch.stack(f, 1).float().cpu())
        logits.append(torch.stack(lg, 1).float().cpu())
    return torch.cat(feats), torch.cat(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', default='./data')
    ap.add_argument('--mlp-ckpt', required=True)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=3)
    ap.add_argument('--out', default='./results/analysis/token_diversity_mlp.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    ds = CIFAR100(args.data_path, train=False, download=False, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                         pin_memory=True)
    print(f'CIFAR-100 test: {len(ds)} images, device {device}')

    deit = timm.create_model('deit_base_distilled_patch16_224.fb_in1k', pretrained=True).eval().to(device)
    ckpt = torch.load(args.mlp_ckpt, map_location='cpu')
    D = deit.num_features
    mlps, fcs = {}, {}
    for k in ['mlp_cls', 'mlp_dist', 'mlp_res', 'mlp_eff']:
        mlps[k] = MLP(D); mlps[k].load_state_dict(ckpt[k]); mlps[k].eval().to(device)
    for k in ['fc_cls', 'fc_dist', 'fc_res', 'fc_eff']:
        fcs[k] = nn.Linear(D, 1000); fcs[k].load_state_dict(ckpt[k]); fcs[k].eval().to(device)
    print(f"Loaded MLPs + fc heads from {args.mlp_ckpt} (epoch {ckpt.get('epoch')})")

    feats, logits = collect(deit, mlps, fcs, loader, device)
    res = pairwise(feats, logits, NAMES)
    res['names'] = NAMES

    print('\n' + '=' * 70 + '\ndeit_b + 4 MLP pseudo-heads (lwf_hydra_v2)\n' + '=' * 70)
    print_matrix('cosine (head input features, mean per image)', NAMES, res['cos'])
    print_matrix('linear CKA (head input features)', NAMES, res['cka'])
    print_matrix('top-1 agreement (IN-1k heads)', NAMES, res['agree'])
    print_matrix('Jensen-Shannon (IN-1k heads, nats)', NAMES, res['js'])
    print(f"\nmean off-diagonal: cos {off_diag_mean(res['cos']):.3f} | cka {off_diag_mean(res['cka']):.3f}"
          f" | agree {off_diag_mean(res['agree']):.3f} | js {off_diag_mean(res['js']):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
