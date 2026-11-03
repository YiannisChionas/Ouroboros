"""Token / head diversity of the pretrained backbones (no CIL training).

Compares how different the prefix tokens are, on the CIFAR-100 test set (224px, in1k normalisation,
same test transform as the CIL runs):
  - vit-hydra : cls + 5 dist tokens  (heads: cls, regnety, convnext, swin, r101, r50)
  - DeiT-B⚗   : cls + dist token     (heads: cls, dist)

Per token pair:
  - cos   : mean per-image cosine similarity of the output token features
  - cka   : linear CKA of the token features across images
  - agree : top-1 agreement of the ImageNet-1k heads
  - js    : mean Jensen-Shannon divergence of the ImageNet-1k head softmaxes (nats, max ln2 = 0.693)
Also: cosine between the learned input token embeddings (before the transformer).
"""
import argparse
import json
import os

import timm
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import InterpolationMode

from networks.vit_hydra_tokens import vit_hydra_base_patch16_224


def linear_cka(x, y):
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    return ((y.T @ x).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm())).item()


def js_div(p_logits, q_logits):
    p, q = F.softmax(p_logits, 1), F.softmax(q_logits, 1)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum(1)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).mean().item()


@torch.no_grad()
def collect(model, kind, loader, device):
    feats, logits = [], []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            if kind == 'vit_hydra':
                tok = model.forward_tokens(x)                              # [B, 6, D]
                cls_l, dist_l = model.forward_head(tok)
                lg = torch.stack([cls_l] + dist_l, 1)                      # [B, 6, 1000]
            else:
                tok = model.forward_features(x)[:, :2]                     # [B, 2, D]
                lg = torch.stack([model.head(tok[:, 0]), model.head_dist(tok[:, 1])], 1)
        feats.append(tok.float().cpu())
        logits.append(lg.float().cpu())
    return torch.cat(feats), torch.cat(logits)


def pairwise(feats, logits, names):
    n = len(names)
    out = {k: [[None] * n for _ in range(n)] for k in ('cos', 'cka', 'agree', 'js')}
    f64 = feats.double()
    pred = logits.argmax(-1)
    for i in range(n):
        for j in range(n):
            out['cos'][i][j]   = F.cosine_similarity(feats[:, i], feats[:, j], dim=-1).mean().item()
            out['cka'][i][j]   = linear_cka(f64[:, i], f64[:, j])
            out['agree'][i][j] = (pred[:, i] == pred[:, j]).float().mean().item()
            out['js'][i][j]    = js_div(logits[:, i], logits[:, j])
    return out


def print_matrix(title, names, m, fmt='{:6.3f}'):
    print(f'\n{title}')
    print(' ' * 10 + ''.join(f'{n:>10s}' for n in names))
    for n, row in zip(names, m):
        print(f'{n:>10s}' + ''.join(f'{fmt.format(v):>10s}' for v in row))


def off_diag_mean(m):
    n = len(m)
    vals = [m[i][j] for i in range(n) for j in range(n) if i != j]
    return sum(vals) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', default='./data')
    ap.add_argument('--vit-hydra-weights', required=True)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=3)
    ap.add_argument('--out', default='./results/analysis/token_diversity.json')
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

    models = {}
    vh = vit_hydra_base_patch16_224()
    vh.load_state_dict(torch.load(args.vit_hydra_weights, map_location='cpu'))
    models['vit_hydra'] = (vh, ['cls', 'regnety', 'convnext', 'swin', 'r101', 'r50'],
                           torch.cat([vh.cls_token, vh.dist_tokens], 1)[0])
    deit = timm.create_model('deit_base_distilled_patch16_224.fb_in1k', pretrained=True)
    models['deit_b'] = (deit, ['cls', 'dist'], torch.cat([deit.cls_token, deit.dist_token], 1)[0])

    results = {}
    for kind, (model, names, in_tok) in models.items():
        model.eval().to(device)
        feats, logits = collect(model, kind, loader, device)
        model.cpu()
        res = pairwise(feats, logits, names)
        in_tok = F.normalize(in_tok.detach().float(), dim=-1)
        res['input_token_cos'] = (in_tok @ in_tok.T).tolist()
        res['names'] = names
        results[kind] = res

        print('\n' + '=' * 70 + f'\n{kind}\n' + '=' * 70)
        print_matrix('cosine (output features, mean per image)', names, res['cos'])
        print_matrix('linear CKA (output features)', names, res['cka'])
        print_matrix('top-1 agreement (IN-1k heads)', names, res['agree'])
        print_matrix('Jensen-Shannon (IN-1k heads, nats)', names, res['js'])
        print_matrix('cosine (input token embeddings)', names, res['input_token_cos'])
        print(f"\nmean off-diagonal: cos {off_diag_mean(res['cos']):.3f} | cka {off_diag_mean(res['cka']):.3f}"
              f" | agree {off_diag_mean(res['agree']):.3f} | js {off_diag_mean(res['js']):.4f}"
              f" | input cos {off_diag_mean(res['input_token_cos']):.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
