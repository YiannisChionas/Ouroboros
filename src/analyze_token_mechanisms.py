"""Direct mechanism measurements of the prefix tokens (pretrained backbones, no CIL training).

Backbones: DeiT-B⚗ (cls, dist) and vit-hydra (cls + 5 dist tokens). Data: CIFAR-100 test, 224px, in1k norm.

1. Attention (per layer):
   for every prefix token, its attention over the 196 patch tokens (mean over heads, renormalised over patches).
   Per token pair: Pearson correlation and Jensen-Shannon divergence of the maps (mean over images).
   Per token: fraction of its attention that goes to patches (rest goes to the prefix tokens).

2. Parameter-Jacobian (per transformer block):
   rows = d(u_k . f_t(x_n)) / d(theta_block) for images n, fixed random unit directions u_k (shared by all tokens),
   f_t = final (normed) feature of token t. Gradients restricted to a fixed random subset of S coordinates per block.
   Per token: effective rank (Roy & Vetterli) of its row matrix.
   Per token pair: subspace overlap of the top row-spaces (90% energy): ||V_a^T V_b||_F^2 / min(r_a, r_b) in [0, 1],
   and mean cosine between the two tokens' gradients for the same (image, direction).
   This is the space in which KD through each token moves the backbone.
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


# ---------------------------------------------------------------- attention
@torch.no_grad()
def attention_stats(model, n_prefix, loader, n_images, device):
    blocks = model.blocks
    captured = {}

    def make_hook(i):
        def hook(mod, args):
            captured[i] = args[0]
        return hook

    hooks = [b.attn.register_forward_pre_hook(make_hook(i)) for i, b in enumerate(blocks)]
    L, P = len(blocks), n_prefix
    corr = torch.zeros(L, P, P, dtype=torch.float64)
    js = torch.zeros(L, P, P, dtype=torch.float64)
    to_patch = torch.zeros(L, P, dtype=torch.float64)
    seen = 0
    for x, _ in loader:
        if seen >= n_images:
            break
        x = x[: n_images - seen].to(device)
        model.forward_features(x)
        for li, b in enumerate(blocks):
            a = b.attn
            h = captured[li]
            B, N, C = h.shape
            qkv = a.qkv(h).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
            q, k = a.q_norm(qkv[0]), a.k_norm(qkv[1])
            att = ((q[:, :, :P] * a.scale) @ k.transpose(-2, -1)).softmax(-1).mean(1)   # [B, P, N]
            to_patch[li] += att[:, :, P:].sum(-1).double().sum(0).cpu()
            m = att[:, :, P:]
            m = m / m.sum(-1, keepdim=True)                                               # [B, P, 196]
            mc = m - m.mean(-1, keepdim=True)
            mc = mc / mc.norm(dim=-1, keepdim=True)
            corr[li] += torch.einsum('bpd,bqd->pq', mc, mc).double().cpu()
            lm = m.clamp_min(1e-12).log()
            for i in range(P):
                for j in range(P):
                    mid = 0.5 * (m[:, i] + m[:, j])
                    lmid = mid.clamp_min(1e-12).log()
                    js[li, i, j] += (0.5 * (m[:, i] * (lm[:, i] - lmid)).sum(-1)
                                     + 0.5 * (m[:, j] * (lm[:, j] - lmid)).sum(-1)).double().sum().cpu()
        seen += x.shape[0]
    for hk in hooks:
        hk.remove()
    return {'corr': (corr / seen).tolist(), 'js': (js / seen).tolist(), 'to_patch': (to_patch / seen).tolist(),
            'n_images': seen}


# ---------------------------------------------------------------- parameter Jacobian
def jacobian_stats(model, n_prefix, images, n_dirs, n_coords, device, seed=0, energy=0.9):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    blocks = list(model.blocks)
    params = [[p for p in b.parameters()] for b in blocks]
    sizes = [sum(p.numel() for p in ps) for ps in params]
    idx = [torch.randperm(s, generator=g)[: min(n_coords, s)].to(device) for s in sizes]
    D = model.embed_dim
    dirs = F.normalize(torch.randn(n_dirs, D, generator=g), dim=-1).to(device)

    P, L = n_prefix, len(blocks)
    rows = [[[] for _ in range(L)] for _ in range(P)]        # rows[t][block] -> list of [S] vectors
    for x in images:
        feats = model.forward_features(x.unsqueeze(0).to(device))[0, :P]       # [P, D]
        for t in range(P):
            for kk in range(n_dirs):
                s = feats[t] @ dirs[kk]
                grads = torch.autograd.grad(s, [p for ps in params for p in ps], retain_graph=True)
                off = 0
                for b, ps in enumerate(params):
                    gb = torch.cat([gr.reshape(-1) for gr in grads[off: off + len(ps)]])
                    rows[t][b].append(gb[idx[b]])
                    off += len(ps)
        del feats

    def erank(G):
        sv = torch.linalg.svdvals(G)
        p = sv / sv.sum()
        return torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item()

    def top_space(G):
        _, sv, Vh = torch.linalg.svd(G, full_matrices=False)
        e = (sv ** 2).cumsum(0) / (sv ** 2).sum()
        r = int((e < energy).sum().item()) + 1
        return Vh[:r]                                          # [r, S]

    out = {'erank': [[0.0] * L for _ in range(P)], 'overlap': [[[0.0] * P for _ in range(P)] for _ in range(L)],
           'grad_cos': [[[0.0] * P for _ in range(P)] for _ in range(L)], 'rank90': [[0] * L for _ in range(P)],
           'rows_per_token': len(images) * n_dirs, 'coords_per_block': [int(i.numel()) for i in idx]}
    for b in range(L):
        Gs = [torch.stack(rows[t][b]).double() for t in range(P)]
        Vs = [top_space(G) for G in Gs]
        for t in range(P):
            out['erank'][t][b] = erank(Gs[t])
            out['rank90'][t][b] = Vs[t].shape[0]
        for i in range(P):
            for j in range(P):
                out['overlap'][b][i][j] = ((Vs[i] @ Vs[j].T).norm() ** 2 / min(len(Vs[i]), len(Vs[j]))).item()
                out['grad_cos'][b][i][j] = F.cosine_similarity(Gs[i], Gs[j], dim=-1).mean().item()
    return out


# ---------------------------------------------------------------- printing
def pair_groups(P):
    cls_dist = [(0, j) for j in range(1, P)]
    dist_dist = [(i, j) for i in range(1, P) for j in range(i + 1, P)]
    return cls_dist, dist_dist


def mean_pairs(m, pairs):
    return sum(m[i][j] for i, j in pairs) / len(pairs) if pairs else float('nan')


def report(name, names, att, jac):
    P = len(names)
    cls_dist, dist_dist = pair_groups(P)
    print('\n' + '=' * 90 + f'\n{name}  tokens: {names}\n' + '=' * 90)
    print('\nATTENTION per layer  (corr / JS: mean over token pairs; to_patch: per token)')
    hdr = f"{'layer':>5} {'corr cls-dist':>14} {'corr dist-dist':>15} {'JS cls-dist':>12} {'JS dist-dist':>13}  to_patch"
    print(hdr)
    for li in range(len(att['corr'])):
        tp = ' '.join(f'{v:.2f}' for v in att['to_patch'][li])
        print(f"{li:>5} {mean_pairs(att['corr'][li], cls_dist):>14.3f} {mean_pairs(att['corr'][li], dist_dist):>15.3f}"
              f" {mean_pairs(att['js'][li], cls_dist):>12.4f} {mean_pairs(att['js'][li], dist_dist):>13.4f}  {tp}")
    print('\nJACOBIAN per block  (erank / rank90: per token; overlap / grad_cos: mean over token pairs)')
    print(f"{'block':>5} {'overlap cls-dist':>17} {'overlap dist-dist':>18} {'cos cls-dist':>13} {'cos dist-dist':>14}"
          f"  erank per token")
    for b in range(len(jac['overlap'])):
        er = ' '.join(f"{jac['erank'][t][b]:.1f}" for t in range(P))
        print(f"{b:>5} {mean_pairs(jac['overlap'][b], cls_dist):>17.3f} {mean_pairs(jac['overlap'][b], dist_dist):>18.3f}"
              f" {mean_pairs(jac['grad_cos'][b], cls_dist):>13.3f} {mean_pairs(jac['grad_cos'][b], dist_dist):>14.3f}  {er}")
    print(f"rows per token: {jac['rows_per_token']}, coords per block: {jac['coords_per_block'][0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', default='./data')
    ap.add_argument('--vit-hydra-weights', required=True)
    ap.add_argument('--att-images', type=int, default=1000)
    ap.add_argument('--jac-images', type=int, default=16)
    ap.add_argument('--jac-dirs', type=int, default=4)
    ap.add_argument('--jac-coords', type=int, default=100000)
    ap.add_argument('--batch-size', type=int, default=50)
    ap.add_argument('--out', default='./results/analysis/token_mechanisms.json')
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    ds = CIFAR100(args.data_path, train=False, download=False, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=3)
    g = torch.Generator().manual_seed(0)
    jac_idx = torch.randperm(len(ds), generator=g)[: args.jac_images].tolist()
    jac_images = [ds[i][0] for i in jac_idx]
    print(f'CIFAR-100 test, device {device}: attention on {args.att_images} images, '
          f'Jacobian on {args.jac_images} images x {args.jac_dirs} directions, {args.jac_coords} coords/block')

    vh = vit_hydra_base_patch16_224()
    vh.load_state_dict(torch.load(args.vit_hydra_weights, map_location='cpu'))
    models = {
        'deit_b': (timm.create_model('deit_base_distilled_patch16_224.fb_in1k', pretrained=True), 2, ['cls', 'dist']),
        'vit_hydra': (vh, 6, ['cls', 'regnety', 'convnext', 'swin', 'r101', 'r50']),
    }
    results = {}
    for name, (model, P, names) in models.items():
        model.eval().to(device)
        att = attention_stats(model, P, loader, args.att_images, device)
        jac = jacobian_stats(model, P, jac_images, args.jac_dirs, args.jac_coords, device)
        model.cpu()
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        results[name] = {'names': names, 'attention': att, 'jacobian': jac}
        report(name, names, att, jac)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f)
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
