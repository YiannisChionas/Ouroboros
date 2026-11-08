"""Per-layer weight drift of the DeiT-B⚗ backbone across the 10 CIFAR-100 CIL tasks.

For each approach, from results/<network>/cifar100/<approach>/10/models/task{t}.ckpt:
  step drift  : ||theta_t - theta_{t-1}|| / ||theta_{t-1}||   (theta_{-1} = timm pretrained weights)
  total drift : ||theta_t - theta_pre||   / ||theta_pre||
per group: embed (patch_embed, cls/dist token, pos_embed), block 0..11, norm (final).
Also the step drift of the old task heads (cls heads and dist heads, tasks < t), pooled over all old heads.
"""
import argparse
import json
import os
import re

import timm
import torch

GROUPS = ['embed'] + [f'block{i}' for i in range(12)] + ['norm']


def group_of(key):
    """key without the 'model.' prefix -> group name, or None (IN-1k heads are not used)."""
    m = re.match(r'blocks\.(\d+)\.', key)
    if m:
        return f'block{m.group(1)}'
    if key.startswith(('patch_embed.', 'cls_token', 'dist_token', 'pos_embed')):
        return 'embed'
    if key.startswith('norm.'):
        return 'norm'
    return None


def backbone(sd):
    return {k[len('model.'):]: v.float() for k, v in sd.items() if k.startswith('model.')}


def rel(a, b, keys):
    num = sum(((a[k] - b[k]) ** 2).sum().item() for k in keys)
    den = sum((b[k] ** 2).sum().item() for k in keys)
    return (num / den) ** 0.5


def old_heads_drift(cur, prev, prefix, t):
    keys = [k for k in prev if re.match(rf'{prefix}\.(\d+)\.', k) and int(k.split('.')[1]) < t]
    return rel({k: cur[k].float() for k in keys}, {k: prev[k].float() for k in keys}, keys) if keys else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='./results/deit_base_distilled_patch16_224_cil/cifar100')
    ap.add_argument('--approaches', nargs='+',
                    default=['lwf_dual', 'lwf_dual_v3', 'lwf_dual_v5', 'lwf_dual_v6', 'lwf_dual_v7'])
    ap.add_argument('--num-tasks', type=int, default=10)
    ap.add_argument('--out', default='./results/analysis/weight_drift.json')
    args = ap.parse_args()

    pre = {k: v.float() for k, v in
           timm.create_model('deit_base_distilled_patch16_224.fb_in1k', pretrained=True).state_dict().items()}
    gkeys = {g: [k for k in pre if group_of(k) == g] for g in GROUPS}
    all_keys = [k for g in GROUPS for k in gkeys[g]]

    results = {}
    for appr in args.approaches:
        d = os.path.join(args.results, appr, str(args.num_tasks), 'models')
        res = {'step': {g: [] for g in GROUPS + ['all']}, 'total': {g: [] for g in GROUPS + ['all']},
               'old_cls_heads': [], 'old_dist_heads': []}
        prev_bb, prev_sd = pre, None
        for t in range(args.num_tasks):
            sd = torch.load(os.path.join(d, f'task{t}.ckpt'), map_location='cpu')
            bb = backbone(sd)
            missing = [k for k in all_keys if k not in bb]
            assert not missing, f'{appr} task{t}: missing backbone keys {missing[:5]}'
            for g in GROUPS:
                res['step'][g].append(rel(bb, prev_bb, gkeys[g]))
                res['total'][g].append(rel(bb, pre, gkeys[g]))
            res['step']['all'].append(rel(bb, prev_bb, all_keys))
            res['total']['all'].append(rel(bb, pre, all_keys))
            if prev_sd is not None:
                res['old_cls_heads'].append(old_heads_drift(sd, prev_sd, 'heads', t))
                res['old_dist_heads'].append(old_heads_drift(sd, prev_sd, 'heads_dist', t))
            prev_bb, prev_sd = bb, sd
        results[appr] = res

        print('\n' + '=' * 100 + f'\n{appr}\n' + '=' * 100)
        print('step drift ||theta_t - theta_{t-1}|| / ||theta_{t-1}||  (x1e-3; task 0 = vs pretrained)')
        print(f"{'group':>8}" + ''.join(f'{f"t{t}":>8}' for t in range(args.num_tasks)) + f"{'mean1-9':>9}")
        for g in GROUPS + ['all']:
            v = res['step'][g]
            print(f'{g:>8}' + ''.join(f'{1e3 * x:8.2f}' for x in v) + f'{1e3 * sum(v[1:]) / len(v[1:]):9.2f}')
        print('total drift vs pretrained at the last task (x1e-3): ' +
              ' '.join(f'{g}={1e3 * res["total"][g][-1]:.1f}' for g in GROUPS + ['all']))
        fmt = lambda v: ' '.join('-' if x is None else f'{1e3 * x:.1f}' for x in v)
        print(f'old cls heads step drift t1..t9 (x1e-3):  {fmt(res["old_cls_heads"])}')
        print(f'old dist heads step drift t1..t9 (x1e-3): {fmt(res["old_dist_heads"])}')

    print('\n' + '=' * 100 + '\nSUMMARY: mean step drift over tasks 1-9 (x1e-3)\n' + '=' * 100)
    print(f"{'group':>8}" + ''.join(f'{a:>14}' for a in args.approaches))
    for g in GROUPS + ['all']:
        print(f'{g:>8}' + ''.join(f'{1e3 * sum(results[a]["step"][g][1:]) / (args.num_tasks - 1):14.2f}' for a in args.approaches))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f)
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
