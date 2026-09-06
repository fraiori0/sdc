"""
Storage accounting for Arm B checkpoints (HANDOUT.md §15.2/1.2).

Computes, from a saved checkpoint's database codes (`outputs/db_best.pth`,
`codes_cont`, [0,1]-domain continuous activations):

  - kappa_mean/std, dead_bits, binarity (via utils.sbdr_metrics.usage_stats)
  - the full per-sample kappa histogram (so the k_topk choice is checkable)
  - k_topk: the smallest k such that >= frac_target (default 0.90) of samples
    have kappa <= k, i.e. "captures the bulk of the active-unit mass" per a
    concrete >=90% rule
  - storage_meankappa = kappa_mean * ceil(log2(d))  bits
  - storage_topk      = k_topk     * ceil(log2(d))  bits

No training, no GPU needed (codes already saved to disk).

Usage:
    python experiments/storage_report.py <logdir> [--frac 0.90]
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sbdr_metrics import usage_stats


def kappa_histogram(codes_cont, threshold=0.5):
    active = (codes_cont > threshold).float()
    kappa_per_sample = active.sum(1).long()
    d = codes_cont.size(1)
    hist = torch.bincount(kappa_per_sample, minlength=d + 1)
    return kappa_per_sample, hist


def k_topk_for_frac(kappa_per_sample, d, frac_target=0.90):
    N = kappa_per_sample.numel()
    for k in range(0, d + 1):
        frac = (kappa_per_sample <= k).float().mean().item()
        if frac >= frac_target:
            return k, frac
    return d, 1.0


def report(logdir, frac_target=0.90):
    logdir = logdir.rstrip('/')
    db = torch.load(os.path.join(logdir, 'outputs', 'db_best.pth'), weights_only=False, map_location='cpu')
    codes_cont = db['codes_cont']
    d = codes_cont.size(1)
    bits_per_index = math.ceil(math.log2(d))

    stats = usage_stats(codes_cont)
    kappa_per_sample, hist = kappa_histogram(codes_cont)
    k_topk, frac_at_k = k_topk_for_frac(kappa_per_sample, d, frac_target)

    storage_meankappa = stats['kappa_mean'] * bits_per_index
    storage_topk = k_topk * bits_per_index

    out = {
        'logdir': logdir,
        'd': d,
        'bits_per_index_ceil_log2_d': bits_per_index,
        'kappa_mean': stats['kappa_mean'],
        'kappa_std': stats['kappa_std'],
        'dead_bits': stats['dead_bits'],
        'binarity': stats['binarity'],
        'k_topk': k_topk,
        'frac_at_k_topk': frac_at_k,
        'frac_target': frac_target,
        'storage_meankappa_bits': storage_meankappa,
        'storage_topk_bits': storage_topk,
        'kappa_histogram_nonzero': {int(k): int(c) for k, c in enumerate(hist.tolist()) if c > 0},
        'N_samples': int(kappa_per_sample.numel()),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir')
    parser.add_argument('--frac', type=float, default=0.90)
    args = parser.parse_args()

    out = report(args.logdir, frac_target=args.frac)
    for k, v in out.items():
        if k == 'kappa_histogram_nonzero':
            continue
        print(f'{k}: {v}')
    print('kappa_histogram (kappa: count), nonzero only:')
    for k in sorted(out['kappa_histogram_nonzero']):
        print(f'  {k}: {out["kappa_histogram_nonzero"][k]}')


if __name__ == '__main__':
    main()
