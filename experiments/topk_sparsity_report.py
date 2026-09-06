"""
mAP-vs-k top-k sparsification sweep for a single Arm B checkpoint (HANDOUT.md
§15, Phase 2 / 2.3). Reuses the existing `topk_eval` mechanism
(`utils.hashing.preprocess_on_codes`) already used in §12.3 and
`experiments/sbdr_report.py`'s `mAP_sweep` -- no new eval code, just a custom
list of k values instead of the fixed {8,16,32} sweep, and no retraining
(operates on the checkpoint's already-saved `outputs/db_best.pth` /
`outputs/test_best.pth` continuous codes).

Usage:
    python experiments/topk_sparsity_report.py <logdir> --ks 2 4 8 16 32
"""

import argparse
import json
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.hashing import calculate_mAP
from utils.sbdr_metrics import usage_stats


def load_outputs(logdir):
    db = torch.load(os.path.join(logdir, 'outputs', 'db_best.pth'), weights_only=False, map_location='cpu')
    test = torch.load(os.path.join(logdir, 'outputs', 'test_best.pth'), weights_only=False, map_location='cpu')
    return db, test


def report(logdir, ks):
    logdir = logdir.rstrip('/')
    cfg = OmegaConf.load(os.path.join(logdir, 'config.yaml'))
    db, test = load_outputs(logdir)
    R = cfg.dataset.R

    stats = usage_stats(db['codes_cont'])

    out = {
        'logdir': logdir,
        'eps': cfg.criterion.get('eps'),
        'nbit': cfg.model.nbit,
        'kappa_mean': stats['kappa_mean'],
        'kappa_std': stats['kappa_std'],
        'native_mAP': calculate_mAP(db['codes_cont'], db['labels'], test['codes_cont'], test['labels'],
                                    Rs=R, dist_metric='overlap', code_domain='unit', topk_eval=None),
        'per_k': {},
    }
    for k in ks:
        mAP = calculate_mAP(db['codes_cont'], db['labels'], test['codes_cont'], test['labels'],
                            Rs=R, dist_metric='overlap', code_domain='unit', topk_eval=k)
        out['per_k'][k] = mAP
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir')
    parser.add_argument('--ks', type=int, nargs='+', required=True)
    args = parser.parse_args()

    out = report(args.logdir, args.ks)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
