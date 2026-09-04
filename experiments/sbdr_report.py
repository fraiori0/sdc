"""
Task C driver: per-run report for the Arm B second-order-critic sweep
(HANDOUT.md §2.2b). For each run logdir under a sweep, computes:

  - mAP: native (0.5 threshold) and via topk_eval at kappa in {8,16,32}
  - realised kappa, binarity, per-bit usage mean/std, dead/saturated bits
  - overlap distribution on the database (mean, std, #distinct values >1% mass,
    overlap=0 fraction), and tie-block size at R=dataset.R
  - positive/negative separation: augmented-view pair overlap vs random pair
    overlap, and their ratio
  - false-negative rate: fraction of top-50 retrieved db items sharing the
    query's class label

Numbers only -- no interpretation -- written to <logdir>/sbdr_report.json and
printed as a table.

Usage:
    python experiments/sbdr_report.py logs/cifar10/sbdr64_40/lambda2_0_* logs/cifar10/sbdr64_40/lambda2_0.4_* ...
"""

import argparse
import glob
import json
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra.utils

from utils.hashing import calculate_mAP
from utils.sbdr_metrics import (_sample_pairs_overlap, false_negative_rate,
                                overlap_distribution, positive_negative_separation,
                                tie_block_sizes, usage_stats)

KAPPA_SWEEP = [8, 16, 32]
TOPK_FN = 50
OVERLAP_SAMPLE = 3000
POSPAIR_SAMPLES = 1024


def load_outputs(logdir):
    db = torch.load(os.path.join(logdir, 'outputs', 'db_best.pth'), weights_only=False, map_location='cpu')
    test = torch.load(os.path.join(logdir, 'outputs', 'test_best.pth'), weights_only=False, map_location='cpu')
    return db, test


def load_trainer(logdir, device):
    load_config = OmegaConf.load(os.path.join(logdir, 'config.yaml'))
    load_config.device = device
    trainer = hydra.utils.instantiate(load_config.trainer, load_config)
    trainer.load_dataset()
    trainer.load_dataloader()
    trainer.load_model()
    trainer.load_criterion()
    trainer.load_model_state(os.path.join(logdir, 'models', 'best.pth'))
    trainer.to_device()
    trainer.model.eval()
    return trainer, load_config


def positive_pair_overlap(trainer, device, n_samples=POSPAIR_SAMPLES):
    codes_a, codes_b = [], []
    n = 0
    with torch.no_grad():
        for data in trainer.dataloader['train']:
            images, labels, index = data
            image_1, image_2 = images
            image_1, image_2 = image_1.to(device), image_2.to(device)
            _, z1, _ = trainer.model(image_1)
            _, z2, _ = trainer.model(image_2)
            codes_a.append((z1 > 0.5).float().cpu())
            codes_b.append((z2 > 0.5).float().cpu())
            n += z1.size(0)
            if n >= n_samples:
                break
    codes_a = torch.cat(codes_a)[:n_samples]
    codes_b = torch.cat(codes_b)[:n_samples]
    return (codes_a * codes_b).sum(1)


def mAP_sweep(db_cont, db_labels, test_cont, test_labels, R):
    out = {}
    mAP_native = calculate_mAP(db_cont, db_labels, test_cont, test_labels,
                               Rs=R, dist_metric='overlap', code_domain='unit', topk_eval=None)
    out['native'] = mAP_native
    for k in KAPPA_SWEEP:
        out[f'topk{k}'] = calculate_mAP(db_cont, db_labels, test_cont, test_labels,
                                        Rs=R, dist_metric='overlap', code_domain='unit', topk_eval=k)
    return out


def report_one(logdir, device='cuda'):
    logdir = logdir.rstrip('/')
    cfg = OmegaConf.load(os.path.join(logdir, 'config.yaml'))
    db, test = load_outputs(logdir)

    db_codes_cont = db['codes_cont']
    test_codes_cont = test['codes_cont']
    db_codes_bin = (db_codes_cont > 0.5).float()
    test_codes_bin = (test_codes_cont > 0.5).float()

    R = cfg.dataset.R

    res = {
        'logdir': logdir,
        'critic_order': cfg.criterion.get('critic_order'),
        'lambda2': cfg.criterion.get('lambda2'),
        'eps': cfg.criterion.get('eps'),
        'nbit': cfg.model.nbit,
        'db_size': db_codes_cont.size(0),
        'test_size': test_codes_cont.size(0),
        'R': R,
    }

    res['mAP'] = mAP_sweep(db_codes_cont, db['labels'], test_codes_cont, test['labels'], R)
    res['usage'] = usage_stats(db_codes_cont)
    res['overlap_distribution'] = overlap_distribution(db_codes_bin, n_sample=OVERLAP_SAMPLE)

    tie = tie_block_sizes(test_codes_bin, db_codes_bin, R=R)
    res['tie_block'] = {
        'R': R,
        'mean': tie.float().mean().item(),
        'std': tie.float().std().item(),
        'median': tie.float().median().item(),
        'max': tie.max().item(),
    }

    fn = false_negative_rate(test_codes_bin, db_codes_bin, test['labels'], db['labels'], k=TOPK_FN)
    res['false_negative'] = fn

    trainer, _ = load_trainer(logdir, device)
    pos_overlap = positive_pair_overlap(trainer, device)
    rand_overlap = _sample_pairs_overlap(db_codes_bin, n_sample=OVERLAP_SAMPLE)
    res['separation'] = positive_negative_separation(pos_overlap, rand_overlap)

    with open(os.path.join(logdir, 'sbdr_report.json'), 'w') as f:
        json.dump(res, f, indent=2)

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdirs', nargs='+')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    logdirs = []
    for pattern in args.logdirs:
        matched = sorted(glob.glob(pattern))
        logdirs.extend(matched if matched else [pattern])

    all_res = []
    for logdir in logdirs:
        print(f'\n=== {logdir} ===')
        try:
            res = report_one(logdir, device=args.device)
        except Exception as e:
            print(f'  FAILED: {e}')
            continue
        all_res.append(res)
        print(json.dumps(res, indent=2))

    out_path = 'experiments/sbdr_sweep_report.json'
    with open(out_path, 'w') as f:
        json.dump(all_res, f, indent=2)
    print(f'\nWrote combined report to {out_path}')


if __name__ == '__main__':
    main()
