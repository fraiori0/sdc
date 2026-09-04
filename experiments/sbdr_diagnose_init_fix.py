"""
Task B (2026-09-04 continued investigation): re-run the section-1 initialization
diagnostic for each `feature_norm` / `head_init_gain` candidate fix in
models/arch/sbdr.py, on ONE fixed real batch (act=clip only, since that's the
arm the fix targets). No training.

Usage:
    CUDA_VISIBLE_DEVICES=3 python experiments/sbdr_diagnose_init_fix.py
"""

import os
import sys

import hydra
import torch
from hydra.core.hydra_config import HydraConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine
from models.loss.sbdr import SBDRCriticLoss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS = 0.31
KAPPA_REF = 9
D = 64

CONFIGS = [
    dict(feature_norm='none', head_init_gain=1.0, label='baseline (none, gain=1.0)'),
    dict(feature_norm='standardize', head_init_gain=1.0, label='standardize, gain=1.0'),
    dict(feature_norm='batchnorm', head_init_gain=1.0, label='batchnorm, gain=1.0'),
    dict(feature_norm='none', head_init_gain=6.3, label='none, gain=6.3 (crude fallback)'),
    dict(feature_norm='standardize', head_init_gain=6.3, label='standardize + gain=6.3 (combined)'),
    dict(feature_norm='none', head_init_gain=15.0, label='none, gain=15.0'),
    dict(feature_norm='none', head_init_gain=30.0, label='none, gain=30.0'),
    dict(feature_norm='none', head_init_gain=60.0, label='none, gain=60.0'),
    dict(feature_norm='standardize', head_init_gain=15.0, label='standardize + gain=15.0'),
    dict(feature_norm='standardize', head_init_gain=30.0, label='standardize + gain=30.0'),
]


def build_cfg():
    with hydra.initialize(config_path='../configs', version_base=None):
        cfg = hydra.compose(config_name='train.yaml',
                            overrides=['model=sbdr', 'dataset=cifar10', 'epochs=1'],
                            return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
    return cfg


def main():
    torch.manual_seed(0)
    cfg = build_cfg()
    train_dataset = hydra.utils.instantiate(cfg.dataset.train_dataset)
    loader = engine.dataloader(train_dataset, cfg.batch_size, shuffle=True, drop_last=True)
    images, labels, index = next(iter(loader))
    image_1, image_2 = images[0].to(DEVICE), images[1].to(DEVICE)
    print(f'fixed batch: {tuple(image_1.shape)}')

    analytic_ref = torch.log(torch.tensor(KAPPA_REF ** 2 / D + EPS)) - torch.log(torch.tensor(KAPPA_REF + EPS))
    print(f'target: loss ~= {analytic_ref.item():.4f} (analytic, random kappa=9, d=64)\n')

    for c in CONFIGS:
        torch.manual_seed(1)  # same backbone/head init seed across configs for comparability
        backbone = hydra.utils.instantiate(cfg.backbone)
        model = hydra.utils.instantiate(cfg.model, backbone=backbone, act='clip',
                                        feature_norm=c['feature_norm'],
                                        head_init_gain=c['head_init_gain']).to(DEVICE).train()

        feats1 = model.backbone(image_1)
        feat1n = model.norm(feats1)
        hidden1 = model.encoder[1](model.encoder[0](feat1n))  # post-ReLU intermediate (1024,)
        logits1 = model.encoder(feat1n)  # pre-activation
        mean_u = logits1.mean(0)
        std_u = logits1.std(0)

        with torch.no_grad():
            z1 = logits1.clamp(0, 1)
            feats2 = model.backbone(image_2)
            feat2n = model.norm(feats2)
            logits2 = model.encoder(feat2n)
            z2 = logits2.clamp(0, 1)
            kappa = (z1 > 0.5).float().sum(1)
            n_distinct = (z1 > 0.5).float().unique(dim=0).size(0)
            zall = torch.cat([z1, z2], 0)
            zbar = zall.mean(0)
            t = (z1 * zbar).sum(1) + EPS
            s = (z1 * z2).sum(1) + EPS

        model.zero_grad()
        _, z1m, _ = model(image_1)
        _, z2m, _ = model(image_2)
        loss_fn = SBDRCriticLoss(eps=EPS, critic_order=1, symmetric=True)
        L = loss_fn(z1m, z2m)
        L.backward()
        head_grad_norm = torch.cat([p.grad.flatten() for p in model.encoder.parameters()
                                    if p.grad is not None]).norm().item()

        print(f'--- {c["label"]} ---')
        print(f'  pre-activation per-unit mean: mean={mean_u.mean().item():.4f} '
             f'min={mean_u.min().item():.4f} max={mean_u.max().item():.4f}')
        print(f'  pre-activation per-unit std : mean={std_u.mean().item():.4f} '
             f'min={std_u.min().item():.4f} max={std_u.max().item():.4f}')
        print(f'  intermediate (post-ReLU, 1024-dim) stats: mean={hidden1.mean().item():.4f} '
             f'std={hidden1.std().item():.4f}')
        print(f'  kappa at init: mean={kappa.mean().item():.3f} std={kappa.std().item():.3f}')
        print(f'  distinct codes in batch of {z1.size(0)}: {n_distinct}')
        print(f'  t_i: mean={t.mean().item():.4f} std={t.std().item():.4f}  |  '
             f's_i: mean={s.mean().item():.4f} std={s.std().item():.4f}')
        print(f'  head-weight grad norm: {head_grad_norm:.6e}')
        print(f'  loss at init: {L.item():.6f}  (target ~{analytic_ref.item():.4f})')
        print()


if __name__ == '__main__':
    main()
