"""
Task A (2026-09-04 continued investigation): confirm or refute the "ratchet"
hypothesis -- does clip's single-batch overfit (which reached loss=-1.96,
kappa=4.4, 62-64 distinct codes at 300 steps in the prior diagnostic) continue
to monotonically accumulate dead bits toward ~57 with kappa->7 if run longer
(5000 steps), or does it stabilize?

Single fixed real batch, act=clip, lambda2=0 (critic_order=1), Adam lr=1e-4
wd=1e-5 (matching §9/§10's actual optimizer config), backbone_lr_scale=1 (both
backbone and head train, matching the real runs). No multi-epoch job -- one
batch, reused every step.

Logs every `log_every` steps: loss, kappa mean/std, distinct codes,
dead_bits_exact (z_bar_u == 0.0 exactly, continuous, same definition as
trainers/sbdr.py), and summary stats (mean/std/min/max) of the 64-dim
per-unit pre-activation mean vector. Full per-step trajectory saved to
experiments/sbdr_ratchet_trajectory.pt for later inspection.

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/sbdr_diagnose_ratchet.py
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
N_STEPS = 5000
LOG_EVERY = 50


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
    print(f'fixed batch: {tuple(image_1.shape)}, batch_size={cfg.batch_size}')

    backbone = hydra.utils.instantiate(cfg.backbone)
    model = hydra.utils.instantiate(cfg.model, backbone=backbone, act='clip').to(DEVICE).train()
    loss_fn = SBDRCriticLoss(eps=EPS, critic_order=1, symmetric=True)

    params = list(model.get_backbone().parameters()) + list(model.get_training_modules().parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999))

    log = {'step': [], 'loss': [], 'kappa_mean': [], 'kappa_std': [], 'distinct_codes': [],
          'dead_bits_exact': [], 'preact_mean_mean': [], 'preact_mean_std': [],
          'preact_mean_min': [], 'preact_mean_max': []}
    preact_mean_history = []  # full (n_logged, 64) trajectory

    for step in range(1, N_STEPS + 1):
        optimizer.zero_grad()

        feats1 = model.backbone(image_1)
        logits1 = model.encoder(feats1)
        z1 = logits1.clamp(0, 1)
        feats2 = model.backbone(image_2)
        logits2 = model.encoder(feats2)
        z2 = logits2.clamp(0, 1)

        L = loss_fn(z1, z2)
        L.backward()
        optimizer.step()

        if step == 1 or step % LOG_EVERY == 0 or step == N_STEPS:
            with torch.no_grad():
                zall = torch.cat([z1, z2], 0)
                zbar = zall.mean(0)
                dead_exact = (zbar == 0).sum().item()
                active = (z1 > 0.5).float()
                kappa = active.sum(1)
                n_distinct = active.unique(dim=0).size(0)
                preact_mean = torch.cat([logits1, logits2], 0).mean(0)  # (64,)

            log['step'].append(step)
            log['loss'].append(L.item())
            log['kappa_mean'].append(kappa.mean().item())
            log['kappa_std'].append(kappa.std().item())
            log['distinct_codes'].append(n_distinct)
            log['dead_bits_exact'].append(dead_exact)
            log['preact_mean_mean'].append(preact_mean.mean().item())
            log['preact_mean_std'].append(preact_mean.std().item())
            log['preact_mean_min'].append(preact_mean.min().item())
            log['preact_mean_max'].append(preact_mean.max().item())
            preact_mean_history.append(preact_mean.cpu().clone())

            if step == 1 or step % 250 == 0 or step == N_STEPS:
                print(f'  step {step:>5}: loss={L.item():>10.6f}  kappa={kappa.mean().item():>6.3f}'
                     f'±{kappa.std().item():>5.3f}  distinct={n_distinct:>3}/64  '
                     f'dead_exact={dead_exact:>3}  preact_mean(mean/min/max)='
                     f'{preact_mean.mean().item():>7.4f}/{preact_mean.min().item():>7.4f}/'
                     f'{preact_mean.max().item():>7.4f}')

    torch.save({'log': log, 'preact_mean_history': torch.stack(preact_mean_history)},
              'experiments/sbdr_ratchet_trajectory.pt')

    # monotonicity check on dead_bits_exact
    dbe = log['dead_bits_exact']
    n_increases = sum(1 for i in range(1, len(dbe)) if dbe[i] > dbe[i - 1])
    n_decreases = sum(1 for i in range(1, len(dbe)) if dbe[i] < dbe[i - 1])
    print(f'\ndead_bits_exact: first={dbe[0]}, last={dbe[-1]}, max={max(dbe)}, '
         f'n_logged_increases={n_increases}, n_logged_decreases={n_decreases}')
    print(f'kappa_mean: first={log["kappa_mean"][0]:.3f}, last={log["kappa_mean"][-1]:.3f}')
    print(f'loss: first={log["loss"][0]:.6f}, last={log["loss"][-1]:.6f}, min={min(log["loss"]):.6f}')
    print('Saved full trajectory to experiments/sbdr_ratchet_trajectory.pt')


if __name__ == '__main__':
    main()
