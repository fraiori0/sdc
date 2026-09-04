"""
Task C (2026-09-04 continued investigation): does a mean offset (`delta`) in the
loss's negative term, combined with each activation, break the ratchet confirmed
by Task A? 5000-step single-batch overfit, one fixed real batch, per
(act, delta) cell. No multi-epoch job.

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/sbdr_diagnose_delta.py --act clip --delta 0.0
"""

import argparse
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
PRINT_EVERY = 250


def build_cfg():
    with hydra.initialize(config_path='../configs', version_base=None):
        cfg = hydra.compose(config_name='train.yaml',
                            overrides=['model=sbdr', 'dataset=cifar10', 'epochs=1'],
                            return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--act', choices=['clip', 'sigmoid', 'ste_clip'], required=True)
    parser.add_argument('--delta', type=float, required=True)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    cfg = build_cfg()

    train_dataset = hydra.utils.instantiate(cfg.dataset.train_dataset)
    loader = engine.dataloader(train_dataset, cfg.batch_size, shuffle=True, drop_last=True)
    images, labels, index = next(iter(loader))
    image_1, image_2 = images[0].to(DEVICE), images[1].to(DEVICE)
    print(f'act={args.act} delta={args.delta}  fixed batch: {tuple(image_1.shape)}')

    backbone = hydra.utils.instantiate(cfg.backbone)
    model = hydra.utils.instantiate(cfg.model, backbone=backbone, act=args.act).to(DEVICE).train()
    loss_fn = SBDRCriticLoss(eps=EPS, critic_order=1, symmetric=True, delta=args.delta)

    params = list(model.get_backbone().parameters()) + list(model.get_training_modules().parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999))

    log = {'step': [], 'loss': [], 'kappa_mean': [], 'kappa_std': [], 'distinct_codes': [],
          'dead_bits_exact': [], 'preact_absmax': []}
    stopped_early_at = None
    failure_reason = None

    for step in range(1, N_STEPS + 1):
        optimizer.zero_grad()

        logits1 = model.encoder(model.norm(model.backbone(image_1)))
        logits2 = model.encoder(model.norm(model.backbone(image_2)))
        preact_absmax = max(logits1.abs().max().item(), logits2.abs().max().item())

        if args.act == 'clip':
            z1, z2 = logits1.clamp(0, 1), logits2.clamp(0, 1)
        elif args.act == 'sigmoid':
            z1, z2 = torch.sigmoid(logits1), torch.sigmoid(logits2)
        else:  # ste_clip
            c1, c2 = logits1.clamp(0, 1), logits2.clamp(0, 1)
            z1 = logits1 + (c1 - logits1).detach()
            z2 = logits2 + (c2 - logits2).detach()

        out_of_domain = (z1.min().item() < -1e-6 or z1.max().item() > 1 + 1e-6 or
                         z2.min().item() < -1e-6 or z2.max().item() > 1 + 1e-6)
        if out_of_domain:
            stopped_early_at = step
            failure_reason = (f'z out of [0,1] domain (float32 cancellation in the STE add/detach '
                              f'at preact_absmax={preact_absmax:.3e}): '
                              f'z1=[{z1.min().item():.4g},{z1.max().item():.4g}] '
                              f'z2=[{z2.min().item():.4g},{z2.max().item():.4g}]')
            print(f'  STOPPED at step {step}: {failure_reason}')
            break

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

            log['step'].append(step)
            log['loss'].append(L.item())
            log['kappa_mean'].append(kappa.mean().item())
            log['kappa_std'].append(kappa.std().item())
            log['distinct_codes'].append(n_distinct)
            log['dead_bits_exact'].append(dead_exact)
            log['preact_absmax'].append(preact_absmax)

            if step == 1 or step % PRINT_EVERY == 0 or step == N_STEPS:
                print(f'  step {step:>5}: loss={L.item():>10.6f}  kappa={kappa.mean().item():>6.3f}'
                     f'±{kappa.std().item():>5.3f}  distinct={n_distinct:>3}/64  '
                     f'dead_exact={dead_exact:>3}  preact_absmax={preact_absmax:>10.3e}')

    tag = f'act{args.act}_delta{args.delta}'.replace('.', 'p')
    torch.save({'log': log, 'stopped_early_at': stopped_early_at, 'failure_reason': failure_reason},
              f'experiments/sbdr_delta_trajectory_{tag}.pt')

    dbe = log['dead_bits_exact']
    print(f'\nSUMMARY act={args.act} delta={args.delta}: '
         f'dead_bits_exact first={dbe[0]} last={dbe[-1]} max={max(dbe)}  '
         f'kappa_mean first={log["kappa_mean"][0]:.3f} last={log["kappa_mean"][-1]:.3f}  '
         f'loss first={log["loss"][0]:.6f} last={log["loss"][-1]:.6f} min={min(log["loss"]):.6f}  '
         f'stopped_early_at={stopped_early_at}  failure_reason={failure_reason}')


if __name__ == '__main__':
    main()
