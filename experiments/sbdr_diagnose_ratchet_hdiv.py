"""
2026-09-04 follow-up to the §11 ratchet diagnostic (`sbdr_diagnose_ratchet.py`),
distinguishing two hypotheses for *why* training is pulled into the all-samples-
one-code collapse (HANDOUT §11, Task A): (1) backbone-feature correlation across
samples couples per-sample gradients through the shared head, vs (2) something
about this repo's head/init/optimizer independent of input correlation.

Minimal extension of the original script:
  - also logs cross-sample diversity of the backbone features h_i (pairwise
    cosine similarity and Euclidean distance, mean+var across the batch), and
    of the binarized output codes (pairwise Hamming distance, mean+var), at
    the same steps as the original loss/kappa/dead-bits table;
  - optional `--frozen_backbone` flag: mirrors trainers/base.py's
    `backbone_lr_scale == 0` path exactly (backbone params excluded from the
    optimizer and requires_grad_(False)), everything else (fixed batch, seed,
    Adam lr=1e-4 wd=1e-5, act=clip, critic_order=1, 5000 steps) unchanged.

Does NOT touch or overwrite `sbdr_ratchet_trajectory.pt` (the original,
already-tabulated-in-HANDOUT trajectory) -- writes to a new file per mode.

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/sbdr_diagnose_ratchet_hdiv.py                    # Task 1: unfrozen, rerun with h_i logging added
    CUDA_VISIBLE_DEVICES=2 python experiments/sbdr_diagnose_ratchet_hdiv.py --frozen_backbone   # Task 2: frozen-backbone ablation
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
TABLE_STEPS = {1, 250, 1000, 1500, 1750, 3000, 5000}


def build_cfg():
    with hydra.initialize(config_path='../configs', version_base=None):
        cfg = hydra.compose(config_name='train.yaml',
                            overrides=['model=sbdr', 'dataset=cifar10', 'epochs=1'],
                            return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
    return cfg


def pairwise_stats(X, metric):
    """Mean and variance of pairwise (i<j) similarity/distance across the batch dim."""
    B = X.shape[0]
    if metric == 'cosine':
        Xn = torch.nn.functional.normalize(X, dim=1)
        M = Xn @ Xn.T
    elif metric == 'euclidean':
        M = torch.cdist(X, X, p=2)
    else:  # hamming, X assumed binary 0/1
        overlap = X @ X.T
        kappa = X.sum(1)
        M = kappa.unsqueeze(0) + kappa.unsqueeze(1) - 2 * overlap
    iu = torch.triu_indices(B, B, offset=1, device=X.device)
    vals = M[iu[0], iu[1]]
    return vals.mean().item(), vals.var().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen_backbone', action='store_true',
                        help='Task 2: backbone_lr_scale=0, mirrors trainers/base.py freeze path')
    args = parser.parse_args()

    torch.manual_seed(0)
    cfg = build_cfg()

    train_dataset = hydra.utils.instantiate(cfg.dataset.train_dataset)
    loader = engine.dataloader(train_dataset, cfg.batch_size, shuffle=True, drop_last=True)
    images, labels, index = next(iter(loader))
    image_1, image_2 = images[0].to(DEVICE), images[1].to(DEVICE)
    print(f'fixed batch: {tuple(image_1.shape)}, batch_size={cfg.batch_size}, '
         f'frozen_backbone={args.frozen_backbone}')

    backbone = hydra.utils.instantiate(cfg.backbone)
    model = hydra.utils.instantiate(cfg.model, backbone=backbone, act='clip').to(DEVICE).train()
    loss_fn = SBDRCriticLoss(eps=EPS, critic_order=1, symmetric=True)

    backbone_params = list(model.get_backbone().parameters())
    head_params = list(model.get_training_modules().parameters())
    if args.frozen_backbone:
        print('Freezing backbone (backbone_lr_scale=0, trainers/base.py path)')
        for p in backbone_params:
            p.requires_grad_(False)
        params = [{'params': head_params}]
    else:
        params = [{'params': backbone_params}, {'params': head_params}]
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999))

    log = {'step': [], 'loss': [], 'kappa_mean': [], 'kappa_std': [], 'distinct_codes': [],
          'dead_bits_exact': [], 'preact_mean_mean': [], 'preact_mean_std': [],
          'preact_mean_min': [], 'preact_mean_max': [],
          'code_hamming_mean': [], 'code_hamming_var': [],
          'feat_cosine_mean': [], 'feat_cosine_var': [],
          'feat_euclidean_mean': [], 'feat_euclidean_var': []}
    preact_mean_history = []

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
                preact_mean = torch.cat([logits1, logits2], 0).mean(0)

                code_ham_mean, code_ham_var = pairwise_stats(active, 'hamming')
                feat_all = torch.cat([feats1, feats2], 0)
                feat_cos_mean, feat_cos_var = pairwise_stats(feat_all, 'cosine')
                feat_euc_mean, feat_euc_var = pairwise_stats(feat_all, 'euclidean')

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
            log['code_hamming_mean'].append(code_ham_mean)
            log['code_hamming_var'].append(code_ham_var)
            log['feat_cosine_mean'].append(feat_cos_mean)
            log['feat_cosine_var'].append(feat_cos_var)
            log['feat_euclidean_mean'].append(feat_euc_mean)
            log['feat_euclidean_var'].append(feat_euc_var)
            preact_mean_history.append(preact_mean.cpu().clone())

            if step in TABLE_STEPS or step % 250 == 0:
                print(f'  step {step:>5}: loss={L.item():>10.6f}  kappa={kappa.mean().item():>6.3f}'
                     f'±{kappa.std().item():>5.3f}  distinct={n_distinct:>3}/64  '
                     f'dead_exact={dead_exact:>3}  code_ham(mean/var)={code_ham_mean:>6.3f}/'
                     f'{code_ham_var:>6.3f}  feat_cos(mean/var)={feat_cos_mean:>7.4f}/'
                     f'{feat_cos_var:>7.4f}  feat_euc(mean/var)={feat_euc_mean:>7.4f}/{feat_euc_var:>7.4f}')

    tag = 'frozen' if args.frozen_backbone else 'unfrozen_hdiv'
    out_path = f'experiments/sbdr_ratchet_trajectory_{tag}.pt'
    torch.save({'log': log, 'preact_mean_history': torch.stack(preact_mean_history),
               'frozen_backbone': args.frozen_backbone},
              out_path)

    dbe = log['dead_bits_exact']
    n_increases = sum(1 for i in range(1, len(dbe)) if dbe[i] > dbe[i - 1])
    n_decreases = sum(1 for i in range(1, len(dbe)) if dbe[i] < dbe[i - 1])
    print(f'\ndead_bits_exact: first={dbe[0]}, last={dbe[-1]}, max={max(dbe)}, '
         f'n_logged_increases={n_increases}, n_logged_decreases={n_decreases}')
    print(f'kappa_mean: first={log["kappa_mean"][0]:.3f}, last={log["kappa_mean"][-1]:.3f}')
    print(f'distinct_codes: first={log["distinct_codes"][0]}, last={log["distinct_codes"][-1]}')
    print(f'loss: first={log["loss"][0]:.6f}, last={log["loss"][-1]:.6f}, min={min(log["loss"]):.6f}')
    print(f'feat_cosine_mean: first={log["feat_cosine_mean"][0]:.4f}, last={log["feat_cosine_mean"][-1]:.4f}')
    print(f'Saved full trajectory to {out_path}')


if __name__ == '__main__':
    main()
