"""
Network-level diagnostics for the §9/§10 Arm B collapse investigation
(2026-09-03 continued). Static inspection / single-batch checks only -- no
multi-epoch training. Covers Task items 1, 2e, 2g, 3, 5, 7 of the diagnostic
request. Numbers only, no interpretation.

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/sbdr_diagnose_init.py
"""

import os
import sys

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine
from models.loss.sbdr import SBDRCriticLoss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS = 0.31
KAPPA_REF = 9
D = 64


def section(title):
    print(f'\n{"=" * 90}\n{title}\n{"=" * 90}')


def build_cfg(**overrides):
    ov = ['model=sbdr', 'dataset=cifar10', 'epochs=1'] + [f'{k}={v}' for k, v in overrides.items()]
    with hydra.initialize(config_path='../configs', version_base=None):
        cfg = hydra.compose(config_name='train.yaml', overrides=ov, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
    return cfg


def get_one_train_batch(cfg):
    train_dataset = hydra.utils.instantiate(cfg.dataset.train_dataset)
    loader = engine.dataloader(train_dataset, cfg.batch_size, shuffle=True, drop_last=True)
    data = next(iter(loader))
    images, labels, index = data
    image_1, image_2 = images
    return image_1.to(DEVICE), image_2.to(DEVICE), labels.to(DEVICE), index, cfg


def fresh_model(cfg, act):
    backbone = hydra.utils.instantiate(cfg.backbone)
    model = hydra.utils.instantiate(cfg.model, backbone=backbone, act=act)
    return model.to(DEVICE).train()


# ---------------------------------------------------------------------------
# Section 1 + 2e (part) + 2g: initialization diagnostic
# ---------------------------------------------------------------------------

def section1_and_2g(image_1, image_2, cfg):
    section('SECTION 1 + 2g: initialization diagnostic (fresh model, one real batch)')

    for act in ('clip', 'sigmoid'):
        print(f'\n--- act={act} ---')
        model = fresh_model(cfg, act)

        with torch.no_grad():
            feats1 = model.backbone(image_1)
            logits1 = model.encoder(feats1)  # pre-activation, (N, d)

        mean_u = logits1.mean(0)  # (d,)
        std_u = logits1.std(0)  # (d,)
        ratio_u = std_u / mean_u.abs().clamp(min=1e-12)

        print(f'  pre-activation per-unit mean: mean={mean_u.mean().item():.4f}, '
             f'min={mean_u.min().item():.4f}, max={mean_u.max().item():.4f}')
        print(f'  pre-activation per-unit std : mean={std_u.mean().item():.4f}, '
             f'min={std_u.min().item():.4f}, max={std_u.max().item():.4f}')
        print(f'  ratio std/|mean| per unit   : mean={ratio_u.mean().item():.4f}, '
             f'median={ratio_u.median().item():.4f}, min={ratio_u.min().item():.4f}, '
             f'max={ratio_u.max().item():.4f}')
        n_out_of_range = ((logits1 < 0) | (logits1 > 1)).sum().item()
        print(f'  fraction of (sample,unit) pre-activations outside [0,1]: '
             f'{n_out_of_range}/{logits1.numel()} = {n_out_of_range / logits1.numel():.4f}')

        with torch.no_grad():
            z1 = logits1.clamp(0, 1) if act == 'clip' else torch.sigmoid(logits1)
            feats2 = model.backbone(image_2)
            logits2 = model.encoder(feats2)
            z2 = logits2.clamp(0, 1) if act == 'clip' else torch.sigmoid(logits2)

            kappa = (z1 > 0.5).float().sum(1)
            binz = (z1 > 0.5).float()
            n_distinct = binz.unique(dim=0).size(0)
            bit_identical_frac = (binz.std(0) == 0).float().mean().item()

        print(f'  kappa per sample: mean={kappa.mean().item():.3f}, std={kappa.std().item():.3f}, '
             f'min={kappa.min().item():.0f}, max={kappa.max().item():.0f}')
        print(f'  distinct binarized codes in batch of {z1.size(0)}: {n_distinct}')
        print(f'  fraction of bits identical (std==0) across all samples: {bit_identical_frac:.4f}')

        loss1 = SBDRCriticLoss(eps=EPS, critic_order=1, symmetric=True)
        z1g = z1.clone().requires_grad_(True)
        z2g = z2.clone().requires_grad_(True)
        L = loss1(z1g, z2g)

        with torch.no_grad():
            zall = torch.cat([z1, z2], 0)
            zbar = zall.mean(0)
            t = (z1 * zbar).sum(1) + EPS
            s = (z1 * z2).sum(1) + EPS
        analytic_ref = torch.log(torch.tensor(KAPPA_REF ** 2 / D + EPS)) - torch.log(torch.tensor(KAPPA_REF + EPS))
        print(f'  loss at init: L={L.item():.6f}  (analytic reference for random kappa=9, d=64: '
             f'{analytic_ref.item():.6f})')
        print(f'  t_i: mean={t.mean().item():.4f} std={t.std().item():.4f} min={t.min().item():.4f} '
             f'max={t.max().item():.4f}')
        print(f'  s_i: mean={s.mean().item():.4f} std={s.std().item():.4f} min={s.min().item():.4f} '
             f'max={s.max().item():.4f}')

        # gradient norm w.r.t. head (encoder) weights at init, fresh forward through the real model
        model.zero_grad()
        _, z1m, _ = model(image_1)
        _, z2m, _ = model(image_2)
        loss_full = SBDRCriticLoss(eps=EPS, critic_order=1, symmetric=True, detach_mean=False)
        Lm = loss_full(z1m, z2m)
        Lm.backward()
        full_grad_norm = torch.cat([p.grad.flatten() for p in model.encoder.parameters()
                                    if p.grad is not None]).norm().item()
        full_grads = [p.grad.clone() for p in model.encoder.parameters()]

        model.zero_grad()
        _, z1m2, _ = model(image_1)
        _, z2m2, _ = model(image_2)
        loss_det = SBDRCriticLoss(eps=EPS, critic_order=1, symmetric=True, detach_mean=True)
        Ld = loss_det(z1m2, z2m2)
        Ld.backward()
        det_grad_norm = torch.cat([p.grad.flatten() for p in model.encoder.parameters()
                                   if p.grad is not None]).norm().item()
        det_grads = [p.grad.clone() for p in model.encoder.parameters()]

        diff_norm = torch.cat([(fg - dg).flatten() for fg, dg in zip(full_grads, det_grads)]).norm().item()
        print(f'  2g: head-weight grad norm (full, detach_mean=False): {full_grad_norm:.6e}')
        print(f'  2g: head-weight grad norm (detach_mean=True)       : {det_grad_norm:.6e}')
        print(f'  2g: norm of (full - detached) grad (zbar-mediated component): {diff_norm:.6e}')


# ---------------------------------------------------------------------------
# Section 2e: activation gradient flow
# ---------------------------------------------------------------------------

def section2e(image_1, cfg):
    section('SECTION 2e: gradient flow through activation for out-of-range pre-activations')

    model = fresh_model(cfg, 'clip')
    with torch.no_grad():
        feats = model.backbone(image_1)
        logits_real = model.encoder(feats)
    n_below0 = (logits_real < 0).sum().item()
    n_above1 = (logits_real > 1).sum().item()
    print(f'  real init batch: {logits_real.numel()} coords total, '
         f'{n_below0} below 0, {n_above1} above 1, '
         f'{n_below0 + n_above1} outside [0,1] ({(n_below0 + n_above1) / logits_real.numel():.4f})')

    # synthetic logits deliberately spanning outside [0,1], both directions
    torch.manual_seed(0)
    logits_syn = (torch.rand(8, 16, device=DEVICE) - 0.3) * 6  # spans roughly [-1.8, 4.2]
    out_of_range_mask = (logits_syn < 0) | (logits_syn > 1)
    print(f'  synthetic batch: {out_of_range_mask.sum().item()}/{logits_syn.numel()} coords outside [0,1]')

    for act in ('clip', 'sigmoid'):
        x = logits_syn.clone().requires_grad_(True)
        z = x.clamp(0, 1) if act == 'clip' else torch.sigmoid(x)
        loss = z.sum()
        loss.backward()
        g = x.grad
        g_out = g[out_of_range_mask]
        g_in = g[~out_of_range_mask]
        print(f'  act={act}: grad at out-of-range coords: mean|g|={g_out.abs().mean().item():.3e}, '
             f'max|g|={g_out.abs().max().item():.3e}, exactly-zero frac={float((g_out == 0).float().mean()):.3f}')
        print(f'  act={act}: grad at in-range coords    : mean|g|={g_in.abs().mean().item():.3e} (reference)')


# ---------------------------------------------------------------------------
# Section 3: positive-pair correctness
# ---------------------------------------------------------------------------

def section3(image_1, image_2, index, cfg):
    section('SECTION 3: positive-pair correctness')

    transform_yaml = OmegaConf.to_yaml(cfg.dataset.train_dataset.transform)
    is_two_view = 'NCropsTransform' in transform_yaml
    print(f'  cfg.dataset.train_dataset.transform uses NCropsTransform: {is_two_view}')
    print(f'  transforms config (first 5 lines):')
    for line in transform_yaml.splitlines()[:5]:
        print(f'    {line}')

    identical = torch.equal(image_1, image_2)
    n_diff_pixels = (image_1 != image_2).float().mean().item()
    print(f'  images_i == images_j elementwise (should be False): {identical}')
    print(f'  fraction of differing pixel-values between the two views: {n_diff_pixels:.4f}')
    print(f'  index tensor shape (shared underlying indices for both views): {tuple(index.shape)}, '
         f'first 8 indices: {index[:8].tolist()}')

    # known-permutation index-alignment check, purely at the loss level
    torch.manual_seed(1)
    N, d, kappa = 12, D, KAPPA_REF
    z1 = torch.zeros(N, d)
    for i in range(N):
        idx = torch.randperm(d)[:kappa]
        z1[i, idx] = 1.0
    z2_aligned = z1.clone()  # perfect positive pairing: view2_i corresponds to view1_i
    perm = torch.randperm(N)
    while (perm == torch.arange(N)).any():
        perm = torch.randperm(N)
    z2_shuffled = z1[perm].clone()  # deliberately misaligned by a known permutation

    loss = SBDRCriticLoss(eps=EPS, critic_order=1, symmetric=True)
    L_aligned = loss(z1, z2_aligned).item()
    L_shuffled = loss(z1, z2_shuffled).item()
    print(f'  loss with correct pairing (z2=z1):              {L_aligned:.6f}')
    print(f'  loss with known-permutation-shuffled pairing:    {L_shuffled:.6f}')
    print(f'  |difference|: {abs(L_aligned - L_shuffled):.6f} (0 would indicate the pairing is not used)')


# ---------------------------------------------------------------------------
# Section 5: config / optimizer audit
# ---------------------------------------------------------------------------

def section5():
    section('SECTION 5: config / optimizer audit')

    import glob
    candidates = sorted(glob.glob('logs/cifar10/sbdr64_100/actclip_lambda2_0_*/config.yaml'))
    if candidates:
        saved_cfg = OmegaConf.load(candidates[0])
        print(f'  loaded actual saved config from: {candidates[0]}')
        print(f'  backbone_lr_scale: {saved_cfg.backbone_lr_scale}')
        print(f'  optim: {OmegaConf.to_yaml(saved_cfg.optim).strip()}')
        print(f'  criterion.critic_order (from saved config): {saved_cfg.criterion.get("critic_order")}')
        print(f'  criterion.lambda2 (from saved config): {saved_cfg.criterion.get("lambda2")}')
    else:
        print('  no saved §10 config found on disk')

    print(f'\n  configs/optim/adam.yaml (raw file):')
    print('  ' + open('configs/optim/adam.yaml').read().replace('\n', '\n  '))

    # instantiate the criterion exactly as hydra would and read back the resolved attributes
    inst = SBDRCriticLoss(eps=0.31, critic_order=2, lambda2=1.6, symmetric=True)
    print(f'  instantiated SBDRCriticLoss(critic_order=2, lambda2=1.6): '
         f'self.critic_order={inst.critic_order}, self.lambda2={inst.lambda2}, '
         f'self._effective_lambda2={inst._effective_lambda2}')

    import subprocess
    amp_hits = subprocess.run(['grep', '-rn', '-i', 'autocast\\|float16\\|\\.half()\\|amp\\.',
                               'trainers/', 'models/', 'engine.py'],
                              capture_output=True, text=True).stdout
    print(f'\n  grep for autocast/float16/.half()/amp. in trainers/ models/ engine.py: '
         f'{"NONE FOUND" if not amp_hits.strip() else amp_hits}')


# ---------------------------------------------------------------------------
# Section 7: backbone feature statistics (no training)
# ---------------------------------------------------------------------------

def section7(cfg):
    section('SECTION 7: pretrained-backbone feature statistics (no training, ~512 images)')

    backbone = hydra.utils.instantiate(cfg.backbone).to(DEVICE).eval()
    train_dataset = hydra.utils.instantiate(cfg.dataset.train_dataset)
    loader = engine.dataloader(train_dataset, 64, shuffle=True, drop_last=True)

    feats = []
    n = 0
    with torch.no_grad():
        for data in loader:
            images, labels, index = data
            image_1, image_2 = images
            f = backbone(image_1.to(DEVICE))
            feats.append(f.cpu())
            n += f.size(0)
            if n >= 512:
                break
    feats = torch.cat(feats)[:512]

    mu = feats.mean(0)
    sigma = feats.std(0)
    print(f'  n_images={feats.size(0)}, feature dim={feats.size(1)}')
    print(f'  ||mean vector|| = {mu.norm().item():.4f}')
    print(f'  ||std vector||  = {sigma.norm().item():.4f}')
    print(f'  ratio ||std|| / ||mean|| = {(sigma.norm() / mu.norm()).item():.4f}')
    print(f'  fraction of feature dims that are 0 for every image (post-ReLU dead units): '
         f'{(feats.max(0).values == 0).float().mean().item():.4f}')
    per_sample_dev_norm = (feats - mu.unsqueeze(0)).norm(dim=1)
    print(f'  mean per-sample deviation norm ||f_i - mean|| = {per_sample_dev_norm.mean().item():.4f} '
         f'(vs ||mean||={mu.norm().item():.4f})')


# ---------------------------------------------------------------------------
# Section 6: single-batch overfit test (one fixed batch, a few hundred steps)
# ---------------------------------------------------------------------------

def section6(image_1, image_2, cfg, n_steps=300, log_every=20):
    section(f'SECTION 6: single-batch overfit test (act=clip, lambda2=0, {n_steps} steps on ONE fixed batch)')

    model = fresh_model(cfg, 'clip')
    loss_fn = SBDRCriticLoss(eps=EPS, critic_order=1, symmetric=True)
    params = list(model.get_backbone().parameters()) + list(model.get_training_modules().parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999))

    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        _, z1, _ = model(image_1)
        _, z2, _ = model(image_2)
        L = loss_fn(z1, z2)
        L.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == n_steps:
            with torch.no_grad():
                kappa = (z1 > 0.5).float().sum(1)
                n_distinct = (z1 > 0.5).float().unique(dim=0).size(0)
            print(f'  step {step:>4}: loss={L.item():>10.6f}  kappa mean={kappa.mean().item():>7.3f} '
                 f'std={kappa.std().item():>6.3f}  distinct codes={n_distinct}/{z1.size(0)}')

    analytic_disjoint = torch.log(torch.tensor(EPS)) - torch.log(torch.tensor(float(KAPPA_REF) + EPS))
    print(f'\n  final loss={L.item():.6f}  (disjoint-limit reference ~{analytic_disjoint.item():.4f}, '
         f'identical-code degenerate value = 0.0)')


def main():
    print(f'device: {DEVICE}')
    cfg = build_cfg()
    image_1, image_2, labels, index, cfg = get_one_train_batch(cfg)
    print(f'batch shapes: image_1={tuple(image_1.shape)}, image_2={tuple(image_2.shape)}')

    section1_and_2g(image_1, image_2, cfg)
    section2e(image_1, cfg)
    section3(image_1, image_2, index, cfg)
    section5()
    section7(cfg)
    section6(image_1, image_2, cfg)

    print('\nDone.')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
