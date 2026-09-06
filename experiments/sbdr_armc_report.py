"""
Arm C (HANDOUT.md §3, §14 Task 3) post-hoc diagnostics: mAP is already in
`<logdir>/test_history.json` (written by the normal training loop, same as
every other arm) -- this script only adds what isn't logged for free by
`trainers.cibhash.CIBHashTrainer`:

  - kappa (signed-domain analog: count of +1 bits per sample post-sign),
    dead bits (bits whose population mean is exactly +1 or exactly -1 --
    i.e. carry zero information, the signed-domain analog of Arm B's
    z_bar_u == 0 exact-dead-bit definition), binarity (trivially 1.0: CIBHash's
    eval-time `codes` become exactly {-1,+1} once sign()-binarized -- there is
    no partial-binarization gap to measure in this domain, unlike Arm B's
    [0,1] codes; flagged explicitly rather than reporting a fake number).
  - separation ratio: TRUE positive-pair (two augmented views of the same
    training image, freshly forwarded through the model, same pattern as
    `experiments/sbdr_report.py`'s `positive_pair_overlap`) vs random-pair
    bit-agreement (fraction of matching signs) on the db split -- the
    signed-domain analog of Arm B's overlap-based separation ratio.
  - H(p): mean Bernoulli entropy of the sigmoid probabilities, reconstructed
    exactly as `p = sigmoid(codes)` from the saved db eval-time codes
    (CIBHash's `z` at eval is the raw pre-sigmoid logit, confirmed in
    `models/arch/cibhash.py` -- no extra forward pass needed for this part).
  - code agreement across two independent samples of `b ~ Bernoulli(p)`.
    IMPORTANT CAVEAT (flagged per the task's own instruction to report
    deviations rather than paper over them): this repo's actual CIBHash hash
    layer (`models/layers/signhash.py`) is a DETERMINISTIC sign()+STE, not
    stochastic Bernoulli sampling -- grepped the whole codebase, no
    `torch.bernoulli` call exists anywhere before this script. So this
    agreement number is a *hypothetical* post-hoc quantity (what would two
    independent Bernoulli draws from this trained p look like), not a
    measurement of anything that happens during this repo's actual training
    or inference. It does not reflect real per-step sampling variance, since
    there isn't any in this implementation.

Usage:
    python experiments/sbdr_armc_report.py <logdir> [<logdir> ...] [--device cuda]
"""

import argparse
import json
import os
import sys

import hydra.utils
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sbdr_metrics import _sample_pairs_overlap, positive_negative_separation

POSPAIR_SAMPLES = 1024
RAND_SAMPLE = 3000


def load_outputs(logdir):
    return torch.load(os.path.join(logdir, 'outputs', 'db_best.pth'), weights_only=False, map_location='cpu')


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
    return trainer


def positive_pair_overlap(trainer, device, n_samples=POSPAIR_SAMPLES):
    """True positive pairs: two augmented views of the same training image,
    freshly forwarded (mirrors experiments/sbdr_report.py's positive_pair_overlap)."""
    b1_list, b2_list = [], []
    n = 0
    with torch.no_grad():
        for data in trainer.dataloader['train']:
            images, labels, index = data
            image_1, image_2 = images
            image_1, image_2 = image_1.to(device), image_2.to(device)
            _, _, z1 = trainer.model(image_1)
            _, _, z2 = trainer.model(image_2)
            # model is in eval() -> z1/z2 are raw logits (CIBHash.forward's
            # train/eval branch), sign() to match the db-side binarization
            b1_list.append((torch.sign(z1) > 0).float().cpu())
            b2_list.append((torch.sign(z2) > 0).float().cpu())
            n += z1.size(0)
            if n >= n_samples:
                break
    b1 = torch.cat(b1_list)[:n_samples]
    b2 = torch.cat(b2_list)[:n_samples]
    return (b1 * b2).sum(1)  # co-active overlap (sum-AND), same definition as Arm B


def report_one(logdir, device='cuda'):
    logdir = logdir.rstrip('/')
    db = load_outputs(logdir)
    codes = db['codes']  # raw pre-sigmoid logits, (N, nbit)
    prob = torch.sigmoid(codes)

    b01 = (torch.sign(codes) > 0).float()  # {0, 1}, matches Arm B's "active" predicate

    kappa = b01.sum(1)
    bit_mean = b01.mean(0)  # (nbit,)
    dead_bits = ((bit_mean == 0.0) | (bit_mean == 1.0)).sum().item()

    torch.manual_seed(0)
    rand_overlap = _sample_pairs_overlap(b01, n_sample=RAND_SAMPLE)

    trainer = load_trainer(logdir, device)
    pos_overlap = positive_pair_overlap(trainer, device)

    sep = positive_negative_separation(pos_overlap, rand_overlap)

    H = (-prob * torch.log(prob.clamp_min(1e-8)) - (1 - prob) * torch.log((1 - prob).clamp_min(1e-8)))
    H_mean = H.mean().item()

    g = torch.Generator().manual_seed(0)
    b1 = torch.bernoulli(prob, generator=g)
    b2 = torch.bernoulli(prob, generator=g)
    agreement = (b1 == b2).float().mean().item()

    res = {
        'logdir': logdir,
        'kappa_mean': kappa.mean().item(),
        'kappa_std': kappa.std().item(),
        'dead_bits': dead_bits,
        'nbit': b01.size(1),
        'binarity': 1.0,  # trivial in signed domain -- see module docstring
        'separation': sep,
        'bernoulli_entropy_mean_nats': H_mean,
        'hypothetical_two_sample_bernoulli_agreement': agreement,
    }
    with open(os.path.join(logdir, 'armc_report.json'), 'w') as f:
        json.dump(res, f, indent=2)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdirs', nargs='+')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    for logdir in args.logdirs:
        res = report_one(logdir, args.device)
        print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
