import logging
import os

import torch

from trainers.base import BaseTrainer


class SBDRTrainer(BaseTrainer):
    """
    Arm B (HANDOUT.md §2.3, §2.4, §3): SBDR's encoder (`models.arch.sbdr.SBDR`)
    returns `(feats, z, z)` -- a single deterministic code in [0,1]^d, no sampling.
    The criterion (`models.loss.sbdr.SBDRCriticLoss`) consumes only the two
    augmented views' codes.

    `inference_one_batch` MUST return both `codes` (binarized at 0.5, unit domain)
    and `codes_cont` (the raw [0,1] code) -- both keys contain 'codes' and are
    picked up automatically by `experiments/train_helper.py` / `test_hashing.py`
    (HANDOUT §1, §2.4). This *is* Experiment 1 (binarization-gap check).

    Collapse-detection diagnostics (continued work, 2026-09-03, §9 follow-up):
    every training batch, `train_one_batch` also computes and logs (via the same
    `meters` dict the rest of the trainer uses, so these flow into
    `train_history.json` as `train_<key>` for free):

      - `usage_mean` / `usage_std`: mean/std across the `d` bits of the CONTINUOUS
        per-bit batch mean `z_bar_u = zall.mean(0)` -- the exact quantity the loss
        consumes (§0.1, §2.2b), not the binarized active-fraction (that one is
        reported post-hoc in `utils.sbdr_metrics.usage_stats` instead).
      - `dead_bits_exact`: bits where `z_bar_u == 0.0` exactly (every sample in
        the batch is exactly 0 for that bit -- the collapse mechanism's claim is
        that this gives *exactly* zero gradient for that bit, permanently, under
        `act=clip`, since a hard-saturated 0 has zero local slope).
      - `dead_bits_near`: bits where `z_bar_u < 1e-4` (inclusive of the exact
        zeros above) -- distinguishing exact-zero from near-zero-but-nonzero is
        the entire point: near-zero bits still receive a (tiny) gradient and can
        in principle recover, exact-zero bits cannot.
      - `overlap_std`: std of the pairwise (binarized, `>0.5`) overlap values
        within the batch (both augmented views pooled) -- a per-batch collapse
        indicator independent of the epoch-end `utils.sbdr_metrics` diagnostics.
      - `kappa_std`: std of per-sample active-bit counts within the batch
        (`kappa` itself, the mean, was already logged before this change).

    A loud `logging.warning` fires whenever the epoch-averaged `dead_bits_exact`
    rises versus the previous epoch. A per-epoch snapshot of the `d`-dim
    `z_bar` vector (epoch-averaged over all training batches) is appended to
    `self.usage_history` and saved to `<logdir>/usage_history.pt` (a
    `(n_epochs_so_far, d)` tensor, overwritten each epoch) so bit death can be
    inspected as progressive vs. sudden.
    """

    def __init__(self, config):
        super().__init__(config)
        self._epoch_zbar_sum = None
        self._epoch_zbar_batches = 0
        self._prev_dead_bits_exact = None
        self.usage_history = []

    def forward_one_batch(self, images):
        feats, z, _ = self.model(images)
        return {
            'feats': feats,
            'codes': z,
        }

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        images, labels, index = data
        if isinstance(images, (tuple, list)):
            image_1, image_2 = images
            images = torch.cat([image_1, image_2], dim=0)
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            _, z, _ = self.model(images)

        return {
            'codes': (z > 0.5).float(),
            'codes_cont': z,
            'labels': labels,
        }

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        images, labels, index = data
        images_i, images_j = images
        images_i, images_j, labels = images_i.to(device), images_j.to(device), labels.to(device)

        # clear gradient
        self.optimizer.zero_grad()

        _, z_i, _ = self.model(images_i)
        _, z_j, _ = self.model(images_j)
        loss = self.criterion(z_i, z_j)

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item())
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item())

        # --- collapse-detection diagnostics (§9 follow-up, see class docstring) ---
        with torch.no_grad():
            zall = torch.cat([z_i, z_j], 0).detach()
            zbar = zall.mean(0)  # (d,), continuous -- exactly what the loss consumes
            active = (zall > 0.5).float()
            kappa = active.sum(1)

            dead_exact = (zbar == 0).sum().item()
            dead_near = (zbar < 1e-4).sum().item()  # inclusive of the exact zeros

            ov = active.matmul(active.t())
            iu = torch.triu_indices(ov.size(0), ov.size(0), offset=1, device=ov.device)
            overlap_std = ov[iu[0], iu[1]].std().item()

            meters['usage_mean'].update(zbar.mean().item())
            meters['usage_std'].update(zbar.std().item())
            meters['dead_bits_exact'].update(dead_exact)
            meters['dead_bits_near'].update(dead_near)
            meters['overlap_std'].update(overlap_std)
            meters['kappa_std'].update(kappa.std().item())

            zbar_cpu = zbar.cpu()
            if self._epoch_zbar_sum is None:
                self._epoch_zbar_sum = zbar_cpu.clone()
            else:
                self._epoch_zbar_sum += zbar_cpu
            self._epoch_zbar_batches += 1

    def train_one_epoch(self, **kwargs):
        self._epoch_zbar_sum = None
        self._epoch_zbar_batches = 0

        meters = super().train_one_epoch(**kwargs)

        if self._epoch_zbar_sum is not None:
            epoch_usage = self._epoch_zbar_sum / self._epoch_zbar_batches
            self.usage_history.append(epoch_usage)
            logdir = self.config.get('logdir')
            if logdir:
                torch.save(torch.stack(self.usage_history), os.path.join(logdir, 'usage_history.pt'))

        if 'dead_bits_exact' in meters:
            cur = meters['dead_bits_exact'].avg
            if self._prev_dead_bits_exact is not None and cur > self._prev_dead_bits_exact:
                logging.warning(f'!!! DEAD-BIT COUNT RISING: epoch-avg dead_bits_exact '
                                f'{self._prev_dead_bits_exact:.2f} -> {cur:.2f} !!!')
            self._prev_dead_bits_exact = cur

        return meters
