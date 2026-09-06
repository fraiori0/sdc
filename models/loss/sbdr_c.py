from models.loss.base import BaseLoss
from models.loss.sbdr import SBDRCriticLoss


class CIBHashSBDROnlyLoss(BaseLoss):
    """
    Arm C (HANDOUT.md §3, 2026-09-05): CIBHash's architecture and hash layer
    (`models/arch/cibhash.py`, `models/layers/signhash.py`) unchanged; the loss
    is *only* our critic (`SBDRCriticLoss`) applied to the sigmoid probabilities
    `p_i, p_j` -- CIBHash's own NtXent+KL loss (`models/loss/cibhash.py`) is not
    used at all. Contrast with Arm D (`models/loss/sbdr_aux.py`), where our loss
    is an auxiliary *addition* to CIBHash's own loss; Arm C is the clean
    ablation -- identical architecture and sampling, only the objective differs.

    Matches `CIBHashLoss`'s forward signature (`prob_i, prob_j, z_i, z_j, f_i,
    f_j`) so this plugs directly into the existing `trainers.cibhash.CIBHashTrainer`
    with no trainer subclass and no arch change (same wiring as Arm D).
    `z_i, z_j, f_i, f_j` are accepted but unused -- per §3's scope caveat, the
    loss touches only `p` and never the sampled/STE `b`, so no gradient flows
    through the hash layer during training (that reduced-STE-variance claim
    belongs to Arm D, not Arm C).

    `p_i, p_j` are already in [0,1] (sigmoid output), matching `SBDRCriticLoss`'s
    domain assertion directly -- no rescaling needed.
    """

    def __init__(self, eps=0.31, symmetric=True, detach_mean=False,
                critic_order=1, lambda2=None, **kwargs):
        super().__init__(**kwargs)
        self.sbdr_loss = SBDRCriticLoss(eps=eps, symmetric=symmetric, detach_mean=detach_mean,
                                        critic_order=critic_order, lambda2=lambda2)

    def forward(self, prob_i, prob_j, z_i, z_j, f_i, f_j):
        loss = self.sbdr_loss(prob_i, prob_j)
        self.losses['contrast'] = self.sbdr_loss.losses['contrast']
        self.losses['kappa'] = self.sbdr_loss.losses['kappa']
        return loss
