from models.loss.base import BaseLoss
from models.loss.cibhash import CIBHashLoss
from models.loss.sbdr import SBDRCriticLoss


class CIBHashSBDRAuxLoss(BaseLoss):
    """
    Arm D (HANDOUT.md §3, "drop-in gain"): CIBHash's architecture and Bernoulli/STE
    sampling are unchanged; our critic is added as an auxiliary regularizer on the
    probabilities `p`:

        loss = CIBHashLoss(prob, z, feats) + beta * SBDRCriticLoss(prob_i, prob_j)

    If beta > 0 beats beta = 0, that is a self-contained result requiring no
    architectural argument -- work-order Step 1's gate.

    Matches CIBHashLoss's forward signature (prob_i, prob_j, z_i, z_j, f_i, f_j), so
    this plugs directly into the existing trainers.cibhash.CIBHashTrainer with no
    trainer subclass and no arch change (HANDOUT §2.4: "Arms C and D import
    models.arch.cibhash.CIBHash untouched and only change what the loss consumes").

    With beta=0 (and temperature/kl_beta left at their CIBHash defaults) this reduces
    exactly to CIBHashLoss, i.e. Arm A -- that identity is what the Step 1 gate
    compares against.
    """

    def __init__(self,
                temperature=0.3, kl_beta=0.001,  # unchanged CIBHash hyperparameters
                eps=0.31, beta=1.0, symmetric=True, detach_mean=False,  # our term
                **kwargs):
        super().__init__(**kwargs)
        self.cibhash_loss = CIBHashLoss(temperature=temperature, beta=kl_beta)
        self.sbdr_loss = SBDRCriticLoss(eps=eps, symmetric=symmetric, detach_mean=detach_mean)
        self.beta = beta

    def forward(self, prob_i, prob_j, z_i, z_j, f_i, f_j):
        cibhash_total = self.cibhash_loss(prob_i, prob_j, z_i, z_j, f_i, f_j)
        sbdr_total = self.sbdr_loss(prob_i, prob_j)
        loss = cibhash_total + self.beta * sbdr_total

        # both wrapped losses use the key 'contrast' internally; prefix rather than
        # let one clobber the other in the merged dict the trainer logs from
        self.losses['cibhash_kl'] = self.cibhash_loss.losses['kl']
        self.losses['cibhash_contrast'] = self.cibhash_loss.losses['contrast']
        self.losses['sbdr_contrast'] = self.sbdr_loss.losses['contrast']
        self.losses['kappa'] = self.sbdr_loss.losses['kappa']

        return loss
