import torch
from torch import nn

from models.arch.base import BaseNet


class SBDR(BaseNet):
    """
    Arm B (HANDOUT.md §2.3, §3): "full proposal" -- CIBHash's encoder head, but with
    a bounded activation instead of Bernoulli sampling + STE. No sign layer, no
    stochasticity: z is deterministic given x, and lives in [0,1]^d throughout
    training and eval (unlike CIBHash, whose train-time `z` is post-sign in {-1,+1}
    while eval-time `z` is raw logits).

    act='clip' (default, HANDOUT-preferred): z = x.clamp(0, 1). Gradient stays
    linear in the interior instead of vanishing at saturation like sigmoid, which
    matters because the dynamics in HANDOUT §0.2 drive codes to the box boundary.
    act='sigmoid': z = sigmoid(x).
    act='ste_clip' (2026-09-04, HANDOUT §11 / Task C): value is clamp(x,0,1), but
    the backward pass is straight-through -- gradient w.r.t. the pre-activation is
    the identity everywhere, including outside [0,1] (unlike plain `clip`, which
    is exactly zero there -- see §11 / diagnostic item 2e). Implemented via the
    standard detach trick: `z = x + (clamp(x,0,1) - x).detach()`.

    Initialization fix (2026-09-04 continued investigation, HANDOUT §11): measured
    pre-activation std at init is ~0.073, far below the ~0.46 needed for inner
    products <z,z> ~ kappa*std^2*d to be commensurate with eps rather than swamped
    by it (§9/§10's diagnosed Failure 2). Two independent knobs, off by default
    (`feature_norm='none'`, `head_init_gain=1.0`, i.e. unchanged from the original
    architecture unless explicitly requested):

    - `feature_norm`: `none` (default) | `standardize` (BatchNorm1d with
      `affine=False` -- running mean/std only, no learnable scale/shift) |
      `batchnorm` (BatchNorm1d with `affine=True`) on the backbone features,
      applied before the encoder's first Linear.
    - `head_init_gain`: multiplies the final Linear layer's default
      (Kaiming-uniform) weight init in-place post-construction. Cruder,
      architecture-independent fallback to widen the pre-activation distribution
      without touching feature statistics.
    """

    def __init__(self,
                backbone: nn.Module,
                nbit: int,
                nclass: int,
                act: str = 'clip',
                feature_norm: str = 'none',
                head_init_gain: float = 1.0,
                **kwargs):
        super().__init__(backbone, nbit, nclass, **kwargs)

        assert act in ('clip', 'sigmoid', 'ste_clip'), \
            f'act must be `clip`, `sigmoid`, or `ste_clip`, got {act!r}'
        self.act = act

        assert feature_norm in ('none', 'standardize', 'batchnorm'), \
            f'feature_norm must be `none`, `standardize`, or `batchnorm`, got {feature_norm!r}'
        self.feature_norm = feature_norm
        if feature_norm == 'none':
            self.norm = nn.Identity()
        elif feature_norm == 'standardize':
            self.norm = nn.BatchNorm1d(self.backbone.features_size, affine=False)
        else:  # batchnorm
            self.norm = nn.BatchNorm1d(self.backbone.features_size, affine=True)

        self.encoder = nn.Sequential(nn.Linear(self.backbone.features_size, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, self.nbit))

        self.head_init_gain = head_init_gain
        if head_init_gain != 1.0:
            with torch.no_grad():
                self.encoder[-1].weight.mul_(head_init_gain)

    def get_training_modules(self):
        return nn.ModuleDict({'encoder': self.encoder, 'norm': self.norm})

    def forward(self, x):
        x = self.backbone(x)
        feat = self.norm(x)
        logits = self.encoder(feat)
        if self.act == 'clip':
            z = logits.clamp(0, 1)
        elif self.act == 'sigmoid':
            z = torch.sigmoid(logits)
        else:  # ste_clip
            clipped = logits.clamp(0, 1)
            z = logits + (clipped - logits).detach()

        # no train/eval branch: z is deterministic and already the code in both
        # modes (unlike CIBHash's sign-then-logits split, HANDOUT §1 table)
        return x, z, z
