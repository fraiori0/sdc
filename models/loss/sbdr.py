import torch

from models.loss.base import BaseLoss


class SBDRCriticLoss(BaseLoss):
    """
    Log-dot-product InfoNCE critic for codes z in [0,1]^d (HANDOUT.md §0, §2.2).

        g(z_a, z_b) = log(<z_a, z_b> + eps)

    exp(g) is linear in z_b, so the InfoNCE denominator collapses to a single mean
    vector z_bar = mean(zall, dim=0):

        (1/K) sum_j exp(g(z_i, z_j)) = <z_i, z_bar> + eps

    which makes this O(K*d) rather than the O(K^2*d) full similarity matrix that
    CIBHash's NtXentLoss builds (models/loss/cibhash.py).

    MUST NOT detach z_bar by default: sample i contributes to z_bar, and that
    self-contribution term is ~49% of the effective gradient (measured, HANDOUT
    §0.1). `detach_mean` is exposed as an ablation flag only, default False.

    Second-order critic (HANDOUT §2.2b, order = 1|2). `g` is the log of the first
    two terms of the Taylor expansion of an exponential critic:

        eps * exp(s/eps) = eps + s + s^2/(2*eps) + ...   (s = <z_i, z_j>)

    order=1 (default) is the plain critic above -- the linear truncation. order=2
    keeps the quadratic term, weighted by lambda2 (default 1/(2*eps), the exact
    Taylor coefficient):

        denominator_i = eps + <z_i, z_bar> + lambda2 * z_i^T C z_i,  C = (1/K) sum_j z_j z_j^T
        numerator_i   = eps + s+_i + lambda2 * (s+_i)^2,             s+_i = <z_i, z_j> (positive pair)
        L_i = log(denominator_i) - log(numerator_i)

    order=infinity recovers a plain dot-product critic with temperature eps, i.e.
    essentially CIBHash -- order is a knob interpolating sparse-binary (1) to
    dense-SOTA (infinity, not implemented here).

    The quadratic form z_i^T C z_i is evaluated through the d x d matrix
    C = zall^T @ zall / K (O(K*d^2) to build, O(N*d^2) to apply to N query rows),
    never through the O(K^2*d) pairwise double sum that C is mathematically
    equivalent to (z_i^T C z_i = (1/K) sum_j <z_i,z_j>^2).

    C is NOT detached, for the same reason z_bar is not (see above): sample i
    contributes to C via zall, and detaching would silently remove part of the
    self-contribution gradient.

    critic_order=1 forces the effective lambda2 to 0 regardless of the `lambda2`
    kwarg, so it is bit-identical to critic_order=2 with lambda2=0 explicitly --
    that identity is what pins order-1 as a special case of order-2, not a
    separate code path.
    """

    def __init__(self, eps=0.31, symmetric=True, detach_mean=False,
                critic_order=1, lambda2=None, delta=0.0, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.symmetric = symmetric
        self.detach_mean = detach_mean

        assert critic_order in (1, 2), f'critic_order must be 1 or 2, got {critic_order!r}'
        self.critic_order = critic_order
        self.lambda2 = (1.0 / (2.0 * eps)) if lambda2 is None else lambda2

        # delta (2026-09-04, HANDOUT §11 / Task C): mean offset in the negative
        # (denominator) term only, t_i = <z_i,z_bar> - delta*||z_i||_1 + eps,
        # clamped at eps/2. For a dead unit (z_bar_u=0, w_u=0) this makes the
        # gradient contribution from that unit exactly -delta/t_i < 0 (descent
        # increases it) instead of exactly 0 -- see class docstring. Default 0.0
        # reproduces the undamped critic exactly (no clamp ever binds when
        # delta=0, since t_i = <z_i,z_bar>+eps >= eps > eps/2 always).
        self.delta = delta

    @property
    def _effective_lambda2(self):
        # order=1 is the lambda2=0 special case of order=2, always -- see class docstring
        return self.lambda2 if self.critic_order == 2 else 0.0

    def _one_way(self, za, zb, zall):
        zbar = zall.mean(0)
        if self.detach_mean:  # ABLATION ONLY -- see class docstring
            zbar = zbar.detach()

        t = (za * zbar).sum(1)
        s = (za * zb).sum(1)  # s+_i, the positive-pair dot product

        lambda2 = self._effective_lambda2
        if lambda2 != 0.0:
            K = zall.size(0)
            C = zall.t().matmul(zall) / K  # (d, d); NOT detached -- see class docstring
            quad_t = (za.matmul(C) * za).sum(1)  # z_i^T C z_i, (N,)
            t = t + lambda2 * quad_t
            s = s + lambda2 * s.pow(2)

        if self.delta != 0.0:
            t = t - self.delta * za.sum(1)  # za >= 0 always, so sum == L1 norm

        t = t + self.eps
        s = s + self.eps
        t = t.clamp(min=self.eps / 2)
        return (t.log() - s.log()).mean()

    def forward(self, z1, z2):
        assert z1.min() >= -1e-6 and z1.max() <= 1 + 1e-6, \
            f'SBDRCriticLoss expects codes in [0,1], got z1 range [{z1.min():.4g}, {z1.max():.4g}]'
        assert z2.min() >= -1e-6 and z2.max() <= 1 + 1e-6, \
            f'SBDRCriticLoss expects codes in [0,1], got z2 range [{z2.min():.4g}, {z2.max():.4g}]'

        zall = torch.cat([z1, z2], 0)
        L = self._one_way(z1, z2, zall)
        if self.symmetric:
            L = 0.5 * (L + self._one_way(z2, z1, zall))

        self.losses['contrast'] = L
        # realised sparsity, matching the > 0.5 predicate used at eval (utils/hashing.py)
        self.losses['kappa'] = (z1 > 0.5).float().sum(1).mean()
        return L
