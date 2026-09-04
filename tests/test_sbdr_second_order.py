"""
Checks for the second-order critic in `models.loss.sbdr.SBDRCriticLoss` (HANDOUT.md
§2.2b / Task A).

Two things are pinned down:

1. The vectorized C-matrix path (O(K*d^2): build C = zall^T @ zall / K once, then
   z_i^T C z_i per row) matches a naive O(K^2*d) double loop that never forms C,
   using the algebraic identity z_i^T C z_i = (1/K) sum_j <z_i, z_j>^2.
2. critic_order=1 is bit-identical to critic_order=2 with lambda2=0 -- i.e. order-1
   is a special case of order-2, not a separate formula.

Run directly (no pytest required, though pytest will also collect it):

    python tests/test_sbdr_second_order.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loss.sbdr import SBDRCriticLoss


def _naive_quad_form(za, zall):
    """
    z_i^T C z_i via the O(K^2*d) double sum (1/K) sum_j <z_i,z_j>^2, never forming
    the d x d matrix C. This is the reference the vectorized C-based path is
    checked against.
    """
    K = zall.size(0)
    N = za.size(0)
    out = torch.zeros(N)
    for i in range(N):
        acc = 0.0
        for j in range(K):
            acc = acc + torch.dot(za[i], zall[j]) ** 2
        out[i] = acc / K
    return out


def _naive_one_way(za, zb, zall, eps, lambda2):
    """Reference implementation of `_one_way`, quad form via `_naive_quad_form`."""
    zbar = zall.mean(0)
    t = (za * zbar).sum(1)
    s = (za * zb).sum(1)
    if lambda2 != 0.0:
        t = t + lambda2 * _naive_quad_form(za, zall)
        s = s + lambda2 * s.pow(2)
    t = t + eps
    s = s + eps
    return (t.log() - s.log()).mean()


def test_c_matrix_path_matches_naive_double_loop():
    """Vectorized O(K*d^2) C-matrix path vs. naive O(K^2*d) double loop, ~1e-6."""
    torch.manual_seed(0)
    N, d, eps, lambda2 = 12, 16, 0.31, 1.6

    z1 = torch.rand(N, d)
    z2 = torch.rand(N, d)
    zall = torch.cat([z1, z2], 0)

    loss = SBDRCriticLoss(eps=eps, critic_order=2, lambda2=lambda2, symmetric=False)
    got = loss._one_way(z1, z2, zall)
    exp = _naive_one_way(z1, z2, zall, eps, lambda2)

    diff = (got - exp).abs().item()
    assert diff < 1e-6, f'C-matrix path vs naive double loop: got={got.item():.8f} ' \
                        f'exp={exp.item():.8f} diff={diff:.2e}'
    print(f'    C-matrix _one_way = {got.item():.8f}')
    print(f'    naive double-loop = {exp.item():.8f}')
    print(f'    |diff|            = {diff:.2e}  (< 1e-6)')


def test_c_matrix_quad_form_matches_naive_per_row():
    """Same check at the level of the raw per-row quadratic form, not just the scalar loss."""
    torch.manual_seed(1)
    N, K, d = 7, 20, 10
    za = torch.randn(N, d, dtype=torch.float64)
    zall = torch.randn(K, d, dtype=torch.float64)

    C = zall.t().matmul(zall) / K
    got = (za.matmul(C) * za).sum(1)
    exp = _naive_quad_form(za, zall)

    max_diff = (got - exp).abs().max().item()
    assert max_diff < 1e-6, f'max per-row diff {max_diff:.2e}'
    print(f'    max |C-matrix - naive| over {N} rows = {max_diff:.2e}  (< 1e-6)')


def test_lambda2_zero_matches_order_1_exactly():
    """critic_order=1 must be bit-identical to critic_order=2 with lambda2=0."""
    torch.manual_seed(2)
    N, d, eps = 8, 24, 0.31
    z1 = torch.rand(N, d)
    z2 = torch.rand(N, d)

    order1 = SBDRCriticLoss(eps=eps, critic_order=1, symmetric=True)
    order2_lambda0 = SBDRCriticLoss(eps=eps, critic_order=2, lambda2=0.0, symmetric=True)
    # also: critic_order=1 must IGNORE a nonzero lambda2 passed alongside it
    order1_nonzero_lambda2_kwarg = SBDRCriticLoss(eps=eps, critic_order=1, lambda2=3.2, symmetric=True)

    L1 = order1(z1, z2)
    L2 = order2_lambda0(z1, z2)
    L3 = order1_nonzero_lambda2_kwarg(z1, z2)

    assert torch.equal(L1, L2), f'order=1 ({L1.item():.10f}) != order=2,lambda2=0 ({L2.item():.10f})'
    assert torch.equal(L1, L3), \
        f'order=1 must ignore a nonzero lambda2 kwarg: {L1.item():.10f} != {L3.item():.10f}'
    print(f'    order=1                          loss = {L1.item():.10f}')
    print(f'    order=2, lambda2=0                loss = {L2.item():.10f}  (bit-identical)')
    print(f'    order=1, lambda2=3.2 kwarg ignored loss = {L3.item():.10f}  (bit-identical)')


def test_lambda2_default_is_taylor_coefficient():
    """Default lambda2 (order=2, lambda2 unset) must be exactly 1/(2*eps)."""
    for eps in (0.31, 1.0, 0.05):
        loss = SBDRCriticLoss(eps=eps, critic_order=2)
        expected = 1.0 / (2.0 * eps)
        assert abs(loss.lambda2 - expected) < 1e-12, f'eps={eps}: {loss.lambda2} != {expected}'
    print('    default lambda2 == 1/(2*eps) for eps in {0.31, 1.0, 0.05}')


def test_nonzero_lambda2_changes_the_loss():
    """Sanity: order=2 with lambda2 != 0 must actually differ from order=1 (no silent no-op)."""
    torch.manual_seed(3)
    N, d, eps = 16, 32, 0.31
    z1 = torch.rand(N, d)
    z2 = torch.rand(N, d)

    order1 = SBDRCriticLoss(eps=eps, critic_order=1, symmetric=True)
    order2 = SBDRCriticLoss(eps=eps, critic_order=2, lambda2=1.6, symmetric=True)

    L1 = order1(z1, z2)
    L2 = order2(z1, z2)
    assert (L1 - L2).abs().item() > 1e-4, f'order=2,lambda2=1.6 suspiciously close to order=1: ' \
                                          f'{L1.item():.8f} vs {L2.item():.8f}'
    print(f'    order=1              loss = {L1.item():.8f}')
    print(f'    order=2, lambda2=1.6 loss = {L2.item():.8f}  (differs, as expected)')


def test_gradient_flows_into_c_and_zbar():
    """z_bar and C must not be detached: grad must reach every row of z1 through both paths."""
    torch.manual_seed(4)
    N, d, eps = 6, 8, 0.31
    z1 = torch.rand(N, d, requires_grad=True)
    z2 = torch.rand(N, d, requires_grad=True)

    loss = SBDRCriticLoss(eps=eps, critic_order=2, lambda2=1.6, symmetric=True)
    L = loss(z1, z2)
    L.backward()

    assert z1.grad is not None and z1.grad.abs().sum().item() > 0
    assert z2.grad is not None and z2.grad.abs().sum().item() > 0
    print(f'    z1.grad abs-sum = {z1.grad.abs().sum().item():.6f}')
    print(f'    z2.grad abs-sum = {z2.grad.abs().sum().item():.6f}')


def main():
    tests = [test_c_matrix_path_matches_naive_double_loop,
             test_c_matrix_quad_form_matches_naive_per_row,
             test_lambda2_zero_matches_order_1_exactly,
             test_lambda2_default_is_taylor_coefficient,
             test_nonzero_lambda2_changes_the_loss,
             test_gradient_flows_into_c_and_zbar]
    for t in tests:
        print(f'\n[{t.__name__}]')
        print(f'  {t.__doc__.strip()}' if t.__doc__ else '')
        t()
    print(f'\nAll {len(tests)} checks passed.')


if __name__ == '__main__':
    main()
