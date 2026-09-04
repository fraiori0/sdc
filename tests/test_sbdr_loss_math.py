"""
Pure loss-level diagnostics for models/loss/sbdr.py (2026-09-03 continued
investigation into the §9/§10 Arm B collapse). No network involved -- all `z`
are hand-constructed float64 tensors in [0,1]. Corresponds to Task items 2a-2d,
2f, and 4 of the diagnostic request.

Run directly (also pytest-collectible):

    python tests/test_sbdr_loss_math.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loss.sbdr import SBDRCriticLoss

torch.set_default_dtype(torch.float64)


def _rand_sparse(n, d, kappa, generator=None):
    z = torch.zeros(n, d)
    for i in range(n):
        idx = torch.randperm(d, generator=generator)[:kappa]
        z[i, idx] = 1.0
    return z


# ---------------------------------------------------------------------------
# 2a. Numerical vs analytic gradient (central difference)
# ---------------------------------------------------------------------------

def test_2a_central_difference_order1_and_order2():
    """Central-difference dL/dz vs autograd, critic_order in {1,2}, tol ~1e-8."""
    torch.manual_seed(0)
    N, d, kappa, eps = 5, 16, 4, 0.31
    z1 = _rand_sparse(N, d, kappa) * 0.97 + 0.01  # keep strictly inside (0,1), avoid clip boundary
    z2 = _rand_sparse(N, d, kappa) * 0.97 + 0.01

    for order, lambda2 in [(1, 0.0), (2, 1.6)]:
        loss = SBDRCriticLoss(eps=eps, critic_order=order, lambda2=lambda2, symmetric=True)

        z1a = z1.clone().requires_grad_(True)
        L = loss(z1a, z2)
        L.backward()
        analytic = z1a.grad.clone()

        h = 1e-6
        numeric = torch.zeros_like(z1)
        for i in range(N):
            for u in range(d):
                zp = z1.clone(); zp[i, u] += h
                zm = z1.clone(); zm[i, u] -= h
                Lp = loss(zp, z2)
                Lm = loss(zm, z2)
                numeric[i, u] = (Lp - Lm) / (2 * h)

        max_abs_diff = (analytic - numeric).abs().max().item()
        max_rel_diff = ((analytic - numeric).abs() / (numeric.abs() + 1e-12)).max().item()
        print(f'    order={order} lambda2={lambda2}: max|analytic-numeric|={max_abs_diff:.3e}, '
             f'max relative={max_rel_diff:.3e}')
        assert max_abs_diff < 1e-7, f'order={order}: {max_abs_diff}'


# ---------------------------------------------------------------------------
# 2b. z_bar carries gradient -- the "w" term
# ---------------------------------------------------------------------------

def test_2b_mean_carries_gradient_w_term():
    """
    Analytic claim: d/dz_k [sum_i log(t_i)] = zbar/t_k + w for k in the t-batch,
    w = (1/K) sum_i z_i/t_i (row-constant). Detaching zbar removes exactly w.
    """
    torch.manual_seed(1)
    N, d, kappa, eps = 6, 20, 5, 0.31
    z1 = _rand_sparse(N, d, kappa) * 0.9 + 0.02
    z2 = _rand_sparse(N, d, kappa) * 0.9 + 0.02

    # asymmetric one-way to isolate the t-term cleanly (matches the derivation)
    loss_full = SBDRCriticLoss(eps=eps, critic_order=1, symmetric=False, detach_mean=False)
    loss_detached = SBDRCriticLoss(eps=eps, critic_order=1, symmetric=False, detach_mean=True)

    z1_full = z1.clone().requires_grad_(True)
    zall = torch.cat([z1_full, z2], 0)
    L_full = loss_full._one_way(z1_full, z2, zall)
    g_full, = torch.autograd.grad(L_full, z1_full)

    z1_det = z1.clone().requires_grad_(True)
    zall2 = torch.cat([z1_det, z2], 0)
    L_det = loss_detached._one_way(z1_det, z2, zall2)
    g_det, = torch.autograd.grad(L_det, z1_det)

    # manual w = (1/K) sum_i z_i / t_i  (t_i computed WITHOUT gradient, plain forward values)
    with torch.no_grad():
        zbar = zall.mean(0)
        t = (z1 * zbar).sum(1) + eps  # (N,)
        K = zall.size(0)
        w = (z1 / t.unsqueeze(1)).sum(0) / K  # (d,), row-constant claim

    diff = g_full - g_det  # should equal w, broadcast to every row, and the LOSS is a .mean(), so
    # g_full = d/dz_k [ (1/N) sum_i log(t_i) ], i.e. everything above scaled by 1/N
    w_scaled = w / N

    print(f'    g_full row0 vs row1 (detach-diff)      : {diff[0, :4].tolist()} / {diff[1, :4].tolist()}')
    print(f'    w_scaled (row-constant, first 4 dims)  : {w_scaled[:4].tolist()}')

    # every row of (g_full - g_det) must equal w_scaled (row-constant) to machine precision.
    # (g_full includes the -log(s_i) numerator term too, but that term does not depend on
    # detach_mean at all, so it cancels exactly in the subtraction -- this diff isolates the
    # t-term's zbar-dependence cleanly without needing to split t from s.)
    row_constant_err = (diff - w_scaled.unsqueeze(0)).abs().max().item()
    print(f'    max |(g_full - g_det)_row - w_scaled| = {row_constant_err:.3e}')
    assert row_constant_err < 1e-10, row_constant_err

    # Independent check of the full analytic decomposition of the T-TERM ALONE (log(t_i) part
    # only, excluding -log(s_i), which is unaffected by zbar and would otherwise have to be
    # added back in separately): d/dz_k [mean_i log(t_i)] == zbar/t_k/N + w_scaled for every k.
    z1_tonly = z1.clone().requires_grad_(True)
    zall_tonly = torch.cat([z1_tonly, z2], 0)
    zbar_tonly = zall_tonly.mean(0)
    t_tonly = (z1_tonly * zbar_tonly).sum(1) + eps
    L_tonly = t_tonly.log().mean()
    g_tonly, = torch.autograd.grad(L_tonly, z1_tonly)

    zbar_over_t = zbar.unsqueeze(0) / t.unsqueeze(1) / N  # (N, d)
    reconstructed = zbar_over_t + w_scaled.unsqueeze(0)
    recon_err = (g_tonly - reconstructed).abs().max().item()
    print(f'    max |d/dz[mean log(t)] - (zbar/t_k + w)| (t-term-only decomposition) = {recon_err:.3e}')
    assert recon_err < 1e-10, recon_err

    # magnitude ratio |w| / |zbar/t_k + w| per row, expect ~0.49
    ratios = w_scaled.norm() / reconstructed.norm(dim=1)
    print(f'    ratio |w| / |zbar/t_k + w| per row: {ratios.tolist()}')
    print(f'    mean ratio: {ratios.mean().item():.4f} (expected ~0.49)')

    # grep the loss file for detach/no_grad/.data on zbar or C
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'models/loss/sbdr.py')).read()
    import re
    hits = []
    for m in re.finditer(r'.*\.detach\(\)|.*no_grad\(\)|.*\.data\b', src):
        line = m.group(0).strip()
        hits.append(line)
    print(f'    grep .detach()/no_grad()/.data in models/loss/sbdr.py: {len(hits)} matches')
    for h in hits:
        print(f'      {h}')


# ---------------------------------------------------------------------------
# 2c. Gradient exactly zero at the degenerate point (all codes identical)
# ---------------------------------------------------------------------------

def test_2c_zero_gradient_at_degenerate_point():
    """All codes identical (kappa=9, d=64) -> dL/dz == 0 to machine precision, lambda2 in {0, 1.6}."""
    d, kappa, eps = 64, 9, 0.31
    idx = torch.randperm(d)[:kappa]
    z_hard = torch.zeros(d); z_hard[idx] = 1.0  # exact 0/1, mimics `clip` saturation
    # near-saturated version, mimics `sigmoid` (never exactly 0/1 but very close)
    z_soft = torch.full((d,), 1e-7)
    z_soft[idx] = 1.0 - 1e-7

    for label, z_row in [('hard (exact 0/1)', z_hard), ('soft (near 0/1, 1e-7)', z_soft)]:
        for lambda2 in (0.0, 1.6):
            order = 1 if lambda2 == 0.0 else 2
            N = 8  # batch of identical rows
            z1 = z_row.unsqueeze(0).repeat(N, 1).clone().requires_grad_(True)
            z2 = z_row.unsqueeze(0).repeat(N, 1).clone()

            loss = SBDRCriticLoss(eps=eps, critic_order=order, lambda2=lambda2, symmetric=True)
            L = loss(z1, z2)
            L.backward()

            gmax = z1.grad.abs().max().item()
            print(f'    {label:<24} lambda2={lambda2}: L={L.item():.3e}, max|dL/dz|={gmax:.3e}')
            assert abs(L.item()) < 1e-6, (label, lambda2, L.item())
            assert gmax < 1e-6, (label, lambda2, gmax)


# ---------------------------------------------------------------------------
# 2d. Gradient sign for dead / saturated units
# ---------------------------------------------------------------------------

def test_2d_dead_and_saturated_unit_gradient_sign():
    """Unit u==0 for every sample -> dL/dz_iu >= 0 for all i. Unit u==1 for every sample -> report sign."""
    torch.manual_seed(2)
    N, d, kappa, eps = 10, 32, 6, 0.31

    z1 = _rand_sparse(N, d, kappa) * 0.9 + 0.02
    z2 = _rand_sparse(N, d, kappa) * 0.9 + 0.02
    dead_u, sat_u = 0, 1
    z1[:, dead_u] = 0.0
    z2[:, dead_u] = 0.0
    z1[:, sat_u] = 1.0
    z2[:, sat_u] = 1.0

    for order, lambda2 in [(1, 0.0), (2, 1.6)]:
        z1a = z1.clone().requires_grad_(True)
        loss = SBDRCriticLoss(eps=eps, critic_order=order, lambda2=lambda2, symmetric=True)
        L = loss(z1a, z2)
        L.backward()
        g_dead = z1a.grad[:, dead_u]
        g_sat = z1a.grad[:, sat_u]
        print(f'    order={order}: dead-unit grad min/max = {g_dead.min().item():.3e} / {g_dead.max().item():.3e}  '
             f'(all >= 0: {bool((g_dead >= -1e-12).all())})')
        print(f'    order={order}: saturated-unit grad min/max = {g_sat.min().item():.3e} / {g_sat.max().item():.3e} '
             f'(sign: {"positive" if g_sat.mean() > 0 else "negative"})')
        assert (g_dead >= -1e-10).all(), g_dead


# ---------------------------------------------------------------------------
# 2f. Second-order C-matrix path vs naive double loop (re-confirmation)
# ---------------------------------------------------------------------------

def test_2f_second_order_path_value_and_gradient():
    """C-matrix path vs naive O(K^2 d) double loop, value AND gradient, ~1e-6; lambda2=0 == order1 bit-for-bit."""
    torch.manual_seed(3)
    N, d, eps, lambda2 = 7, 24, 0.31, 1.6
    z1 = torch.rand(N, d) * 0.9 + 0.02
    z2 = torch.rand(N, d) * 0.9 + 0.02

    loss = SBDRCriticLoss(eps=eps, critic_order=2, lambda2=lambda2, symmetric=False)
    z1a = z1.clone().requires_grad_(True)
    zall = torch.cat([z1a, z2], 0)
    L_matrix = loss._one_way(z1a, z2, zall)
    g_matrix, = torch.autograd.grad(L_matrix, z1a)

    def naive_one_way(za, zb, zall, eps, lambda2):
        K = zall.size(0)
        zbar = zall.mean(0)
        t = (za * zbar).sum(1)
        s = (za * zb).sum(1)
        quad_t = torch.stack([sum(torch.dot(za[i], zall[j]) ** 2 for j in range(K)) / K
                              for i in range(za.size(0))])
        t = t + lambda2 * quad_t + eps
        s = s + lambda2 * s.pow(2) + eps
        return (t.log() - s.log()).mean()

    z1b = z1.clone().requires_grad_(True)
    zallb = torch.cat([z1b, z2], 0)
    L_naive = naive_one_way(z1b, z2, zallb, eps, lambda2)
    g_naive, = torch.autograd.grad(L_naive, z1b)

    vdiff = (L_matrix - L_naive).abs().item()
    gdiff = (g_matrix - g_naive).abs().max().item()
    print(f'    value: matrix={L_matrix.item():.10f} naive={L_naive.item():.10f} |diff|={vdiff:.3e}')
    print(f'    grad : max|matrix-naive|={gdiff:.3e}')
    assert vdiff < 1e-6 and gdiff < 1e-6

    order1 = SBDRCriticLoss(eps=eps, critic_order=1, symmetric=True)
    order2_l0 = SBDRCriticLoss(eps=eps, critic_order=2, lambda2=0.0, symmetric=True)
    L1 = order1(z1, z2)
    L2 = order2_l0(z1, z2)
    print(f'    order1={L1.item():.12f} order2(lambda2=0)={L2.item():.12f} equal={torch.equal(L1, L2)}')
    assert torch.equal(L1, L2)


# ---------------------------------------------------------------------------
# 4. Closed-form reference values
# ---------------------------------------------------------------------------

def test_4_closed_form_reference_values():
    """eps=0.31, d=64. Identical codes -> 0.0 exactly. Disjoint / random kappa=9 -> ~-3.40 / ~-1.77."""
    eps, d, kappa = 0.31, 64, 9

    # (i) all codes identical -> exactly 0.0
    idx = torch.randperm(d)[:kappa]
    z = torch.zeros(d); z[idx] = 1.0
    N = 8
    z1 = z.unsqueeze(0).repeat(N, 1)
    z2 = z.unsqueeze(0).repeat(N, 1)
    loss = SBDRCriticLoss(eps=eps, critic_order=1, symmetric=True)
    L_identical = loss(z1, z2).item()
    print(f'    (i)   identical codes, kappa={kappa}: L = {L_identical:.10f} (expected exactly 0.0)')
    assert abs(L_identical) < 1e-9

    # (ii) "disjoint" limit: 1 probe sample (support S) + K_filler rows on a disjoint support T,
    # K_filler large so self-contribution kappa/K is negligible -> t_i -> eps
    torch.manual_seed(4)
    S = torch.arange(0, kappa)
    T = torch.arange(kappa, 2 * kappa)  # disjoint from S, both within d=64
    z_S = torch.zeros(d); z_S[S] = 1.0
    z_T = torch.zeros(d); z_T[T] = 1.0
    K_filler = 4000
    z1 = torch.cat([z_S.unsqueeze(0), z_T.unsqueeze(0).repeat(K_filler, 1)], 0)
    z2 = z1.clone()  # positive pair identical to the probe / filler itself
    zall = torch.cat([z1, z2], 0)
    zbar = zall.mean(0)
    t0 = (z_S * zbar).sum().item() + eps
    s0 = (z_S * z_S).sum().item() + eps
    L_probe = torch.log(torch.tensor(t0)) - torch.log(torch.tensor(s0))
    analytic_disjoint = torch.log(torch.tensor(eps)) - torch.log(torch.tensor(float(kappa) + eps))
    print(f'    (ii)  disjoint-limit probe (K_filler={K_filler}): t_0={t0:.6f} (eps={eps}), '
         f's_0={s0:.6f}, L_probe = {L_probe.item():.6f}  |  analytic log(eps)-log(kappa+eps) = '
         f'{analytic_disjoint.item():.6f}')

    # (iii) "random" limit: uniform marginal usage kappa/d -> t = kappa^2/d + eps
    K_rand = 4000
    z1r = _rand_sparse(K_rand, d, kappa)
    z2r = z1r.clone()
    zallr = torch.cat([z1r, z2r], 0)
    zbar_r = zallr.mean(0)
    t_r = (z1r * zbar_r.unsqueeze(0)).sum(1) + eps
    s_r = (z1r * z2r).sum(1) + eps
    L_rand = (t_r.log() - s_r.log()).mean().item()
    analytic_random = torch.log(torch.tensor(kappa ** 2 / d + eps)) - torch.log(torch.tensor(float(kappa) + eps))
    print(f'    (iii) random kappa={kappa} codes (K={K_rand}): mean t={t_r.mean().item():.6f} '
         f'(analytic kappa^2/d+eps={kappa**2/d+eps:.6f}), L = {L_rand:.6f}  |  analytic = '
         f'{analytic_random.item():.6f}')

    # permutation invariance and view-swap symmetry, on a small random batch
    torch.manual_seed(5)
    N, dd, kk = 9, 32, 6
    z1p = _rand_sparse(N, dd, kk)
    z2p = _rand_sparse(N, dd, kk)
    loss2 = SBDRCriticLoss(eps=eps, critic_order=1, symmetric=True)
    L_orig = loss2(z1p, z2p).item()
    perm = torch.randperm(N)
    L_perm = loss2(z1p[perm], z2p[perm]).item()
    L_swap = loss2(z2p, z1p).item()
    print(f'    (iv)  permutation invariance: L_orig={L_orig:.10f} L_permuted={L_perm:.10f} '
         f'|diff|={abs(L_orig - L_perm):.3e}')
    print(f'    (v)   view-swap symmetry:     L(z1,z2)={L_orig:.10f} L(z2,z1)={L_swap:.10f} '
         f'|diff|={abs(L_orig - L_swap):.3e}')
    assert abs(L_orig - L_perm) < 1e-9
    assert abs(L_orig - L_swap) < 1e-9


def main():
    tests = [test_2a_central_difference_order1_and_order2,
             test_2b_mean_carries_gradient_w_term,
             test_2c_zero_gradient_at_degenerate_point,
             test_2d_dead_and_saturated_unit_gradient_sign,
             test_2f_second_order_path_value_and_gradient,
             test_4_closed_form_reference_values]
    for t in tests:
        print(f'\n[{t.__name__}]')
        print(f'  {t.__doc__.strip()}' if t.__doc__ else '')
        t()
    print(f'\nAll {len(tests)} checks completed.')


if __name__ == '__main__':
    main()
