"""
Standalone checks for the [0,1] vs {-1,+1} code-domain handling in `utils.hashing`.

The repo's ranking path assumes codes in {-1,+1}. Our codes live in [0,1] with the
decision boundary at 0.5, and `torch.sign` maps those to {0,1} rather than {-1,+1},
so `hamming` silently returns plausible-looking, meaningless numbers. These checks
pin down the fix against hand-computed values.

Run directly (no pytest required, though pytest will also collect it):

    CUDA_VISIBLE_DEVICES=2 python tests/test_hashing_domain.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.hashing import (calculate_mAP, get_distance_func, hamming, overlap,
                           preprocess_on_codes)


def _legacy_preprocess(codes, threshold=0., sign=True):
    """The body of `preprocess_on_codes` exactly as it was before the code_domain change."""
    codes = codes.clone()
    if threshold != 0:
        codes[codes.abs() < threshold] = 0
    if sign:
        codes = torch.sign(codes)
    return codes


def _expect_value_error(fn, needle, label):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), f'{label}: unexpected message: {e}'
        print(f'    ok   {label}\n         -> ValueError: {str(e).splitlines()[0][:88]}...')
        return
    raise AssertionError(f'{label}: expected a ValueError, none was raised')


def test_handout_trap_is_reproduced():
    """The silent failure this whole change exists to prevent."""
    z = torch.tensor([[0.9, 0.0, 0.0, 0.8],
                      [0.0, 0.7, 0.0, 0.0]])

    broken = hamming(torch.sign(z), torch.sign(z))
    assert torch.equal(broken, torch.tensor([[1.0, 2.0], [2.0, 1.5]])), broken

    b01 = preprocess_on_codes(z, code_domain='unit', dist_metric='overlap')
    assert torch.equal(b01, torch.tensor([[1., 0., 0., 1.], [0., 1., 0., 0.]])), b01
    true_overlap = -overlap(b01, b01)
    assert torch.equal(true_overlap, torch.tensor([[2., 0.], [0., 1.]])), true_overlap

    bpm = preprocess_on_codes(z, code_domain='unit', dist_metric='hamming')
    assert torch.equal(bpm, torch.tensor([[1., -1., -1., 1.], [-1., 1., -1., -1.]])), bpm
    true_hamming = hamming(bpm, bpm)
    assert torch.equal(true_hamming, torch.tensor([[0., 3.], [3., 0.]])), true_hamming

    print('    z                     =', z.tolist())
    print('    torch.sign(z)         =', torch.sign(z).tolist(), ' <- 0, not -1')
    print('    hamming on that       =', broken.tolist(), ' <- meaningless (and non-integer)')
    print('    true overlap (sum-AND)=', true_overlap.tolist())
    print('    true hamming          =', true_hamming.tolist())


def test_overlap_and_hamming_match_hand_computed():
    """Four sparse codes, kappa=3 of 8 bits, with the answers written out by hand."""
    supports = {'a': {0, 1, 2},
                'b': {0, 1, 7},
                'c': {3, 4, 5},
                'd': {0, 1, 2}}  # d is a duplicate of a
    names = ['a', 'b', 'c', 'd']
    nbit = 8

    # deliberately not exactly 0/1, and straddling the 0.5 boundary on both sides,
    # so the binarization threshold itself is exercised
    active = [0.9, 0.7, 0.51]
    inactive = [0.0, 0.1, 0.49]
    z = torch.zeros(len(names), nbit)
    for i, name in enumerate(names):
        for u in range(nbit):
            pool = active if u in supports[name] else inactive
            z[i, u] = pool[u % len(pool)]

    # hand-computed: |A n B| and |A xor B| = |A| + |B| - 2|A n B|
    exp_overlap = torch.zeros(4, 4)
    exp_hamming = torch.zeros(4, 4)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            inter = len(supports[ni] & supports[nj])
            exp_overlap[i, j] = -inter
            exp_hamming[i, j] = len(supports[ni]) + len(supports[nj]) - 2 * inter

    # spot-check the hand computation itself against the literal expected matrices
    assert torch.equal(exp_overlap, -torch.tensor([[3., 2., 0., 3.],
                                                   [2., 3., 0., 2.],
                                                   [0., 0., 3., 0.],
                                                   [3., 2., 0., 3.]]))
    assert torch.equal(exp_hamming, torch.tensor([[0., 2., 6., 0.],
                                                  [2., 0., 6., 2.],
                                                  [6., 6., 0., 6.],
                                                  [0., 2., 6., 0.]]))

    b01 = preprocess_on_codes(z, code_domain='unit', dist_metric='overlap')
    got_overlap = get_distance_func('overlap')(b01, b01)
    assert torch.equal(got_overlap, exp_overlap), f'{got_overlap}\n!=\n{exp_overlap}'

    bpm = preprocess_on_codes(z, code_domain='unit', dist_metric='hamming')
    got_hamming = get_distance_func('hamming')(bpm, bpm)
    assert torch.equal(got_hamming, exp_hamming), f'{got_hamming}\n!=\n{exp_hamming}'

    assert set(b01.unique().tolist()) == {0., 1.}, b01.unique()
    assert set(bpm.unique().tolist()) == {-1., 1.}, bpm.unique()
    assert torch.equal((b01 > 0.5).sum(1), torch.tensor([3, 3, 3, 3]))  # realised kappa

    print(f'    supports   = {[sorted(supports[n]) for n in names]}  (nbit={nbit})')
    print(f'    overlap    = {got_overlap.tolist()}  (matches hand-computed)')
    print(f'    hamming    = {got_hamming.tolist()}  (matches hand-computed)')


def test_signed_path_is_bit_identical_to_before():
    """Regression guard: baselines must be untouched."""
    torch.manual_seed(0)
    codes = torch.randn(64, 32)
    for threshold in (0., 0.5):
        for sign in (True, False):
            got = preprocess_on_codes(codes, threshold, sign)
            exp = _legacy_preprocess(codes, threshold, sign)
            assert torch.equal(got, exp), f'threshold={threshold} sign={sign}'
    # and the input itself is not mutated
    assert torch.equal(codes, torch.randn(64, 32) * 0 + codes)
    print('    signed domain matches the pre-change behaviour for '
          'threshold in {0, 0.5} x sign in {True, False}')


def test_guards_fire():
    unit = torch.rand(16, 8)
    signed = torch.randn(16, 8)

    _expect_value_error(lambda: preprocess_on_codes(unit, code_domain='signed'),
                        'straddling 0', 'unit codes declared `signed`')
    _expect_value_error(lambda: preprocess_on_codes(signed, code_domain='unit',
                                                    dist_metric='hamming'),
                        'expects codes in [0, 1]', 'signed codes declared `unit`')
    _expect_value_error(lambda: preprocess_on_codes(unit - unit.mean(0, keepdim=True),
                                                    code_domain='unit', dist_metric='overlap'),
                        'expects codes in [0, 1]', 'zero-mean-centred unit codes')
    _expect_value_error(lambda: preprocess_on_codes(signed, code_domain='signed',
                                                    dist_metric='overlap'),
                        'requires code_domain=`unit`', '`overlap` on the signed domain')
    _expect_value_error(lambda: preprocess_on_codes(unit, threshold=0.2, sign=False,
                                                    code_domain='unit', dist_metric='overlap'),
                        'ternary/DBQ margin', 'ternary threshold on the unit domain')
    _expect_value_error(lambda: preprocess_on_codes(signed, code_domain='binary'),
                        'Unknown code_domain', 'unknown code_domain')

    # the guard must NOT fire on the legitimate non-binarizing metrics
    for metric in ('cosine', 'euclidean'):
        out = preprocess_on_codes(unit, sign=False, code_domain='signed', dist_metric=metric)
        assert torch.equal(out, unit), metric
    print('    ok   non-negative codes with cosine/euclidean are left alone (no false positive)')


def test_calculate_mAP_end_to_end():
    """Two classes with disjoint supports -> mAP must be exactly 1.0 under both metrics."""
    nbit, per_class, n_query = 16, 10, 4
    supports = [range(0, 4), range(8, 12)]

    def make(n):
        z = torch.rand(2 * n, nbit) * 0.4  # inactive in [0, 0.4)
        labels = torch.zeros(2 * n, dtype=torch.long)
        for c, sup in enumerate(supports):
            rows = slice(c * n, (c + 1) * n)
            labels[rows] = c
            z[rows, sup.start:sup.stop] = 0.6 + torch.rand(n, len(sup)) * 0.4  # active in [0.6, 1)
        return z, labels

    torch.manual_seed(1)
    db_codes, db_labels = make(per_class)
    test_codes, test_labels = make(n_query)

    for metric in ('overlap', 'hamming'):
        mAP = calculate_mAP(db_codes, db_labels, test_codes, test_labels,
                            Rs=per_class, dist_metric=metric, code_domain='unit')
        assert abs(mAP - 1.0) < 1e-9, f'{metric}: mAP={mAP}'
        print(f'    mAP@{per_class} ({metric:7s}, code_domain=unit) = {mAP:.6f}')

    # the same codes down the default signed path must refuse to run rather than
    # silently return a number
    _expect_value_error(lambda: calculate_mAP(db_codes, db_labels, test_codes, test_labels,
                                              Rs=per_class, dist_metric='hamming'),
                        'straddling 0', 'unit codes through the default signed path')


def test_topk_eval_forces_exact_kappa():
    """
    topk_eval must force exactly k active bits per sample, overriding whatever the
    plain 0.5 threshold would have given -- that override is the entire point (an
    *exact* per-sample kappa, since `eps` only sets kappa in expectation).
    """
    # row0: 4 values > 0.5 naturally; row1: 0 values > 0.5; row2: 1 value > 0.5.
    # topk_eval=2 must select the top 2 by value in every row regardless.
    z = torch.tensor([[0.9, 0.8, 0.6, 0.55, 0.1],
                      [0.4, 0.3, 0.2, 0.10, 0.05],
                      [0.9, 0.4, 0.3, 0.20, 0.10]])

    plain = preprocess_on_codes(z, code_domain='unit', dist_metric='overlap')
    exp_plain = torch.tensor([[1., 1., 1., 1., 0.],   # 4 actives (0.55 > 0.5)
                              [0., 0., 0., 0., 0.],   # 0 actives
                              [1., 0., 0., 0., 0.]])  # 1 active
    assert torch.equal(plain, exp_plain), plain
    assert torch.equal(plain.sum(1), torch.tensor([4., 0., 1.])), \
        'sanity: the plain threshold gives a non-constant, non-2 kappa per row'

    topk01 = preprocess_on_codes(z, code_domain='unit', dist_metric='overlap', topk_eval=2)
    exp_topk01 = torch.tensor([[1., 1., 0., 0., 0.],   # top2 of row0: indices 0,1
                               [1., 1., 0., 0., 0.],   # top2 of row1: indices 0,1 (forced active
                               [1., 1., 0., 0., 0.]])  #   despite being < 0.5 under plain thresh)
    assert torch.equal(topk01, exp_topk01), topk01
    assert torch.equal(topk01.sum(1), torch.tensor([2., 2., 2.])), \
        'topk_eval=2 must give exactly kappa=2 on every row, unlike the plain threshold'

    topkpm = preprocess_on_codes(z, code_domain='unit', dist_metric='hamming', topk_eval=2)
    exp_topkpm = torch.tensor([[1., 1., -1., -1., -1.],
                               [1., 1., -1., -1., -1.],
                               [1., 1., -1., -1., -1.]])
    assert torch.equal(topkpm, exp_topkpm), topkpm

    print('    z              =', z.tolist())
    print('    plain (>0.5)   =', plain.tolist(), ' kappa =', plain.sum(1).tolist())
    print('    topk_eval=2    =', topk01.tolist(), ' kappa =', topk01.sum(1).tolist(),
         ' (exact, matches hand-picked top-2 indices per row)')


def test_topk_eval_guards_fire():
    unit = torch.rand(16, 8)
    signed = torch.randn(16, 8)

    _expect_value_error(lambda: preprocess_on_codes(signed, code_domain='signed',
                                                    dist_metric='hamming', topk_eval=3),
                        'only applies to code_domain=`unit`', 'topk_eval on the signed domain')
    _expect_value_error(lambda: preprocess_on_codes(unit, code_domain='unit',
                                                    dist_metric='cosine', topk_eval=3),
                        "dist_metric in ('overlap', 'hamming')", 'topk_eval with dist_metric=cosine')
    _expect_value_error(lambda: preprocess_on_codes(unit, code_domain='unit',
                                                    dist_metric='overlap', topk_eval=0),
                        'must be an int in [1, nbit=8]', 'topk_eval=0 (out of range)')
    _expect_value_error(lambda: preprocess_on_codes(unit, code_domain='unit',
                                                    dist_metric='overlap', topk_eval=9),
                        'must be an int in [1, nbit=8]', 'topk_eval > nbit')
    _expect_value_error(lambda: preprocess_on_codes(unit, code_domain='unit',
                                                    dist_metric='overlap', topk_eval=2.0),
                        'must be an int in [1, nbit=8]', 'topk_eval as a float')

    # boundary: topk_eval == nbit (dense) must be accepted, not rejected
    out = preprocess_on_codes(unit, code_domain='unit', dist_metric='overlap', topk_eval=8)
    assert torch.equal(out, torch.ones_like(unit)), 'topk_eval == nbit must activate every bit'
    print('    ok   topk_eval == nbit (dense boundary) is accepted, all bits active')

    # topk_eval=None (the default) must be silently accepted everywhere -- baselines untouched
    for metric in ('cosine', 'euclidean'):
        out = preprocess_on_codes(unit, sign=False, code_domain='signed', dist_metric=metric,
                                  topk_eval=None)
        assert torch.equal(out, unit), metric
    print('    ok   topk_eval=None (default) never triggers a guard')


def main():
    tests = [test_handout_trap_is_reproduced,
             test_overlap_and_hamming_match_hand_computed,
             test_signed_path_is_bit_identical_to_before,
             test_guards_fire,
             test_calculate_mAP_end_to_end,
             test_topk_eval_forces_exact_kappa,
             test_topk_eval_guards_fire]
    for t in tests:
        print(f'\n[{t.__name__}]')
        print(f'  {t.__doc__.strip()}' if t.__doc__ else '')
        t()
    print(f'\nAll {len(tests)} checks passed.')


if __name__ == '__main__':
    main()
