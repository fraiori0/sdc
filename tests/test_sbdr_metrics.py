"""
Hand-computed checks for `utils.sbdr_metrics` (Task C reporting helpers).

Run directly (no pytest required, though pytest will also collect it):

    python tests/test_sbdr_metrics.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sbdr_metrics import (false_negative_rate, overlap_distribution,
                                positive_negative_separation, tie_block_sizes,
                                usage_stats)


def test_usage_stats_hand_computed():
    """4 samples x 4 bits, supports chosen so kappa/usage/dead/saturated are all known by hand."""
    # bit0: active in all 4 (saturated); bit1: active in none (dead);
    # bit2: active in samples 0,1; bit3: active in sample 0 only
    z = torch.tensor([[1.0, 0.0, 1.0, 1.0],
                      [0.9, 0.1, 0.8, 0.0],
                      [0.99, 0.0, 0.0, 0.0],
                      [0.6, 0.2, 0.0, 0.0]])
    stats = usage_stats(z)

    assert stats['nbit'] == 4
    assert stats['dead_bits'] == 1, stats  # bit1
    assert stats['saturated_bits'] == 1, stats  # bit0
    exp_kappa = torch.tensor([3., 2., 1., 1.])  # active-bit counts per row (>0.5)
    assert abs(stats['kappa_mean'] - exp_kappa.mean().item()) < 1e-6
    assert abs(stats['kappa_std'] - exp_kappa.std().item()) < 1e-6
    exp_usage = torch.tensor([1.0, 0.0, 0.5, 0.25])  # per-bit active fraction
    assert abs(stats['usage_mean'] - exp_usage.mean().item()) < 1e-6
    assert abs(stats['usage_std'] - exp_usage.std().item()) < 1e-6
    print(f'    kappa_mean={stats["kappa_mean"]:.4f} (exp {exp_kappa.mean().item():.4f})')
    print(f'    usage_mean={stats["usage_mean"]:.4f} (exp {exp_usage.mean().item():.4f})')
    print(f'    dead_bits={stats["dead_bits"]}, saturated_bits={stats["saturated_bits"]}')


def test_overlap_distribution_hand_computed():
    """3 db items, kappa=2 of 4 bits, all 3 pairwise overlaps known by hand: 2, 0, 1."""
    z = torch.tensor([[1., 1., 0., 0.],   # support {0,1}
                      [1., 1., 0., 0.],   # support {0,1} -- duplicate of row0, overlap=2
                      [0., 0., 1., 1.]])  # support {2,3}, overlap with row0/1 = 0
    # pairwise: (0,1)->2, (0,2)->0, (1,2)->0
    out = overlap_distribution(z, n_sample=3)
    assert out['n_pairs_sampled'] == 3
    exp_mean = (2 + 0 + 0) / 3
    assert abs(out['overlap_mean'] - exp_mean) < 1e-6, out
    assert abs(out['overlap_zero_frac'] - 2 / 3) < 1e-6, out
    assert out['n_distinct_values'] == 2, out  # {0, 2}
    print(f'    overlap_mean={out["overlap_mean"]:.4f} (exp {exp_mean:.4f})')
    print(f'    overlap_zero_frac={out["overlap_zero_frac"]:.4f} (exp {2/3:.4f})')
    print(f'    n_distinct_values={out["n_distinct_values"]} (exp 2)')


def test_tie_block_sizes_hand_computed():
    """
    1 query, 6 db items with overlaps [5, 5, 3, 3, 3, 0] against the query
    (by construction, d=5 bits). Rank R=1 or 2 -> block of the two overlap=5 items
    (size 2). R=3,4,5 -> block of the three overlap=3 items (size 3). R=6 -> the
    lone overlap=0 item (size 1).
    """
    d = 5
    query = torch.tensor([[1., 1., 1., 1., 1.]])  # full support
    # rows engineered to have overlap 5,5,3,3,3,0 with an all-ones query
    db = torch.tensor([
        [1., 1., 1., 1., 1.],  # overlap 5
        [1., 1., 1., 1., 1.],  # overlap 5
        [1., 1., 1., 0., 0.],  # overlap 3
        [1., 1., 0., 1., 0.],  # overlap 3
        [1., 0., 1., 1., 0.],  # overlap 3
        [0., 0., 0., 0., 0.],  # overlap 0
    ])
    for R, exp_size in [(1, 2), (2, 2), (3, 3), (4, 3), (5, 3), (6, 1)]:
        size = tie_block_sizes(query, db, R=R).item()
        assert size == exp_size, f'R={R}: got {size}, expected {exp_size}'
    print('    tie-block sizes for R in {1..6} against overlaps [5,5,3,3,3,0]: '
         '[2,2,3,3,3,1] (all match hand-computed)')


def test_false_negative_rate_hand_computed():
    """
    2 queries (class 0, class 1), 4 db items (labels 0,0,1,1). Query0's top-2 by
    overlap are both class-1 db items (deliberately) -> FN rate for query0 = 0/2
    matches, i.e. 0.0 (no same-class in top-2). Query1's top-2 are both class-1 ->
    FN rate 2/2 = 1.0 (both share query1's label).
    """
    d = 4
    query = torch.tensor([[0., 0., 1., 1.],   # class 0, but engineered to overlap most with class-1 db rows
                          [0., 0., 1., 1.]])  # class 1
    db = torch.tensor([[1., 0., 0., 0.],      # class 0, overlap 0 with either query
                       [0., 1., 0., 0.],      # class 0, overlap 0
                       [0., 0., 1., 1.],      # class 1, overlap 2 with either query
                       [0., 0., 1., 0.]])     # class 1, overlap 1 with either query
    query_labels = torch.tensor([0, 1])
    db_labels = torch.tensor([0, 0, 1, 1])

    out = false_negative_rate(query, db, query_labels, db_labels, k=2)
    # query0 (label 0): top-2 by overlap = db idx {2,3} both label 1 -> 0/2 match query's own label
    # query1 (label 1): top-2 by overlap = db idx {2,3} both label 1 -> 2/2 match
    # mean over queries = (0 + 1) / 2 = 0.5
    assert abs(out['false_negative_rate_mean'] - 0.5) < 1e-6, out
    print(f'    false_negative_rate_mean={out["false_negative_rate_mean"]:.4f} (exp 0.5)')


def test_positive_negative_separation_hand_computed():
    pos = torch.tensor([6.0, 8.0])
    rand = torch.tensor([1.0, 2.0, 3.0])
    out = positive_negative_separation(pos, rand)
    assert abs(out['positive_pair_overlap_mean'] - 7.0) < 1e-6
    assert abs(out['random_pair_overlap_mean'] - 2.0) < 1e-6
    assert abs(out['separation_ratio'] - 3.5) < 1e-6
    print(f'    separation_ratio={out["separation_ratio"]:.4f} (exp 3.5)')


def main():
    tests = [test_usage_stats_hand_computed,
             test_overlap_distribution_hand_computed,
             test_tie_block_sizes_hand_computed,
             test_false_negative_rate_hand_computed,
             test_positive_negative_separation_hand_computed]
    for t in tests:
        print(f'\n[{t.__name__}]')
        print(f'  {t.__doc__.strip()}' if t.__doc__ else '')
        t()
    print(f'\nAll {len(tests)} checks passed.')


if __name__ == '__main__':
    main()
