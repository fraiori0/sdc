"""
Diagnostic metrics for SBDR codes (HANDOUT.md §2.6, Task C in the 2026-09-03
second-order-critic sweep). Pure tensor-in / dict-out functions, no I/O, so they
can be unit tested and reused from any driver script.

All functions expect `codes` as [0,1]-domain floats (continuous) and binarize at
0.5 internally where needed, matching utils.hashing.preprocess_on_codes.
"""

import torch


def usage_stats(codes_cont, threshold=0.5):
    """
    Per-bit usage (active fraction), kappa (active bits per sample), binarity, and
    dead/saturated bit counts.

    codes_cont: (N, d) float tensor in [0,1].
    """
    active = (codes_cont > threshold).float()  # (N, d)
    usage = active.mean(0)  # (d,) fraction of samples with bit u active
    kappa_per_sample = active.sum(1)  # (N,)

    binarity = ((codes_cont - 0.5).abs() >= 0.49).float().mean().item()  # within 1e-2 of {0,1}

    return {
        'kappa_mean': kappa_per_sample.mean().item(),
        'kappa_std': kappa_per_sample.std().item(),
        'binarity': binarity,
        'usage_mean': usage.mean().item(),
        'usage_std': usage.std().item(),
        'dead_bits': int((usage == 0).sum().item()),
        'saturated_bits': int((usage == 1).sum().item()),
        'near_dead_bits': int((usage < 1e-3).sum().item()),
        'near_saturated_bits': int((usage > 1 - 1e-3).sum().item()),
        'nbit': codes_cont.size(1),
    }


def _sample_pairs_overlap(codes_bin, n_sample=3000, generator=None):
    """
    Pairwise overlap (sum-AND) over a random subset of `n_sample` rows of
    `codes_bin` ({0,1}, (N, d)). Returns the flattened off-diagonal upper-triangle
    of the (n_sample, n_sample) overlap matrix -- i.e. each unordered pair once.
    """
    N = codes_bin.size(0)
    n_sample = min(n_sample, N)
    idx = torch.randperm(N, generator=generator)[:n_sample]
    sub = codes_bin[idx].float()
    ov = sub @ sub.t()  # (n_sample, n_sample), integer-valued
    iu = torch.triu_indices(n_sample, n_sample, offset=1)
    return ov[iu[0], iu[1]]


def overlap_distribution(codes_bin, n_sample=3000, mass_threshold=0.01, generator=None):
    """
    Overlap-value distribution over a random sample of database pairs (HANDOUT
    §5.1 / Experiment 8): mean, std, number of distinct overlap values carrying
    > mass_threshold of the pair mass, and the overlap=0 fraction.
    """
    pairs = _sample_pairs_overlap(codes_bin, n_sample=n_sample, generator=generator)
    vals, counts = pairs.unique(return_counts=True)
    freq = counts.float() / counts.sum()

    return {
        'overlap_mean': pairs.mean().item(),
        'overlap_std': pairs.std().item(),
        'n_distinct_values': int(vals.numel()),
        'n_distinct_values_over_1pct_mass': int((freq > mass_threshold).sum().item()),
        'overlap_zero_frac': (pairs == 0).float().mean().item(),
        'n_pairs_sampled': int(pairs.numel()),
    }


def _overlap_counts_query_vs_db(query_bin, db_bin, max_val, chunk=200):
    """
    For each query row, the integer-overlap histogram against every db row:
    returns (Q, max_val+1) tensor where entry [q, v] = #db items with
    overlap(query_q, db) == v. Computed in query chunks to bound memory.
    """
    Q = query_bin.size(0)
    device = query_bin.device
    out = torch.zeros(Q, max_val + 1, dtype=torch.long)

    db = db_bin.float()
    for start in range(0, Q, chunk):
        qb = query_bin[start:start + chunk].float()
        ov = (qb @ db.t()).round().long()  # (chunk, N_db), integer-valued
        ov.clamp_(0, max_val)
        for row in range(ov.size(0)):
            out[start + row] = torch.bincount(ov[row], minlength=max_val + 1)
    return out


def tie_block_sizes(query_bin, db_bin, R, chunk=200):
    """
    HANDOUT §5.1 / Experiment 8: for each query, the size of the overlap-value tie
    block that rank R falls inside, under a `largest overlap first` ranking
    (ties broken by `torch.topk`'s implementation-defined order elsewhere in the
    pipeline; here we report the block size itself, not which side of it R lands).

    Returns a (Q,) tensor of per-query tie-block sizes.
    """
    d = db_bin.size(1)
    counts = _overlap_counts_query_vs_db(query_bin, db_bin, max_val=d, chunk=chunk)  # (Q, d+1), index = overlap value
    counts_desc = counts.flip(1)  # index 0 = overlap value d, index d = overlap value 0
    csum = counts_desc.cumsum(1)  # (Q, d+1)
    # first index where cumulative count reaches R -- the overlap level containing rank R (1-indexed)
    reached = csum >= R
    # if R exceeds the db size (shouldn't happen), fall back to the last bucket
    first_idx = reached.float().argmax(1)
    tie_size = counts_desc.gather(1, first_idx.unsqueeze(1)).squeeze(1)
    return tie_size


def false_negative_rate(query_bin, db_bin, query_labels, db_labels, k=50, chunk=200):
    """
    HANDOUT Task C: among the top-k retrieved db items per query (by overlap,
    largest first), the fraction sharing the query's class label, averaged over
    queries. Labels may be class indices (N,) or one-hot (N, C).
    """
    if query_labels.dim() == 2:
        query_labels = query_labels.argmax(1)
    if db_labels.dim() == 2:
        db_labels = db_labels.argmax(1)

    Q = query_bin.size(0)
    db = db_bin.float()
    fracs = []
    for start in range(0, Q, chunk):
        qb = query_bin[start:start + chunk].float()
        ov = qb @ db.t()  # (chunk, N_db)
        topk_idx = ov.topk(k, dim=1, largest=True, sorted=False).indices  # (chunk, k)
        retrieved_labels = db_labels[topk_idx]  # (chunk, k)
        match = (retrieved_labels == query_labels[start:start + chunk].unsqueeze(1)).float()
        fracs.append(match.mean(1))
    fracs = torch.cat(fracs)
    return {
        'false_negative_rate_mean': fracs.mean().item(),
        'false_negative_rate_std': fracs.std().item(),
        'k': k,
    }


def positive_negative_separation(pos_pair_overlap, random_pair_overlap):
    """
    HANDOUT Task C: mean overlap for augmented-view (positive) pairs vs. mean
    overlap for random pairs, and the ratio.
    """
    pos_mean = pos_pair_overlap.mean().item()
    rand_mean = random_pair_overlap.mean().item()
    return {
        'positive_pair_overlap_mean': pos_mean,
        'positive_pair_overlap_std': pos_pair_overlap.std().item(),
        'random_pair_overlap_mean': rand_mean,
        'random_pair_overlap_std': random_pair_overlap.std().item(),
        'separation_ratio': pos_mean / rand_mean if rand_mean != 0 else float('inf'),
        'n_positive_pairs': int(pos_pair_overlap.numel()),
    }
