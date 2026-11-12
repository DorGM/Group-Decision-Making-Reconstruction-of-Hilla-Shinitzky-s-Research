"""Partition of a set of responses into sub-sets.

Plan §3.3.1 and §8 OQ6 (TH §3.2.1 p.27: "we sampled sub-sets of 30
responses. If the original set of responses was smaller than 30, the
sampling of sub-groups was performed by dividing the set of responses into
two sub-sets of equal size"), and plan §2.2 (the same rule gives the 77
World-B sub-sets of TH Table 4.1). Step A5 of plan §9.3.

Default rule (OQ6 option a): a disjoint random partition into
k = floor(N / 30) near-equal sub-sets that use every response; if k < 2,
two halves (for odd N the halves differ by one). The seed is an explicit
argument so that it can be stored.
"""
import numpy as np

SCHEMES = (
    "a_disjoint_floor_n_over_30",
    "b_overlapping_samples_of_30",
    "c_ceil_n_over_30",
    "d_floor_n_over_30_exactly_30_leftover_unused",
)


def n_subsets(n, subset_size=30, scheme="a_disjoint_floor_n_over_30",
              halves_if_k_below_2=True):
    """Number of sub-sets k for a set of n responses. Plan §8 OQ6.

    a, d : k = floor(n / subset_size)
    c    : k = ceil(n / subset_size)
    n < subset_size: always k = 2, two halves. This is stated in TH p.27 and
    is not an open question, so the switch does not affect it.
    subset_size <= n < 2*subset_size (k = 1 under a/d): k = 2 if
    halves_if_k_below_2 (OQ6 default, extending the TH rule); otherwise
    k = 1, a single sub-set, for which the SG features are degenerate
    (SG_B_Amax 0, all spreads 0; see features_18_27).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}")
    if scheme == "b_overlapping_samples_of_30":
        # Plan §8 OQ6 (b): "the number of samples is then a further unknown".
        raise NotImplementedError("OQ6 option (b): number of samples is not specified")
    if scheme == "c_ceil_n_over_30":
        k = -(-n // subset_size)
    else:
        k = n // subset_size
    if k < 2 and (n < subset_size or halves_if_k_below_2):
        k = 2
    return k


def partition(n, seed, subset_size=30, scheme="a_disjoint_floor_n_over_30",
              halves_if_k_below_2=True, shuffle=True):
    """Split the indices 0..n-1 into disjoint sub-sets. Plan §3.3.1, OQ6.

    n       : number of responses |R|.
    seed    : seed of the random assignment (numpy default_rng). Stored by
              the caller; the same (n, seed) always gives the same split.
    shuffle : False keeps the given order and cuts it into consecutive
              blocks (deterministic; for tests and for a caller that
              supplies its own order).

    Returns a list of k lists of indices.
      schemes a, c : sizes differ by at most one and every index is used.
      scheme d     : k = floor(n/30) sub-sets of exactly `subset_size`;
                     leftover indices are unused (only when the halves
                     fallback is not triggered).
    """
    n = int(n)
    if n < 2:
        raise ValueError("a partition needs at least two responses")
    k = n_subsets(n, subset_size, scheme, halves_if_k_below_2)
    if k < 1:
        raise ValueError(f"n={n} gives no sub-set under scheme {scheme!r}")
    order = np.arange(n)
    if shuffle:
        order = np.random.default_rng(seed).permutation(n)
    exact = (scheme == "d_floor_n_over_30_exactly_30_leftover_unused"
             and n // subset_size >= k)
    if exact:
        order = order[: k * subset_size]
    return [[int(i) for i in part] for part in np.array_split(order, k)]


def subset_sizes(n, subset_size=30, scheme="a_disjoint_floor_n_over_30",
                 halves_if_k_below_2=True):
    """Sizes of the sub-sets for n responses, largest first (no randomness)."""
    parts = partition(n, None, subset_size, scheme, halves_if_k_below_2,
                      shuffle=False)
    return sorted((len(p) for p in parts), reverse=True)
