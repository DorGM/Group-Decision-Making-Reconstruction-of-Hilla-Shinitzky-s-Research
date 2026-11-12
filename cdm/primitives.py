"""Primitive quantities of plan §3.1 (TH §3.1 p.21-24, Table 3.1).

All functions take a set of responses R (any sequence of Response) and,
where needed, the answer list A_P. R may be the full R_P or a sub-set R'.
"""
import math

import numpy as np

NAN = float("nan")

#: Absolute tolerance for ties between FLOAT scores (WC, HAC, SP and, through
#: f_HAC, feature 12) and for the AvgC / AvgPSV comparisons of features 16/17.
#: Amendments AM3 / AM9. It mirrors `tie_abs_tol`
#: in config/defaults.yaml (set that to 0 for exact comparison). It is a
#: numeric safeguard, not a modelling choice. It is NOT used for integer or
#: label comparisons (MR, a_max, a_min, DA counters, OQ8b, SG_B_Amax).
DEFAULT_TIE_ABS_TOL = 1e-9

#: Tie policies of plan §8 OQ8 / Step D5 item 1.
TIE_POLICIES = ("canonical_first", "canonical_second", "random",
                "sp_tie_half_credit")


# ---- sets of responses -------------------------------------------------

def supporters(a, R):
    """R(a) = {r : r.v = a}. Plan §3.1."""
    return [r for r in R if r.v == a]


def non_supporters(a, R):
    """R-bar(a) = R minus R(a). Plan §3.1."""
    return [r for r in R if r.v != a]


def in_group(r, R):
    """In-group of r = R(r.v); it includes r itself. Plan §3.1 (MCF §3)."""
    return supporters(r.v, R)


def out_group(r, R):
    """Out-group of r = R-bar(r.v). Plan §3.1 (MCF §3)."""
    return non_supporters(r.v, R)


# ---- support -----------------------------------------------------------

def vote_counts(R, answers):
    """Number of votes per answer, over all of A_P, in the order of A_P."""
    return [sum(1 for r in R if r.v == a) for a in answers]


def support(a, R):
    """S(a, R) = |R(a)| / |R|. Plan §3.1. NaN if R is empty."""
    if len(R) == 0:
        return NAN
    return len(supporters(a, R)) / len(R)


def support_vector(R, answers):
    """S(R): support of every a in A_P, zero-vote answers included.

    Plan §3.1 and §8 OQ7 (the vector is over all of A_P).
    """
    return [support(a, R) for a in answers]


# ---- tie policy --------------------------------------------------------

def check_tol(tol):
    """Validate an absolute tie tolerance (a finite number >= 0)."""
    if isinstance(tol, (bool, str)) or tol is None:
        raise ValueError(f"tie tolerance must be a number >= 0, got {tol!r}")
    tol = float(tol)
    if not (math.isfinite(tol) and tol >= 0.0):
        raise ValueError(f"tie tolerance must be a finite number >= 0, got {tol!r}")
    return tol


def tied_best(scores, answers, largest=True, tol=0.0):
    """All answers that share the best (largest or smallest) score.

    Returned in the canonical order of A_P. NaN scores are never best.

    tol : absolute tolerance; a score s is tied with the best one when
          |s - best| <= tol. The default 0 is exact equality, which is
          right for integer vote counts (a_max, a_min, MR). Float scores
          (WC, HAC, SP) are passed with tol = DEFAULT_TIE_ABS_TOL by
          cdm.aggregators (amendment AM3).
    """
    tol = check_tol(tol)
    valid = [s for s in scores if not _isnan(s)]
    if not valid:
        return []
    best = max(valid) if largest else min(valid)
    return [a for a, s in zip(answers, scores)
            if not _isnan(s) and abs(s - best) <= tol]


def break_tie(tied, policy="canonical_first"):
    """Pick one answer from a tied list. Plan §8 OQ7/OQ8, Step D5 item 1.

    canonical_first  : the first tied answer in the canonical order of A_P
                       (the default of OQ7 and OQ8; deterministic).
    canonical_second : the second tied answer in the canonical order (Step
                       D5 ladder, item 1). Defined for a tie between exactly
                       two answers, which is every tie of a binary case.
                       With more than two tied answers the plan does not say
                       which one is meant, so this raises.
    random, sp_tie_half_credit : not answer-level rules that the plan
                       defines; they raise NotImplementedError (see
                       cdm.aggregators.choose / success_credit).

    A list with one answer is not a tie; that answer is returned under
    every implemented policy.
    """
    if policy not in TIE_POLICIES:
        raise ValueError(f"unknown tie policy {policy!r}; expected one of {TIE_POLICIES}")
    if policy == "random":
        raise NotImplementedError(
            "tie policy 'random' is not implemented: plan §8 OQ8 names it as "
            "an alternative but defines no generator or seed convention, and "
            "it is not part of the Step D5 ladder")
    if policy == "sp_tie_half_credit":
        raise NotImplementedError(
            "tie policy 'sp_tie_half_credit' scores a tie as 0.5 correct; it "
            "does not pick an answer. Use cdm.aggregators.success_credit")
    if len(tied) == 0:
        raise ValueError("no answer to choose from")
    if policy == "canonical_first" or len(tied) == 1:
        return tied[0]
    if len(tied) > 2:
        raise NotImplementedError(
            "tie policy 'canonical_second' is defined only for a tie between "
            "two answers (plan Step D5 item 1 gives no rule for a larger tie)")
    return tied[1]


def a_max_ties(R, answers):
    """All answers with the largest support (for tie logging, OQ7).

    Integer vote counts are compared, exactly (no tolerance). An empty R
    has no a_max: the list is empty.
    """
    if len(R) == 0:
        return []
    return tied_best(vote_counts(R, answers), answers, largest=True)


def a_min_ties(R, answers):
    """All answers with the smallest support, zero-vote answers included.

    Integer vote counts are compared, exactly. Empty R -> empty list.
    """
    if len(R) == 0:
        return []
    return tied_best(vote_counts(R, answers), answers, largest=False)


def a_max(R, answers, policy="canonical_first"):
    """a_max = argmax_a S(a, R). Plan §3.1; ties by OQ7.

    Raises ValueError for an empty R (a_max is undefined there).
    """
    if len(R) == 0:
        raise ValueError("a_max is undefined for an empty set of responses")
    return break_tie(a_max_ties(R, answers), policy)


def a_min(R, answers, policy="canonical_first"):
    """a_min = argmin_a S(a, R) over all of A_P. Plan §3.1; ties by OQ7.

    Raises ValueError for an empty R (a_min is undefined there).
    """
    if len(R) == 0:
        raise ValueError("a_min is undefined for an empty set of responses")
    return break_tie(a_min_ties(R, answers), policy)


# ---- confidence and predicted support ------------------------------------

def confidences(R):
    """C(R) = {r.c}. Plan §3.1."""
    return [r.c for r in R]


def predicted_support(a, R, answers):
    """PS(a, R) = {r.ps[a]}. Plan §3.1."""
    i = list(answers).index(a)
    return [r.ps[i] for r in R]


def predicted_support_of_vote(R, answers):
    """PSV(R) = {r.ps[r.v]}. Plan §3.1."""
    index = {a: i for i, a in enumerate(answers)}
    return [r.ps[index[r.v]] for r in R]


# ---- functions (TH p.24) -----------------------------------------------

def average(xs):
    """Average. NaN for an empty set (plan §8 OQ7: undefined -> NaN)."""
    if len(xs) == 0:
        return NAN
    return float(np.mean(xs))


def var(xs, kind="population_sd"):
    """The thesis's "Var".

    Plan §3.1 and §8 OQ9: "Var" is the population standard deviation
    (inferred from TH Table A.2). kind="population_variance" is the switch.
    NaN for an empty set.
    """
    if len(xs) == 0:
        return NAN
    if kind == "population_sd":
        return float(np.std(xs, ddof=0))
    if kind == "population_variance":
        return float(np.var(xs, ddof=0))
    raise ValueError(f"unknown kind {kind!r}")


def maximum(xs):
    """Max. NaN for an empty set, and NaN if any value is NaN."""
    if len(xs) == 0 or any(_isnan(x) for x in xs):
        return NAN
    return float(max(xs))


def minimum(xs):
    """Min. NaN for an empty set, and NaN if any value is NaN."""
    if len(xs) == 0 or any(_isnan(x) for x in xs):
        return NAN
    return float(min(xs))


def entropy(ps, base=2):
    """Entropy of a share vector, log base 2 (plan §3.1, §8 OQ10).

    Zero shares contribute 0, so zero-vote answers can stay in S(R) (OQ7).
    NaN for an empty vector or if any share is NaN (e.g. the support vector
    of an empty set). Raises ValueError for a share outside [0, 1] (counts
    passed instead of shares) or a base <= 1.
    """
    if isinstance(base, bool) or not base > 1:
        raise ValueError(f"entropy base must be > 1, got {base!r}")
    ps = list(ps)
    if len(ps) == 0 or any(_isnan(p) for p in ps):
        return NAN
    total = 0.0
    for p in ps:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"share {p!r} is not in [0, 1]")
        if p > 0:
            total -= p * math.log(p, base)
    return total


def _isnan(x):
    """True for any real NaN (Python float, numpy float32/float64, ...)."""
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False
