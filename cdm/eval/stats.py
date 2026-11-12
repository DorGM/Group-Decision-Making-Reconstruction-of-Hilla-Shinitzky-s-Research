"""Metrics and significance tests (docs/RECONSTRUCTION_PLAN.md §5.1, §5.8; step A3).

No data is read here: every function maps counts (or rates) to a number.

Conventions, all taken from the plan:

* success rate, PRE ............ §5.1 (TH §4.1.3, TH p.65)
* pooled two-proportion z-test . §5.8 items 1, 2, 5 (TH Table 5.3, TH Table
  5.11, MLA Table 7 proportion column); two-tailed form for the ablations
  (§5.4, TH p.70)
* one-sample z-test ............ §5.8 item 3 (TH Table 5.4, MLA Fig 2)
* McNemar ...................... §5.8 item 4 / changelog V2 (two-sided
  chi-square with continuity correction; TH Table 5.11, MLA Table 7)

Tail probability. The plan (§5.8 item 1) defines the p-value as the one given
by ``statsmodels ... proportions_ztest(..., alternative='larger')``, i.e. the
exact upper tail (survival function). That is the default here
(``tail="sf"``). The printed tables are also consistent with p computed as
``1 - Phi(z)`` in double precision: TH Table 5.3 HAC row (3.18856E-13,
1.57874E-13) and every printed "0" (TH Table 5.3 DA, TH Table 5.4 DA, MLA
Table 7 HAC/DA proportion cells). ``tail="one_minus_cdf"`` gives that form.
It is paper evidence, not a plan convention, so it is opt-in.
"""
import math
from fractions import Fraction

import numpy as np
from scipy.stats import chi2, norm
from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar

_ALTERNATIVES = ("larger", "smaller", "two-sided")
_TAILS = ("sf", "one_minus_cdf")


# --------------------------------------------------------------------------
# Metrics (plan §5.1)
# --------------------------------------------------------------------------

def success_rate(n_correct, n_total):
    """Success rate = share of cases where the returned answer equals a*.

    Plan §5.1 (TH §4.1.3, §4.2.3). Example: T1, 935/1209 = 0.773.
    """
    if n_total <= 0:
        raise ValueError("n_total must be positive")
    if not 0 <= n_correct <= n_total:
        raise ValueError("n_correct must lie in [0, n_total]")
    return n_correct / n_total


def success_rate_from_outcomes(correct):
    """Success rate of a sequence of per-case outcomes (True = correct).

    Plan §5.1. Same quantity as :func:`success_rate`.
    """
    correct = np.asarray(correct, dtype=bool)
    return success_rate(int(correct.sum()), int(correct.size))


def round_half_up(numerator, denominator=1, decimals=3):
    """numerator / denominator rounded half UP to `decimals` places.

    For comparing a count ratio with a printed table cell. The thesis
    rounds half up: TH Table 5.5 Politics SP prints 0.813 for 13/16 =
    0.8125, where Python's round() (half to even) gives 0.812; a negative
    half is rounded away from zero
    (decision ST4). The ratio is taken exactly (integers or
    Fractions), so no float noise enters. Returns a float.
    """
    value = Fraction(numerator) / Fraction(denominator)
    scaled = abs(value) * 10 ** decimals
    rounded = math.floor(scaled + Fraction(1, 2))       # exact; half goes up
    sign = -1 if value < 0 else 1
    return sign * rounded / 10 ** decimals


def pre(success_1, success_2):
    """Proportional Reduction in Error, (E1 - E2) / E1 with E = 1 - success.

    Plan §5.1 (TH p.65). ``success_1`` is the reference rate (the rule, or
    P(success by AG_i)); ``success_2`` is the compared rate (the approach, or
    P(success by AG_i | AMP selected AG_i)). Negative when success_2 is lower.
    """
    e1 = 1.0 - success_1
    e2 = 1.0 - success_2
    if e1 == 0:
        raise ZeroDivisionError("PRE is undefined when the reference error is 0")
    return (e1 - e2) / e1


def pre_from_rounded_rates(success_1, success_2, decimals=3):
    """PRE computed from rates rounded to ``decimals`` places first.

    Plan T4, T25 and D22: the headline PRE values (.204/.211 for
    AMP/DAP vs SP; .348/.185 for RP/AP vs MR) were computed from the printed
    3-decimal rates, not from counts.
    """
    return pre(round(success_1, decimals), round(success_2, decimals))


def pre_from_counts(k_1, n_1, k_2, n_2):
    """PRE computed from exact counts: reference k_1/n_1, compared k_2/n_2.

    Plan §5.8 item 3 and D22: the PRE rows of TH Tables 5.4, 5.5 and
    A.10 use exact counts (k_1/n_1 = the rule's uniform success, k_2/n_2 =
    its success in the cases where AMP selected it).
    """
    return pre(success_rate(k_1, n_1), success_rate(k_2, n_2))


# --------------------------------------------------------------------------
# z-tests for proportions (plan §5.8 items 1, 2, 3, 5; §5.4)
# --------------------------------------------------------------------------

def _p_from_z(z, alternative, tail):
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}")
    if tail not in _TAILS:
        raise ValueError(f"tail must be one of {_TAILS}")
    if tail == "sf":
        upper = lambda x: float(norm.sf(x))          # noqa: E731
    else:
        upper = lambda x: float(1.0 - norm.cdf(x))   # noqa: E731
    if alternative == "larger":
        return upper(z)
    if alternative == "smaller":
        return upper(-z)
    return 2.0 * upper(abs(z))


def two_proportion_ztest(x1, n1, x2, n2, alternative="larger", tail="sf"):
    """Pooled two-sample z-test for proportions. Returns (z, p).

    Plan §5.8 item 1: TH Table 5.3 is this test, one-tailed with
    H1: x1/n1 > x2/n2 (approach > rule), n1 = n2 = 1209. With the default
    ``tail="sf"`` it equals ``statsmodels.stats.proportion.proportions_ztest(
    [x1, x2], [n1, n2], alternative='larger')``. The same test with n = 77 is
    the proportion column of TH Table 5.11 (item 2) and, with the LOO counts
    975/982, the proportion column of MLA Table 7 (item 5).

    ``alternative="two-sided"`` is the two-tailed form used for the ablation
    rows against the full version (plan §5.4, TH p.70).

    Pooled SE: sqrt(p(1-p)(1/n1 + 1/n2)) with p = (x1+x2)/(n1+n2).
    ``tail``: see the module docstring.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 and n2 must be positive")
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        raise ZeroDivisionError("pooled standard error is 0")
    z = (p1 - p2) / se
    return z, _p_from_z(z, alternative, tail)


def one_sample_proportion_ztest(k, n, p0, alternative="larger", tail="sf"):
    """One-sample z-test of k/n against a fixed p0. Returns (z, p).

    Plan §5.8 item 3: the per-method p-values of TH Table 5.4 (and MLA Fig 2)
    test P(success | selected) = k/n, n = number of cases where AMP selected
    the method, against the fixed p0 = P(success) of that method over all
    cases, with the null standard error sqrt(p0(1-p0)/n). One-tailed,
    H1: k/n > p0. The pooled two-sample variant does not match the table.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < p0 < 1:
        raise ValueError("p0 must lie strictly between 0 and 1")
    z = (k / n - p0) / math.sqrt(p0 * (1.0 - p0) / n)
    return z, _p_from_z(z, alternative, tail)


# --------------------------------------------------------------------------
# McNemar (plan §5.8 item 4; changelog V2)
# --------------------------------------------------------------------------

def mcnemar_test(b, c):
    """McNemar test from the discordant counts. Returns (chi2, p).

    Plan §5.8 item 4: two-sided chi-square with continuity correction,
    statistic (|b - c| - 1)^2 / (b + c), 1 degree of freedom. ``b`` = cases
    only the first method got right, ``c`` = cases only the second got right.
    Equal to ``statsmodels.stats.contingency_tables.mcnemar(table,
    exact=False, correction=True)``, which is what is called here. The
    concordant cells do not enter the statistic. The plan gives no rule for
    b + c = 0; that input is rejected.
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    if b + c == 0:
        raise ValueError("McNemar is undefined when b + c = 0 (plan is silent)")
    res = _sm_mcnemar([[0, b], [c, 0]], exact=False, correction=True)
    return float(res.statistic), float(res.pvalue)


def mcnemar_statistic(b, c):
    """The printed formula (|b - c| - 1)^2 / (b + c) and its chi2(1) p-value.

    Plan §5.8 item 4. Independent of statsmodels; used to cross-check
    :func:`mcnemar_test`.
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    if b + c == 0:
        raise ValueError("McNemar is undefined when b + c = 0 (plan is silent)")
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return stat, float(chi2.sf(stat, 1))


def discordant_counts(correct_a, correct_b):
    """(b, c) for two paired outcome vectors (True = correct).

    b = cases where only A is correct, c = cases where only B is correct
    (the order used in plan T22: approach-only, rule-only).
    """
    a = np.asarray(correct_a, dtype=bool)
    r = np.asarray(correct_b, dtype=bool)
    if a.shape != r.shape:
        raise ValueError("outcome vectors must have the same length")
    return int(np.sum(a & ~r)), int(np.sum(~a & r))


def mcnemar_from_outcomes(correct_a, correct_b):
    """McNemar (plan §5.8 item 4) for two paired outcome vectors."""
    return mcnemar_test(*discordant_counts(correct_a, correct_b))
