"""A2 acceptance (plan §9.3): S=(67,23)/90 -> ES 0.820, VarS 0.244,
D_S_Uniform 0.346, D 0.489 (TH Table A.2)."""
import math

import numpy as np
import pytest

from cdm import primitives as P
from cdm.schema import Response

ANSWERS = ("a1", "a2")


def _responses(n1, n2, c1=0.8, c2=0.6):
    return ([Response("a1", c1, (0.7, 0.3))] * n1
            + [Response("a2", c2, (0.4, 0.6))] * n2)


R90 = _responses(67, 23)


def test_table_a2_values():
    s = P.support_vector(R90, ANSWERS)
    assert s == pytest.approx([67 / 90, 23 / 90])
    # feature 3, ES
    assert round(P.entropy(s), 3) == 0.820
    # feature 4, VarS (population SD)
    assert round(P.var(s), 3) == 0.244
    # feature 5, D_S_Uniform (Euclidean distance to the uniform vector)
    uniform = np.full(len(ANSWERS), 1 / len(ANSWERS))
    assert round(float(np.linalg.norm(np.array(s) - uniform)), 3) == 0.346
    # feature 2, D_Smax_Smin
    d = (P.support(P.a_max(R90, ANSWERS), R90)
         - P.support(P.a_min(R90, ANSWERS), R90))
    assert round(d, 3) == 0.489


def test_var_switch():
    s = [67 / 90, 23 / 90]
    assert P.var(s) == pytest.approx(abs(67 / 90 - 0.5))  # binary: SD = |S - .5|
    assert P.var(s, kind="population_variance") == pytest.approx((67 / 90 - 0.5) ** 2)
    assert P.var([1, 2, 3, 4]) == pytest.approx(math.sqrt(1.25))  # ddof = 0
    with pytest.raises(ValueError):
        P.var(s, kind="sample")


def test_entropy_base_2_and_zero_shares():
    assert P.entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert P.entropy([0.25] * 4) == pytest.approx(2.0)
    assert P.entropy([1.0, 0.0]) == 0.0
    assert P.entropy([0.5, 0.5, 0.0]) == pytest.approx(1.0)


def test_supporters_and_groups():
    assert len(P.supporters("a1", R90)) == 67
    assert len(P.non_supporters("a1", R90)) == 23
    r = R90[0]
    assert len(P.in_group(r, R90)) == 67  # includes r itself
    assert len(P.out_group(r, R90)) == 23
    assert P.vote_counts(R90, ANSWERS) == [67, 23]


def test_support_vector_includes_zero_vote_answers():
    answers = ("a", "b", "c")
    R = [Response("a", 0.5, (0.5, 0.3, 0.2))] * 3 + [Response("b", 0.5, (0.5, 0.3, 0.2))]
    assert P.support_vector(R, answers) == [0.75, 0.25, 0.0]
    assert P.a_min(R, answers) == "c"
    assert P.a_max(R, answers) == "a"


def test_c_ps_psv():
    R = [Response("a1", 0.9, (0.7, 0.3)), Response("a2", 0.2, (0.4, 0.6))]
    assert P.confidences(R) == [0.9, 0.2]
    assert P.predicted_support("a1", R, ANSWERS) == [0.7, 0.4]
    assert P.predicted_support("a2", R, ANSWERS) == [0.3, 0.6]
    assert P.predicted_support_of_vote(R, ANSWERS) == [0.7, 0.6]
    assert P.average(P.confidences(R)) == pytest.approx(0.55)
    assert P.maximum(P.confidences(R)) == 0.9
    assert P.minimum(P.confidences(R)) == 0.2


def test_table_a2_avgc_identity():
    # plan §3.3.2: AvgC = (67*.788 + 23*.634)/90 = 0.749
    R = _responses(67, 23, c1=0.788, c2=0.634)
    assert round(P.average(P.confidences(R)), 3) == 0.749
    assert round(P.average(P.confidences(P.supporters("a1", R))), 3) == 0.788


def test_empty_sets_give_nan():
    assert math.isnan(P.average([]))
    assert math.isnan(P.var([]))
    assert math.isnan(P.maximum([]))
    assert math.isnan(P.minimum([]))
    assert math.isnan(P.support("a1", []))
    assert math.isnan(P.average(P.confidences(P.supporters("a2", _responses(5, 0)))))


def test_tie_policy_canonical_first():
    R = _responses(5, 5)
    assert P.a_max_ties(R, ANSWERS) == ["a1", "a2"]
    assert P.a_min_ties(R, ANSWERS) == ["a1", "a2"]
    assert P.a_max(R, ANSWERS) == "a1"
    assert P.a_min(R, ANSWERS) == "a1"
    assert P.a_max(R, ("a2", "a1")) == "a2"  # order of A_P decides
    assert P.tied_best([0.2, 0.9, 0.9], ("a", "b", "c")) == ["b", "c"]
    assert P.tied_best([0.2, 0.9, 0.2], ("a", "b", "c"), largest=False) == ["a", "c"]
    assert P.tied_best([float("nan"), 0.3], ("a", "b")) == ["b"]
    with pytest.raises(NotImplementedError):
        P.break_tie(["a", "b"], policy="random")


# ---- ------------------------------------------------

def test_f6_tied_best_tolerance_am3():
    """Amendment AM3: float scores within tol are tied."""
    # WC true tie: 2 supporters at c=0.3 vs 3 at c=0.2 (N=5), sum-of-c / N
    a, b = math.fsum([0.3] * 2) / 5, math.fsum([0.2] * 3) / 5
    assert a != b                                    # rounding noise
    assert P.tied_best([a, b], ("a", "b")) == ["b"]                 # exact: missed
    assert P.tied_best([a, b], ("a", "b"), tol=1e-9) == ["a", "b"]
    assert P.tied_best([a, b], ("a", "b"), largest=False, tol=1e-9) == ["a", "b"]
    # HAC: mean([.1, .2, .3]) is 0.20000000000000004
    m = sum([0.1, 0.2, 0.3]) / 3
    assert m != 0.2
    assert P.tied_best([m, 0.2], ("a", "b"), tol=P.DEFAULT_TIE_ABS_TOL) == ["a", "b"]
    # a real gap is kept: the smallest true gap on a 0.01 grid at N=1000 is ~4e-8
    assert P.tied_best([0.5, 0.5 + 4e-8], ("a", "b"), tol=1e-9) == ["b"]
    # default tol is 0 (exact): integer vote counts need no tolerance
    assert P.tied_best([3, 3, 2], ("a", "b", "c")) == ["a", "b"]
    assert P.DEFAULT_TIE_ABS_TOL == 1e-9
    for bad in (-1e-9, float("nan"), None, "1e-9", True):
        with pytest.raises(ValueError):
            P.tied_best([0.1, 0.2], ("a", "b"), tol=bad)


def test_f6_sp_true_ties_over_all_splits():
    # Everyone predicts the true shares, so SP is an exact tie (0 for both
    # answers). Exact float equality misses most of them; the tolerance none.
    from cdm import aggregators as AG
    missed_exact = missed_tol = total = 0
    for n in range(2, 131):
        for k in range(1, n):
            ps = (k / n, 1 - k / n)
            rs = [Response("a1", 0.5, ps)] * k + [Response("a2", 0.5, ps)] * (n - k)
            total += 1
            missed_exact += AG.tied_answers("SP", rs, ANSWERS, tol=0) != ["a1", "a2"]
            missed_tol += AG.tied_answers("SP", rs, ANSWERS) != ["a1", "a2"]
    assert total == 8385
    assert missed_exact > 1000
    assert missed_tol == 0


def test_f7_empty_or_nan_input_gives_no_silent_value():
    assert math.isnan(P.entropy([]))
    assert math.isnan(P.entropy([float("nan"), float("nan")]))
    assert math.isnan(P.entropy([0.5, float("nan")]))
    assert math.isnan(P.entropy(P.support_vector([], ANSWERS)))
    assert P.a_max_ties([], ANSWERS) == [] and P.a_min_ties([], ANSWERS) == []
    with pytest.raises(ValueError, match="empty"):
        P.a_max([], ANSWERS)
    with pytest.raises(ValueError, match="empty"):
        P.a_min([], ANSWERS)


def test_f8_nan_handling_does_not_depend_on_order():
    nan = float("nan")
    assert math.isnan(P.maximum([nan, 1])) and math.isnan(P.maximum([1, nan]))
    assert math.isnan(P.minimum([nan, 1])) and math.isnan(P.minimum([1, nan]))
    assert P.maximum([1, 3, 2]) == 3.0 and P.minimum([1, 3, 2]) == 1.0
    nan32 = np.float32("nan")
    assert P.tied_best([nan32, 0.3], ("a", "b")) == ["b"]
    assert P.tied_best([np.float32(0.25), nan32], ("a", "b")) == ["a"]
    assert P.tied_best([nan32, nan], ("a", "b")) == []


def test_f9_entropy_guards_its_input():
    with pytest.raises(ValueError):
        P.entropy([-0.1, 1.1])
    with pytest.raises(ValueError):
        P.entropy([67, 23])                  # counts instead of shares
    for base in (1, 0, -2, 0.5):
        with pytest.raises(ValueError):
            P.entropy([0.5, 0.5], base=base)
    assert P.entropy([0.5, 0.5], base=math.e) == pytest.approx(math.log(2))


def test_break_tie_policies_of_step_d5():
    # plan Step D5 item 1: canonical first -> canonical second
    assert P.break_tie(["a", "b"]) == "a"
    assert P.break_tie(["a", "b"], "canonical_second") == "b"
    assert P.break_tie(["a"], "canonical_second") == "a"      # no tie
    with pytest.raises(NotImplementedError):                  # > 2 tied: undefined
        P.break_tie(["a", "b", "c"], "canonical_second")
    with pytest.raises(NotImplementedError):
        P.break_tie(["a", "b"], "sp_tie_half_credit")
    with pytest.raises(NotImplementedError):
        P.break_tie(["a"], "random")
    with pytest.raises(ValueError):
        P.break_tie(["a", "b"], "coin")
    with pytest.raises(ValueError):
        P.break_tie([], "canonical_first")
