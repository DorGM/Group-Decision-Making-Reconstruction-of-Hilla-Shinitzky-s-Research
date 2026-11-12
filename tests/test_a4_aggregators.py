"""A4 acceptance (plan §9.3): aggregators MR/WC/HAC/SP + DA with method
sub-sets.

Checks:
  * the two DA examples of TH §3.3.1.1 p.35 (plan §3.2 "Deterministic unit
    tests from TH p.35");
  * binary identity: DA = MR's answer iff MR is the lone dissenter, else
    the answer MR did not choose (plan §3.2 "Binary sanity checks");
  * DA recomputed under ablation sub-sets (TH §4.1.5 p.48; plan §5.4).
"""
import itertools
import math
import random

import pytest

from cdm import aggregators as AG
from cdm import primitives as P
from cdm.schema import Response

A2 = ("a1", "a2")


def R(*rows):
    """rows: (vote, confidence, ps tuple, count)."""
    out = []
    for v, c, ps, k in rows:
        out += [Response(v, c, ps)] * k
    return out


# ---- MR / WC / HAC / SP definitions (TH §3.1 p.24-25) ---------------------

# 6 vote a1 with low confidence, 4 vote a2 with high confidence.
# Everybody predicts 70% for a1.
CASE = R(("a1", 0.25, (0.7, 0.3), 6), ("a2", 0.75, (0.7, 0.3), 4))


def test_scores_follow_the_thesis_formulas():
    assert AG.mr_scores(CASE, A2) == pytest.approx([0.6, 0.4])
    # WC: S(a) * Average(C(R(a)))  (TH p.25)
    assert AG.wc_scores(CASE, A2) == pytest.approx([0.6 * 0.25, 0.4 * 0.75])
    # HAC: Average(C(R(a)))  (TH p.25)
    assert AG.hac_scores(CASE, A2) == pytest.approx([0.25, 0.75])
    # SP: S(a) - Average(PS(a, R))  (TH p.25)
    assert AG.sp_scores(CASE, A2) == pytest.approx([0.6 - 0.7, 0.4 - 0.3])


def test_choices():
    assert AG.mr(CASE, A2) == "a1"
    assert AG.wc(CASE, A2) == "a2"
    assert AG.hac(CASE, A2) == "a2"
    assert AG.sp(CASE, A2) == "a2"


def test_wc_equals_confidence_sum_over_n():
    # plan §3.2 WC row: S(a)*Average(C(R(a))) = sum_{r in R(a)} r.c / |R|
    rng = random.Random(1)
    for _ in range(50):
        rs = [Response(rng.choice(A2), rng.random(), (0.5, 0.5))
              for _ in range(rng.randint(2, 40))]
        for a, w in zip(A2, AG.wc_scores(rs, A2)):
            sup = P.supporters(a, rs)
            lit = P.support(a, rs) * P.average(P.confidences(sup)) if sup else 0.0
            assert w == pytest.approx(lit, abs=1e-12)


def test_hac_single_confident_supporter_wins():
    # plan §3.2 HAC row: "An answer with one very confident supporter can win"
    rs = R(("a1", 0.5, (0.5, 0.5), 9), ("a2", 1.0, (0.5, 0.5), 1))
    assert AG.hac(rs, A2) == "a2"
    assert AG.mr(rs, A2) == "a1"


def test_zero_vote_answers():
    A3 = ("a1", "a2", "a3")
    rs = R(("a1", 0.5, (0.5, 0.3, 0.2), 3), ("a2", 0.75, (0.5, 0.3, 0.2), 1))
    assert AG.wc_scores(rs, A3)[2] == 0.0            # empty sum / |R|
    assert math.isnan(AG.hac_scores(rs, A3)[2])      # undefined average
    assert AG.sp_scores(rs, A3)[2] == pytest.approx(0 - 0.2)
    assert AG.hac(rs, A3) == "a2"
    assert AG.tied_answers("HAC", rs, A3) == ["a2"]


def test_sp_difference_form_for_three_answers():
    # plan §8 OQ38: thesis difference form for m > 2
    A3 = ("a1", "a2", "a3")
    rs = R(("a1", 0.5, (0.6, 0.3, 0.1), 5), ("a2", 0.5, (0.6, 0.1, 0.3), 3),
           ("a3", 0.5, (0.5, 0.25, 0.25), 2))
    s = AG.sp_scores(rs, A3)
    assert s == pytest.approx([0.5 - 0.58, 0.3 - 0.23, 0.2 - 0.19])
    assert AG.sp(rs, A3) == "a2"


# ---- ties (plan §8 OQ8: canonical first, tied set exposed) ----------------

def test_ties_canonical_first_and_exposed():
    rs = R(("a1", 0.5, (0.5, 0.5), 2), ("a2", 0.5, (0.5, 0.5), 2))
    for m in AG.RULES:
        assert AG.tied_answers(m, rs, A2) == ["a1", "a2"]
        assert AG.choose(m, rs, A2) == "a1"
        # the canonical order decides, not the names
        assert AG.choose(m, rs, ("a2", "a1")) == "a2"


def test_wc_tie_with_different_group_sizes_is_exact_for_dyadic_values():
    # 3 x 0.5 = 2 x 0.75 = 1.5 ; N = 9. Sum form gives an exact float tie.
    rs = R(("a1", 0.5, (0.5, 0.5), 3), ("a2", 0.75, (0.5, 0.5), 2),
           ("a1", 0.0, (0.5, 0.5), 4))
    assert AG.tied_answers("WC", rs, A2) == ["a1", "a2"]


def test_scores_do_not_depend_on_response_order():
    rng = random.Random(7)
    rs = [Response(rng.choice(A2), rng.random(), (0.3, 0.7)) for _ in range(60)]
    base = {m: AG.scores(m, rs, A2) for m in AG.RULES}
    for _ in range(20):
        rng.shuffle(rs)
        for m in AG.RULES:
            assert AG.scores(m, rs, A2) == base[m]   # exact equality


def test_unknown_policy_and_method():
    with pytest.raises(NotImplementedError):
        AG.mr(CASE, A2, policy="random")
    with pytest.raises(ValueError):
        AG.scores("XX", CASE, A2)


# ---- DA: the two examples of TH p.35 --------------------------------------

def test_da_example_1_th_p35():
    # TH p.35: f_MR=a1, f_WC=a1, f_HAC=a2, f_SP=a1 -> DA = a2
    ch = {"MR": "a1", "WC": "a1", "HAC": "a2", "SP": "a1"}
    d = AG.da_detail(ch, A2)
    assert d["answer"] == "a2"
    assert d["counts"] == {"a1": 3, "a2": 1}
    assert d["mr_tie_break"] is False


def test_da_example_2_th_p35_tie_break():
    # TH p.35: f_MR=a1, f_WC=a1, f_HAC=a2, f_SP=a2 -> DA = a2 (not chosen by MR)
    ch = {"MR": "a1", "WC": "a1", "HAC": "a2", "SP": "a2"}
    d = AG.da_detail(ch, A2)
    assert d["answer"] == "a2"
    assert d["counts"] == {"a1": 2, "a2": 2}
    assert d["mr_tie_break"] is True
    # the tie-break follows MR, not the canonical order
    ch2 = {"MR": "a2", "WC": "a2", "HAC": "a1", "SP": "a1"}
    assert AG.da_from_choices(ch2, A2) == "a1"


# ---- DA: binary identity (plan §3.2 "Binary sanity checks") ----------------

def _other(a):
    return "a2" if a == "a1" else "a1"


SUBSETS = [("MR", "WC", "HAC", "SP"),   # rows 1-4, 8 of plan §5.4
           ("MR", "SP"),                # rows 5, 9  (WC, HAC removed)
           ("MR", "WC", "HAC"),         # rows 6, 10 (SP removed)
           ("MR",)]                     # rows 7, 11 (WC, HAC, SP removed)


@pytest.mark.parametrize("methods", SUBSETS)
def test_da_binary_identity_all_combinations(methods):
    """DA = MR's answer iff MR is the lone dissenter (1 vs >= 2), else not-MR."""
    for combo in itertools.product(A2, repeat=len(methods)):
        ch = dict(zip(methods, combo))
        others = [ch[m] for m in methods if m != "MR"]
        lone = len(others) >= 2 and all(o != ch["MR"] for o in others)
        expected = ch["MR"] if lone else _other(ch["MR"])
        assert AG.da_from_choices(ch, A2, methods) == expected, (methods, ch)


def test_da_all_four_agree_returns_other_answer():
    # plan §3.2: "When all 4 rules agree, DA returns the other answer."
    for a in A2:
        ch = {m: a for m in AG.RULES}
        assert AG.da_from_choices(ch, A2) == _other(a)


def test_da_binary_identity_on_random_cases():
    rng = random.Random(0)
    seen = set()
    for _ in range(400):
        n = rng.randint(4, 40)
        rs = []
        for _ in range(n):
            x = rng.choice([0.2, 0.4, 0.6, 0.8])
            v = rng.choice(A2)
            ps = (x, 1 - x) if v == "a1" else (1 - x, x)
            rs.append(Response(v, rng.choice([0, 0.25, 0.5, 0.75, 1.0]), ps))
        out = AG.aggregate_all(rs, A2)
        assert set(out) == {"MR", "WC", "HAC", "SP", "DA"}
        lone = all(out[m] != out["MR"] for m in ("WC", "HAC", "SP"))
        assert out["DA"] == (out["MR"] if lone else _other(out["MR"]))
        assert AG.da(rs, A2) == out["DA"]
        seen.add(lone)
    assert seen == {True, False}       # both branches were exercised


# ---- DA under ablation sub-sets (TH §4.1.5 p.48; plan §5.4) ---------------

def test_da_recomputed_under_subsets():
    # MR=a1, WC=a2, HAC=a2, SP=a2: MR is the lone dissenter -> DA = a1.
    ch = {"MR": "a1", "WC": "a2", "HAC": "a2", "SP": "a2"}
    assert AG.da_from_choices(ch, A2) == "a1"
    # SP removed: counts a1=1, a2=2 -> still a1
    assert AG.da_from_choices(ch, A2, ("MR", "WC", "HAC")) == "a1"
    # WC, HAC removed: counts 1/1 -> tie -> not MR -> a2 (the answer changes)
    assert AG.da_from_choices(ch, A2, ("MR", "SP")) == "a2"
    # only MR left: "with only MR + DA, DA = not MR" (plan §5.4)
    assert AG.da_from_choices(ch, A2, ("MR",)) == "a2"


def test_da_ignores_excluded_methods():
    # TH p.48: "if a method's label was excluded, the DA did not consider
    # this method's choice in its aggregation procedure."
    ch = {"MR": "a1", "WC": "a1", "HAC": "a2", "SP": "a2"}
    d = AG.da_detail(ch, A2, ("MR", "WC", "HAC"))
    assert d["counts"] == {"a1": 2, "a2": 1}
    assert d["answer"] == "a2"
    # only the choices of the kept methods are needed
    assert AG.da_from_choices({"MR": "a1", "SP": "a2"}, A2, ("MR", "SP")) == "a2"


def test_aggregate_all_with_subset_returns_only_kept_methods():
    out = AG.aggregate_all(CASE, A2, ("MR", "SP"))
    assert set(out) == {"MR", "SP", "DA"}
    assert out == {"MR": "a1", "SP": "a2", "DA": "a2"}
    out = AG.aggregate_all(CASE, A2, ("MR",))
    assert out == {"MR": "a1", "DA": "a2"}
    # full set on CASE: MR=a1 is the lone dissenter -> DA = a1
    assert AG.aggregate_all(CASE, A2)["DA"] == "a1"


def test_da_requires_mr_and_known_methods():
    ch = {"MR": "a1", "WC": "a1", "HAC": "a2", "SP": "a2"}
    with pytest.raises(ValueError):
        AG.da_from_choices(ch, A2, ("WC", "SP"))
    with pytest.raises(ValueError):
        AG.da_from_choices(ch, A2, ("MR", "XX"))
    with pytest.raises(ValueError):
        AG.da_from_choices({"MR": "a1"}, A2)          # missing choices
    with pytest.raises(ValueError):
        AG.da_from_choices({**ch, "SP": "zz"}, A2)    # not an answer


# ---- DA for |A| > 2 (plan §8 OQ8b; not needed for World A) ----------------

def test_da_counts_zero_vote_answers_and_oq8b():
    A3 = ("a1", "a2", "a3")
    ch = {"MR": "a1", "WC": "a1", "HAC": "a2", "SP": "a1"}
    # counts a1=3, a2=1, a3=0 -> a3 (an answer nobody chose; TH p.34-35:
    # "considering every option available in A_P")
    assert AG.da_from_choices(ch, A3) == "a3"
    # all agree: a2 and a3 tie at 0, both non-MR -> OQ8b: lowest support
    ch = {m: "a1" for m in AG.RULES}
    with pytest.raises(ValueError):
        AG.da_from_choices(ch, A3)                    # support is needed
    d = AG.da_detail(ch, A3, support={"a1": 0.5, "a2": 0.3, "a3": 0.2})
    assert d["answer"] == "a3" and d["oq8b_used"] is True
    d = AG.da_detail(ch, A3, support={"a1": 0.5, "a2": 0.2, "a3": 0.3})
    assert d["answer"] == "a2"
    # MR's answer among the tied lowest: it is removed first (TH p.35)
    ch = {"MR": "a1", "WC": "a2", "HAC": "a3", "SP": "a3"}
    A4 = ("a1", "a2", "a3", "a4")
    d = AG.da_detail(ch, ("a1", "a2", "a3"))
    assert d["tied_lowest"] == ["a1", "a2"] and d["answer"] == "a2"
    assert d["oq8b_used"] is False
    assert AG.da_from_choices(ch, A4) == "a4"


# ---- tie tolerance for the float scores ----------------
# Amendment AM3 (config `tie_abs_tol`, 0 = exact).

def _cfg(**changes):
    import copy
    from cdm.config import load_config
    cfg = copy.deepcopy(load_config())
    for key, value in changes.items():
        if key == "tie_abs_tol":
            cfg[key] = value
        else:
            cfg["open_questions"][key]["default"] = value
    return cfg


# True ties on non-dyadic values. Each row: method, responses.
_HALF = (0.5, 0.5)
TRUE_TIES = {
    # WC: (0.1 + 0.2)/3 vs 0.3/3
    "WC_.1+.2_vs_.3": ("WC", R(("a1", 0.1, _HALF, 1), ("a1", 0.2, _HALF, 1),
                               ("a2", 0.3, _HALF, 1))),
    # WC: 2 supporters at 0.3 vs 3 at 0.2
    "WC_2x.3_vs_3x.2": ("WC", R(("a1", 0.3, _HALF, 2), ("a2", 0.2, _HALF, 3))),
    # HAC: mean(.1, .2, .3) vs .2
    "HAC_mean_vs_.2": ("HAC", R(("a1", 0.1, _HALF, 1), ("a1", 0.2, _HALF, 1),
                                ("a1", 0.3, _HALF, 1), ("a2", 0.2, _HALF, 1))),
}


@pytest.mark.parametrize("name", TRUE_TIES)
def test_r1_true_float_ties_are_detected(name):
    method, rs = TRUE_TIES[name]
    sc = AG.scores(method, rs, A2)
    assert sc[0] != sc[1] and abs(sc[0] - sc[1]) < 1e-15   # rounding noise only
    assert len(AG.tied_answers(method, rs, A2, tol=0)) == 1       # exact: missed
    assert AG.tied_answers(method, rs, A2) == ["a1", "a2"]        # default tol
    assert AG.choose(method, rs, A2) == "a1"                      # canonical first


@pytest.mark.parametrize("name", TRUE_TIES)
def test_r1_tie_choice_is_invariant_to_answer_listing_and_response_order(name):
    """With the tolerance, a true tie goes to the first listed answer for
    either listing order and for any order of the responses; with exact
    comparison the winner is fixed by rounding noise."""
    method, rs = TRUE_TIES[name]
    swapped = [Response({"a1": "a2", "a2": "a1"}[r.v], r.c, r.ps) for r in rs]
    # relabelling the answers must not change which LISTED position wins
    assert AG.choose(method, rs, A2) == "a1"
    assert AG.choose(method, swapped, A2) == "a1"
    assert AG.choose(method, rs, ("a2", "a1")) == "a2"
    # exact comparison: the two relabelled data sets give different winners
    assert AG.choose(method, rs, A2, tol=0) != AG.choose(method, swapped, A2, tol=0)
    rng = random.Random(3)
    shuffled = list(rs)
    for _ in range(20):
        rng.shuffle(shuffled)
        assert AG.tied_answers(method, shuffled, A2) == ["a1", "a2"]


def test_r1_tolerance_keeps_real_gaps_and_is_not_used_for_counts():
    # smallest real WC gap on a 0.01 grid: 0.01/N. N = 1000 -> 1e-5 >> 1e-9
    rs = R(("a1", 0.50, _HALF, 500), ("a2", 0.50, _HALF, 499), ("a2", 0.51, _HALF, 1))
    assert AG.tied_answers("WC", rs, A2) == ["a2"]
    # MR compares integer counts: a huge tol does not create a tie
    assert AG.tied_answers("MR", CASE, A2, tol=0.5) == ["a1"]
    # DA counters and the OQ8b support comparison are exact: 0.3 vs 0.3+1e-12
    d = AG.da_detail({"MR": "a", "WC": "a", "HAC": "a", "SP": "a"}, ("a", "b", "c"),
                     support={"a": 0.4, "b": 0.3 + 1e-12, "c": 0.3})
    assert d["answer"] == "c" and d["oq8b_tied"] == ["c"]
    with pytest.raises(ValueError):
        AG.tied_answers("WC", CASE, A2, tol=-1)


def test_r1_feature_12_uses_the_tolerance_through_f_hac():
    from cdm.features import cdmcr
    _, rs = TRUE_TIES["HAC_mean_vs_.2"]                 # a_max = a1, HAC tied
    assert cdmcr.features_1_17(rs, A2)["B_Amax_MaxCa"] == 1
    assert cdmcr.features_1_17(rs, A2, tol=0)["B_Amax_MaxCa"] == 0   # noise decides


# ---- OQ8 / OQ8b / OQ38 wiring, Step D5 tie ladder ------

def test_r3_options_from_config_defaults():
    assert AG.options_from_config() == {"policy": "canonical_first", "tol": 1e-9}
    tied = R(("a1", 0.5, _HALF, 2), ("a2", 0.5, _HALF, 2))
    opts = AG.options_from_config()
    assert AG.aggregate_all(tied, A2, **opts) == AG.aggregate_all(tied, A2)


def test_r3_changing_oq8_changes_the_result():
    tied = R(("a1", 0.5, _HALF, 2), ("a2", 0.5, _HALF, 2))
    opts = AG.options_from_config(_cfg(OQ8="canonical_second"))
    assert opts["policy"] == "canonical_second"
    for m in AG.RULES:
        assert AG.choose(m, tied, A2, **opts) == "a2"
        assert AG.choose(m, tied, A2) == "a1"
    # untied case: the policy changes nothing
    assert AG.aggregate_all(CASE, A2, **opts) == AG.aggregate_all(CASE, A2)


def test_r3_changing_tie_abs_tol_changes_the_result():
    method, rs = TRUE_TIES["WC_2x.3_vs_3x.2"]
    exact = AG.options_from_config(_cfg(tie_abs_tol=0))
    assert exact["tol"] == 0.0
    assert AG.wc(rs, A2, **exact) == "a2"
    assert AG.wc(rs, A2, **AG.options_from_config()) == "a1"


@pytest.mark.parametrize("key,value,error", [
    ("OQ8", "random", NotImplementedError),
    ("OQ8", "coin", ValueError),
    ("OQ8b", "canonical_first", NotImplementedError),
    ("OQ38", "prelec_normalized_form", NotImplementedError),
    ("tie_abs_tol", "1e-9", ValueError),
])
def test_r3_unimplemented_switch_values_raise(key, value, error):
    with pytest.raises(error):
        AG.options_from_config(_cfg(**{key: value}))


def test_r3_canonical_second_needs_a_two_way_tie():
    a3 = ("a", "b", "c")
    p3 = (0.4, 0.3, 0.3)
    three = R(("a", 0.5, p3, 1), ("b", 0.5, p3, 1), ("c", 0.5, p3, 1))
    with pytest.raises(NotImplementedError):
        AG.mr(three, a3, policy="canonical_second")
    two = three + R(("b", 0.5, p3, 1), ("c", 0.5, p3, 1))       # b, c tied on top
    assert AG.mr(two, a3, policy="canonical_second") == "c"
    assert AG.mr(two, a3) == "b"


def test_r3_sp_half_credit_policy():
    """Plan §8 OQ8 "SP tie = 0.5 (Rutchick)", Step D5 item 1."""
    hc = "sp_tie_half_credit"
    # SP tie: everybody predicts the true shares (6:4)
    ps = (0.6, 0.4)
    tied = R(("a1", 0.25, ps, 6), ("a2", 0.75, ps, 4))
    assert AG.tied_answers("SP", tied, A2) == ["a1", "a2"]
    assert AG.success_credit("SP", tied, A2, "a1", policy=hc) == 0.5
    assert AG.success_credit("SP", tied, A2, "a2", policy=hc) == 0.5
    # default policy: canonical first -> full or no credit
    assert AG.success_credit("SP", tied, A2, "a1") == 1.0
    assert AG.success_credit("SP", tied, A2, "a2") == 0.0
    assert AG.success_credit("SP", tied, A2, "a2", policy="canonical_second") == 1.0
    # the other rules and untied SP cases are scored on their answer
    assert AG.success_credit("MR", tied, A2, "a1", policy=hc) == 1.0
    assert AG.success_credit("HAC", tied, A2, "a1", policy=hc) == 0.0
    assert AG.success_credit("SP", CASE, A2, "a2", policy=hc) == 1.0
    assert AG.choose("SP", CASE, A2, policy=hc) == "a2"
    assert AG.choose("MR", tied, A2, policy=hc) == "a1"
    # the single SP answer of a tied case (and so DA) is not defined by the plan
    with pytest.raises(NotImplementedError):
        AG.choose("SP", tied, A2, policy=hc)
    with pytest.raises(NotImplementedError):
        AG.aggregate_all(tied, A2, policy=hc)
    # more than two answers: no rule
    a3 = ("a", "b", "c")
    p3 = (0.5, 0.25, 0.25)
    t3 = R(("a", 0.5, p3, 2), ("b", 0.5, p3, 1), ("c", 0.5, p3, 1))
    assert len(AG.tied_answers("SP", t3, a3)) == 3
    with pytest.raises(NotImplementedError):
        AG.success_credit("SP", t3, a3, "a", policy=hc)
    with pytest.raises(ValueError):
        AG.success_credit("SP", tied, A2, "zz")
    with pytest.raises(ValueError):
        AG.success_credit("XX", tied, A2, "a1")
