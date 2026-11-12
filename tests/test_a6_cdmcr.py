"""A6 acceptance (plan §9.3): CDMCR features 1-17.

"A synthetic N=90 case matching A.2's free parameters reproduces A.2
features 1-17 to 3 d.p." (TH Table A.2 p.110; plan §3.3.2). The case is in
tests/a2_case.py.
"""
import math

import pytest

from a2_case import ANSWERS, R90, TABLE_A2
from cdm import aggregators as AG
from cdm.features import cdmcr
from cdm.schema import Response


def R(*rows):
    out = []
    for v, c, ps, k in rows:
        out += [Response(v, c, ps)] * k
    return out


F = cdmcr.features_1_17(R90, ANSWERS)


def test_names_and_order():
    # TH Table 3.2 p.28, rows 1-17
    assert tuple(F) == cdmcr.FEATURES_1_17
    assert len(cdmcr.FEATURES_1_17) == 17
    assert len(cdmcr.FEATURE_NAMES) == 27


# What each Table A.2 comparison is worth.
#   evidence        : the printed value follows from N = 90 and the 67/23
#                     split alone and rules out the alternative readings
#   partial         : part of the value is evidence (see the comment)
#   input           : the value was used to build the synthetic case
#   by_construction : the synthetic case was built to meet it; NO evidence
A2_EVIDENCE = {
    "N": "input",
    "D_Smax_Smin": "input",            # the 67/23 split is derived from 0.489
    "ES": "evidence",                  # base e gives 0.568, base 10 gives 0.247
    "VarS": "evidence",                # variance 0.060; sample SD 0.346
    "D_S_Uniform": "evidence",         # Manhattan distance gives 0.489
    "CAmin": "by_construction",
    "CAmax": "by_construction",
    "MinC": "by_construction",
    "AvgC": "partial",                 # the printed .788/.634/.749 are consistent
    "VarC": "by_construction",         # evidence = the feasibility bound below
    "D_MaxC_AvgC": "by_construction",
    "B_Amax_MaxCa": "partial",         # one bit: CAmax .788 > CAmin .634
    "MaxPSa": "by_construction",
    "MinPSa": "by_construction",
    "AvgPSV": "by_construction",
    "P_lowC_highPSv": "partial",       # 0.122 = 11/90: "share of all N"; count built in
    "P_lowPSv_highC": "partial",       # 0.089 = 8/90
}


def test_a2_evidence_labels_cover_all_17_features():
    assert tuple(A2_EVIDENCE) == cdmcr.FEATURES_1_17
    assert sum(v == "by_construction" for v in A2_EVIDENCE.values()) == 8
    assert [k for k, v in A2_EVIDENCE.items() if v == "evidence"] == \
        ["ES", "VarS", "D_S_Uniform"]


@pytest.mark.parametrize(
    "name", cdmcr.FEATURES_1_17,
    ids=[f"{n}[{A2_EVIDENCE[n]}]" for n in cdmcr.FEATURES_1_17])
def test_synthetic_a2_case_is_consistent_with_table_a2(name):
    """The synthetic case of tests/a2_case.py gives TH Table A.2 (p.110) to
    3 d.p. This shows the formulas are CONSISTENT with Table A.2. Only the
    features labelled "evidence" (and partly "partial") test the code
    against the thesis; the "by_construction" ones carry no evidence."""
    assert round(F[name], 3) == TABLE_A2[name], (name, F[name])


def test_table_a2_evidence_on_the_printed_numbers():
    """Checks on the printed values of Table A.2 that discriminate between
    readings. They do not depend on the synthetic case."""
    s = [67 / 90, 23 / 90]
    # ES: only base 2 gives the printed 0.820
    h = lambda base: -sum(x * math.log(x, base) for x in s)
    assert round(h(2), 3) == TABLE_A2["ES"]
    assert round(h(math.e), 3) == 0.568 and round(h(10), 3) == 0.247
    # VarS: only the population SD gives 0.244
    mean = 0.5
    pop_var = sum((x - mean) ** 2 for x in s) / 2
    assert round(math.sqrt(pop_var), 3) == TABLE_A2["VarS"]
    assert round(pop_var, 3) == 0.060
    assert round(math.sqrt(2 * pop_var), 3) == 0.346          # sample SD (ddof 1)
    # D_S_Uniform: Euclidean 0.346; Manhattan would be 0.489
    assert round(math.sqrt(sum((x - mean) ** 2 for x in s)), 3) == TABLE_A2["D_S_Uniform"]
    assert round(sum(abs(x - mean) for x in s), 3) == 0.489
    # VarC = 0.222 must be an SD: with mean 0.749 on [MinC, 1] = [0.05, 1] the
    # largest possible variance is (max - mean)(mean - min) = 0.175 < 0.222
    bound = (1.0 - TABLE_A2["AvgC"]) * (TABLE_A2["AvgC"] - TABLE_A2["MinC"])
    assert round(bound, 3) == 0.175
    assert bound < TABLE_A2["VarC"]
    assert TABLE_A2["VarC"] ** 2 < bound                       # as an SD it is feasible
    # features 16/17 are k/90: shares of all N
    assert round(11 / 90, 3) == TABLE_A2["P_lowC_highPSv"]
    assert round(8 / 90, 3) == TABLE_A2["P_lowPSv_highC"]


def test_table_a2_values_are_not_on_a_rounding_edge():
    # every reproduced value is at least 5e-5 away from a 3-d.p. rounding
    # boundary, so the check above does not hinge on float noise
    for name in cdmcr.FEATURES_1_17:
        x = F[name] * 1000
        assert abs((x % 1) - 0.5) > 0.05 or float(F[name]).is_integer(), (name, F[name])


def test_table_a2_consistency_checks_of_plan_3_3_2():
    # plan §3.3.2: values that follow from N=90 and S(a_max)=67/90 alone
    assert F["D_Smax_Smin"] == pytest.approx(44 / 90)
    assert F["ES"] == pytest.approx(0.8200, abs=5e-5)
    assert F["VarS"] == pytest.approx(abs(67 / 90 - 0.5))      # binary: SD = |S-.5|
    assert F["D_S_Uniform"] == pytest.approx(0.3457, abs=5e-5)
    # AvgC = (67*CAmax + 23*CAmin)/90 ; with the printed .788/.634 -> .749
    assert F["AvgC"] == pytest.approx((67 * F["CAmax"] + 23 * F["CAmin"]) / 90)
    assert round((67 * 0.788 + 23 * 0.634) / 90, 3) == TABLE_A2["AvgC"]
    # features 16/17 are 11/90 and 8/90
    assert F["P_lowC_highPSv"] == pytest.approx(11 / 90)
    assert F["P_lowPSv_highC"] == pytest.approx(8 / 90)
    # binary predicted shares sum to 1
    assert F["MaxPSa"] + F["MinPSa"] == pytest.approx(1.0)
    assert TABLE_A2["MaxPSa"] + TABLE_A2["MinPSa"] == pytest.approx(1.0)
    # Max(C) = AvgC + D_MaxC_AvgC = 1.0
    assert F["AvgC"] + F["D_MaxC_AvgC"] == pytest.approx(1.0)
    assert TABLE_A2["AvgC"] + TABLE_A2["D_MaxC_AvgC"] == pytest.approx(1.0)


def test_varc_is_population_sd_not_variance():
    # plan §3.3 row 10 / OQ9: 0.222 exceeds the largest possible variance here, so Var is read as SD
    v = cdmcr.features_1_17(R90, ANSWERS, var_kind="population_variance")
    assert v["VarC"] == pytest.approx(F["VarC"] ** 2)
    assert round(v["VarC"], 3) != TABLE_A2["VarC"]
    assert round(v["VarS"], 3) != TABLE_A2["VarS"]


# ---- definitions on a small hand-computed case ----------------------------

A3 = ("a1", "a2", "a3")
# votes: a1 x4, a2 x1, a3 x0 ; N = 5
SMALL = R(("a1", 0.25, (0.5, 0.3, 0.2), 2),
          ("a1", 0.75, (0.9, 0.1, 0.0), 2),
          ("a2", 1.00, (0.2, 0.6, 0.2), 1))


def test_small_case_by_hand():
    f = cdmcr.features_1_17(SMALL, A3)
    s = [0.8, 0.2, 0.0]
    assert f["N"] == 5
    # a_min is the zero-vote answer a3 (S over all of A_P, OQ7)
    assert f["D_Smax_Smin"] == pytest.approx(0.8)
    assert f["ES"] == pytest.approx(-(0.8 * math.log2(0.8) + 0.2 * math.log2(0.2)))
    mean = 1 / 3
    assert f["VarS"] == pytest.approx(math.sqrt(sum((x - mean) ** 2 for x in s) / 3))
    assert f["D_S_Uniform"] == pytest.approx(math.sqrt(sum((x - mean) ** 2 for x in s)))
    assert math.isnan(f["CAmin"])                 # a3 has no supporter (OQ7)
    assert f["CAmax"] == pytest.approx(0.5)
    assert f["MinC"] == 0.25
    assert f["AvgC"] == pytest.approx(0.6)
    c = [0.25, 0.25, 0.75, 0.75, 1.0]
    assert f["VarC"] == pytest.approx(math.sqrt(sum((x - 0.6) ** 2 for x in c) / 5))
    assert f["D_MaxC_AvgC"] == pytest.approx(0.4)
    assert f["B_Amax_MaxCa"] == 0                 # HAC picks a2 (1.0 > 0.5)
    # Average(PS(a_i, R)): a1 (1.0+1.8+0.2)/5=.6 ; a2 (.6+.2+.6)/5=.28 ; a3 .12
    assert f["MaxPSa"] == pytest.approx(0.6)
    assert f["MinPSa"] == pytest.approx(0.12)
    # PSV = ps[r.v]: .5,.5,.9,.9,.6 -> mean .68
    assert f["AvgPSV"] == pytest.approx(0.68)
    # 16: c < .6 and psv > .68 -> none ; 17: c > .6 and psv < .68 -> the a2 voter
    assert f["P_lowC_highPSv"] == 0
    assert f["P_lowPSv_highC"] == pytest.approx(1 / 5)


def test_b_amax_maxca_follows_f_hac():
    # TH Table 3.2 row 12: 1 iff f_HAC(R_P) == a_max
    rs = R(("a1", 0.75, (0.5, 0.5), 3), ("a2", 0.5, (0.5, 0.5), 2))
    assert cdmcr.features_1_17(rs, ("a1", "a2"))["B_Amax_MaxCa"] == 1
    rs = R(("a1", 0.25, (0.5, 0.5), 3), ("a2", 0.5, (0.5, 0.5), 2))
    assert cdmcr.features_1_17(rs, ("a1", "a2"))["B_Amax_MaxCa"] == 0
    # HAC tie (plan §8 OQ14): f_HAC with the OQ8 rule = canonical first.
    rs = R(("a1", 0.5, (0.5, 0.5), 3), ("a2", 0.5, (0.5, 0.5), 2))
    assert AG.tied_answers("HAC", rs, ("a1", "a2")) == ["a1", "a2"]
    assert cdmcr.features_1_17(rs, ("a1", "a2"))["B_Amax_MaxCa"] == 1
    # same data, other canonical order: f_HAC = a2 != a_max = a1 -> 0.
    # (The MLA wording, "a_max has the highest average confidence", would
    # give 1 here; the plan follows TH + OQ8.)
    assert cdmcr.features_1_17(rs, ("a2", "a1"))["B_Amax_MaxCa"] == 0


def test_strict_inequalities_in_16_17():
    # plan §8 OQ13: strict, as printed. Everybody equals the averages.
    rs = R(("a1", 0.5, (0.6, 0.4), 4), ("a2", 0.5, (0.4, 0.6), 2))
    f = cdmcr.features_1_17(rs, ("a1", "a2"))
    assert f["P_lowC_highPSv"] == 0 and f["P_lowPSv_highC"] == 0
    g = cdmcr.features_1_17(rs, ("a1", "a2"), strict=False)
    assert g["P_lowC_highPSv"] == 1 and g["P_lowPSv_highC"] == 1


def test_amax_amin_tie_is_canonical_first():
    # plan §8 OQ7
    rs = R(("a1", 0.25, (0.5, 0.5), 2), ("a2", 0.75, (0.5, 0.5), 2))
    f = cdmcr.features_1_17(rs, ("a1", "a2"))
    assert f["D_Smax_Smin"] == 0
    assert f["CAmax"] == 0.25 and f["CAmin"] == 0.25      # both are a1
    g = cdmcr.features_1_17(rs, ("a2", "a1"))
    assert g["CAmax"] == 0.75 and g["CAmin"] == 0.75


def test_feature_groups_of_th_p47():
    # plan §3.3: Vote = {1-5, 18-21}; Conf = {6-12, 16, 17, 22-27}; PS = {13-17}
    g = cdmcr.GROUPS
    assert set(g["Vote"]) == {1, 2, 3, 4, 5, 18, 19, 20, 21}
    assert set(g["Conf"]) == set(range(6, 13)) | {16, 17} | set(range(22, 28))
    assert set(g["PS"]) == {13, 14, 15, 16, 17}
    assert set(g["Vote"]) | set(g["Conf"]) | set(g["PS"]) == set(range(1, 28))
    assert set(g["Conf"]) & set(g["PS"]) == {16, 17}
    assert not set(g["Vote"]) & (set(g["Conf"]) | set(g["PS"]))


# ---- tolerance in features 16/17 -----------------------
# Amendment AM9 (extension of AM3).

def _one(c, x):
    return Response("a1", c, (x, 1 - x))


# 20 responses; confidences 3 x 0.2, 14 x 0.6, 3 x 1.0 (mean exactly 0.6);
# ps[v]: mean 0.53. True values: feature 16 = 3/20, feature 17 = 3/20.
ORDER_CASE = ([_one(0.2, 0.9)] * 3 + [_one(0.6, 0.3)] * 7
              + [_one(0.6, 0.7)] * 7 + [_one(1.0, 0.3)] * 3)


def test_r2_exact_comparison_depends_on_response_order():
    # the defect: the same 20 responses give 3/20 or 10/20
    a = cdmcr.features_1_17(ORDER_CASE, ("a1", "a2"), tol=0)
    b = cdmcr.features_1_17(ORDER_CASE[::-1], ("a1", "a2"), tol=0)
    assert a["P_lowPSv_highC"] == pytest.approx(3 / 20)       # the true value
    assert b["P_lowPSv_highC"] == pytest.approx(10 / 20)      # reversed order


def test_r2_features_16_17_are_invariant_to_response_order():
    import random
    rng = random.Random(0)
    rs = list(ORDER_CASE)
    for _ in range(200):
        rng.shuffle(rs)
        f = cdmcr.features_1_17(rs, ("a1", "a2"))
        assert f["P_lowC_highPSv"] == pytest.approx(3 / 20)
        assert f["P_lowPSv_highC"] == pytest.approx(3 / 20)
        g = cdmcr.features_1_17(rs, ("a1", "a2"), strict=False)
        # non-strict: the 14 at the mean confidence count on the matching side
        assert g["P_lowC_highPSv"] == pytest.approx((3 + 7) / 20)
        assert g["P_lowPSv_highC"] == pytest.approx((3 + 7) / 20)


def test_r2_identical_responses_give_zero_for_every_n():
    # everybody has c = 0.7 and ps[v] = 0.6: nobody is below or above average
    wrong_exact = 0
    for n in range(4, 131):
        rs = [_one(0.7, 0.6)] * n
        f = cdmcr.features_1_17(rs, ("a1", "a2"))
        assert f["P_lowC_highPSv"] == 0 and f["P_lowPSv_highC"] == 0, n
        g = cdmcr.features_1_17(rs, ("a1", "a2"), strict=False)
        assert g["P_lowC_highPSv"] == 1 and g["P_lowPSv_highC"] == 1, n
        e = cdmcr.features_1_17(rs, ("a1", "a2"), tol=0)
        wrong_exact += bool(e["P_lowC_highPSv"] or e["P_lowPSv_highC"])
    assert wrong_exact == 30          # exact comparison is wrong for 30 values of N


def test_r2_tolerance_does_not_change_table_a2_or_real_gaps():
    assert cdmcr.features_1_17(R90, ANSWERS, tol=0) == F
    # a real gap of 0.01 around the mean is kept
    rs = [_one(0.59, 0.61)] * 5 + [_one(0.61, 0.59)] * 5
    f = cdmcr.features_1_17(rs, ("a1", "a2"))
    assert f["P_lowC_highPSv"] == 0.5 and f["P_lowPSv_highC"] == 0.5


# ---- separate OQ7 / OQ8 policies, strict validated -----

def test_r3_amax_policy_and_hac_policy_are_separate():
    # 2:2 vote tie (a_max tie) and a HAC tie
    rs = R(("a1", 0.5, (0.5, 0.5), 2), ("a2", 0.5, (0.5, 0.5), 2))
    a = ("a1", "a2")
    base = cdmcr.features_1_17(rs, a)
    assert base["B_Amax_MaxCa"] == 1                    # a_max = a1, f_HAC = a1
    only_hac = cdmcr.features_1_17(rs, a, hac_policy="canonical_second")
    assert only_hac["B_Amax_MaxCa"] == 0                # f_HAC = a2, a_max = a1
    only_amax = cdmcr.features_1_17(rs, a, amax_policy="canonical_second")
    assert only_amax["B_Amax_MaxCa"] == 0               # a_max = a2, f_HAC = a1
    both = cdmcr.features_1_17(rs, a, policy="canonical_second")
    assert both["B_Amax_MaxCa"] == 1                    # legacy argument sets both
    assert both == cdmcr.features_1_17(rs, a, amax_policy="canonical_second",
                                       hac_policy="canonical_second")
    with pytest.raises(NotImplementedError):
        cdmcr.features_1_17(rs, a, hac_policy="random")


def test_r3_strict_must_be_a_bool():
    for bad in ("strict", "non_strict", "nonstrict", 1, None):
        with pytest.raises(ValueError):
            cdmcr.features_1_17(R90, ANSWERS, strict=bad)


def test_r7_features_1_17_return_info():
    rs = R(("a1", 0.5, (0.5, 0.5), 2), ("a2", 0.5, (0.5, 0.5), 2))
    f, info = cdmcr.features_1_17(rs, ("a1", "a2"), return_info=True)
    assert f == cdmcr.features_1_17(rs, ("a1", "a2"))
    assert info == {"amax_tied": ["a1", "a2"], "amin_tied": ["a1", "a2"],
                    "hac_tied": ["a1", "a2"], "amax_tie": True,
                    "amin_tie": True, "hac_tie": True}
    _, info = cdmcr.features_1_17(R90, ANSWERS, return_info=True)
    assert not (info["amax_tie"] or info["amin_tie"] or info["hac_tie"])
