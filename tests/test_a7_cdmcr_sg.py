"""A7 acceptance (plan §9.3): CDMCR sub-group features 18-27.

"The (21,23,23) case gives .031/.067/.098 and SG_B_Amax=0 (A.2). Feature 22
is not an oracle." (TH Table A.2 p.110; plan §3.3.1, §3.3.2.)

Features 22-27 of Table A.2 depend on the unpublished raw confidences of
each sub-set, so they are not compared with Table A.2; they are checked
against hand-computed values on small cases.
"""
import math

import pytest

from a2_case import ANSWERS, R90, TABLE_A2, subsets_21_23_23
from cdm import primitives as P
from cdm.features import cdmcr
from cdm.partition import partition
from cdm.schema import Response

A2 = ("a1", "a2")


def R(*rows):
    out = []
    for v, c, ps, k in rows:
        out += [Response(v, c, ps)] * k
    return out


def H(p):
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# ---- Table A.2, features 18-21 ----------------------------------------------

SUBSETS = subsets_21_23_23()
SG = cdmcr.features_18_27(R90, ANSWERS, SUBSETS)


def test_the_partition_is_3x30_with_amax_counts_21_23_23():
    assert [len(s) for s in SUBSETS] == [30, 30, 30]
    assert sorted(i for s in SUBSETS for i in s) == list(range(90))
    counts = [sum(1 for i in s if R90[i].v == "a1") for s in SUBSETS]
    assert counts == [21, 23, 23]


def test_names_and_order():
    # TH Table 3.2 p.28, rows 18-27
    assert tuple(SG) == cdmcr.FEATURES_18_27
    assert len(SG) == 10


@pytest.mark.parametrize("name", ["SG_B_Amax", "SG_VarSAmax",
                                  "SG_D_MaxVarS_MinVarS", "SG_D_MaxES_MinES"])
def test_table_a2_features_18_21_to_3dp(name):
    # TH Table A.2 p.110: SG_B_Amax 0, SG_VarSAmax 0.031,
    # SG_D_MaxVarS_MinVarS 0.067, SG_D_MaxES_MinES 0.098
    assert round(SG[name], 3) == TABLE_A2[name], (name, SG[name])


def test_table_a2_features_19_21_exact_forms():
    # plan §3.3.1
    assert SG["SG_VarSAmax"] == pytest.approx(P.var([21 / 30, 23 / 30, 23 / 30]))
    assert SG["SG_VarSAmax"] == pytest.approx(0.0314, abs=5e-5)
    assert SG["SG_D_MaxVarS_MinVarS"] == pytest.approx(2 / 30)
    assert SG["SG_D_MaxES_MinES"] == pytest.approx(H(21 / 30) - H(23 / 30))
    assert SG["SG_D_MaxES_MinES"] == pytest.approx(0.0975, abs=5e-5)


def test_21_23_23_is_the_only_3x30_split_matching_a2():
    # plan §3.3.1: "(21,23,23) is the unique 3x30 split giving these values"
    hits = []
    for a in range(0, 31):
        for b in range(a, 31):
            c = 67 - a - b
            if not b <= c <= 30:
                continue
            s = [a / 30, b / 30, c / 30]
            amax_share = [max(x, 1 - x) for x in s]       # per-subset a_max
            var_s = [abs(x - 0.5) for x in s]             # binary: SD = |S-.5|
            es = [P.entropy([x, 1 - x]) for x in s]
            if (round(P.var(amax_share), 3) == 0.031
                    and round(max(var_s) - min(var_s), 3) == 0.067
                    and round(max(es) - min(es), 3) == 0.098):
                hits.append((a, b, c))
    assert hits == [(21, 23, 23)]


def test_population_variance_switch_does_not_match_a2():
    # plan §8 OQ9
    v = cdmcr.features_18_27(R90, ANSWERS, SUBSETS, var_kind="population_variance")
    assert round(v["SG_VarSAmax"], 3) != TABLE_A2["SG_VarSAmax"]


def test_oq12_cannot_be_decided_by_a2():
    # plan §8 OQ12: "A.2 cannot discriminate" (a_max is a1 in every sub-set)
    g = cdmcr.features_18_27(R90, ANSWERS, SUBSETS, amax_mode="global")
    assert g == SG


def test_features_22_27_not_compared_with_a2():
    # Plan §3.3.2: feature 22 (0.098) is not an oracle; 23-27 depend on the
    # unpublished per-sub-set confidences. Only sanity bounds are checked.
    for name in cdmcr.FEATURES_18_27[4:]:
        assert 0 <= SG[name] <= 1 and not math.isnan(SG[name])
    # the mean over equal-size sub-sets of AvgC(R') is AvgC(R)
    f = cdmcr.features_1_17(R90, ANSWERS)
    assert f["CAmin"] <= 1 and f["CAmax"] <= 1


# ---- hand-computed case for all ten features --------------------------------

# Two sub-sets of 4 responses.
#   sub-set 0: a1 x3 (c = .5, .5, 1.0), a2 x1 (c = .25)
#   sub-set 1: a1 x1 (c = 1.0),         a2 x3 (c = .25, .75, .5)
HAND = (R(("a1", 0.5, (0.5, 0.5), 2), ("a1", 1.0, (0.5, 0.5), 1),
          ("a2", 0.25, (0.5, 0.5), 1))
        + R(("a1", 1.0, (0.5, 0.5), 1), ("a2", 0.25, (0.5, 0.5), 1),
            ("a2", 0.75, (0.5, 0.5), 1), ("a2", 0.5, (0.5, 0.5), 1)))
HAND_SUBS = [[0, 1, 2, 3], [4, 5, 6, 7]]


def _sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def test_hand_case_per_subset():
    f, info = cdmcr.features_18_27(HAND, A2, HAND_SUBS, return_info=True)
    # 18: a_max is a1 in sub-set 0 and a2 in sub-set 1
    assert f["SG_B_Amax"] == 1
    # 19: S(a_max', R') = .75 and .75 -> SD 0   (per-subset a_max, OQ12)
    assert f["SG_VarSAmax"] == pytest.approx(0.0)
    # 20, 21: both sub-sets have shares (.75, .25) -> 0
    assert f["SG_D_MaxVarS_MinVarS"] == pytest.approx(0.0)
    assert f["SG_D_MaxES_MinES"] == pytest.approx(0.0)
    # 22: Average(C(R')) = 2.25/4 = .5625 and 2.5/4 = .625
    assert f["SG_D_MaxAvgC_MinAvgC"] == pytest.approx(0.0625)
    # 23: population SD of C(R')
    sd0 = _sd([0.5, 0.5, 1.0, 0.25])
    sd1 = _sd([1.0, 0.25, 0.75, 0.5])
    assert f["SG_D_MaxVarC_MinVarC"] == pytest.approx(abs(sd0 - sd1))
    # 24, 26: a_min' = a2 (c=.25) in sub-set 0; a1 (c=1.0) in sub-set 1
    assert f["SG_VarCAmin"] == pytest.approx(_sd([0.25, 1.0]))
    assert f["SG_AvgCAmin"] == pytest.approx(0.625)
    # 25, 27: a_max' = a1 (mean 2/3) in sub-set 0; a2 (mean .5) in sub-set 1
    assert f["SG_VarCAmax"] == pytest.approx(_sd([2 / 3, 0.5]))
    assert f["SG_AvgCAmax"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert info == {"imputed_CAmin": 0, "imputed_CAmax": 0, "subset_amax_ties": 0,
                    "subset_amin_ties": 0, "n_subsets": 2, "single_subset": False}


def test_hand_case_global_amax():
    # plan §8 OQ12 alternative: global a_max/a_min. Votes are 4:4, so the
    # global a_max = a_min = a1 (canonical first, OQ7).
    f = cdmcr.features_18_27(HAND, A2, HAND_SUBS, amax_mode="global")
    assert f["SG_B_Amax"] == 1                       # still the sub-sets' own a_max
    assert f["SG_VarSAmax"] == pytest.approx(_sd([0.75, 0.25]))
    assert f["SG_VarCAmax"] == pytest.approx(_sd([2 / 3, 1.0]))
    assert f["SG_AvgCAmax"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert f["SG_VarCAmin"] == f["SG_VarCAmax"]
    # features that do not use a_max are unchanged
    g = cdmcr.features_18_27(HAND, A2, HAND_SUBS)
    for name in ("SG_D_MaxVarS_MinVarS", "SG_D_MaxES_MinES",
                 "SG_D_MaxAvgC_MinAvgC", "SG_D_MaxVarC_MinVarC"):
        assert f[name] == g[name]


def test_feature_22_uses_confidence_not_support():
    # plan §3.3 row 22 / OQ36: TH Table 3.2 prints Average(S(R')).
    # Average(S(R')) is 1/|A| in every sub-set, so that reading would
    # always give 0.
    f = cdmcr.features_18_27(HAND, A2, HAND_SUBS)
    assert f["SG_D_MaxAvgC_MinAvgC"] > 0


# ---- empty supporter sets in a sub-set (plan §8 OQ7) --------------------------

# sub-set 0 is unanimous for a1 (c = .5, 1.0); sub-set 1: a1 (c=.25), a2 (c=.75)
UNAN = R(("a1", 0.5, (0.5, 0.5), 1), ("a1", 1.0, (0.5, 0.5), 1),
         ("a1", 0.25, (0.5, 0.5), 1), ("a2", 0.75, (0.5, 0.5), 1))
UNAN_SUBS = [[0, 1], [2, 3]]


def test_empty_amin_is_imputed_with_subset_avgc():
    f, info = cdmcr.features_18_27(UNAN, A2, UNAN_SUBS, return_info=True)
    # sub-set 0: a_min' = a2 has no supporter -> AvgC(R') = .75 (OQ7 default)
    # sub-set 1: 1:1 tie -> a_max' = a_min' = a1 (canonical first) -> .25
    assert info["imputed_CAmin"] == 1
    assert info["subset_amax_ties"] == 1
    assert f["SG_AvgCAmin"] == pytest.approx((0.75 + 0.25) / 2)
    assert f["SG_VarCAmin"] == pytest.approx(_sd([0.75, 0.25]))
    assert f["SG_AvgCAmax"] == pytest.approx((0.75 + 0.25) / 2)
    assert f["SG_B_Amax"] == 0
    # 20/21: unanimous sub-set has SD .5 and entropy 0; tied one SD 0, entropy 1
    assert f["SG_D_MaxVarS_MinVarS"] == pytest.approx(0.5)
    assert f["SG_D_MaxES_MinES"] == pytest.approx(1.0)


def test_empty_amin_alternative_zero():
    f = cdmcr.features_18_27(UNAN, A2, UNAN_SUBS, empty_supporters="zero")
    assert f["SG_AvgCAmin"] == pytest.approx((0.0 + 0.25) / 2)


def test_bad_arguments():
    with pytest.raises(ValueError):
        cdmcr.features_18_27(UNAN, A2, [[0, 1, 2, 3]])           # one sub-set
    with pytest.raises(ValueError):
        cdmcr.features_18_27(UNAN, A2, [[0, 1, 2, 3], []])       # empty sub-set
    with pytest.raises(ValueError):
        cdmcr.features_18_27(UNAN, A2, UNAN_SUBS, amax_mode="x")
    with pytest.raises(ValueError):
        cdmcr.features_18_27(UNAN, A2, UNAN_SUBS, empty_supporters="x")


# ---- the 27 features together -------------------------------------------------

def test_all_27_on_the_a2_case():
    f = cdmcr.cdmcr_features(R90, ANSWERS, subsets=SUBSETS)
    assert tuple(f) == cdmcr.FEATURE_NAMES and len(f) == 27
    for name in cdmcr.FEATURE_NAMES[:21]:                 # features 1-21
        assert round(f[name], 3) == TABLE_A2[name], (name, f[name])


def test_seeded_partition_is_used_and_reproducible():
    f0 = cdmcr.cdmcr_features(R90, ANSWERS, seed=0)
    assert f0 == cdmcr.cdmcr_features(R90, ANSWERS, seed=0)
    assert f0 == cdmcr.cdmcr_features(R90, ANSWERS, subsets=partition(90, 0))
    # features 1-17 do not depend on the partition
    f1 = cdmcr.cdmcr_features(R90, ANSWERS, seed=1)
    for name in cdmcr.FEATURES_1_17:
        assert f0[name] == f1[name]
    assert any(f0[n] != f1[n] for n in cdmcr.FEATURES_18_27)
    with pytest.raises(ValueError):
        cdmcr.cdmcr_features(R90, ANSWERS)                # no seed, no sub-sets


def test_small_n_uses_two_halves():
    # TH p.27 (N < 30) and plan §8 OQ6 (30 <= N < 60): two halves
    rs = R(("a1", 0.5, (0.5, 0.5), 7), ("a2", 0.75, (0.5, 0.5), 4))
    f = cdmcr.cdmcr_features(rs, A2, seed=0)
    assert f["N"] == 11 and not any(math.isnan(v) for v in f.values())


def test_options_from_config_are_the_plan_defaults():
    opts = cdmcr.options_from_config()
    assert opts == {
        "partition_kwargs": {"subset_size": 30,
                             "scheme": "a_disjoint_floor_n_over_30",
                             "halves_if_k_below_2": True},
        "var_kind": "population_sd",          # OQ9
        "strict": True,                       # OQ13
        "amax_mode": "per_subset",            # OQ12
        "empty_supporters": "nan_then_impute_subset_avgc",   # OQ7
        "amax_policy": "canonical_first",     # OQ7 (a_max / a_min)
        "hac_policy": "canonical_first",      # OQ8 through OQ14 (feature 12)
        "tol": 1e-9,                          # tie_abs_tol (AM3/AM9)
    }
    f = cdmcr.cdmcr_features(R90, ANSWERS, seed=0, **opts)
    assert f == cdmcr.cdmcr_features(R90, ANSWERS, seed=0)


# ---- wiring, single sub-set, info ---------------------

def _config(**changes):
    """A copy of the default config with some OQ defaults replaced."""
    import copy
    from cdm.config import load_config
    cfg = copy.deepcopy(load_config())
    for oq, value in changes.items():
        if oq == "tie_abs_tol":
            cfg["tie_abs_tol"] = value
        elif isinstance(value, dict):
            cfg["open_questions"][oq]["default"].update(value)
        else:
            cfg["open_questions"][oq]["default"] = value
    return cfg


@pytest.mark.parametrize("oq,value", [
    ("OQ10", 10), ("OQ10", "e"),
    ("OQ11", "manhattan"),
    ("OQ14", "mla_amax_among_highest_avg_confidence"),
    ("OQ36", "use_support"),
    ("OQ7", {"support_vector_includes_zero_vote_answers": False}),
    ("OQ7", {"ties": "canonical_second"}),
    ("OQ8", "random"),
])
def test_r3_unimplemented_switch_value_raises(oq, value):
    # a changed switch must never be silently ignored
    with pytest.raises(NotImplementedError):
        cdmcr.options_from_config(_config(**{oq: value}))


@pytest.mark.parametrize("oq,value", [
    ("OQ13", "nonstrict"), ("OQ13", True),
    ("OQ9", "sample_sd"), ("OQ12", "both"),
    ("OQ7", {"empty_supporters_in_subset": "drop"}),
    ("OQ8", "coin"),
    ("tie_abs_tol", "1e-9"), ("tie_abs_tol", -1.0),
])
def test_r3_unknown_switch_value_raises(oq, value):
    with pytest.raises(ValueError):
        cdmcr.options_from_config(_config(**{oq: value}))


def test_r3_oq13_non_strict_is_wired():
    assert cdmcr.options_from_config(_config(OQ13="non_strict"))["strict"] is False
    assert cdmcr.options_from_config(_config(OQ13="strict"))["strict"] is True


def test_r3_oq8_reaches_feature_12_but_not_amax():
    # HAC tie: 3 x .5 against 2 x .5. OQ8 = canonical_second -> f_HAC = a2,
    # while a_max stays a1 (OQ7 is a separate switch).
    rs = R(("a1", 0.5, (0.5, 0.5), 3), ("a2", 0.5, (0.5, 0.5), 2)) * 2
    subs = [list(range(0, 5)), list(range(5, 10))]
    first = cdmcr.options_from_config()
    second = cdmcr.options_from_config(_config(OQ8="canonical_second"))
    assert second["hac_policy"] == "canonical_second"
    assert second["amax_policy"] == "canonical_first"
    assert first.pop("partition_kwargs") == second.pop("partition_kwargs")
    f1 = cdmcr.cdmcr_features(rs, A2, subsets=subs, **first)
    f2 = cdmcr.cdmcr_features(rs, A2, subsets=subs, **second)
    assert f1["B_Amax_MaxCa"] == 1 and f2["B_Amax_MaxCa"] == 0
    assert {k: v for k, v in f1.items() if k != "B_Amax_MaxCa"} == \
           {k: v for k, v in f2.items() if k != "B_Amax_MaxCa"}


def test_r3_tie_abs_tol_is_wired():
    assert cdmcr.options_from_config(_config(tie_abs_tol=0))["tol"] == 0.0
    assert cdmcr.options_from_config(_config(tie_abs_tol=0.0))["tol"] == 0.0
    assert cdmcr.options_from_config()["tol"] == P.DEFAULT_TIE_ABS_TOL


def test_r4_single_subset_is_degenerate_only_when_halves_are_switched_off():
    rs = R(("a1", 0.5, (0.5, 0.5), 25), ("a2", 0.75, (0.5, 0.5), 20))    # N = 45
    opts = cdmcr.options_from_config(_config(OQ6={"halves_if_k_below_2": False}))
    f, info = cdmcr.cdmcr_features(rs, A2, seed=0, return_info=True, **opts)
    assert info["single_subset"] is True and info["n_subsets"] == 1
    assert f["SG_B_Amax"] == 0
    for name in ("SG_VarSAmax", "SG_D_MaxVarS_MinVarS", "SG_D_MaxES_MinES",
                 "SG_D_MaxAvgC_MinAvgC", "SG_D_MaxVarC_MinVarC",
                 "SG_VarCAmin", "SG_VarCAmax"):
        assert f[name] == 0
    assert f["SG_AvgCAmin"] == pytest.approx(f["CAmin"])
    assert f["SG_AvgCAmax"] == pytest.approx(f["CAmax"])
    # N < 30 keeps two halves with the switch off (TH p.27)
    small = rs[:10] + rs[25:36]                                           # N = 21
    g, ginfo = cdmcr.cdmcr_features(small, A2, seed=0, return_info=True, **opts)
    assert ginfo["n_subsets"] == 2 and ginfo["single_subset"] is False
    # explicit single sub-set, or the default switch: still an error
    with pytest.raises(ValueError):
        cdmcr.cdmcr_features(rs, A2, subsets=[list(range(45))])
    one = cdmcr.features_18_27(rs, A2, [list(range(45))], allow_single_subset=True)
    assert one["SG_B_Amax"] == 0


def test_r7_cdmcr_features_returns_tie_info():
    # full-set 2:2 vote tie and an HAC tie; sub-set 1 is tied 1:1
    f, info = cdmcr.cdmcr_features(UNAN, A2, subsets=UNAN_SUBS, return_info=True)
    assert tuple(f) == cdmcr.FEATURE_NAMES
    assert f == cdmcr.cdmcr_features(UNAN, A2, subsets=UNAN_SUBS)
    full, finfo = cdmcr.features_1_17(UNAN, A2, return_info=True)
    for key in ("amax_tied", "amin_tied", "hac_tied",
                "amax_tie", "amin_tie", "hac_tie"):
        assert info[key] == finfo[key]
    assert info["imputed_CAmin"] == 1 and info["subset_amax_ties"] == 1
    assert info["subset_amin_ties"] == 1
    assert info["subsets"] == UNAN_SUBS
    # a full-set vote tie is flagged (OQ7 "flag", "log the frequency")
    tie = R(("a1", 0.25, (0.5, 0.5), 2), ("a2", 0.75, (0.5, 0.5), 2))
    _, ti = cdmcr.cdmcr_features(tie, A2, subsets=[[0, 2], [1, 3]], return_info=True)
    assert ti["amax_tie"] is True and ti["amax_tied"] == ["a1", "a2"]
    assert ti["amin_tie"] is True and ti["amin_tied"] == ["a1", "a2"]
    assert ti["hac_tie"] is False and ti["hac_tied"] == ["a2"]
    _, ni = cdmcr.cdmcr_features(R90, ANSWERS, subsets=SUBSETS, return_info=True)
    assert ni["amax_tie"] is False and ni["amax_tied"] == ["a1"]
    assert ni["amin_tied"] == ["a2"] and ni["hac_tie"] is False
