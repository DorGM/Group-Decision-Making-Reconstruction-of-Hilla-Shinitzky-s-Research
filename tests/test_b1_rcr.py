"""B1 acceptance (plan §9.3): RCR features.

"A synthetic sub-set matching A.4 (S=8/33, ps[v]=1, IG/OG means) gives
ChiSq .610, PChiSq .682, PChiSq_IG .255, PChiSq_OG .887, ConfD
.203/.175/.212, dPSv -.758, PSvD .811, PSvD_IG .406; PSvD_OG = .940 (D12)"
(TH Table A.4 p.112; plan §3.4). The sub-set is in tests/a4_case.py.
"""
import math
import random

import pytest

from a4_case import (ANSWERS, IN_GROUP, OUT_GROUP, PSVD_OG_EXPECTED, R0, R33,
                     TABLE_A4)
from cdm.config import load_config
from cdm.features import rcr
from cdm.schema import Response

ROWS, INFO = rcr.rcr_features(R33, ANSWERS, return_info=True)
F = ROWS[0]                      # the example response of Table A.4


def test_names_and_order():
    # TH Table 3.3 p.30, rows 1-14
    assert len(rcr.FEATURE_NAMES) == 14
    assert rcr.FEATURE_NAMES == (
        "ChosenMajAns", "Conf", "ConfD", "ConfD_IG", "ConfD_OG", "PSv", "dPSv",
        "PSvD", "PSvD_IG", "PSvD_OG", "ChiSq", "PChiSq", "PChiSq_IG",
        "PChiSq_OG")
    assert len(ROWS) == len(R33) == 33
    for row in ROWS:
        assert tuple(row) == rcr.FEATURE_NAMES


# What each Table A.4 comparison is worth.
#   evidence        : the printed value follows from OTHER printed values
#                     through the formula alone, whatever the rest of the
#                     sub-set looks like (test_table_a4_evidence_...)
#   input           : the value was used to build the synthetic sub-set
#   by_construction : the sub-set was built to meet it; NO evidence
#   printed_differs : printed 0.904; the other printed entries imply 0.940
#                     (plan D12 / OQ37)
A4_EVIDENCE = {
    "ChosenMajAns": "by_construction",
    "Conf": "input",
    "ConfD": "by_construction",      # the three implied means are mutually
    "ConfD_IG": "by_construction",   # consistent with S = 8/33 (test below)
    "ConfD_OG": "by_construction",
    "PSv": "input",
    "dPSv": "input",                 # gives S(v) = 8/33
    "PSvD": "by_construction",
    "PSvD_IG": "by_construction",
    "PSvD_OG": "printed_differs",
    "ChiSq": "evidence",             # from S = 8/33 and ps[v] = 1 alone
    "PChiSq": "evidence",            # from PSvD .811 alone (needs sum ps = 1)
    "PChiSq_IG": "evidence",         # from PSvD_IG .406 alone
    "PChiSq_OG": "evidence",         # from A.6 AvgPS_OG .06; refutes .904
}


def test_a4_evidence_labels_cover_all_14_features():
    assert tuple(A4_EVIDENCE) == rcr.FEATURE_NAMES
    assert [k for k, v in A4_EVIDENCE.items() if v == "evidence"] == \
        ["ChiSq", "PChiSq", "PChiSq_IG", "PChiSq_OG"]


@pytest.mark.parametrize(
    "name", [n for n in rcr.FEATURE_NAMES if n != "PSvD_OG"],
    ids=[f"{n}[{A4_EVIDENCE[n]}]" for n in rcr.FEATURE_NAMES if n != "PSvD_OG"])
def test_synthetic_a4_subset_is_consistent_with_table_a4(name):
    """The sub-set of tests/a4_case.py gives TH Table A.4 (p.112) to 3 d.p.
    Only the features labelled "evidence" test the code against the thesis."""
    assert round(F[name], 3) == TABLE_A4[name], (name, F[name])


def test_psvd_og_is_0940_not_the_printed_0904():
    # plan §7 D12, §8 OQ37: the printed 0.904 contradicts A.6 AvgPS_OG = 0.06
    # and the printed PChiSq_OG = 0.887
    assert round(F["PSvD_OG"], 3) == PSVD_OG_EXPECTED == 0.940
    assert round(F["PSvD_OG"], 3) != TABLE_A4["PSvD_OG"]


def test_plan_b1_acceptance_values():
    # the list of plan §9.3 row B1, verbatim
    want = {"ChiSq": .610, "PChiSq": .682, "PChiSq_IG": .255, "PChiSq_OG": .887,
            "ConfD": .203, "ConfD_IG": .175, "ConfD_OG": .212, "dPSv": -.758,
            "PSvD": .811, "PSvD_IG": .406, "PSvD_OG": .940}
    for name, value in want.items():
        assert round(F[name], 3) == value, (name, F[name])


def test_table_a4_values_are_not_on_a_rounding_edge():
    # every reproduced value is at least 1e-5 away from a 3-d.p. rounding
    # boundary, far above float noise. The margin cannot be made much wider:
    # the printed tables leave a narrow joint window. PChiSq = .682 needs
    # AvgPS(v) <= .18941, while A.6 dPS_OG = .182 needs an out-group mean
    # >= .05992 and dPS_IG = -.351 an in-group mean >= .5935, which give
    # AvgPS(v) >= .18927. The fixture (.59375, .06) sits at .18939.
    for name in rcr.FEATURE_NAMES:
        x = abs(F[name]) * 1000
        assert abs((x % 1) - 0.5) > 0.01 or float(F[name]).is_integer(), (name, F[name])


def _chi(x, y):
    return 0.5 * sum((a - b) ** 2 / (a + b) for a, b in zip(x, y) if a + b > 0)


def test_table_a4_evidence_on_the_printed_numbers():
    """Checks on the PRINTED values that do not depend on the synthetic
    sub-set (plan §3.4 "Table A.4 checks")."""
    # S(v) = 8/33 is the only k/n with 30 <= n <= 45 (TH p.53: sub-sets of
    # 30-45 responses) that gives the printed dPSv = -0.758 with PSv = 1
    fits = [(k, n) for n in range(30, 46) for k in range(1, n)
            if round(k / n - 1, 3) == TABLE_A4["dPSv"]]
    assert fits == [(8, 33)]
    s = 8 / 33
    # ChiSq: the response predicts 1 for its own answer and 0 elsewhere, so
    # chi2 = 1/2 [(1-S)^2/(1+S) + sum of the other supports (= 1-S)]
    chi = 0.5 * ((1 - s) ** 2 / (1 + s) + (1 - s))
    assert round(chi, 3) == TABLE_A4["ChiSq"]
    assert round(2 * chi, 3) == 1.22                 # without the 1/2 factor
    # the same formula with the reference "average prediction": the other
    # answers' average predictions sum to 1 - AvgPS(v) only if the
    # predicted shares sum to 1 (plan §2.3 [INFERRED], V15)
    def pchi(avg_v):
        return 0.5 * ((1 - avg_v) ** 2 / (1 + avg_v) + (1 - avg_v))
    assert round(pchi(1 - TABLE_A4["PSvD"]), 3) == TABLE_A4["PChiSq"]
    assert round(pchi(1 - TABLE_A4["PSvD_IG"]), 3) == TABLE_A4["PChiSq_IG"]
    # out-group: A.6 AvgPS_OG = 0.06 reproduces PChiSq_OG; the printed
    # PSvD_OG = 0.904 (out-group mean 0.096) does not
    assert round(pchi(0.06), 3) == TABLE_A4["PChiSq_OG"]
    assert round(pchi(1 - TABLE_A4["PSvD_OG"]), 3) == 0.825
    assert round(pchi(1 - PSVD_OG_EXPECTED), 3) == TABLE_A4["PChiSq_OG"]
    # the three implied confidence means are consistent with S = 8/33:
    # S * in-group + (1 - S) * out-group = overall
    avg, ig, og = (1 - TABLE_A4["ConfD"], 1 - TABLE_A4["ConfD_IG"],
                   1 - TABLE_A4["ConfD_OG"])
    assert round(s * ig + (1 - s) * og, 3) == round(avg, 3) == 0.797
    # in-group INCLUDES r: A.4's in-group mean confidence .825 equals A.6
    # AvgConf .825 (all supporters). If the in-group excluded r, the mean of
    # all 8 supporters would be (7 * .825 + 1) / 8 = .847
    assert round(ig, 3) == 0.825
    assert round((7 * ig + 1) / 8, 3) == 0.847


def test_fixture_matches_its_description():
    assert len(IN_GROUP) == 8 and len(OUT_GROUP) == 25 and R33[0] is R0
    assert R0.c == 1.0 and R0.ps[ANSWERS.index(R0.v)] == 1.0
    for r in R33:
        assert sum(r.ps) == pytest.approx(1.0)       # shares sum to 1
    assert INFO == {"majority_answers": ["a1"], "majority_tie": False,
                    "empty_out_group": 0}


# ---- formulas on a small hand-computed set -------------------------------------

A3 = ("x", "y", "z")
SMALL = [
    Response("x", 0.9, (0.6, 0.3, 0.1)),
    Response("x", 0.5, (0.8, 0.2, 0.0)),
    Response("y", 0.7, (0.5, 0.5, 0.0)),
    Response("y", 0.3, (0.2, 0.7, 0.1)),
    Response("x", 0.4, (0.4, 0.4, 0.2)),
]


def test_hand_computed_small_set():
    rows = rcr.rcr_features(SMALL, A3)
    f = rows[2]                              # votes y, c = .7, ps = (.5, .5, 0)
    assert f["ChosenMajAns"] == 0 and rows[0]["ChosenMajAns"] == 1
    assert f["Conf"] == 0.7
    assert f["ConfD"] == pytest.approx(0.7 - 2.8 / 5)
    assert f["ConfD_IG"] == pytest.approx(0.7 - 0.5)        # (0.7 + 0.3) / 2
    assert f["ConfD_OG"] == pytest.approx(0.7 - 0.6)        # (0.9 + 0.5 + 0.4) / 3
    assert f["PSv"] == 0.5
    assert f["dPSv"] == pytest.approx(2 / 5 - 0.5)
    assert f["PSvD"] == pytest.approx(0.5 - 2.1 / 5)
    assert f["PSvD_IG"] == pytest.approx(0.5 - 0.6)         # (0.5 + 0.7) / 2
    assert f["PSvD_OG"] == pytest.approx(0.5 - 0.3)         # (0.3 + 0.2 + 0.4) / 3
    assert f["ChiSq"] == pytest.approx(_chi((.5, .5, 0), (.6, .4, 0)))
    assert f["PChiSq"] == pytest.approx(_chi((.5, .5, 0), (.5, .42, .08)))
    assert f["PChiSq_IG"] == pytest.approx(_chi((.5, .5, 0), (.35, .6, .05)))
    assert f["PChiSq_OG"] == pytest.approx(_chi((.5, .5, 0), (.6, .3, .1)))


def test_chi_sq_distance():
    # TH Table 3.3 rows 11-14: 1/2 sum (x - y)^2 / (x + y)
    assert rcr.chi_sq_distance((1, 0), (0, 1)) == 1.0        # disjoint: maximum
    assert rcr.chi_sq_distance((.3, .7), (.3, .7)) == 0.0
    assert rcr.chi_sq_distance((.5, .5), (.25, .75)) == pytest.approx(
        0.5 * (.0625 / .75 + .0625 / 1.25))
    # 0/0 := 0 (plan §3.4 [INFERRED])
    assert rcr.chi_sq_distance((1, 0, 0), (1, 0, 0)) == 0.0
    assert rcr.chi_sq_distance((.5, .5, 0), (.5, .5, 0)) == 0.0
    # symmetric
    assert rcr.chi_sq_distance((.2, .8), (.6, .4)) == rcr.chi_sq_distance((.6, .4), (.2, .8))
    assert math.isnan(rcr.chi_sq_distance((.5, .5), (float("nan"), .5)))
    with pytest.raises(ValueError):
        rcr.chi_sq_distance((.5, .5), (.5,))
    with pytest.raises(ValueError):
        rcr.chi_sq_distance((-.1, 1.1), (.5, .5))


def test_zero_vote_answers_enter_the_chi_square_sums():
    # the sums run over all of A_P (TH Table 3.3: "a_i in A_P"): a prediction
    # for an answer nobody chose adds (p - 0)^2 / (p + 0) = p
    R = [Response("x", .5, (.5, .3, .2)), Response("y", .5, (.5, .5, 0))]
    f = rcr.rcr_features(R, A3)[0]
    assert f["ChiSq"] == pytest.approx(0.5 * (0 + .04 / .8 + .2))


def test_order_of_responses_does_not_matter():
    rnd = random.Random(1)
    idx = list(range(len(R33)))
    rnd.shuffle(idx)
    shuffled = rcr.rcr_features([R33[i] for i in idx], ANSWERS)
    for pos, i in enumerate(idx):
        for name in rcr.FEATURE_NAMES:
            assert shuffled[pos][name] == pytest.approx(ROWS[i][name], abs=1e-12)


def test_identities():
    # in-group / out-group means recombine into the overall mean
    s = {a: sum(1 for r in R33 if r.v == a) / 33 for a in ANSWERS}
    for r, f in zip(R33, ROWS):
        sv = s[r.v]
        assert sv * (f["Conf"] - f["ConfD_IG"]) + (1 - sv) * (f["Conf"] - f["ConfD_OG"]) \
            == pytest.approx(f["Conf"] - f["ConfD"])
        assert sv * (f["PSv"] - f["PSvD_IG"]) + (1 - sv) * (f["PSv"] - f["PSvD_OG"]) \
            == pytest.approx(f["PSv"] - f["PSvD"])
        assert f["dPSv"] == pytest.approx(sv - f["PSv"])
        for name in ("ChiSq", "PChiSq", "PChiSq_IG", "PChiSq_OG"):
            assert 0.0 <= f[name] <= 1.0


# ---- OQ28: tied majorities ----------------------------------------------------------

TIE = [Response("x", .5, (.5, .5, 0)), Response("x", .5, (.5, .5, 0)),
       Response("y", .5, (.5, .5, 0)), Response("y", .5, (.5, .5, 0)),
       Response("z", .5, (.2, .2, .6))]


def test_oq28_default_one_for_all_tied_top_answers():
    rows, info = rcr.rcr_features(TIE, A3, return_info=True)
    assert [f["ChosenMajAns"] for f in rows] == [1, 1, 1, 1, 0]
    assert info["majority_tie"] is True and info["majority_answers"] == ["x", "y"]


def test_oq28_alternative_canonical_first():
    rows, info = rcr.rcr_features(TIE, A3, majority_rule="canonical_first",
                                  return_info=True)
    assert [f["ChosenMajAns"] for f in rows] == [1, 1, 0, 0, 0]
    assert info["majority_tie"] is True and info["majority_answers"] == ["x"]
    with pytest.raises(ValueError):
        rcr.rcr_features(TIE, A3, majority_rule="random")


def test_options_from_config():
    assert rcr.options_from_config() == {"majority_rule": "one_for_all_tied_top_answers"}
    cfg = load_config()
    cfg["open_questions"]["OQ28"]["default"] = "canonical_first"
    assert rcr.options_from_config(cfg) == {"majority_rule": "canonical_first"}
    cfg["open_questions"]["OQ28"]["default"] = "something_else"
    with pytest.raises(ValueError):
        rcr.options_from_config(cfg)


# ---- undefined out-group ---------------------------------------------------------------

def test_unanimous_set_has_no_out_group():
    # no rule in the thesis or the plan: NaN + flag (decision B1-2)
    R = [Response("x", .6, (.7, .2, .1)), Response("x", .8, (.9, .1, 0))]
    rows, info = rcr.rcr_features(R, A3, return_info=True)
    assert info["empty_out_group"] == 2
    for f in rows:
        for name in ("ConfD_OG", "PSvD_OG", "PChiSq_OG"):
            assert math.isnan(f[name])
        for name in rcr.FEATURE_NAMES:
            if name not in ("ConfD_OG", "PSvD_OG", "PChiSq_OG"):
                assert math.isfinite(f[name])
    assert rows[0]["ChosenMajAns"] == 1 and rows[0]["ConfD_IG"] == pytest.approx(-.1)


def test_empty_set_raises():
    with pytest.raises(ValueError):
        rcr.rcr_features([], A3)
