"""B2 acceptance (plan §9.3): ACR features.

"The same sub-set gives A.6 (.242, .189, .594, .06, .053, -.352, .182,
.825)" (TH Table A.6 p.113; plan §3.5). The sub-set is in tests/a4_case.py.

dPS_IG: Table A.6 prints -0.351; the plan's -0.352 is 8/33 - 0.594 computed
from the rounded AvgPS_IG. The fixture's in-group mean (0.59375) meets the
PRINTED value; both readings are asserted below (decision B2-2).
"""
import math

import pytest

from a4_case import ANSWER, ANSWERS, R33, TABLE_A4, TABLE_A6
from cdm import aggregators as AG
from cdm.features import acr, rcr
from cdm.schema import Response

ROWS, INFO = acr.acr_features(R33, ANSWERS, return_info=True)
BY_ANSWER = dict(ROWS)
F = BY_ANSWER[ANSWER]                 # the example answer of Table A.6


def test_names_and_order():
    # TH Table 3.4 p.31, rows 1-10
    assert acr.FEATURE_NAMES == (
        "IsMajority", "Support", "AvgPS", "AvgPS_IG", "AvgPS_OG", "dPS",
        "dPS_IG", "dPS_OG", "AvgChiSq", "AvgConf")
    for _, f in ROWS:
        assert tuple(f) == acr.FEATURE_NAMES


def test_one_instance_per_answer_with_at_least_one_vote():
    # TH p.30: an answer "which has at least one vote in the given set"
    assert [a for a, _ in ROWS] == ["a1", "a2", "a3", "a4"]      # a5, a6: no vote
    assert acr.voted_answers(R33, ANSWERS) == ["a1", "a2", "a3", "a4"]


# What each Table A.6 comparison is worth (labels as in test_b1_rcr.py, plus
#   sign_only : a difference of two fixture inputs; the comparison shows the
#               sign convention (S - AvgPS) and nothing else).
A6_EVIDENCE = {
    "IsMajority": "by_construction",
    "Support": "input",              # 8/33
    "AvgPS": "evidence",             # = S * AvgPS_IG + (1 - S) * AvgPS_OG
    "AvgPS_IG": "by_construction",   # = 1 - A.4 PSvD_IG (same answer, same sub-set)
    "AvgPS_OG": "by_construction",
    "dPS": "sign_only",              # = Support - AvgPS
    "dPS_IG": "sign_only",           # = Support - AvgPS_IG
    "dPS_OG": "sign_only",           # = Support - AvgPS_OG
    "AvgChiSq": "by_construction",   # the fixture was tuned to 0.28
    "AvgConf": "by_construction",    # = 1 - A.4 ConfD_IG
}


def test_a6_evidence_labels_cover_all_10_features():
    assert tuple(A6_EVIDENCE) == acr.FEATURE_NAMES


@pytest.mark.parametrize(
    "name", acr.FEATURE_NAMES,
    ids=[f"{n}[{A6_EVIDENCE[n]}]" for n in acr.FEATURE_NAMES])
def test_synthetic_a4_subset_is_consistent_with_table_a6(name):
    """The sub-set of tests/a4_case.py gives TH Table A.6 (p.113) as
    printed, to 3 d.p. (dPS_IG: the printed -0.351)."""
    assert round(F[name], 3) == TABLE_A6[name], (name, F[name])


def test_plan_b2_acceptance_values():
    # the list of plan §9.3 row B2: Support, AvgPS, AvgPS_IG, AvgPS_OG, dPS,
    # dPS_IG, dPS_OG, AvgConf
    want = {"Support": .242, "AvgPS": .189, "AvgPS_IG": .594, "AvgPS_OG": .06,
            "dPS": .053, "dPS_OG": .182, "AvgConf": .825}
    for name, value in want.items():
        assert round(F[name], 3) == value, (name, F[name])
    # dPS_IG: plan -.352 (from the rounded inputs), thesis prints -.351.
    # The unrounded fixture value rounds to the printed one and is within
    # one unit of the last digit of the plan's value.
    assert round(F["dPS_IG"], 3) == -0.351
    assert abs(F["dPS_IG"] - (-0.352)) < 0.001
    assert round(round(8 / 33, 3) - TABLE_A6["AvgPS_IG"], 3) == -0.352


def test_table_a6_values_are_not_on_a_rounding_edge():
    # at least 1e-5 from a 3-d.p. rounding boundary (see test_b1_rcr.py)
    for name in acr.FEATURE_NAMES:
        x = abs(F[name]) * 1000
        assert abs((x % 1) - 0.5) > 0.01 or float(F[name]).is_integer(), (name, F[name])


def test_table_a6_evidence_on_the_printed_numbers():
    """Checks on the PRINTED values, independent of the synthetic sub-set
    (plan §3.5 "Table A.6 checks")."""
    s = 8 / 33
    t = TABLE_A6
    assert round(s, 3) == t["Support"]
    # AvgPS is the support-weighted mix of the in-group and out-group means
    assert round(s * t["AvgPS_IG"] + (1 - s) * t["AvgPS_OG"], 3) == t["AvgPS"]
    assert round(s - t["AvgPS"], 3) == t["dPS"]
    assert round(s - t["AvgPS_OG"], 3) == t["dPS_OG"]
    # dPS_IG = -0.351 needs an in-group mean in [0.5935, 0.5940): possible,
    # and it still rounds to the printed AvgPS_IG 0.594
    assert round(s - 0.59375, 3) == t["dPS_IG"] and round(0.59375, 3) == t["AvgPS_IG"]
    # A.4 and A.6 describe the same answer in the same sub-set
    assert round(1 - TABLE_A4["PSvD_IG"], 3) == t["AvgPS_IG"]
    assert round(1 - TABLE_A4["PSvD"], 3) == t["AvgPS"]
    assert round(1 - TABLE_A4["ConfD_IG"], 3) == t["AvgConf"]
    assert round(1 + TABLE_A4["dPSv"], 3) == t["Support"]


def test_acr_agrees_with_rcr_and_the_rule_scores():
    rrows = rcr.rcr_features(R33, ANSWERS)
    mr = dict(zip(ANSWERS, AG.mr_scores(R33, ANSWERS)))
    hac = dict(zip(ANSWERS, AG.hac_scores(R33, ANSWERS)))
    sp = dict(zip(ANSWERS, AG.sp_scores(R33, ANSWERS)))
    for a, f in ROWS:
        mine = [x for r, x in zip(R33, rrows) if r.v == a]
        # 9: mean ChiSq of the supporters (RCR feature 11)
        assert f["AvgChiSq"] == pytest.approx(sum(x["ChiSq"] for x in mine) / len(mine))
        # 2, 6, 10 are the MR, SP and HAC criteria of the answer (TH p.31)
        assert f["Support"] == pytest.approx(mr[a])
        assert f["dPS"] == pytest.approx(sp[a])
        assert f["AvgConf"] == pytest.approx(hac[a])
        # 3 = S * 4 + (1 - S) * 5
        assert f["AvgPS"] == pytest.approx(
            f["Support"] * f["AvgPS_IG"] + (1 - f["Support"]) * f["AvgPS_OG"])
        # in-group / out-group means are those RCR uses for the supporters
        assert f["AvgPS_IG"] == pytest.approx(mine[0]["PSv"] - mine[0]["PSvD_IG"])
        assert f["AvgPS_OG"] == pytest.approx(mine[0]["PSv"] - mine[0]["PSvD_OG"])
        assert f["IsMajority"] == mine[0]["ChosenMajAns"]
    assert sum(f["Support"] for _, f in ROWS) == pytest.approx(1.0)
    assert INFO == {"majority_answers": ["a1"], "majority_tie": False,
                    "empty_out_group": False}


def test_hand_computed_small_set():
    A3 = ("x", "y", "z")
    R = [Response("x", 0.9, (0.6, 0.3, 0.1)),
         Response("x", 0.5, (0.8, 0.2, 0.0)),
         Response("y", 0.7, (0.5, 0.5, 0.0)),
         Response("y", 0.3, (0.2, 0.7, 0.1)),
         Response("x", 0.4, (0.4, 0.4, 0.2))]
    rows = dict(acr.acr_features(R, A3))
    assert list(rows) == ["x", "y"]
    f = rows["y"]
    assert f["IsMajority"] == 0 and rows["x"]["IsMajority"] == 1
    assert f["Support"] == pytest.approx(0.4)
    assert f["AvgPS"] == pytest.approx(2.1 / 5)
    assert f["AvgPS_IG"] == pytest.approx(0.6)
    assert f["AvgPS_OG"] == pytest.approx(0.3)
    assert f["dPS"] == pytest.approx(0.4 - 0.42)
    assert f["dPS_IG"] == pytest.approx(-0.2)
    assert f["dPS_OG"] == pytest.approx(0.1)
    chi = lambda ps: 0.5 * sum((p - s) ** 2 / (p + s)
                               for p, s in zip(ps, (.6, .4, 0)) if p + s > 0)
    assert f["AvgChiSq"] == pytest.approx((chi((.5, .5, 0)) + chi((.2, .7, .1))) / 2)
    assert f["AvgConf"] == pytest.approx(0.5)


def test_oq28_tied_majorities():
    A3 = ("x", "y", "z")
    R = [Response("x", .5, (.5, .5, 0)), Response("y", .5, (.5, .5, 0)),
         Response("z", .5, (.2, .2, .6)), Response("y", .5, (.5, .5, 0)),
         Response("x", .5, (.5, .5, 0))]
    rows, info = acr.acr_features(R, A3, return_info=True)
    assert [f["IsMajority"] for _, f in rows] == [1, 1, 0]
    assert info["majority_tie"] is True
    rows = acr.acr_features(R, A3, majority_rule="canonical_first")
    assert [f["IsMajority"] for _, f in rows] == [1, 0, 0]
    assert acr.options_from_config() == {"majority_rule": "one_for_all_tied_top_answers"}


def test_unanimous_set_has_no_out_group():
    # no rule in the thesis or the plan: NaN + flag (decision B1-2)
    R = [Response("x", .6, (.7, .3)), Response("x", .8, (.9, .1))]
    rows, info = acr.acr_features(R, ("x", "y"), return_info=True)
    assert [a for a, _ in rows] == ["x"] and info["empty_out_group"] is True
    f = rows[0][1]
    assert math.isnan(f["AvgPS_OG"]) and math.isnan(f["dPS_OG"])
    assert f["Support"] == 1.0 and f["AvgPS_IG"] == pytest.approx(0.8)
    assert f["AvgPS"] == pytest.approx(0.8) and f["AvgConf"] == pytest.approx(0.7)
