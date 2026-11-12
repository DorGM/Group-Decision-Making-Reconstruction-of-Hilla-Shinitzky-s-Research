"""Synthetic sub-set of 33 responses matching the free parameters of TH
Table A.4 (RCR example, p.112) and Table A.6 (ACR example, p.113). Shared
by test_b1_rcr.py and test_b2_acr.py.

The raw responses behind the two tables are not published. This sub-set is
built so that the printed values are met; it shows that the formulas of
plan §3.4 / §3.5 are consistent with the tables, as plan §9.3 B1/B2 asks.
It is NOT the original data. Each feature is labelled in the test files
with what the comparison is worth (evidence / input / by construction).

What the printed tables imply (plan §3.4 "Table A.4 checks"):
  Conf = 1, PSv = 1                 -> the response is fully confident and
                                       predicts 100% for its own answer, so
                                       its ps is 0 for every other answer
                                       (shares sum to 1, plan §2.3 [INFERRED])
  dPSv = -0.758 (= Support 0.242)   -> S(v) = 8/33, the only k/n with
                                       30 <= n <= 45 that rounds to 0.242
  ConfD .203 / .175 / .212          -> Average C: all .797, in-group .825
                                       (= A.6 AvgConf), out-group .788
  PSvD .811, PSvD_IG .406           -> Average PS(v): all .189, in-group .594
  A.6 AvgPS_OG = 0.06               -> out-group Average PS(v) = .060, so
                                       PSvD_OG = 0.940 (printed 0.904: plan
                                       D12 / OQ37)

Free choices of this fixture (no evidence; [INVENTED] unless marked):
  * 6 answers [INFERRED, weak: sub-sets of 33 arise only from Knapsack 2
    (99 = 3 x 33) and Pills (66 = 2 x 33) in TH Table 4.1, both |A_P| = 6];
  * the votes of the other 25 responses: 15 / 6 / 4 on a1 / a3 / a4, none
    on a5 and a6 (a1 is the majority answer: ChosenMajAns = IsMajority = 0);
  * the individual confidences and predictions behind the group means;
  * the predictions of the 7 other supporters on the other answers, chosen
    so that A.6 AvgChiSq = 0.28 is met (by construction).

The in-group mean of ps[v] is 0.59375, not the rounded 0.594: with it the
fixture gives dPS_IG = -0.351 as PRINTED in Table A.6 (the plan's -0.352 is
8/33 - 0.594 computed from the rounded value).
"""
from cdm.schema import Response

ANSWERS = ("a1", "a2", "a3", "a4", "a5", "a6")
#: The answer of the example (TH Table A.6) and the vote of the example
#: response (TH Table A.4).
ANSWER = "a2"

#: The example response of TH Table A.4: Conf = 1, PSv = 1.
R0 = Response("a2", 1.0, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0))


def _supporter(x):
    # one of the 7 other supporters of a2: confidence 0.8, ps[a2] = x
    return Response("a2", 0.8, (1 - x - 0.1 - 0.154, x, 0.05, 0.05, 0.154, 0.0))


#: In-group of the example: 8 supporters of a2.
#:   confidences 1 + 7 * 0.8 = 6.6                 -> mean 0.825
#:   ps[a2]      1 + 4*.5 + .55 + 2*.6 = 4.75      -> mean 0.59375
IN_GROUP = [R0] + [_supporter(x) for x in (0.5, 0.5, 0.5, 0.5, 0.55, 0.6, 0.6)]

#: Out-group: 25 responses, confidence 0.788 each, ps[a2] = 0.06 each.
_OUT_PS = (0.50, 0.06, 0.24, 0.20, 0.0, 0.0)
OUT_GROUP = ([Response("a1", 0.788, _OUT_PS)] * 15
             + [Response("a3", 0.788, _OUT_PS)] * 6
             + [Response("a4", 0.788, _OUT_PS)] * 4)

R33 = IN_GROUP + OUT_GROUP          # index 0 is the example response

# TH Table A.4 p.112, as printed.
TABLE_A4 = {
    "ChosenMajAns": 0,
    "Conf": 1,
    "ConfD": 0.203,
    "ConfD_IG": 0.175,
    "ConfD_OG": 0.212,
    "PSv": 1,
    "dPSv": -0.758,
    "PSvD": 0.811,
    "PSvD_IG": 0.406,
    "PSvD_OG": 0.904,      # as printed; implied by the other entries: 0.940 (plan D12, OQ37)
    "ChiSq": 0.61,
    "PChiSq": 0.682,
    "PChiSq_IG": 0.255,
    "PChiSq_OG": 0.887,
}
#: Plan §7 D12 / §8 OQ37: the value that A.6 and PChiSq_OG imply.
PSVD_OG_EXPECTED = 0.940

# TH Table A.6 p.113, as printed.
TABLE_A6 = {
    "IsMajority": 0,
    "Support": 0.242,
    "AvgPS": 0.189,
    "AvgPS_IG": 0.594,
    "AvgPS_OG": 0.06,
    "dPS": 0.053,
    "dPS_IG": -0.351,
    "dPS_OG": 0.182,
    "AvgChiSq": 0.28,
    "AvgConf": 0.825,
}
