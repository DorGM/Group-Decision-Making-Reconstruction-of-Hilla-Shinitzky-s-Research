"""Printed thesis values of the World-B plan items T23, T25-T30 (plan §1.1).

Typed from the thesis text / rendered pages; nothing here is computed from
data and nothing is used to tune anything. They are kept in this module
beside the World-B code (decisions R1, SH10, B6-9); T24 is in cdm/eval/worldB_rules.py.

A test (tests/test_b9_stats.py) recomputes Table 5.11 from the counts of
plan §5.8 with cdm.eval.stats.
"""
from cdm.approaches import rp
from cdm.features import acr, rcr

N_SUBSETS = 77

#: T23. TH Fig 5.3 p.76 ("64% and 55%"); MCF §5. Counts: plan T23 [INFERRED].
T23_RATE = {"RP": 0.636, "AP": 0.545}
T23_COUNT = {"RP": 49, "AP": 42}
#: T25. TH §6.1; MCF p.105: PRE against MR, computed from rounded rates.
T25_PRE_VS_MR = {"RP": 0.348, "AP": 0.185}

#: T26. TH Table 4.2 p.57 = MCF Table 4: strategy -> (success %, tie %).
TABLE_4_2 = {
    "maxp": (58, 0), "avgp": (61, 0), "wm": (47, 0),
    "maj+avgp": (62, 16), "maj+maxp": (60, 17), "maj+wm": (57, 16),
    "prop+avgp": (64, 18), "prop+maxp": (60, 16), "prop+wm": (57, 17),
}
assert tuple(TABLE_4_2) == rp.STRATEGY_NAMES

#: T27. TH Table A.7 p.114: classifier -> strategy -> (success %, tie %).
#: The rows of "Vote soft" repeat Table 4.2. The tie-breaker column prints
#: "rnd" for maxp / avgp / wm.
TABLE_A_7 = {
    "soft_vote": dict(TABLE_4_2),
    "bag_knn": {
        "maxp": (51, 0), "avgp": (56, 0), "wm": (48, 0),
        "maj+avgp": (49, 8), "maj+maxp": (49, 16), "maj+wm": (48, 12),
        "prop+avgp": (53, 10), "prop+maxp": (53, 6), "prop+wm": (52, 5),
    },
    "svm": {
        "maxp": (43, 0), "avgp": (43, 0), "wm": (45, 0),
        "maj+avgp": (43, 91), "maj+maxp": (43, 91), "maj+wm": (45, 91),
        "prop+avgp": (43, 91), "prop+maxp": (43, 91), "prop+wm": (45, 91),
    },
}
#: Row order of Table A.7 / A.8.
A7_CLASSIFIER_ORDER = ("soft_vote", "bag_knn", "svm")
A8_CLASSIFIER_ORDER = ("bag_knn", "svm", "soft_vote")
#: T28. TH Table A.8 p.114: success % of AP.
TABLE_A_8 = {"bag_knn": 44, "svm": 43, "soft_vote": 55}

#: T29. TH Table 5.11 p.76 = MCF Table 5 (printed p-values), row order SP,
#: MR, WC, HAC: rule -> approach -> (McNemar, proportion test).
TABLE_5_11_ORDER = ("SP", "MR", "WC", "HAC")
TABLE_5_11 = {
    "SP": {"RP": ("0.001", "0.005"), "AP": ("0.027", "0.073")},
    "MR": {"RP": ("3.0E-04", "0.008"), "AP": ("0.027", "0.099")},
    "WC": {"RP": ("3.0E-04", "0.008"), "AP": ("0.027", "0.099")},
    "HAC": {"RP": ("4.9E-05", "6.4E-06"), "AP": ("0.003", "0.001")},
}
#: Significance level of the thesis's tests (TH p.65: "a P-value lower than
#: 0.05").
ALPHA = 0.05

#: T30 (World-B part). Rank orders read from the rendered figures:
#: TH Fig 5.8 p.82 (= MCF Fig 2) and TH Fig 5.9 p.83 (= MCF Fig 3), top to
#: bottom. All features are shown (14 and 10).
SHAP_RANKING = {
    "RP": ("dPSv", "PSvD_OG", "PSvD", "ConfD_IG", "PSv", "ChosenMajAns", "ConfD_OG",
           "ConfD", "PSvD_IG", "ChiSq", "PChiSq_IG", "Conf", "PChiSq_OG", "PChiSq"),
    "AP": ("dPS_OG", "AvgConf", "AvgPS_OG", "AvgPS_IG", "dPS", "AvgChiSq", "dPS_IG",
           "AvgPS", "Support", "IsMajority"),
}
SHAP_FIGURE = {"RP": "TH Fig 5.8 p.82 (MCF Fig 2)", "AP": "TH Fig 5.9 p.83 (MCF Fig 3)"}
assert sorted(SHAP_RANKING["RP"]) == sorted(rcr.FEATURE_NAMES)
assert sorted(SHAP_RANKING["AP"]) == sorted(acr.FEATURE_NAMES)
