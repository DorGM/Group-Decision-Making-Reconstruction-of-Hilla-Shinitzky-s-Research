"""Synthetic N=90 binary case matching the free parameters of TH Table A.2
(p.110). Shared by test_a6_cdmcr.py and test_a7_cdmcr_sg.py.

The raw responses behind Table A.2 are not published. This case is built
so that the published values are met; it shows that the formulas of plan
§3.3 are consistent with Table A.2, as plan §9.3 A6/A7 asks. It is not the
original data.

Construction (plan §3.3.2):
  67 responses vote a1 (= a_max), 23 vote a2      -> S(a_max) = 67/90
  ps is binary: ps[r.v] = x, ps[other] = 1 - x
  rows: (vote, confidence, x = ps[r.v], count)
"""
from cdm.schema import Response

ANSWERS = ("a1", "a2")

ROWS_A1 = [            # 67 supporters of a_max
    ("a1", 0.896, 0.800, 45),   # c > AvgC, x > AvgPSV
    ("a1", 0.444, 0.800, 7),    # c < AvgC, x > AvgPSV  (feature 16)
    ("a1", 0.896, 0.600, 6),    # c > AvgC, x < AvgPSV  (feature 17)
    ("a1", 0.444, 0.595, 9),    # c < AvgC, x < AvgPSV
]
ROWS_A2 = [            # 23 supporters of a_min
    ("a2", 1.000, 0.750, 5),    # the maximal confidence 1.0
    ("a2", 0.502, 0.750, 4),    # feature 16
    ("a2", 1.000, 0.400, 2),    # feature 17
    ("a2", 0.502, 0.355, 11),
    ("a2", 0.050, 0.355, 1),    # MinC = 0.05
]


def _expand(rows):
    out = []
    for v, c, x, k in rows:
        ps = (x, 1 - x) if v == "a1" else (1 - x, x)
        out += [Response(v, c, ps)] * k
    return out


R_A1 = _expand(ROWS_A1)     # 67 responses
R_A2 = _expand(ROWS_A2)     # 23 responses
R90 = R_A1 + R_A2           # indices 0..66 vote a1, 67..89 vote a2

# TH Table A.2 p.110, as printed.
TABLE_A2 = {
    "N": 90,
    "D_Smax_Smin": 0.489,
    "ES": 0.82,
    "VarS": 0.244,
    "D_S_Uniform": 0.346,
    "CAmin": 0.634,
    "CAmax": 0.788,
    "MinC": 0.05,
    "AvgC": 0.749,
    "VarC": 0.222,
    "D_MaxC_AvgC": 0.251,
    "B_Amax_MaxCa": 1,
    "MaxPSa": 0.686,
    "MinPSa": 0.314,
    "AvgPSV": 0.693,
    "P_lowC_highPSv": 0.122,
    "P_lowPSv_highC": 0.089,
    "SG_B_Amax": 0,
    "SG_VarSAmax": 0.031,
    "SG_D_MaxVarS_MinVarS": 0.067,
    "SG_D_MaxES_MinES": 0.098,
    "SG_D_MaxAvgC_MinAvgC": 0.098,
    "SG_D_MaxVarC_MinVarC": 0.045,
    "SG_VarCAmin": 0.113,
    "SG_VarCAmax": 0.037,
    "SG_AvgCAmin": 0.647,
    "SG_AvgCAmax": 0.789,
}


def subsets_21_23_23():
    """Three disjoint sub-sets of 30 whose a_max counts are (21, 23, 23).

    Plan §3.3.1: the only 3 x 30 split that gives Table A.2's features
    19-21. Indices 0..66 vote a1 and 67..89 vote a2.
    """
    a1 = list(range(0, 67))
    a2 = list(range(67, 90))
    return [a1[0:21] + a2[0:9],
            a1[21:44] + a2[9:16],
            a1[44:67] + a2[16:23]]
