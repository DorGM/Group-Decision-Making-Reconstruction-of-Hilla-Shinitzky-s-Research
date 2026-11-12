"""Thesis / paper numbers that a run is shown next to (docs/RECONSTRUCTION_PLAN.md §2).

Printed values and the plan's back-solved counts, with their plan IDs.
Nothing here is computed from the data. Used by the driver scripts in
scripts/ so that every result file carries the thesis-reported value beside
the result (docs/RULE_BASELINE_CHECK.md).
"""
N_CASES = 1209
DOMAIN_N = {"General": 500, "NBA": 393, "NFL": 212, "Politics": 104}

#: Rule baselines printed in the thesis (correct cases of 1209); plan T3 /
#: the rule gate (cdm.eval.rule_gate).
RULES_THESIS = {"SP": 864, "MR": 831, "WC": 830, "HAC": 774, "DA": 372}

# ---- single LOO, BR+RF / RF (MLA §5.5; TH Tables A.9, A.10) ---------------------------
#: T19: successes of 1209.
LOO_SUCCESSES = {"AMP": 975, "DAP": 982}
#: T12: successes per domain (TH Table A.9).
LOO_DOMAIN_SUCCESSES = {
    "AMP": {"General": 416, "NBA": 362, "NFL": 117, "Politics": 80},
    "DAP": {"General": 426, "NBA": 357, "NFL": 123, "Politics": 76},
}
#: T20: AMP selection counts (TH Table A.10 / MLA Fig 3).
LOO_AMP_SELECTED = {"SP": 422, "MR": 268, "WC": 185, "HAC": 180, "DA": 154}
#: T21: correct-when-selected counts; rates .832/.828/.805/.706/.818.
LOO_AMP_SELECTED_SUCCESS = {"SP": 351, "MR": 222, "WC": 149, "HAC": 127, "DA": 126}
#: T22: discordant pairs (approach-only correct, rule-only correct)
#: [INFERRED in the plan from MLA Table 7].
LOO_DISCORDANT = {
    "AMP": {"SP": (159, 48), "MR": (201, 57), "WC": (195, 50), "HAC": (279, 78),
            "DA": (782, 179)},
    "DAP": {"SP": (189, 71), "MR": (225, 74), "WC": (226, 74), "HAC": (299, 91),
            "DA": (763, 153)},
}

# ---- leave-one-out within 10-fold CV (TH §5.1; Tables 5.1, 5.2, 5.4) -----------------
#: T1 / T2: accumulated test successes of 1209.
NESTED_SUCCESSES = {"AMP": 935, "DAP": 937}
#: T10: AMP successes per domain (10-fold).
NESTED_AMP_DOMAIN_SUCCESSES = {"General": 412, "NBA": 361, "NFL": 100, "Politics": 62}
#: T8: AMP selected / correct-when-selected counts (TH Table 5.4).
NESTED_AMP_SELECTED = {"SP": 349, "MR": 244, "WC": 241, "HAC": 132, "DA": 243}
NESTED_AMP_SELECTED_SUCCESS = {"SP": 287, "MR": 200, "WC": 201, "HAC": 93, "DA": 154}

AMP_TECHNIQUES = ("BR+KNN", "BR+LR", "BR+RF", "CC+KNN", "CC+LR", "CC+RF",
                  "LP+KNN", "LP+LR", "LP+RF")
DAP_TECHNIQUES = ("KNN", "LR", "RF")
#: TH Table 5.1 p.63: per fold the inner-LOO success rate of the 9 AMP
#: techniques (order AMP_TECHNIQUES), then the test score. Checked cell by
#: cell against the rendered page.
TH_TABLE_5_1 = (
    (0.776, 0.759, 0.800, 0.774, 0.749, 0.790, 0.776, 0.767, 0.789, 0.851),
    (0.783, 0.772, 0.812, 0.789, 0.767, 0.809, 0.783, 0.784, 0.806, 0.793),
    (0.778, 0.767, 0.808, 0.774, 0.754, 0.796, 0.778, 0.772, 0.792, 0.826),
    (0.788, 0.767, 0.803, 0.784, 0.748, 0.792, 0.788, 0.777, 0.792, 0.835),
    (0.769, 0.748, 0.793, 0.768, 0.731, 0.777, 0.769, 0.756, 0.789, 0.934),
    (0.768, 0.750, 0.789, 0.763, 0.726, 0.790, 0.768, 0.757, 0.789, 0.917),
    (0.779, 0.767, 0.805, 0.778, 0.754, 0.801, 0.779, 0.774, 0.797, 0.860),
    (0.791, 0.779, 0.827, 0.788, 0.778, 0.808, 0.791, 0.798, 0.823, 0.645),
    (0.813, 0.788, 0.830, 0.810, 0.768, 0.836, 0.813, 0.799, 0.824, 0.479),
    (0.796, 0.781, 0.817, 0.789, 0.764, 0.809, 0.796, 0.788, 0.813, 0.592),
)
#: TH Table 5.2 p.63: KNN, LR, RF, then the test score.
TH_TABLE_5_2 = (
    (0.763, 0.695, 0.793, 0.810), (0.767, 0.708, 0.804, 0.826),
    (0.757, 0.707, 0.802, 0.901), (0.760, 0.699, 0.792, 0.868),
    (0.757, 0.730, 0.792, 0.826), (0.759, 0.733, 0.781, 0.826),
    (0.765, 0.720, 0.802, 0.835), (0.790, 0.737, 0.824, 0.727),
    (0.797, 0.720, 0.827, 0.579), (0.782, 0.714, 0.820, 0.550),
)


def thesis_nested_table(approach):
    """(technique names, rows) of TH Table 5.1 (AMP) / 5.2 (DAP)."""
    if approach == "AMP":
        return AMP_TECHNIQUES, TH_TABLE_5_1
    if approach == "DAP":
        return DAP_TECHNIQUES, TH_TABLE_5_2
    raise ValueError(f"unknown approach {approach!r}")


# ---- AMP effectiveness tables (step A15; TH Tables 5.4, 5.5, A.9, A.10) -------------
#: Column order of TH Tables 5.4 / 5.5 / A.10.
METHODS = ("SP", "MR", "WC", "HAC", "DA")
DOMAINS = ("General", "NBA", "NFL", "Politics")
EFFECTIVENESS_ROWS = ("p_selected", "p_success", "p_success_given_selected", "pre")

#: T3b: correct cases of every rule per domain (TH Table 5.5 = A.10, row 2
#: of each block; back-solved in the plan, column sums = RULES_THESIS).
RULES_THESIS_DOMAIN = {
    "General": {"SP": 413, "MR": 385, "WC": 380, "HAC": 338, "DA": 108},
    "NBA": {"SP": 258, "MR": 252, "WC": 253, "HAC": 237, "DA": 142},
    "NFL": {"SP": 109, "MR": 111, "WC": 114, "HAC": 131, "DA": 101},
    "Politics": {"SP": 84, "MR": 83, "WC": 83, "HAC": 68, "DA": 21},
}


def _pairs(sp, mr, wc, hac, da):
    """((selected, correct when selected) x 5) -> two dicts by method."""
    cells = dict(zip(METHODS, (sp, mr, wc, hac, da)))
    return ({m: c[0] for m, c in cells.items()}, {m: c[1] for m, c in cells.items()})


#: T11: 10-fold AMP selected / correct-when-selected counts per domain
#: [INFERRED in the plan, unique] (TH Table 5.5).
NESTED_AMP_DOMAIN_SELECTED, NESTED_AMP_DOMAIN_SELECTED_SUCCESS = {}, {}
for _d, _cells in {
        "General": ((230, 195), (119, 97), (82, 71), (50, 42), (19, 7)),
        "NBA": ((79, 67), (78, 76), (102, 94), (19, 13), (115, 111)),
        "NFL": ((24, 12), (28, 10), (49, 28), (37, 22), (74, 28)),
        "Politics": ((16, 13), (19, 17), (8, 8), (26, 16), (35, 8))}.items():
    NESTED_AMP_DOMAIN_SELECTED[_d], NESTED_AMP_DOMAIN_SELECTED_SUCCESS[_d] = _pairs(*_cells)
#: T13: the same for the single LOO (TH Table A.10).
LOO_AMP_DOMAIN_SELECTED, LOO_AMP_DOMAIN_SELECTED_SUCCESS = {}, {}
for _d, _cells in {
        "General": ((261, 221), (83, 72), (83, 70), (65, 50), (8, 3)),
        "NBA": ((91, 82), (107, 100), (54, 49), (21, 16), (120, 115)),
        "NFL": ((41, 25), (45, 22), (30, 16), (73, 46), (23, 8)),
        "Politics": ((29, 23), (33, 28), (18, 14), (21, 15), (3, 0))}.items():
    LOO_AMP_DOMAIN_SELECTED[_d], LOO_AMP_DOMAIN_SELECTED_SUCCESS[_d] = _pairs(*_cells)
del _d, _cells

#: TH Table 5.4 p.66 as printed (columns SP, MR, WC, HAC, DA). The p row:
#: one-tailed, one-sample z-test (plan §5.8 item 3); DA is printed as 0.
TH_TABLE_5_4 = {
    "p_selected": (0.289, 0.202, 0.199, 0.109, 0.201),
    "p_success": (0.715, 0.687, 0.687, 0.640, 0.308),
    "p_success_given_selected": (0.822, 0.820, 0.834, 0.705, 0.634),
    "pre": (0.377, 0.423, 0.471, 0.179, 0.471),
    "p": (4.2e-6, 4.1e-6, 4.0e-7, 0.062, 0.0),
}
#: TH Table 5.5 p.69 as printed (10-fold), per domain.
TH_TABLE_5_5 = {
    "General": {"p_selected": (0.460, 0.238, 0.164, 0.100, 0.038),
                "p_success": (0.826, 0.770, 0.760, 0.676, 0.216),
                "p_success_given_selected": (0.848, 0.815, 0.866, 0.840, 0.368),
                "pre": (0.125, 0.196, 0.441, 0.506, 0.194)},
    "NBA": {"p_selected": (0.201, 0.198, 0.260, 0.048, 0.293),
            "p_success": (0.656, 0.641, 0.644, 0.603, 0.361),
            "p_success_given_selected": (0.848, 0.974, 0.922, 0.684, 0.965),
            "pre": (0.558, 0.929, 0.780, 0.204, 0.946)},
    "NFL": {"p_selected": (0.113, 0.132, 0.231, 0.175, 0.349),
            "p_success": (0.514, 0.524, 0.538, 0.618, 0.476),
            "p_success_given_selected": (0.500, 0.357, 0.571, 0.595, 0.378),
            "pre": (-0.029, -0.349, 0.073, -0.061, -0.187)},
    "Politics": {"p_selected": (0.154, 0.183, 0.077, 0.250, 0.337),
                 "p_success": (0.808, 0.798, 0.798, 0.654, 0.202),
                 "p_success_given_selected": (0.813, 0.895, 1.000, 0.615, 0.229),
                 "pre": (0.025, 0.479, 1.000, -0.111, 0.033)},
}
#: TH Table A.10 p.115 as printed (single LOO), per domain.
TH_TABLE_A_10 = {
    "General": {"p_selected": (0.522, 0.166, 0.166, 0.130, 0.016),
                "p_success": (0.826, 0.770, 0.760, 0.676, 0.216),
                "p_success_given_selected": (0.847, 0.867, 0.843, 0.769, 0.375),
                "pre": (0.119, 0.424, 0.347, 0.288, 0.203)},
    "NBA": {"p_selected": (0.232, 0.272, 0.137, 0.053, 0.305),
            "p_success": (0.656, 0.641, 0.644, 0.603, 0.361),
            "p_success_given_selected": (0.901, 0.935, 0.907, 0.762, 0.958),
            "pre": (0.712, 0.818, 0.740, 0.400, 0.935)},
    "NFL": {"p_selected": (0.193, 0.212, 0.142, 0.344, 0.108),
            "p_success": (0.514, 0.524, 0.538, 0.618, 0.476),
            "p_success_given_selected": (0.610, 0.489, 0.533, 0.630, 0.348),
            "pre": (0.197, -0.073, -0.010, 0.032, -0.246)},
    "Politics": {"p_selected": (0.279, 0.317, 0.173, 0.202, 0.029),
                 "p_success": (0.808, 0.798, 0.798, 0.654, 0.202),
                 "p_success_given_selected": (0.793, 0.848, 0.778, 0.714, 0.000),
                 "pre": (-0.076, 0.250, -0.101, 0.175, -0.253)},
}
#: TH Table A.9 p.115 as printed: success rate per domain, single LOO.
TH_TABLE_A_9 = {
    "AMP": {"General": 0.832, "NBA": 0.921, "NFL": 0.552, "Politics": 0.769},
    "DAP": {"General": 0.852, "NBA": 0.908, "NFL": 0.580, "Politics": 0.731},
}
#: T10 (TH §5.1.0.1 p.68, text): AMP success rate per domain, 10-fold.
TH_NESTED_AMP_DOMAIN_RATES = {"General": 0.824, "NBA": 0.919, "NFL": 0.472,
                              "Politics": 0.596}
#: T20 / T21 as printed in MLA Figs 2-3 (global, single LOO): selection
#: share in %, P(success | selected), p (DA printed as 0).
MLA_LOO_PRINTED = {
    "p_selected_percent": (35, 22, 15, 15, 13),
    "p_success_given_selected": (0.832, 0.828, 0.805, 0.706, 0.818),
    "p": (5.0e-8, 3.2e-7, 2.5e-4, 3.4e-2, 0.0),
}

# ---- classic meta-learning (step A17; TH Table 5.8 p.73) -----------------------------
#: Row -> (f_WC + f_HAC, f_SP, f_DA included; f_MR always), printed rate.
TH_TABLE_5_8 = {
    1: ((1, 1, 1), 0.713),
    2: ((0, 1, 1), 0.708),
    3: ((1, 0, 1), 0.695),
    4: ((0, 0, 1), 0.687),
    5: ((1, 1, 0), 0.713),
}

# ---- bagging (step A18; TH Tables 5.9, 5.10 p.74) ------------------------------------
#: F -> the ten fold success rates as printed. The "-" row of the thesis
#: tables is the Test column of TH_TABLE_5_1 / TH_TABLE_5_2.
BAGGING_FACTORS = (10, 5, 3, 2)
TH_TABLE_5_9 = {
    10: (0.868, 0.760, 0.851, 0.884, 0.909, 0.744, 0.727, 0.636, 0.603, 0.592),
    5: (0.860, 0.777, 0.860, 0.868, 0.884, 0.785, 0.669, 0.628, 0.579, 0.667),
    3: (0.860, 0.793, 0.835, 0.851, 0.884, 0.777, 0.669, 0.612, 0.570, 0.617),
    2: (0.860, 0.793, 0.868, 0.851, 0.893, 0.793, 0.686, 0.645, 0.562, 0.567),
}
TH_TABLE_5_10 = {
    10: (0.843, 0.826, 0.802, 0.785, 0.512, 0.702, 0.678, 0.628, 0.471, 0.467),
    5: (0.826, 0.802, 0.893, 0.917, 0.769, 0.686, 0.669, 0.653, 0.570, 0.533),
    3: (0.843, 0.826, 0.876, 0.884, 0.818, 0.694, 0.669, 0.636, 0.554, 0.558),
    2: (0.826, 0.802, 0.860, 0.876, 0.810, 0.711, 0.719, 0.653, 0.579, 0.592),
}


def thesis_bagging_table(approach):
    """{F: ten printed fold rates} of TH Table 5.9 (AMP) / 5.10 (DAP), and
    the printed "-" row (the Test column of TH Table 5.1 / 5.2)."""
    if approach == "AMP":
        return TH_TABLE_5_9, tuple(r[-1] for r in TH_TABLE_5_1)
    if approach == "DAP":
        return TH_TABLE_5_10, tuple(r[-1] for r in TH_TABLE_5_2)
    raise ValueError(f"unknown approach {approach!r}")
