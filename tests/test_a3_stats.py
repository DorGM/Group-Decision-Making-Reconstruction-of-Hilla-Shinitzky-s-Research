"""A3 acceptance (docs/RECONSTRUCTION_PLAN.md §9.3 row A3, §5.8, §5.1).

Published numbers are typed from TH Tables 5.3, 5.4, 5.5, A.10 and MLA
Table 7 / Fig 2; counts are the back-solved ones of plan T1-T3b, T8, T11,
T13, T19-T22.
"""
import pytest
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.proportion import proportions_ztest

from cdm.eval import stats as S

N = 1209
METHODS = ("SP", "MR", "WC", "HAC", "DA")
RULE = dict(SP=864, MR=831, WC=830, HAC=774, DA=372)        # T3 (WC=830, V1)
AMP_10F, DAP_10F = 935, 937                                 # T1, T2
AMP_LOO, DAP_LOO = 975, 982                                 # T19


def sig(x, n):
    """x rounded to n significant figures."""
    return float(f"{x:.{n - 1}e}")


# ---------------------------------------------------------------- §5.1 ----

def test_success_rates_T1_T2_T3_T19():
    assert round(S.success_rate(AMP_10F, N), 3) == 0.773      # T1
    assert round(S.success_rate(DAP_10F, N), 3) == 0.775      # T2
    printed = dict(SP=.715, MR=.687, WC=.687, HAC=.640, DA=.308)  # T3
    for m in METHODS:
        assert round(S.success_rate(RULE[m], N), 3) == printed[m]
    assert round(S.success_rate(AMP_LOO, N), 3) == 0.806      # T19
    assert round(S.success_rate(DAP_LOO, N), 3) == 0.812
    assert S.success_rate_from_outcomes([True, False, True, True]) == 0.75
    with pytest.raises(ValueError):
        S.success_rate(5, 4)


def test_pre_from_rounded_rates_T4_T25():
    # T4: headline PRE of AMP / DAP vs SP = .204 / .211 (rounded rates)
    sp = RULE["SP"] / N
    assert round(S.pre_from_rounded_rates(sp, AMP_10F / N), 3) == 0.204
    assert round(S.pre_from_rounded_rates(sp, DAP_10F / N), 3) == 0.211
    # T25: RP / AP vs MR on 77 sub-sets = .348 / .185 (49, 42 vs 34 of 77)
    assert round(S.pre_from_rounded_rates(34 / 77, 49 / 77), 3) == 0.348
    assert round(S.pre_from_rounded_rates(34 / 77, 42 / 77), 3) == 0.185
    # from counts the headline would be .2058 / .2116
    assert round(S.pre_from_counts(864, N, AMP_10F, N), 4) == 0.2058
    assert round(S.pre_from_counts(864, N, DAP_10F, N), 4) == 0.2116
    assert round(S.pre_from_counts(34, 77, 49, 77), 4) == 0.3488   # T25


# ------------------------------------------- Table 5.4 PRE (T8), 5 values --

T8_COUNTS = dict(SP=(349, 287), MR=(244, 200), WC=(241, 201),
                 HAC=(132, 93), DA=(243, 154))
T8_PRE = dict(SP=.377, MR=.423, WC=.471, HAC=.179, DA=.471)
T8_PRE_4DP = dict(SP=.3775, MR=.4232, WC=.4705, HAC=.1788, DA=.4710)
T8_P_SELECTED = dict(SP=.289, MR=.202, WC=.199, HAC=.109, DA=.201)
T8_P_SUCC_SEL = dict(SP=.822, MR=.820, WC=.834, HAC=.705, DA=.634)


@pytest.mark.parametrize("m", METHODS)
def test_table_5_4_pre_from_exact_counts_T8(m):
    sel, cor = T8_COUNTS[m]
    val = S.pre_from_counts(RULE[m], N, cor, sel)
    assert round(val, 3) == T8_PRE[m]
    assert round(val, 4) == T8_PRE_4DP[m]
    assert round(S.success_rate(sel, N), 3) == T8_P_SELECTED[m]
    assert round(S.success_rate(cor, sel), 3) == T8_P_SUCC_SEL[m]


def test_table_5_4_counts_close_T8():
    assert sum(s for s, _ in T8_COUNTS.values()) == N
    assert sum(c for _, c in T8_COUNTS.values()) == AMP_10F


# ------------------------------ Tables 5.5 (T11) and A.10 (T13), 40 values --

DOMAIN_N = dict(General=500, NBA=393, NFL=212, Politics=104)
T3B = dict(  # per-domain rule counts, order SP, MR, WC, HAC, DA
    General=(413, 385, 380, 338, 108),
    NBA=(258, 252, 253, 237, 142),
    NFL=(109, 111, 114, 131, 101),
    Politics=(84, 83, 83, 68, 21),
)
T11_COUNTS = dict(  # TH Table 5.5 (10-fold): (selected, correct)
    General=((230, 195), (119, 97), (82, 71), (50, 42), (19, 7)),
    NBA=((79, 67), (78, 76), (102, 94), (19, 13), (115, 111)),
    NFL=((24, 12), (28, 10), (49, 28), (37, 22), (74, 28)),
    Politics=((16, 13), (19, 17), (8, 8), (26, 16), (35, 8)),
)
T11_PRE = dict(  # TH Table 5.5 p.69, PRE rows as printed
    General=(.125, .196, .441, .506, .194),
    NBA=(.558, .929, .780, .204, .946),
    NFL=(-.029, -.349, .073, -.061, -.187),
    Politics=(.025, .479, 1.000, -.111, .033),
)
T13_COUNTS = dict(  # TH Table A.10 (LOO): (selected, correct)
    General=((261, 221), (83, 72), (83, 70), (65, 50), (8, 3)),
    NBA=((91, 82), (107, 100), (54, 49), (21, 16), (120, 115)),
    NFL=((41, 25), (45, 22), (30, 16), (73, 46), (23, 8)),
    Politics=((29, 23), (33, 28), (18, 14), (21, 15), (3, 0)),
)
T13_PRE = dict(  # TH Table A.10 p.115, PRE rows as printed
    General=(.119, .424, .347, .288, .203),
    NBA=(.712, .818, .740, .400, .935),
    NFL=(.197, -.073, -.010, .032, -.246),
    Politics=(-.076, .250, -.101, .175, -.253),
)

_DOMAIN_CELLS = [(tid, counts, printed, d, i)
                 for tid, counts, printed in (("T11", T11_COUNTS, T11_PRE),
                                              ("T13", T13_COUNTS, T13_PRE))
                 for d in DOMAIN_N for i in range(5)]


@pytest.mark.parametrize(
    "tid,counts,printed,domain,i", _DOMAIN_CELLS,
    ids=[f"{t}-{d}-{METHODS[i]}" for t, _, _, d, i in _DOMAIN_CELLS])
def test_per_domain_pre_from_exact_counts_T11_T13(tid, counts, printed,
                                                  domain, i):
    sel, cor = counts[domain][i]
    val = S.pre_from_counts(T3B[domain][i], DOMAIN_N[domain], cor, sel)
    assert round(val, 3) == printed[domain][i]


# The other 80 printed cells of Tables 5.5 and A.10: P(selected) and P(success | selected), order SP, MR, WC, HAC, DA.
# Typed from the rendered pages TH p.69 and p.115.
T11_P_SELECTED = dict(
    General=(.460, .238, .164, .100, .038), NBA=(.201, .198, .260, .048, .293),
    NFL=(.113, .132, .231, .175, .349), Politics=(.154, .183, .077, .250, .337))
T11_P_SUCC_SEL = dict(
    General=(.848, .815, .866, .840, .368), NBA=(.848, .974, .922, .684, .965),
    NFL=(.500, .357, .571, .595, .378), Politics=(.813, .895, 1.000, .615, .229))
T13_P_SELECTED = dict(
    General=(.522, .166, .166, .130, .016), NBA=(.232, .272, .137, .053, .305),
    NFL=(.193, .212, .142, .344, .108), Politics=(.279, .317, .173, .202, .029))
T13_P_SUCC_SEL = dict(
    General=(.847, .867, .843, .769, .375), NBA=(.901, .935, .907, .762, .958),
    NFL=(.610, .489, .533, .630, .348), Politics=(.793, .848, .778, .714, .000))

_SEL_CELLS = [(tid, counts, psel, psucc, d, i)
              for tid, counts, psel, psucc in (
                  ("T11", T11_COUNTS, T11_P_SELECTED, T11_P_SUCC_SEL),
                  ("T13", T13_COUNTS, T13_P_SELECTED, T13_P_SUCC_SEL))
              for d in DOMAIN_N for i in range(5)]


@pytest.mark.parametrize(
    "tid,counts,psel,psucc,domain,i", _SEL_CELLS,
    ids=[f"{t}-{d}-{METHODS[i]}" for t, _, _, _, d, i in _SEL_CELLS])
def test_per_domain_selection_cells_T11_T13(tid, counts, psel, psucc, domain, i):
    """P(selected) = selected / N_domain and P(success | selected) =
    correct / selected, rounded half up as the thesis does."""
    sel, cor = counts[domain][i]
    assert S.round_half_up(sel, DOMAIN_N[domain]) == psel[domain][i]
    assert S.round_half_up(cor, sel) == psucc[domain][i]


def test_round_half_up_I4():
    # TH Table 5.5 Politics SP: 13/16 = 0.8125 is printed 0.813
    assert S.round_half_up(13, 16) == 0.813 == T11_P_SUCC_SEL["Politics"][0]
    assert round(13 / 16, 3) == 0.812                 # Python rounds half to even
    assert S.round_half_up(1, 3) == 0.333 and S.round_half_up(2, 3) == 0.667
    assert S.round_half_up(1, 8, decimals=2) == 0.13
    assert S.round_half_up(-1, 16) == -0.063
    assert S.round_half_up(8, 8) == 1.0
    # Python round agrees on every other one of the 80 cells
    diff = [(t, d, i) for t, c, _, _, d, i in _SEL_CELLS
            if round(c[d][i][1] / c[d][i][0], 3) != S.round_half_up(c[d][i][1], c[d][i][0])
            or round(c[d][i][0] / DOMAIN_N[d], 3) != S.round_half_up(c[d][i][0], DOMAIN_N[d])]
    assert diff == [("T11", "Politics", 0)]


def test_per_domain_counts_close_T3b_T8_T10_T12_T20_T21():
    for i, m in enumerate(METHODS):
        assert sum(T3B[d][i] for d in DOMAIN_N) == RULE[m]             # T3b
        assert (sum(T11_COUNTS[d][i][0] for d in DOMAIN_N),
                sum(T11_COUNTS[d][i][1] for d in DOMAIN_N)) == T8_COUNTS[m]
    for d in DOMAIN_N:
        assert sum(s for s, _ in T11_COUNTS[d]) == DOMAIN_N[d]
        assert sum(s for s, _ in T13_COUNTS[d]) == DOMAIN_N[d]
    # T10 / T12: per-domain AMP successes
    assert [sum(c for _, c in T11_COUNTS[d]) for d in DOMAIN_N] == [412, 361, 100, 62]
    assert [sum(c for _, c in T13_COUNTS[d]) for d in DOMAIN_N] == [416, 362, 117, 80]
    # T20 / T21: LOO totals
    assert [sum(T13_COUNTS[d][i][0] for d in DOMAIN_N) for i in range(5)] == \
        [422, 268, 185, 180, 154]
    assert [sum(T13_COUNTS[d][i][1] for d in DOMAIN_N) for i in range(5)] == \
        [351, 222, 149, 127, 126]


# ----------------------------------------------------- Table 5.3 (T7) ------

T7 = {  # TH Table 5.3 p.65, as printed
    ("AMP", "SP"): 0.000469057, ("DAP", "SP"): 0.000330493,
    ("AMP", "MR"): 9.40195e-07, ("DAP", "MR"): 5.8033e-07,
    ("AMP", "WC"): 7.57021e-07, ("DAP", "WC"): 4.65415e-07,
    ("AMP", "HAC"): 3.18856e-13, ("DAP", "HAC"): 1.57874e-13,
}
_APPROACH_10F = dict(AMP=AMP_10F, DAP=DAP_10F)
# Exact upper tail (the statistically correct p-value) at 4 s.f. It equals
# the printed cell at 4 s.f. everywhere except DAP-HAC, where the printed
# 1.57874E-13 is 1 - Phi(z) in double precision and the exact tail is
# 1.57831e-13 (plan §5.8 item 1 as amended by AM1).
T7_EXACT_TAIL_4SF = {cell: sig(v, 4) for cell, v in T7.items()}
T7_EXACT_TAIL_4SF[("DAP", "HAC")] = 1.578e-13


@pytest.mark.parametrize("approach,m", list(T7))
def test_table_5_3_one_tailed_4sf_T7(approach, m):
    """Exact upper tail, 4 s.f. Follows amendment AM1:
    7 cells equal the printed value at 4 s.f.; DAP-HAC is asserted at its
    exact-tail value 1.578e-13 (printed 1.579E-13 at 4 s.f.), see
    plan §5.8 item 1."""
    _, p = S.two_proportion_ztest(_APPROACH_10F[approach], N, RULE[m], N)
    assert sig(p, 4) == T7_EXACT_TAIL_4SF[(approach, m)]
    if (approach, m) == ("DAP", "HAC"):
        assert sig(p, 4) == 1.578e-13 != sig(T7[(approach, m)], 4) == 1.579e-13
        assert sig(p, 6) == 1.57831e-13
    else:
        assert sig(p, 4) == sig(T7[(approach, m)], 4)


def test_table_5_3_dap_hac_printed_cell_is_one_minus_cdf_AM1():
    """Amendment AM1: the printed DAP-HAC cell is
    1 - Phi(z) in double precision, an exact multiple of 2**-53."""
    _, p = S.two_proportion_ztest(DAP_10F, N, RULE["HAC"], N, tail="one_minus_cdf")
    assert sig(p, 6) == T7[("DAP", "HAC")] == 1.57874e-13
    assert p == 1422 * 2.0 ** -53
    _, p = S.two_proportion_ztest(AMP_10F, N, RULE["HAC"], N, tail="one_minus_cdf")
    assert sig(p, 6) == T7[("AMP", "HAC")] and p == 2872 * 2.0 ** -53


@pytest.mark.parametrize("approach,m", list(T7))
def test_table_5_3_one_minus_cdf_T7(approach, m):
    """All 8 printed cells to the printed 5-6 s.f. with p = 1 - Phi(z), the
    as-printed form. Follows amendment AM1."""
    _, p = S.two_proportion_ztest(_APPROACH_10F[approach], N, RULE[m], N,
                                  tail="one_minus_cdf")
    printed = T7[(approach, m)]
    digits = len(f"{printed:e}".split("e")[0].rstrip("0").replace(".", ""))
    assert sig(p, digits) == printed


def test_table_5_3_plan_recomputed_values():
    # plan §5.8 item 1, the plan's own recomputation (6 s.f. / 5 s.f.)
    plan = {("AMP", "SP"): 4.69057e-4, ("AMP", "MR"): 9.40195e-7,
            ("AMP", "WC"): 7.57021e-7, ("AMP", "HAC"): 3.1885e-13,
            ("DAP", "SP"): 3.30493e-4, ("DAP", "MR"): 5.8033e-7,
            ("DAP", "WC"): 4.65415e-7, ("DAP", "HAC"): 1.5783e-13}
    for (a, m), v in plan.items():
        _, p = S.two_proportion_ztest(_APPROACH_10F[a], N, RULE[m], N)
        assert p == pytest.approx(v, rel=5e-5)


def test_table_5_3_DA_row_T7():
    # printed 0; true p ~ 1e-117 (plan §5.8 item 1)
    for a in (AMP_10F, DAP_10F):
        _, p = S.two_proportion_ztest(a, N, RULE["DA"], N)
        assert 0 < p < 1e-116
        _, p0 = S.two_proportion_ztest(a, N, RULE["DA"], N,
                                       tail="one_minus_cdf")
        assert p0 == 0.0


def test_two_proportion_equals_statsmodels():
    for x1, x2, n in ((935, 864, 1209), (982, 830, 1209), (49, 33, 77)):
        for alt in ("larger", "smaller", "two-sided"):
            z, p = S.two_proportion_ztest(x1, n, x2, n, alternative=alt)
            z_sm, p_sm = proportions_ztest([x1, x2], [n, n], alternative=alt)
            assert z == pytest.approx(z_sm, rel=1e-12)
            assert p == pytest.approx(p_sm, rel=1e-10)


def test_table_5_11_proportion_column_T29():
    # plan §5.8 item 2 (n = 77)
    cases = [(49, 33, 0.005, 3), (49, 34, 0.008, 3),
             (42, 33, 0.073, 3), (42, 34, 0.099, 3)]
    for x1, x2, printed, dp in cases:
        _, p = S.two_proportion_ztest(x1, 77, x2, 77)
        assert round(p, dp) == printed
    _, p = S.two_proportion_ztest(49, 77, 22, 77)
    assert sig(p, 2) == 6.4e-6


# --------------------------------- MLA Table 7 proportion column (T22) -----

T22_PROP = {  # MLA Table 7 p.17, as printed (3 s.f.)
    ("AMP", "SP"): 6.13e-08, ("DAP", "SP"): 8.18e-09,
    ("AMP", "MR"): 8.18e-12, ("DAP", "MR"): 6.72e-13,
    ("AMP", "WC"): 6.08e-12, ("DAP", "WC"): 4.92e-13,
}
_APPROACH_LOO = dict(AMP=AMP_LOO, DAP=DAP_LOO)


@pytest.mark.parametrize("approach,m", list(T22_PROP))
def test_mla_table_7_proportion_column_T22(approach, m):
    _, p = S.two_proportion_ztest(_APPROACH_LOO[approach], N, RULE[m], N)
    assert sig(p, 3) == T22_PROP[(approach, m)]


def test_mla_table_7_proportion_zero_cells_T22():
    # HAC and DA cells are printed as 0; true values are < 1e-19
    for a in (AMP_LOO, DAP_LOO):
        for m in ("HAC", "DA"):
            _, p = S.two_proportion_ztest(a, N, RULE[m], N)
            assert 0 < p < 1e-19
            _, p0 = S.two_proportion_ztest(a, N, RULE[m], N,
                                           tail="one_minus_cdf")
            assert p0 == 0.0


# --------------------------- one-sample z: Table 5.4 (T8), MLA Fig 2 (T21) --

T8_P = dict(SP=4.2e-6, MR=4.1e-6, WC=4.0e-7, HAC=0.062)   # TH Table 5.4


@pytest.mark.parametrize("m", list(T8_P))
def test_table_5_4_one_sample_p_T8(m):
    sel, cor = T8_COUNTS[m]
    _, p = S.one_sample_proportion_ztest(cor, sel, RULE[m] / N)
    assert sig(p, 2) == T8_P[m]


def test_table_5_4_wc_needs_830_V1():
    # plan §5.8 item 3: WC with p0 = 830/1209 -> 3.98e-7
    _, p = S.one_sample_proportion_ztest(201, 241, 830 / N)
    assert sig(p, 3) == 3.98e-7


def test_table_5_4_DA_printed_zero_T8():
    sel, cor = T8_COUNTS["DA"]
    _, p = S.one_sample_proportion_ztest(cor, sel, RULE["DA"] / N)
    assert 0 < p < 1e-20
    _, p0 = S.one_sample_proportion_ztest(cor, sel, RULE["DA"] / N,
                                          tail="one_minus_cdf")
    assert p0 == 0.0


def test_pooled_two_sample_does_not_match_table_5_4():
    # plan §5.8 item 3: "The pooled two-sample variant does not match"
    mismatches = 0
    for m, printed in T8_P.items():
        sel, cor = T8_COUNTS[m]
        _, p = S.two_proportion_ztest(cor, sel, RULE[m], N)
        mismatches += sig(p, 2) != printed
    assert mismatches == len(T8_P)


T21_COUNTS = dict(SP=(422, 351), MR=(268, 222), WC=(185, 149),
                  HAC=(180, 127), DA=(154, 126))
T21_RATE = dict(SP=.832, MR=.828, WC=.805, HAC=.706, DA=.818)
T21_P = dict(SP=5.0e-8, MR=3.2e-7, WC=2.5e-4, HAC=3.4e-2)  # MLA Fig 2


@pytest.mark.parametrize("m", METHODS)
def test_mla_fig_2_T20_T21(m):
    sel, cor = T21_COUNTS[m]
    assert round(S.success_rate(cor, sel), 3) == T21_RATE[m]
    _, p = S.one_sample_proportion_ztest(cor, sel, RULE[m] / N)
    if m == "DA":                     # printed 0
        assert 0 < p < 1e-40
        assert S.one_sample_proportion_ztest(
            cor, sel, RULE[m] / N, tail="one_minus_cdf")[1] == 0.0
    else:
        assert sig(p, 2) == T21_P[m]


def test_mla_fig_3_selection_shares_T20():
    printed = dict(SP=35, MR=22, WC=15, HAC=15, DA=13)
    for m in METHODS:
        assert round(100 * S.success_rate(T21_COUNTS[m][0], N)) == printed[m]
    assert sum(c for _, c in T21_COUNTS.values()) == AMP_LOO


# ------------------------------------------------- McNemar (T22, T29) ---- --

T22_MCNEMAR = {  # (approach, rule): ((b, c), p printed in MLA Table 7)
    ("AMP", "SP"): ((159, 48), 2.08e-14),
    ("AMP", "MR"): ((201, 57), 5.45e-19),
    ("AMP", "WC"): ((195, 50), 3.59e-20),
    ("AMP", "HAC"): ((279, 78), 3.49e-26),
    ("AMP", "DA"): ((782, 179), 5.30e-84),
    ("DAP", "SP"): ((189, 71), 3.99e-13),
    ("DAP", "MR"): ((225, 74), 4.15e-18),
    ("DAP", "WC"): ((226, 74), 2.83e-18),
    ("DAP", "HAC"): ((299, 91), 1.05e-25),
    ("DAP", "DA"): ((763, 153), 4.74e-90),
}


@pytest.mark.parametrize("approach,m", list(T22_MCNEMAR))
def test_mla_table_7_mcnemar_T22(approach, m):
    (b, c), printed = T22_MCNEMAR[(approach, m)]
    assert b - c == _APPROACH_LOO[approach] - RULE[m]   # pairs fit the counts
    _, p = S.mcnemar_test(b, c)
    assert sig(p, 3) == printed


@pytest.mark.parametrize("b,c,printed,digits", [
    (15, 0, 3.0e-4, 2),    # RP-MR
    (34, 7, 4.9e-5, 2),    # RP-HAC
    (11, 2, 0.027, 2),     # AP-SP
    (9, 1, 0.027, 2),      # AP-MR
])
def test_table_5_11_mcnemar_unit_cases_T29(b, c, printed, digits):
    _, p = S.mcnemar_test(b, c)
    assert sig(p, digits) == printed


def test_table_5_11_mcnemar_two_solution_cells_T29():
    # plan §5.8 item 4: RP-SP c in {2,3} -> .001; AP-HAC c in {10,11} -> .003
    for b, c in ((18, 2), (19, 3)):
        assert round(S.mcnemar_test(b, c)[1], 3) == 0.001
    for b, c in ((30, 10), (31, 11)):
        assert round(S.mcnemar_test(b, c)[1], 3) == 0.003


def test_mcnemar_is_chi2_with_continuity_correction_V2():
    for b, c in ((159, 48), (15, 0), (9, 1), (7, 7), (3, 12)):
        stat, p = S.mcnemar_test(b, c)
        stat2, p2 = S.mcnemar_statistic(b, c)
        assert stat == pytest.approx((abs(b - c) - 1) ** 2 / (b + c))
        assert (stat, p) == pytest.approx((stat2, p2))
        ref = sm_mcnemar([[100, b], [c, 50]], exact=False, correction=True)
        assert p == pytest.approx(ref.pvalue)        # concordant cells unused
        assert S.mcnemar_test(c, b)[1] == pytest.approx(p)    # two-sided
    with pytest.raises(ValueError):
        S.mcnemar_test(0, 0)
    with pytest.raises(ValueError):                  # same guard
        S.mcnemar_statistic(0, 0)
    with pytest.raises(ValueError):
        S.mcnemar_statistic(-1, 3)


def test_mcnemar_from_outcomes():
    a = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 3
    r = [False] * 20 + [True] * 5 + [True] * 10 + [False] * 3
    assert S.discordant_counts(a, r) == (20, 5)
    assert S.mcnemar_from_outcomes(a, r) == pytest.approx(S.mcnemar_test(20, 5))


# -------------------------------------------- ablation two-tailed (§5.4) ---

def test_ablation_two_tailed_significance_TH_p70():
    """TH p.70 / MLA §7: AMP rows 8 (.754) and 11 (.763) differ significantly
    from the full version (.806), two-tailed proportion test, p < 0.05."""
    for rate in (0.754, 0.763):
        _, p = S.two_proportion_ztest(0.806 * N, N, rate * N, N,
                                      alternative="two-sided")
        assert p < 0.05
        for k in (round(rate * N) - 1, round(rate * N), round(rate * N) + 1):
            _, p = S.two_proportion_ztest(AMP_LOO, N, k, N,
                                          alternative="two-sided")
            assert p < 0.05


T14_AMP = (.806, .800, .816, .795, .808, .797, .791, .754, .806, .799, .763)
T15_DAP = (.812, .812, .825, .803, .804, .805, .806, .813, .811, .815, .798)


def _counts_for_rate(rate):
    """Every integer count k with k/1209 printed as `rate` (3 d.p.)."""
    return [k for k in range(N + 1) if round(k / N, 3) == rate]


def _two_tailed(full, k):
    return S.two_proportion_ztest(full, N, k, N, alternative="two-sided")[1]


def test_ablation_two_tailed_values_and_significance_pattern_AM2():
    """Follows amendment AM2.

    The thesis prints no ablation p-values, only the significance pattern
    (TH pp.70-71; MLA §8): AMP rows 8 and 11 differ from the full version,
    no other AMP row and no DAP row does. The full-version LOO counts are
    975 (AMP) and 982 (DAP) [T19]. The plan's earlier .0023 / .0102 came
    from 974 = round(.806 * 1209), which contradicts T19, and are withdrawn.
    """
    # the counts compatible with the printed rates
    assert _counts_for_rate(.754) == [911, 912]           # AMP row 8
    assert _counts_for_rate(.763) == [922, 923]           # AMP row 11
    assert AMP_LOO in _counts_for_rate(.806) and DAP_LOO in _counts_for_rate(.812)
    # derived p-values (4 d.p.) for 975
    assert round(_two_tailed(AMP_LOO, 911), 4) == 0.0017
    assert round(_two_tailed(AMP_LOO, 912), 4) == 0.0020
    assert round(_two_tailed(AMP_LOO, 922), 4) == 0.0088
    assert round(_two_tailed(AMP_LOO, 923), 4) == 0.0101
    # significance pattern over every admissible count
    lowest_other_amp = 1.0
    for row, rate in enumerate(T14_AMP, start=1):
        for k in _counts_for_rate(rate):
            p = _two_tailed(AMP_LOO, k)
            if row in (8, 11):
                assert p < 0.05, (row, k, p)
            else:
                assert p >= 0.33, (row, k, p)
                lowest_other_amp = min(lowest_other_amp, p)
    assert round(lowest_other_amp, 3) == 0.335            # row 7 (.791)
    lowest_dap = 1.0
    for row, rate in enumerate(T15_DAP[1:], start=2):     # the ten ablated rows
        for k in _counts_for_rate(rate):
            p = _two_tailed(DAP_LOO, k)
            assert p >= 0.38, (row, k, p)
            lowest_dap = min(lowest_dap, p)
    assert round(lowest_dap, 3) == 0.383                  # row 11 (.798)
    # the withdrawn plan figures are not reproduced by the T19 count
    assert round(_two_tailed(AMP_LOO, 912), 4) != 0.0023
    assert round(_two_tailed(AMP_LOO, 922), 4) != 0.0102


def test_argument_checks():
    with pytest.raises(ValueError):
        S.two_proportion_ztest(1, 2, 1, 2, alternative="greater")
    with pytest.raises(ValueError):
        S.two_proportion_ztest(1, 2, 1, 2, tail="exact")
    with pytest.raises(ValueError):
        S.one_sample_proportion_ztest(1, 2, 1.0)
    with pytest.raises(ZeroDivisionError):
        S.pre(1.0, 0.5)
