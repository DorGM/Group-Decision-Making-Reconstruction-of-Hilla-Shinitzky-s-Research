"""A8 acceptance (plan §9.3; plan §2.1.1 Steps D1-D2): World-A loaders.

Expected numbers are counts taken from the raw files and the authors' own
totals in the "Analysis" workbooks. Table A.2 = TH p.110.

All tests need the raw files (CDM_DATA_ROOT); they skip without them.
"""
import collections

import pytest

from a2_case import TABLE_A2
from cdm.config import data_root
from cdm.data.worldA import common, lee, martinie, rutchick
from cdm.features import cdmcr
from cdm.partition import partition
from cdm.schema import Case

ROOT = data_root()
HAVE_DATA = all((ROOT / d / "extracted").is_dir() for d in (
    "Martinie2020_General", "Rutchick2020_NBA_NFL_Politics", "Lee2018_NFL"))
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason=f"raw data not found under {ROOT}")

OPT_C = "c_p_if_true_else_100_minus_p"


def n_range(raws, **kw):
    ns = [len(common.filter_responses(r, **kw)[0]) for r in raws]
    return min(ns), max(ns), sum(ns) / len(ns)


def rate_counts(raws, **kw):
    """(#cases at 0% correct, #cases at 100% correct)."""
    zero = full = 0
    for r in raws:
        n, k = common.count_correct(r, common.filter_responses(r, **kw)[0])
        zero += k == 0
        full += k == n
    return zero, full


# ---- Martinie ----------------------------------------------------------------

@pytest.fixture(scope="module")
def m_raw():
    return martinie.load_raw()


def test_martinie_raw_counts(m_raw):
    # raw files: 500 items, 459 workers = 90+89+93+95+92
    assert len(m_raw) == 500
    per_file = collections.OrderedDict()
    for r in m_raw:
        per_file.setdefault(r.meta["file"], len(r.responses))
    assert list(per_file.values()) == [90, 89, 93, 95, 92]
    assert sum(per_file.values()) == 459
    assert [r.case_id for r in m_raw] == [str(i) for i in range(1, 501)]
    assert sum(r.winner == martinie.TRUE_CODE for r in m_raw) == 250
    lo, hi, mean = n_range(m_raw)
    assert (lo, hi) == (89, 95) and round(mean, 2) == 91.80
    assert rate_counts(m_raw) == (0, 0)


def test_martinie_values_are_whole_percentages_and_complete(m_raw):
    # raw files (no missing cell)
    for raw in m_raw:
        for r in raw.responses:
            assert r.vote in (1, 2)
            assert isinstance(r.conf, int) and 0 <= r.conf <= 100
            assert isinstance(r.others, int) and 0 <= r.others <= 100


def test_martinie_authors_exclusion_counts(m_raw):
    # raw files: 5,168 inconsistent responses; N 61-95,
    # mean 81.46; item 50 becomes 100% correct (81/81)
    kw = {"martinie_drop_vote_inconsistent": True}
    dropped = sum(common.filter_responses(r, **kw)[1]["martinie_vote_inconsistent"]
                  for r in m_raw)
    assert dropped == 5168
    lo, hi, mean = n_range(m_raw, **kw)
    assert (lo, hi) == (61, 95) and round(mean, 2) == 81.46
    assert rate_counts(m_raw, **kw) == (0, 1)
    kept = common.filter_responses(m_raw[49], **kw)[0]
    assert common.count_correct(m_raw[49], kept) == (81, 81)


@pytest.fixture(scope="module")
def item1():
    """Martinie item 1 under option (c) with all 90 responses kept."""
    return martinie.load_cases(confidence=OPT_C, drop_vote_inconsistent=False)[0]


def test_item1_is_the_table_a2_case(item1):
    assert item1.case_id == "General:1" and item1.domain == "General"
    assert item1.answers == ("True", "False") and item1.correct == "True"
    assert item1.n == 90
    assert sum(r.v == "True" for r in item1.responses) == 67


@pytest.mark.parametrize("name", cdmcr.FEATURES_1_17)
def test_table_a2_features_1_17_on_item1(item1, name):
    # TH Table A.2 p.110, all 17 non-sub-group features, to 3 decimals
    f = cdmcr.features_1_17(list(item1.responses), item1.answers)
    assert round(f[name], 3) == TABLE_A2[name], (name, f[name])


SG_18_21 = ("SG_B_Amax", "SG_VarSAmax", "SG_D_MaxVarS_MinVarS", "SG_D_MaxES_MinES")
SG_22_27 = ("SG_D_MaxAvgC_MinAvgC", "SG_D_MaxVarC_MinVarC", "SG_VarCAmin",
            "SG_VarCAmax", "SG_AvgCAmin", "SG_AvgCAmax")


def test_table_a2_features_18_21_need_counts_21_23_23(item1):
    # plan §3.3.1: features 18-21 of Table A.2 depend only on the number of
    # a_max votes per sub-set, and are met by ANY 3 x 30 partition with
    # True-counts {21, 23, 23}. The file-order thirds used here are ONE OF
    # MANY such partitions: 11.3% of random disjoint 3 x 30 partitions have
    # these counts, and so has the interleaved split. This does not establish
    # how the thesis formed its sub-sets.
    subsets = partition(90, None, shuffle=False)
    assert subsets == [list(range(0, 30)), list(range(30, 60)), list(range(60, 90))]
    R = list(item1.responses)
    assert [sum(R[i].v == "True" for i in s) for s in subsets] == [21, 23, 23]
    sg = cdmcr.features_18_27(R, item1.answers, subsets)
    for name in SG_18_21:
        assert round(sg[name], 3) == TABLE_A2[name], (name, sg[name])
    # the interleaved split has the same counts and the same four values
    inter = [list(range(k, 90, 3)) for k in range(3)]
    assert sorted(sum(R[i].v == "True" for i in s) for s in inter) == [21, 23, 23]
    sg2 = cdmcr.features_18_27(R, item1.answers, inter)
    for name in SG_18_21[1:]:
        assert round(sg2[name], 3) == TABLE_A2[name], (name, sg2[name])


def test_table_a2_features_22_27_are_not_reproduced_by_file_order_thirds(item1):
    # File-order thirds differ from all six printed values (and so does the
    # interleaved split). A random search found disjoint partitions that
    # meet 23-27 together, none that also meets feature 22.
    # So features 22-27 of Table A.2 are NOT used as a regression value, and the
    # sub-sets are not established to be file-order blocks (AM13).
    R = list(item1.responses)
    thirds = partition(90, None, shuffle=False)
    inter = [list(range(k, 90, 3)) for k in range(3)]
    for subsets in (thirds, inter):
        sg = cdmcr.features_18_27(R, item1.answers, subsets)
        assert all(round(sg[name], 3) != TABLE_A2[name] for name in SG_22_27)
    sg = cdmcr.features_18_27(R, item1.answers, thirds)
    assert [round(sg[name], 3) for name in SG_22_27] == [
        .032, .071, .028, .022, .631, .787]


def test_plan_default_does_not_reproduce_table_a2():
    # plan §3.3.2 decision rule: with whole percentages (a) cannot give
    # MinC = 0.05; with the authors' exclusion item 1 has 79 responses
    plan = martinie.load_cases()[0]
    assert plan.n == 79
    a_all = martinie.load_cases(drop_vote_inconsistent=False)[0]
    b_all = martinie.load_cases(confidence="b_max_p_100_minus_p_over_100",
                                drop_vote_inconsistent=False)[0]
    assert min(r.c for r in a_all.responses) == 0.0
    assert min(r.c for r in b_all.responses) == 0.5
    assert round(sum(r.c for r in a_all.responses) / 90, 4) == 0.5813
    assert round(sum(r.c for r in b_all.responses) / 90, 4) == 0.7907


def test_martinie_ps_is_share_for_true_and_its_complement(m_raw, item1):
    # plan §2.1 step 2, OQ3: ps[True] = y/100, ps[False] = 1 - y/100
    for raw_r, r in zip(m_raw[0].responses, item1.responses):
        assert r.ps[0] == pytest.approx(raw_r.others / 100)
        assert r.ps[0] + r.ps[1] == pytest.approx(1.0)
        assert isinstance(r.c, float) and all(isinstance(x, float) for x in r.ps)


# ---- Rutchick S1: NFL --------------------------------------------------------

@pytest.fixture(scope="module", params=rutchick.VERSIONS)
def nfl(request):
    return rutchick.load_raw_nfl(request.param)


def test_nfl_raw_counts(nfl):
    # raw files
    assert len(nfl) == 216
    assert n_range(nfl, complete_only=False)[:2] == (121, 161)
    lo, hi, mean = n_range(nfl)
    assert (lo, hi) == (88, 129) and round(mean, 2) == 105.92
    assert [r.case_id for r in nfl
            if len(common.filter_responses(r)[0]) == 129] == ["W3_OAK_MIA"]
    assert rate_counts(nfl) == (0, 0)
    assert collections.Counter(r.winner for r in nfl) == {2: 132, 1: 84}
    active = {x.participant for r in nfl for x in r.responses if x.vote is not None}
    assert len(active) == 202
    weeks = collections.Counter(r.meta["week"] for r in nfl)
    assert [weeks[w] for w in range(1, 16)] == [16, 16, 16, 14, 15, 14, 13, 13, 13,
                                                12, 12, 15, 16, 15, 16]


def test_nfl_header_irregularities(nfl):
    # raw files: 4 token mismatches, 9 non-exact triples
    assert [r.case_id for r in nfl if not r.meta["header_tokens_match"]] == [
        "W1_HOU_NE", "W6_BAL_TEN", "W12_ATL_NO", "W12_PIT_DEN"]
    assert sum(not r.meta["header_exact_match"] for r in nfl) == 9


def test_nfl_original_confidence_coding_and_recode():
    # raw files: 25 columns coded {1,4,5,6,7}; 2,355 cells differ;
    # recoding the original gives the REVISED confidence cells
    orig = rutchick.load_raw_nfl("original")
    rev = rutchick.load_raw_nfl("revised")
    assert sum(r.meta["conf_out_of_scale"] for r in orig) == 25
    assert sum(r.meta["conf_out_of_scale"] for r in rev) == 0
    diff = sum(a.conf != b.conf for o, v in zip(orig, rev)
               for a, b in zip(o.responses, v.responses))
    assert diff == 2355
    votes_equal = all((a.vote, a.others) == (b.vote, b.others)
                      for o, v in zip(orig, rev) for a, b in zip(o.responses, v.responses))
    assert votes_equal
    rutchick.recode_nfl_original_confidence(orig)
    assert all(a.conf == b.conf for o, v in zip(orig, rev)
               for a, b in zip(o.responses, v.responses))


@pytest.mark.parametrize("mode", rutchick.NFL_ORIGINAL_CONF)
def test_nfl_original_confidence_stays_in_unit_interval(mode):
    cases = rutchick.load_cases("nfl", "original", nfl_original_conf=mode)
    assert len(cases) == 216
    assert max(r.c for c in cases for r in c.responses) == 1.0
    lowest = 1 / 7 if mode == "c_over_7" else 0.0
    assert min(r.c for c in cases for r in c.responses) == pytest.approx(lowest)


def test_nfl_original_confidence_default_equals_the_revised_file():
    # the default treatment (1,4,5,6,7 -> 1..5) gives the REVISED confidences
    a = rutchick.load_cases("nfl", "original")
    b = rutchick.load_cases("nfl", "revised")
    assert [[r.c for r in c.responses] for c in a] == [[r.c for r in c.responses] for c in b]


def test_likert_settings_per_treatment():
    orig = rutchick.load_raw_nfl("original")
    odd = next(r for r in orig if r.meta["conf_out_of_scale"])
    plain = next(r for r in orig if not r.meta["conf_out_of_scale"])
    nba = rutchick.load_raw_nba("original")[0]
    A = "a_c_minus_1_over_4"
    assert rutchick.likert_settings(odd, A, "recode_to_revised") == (A, 5, False)
    assert rutchick.likert_settings(plain, A, "domain_minmax_1_7") == (A, 7, False)
    assert rutchick.likert_settings(plain, A, "column_minmax_1_7") == (A, 5, False)
    assert rutchick.likert_settings(odd, A, "column_minmax_1_7") == (A, 7, False)
    assert rutchick.likert_settings(odd, A, "clip") == (A, 5, True)
    assert rutchick.likert_settings(plain, A, "c_over_7") == ("b_c_over_5", 7, False)
    assert rutchick.likert_settings(nba, A, "c_over_7") == (A, 5, False)
    assert rutchick.likert_settings(nba, A, "domain_minmax_1_7") == (A, 5, False)
    assert rutchick.likert_settings(nba, A, "all_likert_minmax_1_7") == (A, 7, False)
    # REVISED files are never touched
    rev = rutchick.load_raw_nfl("revised")[0]
    assert rutchick.likert_settings(rev, A, "all_likert_minmax_1_7") == (A, 5, False)
    with pytest.raises(ValueError):
        rutchick.likert_settings(odd, A, "something")


# ---- Rutchick S3: NBA --------------------------------------------------------

@pytest.fixture(scope="module", params=rutchick.VERSIONS)
def nba(request):
    return rutchick.load_raw_nba(request.param)


def test_nba_raw_counts(nba):
    # raw files
    assert len(nba) == 465
    lo, hi, mean = n_range(nba)
    assert (lo, hi) == (10, 61) and round(mean, 2) == 21.05
    assert rate_counts(nba) == (17, 52)
    assert collections.Counter(r.winner for r in nba) == {2: 266, 1: 199}
    active = {x.participant for r in nba for x in r.responses if x.vote is not None}
    assert len(active) == 113
    assert nba[0].case_id == "W9_9"                      # file column order
    assert all(r.meta["header_exact_match"] for r in nba)


def test_nba_original_has_one_vote_coded_zero():
    # raw files: AGM26 (participant 25, W3_2) 0 -> 2 in REVISED
    orig = {r.case_id: r for r in rutchick.load_raw_nba("original")}["W3_2"]
    rev = {r.case_id: r for r in rutchick.load_raw_nba("revised")}["W3_2"]
    kept, n = common.filter_responses(orig)
    assert n["vote_not_an_answer"] == 1 and len(kept) == 36
    assert len(common.filter_responses(rev)[0]) == 37


def test_nba_three_games_with_one_of_61_correct(nba):
    # raw files
    low = [r.case_id for r in nba
           if common.count_correct(r, common.filter_responses(r)[0]) == (61, 1)]
    assert sorted(low) == ["W2_10", "W2_17", "W2_9"]


# ---- Rutchick S2: Elections --------------------------------------------------

@pytest.fixture(scope="module", params=rutchick.VERSIONS)
def elections(request):
    return rutchick.load_raw_elections(request.param)


def test_elections_raw_counts(elections):
    # raw files
    assert len(elections) == 110
    lo, hi, mean = n_range(elections)
    assert (lo, hi) == (63, 76) and round(mean, 2) == 70.23
    assert n_range(elections, complete_only=False)[:2] == (63, 76)
    assert rate_counts(elections) == (0, 0)
    people = {x.participant for r in elections for x in r.responses if x.vote is not None}
    assert len(people) == 309
    assert collections.Counter(r.winner for r in elections) == {1: 63, 2: 45, 3: 2}
    votes = collections.Counter(x.vote for r in elections for x in r.responses
                                if x.vote is not None)
    assert votes == {1: 3957, 2: 3617, 3: 127, 4: 24}


def test_elections_are_not_all_binary(elections):
    # raw files: 46 races offered 3 or 4 options
    assert all(r.meta["options_source"] == "qsf" for r in elections)
    offered = collections.Counter(len(r.options) for r in elections)
    assert offered == {2: 64, 3: 45, 4: 1}
    assert sorted(r.case_id for r in elections if r.winner == 3) == ["ME_sen", "MS_sen1"]
    ids = [r.case_id for r in elections]
    assert "AK_gub" in ids and "AK_gub#2" in ids and len(set(ids)) == 110


def test_elections_header_irregularities():
    # raw files
    six = ["TX_sen", "MT_atlarge", "CA_45th", "NY_sen", "IL_gub", "SC_gub"]
    orig = rutchick.load_raw_elections("original")
    rev = rutchick.load_raw_elections("revised")
    assert [r.case_id for r in orig if not r.meta["name_match_strict"]] == six
    assert [r.case_id for r in orig if not r.meta["name_match_loose"]] == six
    assert [r.case_id for r in rev if not r.meta["name_match_strict"]] == six
    assert [r.case_id for r in rev if not r.meta["name_match_loose"]] == six[1:]
    # which of the three names differ (rule gate report §5.3)
    assert [r.case_id for r in orig if not r.meta["pred_agree_match"]] == [
        "TX_sen", "CA_45th", "IL_gub", "SC_gub"]
    assert [r.case_id for r in orig if not r.meta["pred_conf_match"]] == [
        "MT_atlarge", "CA_45th", "NY_sen"]


def test_elections_third_option_votes_and_observed_answer_set(elections):
    # 151 votes for options 3-4 in 38
    # races; six races have 6 or more; in 8 three-option races nobody chose
    # the third option
    votes = {r.case_id: r.meta["third_option_votes"] for r in elections}
    assert sum(votes.values()) == 151 and sum(v > 0 for v in votes.values()) == 38
    assert {k: v for k, v in votes.items() if v >= 6} == {
        "MS_sen1": 36, "ME_sen": 17, "NM_sen": 10, "WI_1": 8, "TX_32": 6, "AK_gub": 6}
    assert sum(v >= 5 for v in votes.values()) == 7       # threshold 5 -> 103
    assert sum(v >= 7 for v in votes.values()) == 4       # threshold 7 -> 106
    observed = collections.Counter(len(r.meta["observed_options"]) for r in elections)
    assert observed == {2: 72, 3: 37, 4: 1}
    for r in elections:
        assert set(r.meta["observed_options"]) <= set(r.options)
        assert {1, 2, r.winner} <= set(r.meta["observed_options"])


# ---- authors' own totals (plan §9.3 A8 sanity) --------------------------------

def _authors_totals(raws):
    """Democratic, confidence-weighted and SP totals as the Analysis
    workbooks compute them: every vote counts, each mean is over whoever
    gave that field, ties score 1/k."""
    dem = cw = sp = 0.0
    for raw in raws:
        cnt = {o: 0 for o in raw.options}
        cs = {o: [] for o in raw.options}
        os_ = {o: [] for o in raw.options}
        for r in raw.responses:
            if r.vote in cnt:
                cnt[r.vote] += 1
                if r.conf is not None:
                    cs[r.vote].append(r.conf)
                if r.others is not None:
                    os_[r.vote].append(r.others)

        def credit(score):
            m = max(score.values())
            best = [o for o in score if abs(score[o] - m) < 1e-9]
            return 1.0 / len(best) if raw.winner in best else 0.0

        dem += credit(cnt)
        cw += credit({o: cnt[o] * sum(cs[o]) / len(cs[o]) if cs[o] else 0.0 for o in cnt})
        if len(raw.options) == 2:
            n = cnt[1] + cnt[2]
            k1 = sum(os_[1]) / len(os_[1]) if os_[1] else 0.0
            k2 = sum(os_[2]) / len(os_[2]) if os_[2] else 0.0
            pred = 1 if cnt[1] * 100 / n > cnt[1] / n * k1 + cnt[2] / n * (100 - k2) else 2
            sp += pred == raw.winner
    return dem, cw, sp


def test_authors_totals_nfl(nfl):
    # NFL Analysis workbook rows 241-243: 116.5 / 120 / 111 of 216
    assert _authors_totals(nfl) == (116.5, 120.0, 111.0)


def test_authors_totals_nba(nba):
    # NBA Analysis workbook rows 140-142: 306.5 / 304.5 / 313 of 465
    assert _authors_totals(nba) == (306.5, 304.5, 313.0)


def test_authors_totals_elections(elections):
    # Election Analysis workbook rows 349-350: democratic 87, CW 87.5 of 110
    # (the four-option SP formula is not re-implemented here)
    dem, cw, _ = _authors_totals(elections)
    assert (dem, cw) == (87.0, 87.5)


# ---- Case conversion ---------------------------------------------------------

@pytest.mark.parametrize("study, n_cases, first_answers", [
    ("nba", 465, ("home", "away")),
    ("nfl", 216, ("PHI", "ATL")),            # W1_ATL_PHI: second-named first
    ("elections", 110, ("option1", "option2")),
])
def test_rutchick_cases(study, n_cases, first_answers):
    cases = rutchick.load_cases(study)
    assert len(cases) == n_cases and all(isinstance(c, Case) for c in cases)
    assert cases[0].answers == first_answers                 # OQ5 (a)
    raw = rutchick.load_raw(study)
    for c, r in zip(cases, raw):
        assert c.case_id == f"{r.domain}:{r.case_id}" and c.domain == r.domain
        assert c.n == len(common.filter_responses(r)[0])
    for c in cases[:40]:
        for r in c.responses:
            assert r.c in (0.0, 0.25, 0.5, 0.75, 1.0)         # OQ2 Likert (a)
            if len(c.answers) == 2:
                assert sum(r.ps) == pytest.approx(1.0)
            assert isinstance(r.c, float) and isinstance(r.ps, tuple)
    other = rutchick.load_cases(study, likert_conf="b_c_over_5")
    assert {r.c for c in other[:40] for r in c.responses} <= {0.2, 0.4, 0.6, 0.8, 1.0}


def test_binary_predicted_support_from_the_single_number():
    # plan §2.1 step 2 / OQ3: ps[v] = x/100, ps[other] = 1 - x/100
    raw = rutchick.load_raw_nba()[0]
    kept, _ = common.filter_responses(raw)
    case = common.make_case(raw, kept)
    for rr, r in zip(kept, case.responses):
        i = case.answers.index(r.v)
        assert r.ps[i] == pytest.approx(rr.others / 100)
        assert r.ps[1 - i] == pytest.approx(1 - rr.others / 100)


def test_multi_answer_predicted_support_rules():
    # more than two answers (plan silent, AM15). Default: Rutchick's own SP
    # formula, 100 - x for EVERY other option; the even split is a switch.
    raw = next(r for r in rutchick.load_raw_elections()
               if len(r.meta["observed_options"]) == 3)
    opts = raw.meta["observed_options"]
    kept, _ = common.filter_responses(raw, options=opts)
    comp = common.make_case(raw, kept, options=opts)
    split = common.make_case(raw, kept, options=opts, ps_other_answers="split_evenly")
    assert len(comp.answers) == 3
    for rr, a, b in zip(kept, comp.responses, split.responses):
        x = rr.others / 100
        i = comp.answers.index(a.v)
        assert a.ps[i] == pytest.approx(x) == b.ps[i]
        assert all(a.ps[j] == pytest.approx(1 - x) for j in range(3) if j != i)
        assert all(b.ps[j] == pytest.approx((1 - x) / 2) for j in range(3) if j != i)
        assert sum(b.ps) == pytest.approx(1.0)
        assert sum(a.ps) == pytest.approx(x + 2 * (1 - x))     # not 1
    with pytest.raises(ValueError):
        common.make_case(raw, kept, options=opts, ps_other_answers="other")
    # a binary case is the same under both rules
    nba = rutchick.load_raw_nba()[0]
    assert common.make_case(nba) == common.make_case(nba, ps_other_answers="split_evenly")


def test_elections_answer_set_switch():
    obs = rutchick.load_cases("elections")
    qsf = rutchick.load_cases("elections", elections_answer_set="offered_qsf")
    size = lambda cases: collections.Counter(len(c.answers) for c in cases)
    assert size(obs) == {2: 72, 3: 37, 4: 1}       # codes that occur in the workbook
    assert size(qsf) == {2: 64, 3: 45, 4: 1}       # options offered in the survey
    assert [c.n for c in obs] == [c.n for c in qsf]
    with pytest.raises(ValueError):
        rutchick.load_cases("elections", elections_answer_set="x")


def test_answer_coding_options():
    # plan §8 OQ5
    raw = rutchick.load_raw_nba()[:30]
    a = [common.make_case(r) for r in raw]
    b = [common.make_case(r, answer_coding="b_raw_file_option_order") for r in raw]
    c1 = [common.make_case(r, answer_coding="c_random_per_case_seeded", coding_seed=0)
          for r in raw]
    c2 = [common.make_case(r, answer_coding="c_random_per_case_seeded", coding_seed=0)
          for r in raw]
    assert all(c.answers == ("home", "away") for c in a)
    assert all(c.answers == ("away", "home") for c in b)
    assert [c.answers for c in c1] == [c.answers for c in c2]
    assert len({c.answers for c in c1}) == 2
    m = martinie.load_cases(answer_coding="b_raw_file_option_order")[0]
    assert m.answers == ("False", "True")
    # the coding only reorders: the vote, confidence and ps of each answer stay
    for x, y in zip(a[0].responses, b[0].responses):
        assert (x.v, x.c) == (y.v, y.c) and x.ps == y.ps[::-1]


# ---- Lee 2018 ----------------------------------------------------------------

def test_lee_responses_only():
    # raw files: 257 game columns (256 distinct games), exactly
    # 100 complete responses each, no outcomes
    raws = lee.load_raw()
    assert len(raws) == 257
    assert [r.case_id for r in raws if r.meta["same_pair_appears_later"]] == ["W1_4"]
    assert n_range(raws)[:2] == (100, 100)
    assert all(r.winner is None for r in raws)
    weeks = collections.Counter(r.meta["week"] for r in raws)
    assert [weeks[w] for w in range(1, 18)] == [16, 16, 16, 16, 14, 14, 15, 13, 13,
                                                14, 14, 16, 16, 16, 16, 16, 16]
    conf = collections.Counter(x.conf for r in raws for x in r.responses)
    assert conf == {1: 1429, 2: 4010, 3: 8199, 4: 8919, 5: 3143}
    experts = sum(1 for w in range(1, 18)
                  for x in next(r for r in raws if r.meta["week"] == w).responses
                  if x.extra == lee.EXPERT_LABEL)
    assert experts == 310


def test_lee_cases_need_outcomes():
    with pytest.raises(lee.LeeOutcomesMissing):
        lee.load_cases()
    with pytest.raises(ValueError):
        common.make_case(lee.load_raw()[0])
