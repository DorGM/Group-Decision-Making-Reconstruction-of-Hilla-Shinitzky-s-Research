"""A9 acceptance (plan §9.3; plan §2.1.1 Steps D3, D4, D6): World-A build.

Thesis-reported values (TH p.43, p.68): 1209 cases = 500 General / 393 NBA / 212 NFL /
104 Politics; N mean 70, SD 35.5, min 10, max 129; correct mean 62%,
SD 23.8, min 2.7%, max 98.9%.

STATUS: no build is established as the thesis data set. Every scenario
is a hypothesis. The `h1209_*` scenarios
(formerly `best_evidence_*`) reach the four counts through three count
hypotheses; their SDs differ from both printed SDs. `waiver_main` /
`waiver_politics_alt` are the builds used for the model runs
(docs/RULE_BASELINE_CHECK.md); they are not defaults. The rule gate itself is tested in tests/test_a10_rule_gate.py;
`freeze` writes its result into the manifest.
All tests need the raw files (CDM_DATA_ROOT); they skip without them.
"""
import json

import pytest

from cdm.config import data_root
from cdm.data.worldA import build_worldA as B
from cdm.data.worldA import lee
from cdm.schema import Case

ROOT = data_root()
HAVE_DATA = all((ROOT / d / "extracted").is_dir() for d in (
    "Martinie2020_General", "Rutchick2020_NBA_NFL_Politics"))
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason=f"raw data not found under {ROOT}")

H1209 = ("h1209_nba_games", "h1209_nba_participants")
WAIVER = ("waiver_main", "waiver_politics_alt")
ORDER = ["General", "NBA", "NFL", "Politics"]


@pytest.fixture(scope="module")
def builds():
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = B.build(scenario=name)
        return cache[name]
    return get


# ---- h1209 hypothesis scenarios: counts and TH p.43 statistics ----------------

@pytest.mark.parametrize("name", H1209 + WAIVER)
def test_h1209_and_waiver_counts(builds, name):
    b = builds(name)
    assert b.counts() == {"General": 500, "NBA": 393, "NFL": 212, "Politics": 104}
    assert len(b.records) == len(b.cases) == 1209
    assert sum(1 for r in b.records if r["domain"] != "General") == 709   # TH p.43


@pytest.mark.parametrize("name", H1209)
def test_h1209_statistics_that_match_at_printed_precision(builds, name):
    s = builds(name).stats()
    assert round(s["n_mean"]) == 70                 # 69.66 / 69.59
    assert s["n_min"] == 10 and s["n_max"] == 129
    assert round(s["rate_mean"]) == 62              # 61.95 / 61.96
    assert round(s["rate_min"], 1) == 2.7           # 1/37
    assert round(s["rate_max"], 1) == 98.9          # 94/95 = 98.947


@pytest.mark.parametrize("name", H1209)
def test_h1209_sds_differ_from_the_printed_values_at_one_decimal(builds, name):
    # Thesis: N SD 35.5, correct SD 23.8. These builds: N SD 35.59-35.63
    # (prints as 35.6: differs); correct SD 23.72-23.76 (prints as 23.7:
    # differs, except the sample SD of the participants variant, 23.76).
    # A random search found almost no choice of 4 NFL games + 6 races
    # that meets both SDs once the three NBA games are out (AM14).
    # The difference is asserted; 0.15 only bounds its size.
    s = builds(name).stats()
    assert round(s["n_sd_pop"], 1) == 35.6 != 35.5
    assert round(s["rate_sd_pop"], 1) in (23.7, 23.8) and round(s["rate_sd_pop"], 2) < 23.8
    for key in ("n_sd_pop", "n_sd_sample"):
        assert abs(s[key] - 35.5) <= 0.15, (key, s[key])
    for key in ("rate_sd_pop", "rate_sd_sample"):
        assert abs(s[key] - 23.8) <= 0.15, (key, s[key])


def test_recorded_sd_values(builds):
    # exact values behind the previous test (2 decimals)
    g = builds("h1209_nba_games").stats()
    p = builds("h1209_nba_participants").stats()
    assert (round(g["n_sd_pop"], 2), round(g["n_sd_sample"], 2)) == (35.59, 35.61)
    assert (round(g["rate_sd_pop"], 2), round(g["rate_sd_sample"], 2)) == (23.72, 23.73)
    assert (round(p["n_sd_pop"], 2), round(p["n_sd_sample"], 2)) == (35.61, 35.63)
    assert (round(p["rate_sd_pop"], 2), round(p["rate_sd_sample"], 2)) == (23.75, 23.76)
    # the stated-filters build (1222 cases) is the one that matches both SDs
    s2 = builds("s2_stated_filters_only").stats()
    assert (round(s2["n_sd_pop"], 1), round(s2["rate_sd_pop"], 1)) == (35.5, 23.8)
    assert round(s2["rate_min"], 2) == 1.64                    # but not the 2.7% minimum


def test_waiver_build_statistics_are_recorded(builds):
    # the SDs of the waiver builds also differ from both printed SDs (same family of builds)
    for name in WAIVER:
        s = builds(name).stats()
        assert round(s["n_mean"]) == 70 and s["n_min"] == 10 and s["n_max"] == 129
        assert round(s["rate_min"], 1) == 2.7 and round(s["rate_max"], 1) == 98.9
        assert round(s["n_sd_pop"], 1) == 35.6 and round(s["rate_sd_pop"], 1) != 23.8


# ---- other scenarios: what each switch does to the counts ----------------------

EXPECTED_COUNTS = {
    # the plan's defaults: Martinie consistent-only loses item 50 (100%)
    "plan_default": (499, 396, 216, 110),
    # the stated 0%/100% filter alone: 396, 216, 110
    "s2_stated_filters_only": (500, 396, 216, 110),
    "s1_votes_only": (500, 396, 216, 110),
    "h1209_nba_games": (500, 393, 212, 104),
    "h1209_nba_participants": (500, 393, 212, 104),
    "h1209_revised_files": (500, 393, 212, 104),
    "h1209_revised_name_loose": (500, 393, 212, 105),
    "s4_martinie_consistent_only": (499, 393, 212, 104),
    "s5_nfl_votes_only": (500, 393, 212, 104),
    "h1209_no_nba_exclusion": (500, 396, 212, 104),
    "h1209_nfl_exact_prefix": (500, 393, 207, 104),
    "h1209_elections_binary": (500, 393, 212, 102),
    "h1209_elections_two_option_races": (500, 393, 212, 60),
    "h1209_min_n_10": (500, 393, 212, 104),
    # the builds used for the model runs (not defaults)
    "waiver_main": (500, 393, 212, 104),
    "waiver_politics_alt": (500, 393, 212, 104),
}


def test_every_scenario_in_the_yaml_is_covered():
    assert set(B.scenario_names()) == set(EXPECTED_COUNTS)


@pytest.fixture(scope="module")
def table():
    return {r["scenario"]: r for r in B.scenario_table()}


@pytest.mark.parametrize("name", sorted(EXPECTED_COUNTS))
def test_scenario_counts(table, name):
    r = table[name]
    assert tuple(r[d] for d in ORDER) == EXPECTED_COUNTS[name]
    assert r["n_cases"] == sum(EXPECTED_COUNTS[name])


def test_scenario_statistics_match_the_raw_file_counts(table):
    # scenarios S1-S5 (N mean, N max, min %)
    def row(name):
        r = table[name]
        return (round(r["n_mean"], 2), r["n_max"], round(r["rate_min"], 2))
    assert row("s1_votes_only") == (76.04, 161, 1.64)
    assert row("s2_stated_filters_only") == (69.76, 129, 1.64)
    assert row("h1209_nba_games") == (69.66, 129, 2.70)
    assert row("s4_martinie_consistent_only") == (65.38, 129, 2.70)
    assert row("s5_nfl_votes_only")[1:] == (161, 2.70)
    assert all(r["n_min"] == 10 for r in table.values())
    assert all(round(r["rate_max"], 2) == 98.95 for r in table.values())


def test_format_table_has_one_line_per_scenario(table):
    text = B.format_table(list(table.values()))
    assert len(text.splitlines()) == 3 + len(table)


# ---- filter log ----------------------------------------------------------------

def test_filter_log_names_the_dropped_cases(builds):
    log = builds("h1209_nba_games").log
    assert log["General"]["cases_dropped"] == {}
    assert log["NFL"]["cases_dropped"] == {"nfl_header_tokens_mismatch": [
        "W1_HOU_NE", "W6_BAL_TEN", "W12_ATL_NO", "W12_PIT_DEN"]}
    assert log["Politics"]["cases_dropped"] == {"elections_header_name_mismatch": [
        "TX_sen", "MT_atlarge", "CA_45th", "NY_sen", "IL_gub", "SC_gub"]}
    nba = log["NBA"]["cases_dropped"]
    assert sorted(nba["nba_named_game"]) == ["W2_10", "W2_17", "W2_9"]
    assert len(nba["0_percent_correct"]) == 17 and len(nba["100_percent_correct"]) == 52
    assert log["NBA"]["cases_in_file"] == 465 and log["NBA"]["cases_kept"] == 393
    # Step D2: 7,583 + 56 + 13 NFL votes lack confidence or "others %"
    #, minus those in the 4 dropped games
    assert log["NFL"]["responses_dropped"]["incomplete"] > 7000
    # original NBA file: one vote coded 0; 27 responses without "others %"
    assert log["NBA"]["responses_dropped"] == {"vote_not_an_answer": 1, "incomplete": 27}


def test_participant_variant_sends_the_three_games_to_the_0_percent_filter(builds):
    nba = builds("h1209_nba_participants").log["NBA"]["cases_dropped"]
    assert "nba_named_game" not in nba
    assert len(nba["0_percent_correct"]) == 20 and len(nba["100_percent_correct"]) == 52
    assert {"W2_9", "W2_10", "W2_17"} <= set(nba["0_percent_correct"])


def test_plan_default_loses_one_martinie_item(builds):
    assert builds("plan_default").log["General"]["cases_dropped"] == {
        "100_percent_correct": ["50"]}


def test_min_n_10_changes_nothing(builds):
    a, b = builds("h1209_nba_games"), builds("h1209_min_n_10")
    assert [r["case_id"] for r in a.records] == [r["case_id"] for r in b.records]


# ---- order (Step D6, OQ16) -----------------------------------------------------

@pytest.mark.parametrize("name", H1209 + WAIVER)
def test_domain_order_and_fold_boundaries(builds, name):
    # plan §2.1: General rows 0-499, NBA 500-892, NFL 893-1104, Politics 1105-1208
    b = builds(name)
    domains = [c.domain for c in b.cases]
    assert domains == (["General"] * 500 + ["NBA"] * 393 + ["NFL"] * 212
                       + ["Politics"] * 104)
    assert [r["index"] for r in b.records] == list(range(1209))
    assert [r["case_id"] for r in b.records] == [c.case_id for c in b.cases]


def test_within_domain_order_is_source_file_order(builds):
    b = builds("h1209_nba_games")
    ids = [c.case_id for c in b.cases]
    assert ids[:500] == [f"General:{i}" for i in range(1, 501)]
    assert ids[500] == "NBA:W9_9"              # NBA file starts with week 9
    assert ids[893] == "NFL:W1_ATL_PHI"
    assert ids[1105] == "Politics:FL_sen"
    weeks = [int(i.split(":W")[1].split("_")[0]) for i in ids[893:1105]]
    assert weeks == sorted(weeks)              # NFL weeks 1..15 in order


# ---- cases ---------------------------------------------------------------------

@pytest.mark.parametrize("name", H1209 + WAIVER)
def test_cases_are_valid_and_non_degenerate(builds, name):
    b = builds(name)
    for case, rec in zip(b.cases, b.records):
        assert isinstance(case, Case)
        k = sum(r.v == case.correct for r in case.responses)
        assert (case.n, k) == (rec["n"], rec["n_correct"])
        assert 0 < k < case.n
        for r in case.responses:
            assert isinstance(r.c, float) and 0.0 <= r.c <= 1.0
            assert isinstance(r.ps, tuple) and len(r.ps) == len(case.answers)
            if len(case.answers) == 2:
                assert abs(sum(r.ps) - 1.0) < 1e-9
    n_answers = {len(c.answers) for c in b.cases if c.domain != "Politics"}
    assert n_answers == {2}


def test_politics_is_not_all_binary_in_the_h1209_build(builds):
    # plan/data conflict, reported, not resolved
    # here. Answer set = the codes that occur in the workbook (default).
    b = builds("h1209_nba_games")
    sizes = [len(c.answers) for c in b.cases if c.domain == "Politics"]
    assert (sizes.count(2), sizes.count(3), sizes.count(4)) == (68, 35, 1)
    # with the options offered in the survey (.qsf): 8 more three-answer
    # cases, in which nobody chose the third option
    q = B.build(scenario="h1209_nba_games", switches={"elections_answer_set": "offered_qsf"})
    sizes = [len(c.answers) for c in q.cases if c.domain == "Politics"]
    assert (sizes.count(2), sizes.count(3), sizes.count(4)) == (60, 43, 1)
    assert [c.n for c in q.cases] == [c.n for c in b.cases]
    # multi-answer predicted support: complement (default) vs even split
    multi = next(c for c in b.cases if len(c.answers) == 3)
    r = multi.responses[0]
    x = r.ps[multi.answers.index(r.v)]
    assert sorted(r.ps) == pytest.approx(sorted([x, 1 - x, 1 - x]))
    e = B.build(scenario="h1209_nba_games",
                switches={"elections_ps_other_answers": "split_evenly"})
    multi_e = next(c for c in e.cases if c.case_id == multi.case_id)
    assert sum(multi_e.responses[0].ps) == pytest.approx(1.0)
    bb = builds("h1209_elections_binary")
    assert {len(c.answers) for c in bb.cases} == {2}
    assert bb.log["Politics"]["cases_dropped"]["winner_not_among_answers"] == [
        "MS_sen1", "ME_sen"]


# ---- waiver scenarios (hypotheses) ---------------------------------------------

def test_waiver_main_politics_is_binary_minus_four_pred_agree_races(builds):
    b = builds("waiver_main")
    assert {len(c.answers) for c in b.cases} == {2}
    assert b.log["Politics"]["cases_dropped"] == {
        "elections_pred_agree_name_mismatch": ["TX_sen", "CA_45th", "IL_gub", "SC_gub"],
        "winner_not_among_answers": ["MS_sen1", "ME_sen"]}
    # 151 votes for options 3-4 in the file; those in the 6 dropped races
    # go with the race, the rest are dropped as votes
    assert b.log["Politics"]["responses_dropped"]["vote_not_an_answer"] > 0
    assert sorted(b.log["NBA"]["cases_dropped"]["nba_named_game"]) == [
        "W2_10", "W2_17", "W2_9"]
    assert b.log["NFL"]["cases_dropped"] == {"nfl_header_tokens_mismatch": [
        "W1_HOU_NE", "W6_BAL_TEN", "W12_ATL_NO", "W12_PIT_DEN"]}
    sw = b.switches
    assert sw["rutchick_version"] == "original"
    assert sw["nfl_original_conf"] == "recode_to_revised"      # the 1-5 recode
    assert sw["elections_options"] == "binary_drop_other_votes"
    assert sw["elections_column_matching"] == "pred_agree_names"
    assert sw["answer_coding_by_domain"] == {}


def test_waiver_politics_alt_differs_from_waiver_main_in_politics_only(builds):
    a, b = builds("waiver_main"), builds("waiver_politics_alt")
    assert {len(c.answers) for c in b.cases} == {2}
    assert sorted(b.log["Politics"]["cases_dropped"]["elections_third_option_votes_at_least_k"]) \
        == sorted(["MS_sen1", "ME_sen", "NM_sen", "WI_1", "TX_32", "AK_gub"])
    assert set(b.log["Politics"]["cases_dropped"]) == {"elections_third_option_votes_at_least_k"}
    assert a.cases[:1105] == b.cases[:1105]                    # General, NBA, NFL identical
    ida = {c.case_id for c in a.cases[1105:]}
    idb = {c.case_id for c in b.cases[1105:]}
    assert len(ida) == len(idb) == 104 and len(ida ^ idb) == 8  # 4 races swapped for 4
    diff = {k: (a.switches[k], b.switches[k]) for k in a.switches
            if a.switches[k] != b.switches[k]}
    assert diff == {"elections_column_matching": ("pred_agree_names", "positional"),
                    "elections_drop_third_option_races_min_votes": (None, 6)}


def test_waiver_general_nba_nfl_equal_the_h1209_build(builds):
    # "General as now; NBA minus the three 1/61 games; NFL with the 1-5 recode"
    a, h = builds("waiver_main"), builds("h1209_nba_games")
    assert a.cases[:1105] == h.cases[:1105]


def test_third_option_threshold_switch():
    counts = {k: B.build(scenario="s2_stated_filters_only", make_cases=False, switches={
        "elections_drop_third_option_races_min_votes": k}).counts()["Politics"]
        for k in (5, 6, 7)}
    assert counts == {5: 103, 6: 104, 7: 106}          # AM14
    with pytest.raises(ValueError):
        B.build(switches={"elections_drop_third_option_races_min_votes": 0})


def test_every_scenario_is_labelled_a_hypothesis_and_none_is_a_default():
    config = B.load_scenarios()
    for name in B.scenario_names(config):
        info = B.scenario_info(name, config)
        assert "hypothesis" in info["status"], name
        assert info["description"], name
    for name in WAIVER:
        assert "rule-baseline differences documented" in B.scenario_info(name)["status"]
    assert "not determined by the rule-baseline check" in \
        B.scenario_info("waiver_main")["description"]
    # the defaults are still the plan's values: no scenario was made the default
    d = B.default_switches()
    assert d["elections_options"] == "all_offered_options"
    assert d["elections_column_matching"] == "positional"
    assert d["nba_extra_exclusion"] == "none" and d["nfl_column_matching"] == "positional"
    assert d["elections_drop_third_option_races_min_votes"] is None
    assert d["nfl_original_conf"] == "recode_to_revised"
    assert d["elections_ps_other_answers"] == "complement_each"
    assert d["elections_answer_set"] == "observed_codes"
    # no base scenario picks a 1-7 reading of the NFL confidence
    for name in B.scenario_names(config):
        assert B.scenario_switches(name, config)["nfl_original_conf"] == "recode_to_revised"
    text = B.SCENARIOS_PATH.read_text(encoding="utf-8")
    for switch in ("nfl_column_matching", "elections_column_matching",
                   "nba_extra_exclusion", "elections_drop_third_option_races_min_votes"):
        line = next(l for l in text.splitlines() if l.strip().startswith(switch + ":"))
        assert "[HYPOTHESIS]" in line, switch


def test_old_scenario_names_are_deprecated_aliases(builds):
    assert B.resolve_scenario("best_evidence_nba_games") == "h1209_nba_games"
    assert B.resolve_scenario("waiver_main") == "waiver_main"
    assert not any(n.startswith("best_evidence") for n in B.scenario_names())
    assert B.scenario_switches("best_evidence_nba_games") == B.scenario_switches("h1209_nba_games")
    b = B.build(scenario="best_evidence_min_n_10", make_cases=False)
    assert b.name == "h1209_min_n_10"


def test_answer_coding_per_domain(builds):
    base = builds("waiver_main")
    b = B.build(scenario="waiver_main",
                switches={"answer_coding_by_domain": {"NBA": "b_raw_file_option_order"}})
    first = lambda build, d: {c.answers for c in build.cases if c.domain == d}
    assert first(base, "NBA") == {("home", "away")}
    assert first(b, "NBA") == {("away", "home")}               # raw code 1 first
    assert [c for c in b.cases if c.domain != "NBA"] == [
        c for c in base.cases if c.domain != "NBA"]
    with pytest.raises(ValueError):
        B.build(switches={"answer_coding_by_domain": {"NBA": "home_first"}})
    with pytest.raises(ValueError):
        B.build(switches={"answer_coding_by_domain": {"Hockey": "b_raw_file_option_order"}})


def test_votes_only_scenario_has_statistics_but_no_cases(builds):
    b = builds("s5_nfl_votes_only")
    assert b.cases is None and len(b.records) == 1209
    with pytest.raises(ValueError):
        B.freeze(b)


# ---- switches ------------------------------------------------------------------

def test_scenario_switches_are_complete_and_checked():
    sw = B.scenario_switches("h1209_nba_games")
    assert set(sw) == set(B.default_switches())
    assert sw["martinie_confidence"] == "c_p_if_true_else_100_minus_p"
    assert sw["martinie_drop_vote_inconsistent"] is False
    assert sw["rutchick_version"] == "original"
    plan = B.default_switches()
    assert plan["martinie_confidence"] == "a_abs_p_minus_50_over_50"   # plan OQ2
    assert plan["likert_confidence"] == "a_c_minus_1_over_4"
    assert plan["martinie_drop_vote_inconsistent"] is True              # plan OQ4
    assert plan["rutchick_version"] == "revised"                        # plan OQ44
    assert plan["min_n"] is None and plan["nfl_source"] == "rutchick_s1"
    assert plan["domain_order"] == ORDER                                # plan OQ16
    with pytest.raises(KeyError):
        B.build(switches={"no_such_switch": 1})
    with pytest.raises(ValueError):
        B.build(switches={"rutchick_version": "draft"})


def test_lee_as_nfl_source_is_refused_without_outcomes():
    # raw files: the Lee folder holds no outcomes; none is invented
    with pytest.raises(lee.LeeOutcomesMissing):
        B.build(switches={"nfl_source": "lee2018"})


# ---- freeze --------------------------------------------------------------------

def test_freeze_round_trip_manifest_and_hash(builds, tmp_path):
    b = builds("h1209_nba_games")
    out = B.freeze(b, tmp_path / "a")
    cases, manifest = B.load_frozen(out)
    assert cases == b.cases                       # exact, floats included
    assert manifest["switches"] == b.switches
    assert set(manifest["switches"]) == set(B.default_switches())
    assert manifest["counts"] == {"General": 500, "NBA": 393, "NFL": 212, "Politics": 104}
    assert manifest["n_cases"] == 1209
    assert manifest["n_responses"] == sum(c.n for c in b.cases)
    assert manifest["statistics"]["all"]["n_max"] == 129
    assert set(manifest["statistics"]) == {"all", *ORDER}
    assert manifest["filter_log"]["NBA"]["cases_kept"] == 393
    # 6 Martinie files + 3 Rutchick workbooks + the election .qsf
    assert len(manifest["inputs"]) == 6 + 3 + 1
    assert any(k.endswith("Real_SP_Election.qsf") for k in manifest["inputs"])
    assert all(len(h) == 64 for h in manifest["inputs"].values())
    # status and rule-gate result travel with the build
    assert manifest["status"] == B.scenario_info("h1209_nba_games")["status"]
    assert "hypothesis" in manifest["status"] and manifest["description"]
    gate = manifest["rule_gate"]
    assert gate["passes"] is False and gate["verdict"] == "NOT MET"
    assert gate["tie_policy"] == "canonical_first" and gate["da_tie_rule"] == "not_mr"
    assert gate["n_cases"] == gate["n_cases_thesis"]
    assert len([1 for m in gate["per_domain"] for d in gate["per_domain"][m]]) == 20
    cell = gate["per_domain"]["SP"]["General"]
    assert cell == {"recomputed": 413, "thesis": 413, "difference": 0}
    assert gate["da_depends_on_oq8b"]["n_cases"] > 0           # multi-option Politics
    assert "JSON lines" in manifest["format"]
    # deterministic: a second freeze gives the same data hash
    out2 = B.freeze(b, tmp_path / "b")
    m2 = json.loads((out2 / B.MANIFEST_FILE).read_text())
    assert m2["files"] == manifest["files"]
    # a changed file is detected
    with open(out2 / B.CASES_FILE, "a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(ValueError):
        B.load_frozen(out2)
    assert not any(p.suffix in (".xlsx", ".csv") and "worldA_case_table" not in p.name
                   for p in out.iterdir())


def test_waiver_manifest_carries_the_scenario_status(builds, tmp_path):
    out = B.freeze(builds("waiver_main"), tmp_path / "w")
    cases, manifest = B.load_frozen(out)
    assert "rule-baseline differences documented" in manifest["status"]
    assert manifest["rule_gate"]["passes"] is False
    assert manifest["rule_gate"]["domains_passing"] == ["Politics"]
    assert manifest["rule_gate"]["da_depends_on_oq8b"]["n_cases"] == 0   # all binary
    assert cases == builds("waiver_main").cases
