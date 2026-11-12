"""A10 acceptance (plan §9.3 row A10; plan §2.1.1 Step D5): the rule gate.

Thesis-reported values: T3b per domain (20 counts, +-1), T3 864/831/830/774/372 of 1209,
T9 union 1007 / DA-only 202 / DA-and-rules 170.

Part 1: unit tests of the gate logic on small hand-made cases (no data).
Part 2: data tests (skipped without the raw data) on three scenarios, all
HYPOTHESES (none is established as the thesis data set):
  h1209_nba_games       the three count hypotheses (formerly
                        `best_evidence_nba_games`);
  waiver_main           the build used for the model runs;
  waiver_politics_alt   its Politics sensitivity build.
Every thesis-reported count is encoded for each of them. A count that a
scenario does not reach is marked xfail(strict=True) with the measured difference, so a
comparison is never loosened silently and a change of any count (better or
worse) shows up as a test failure.

STATE OF THE CHECK (docs/RULE_BASELINE_CHECK.md): on every scenario some
rule-baseline counts differ from the thesis-reported counts. General
equals the thesis in MR/WC/HAC/SP. NBA and NFL differ in every scenario.
With Politics treated as binary (waiver builds) the Politics domain is
within +-1, but 74% of all 4-race removals do that, so it identifies no
race set. The model runs use `waiver_main` with the differences documented
(config/gate_status.yaml, AM17).
"""
import pytest

from cdm import aggregators as A
from cdm.config import data_root
from cdm.eval import rule_gate as G
from cdm.schema import Case, Response


# ---- helpers -------------------------------------------------------------------

def resp(v, c, ps_first, answers=("a1", "a2")):
    """Binary response: ps = (ps_first, 1 - ps_first) in the order of answers."""
    return Response(v=v, c=c, ps=(ps_first, 1.0 - ps_first))


def case(responses, correct="a1", domain="General", case_id="c", answers=("a1", "a2")):
    return Case(case_id=case_id, answers=answers, correct=correct,
                responses=tuple(responses), domain=domain)


def fake_result(rows):
    """GateResult from (domain, {method: credit}) rows, for count logic."""
    recs = []
    for i, (domain, credit) in enumerate(rows):
        choice = {m: ("ok" if credit[m] else "bad") for m in G.METHODS}
        recs.append({"case_id": f"{domain}:{i}", "domain": domain, "n": 10,
                     "correct": "ok", "tied": {m: (choice[m],) for m in A.RULES},
                     "choice": choice, "credit": {m: float(credit[m]) for m in G.METHODS},
                     "sp_half": False,
                     "range": {m: (float(credit[m]), float(credit[m])) for m in G.METHODS}})
    return G.GateResult(recs, "canonical_first", 1e-9)


def credits(sp, mr, wc, hac, da):
    return {"SP": sp, "MR": mr, "WC": wc, "HAC": hac, "DA": da}


# ---- Part 1a: the thesis-reported counts are internally consistent (plan §1.1) -----------------

def test_t3b_columns_sum_to_t3():
    for m in G.METHODS:
        assert sum(G.THESIS_T3B[m].values()) == G.THESIS_T3[m]
    assert sum(G.THESIS_N.values()) == G.THESIS_N_CASES == 1209


def test_t3b_values_are_the_plan_values():
    order = ("General", "NBA", "NFL", "Politics")
    assert [G.THESIS_T3B["SP"][d] for d in order] == [413, 258, 109, 84]
    assert [G.THESIS_T3B["MR"][d] for d in order] == [385, 252, 111, 83]
    assert [G.THESIS_T3B["WC"][d] for d in order] == [380, 253, 114, 83]
    assert [G.THESIS_T3B["HAC"][d] for d in order] == [338, 237, 131, 68]
    assert [G.THESIS_T3B["DA"][d] for d in order] == [108, 142, 101, 21]
    assert G.THESIS_T3 == {"SP": 864, "MR": 831, "WC": 830, "HAC": 774, "DA": 372}


def test_t9_arithmetic():
    t = G.THESIS_T9
    assert t == {"union_rules": 1007, "da_only": 202, "da_and_rules": 170}
    assert t["union_rules"] + t["da_only"] == 1209            # 83.3% -> 100%
    assert t["da_only"] + t["da_and_rules"] == G.THESIS_T3["DA"]
    assert round(t["union_rules"] / 1209, 3) == 0.833


def test_tolerances_are_the_plan_tolerances():
    assert G.TOL_T3B == 1               # Step D5: "+-1 case per cell for ties"
    assert G.TOL_GLOBAL_FROM_DOMAINS == 4


# ---- Part 1b: one case ------------------------------------------------------------

def test_evaluate_case_plain_binary():
    # 3 vote a1 (c .9,.8,.7), 2 vote a2 (c .2,.1). Everyone predicts 80% for a1.
    # MR a1 (3 v 2); WC a1 (2.4 v 0.3); HAC a1 (.8 v .15);
    # SP: S(a1) - mean ps[a1] = .6 - .8 < 0, S(a2) - .2 = .2 > 0 -> a2.
    c = case([resp("a1", .9, .8), resp("a1", .8, .8), resp("a1", .7, .8),
              resp("a2", .2, .8), resp("a2", .1, .8)], correct="a1")
    r = G.evaluate_case(c)
    assert r["choice"] == {"MR": "a1", "WC": "a1", "HAC": "a1", "SP": "a2", "DA": "a2"}
    assert r["credit"] == {"SP": 0.0, "MR": 1.0, "WC": 1.0, "HAC": 1.0, "DA": 0.0}
    assert all(len(t) == 1 for t in r["tied"].values())
    assert r["range"]["MR"] == (1.0, 1.0) and r["range"]["DA"] == (0.0, 0.0)
    assert r["n"] == 5 and r["domain"] == "General"


def test_da_examples_of_th_p35_through_the_gate():
    # (MR,WC,HAC,SP) = (a1,a1,a2,a1) -> DA a2: two a1 voters with low
    # confidence, one a2 voter with high confidence, ps makes SP = a1.
    c = case([resp("a1", .3, .5), resp("a1", .3, .5), resp("a2", .5, .5)], correct="a2")
    r = G.evaluate_case(c)
    assert (r["choice"]["MR"], r["choice"]["WC"], r["choice"]["HAC"],
            r["choice"]["SP"]) == ("a1", "a1", "a2", "a1")
    assert r["choice"]["DA"] == "a2" and r["credit"]["DA"] == 1.0
    # (a1,a1,a2,a2) -> DA a2 (tie -> not MR): same votes, SP now a2.
    c = case([resp("a1", .3, .9), resp("a1", .3, .9), resp("a2", .5, .9)], correct="a1")
    r = G.evaluate_case(c)
    assert (r["choice"]["MR"], r["choice"]["WC"], r["choice"]["HAC"],
            r["choice"]["SP"]) == ("a1", "a1", "a2", "a2")
    assert r["choice"]["DA"] == "a2" and r["credit"]["DA"] == 0.0


def test_all_four_rules_wrong_means_da_right():
    c = case([resp("a1", .9, .5), resp("a1", .9, .5), resp("a2", .1, .5)], correct="a2")
    r = G.evaluate_case(c)
    assert [r["credit"][m] for m in A.RULES] == [0.0] * 4
    assert r["credit"]["DA"] == 1.0


def mr_tie_case(correct):
    # 2 v 2 votes. a2 has the higher confidence -> WC a2, HAC a2.
    # SP: S = .5 each; mean ps[a1] = .4 -> SP a1 (.1 v -.1).
    return case([resp("a1", .2, .4), resp("a1", .2, .4),
                 resp("a2", .9, .4), resp("a2", .9, .4)], correct=correct)


def test_mr_tie_is_logged_and_broken_by_policy():
    r1 = G.evaluate_case(mr_tie_case("a1"), "canonical_first")
    r2 = G.evaluate_case(mr_tie_case("a1"), "canonical_second")
    assert r1["tied"]["MR"] == ("a1", "a2") == r2["tied"]["MR"]
    assert r1["choice"]["MR"] == "a1" and r2["choice"]["MR"] == "a2"
    assert r1["credit"]["MR"] == 1.0 and r2["credit"]["MR"] == 0.0
    assert r1["range"]["MR"] == (0.0, 1.0)            # depends on the tie rule
    assert r1["range"]["WC"] == (0.0, 0.0)            # no tie, wrong
    # DA follows MR: (a1,a2,a2,a1) -> 2-2 -> not MR = a2; (a2,a2,a2,a1) -> a1.
    assert r1["choice"]["DA"] == "a2" and r2["choice"]["DA"] == "a1"
    assert r1["range"]["DA"] == (0.0, 1.0)


def sp_tie_case(correct="a1"):
    # 1 v 1 votes, ps[a1] = .5 for both -> SP scores 0 and 0 (tie); MR tie too.
    return case([resp("a1", .9, .5), resp("a2", .1, .5)], correct=correct)


def test_sp_tie_half_credit():
    r = G.evaluate_case(sp_tie_case(), "sp_tie_half_credit")
    assert r["tied"]["SP"] == ("a1", "a2")
    assert r["credit"]["SP"] == 0.5 and r["sp_half"] is True
    # MR/WC/HAC keep the canonical-first answer; DA is built on the
    # canonical-first SP answer (documented choice of the gate).
    assert r["choice"]["MR"] == "a1" and r["choice"]["SP"] == "a1"
    assert r["credit"]["MR"] == 1.0
    plain = G.evaluate_case(sp_tie_case(), "canonical_first")
    assert plain["credit"]["SP"] == 1.0 and plain["sp_half"] is False
    assert r["choice"]["DA"] == plain["choice"]["DA"]


def test_sp_tie_half_credit_needs_a_binary_case():
    c = Case(case_id="m", answers=("a1", "a2", "a3"), correct="a1", domain="Politics",
             responses=(Response("a1", .5, (.5, .5, 0.0)), Response("a2", .5, (.5, .5, 0.0))))
    with pytest.raises(NotImplementedError):
        G.evaluate_case(c, "sp_tie_half_credit")


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        G.evaluate_case(sp_tie_case(), "random")


def test_gate_choices_equal_the_aggregators():
    for c in (mr_tie_case("a1"), sp_tie_case(), case(
            [resp("a1", .9, .8), resp("a2", .2, .8), resp("a2", .1, .3)])):
        for policy in ("canonical_first", "canonical_second"):
            assert G.evaluate_case(c, policy)["choice"] == A.aggregate_all(
                c.responses, c.answers, policy=policy)


def test_three_answer_case_uses_oq8b_and_never_crashes():
    c = Case(case_id="m", answers=("a1", "a2", "a3"), correct="a3", domain="Politics",
             responses=(Response("a1", .5, (.6, .2, .2)), Response("a1", .5, (.6, .2, .2)),
                        Response("a2", .9, (.2, .6, .2)), Response("a3", .1, (.1, .1, .8))))
    r = G.evaluate_case(c)
    # MR a1, WC a1 (1.0 v .9 v .1), HAC a2, SP: a1 .5-.375, a2 .25-.275, a3 .25-.35 -> a1
    assert (r["choice"]["MR"], r["choice"]["WC"], r["choice"]["HAC"],
            r["choice"]["SP"]) == ("a1", "a1", "a2", "a1")
    assert r["choice"]["DA"] == "a3"        # never chosen, count 0
    assert r["credit"]["DA"] == 1.0


# ---- Part 1b': DA tie-rule switch and the OQ8b caveat ----------------------------

def test_da_tie_rule_default_is_the_thesis_text():
    from cdm.config import load_config
    assert A.DA_TIE_RULES == ("not_mr", "first_answer", "second_answer")
    assert A.DEFAULT_DA_TIE_RULE == "not_mr"
    assert load_config()["da_tie_rule"] == "not_mr"
    assert A.da_tie_rule_from_config() == "not_mr"
    assert A.da_tie_rule_from_config({}) == "not_mr"
    assert A.da_tie_rule_from_config({"da_tie_rule": "first_answer"}) == "first_answer"
    with pytest.raises(ValueError):
        A.da_tie_rule_from_config({"da_tie_rule": "coin"})
    with pytest.raises(ValueError):
        A.da_from_choices({m: "a1" for m in A.RULES}, ("a1", "a2"), tie_rule="coin")
    assert G.run_gate([mr_tie_case("a1")]).da_tie_rule == "not_mr"


def test_da_tie_rule_positional_variants_differ_only_on_a_split():
    ans = ("a1", "a2")
    split_mr_a1 = {"MR": "a1", "WC": "a1", "HAC": "a2", "SP": "a2"}   # TH p.35-36 example
    split_mr_a2 = {"MR": "a2", "WC": "a1", "HAC": "a1", "SP": "a2"}
    three_one = {"MR": "a1", "WC": "a1", "HAC": "a2", "SP": "a1"}
    # thesis text: the answer not chosen by MR
    assert A.da_from_choices(split_mr_a1, ans) == "a2"
    assert A.da_from_choices(split_mr_a2, ans) == "a1"
    # sensitivity options: position in the canonical answer order
    assert A.da_from_choices(split_mr_a1, ans, tie_rule="first_answer") == "a1"
    assert A.da_from_choices(split_mr_a2, ans, tie_rule="first_answer") == "a1"
    assert A.da_from_choices(split_mr_a1, ans, tie_rule="second_answer") == "a2"
    assert A.da_from_choices(split_mr_a2, ans, tie_rule="second_answer") == "a2"
    # the thesis's only worked tie example (MR a1 -> a2) cannot separate
    # "not MR" from "the second answer"
    assert (A.da_from_choices(split_mr_a1, ans)
            == A.da_from_choices(split_mr_a1, ans, tie_rule="second_answer"))
    for rule in A.DA_TIE_RULES:                       # no tie: the rule is not used
        assert A.da_from_choices(three_one, ans, tie_rule=rule) == "a2"
    d = A.da_detail(split_mr_a1, ans, tie_rule="first_answer")
    assert d["positional_tie_break"] and not d["mr_tie_break"] and d["tie_rule"] == "first_answer"
    d = A.da_detail(split_mr_a1, ans)
    assert d["mr_tie_break"] and not d["positional_tie_break"]


def test_da_tie_rule_reaches_the_gate_and_the_instances():
    from cdm.approaches import instances as I
    # rules (a1, a2, a2, a1): a 2-2 split with MR = a1
    cases = [Case(case_id=f"g{i}", answers=c.answers, correct=c.correct,
                  responses=c.responses, domain="General")
             for i, c in enumerate([mr_tie_case("a1"), mr_tie_case("a2")])]
    text = G.run_gate(cases, policy="canonical_first", tol=1e-9)
    first = G.run_gate(cases, policy="canonical_first", tol=1e-9, da_tie_rule="first_answer")
    assert [r["choice"]["DA"] for r in text.records] == ["a2", "a2"]
    assert [r["choice"]["DA"] for r in first.records] == ["a1", "a1"]
    assert first.da_tie_rule == "first_answer"
    assert "SENSITIVITY OPTION" in G.format_table(first, title="x")
    assert "thesis text" in G.format_table(text, title="x")
    for rule in A.DA_TIE_RULES:
        res = G.run_gate(cases, policy="canonical_first", tol=1e-9, da_tie_rule=rule)
        assert res.count("DA", "General") == G.exploratory_da_count(
            cases, text, "General", rule)
        inst = I.build_instances(cases, da_tie_rule=rule)
        assert inst.da() == [r["choice"]["DA"] for r in res.records]
        assert inst.options["da_tie_rule"] == rule
    assert I.build_instances(cases).options["da_tie_rule"] == "not_mr"


def test_oq8b_caveat_is_reported_for_multi_option_cases_only():
    # three answers; MR a1, WC a1, HAC a2 (0.9), SP a1; a3 and a4 never
    # chosen -> DA must pick between two unchosen answers (OQ8b)
    c4 = Case(case_id="m", answers=("a1", "a2", "a3", "a4"), correct="a4", domain="Politics",
              responses=(Response("a1", .5, (.6, .2, .1, .1)), Response("a1", .5, (.6, .2, .1, .1)),
                         Response("a2", .9, (.2, .6, .1, .1)), Response("a3", .1, (.1, .1, .7, .1))))
    r = G.evaluate_case(c4)
    assert r["choice"]["DA"] == "a4"              # lowest support among a3, a4 (plan OQ8b)
    assert r["da_oq8b"] == (0.0, 1.0)             # a3 would have been wrong
    res = G.run_gate([c4, Case(case_id="b", answers=("a1", "a2"), correct="a1",
                               domain="Politics", responses=sp_tie_case().responses)])
    o = res.da_oq8b("Politics")
    assert o["n_cases"] == 1 and (o["low"], o["count"], o["high"]) == (
        res.count("DA") - 1, res.count("DA"), res.count("DA"))
    assert "OQ8b caveat" in G.format_table(res)
    binary = G.run_gate([mr_tie_case("a1")])
    assert binary.da_oq8b()["n_cases"] == 0 and binary.records[0]["da_oq8b"] is None
    assert "OQ8b caveat" not in G.format_table(binary)


# ---- Part 1c: counts, T9, ties, comparison ---------------------------------------

def test_run_gate_counts_and_tie_log():
    cases = [mr_tie_case("a1"), mr_tie_case("a2"), sp_tie_case("a2")]
    cases = [Case(case_id=f"g{i}", answers=c.answers, correct=c.correct,
                  responses=c.responses, domain="NBA") for i, c in enumerate(cases)]
    res = G.run_gate(cases, policy="canonical_first", tol=1e-9)
    assert res.n_cases() == 3 and res.n_cases("NBA") == 3 and res.n_cases("NFL") == 0
    assert res.count("MR") == 1                       # a1, a1, a1 vs a1, a2, a2
    t = res.ties("MR", "NBA")
    assert t["n_tied"] == 3 and t["scored_correct"] == 1 and t["correct_among_tied"] == 3
    assert t["case_ids"] == ["g0", "g1", "g2"]
    assert res.tie_range("MR") == (0, 3)
    assert res.ties("SP")["n_tied"] == 1 and res.ties("WC")["n_tied"] == 0
    res2 = G.run_gate(cases, policy="canonical_second", tol=1e-9)
    assert res2.count("MR") == 2
    half = G.run_gate(cases, policy="sp_tie_half_credit", tol=1e-9)
    assert half.sp_half_credit_cases() == ["g2"]
    assert half.count("SP") == res.count("SP") + 0.5   # canonical first was wrong there
    with pytest.raises(ValueError):
        res.ties("DA")


def test_run_gate_defaults_come_from_the_config():
    res = G.run_gate([mr_tie_case("a1")])
    opts = A.options_from_config()
    assert res.policy == opts["policy"] == "canonical_first"
    assert res.tol == opts["tol"]


def test_t9_counts():
    rows = [("General", credits(1, 1, 1, 1, 0)),      # all rules right
            ("General", credits(0, 0, 0, 0, 1)),      # DA only
            ("NBA", credits(1, 0, 0, 0, 1)),          # DA and one rule
            ("NBA", credits(0, 1, 1, 1, 0)),
            ("NFL", credits(0, 0, 0, 0, 1))]          # DA only
    res = fake_result(rows)
    assert res.t9() == {"union_rules": 3, "da_only": 2, "da_and_rules": 1,
                        "covered_with_da": 5, "n_cases": 5}
    assert res.t9("NBA")["union_rules"] == 2


def result_at_targets(shift=None):
    """A fake result that hits every T3b cell (plus `shift` on given cells)."""
    shift = shift or {}
    rows = []
    for d in G.DOMAINS:
        n = G.THESIS_N[d]
        k = {m: G.THESIS_T3B[m][d] + shift.get((m, d), 0) for m in G.METHODS}
        for i in range(n):
            rows.append((d, {m: int(i < k[m]) for m in G.METHODS}))
    return fake_result(rows)


def test_compare_exact_match_passes():
    res = result_at_targets()
    rows = G.compare(res)
    assert all(r["delta"] == 0 and r["ok"] for r in rows if r["block"] in ("N", "T3b", "T3"))
    s = G.gate_summary(res)
    assert s["passes_t3b"] and s["passes_t3_exact"]
    assert s["cells_exact"] == 20 and s["domains_passing"] == list(G.DOMAINS)


def test_compare_tolerance_is_plus_minus_one_per_cell():
    res = result_at_targets({("MR", "NBA"): +1, ("HAC", "NFL"): -1})
    s = G.gate_summary(res)
    assert s["passes_t3b"] and s["cells_within_tolerance"] == 20 and s["cells_exact"] == 18
    assert not s["passes_t3_exact"]            # T3 is exact: 832 != 831
    t3 = {r["key"]: r for r in G.compare(res) if r["block"] == "T3"}
    assert t3["MR"]["delta"] == 1 and t3["MR"]["within_domain_tolerance"]
    res = result_at_targets({("SP", "Politics"): +2})
    s = G.gate_summary(res)
    assert not s["passes_t3b"]
    assert "Politics" not in s["domains_passing"] and len(s["domains_passing"]) == 3
    assert ("T3b", "SP", "Politics", 86, 84, 2) in s["outside_tolerance"]
    assert "**DIFF**" in G.format_table(res)


def test_domain_needs_the_thesis_number_of_cases():
    res = result_at_targets()
    res = G.without_cases(res, [res.records[-1]["case_id"]])    # a Politics case, all wrong
    assert res.n_cases("Politics") == 103
    assert not G.domain_passes(res, "Politics") and G.domain_passes(res, "NFL")
    assert not G.gate_summary(res)["passes_t3b"]
    with pytest.raises(KeyError):
        G.without_cases(res, ["no such case"])


def test_format_functions_run():
    res = result_at_targets({("DA", "General"): 3})
    text = G.format_table(res, title="x")
    assert "111 / 108 / +3 **DIFF**" in text and "union_rules" in text
    assert "0 tied" in G.format_ties(res)
    assert "19/20" in G.format_delta_table([("x", res)])


# ---- Part 1d: ladder definition and exploratory helpers ---------------------------

def test_ladder_follows_the_plan_order():
    steps = [s for s, _ in G.LADDER]
    assert [s.split()[0] for s in steps] == ["1", "2", "3", "4", "5"]
    assert "tie rule" in steps[0] and "confidence" in steps[1] and "Martinie" in steps[2]
    assert "minimum N" in steps[3] and "Rutchick revisions" in steps[4]
    # every variant changes exactly one thing: a tie policy or one switch group
    for _, variants in G.LADDER:
        for _, policy, overrides in variants:
            assert (policy is None) != (overrides == {})
    assert [p for _, p, _ in G.LADDER[0][1]] == ["canonical_second", "sp_tie_half_credit"]


def test_exploratory_removal_search():
    rows = ([("NFL", credits(1, 1, 0, 0, 0))] * 3 + [("NFL", credits(0, 0, 0, 0, 1))] * 2
            + [("NFL", credits(1, 0, 0, 0, 1))])
    res = fake_result(rows)                      # SP 4, MR 3

    def search(k, sp_target, mr_target):
        old = (G.THESIS_T3B["SP"]["NFL"], G.THESIS_T3B["MR"]["NFL"])
        G.THESIS_T3B["SP"]["NFL"], G.THESIS_T3B["MR"]["NFL"] = sp_target, mr_target
        try:
            return G.exploratory_removal_search(res, "NFL", k, ("MR", "SP"))
        finally:
            G.THESIS_T3B["SP"]["NFL"], G.THESIS_T3B["MR"]["NFL"] = old

    s = search(2, 2, 1)                          # remove two (MR 1, SP 1) cases
    assert s["feasible"] and s["solutions"] == [{(1, 1): 2}]
    assert s["needed"] == {"MR": 2, "SP": 2}
    s = search(2, 1, 1)                          # SP -3 with 2 removals: impossible
    assert not s["feasible"] and s["solutions"] == []
    s = search(1, 5, 3)                          # target above the count: impossible
    assert not s["feasible"]


def test_exploratory_da_count_default_rule_equals_the_gate():
    cases = [Case(case_id=f"g{i}", answers=c.answers, correct=c.correct,
                  responses=c.responses, domain="General")
             for i, c in enumerate([mr_tie_case("a1"), mr_tie_case("a2"), sp_tie_case()])]
    res = G.run_gate(cases, policy="canonical_first", tol=1e-9)
    assert G.exploratory_da_count(cases, res, "General", "not_mr") == res.count("DA", "General")
    # case 0 and 1: rules (a1, a2, a2, a1) -> 2-2. first_answer -> a1.
    first = G.exploratory_da_count(cases, res, "General", "first_answer")
    second = G.exploratory_da_count(cases, res, "General", "second_answer")
    assert first + second >= 2
    with pytest.raises(ValueError):
        G.exploratory_da_count(cases, res, "General", "random")
    with pytest.raises(ValueError):
        G.exploratory_da_count(cases[:2], res, "General", "not_mr")


def test_exploratory_complement_ps():
    c3 = Case(case_id="m", answers=("a1", "a2", "a3"), correct="a1", domain="Politics",
              responses=(Response("a1", .5, (.6, .2, .2)), Response("a3", .5, (.05, .05, .9))))
    c2 = sp_tie_case()
    out = G.exploratory_complement_ps([c3, c2])
    assert out[1] is c2
    assert out[0].responses[0].ps == pytest.approx((.6, .4, .4))
    assert out[0].responses[1].ps == pytest.approx((.1, .1, .9))
    assert out[0].responses[0].v == "a1" and out[0].correct == "a1"


# ---- Part 2: data tests (hypothesis scenarios) ------------------------------------

ROOT = data_root()
HAVE_DATA = all((ROOT / d / "extracted").is_dir() for d in (
    "Martinie2020_General", "Rutchick2020_NBA_NFL_Politics"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason=f"raw data not found under {ROOT}")

#: The three-count-hypotheses scenario (formerly `best_evidence_nba_games`).
BEST = "h1209_nba_games"
WAIVER_MAIN = "waiver_main"
WAIVER_ALT = "waiver_politics_alt"
SCENARIOS = (BEST, WAIVER_MAIN, WAIVER_ALT)

#: What each scenario gives (tie policy canonical_first, tol 1e-9, DA tie
#: rule = thesis text), measured on 2026-11-11. Order: General, NBA, NFL, Politics. General / NBA / NFL are the
#: same cases in all three scenarios.
#: Change against the 2026-11-10 numbers of h1209 (then best_evidence):
#: NFL HAC 132 -> 131 and global HAC 762 -> 761, because the NFL original
#: confidence is now the 1-5 recode (= REVISED) instead of the unsupported
#: 1-7 min-max. Nothing else moved.
MEASURED = {
    BEST: {"SP": (413, 260, 112, 82), "MR": (385, 256, 114, 81),
           "WC": (380, 252, 117, 80), "HAC": (338, 230, 131, 62),
           "DA": (111, 137, 99, 17)},
    WAIVER_MAIN: {"SP": (413, 260, 112, 84), "MR": (385, 256, 114, 83),
                  "WC": (380, 252, 117, 82), "HAC": (338, 230, 131, 68),
                  "DA": (111, 137, 99, 21)},
    WAIVER_ALT: {"SP": (413, 260, 112, 85), "MR": (385, 256, 114, 84),
                 "WC": (380, 252, 117, 84), "HAC": (338, 230, 131, 69),
                 "DA": (111, 137, 99, 20)},
}
MEASURED_T3 = {
    BEST: {"SP": 867, "MR": 836, "WC": 829, "HAC": 761, "DA": 364},
    WAIVER_MAIN: {"SP": 869, "MR": 838, "WC": 831, "HAC": 767, "DA": 368},
    WAIVER_ALT: {"SP": 870, "MR": 839, "WC": 833, "HAC": 768, "DA": 367},
}
MEASURED_T9 = {
    BEST: {"union_rules": 1006, "da_only": 201, "da_and_rules": 163},
    WAIVER_MAIN: {"union_rules": 1006, "da_only": 203, "da_and_rules": 165},
    WAIVER_ALT: {"union_rules": 1007, "da_only": 202, "da_and_rules": 165},
}
#: (cells within +-1, cells exact, domains passing)
MEASURED_SUMMARY = {
    BEST: (6, 5, []),
    WAIVER_MAIN: (11, 9, ["Politics"]),
    WAIVER_ALT: (11, 5, ["Politics"]),
}
STATUS = {BEST: "hypothesis", WAIVER_MAIN: "hypothesis, used with the differences documented",
          WAIVER_ALT: "hypothesis, used with the differences documented"}


@pytest.fixture(scope="module")
def runs():
    from cdm.data.worldA import build_worldA as B
    cache = {}

    def get(name):
        if name not in cache:
            b = B.build(scenario=name)
            cache[name] = (b.cases, G.run_gate(b.cases, name=name))
        return cache[name]
    return get


@pytest.fixture(scope="module")
def best(runs):
    return runs(BEST)


def _t3b_params():
    out = []
    for name in SCENARIOS:
        for m in G.METHODS:
            for i, d in enumerate(G.DOMAINS):
                value = MEASURED[name][m][i]
                delta = value - G.THESIS_T3B[m][d]
                marks = []
                if abs(delta) > G.TOL_T3B:
                    marks = [pytest.mark.xfail(strict=True, reason=(
                        f"rule-baseline count differs from the thesis-reported count; "
                        f"see docs/RULE_BASELINE_CHECK.md (scenario {name}, "
                        f"{STATUS[name]}: {m} {d} recomputed {value}, thesis "
                        f"{G.THESIS_T3B[m][d]}, {delta:+d})"))]
                out.append(pytest.param(name, m, d, marks=marks, id=f"{name}-{m}-{d}"))
    return out


def _global_params(measured, target, what):
    out = []
    for name in SCENARIOS:
        for k in target:
            delta = measured[name][k] - target[k]
            marks = []
            if delta:
                marks = [pytest.mark.xfail(strict=True, reason=(
                    f"rule-baseline count differs from the thesis-reported count; "
                    f"see docs/RULE_BASELINE_CHECK.md (scenario {name}, "
                    f"{STATUS[name]}: {what} {k} recomputed {measured[name][k]}, "
                    f"thesis {target[k]}, {delta:+d})"))]
            out.append(pytest.param(name, k, marks=marks, id=f"{name}-{k}"))
    return out


@needs_data
@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario_has_the_thesis_case_counts(runs, name):
    _, res = runs(name)
    assert {d: res.n_cases(d) for d in G.DOMAINS} == G.THESIS_N
    assert res.n_cases() == 1209


@needs_data
@pytest.mark.parametrize("name,method,domain", _t3b_params())
def test_t3b_cell_within_plus_minus_one(runs, name, method, domain):
    """Plan Step D5: every T3b cell within +-1. h1209: 6 cells pass, 14 are
    xfail. Waiver builds: 11 pass, 9 are xfail (all in NBA, NFL, General DA)."""
    _, res = runs(name)
    assert abs(res.count(method, domain) - G.THESIS_T3B[method][domain]) <= G.TOL_T3B


@needs_data
@pytest.mark.parametrize("name,method", _global_params(MEASURED_T3, G.THESIS_T3, "global"))
def test_t3_global_exact(runs, name, method):
    _, res = runs(name)
    assert res.count(method) == G.THESIS_T3[method]


@needs_data
@pytest.mark.parametrize("name,key", _global_params(MEASURED_T9, G.THESIS_T9, "T9"))
def test_t9_exact(runs, name, key):
    """waiver_politics_alt matches union 1007 and DA-only 202 exactly; 9
    per-domain cells of that build differ from the thesis-reported counts."""
    _, res = runs(name)
    assert res.t9()[key] == G.THESIS_T9[key]


@needs_data
@pytest.mark.parametrize("name", SCENARIOS)
def test_gate_does_not_pass(runs, name):
    """Recorded state: on every scenario some counts differ from the
    thesis-reported counts, so an ML step on the real data needs the
    recorded authorisation (plan §9.3; config/gate_status.yaml)."""
    _, res = runs(name)
    s = G.gate_summary(res)
    within, exact, passing = MEASURED_SUMMARY[name]
    assert not s["passes_t3b"]
    assert s["domains_passing"] == passing
    assert s["cells_within_tolerance"] == within and s["cells_exact"] == exact
    for d in ("General", "NBA", "NFL"):
        assert not G.domain_passes(res, d)


@needs_data
@pytest.mark.parametrize("name", SCENARIOS)
def test_measured_counts_are_stable_per_scenario(runs, name):
    """Regression: the reproduced numbers quoted in the report."""
    _, res = runs(name)
    for m in G.METHODS:
        assert tuple(res.count(m, d) for d in G.DOMAINS) == MEASURED[name][m]
        assert res.count(m) == MEASURED_T3[name][m]
    t9 = res.t9()
    assert {k: t9[k] for k in MEASURED_T9[name]} == MEASURED_T9[name]


@needs_data
def test_politics_da_carries_the_oq8b_range(runs):
    """With three or four options the thesis does not
    say which unchosen answer DA returns. The reported 17 is the plan's
    OQ8b default; over every choice the cell is 17-20, the global DA
    364-367. The waiver builds are binary, so they carry no such caveat."""
    _, res = runs(BEST)
    o = res.da_oq8b("Politics")
    assert (o["count"], o["low"], o["high"]) == (17, 17, 20) and o["n_cases"] > 0
    g = res.da_oq8b()
    assert (g["low"], g["high"]) == (364, 367)
    assert all(res.da_oq8b(d)["n_cases"] == 0 for d in ("General", "NBA", "NFL"))
    assert "OQ8b caveat" in G.format_table(res) and "range 17-20" in G.format_table(res)
    # 21 (thesis) is outside the range: "all options kept" stays refuted
    assert G.THESIS_T3B["DA"]["Politics"] > o["high"]
    for name in (WAIVER_MAIN, WAIVER_ALT):
        _, w = runs(name)
        assert w.da_oq8b()["n_cases"] == 0
        assert "OQ8b caveat" not in G.format_table(w)


@needs_data
def test_da_tie_rule_sensitivity_on_the_waiver_build(runs):
    """LABELLED SENSITIVITY, not adopted: the positional DA rule. Under
    answer coding (a) the first answer is raw code 2 in General, NBA and
    NFL. General becomes exact (108); the default stays the thesis text."""
    cases, text = runs(WAIVER_MAIN)
    assert text.da_tie_rule == "not_mr"
    first = G.run_gate(cases, da_tie_rule="first_answer")
    # thesis 108 / 142 / 101 / 21: differences 0, -1, -1, +1
    assert tuple(first.count("DA", d) for d in G.DOMAINS) == (108, 141, 100, 22)
    for m in A.RULES:                                     # only DA moves
        assert first.counts()[m] == text.counts()[m]
    assert tuple(text.count("DA", d) for d in G.DOMAINS) == (111, 137, 99, 21)
    assert first.count("DA", "General") == G.exploratory_da_count(
        cases, text, "General", "first_answer")
    assert not G.gate_summary(first)["passes_t3b"]        # NBA and NFL still differ


@needs_data
def test_general_reproduces_the_four_rules_exactly(best):
    """The General domain: Martinie, all responses, confidence (c)."""
    _, res = best
    for m in ("SP", "MR", "WC", "HAC"):
        assert res.count(m, "General") == G.THESIS_T3B[m]["General"]
    # DA is 3 above the thesis and no resolution of the 3 MR ties reaches 108
    assert res.count("DA", "General") == 111
    assert res.tie_range("DA", "General") == (110, 112)


@needs_data
def test_tie_log_of_the_h1209_scenario(best):
    _, res = best
    n_tied = {m: tuple(res.ties(m, d)["n_tied"] for d in G.DOMAINS) for m in A.RULES}
    assert n_tied == {"MR": (3, 16, 2, 0), "WC": (0, 3, 0, 0),
                      "HAC": (0, 7, 0, 1), "SP": (0, 0, 0, 0)}
    assert res.tie_range("MR", "NBA") == (247, 263)
    assert res.tie_range("MR", "NFL") == (113, 115)
    assert res.ties("HAC", "NBA")["scored_correct"] == 0


@needs_data
def test_frozen_build_gives_the_same_gate_result(runs):
    from cdm.data.worldA.build_worldA import build_root
    for name in SCENARIOS:
        if not (build_root() / "worldA" / name / "manifest.json").is_file():
            pytest.skip(f"no frozen build of {name}")
    import json
    for name in SCENARIOS:
        _, res = runs(name)
        frozen = G.run_gate_on_frozen(scenario=name)
        assert [r["choice"] for r in frozen.records] == [r["choice"] for r in res.records]
        # the gate result stored in the manifest is the one measured here
        m = json.loads((build_root() / "worldA" / name / "manifest.json").read_text())
        assert "hypothesis" in m["status"]
        gate = m["rule_gate"]
        assert gate["verdict"] == "NOT MET"
        for meth in G.METHODS:
            assert tuple(gate["per_domain"][meth][d]["recomputed"]
                         for d in G.DOMAINS) == MEASURED[name][meth]
            assert gate["global"][meth]["recomputed"] == MEASURED_T3[name][meth]
        assert {k: v["recomputed"] for k, v in gate["t9"].items()} == MEASURED_T9[name]


@needs_data
def test_ladder_step1_canonical_second_does_not_pass(best):
    """Ladder step 1. NBA WC and HAC become exact, MR/SP stay +2: not adopted
    (the plan needs all five rules of the domain within +-1)."""
    cases, _ = best
    res = G.run_gate(cases, policy="canonical_second")
    assert res.counts("NBA") == {"SP": 260, "MR": 254, "WC": 253, "HAC": 237, "DA": 138}
    assert res.count("MR", "General") == 386
    assert G.gate_summary(res)["domains_passing"] == []


@needs_data
def test_no_removal_of_four_nfl_games_reaches_the_sp_target():
    """EXPLORATORY fact used in the report: with complete triples and the
    file's winners, SP on the 216 NFL games is 115 and has no tie, so any
    212 of them give SP >= 111 (thesis 109, tolerance 110)."""
    res = G.run_gate_on_frozen(scenario="s2_stated_filters_only") if _frozen(
        "s2_stated_filters_only") else G.run_gate(G.build_cases("s2_stated_filters_only"))
    assert res.n_cases("NFL") == 216
    assert res.count("SP", "NFL") == 115 and res.ties("SP", "NFL")["n_tied"] == 0
    assert res.count("SP", "NFL") - 4 > G.THESIS_T3B["SP"]["NFL"] + G.TOL_T3B
    assert not G.exploratory_removal_search(res, "NFL", 4, ("MR", "SP"))["feasible"]


def _frozen(name):
    from cdm.data.worldA.build_worldA import build_root
    return (build_root() / "worldA" / name / "manifest.json").is_file()


@needs_data
def test_exploratory_politics_binary_variant_is_within_tolerance():
    """EXPLORATORY, not adopted: election votes for options 3-4 removed (108
    races) and the four races whose _pred and _agree header names differ
    dropped -> 104 races, all five rules within +-1."""
    cases = G.build_cases(BEST, {"elections_options": "binary_drop_other_votes",
                                 "elections_column_matching": "positional"})
    res = G.run_gate(cases)
    assert res.n_cases("Politics") == 108
    assert res.counts("Politics") == {"SP": 88, "MR": 87, "WC": 86, "HAC": 71, "DA": 21}
    res = G.without_cases(res, ["Politics:TX_sen", "Politics:CA_45th",
                                "Politics:IL_gub", "Politics:SC_gub"])
    assert res.counts("Politics") == {"SP": 84, "MR": 83, "WC": 82, "HAC": 68, "DA": 21}
    assert G.domain_passes(res, "Politics")
    # this is exactly the Politics part of the scenario waiver_main
    assert tuple(res.count(m, "Politics") for m in G.METHODS) == tuple(
        MEASURED[WAIVER_MAIN][m][3] for m in G.METHODS)


@needs_data
def test_predicted_support_rule_changes_no_politics_count(best):
    """"1 - x for every other option" (now the build
    default) against the invented even split: no Politics count changes."""
    cases, res = best
    split = G.run_gate(G.build_cases(BEST, {"elections_ps_other_answers": "split_evenly"}))
    assert split.counts("Politics") == res.counts("Politics")
    again = G.run_gate(G.exploratory_complement_ps(cases))
    assert again.counts("Politics") == res.counts("Politics")


@needs_data
def test_answer_set_changes_no_politics_count(best):
    """The options offered in the survey (.qsf) instead of the codes that
    occur in the workbook: 8 more three-answer cases, no count changes."""
    _, res = best
    qsf = G.run_gate(G.build_cases(BEST, {"elections_answer_set": "offered_qsf"}))
    assert qsf.counts("Politics") == res.counts("Politics")
    assert qsf.da_oq8b("Politics")["low"] == 17 and qsf.da_oq8b("Politics")["high"] == 20


@needs_data
@pytest.mark.parametrize("treatment,hac", [
    ("recode_to_revised", 131), ("clip", 131), ("domain_minmax_1_7", 132),
    ("column_minmax_1_7", 132)])
def test_nfl_original_confidence_treatments_are_not_separated_by_the_gate(treatment, hac):
    """WC is 117 (+3) under every
    treatment and HAC is within +-1 under every treatment, so the gate
    supports none over another. The base scenarios use the 1-5 recode,
    which equals the REVISED file; the others are switches only."""
    res = G.run_gate(G.build_cases(BEST, {"nfl_original_conf": treatment}))
    assert res.count("WC", "NFL") == 117
    assert res.count("HAC", "NFL") == hac
    assert abs(hac - G.THESIS_T3B["HAC"]["NFL"]) <= G.TOL_T3B
