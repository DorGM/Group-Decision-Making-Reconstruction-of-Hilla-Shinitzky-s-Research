"""Rule-baseline check ("rule gate"; step A10 = plan §2.1.1 Step D5).

What is compared and the result on the published build:
docs/RULE_BASELINE_CHECK.md.

frozen World-A build -> f_MR, f_WC, f_HAC, f_SP, f_DA per case -> success
counts per domain and overall -> comparison with the thesis-reported counts

  T3b  per-domain counts (TH Table 5.5 p.69 = Table A.10 p.115), 20 cells
  T3   global counts 864/831/830/774/372 of 1209 (TH Fig 5.1, Table 5.3)
  T9   union of the 4 rules 1007, DA-only 202, DA and >=1 rule 170

Tolerance (plan §1.1 and Step D5): "+-1 case per cell for ties" on T3b.
T3 and T9 are "exact (+-1 case per domain for tie conventions)"; the gate
reports the exact difference and two flags for them: `exact` (difference
0) and `within_domain_tolerance` (|difference| <= number of domains = 4,
the most that +-1 per domain can add up to).

The aggregators are those of cdm/aggregators.py with the options of
`options_from_config` (OQ8 tie policy, AM3 tie tolerance); DA listens to
{MR, WC, HAC, SP} (TH p.35).

Tie logging (plan §8 OQ8 "logged"): for every rule and domain the gate
counts the tied cases, how many of them the policy resolved to the correct
answer, and the range [low, high] of the success count over all possible
resolutions of the ties.

Tie policies (Step D5 ladder, item 1):
  canonical_first / canonical_second : answer-level rules of
      cdm.aggregators.choose.
  sp_tie_half_credit : a tied SP case of a binary question scores 0.5
      (Rutchick). MR/WC/HAC use the canonical-first answer. The plan does
      not define the single SP answer that DA needs for such a case
      (aggregators.choose raises); the gate uses the canonical-first SP
      answer for DA and for T9 and counts these cases in
      `sp_half_credit_cases`.

DA tie rule (`da_tie_rule`, cdm.aggregators.DA_TIE_RULES): the default is
the thesis text, "the answer not chosen by MR" (config `da_tie_rule`).
'first_answer' / 'second_answer' are labelled sensitivity options
(decision G2); a run that uses one says so in its table.

DA with more than two answers (plan §8 OQ8b): when several non-MR answers
share the lowest counter the thesis gives no rule. The gate uses the plan's
OQ8b default (lowest support) and reports, per domain, how many cases this
touches and the range of the DA count over every choice among the tied
answers (`GateResult.da_oq8b`). In a multi-option Politics build the DA
cell, the global DA count and T9 therefore carry that range as a caveat.
It is empty in an all-binary build.

This module adopts nothing. `ladder` runs the ordered switches of Step D5
one at a time and reports; the functions in the EXPLORATORY section are
side computations outside the ladder and are labelled as such in every output.

Usage:
  python -m cdm.eval.rule_gate --scenario h1209_nba_games
  python -m cdm.eval.rule_gate --scenario waiver_main --scenario waiver_politics_alt
  python -m cdm.eval.rule_gate --all
"""
import argparse
import itertools

from cdm import aggregators as A
from cdm import primitives as P

#: Thesis domain order (plan §2.1 case ordering) and rule order of TH Table 5.5.
DOMAINS = ("General", "NBA", "NFL", "Politics")
METHODS = ("SP", "MR", "WC", "HAC", "DA")

#: T3b. TH Table 5.5 p.69 (= Table A.10 p.115), n = 500/393/212/104.
THESIS_N = {"General": 500, "NBA": 393, "NFL": 212, "Politics": 104}
THESIS_T3B = {
    "SP": {"General": 413, "NBA": 258, "NFL": 109, "Politics": 84},
    "MR": {"General": 385, "NBA": 252, "NFL": 111, "Politics": 83},
    "WC": {"General": 380, "NBA": 253, "NFL": 114, "Politics": 83},
    "HAC": {"General": 338, "NBA": 237, "NFL": 131, "Politics": 68},
    "DA": {"General": 108, "NBA": 142, "NFL": 101, "Politics": 21},
}
#: T3. TH Fig 5.1; Table 5.4 row 2; Table 5.3 (WC = 830 back-solved, plan V1).
THESIS_T3 = {"SP": 864, "MR": 831, "WC": 830, "HAC": 774, "DA": 372}
THESIS_N_CASES = 1209
#: T9. TH Fig 5.2 p.67, §6.1 p.87 (plan V18).
THESIS_T9 = {"union_rules": 1007, "da_only": 202, "da_and_rules": 170}

#: Plan Step D5: "+-1 case per cell for ties".
TOL_T3B = 1
#: Plan §1.1: T3/T9 "exact (+-1 case per domain for tie conventions)".
TOL_GLOBAL_FROM_DOMAINS = len(DOMAINS)

POLICIES = ("canonical_first", "canonical_second", "sp_tie_half_credit")


# ---- one case ----------------------------------------------------------------

def _da_detail(choices, case, da_tie_rule=A.DEFAULT_DA_TIE_RULE):
    support = dict(zip(case.answers, P.support_vector(case.responses, case.answers)))
    return A.da_detail(choices, case.answers, A.RULES, support, da_tie_rule)


def _da(choices, case, da_tie_rule=A.DEFAULT_DA_TIE_RULE):
    return _da_detail(choices, case, da_tie_rule)["answer"]


def evaluate_case(case, policy="canonical_first", tol=P.DEFAULT_TIE_ABS_TOL,
                  da_tie_rule=A.DEFAULT_DA_TIE_RULE):
    """Rule answers, ties and credits of one case.

    Returns a dict:
      case_id, domain, n, correct
      tied    : {rule: tuple of answers sharing the best score} (4 rules)
      choice  : {method: answer} for MR, WC, HAC, SP, DA
      credit  : {method: 1.0 / 0.0 / 0.5}
      sp_half : True if SP is tied and the policy is sp_tie_half_credit
      range   : {method: (worst, best)} credit over all resolutions of the
                ties of this case (0/1 each)
      da_oq8b : None, or (worst, best) DA credit over every choice among
                the tied non-MR answers when the OQ8b rule was needed
                (more than two answers only)
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown tie policy {policy!r}; expected one of {POLICIES}")
    R, answers = case.responses, case.answers
    tied = {m: tuple(A.tied_answers(m, R, answers, tol)) for m in A.RULES}
    answer_policy = "canonical_first" if policy == "sp_tie_half_credit" else policy
    choice = {m: P.break_tie(list(tied[m]), answer_policy) for m in A.RULES}
    detail = _da_detail(choice, case, da_tie_rule)
    choice["DA"] = detail["answer"]
    da_oq8b = None
    if detail["oq8b_used"]:
        cand = detail["oq8b_candidates"]
        da_oq8b = (1.0 if all(a == case.correct for a in cand) else 0.0,
                   1.0 if case.correct in cand else 0.0)
    credit = {m: 1.0 if choice[m] == case.correct else 0.0 for m in METHODS}
    sp_half = policy == "sp_tie_half_credit" and len(tied["SP"]) > 1
    if sp_half:
        if len(answers) != 2:
            raise NotImplementedError(
                "half credit for an SP tie is defined for binary cases only "
                "(plan §8 OQ8)")
        credit["SP"] = 0.5
    rng = {m: (1.0 if all(a == case.correct for a in tied[m]) else 0.0,
               1.0 if case.correct in tied[m] else 0.0) for m in A.RULES}
    if all(len(tied[m]) == 1 for m in A.RULES):
        rng["DA"] = (credit["DA"], credit["DA"])
    else:
        outcomes = set()
        for combo in itertools.product(*(tied[m] for m in A.RULES)):
            outcomes.add(_da(dict(zip(A.RULES, combo)), case, da_tie_rule)
                         == case.correct)
        rng["DA"] = (0.0 if False in outcomes else 1.0, 1.0 if True in outcomes else 0.0)
    return {"case_id": case.case_id, "domain": case.domain, "n": len(R),
            "correct": case.correct, "tied": tied, "choice": choice,
            "credit": credit, "sp_half": sp_half, "range": rng,
            "da_oq8b": da_oq8b}


# ---- the gate ----------------------------------------------------------------

class GateResult:
    """Per-case records of one gate run and the counts derived from them."""

    def __init__(self, records, policy, tol, name=None,
                 da_tie_rule=A.DEFAULT_DA_TIE_RULE):
        self.records, self.policy, self.tol, self.name = records, policy, tol, name
        self.da_tie_rule = da_tie_rule

    def da_oq8b(self, domain=None):
        """OQ8b caveat of the DA count: {"n_cases": cases where several
        non-MR answers shared the lowest DA counter (more than two answers
        only), "count": the DA count reported, "low", "high": the DA count
        over every choice among those tied answers}. n_cases = 0 in an
        all-binary build."""
        recs = self._recs(domain)
        hit = [r for r in recs if r.get("da_oq8b")]
        base = sum(r["credit"]["DA"] for r in recs if not r.get("da_oq8b"))
        return {"n_cases": len(hit), "count": self.count("DA", domain),
                "low": int(base + sum(r["da_oq8b"][0] for r in hit)),
                "high": int(base + sum(r["da_oq8b"][1] for r in hit))}

    def domains(self):
        seen = []
        for r in self.records:
            if r["domain"] not in seen:
                seen.append(r["domain"])
        return [d for d in DOMAINS if d in seen] + [d for d in seen if d not in DOMAINS]

    def _recs(self, domain=None):
        return [r for r in self.records if domain is None or r["domain"] == domain]

    def n_cases(self, domain=None):
        return len(self._recs(domain))

    def count(self, method, domain=None):
        """Success count (sum of credits). An int unless half credits occur."""
        s = sum(r["credit"][method] for r in self._recs(domain))
        return int(s) if float(s).is_integer() else s

    def counts(self, domain=None):
        return {m: self.count(m, domain) for m in METHODS}

    def tie_range(self, method, domain=None):
        """(low, high) success count over every resolution of the ties."""
        recs = self._recs(domain)
        return (int(sum(r["range"][method][0] for r in recs)),
                int(sum(r["range"][method][1] for r in recs)))

    def ties(self, method, domain=None):
        """Tie log of one rule: number of tied cases; how many the policy
        scored correct; how many have the correct answer among the tied
        answers (= could be correct under some resolution); their IDs."""
        if method not in A.RULES:
            raise ValueError("ties are defined for MR, WC, HAC and SP")
        recs = [r for r in self._recs(domain) if len(r["tied"][method]) > 1]
        return {"n_tied": len(recs),
                "scored_correct": sum(r["credit"][method] for r in recs),
                "correct_among_tied": sum(1 for r in recs if r["correct"] in r["tied"][method]),
                "case_ids": [r["case_id"] for r in recs]}

    def sp_half_credit_cases(self, domain=None):
        return [r["case_id"] for r in self._recs(domain) if r["sp_half"]]

    def t9(self, domain=None):
        """T9 coverage: cases solved by >= 1 of the four rules, by DA only,
        and by DA and >= 1 rule. On the chosen answers (not on credits)."""
        union = da_only = da_and = 0
        for r in self._recs(domain):
            any_rule = any(r["choice"][m] == r["correct"] for m in A.RULES)
            da_ok = r["choice"]["DA"] == r["correct"]
            union += any_rule
            da_only += da_ok and not any_rule
            da_and += da_ok and any_rule
        n = len(self._recs(domain))
        return {"union_rules": union, "da_only": da_only, "da_and_rules": da_and,
                "covered_with_da": union + da_only, "n_cases": n}


def run_gate(cases, policy=None, tol=None, config=None, name=None,
             da_tie_rule=None):
    """Run Step D5 on a list of Case. `policy` / `tol` default to the config
    (cdm.aggregators.options_from_config: OQ8, tie_abs_tol); `da_tie_rule`
    defaults to the config's `da_tie_rule` (the thesis text, 'not_mr')."""
    if policy is None or tol is None:
        opts = A.options_from_config(config)
        policy = opts["policy"] if policy is None else policy
        tol = opts["tol"] if tol is None else tol
    if da_tie_rule is None:
        da_tie_rule = A.da_tie_rule_from_config(config)
    return GateResult([evaluate_case(c, policy, tol, da_tie_rule) for c in cases],
                      policy, tol, name, da_tie_rule)


def run_gate_on_frozen(scenario=None, path=None, policy=None, tol=None, config=None,
                       da_tie_rule=None):
    """Run the gate on a frozen build (cdm.data.worldA.build_worldA.load_frozen)."""
    from cdm.data.worldA.build_worldA import load_frozen
    cases, manifest = load_frozen(path=path, scenario=scenario)
    return run_gate(cases, policy, tol, config, name=manifest.get("scenario", scenario),
                    da_tie_rule=da_tie_rule)


# ---- comparison with the thesis-reported counts -------------------------------

def compare(result):
    """Rows of the comparison with T3b, T3 and T9.

    Each row: block ('T3b' / 'T3' / 'T9' / 'N'), key, domain, value, thesis,
    delta, tol, ok. For T3/T9 `ok` means delta == 0 and the extra field
    `within_domain_tolerance` means |delta| <= 4.
    """
    rows = []
    for d in DOMAINS:
        n = result.n_cases(d)
        rows.append({"block": "N", "key": "cases", "domain": d, "value": n,
                     "thesis": THESIS_N[d], "delta": n - THESIS_N[d], "tol": 0,
                     "ok": n == THESIS_N[d]})
    n = result.n_cases()
    rows.append({"block": "N", "key": "cases", "domain": "all", "value": n,
                 "thesis": THESIS_N_CASES, "delta": n - THESIS_N_CASES, "tol": 0,
                 "ok": n == THESIS_N_CASES})
    for m in METHODS:
        for d in DOMAINS:
            v, t = result.count(m, d), THESIS_T3B[m][d]
            rows.append({"block": "T3b", "key": m, "domain": d, "value": v, "thesis": t,
                         "delta": v - t, "tol": TOL_T3B, "ok": abs(v - t) <= TOL_T3B,
                         "tie_range": result.tie_range(m, d)})
    for m in METHODS:
        v, t = result.count(m), THESIS_T3[m]
        rows.append({"block": "T3", "key": m, "domain": "all", "value": v, "thesis": t,
                     "delta": v - t, "tol": 0, "ok": v == t,
                     "within_domain_tolerance": abs(v - t) <= TOL_GLOBAL_FROM_DOMAINS,
                     "tie_range": result.tie_range(m)})
    t9 = result.t9()
    for k, t in THESIS_T9.items():
        v = t9[k]
        rows.append({"block": "T9", "key": k, "domain": "all", "value": v, "thesis": t,
                     "delta": v - t, "tol": 0, "ok": v == t,
                     "within_domain_tolerance": abs(v - t) <= TOL_GLOBAL_FROM_DOMAINS})
    return rows


def domain_passes(result, domain):
    """Plan Step D5 adoption rule: all five rules of the domain within +-1,
    and the domain has the thesis number of cases."""
    return (result.n_cases(domain) == THESIS_N[domain] and
            all(abs(result.count(m, domain) - THESIS_T3B[m][domain]) <= TOL_T3B
                for m in METHODS))


def gate_summary(result):
    """Whether the check is met, and the cells outside the tolerance."""
    rows = compare(result)
    bad = [r for r in rows if not r["ok"]]
    return {
        "passes_t3b": all(r["ok"] for r in rows if r["block"] in ("T3b", "N")),
        "passes_t3_exact": all(r["ok"] for r in rows if r["block"] == "T3"),
        "passes_t9_exact": all(r["ok"] for r in rows if r["block"] == "T9"),
        "domains_passing": [d for d in DOMAINS if domain_passes(result, d)],
        "cells_within_tolerance": sum(1 for r in rows if r["block"] == "T3b" and r["ok"]),
        "cells_exact": sum(1 for r in rows if r["block"] == "T3b" and r["delta"] == 0),
        "outside_tolerance": [(r["block"], r["key"], r["domain"], r["value"], r["thesis"], r["delta"])
                   for r in bad],
    }


def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def _delta(d):
    return f"{d:+g}" if d else "0"


def format_table(result, title=None):
    """Markdown: domain x rule -> recomputed, thesis-reported, delta; T3; T9; ties."""
    out = []
    if title:
        da_rule = getattr(result, "da_tie_rule", A.DEFAULT_DA_TIE_RULE)
        label = ("thesis text" if da_rule == A.DEFAULT_DA_TIE_RULE
                 else "SENSITIVITY OPTION, not the thesis text")
        out.append(f"**{title}** (tie policy `{result.policy}`, tol {result.tol:g}, "
                   f"DA tie rule `{da_rule}` = {label})\n")
    out.append("| Rule | " + " | ".join(
        f"{d} (n={result.n_cases(d)}, thesis {THESIS_N[d]})" for d in DOMAINS)
        + f" | All (n={result.n_cases()}, thesis {THESIS_N_CASES}) |")
    out.append("|---|" + "---|" * (len(DOMAINS) + 1))
    for m in METHODS:
        cells = []
        for d in DOMAINS:
            v, t = result.count(m, d), THESIS_T3B[m][d]
            mark = "" if abs(v - t) <= TOL_T3B else " **DIFF**"
            cells.append(f"{_fmt(v)} / {t} / {_delta(v - t)}{mark}")
        v, t = result.count(m), THESIS_T3[m]
        cells.append(f"{_fmt(v)} / {t} / {_delta(v - t)}")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Cells: recomputed / thesis / difference. DIFF = outside +-1.")
    t9 = result.t9()
    out.append("")
    out.append("| T9 | recomputed | thesis | difference |")
    out.append("|---|---|---|---|")
    for k, t in THESIS_T9.items():
        out.append(f"| {k} | {t9[k]} | {t} | {_delta(t9[k] - t)} |")
    caveat = format_oq8b_caveat(result)
    if caveat:
        out += ["", caveat]
    return "\n".join(out)


def format_oq8b_caveat(result):
    """One paragraph, or '' for an all-binary run: where the DA count
    depends on the OQ8b rule (several unchosen answers, more than two
    answers) and the range of the count over that open choice."""
    if not hasattr(result, "da_oq8b"):
        return ""
    parts = []
    for d in result.domains():
        o = result.da_oq8b(d)
        if o["n_cases"]:
            parts.append(f"{d} DA = {o['count']}, range {o['low']}-{o['high']} "
                         f"({o['n_cases']} cases)")
    if not parts:
        return ""
    o = result.da_oq8b()
    return ("OQ8b caveat: in cases with more than two answers the thesis does not "
            "say which of several unchosen answers DA returns. The count shown uses "
            "the plan default (lowest support). Over every such choice: "
            + "; ".join(parts) + f"; global DA {o['low']}-{o['high']}. "
            "T9 (DA-only, DA-and-rules) moves with it.")


def format_ties(result):
    """Markdown tie log: per rule and domain, tied cases, how the policy
    scored them, and the range of the count over all resolutions."""
    out = ["| Rule | " + " | ".join(DOMAINS) + " | All |", "|---|" + "---|" * (len(DOMAINS) + 1)]
    for m in A.RULES:
        cells = []
        for d in list(DOMAINS) + [None]:
            t = result.ties(m, d)
            lo, hi = result.tie_range(m, d)
            cells.append(f"{t['n_tied']} tied, {_fmt(float(t['scored_correct']))} scored correct; "
                         f"range {lo}-{hi}" if t["n_tied"] else "0 tied")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    cells = []
    for d in list(DOMAINS) + [None]:
        lo, hi = result.tie_range("DA", d)
        cells.append(f"range {lo}-{hi}")
    out.append("| DA (follows the four) | " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---- Step D5 ladder ------------------------------------------------------------

#: The ordered switches of plan Step D5. Each step is a list of (label,
#: tie policy or None, build-switch overrides). Steps that the staged data
#: cannot run are listed with the reason instead.
LADDER = (
    ("1 tie rule (OQ8)", [
        ("canonical_second", "canonical_second", {}),
        ("sp_tie_half_credit", "sp_tie_half_credit", {}),
    ]),
    ("2 confidence normalization (OQ2), judged on WC", [
        ("Likert c/5", None, {"likert_confidence": "b_c_over_5"}),
        ("Martinie (a) |p-50|/50", None, {"martinie_confidence": "a_abs_p_minus_50_over_50"}),
        ("Martinie (b) max(p,100-p)/100", None,
         {"martinie_confidence": "b_max_p_100_minus_p_over_100"}),
    ]),
    ("3 Martinie exclusions (OQ4)", [
        ("vote-inconsistent responses dropped", None,
         {"martinie_drop_vote_inconsistent": True}),
        # cheaters: the 41 workers are absent from the files and cannot be
        # identified; the switch has no effect.
    ]),
    ("4 minimum N (OQ4)", [
        ("N >= 10", None, {"min_n": 10}),
    ]),
    ("5 Rutchick revisions (OQ44)", [
        ("REVISED files (base uses the originals)", None,
         {"rutchick_version": "revised"}),
    ]),
    # 6 NFL source = Lee 2018: cannot be built, the Lee folder holds no game
    # outcomes (build_worldA raises LeeOutcomesMissing).
)


def build_cases(base_scenario, overrides=None):
    """Cases of a scenario with overridden build switches (needs the raw
    data, CDM_DATA_ROOT). Nothing is frozen and no file is changed."""
    from cdm.data.worldA import build_worldA as B
    b = B.build(scenario=base_scenario, switches=overrides or None)
    if b.cases is None:
        raise ValueError("this set of switches keeps incomplete responses (no cases)")
    return b.cases


def ladder(base_scenario, config=None):
    """Run every step of the Step D5 ladder, one switch at a time, from
    `base_scenario`. Returns a list of (step, label, GateResult). The first
    entry is the base run. Nothing is adopted here."""
    base_cases = build_cases(base_scenario)
    base = run_gate(base_cases, config=config, name=base_scenario)
    out = [("0 base", base_scenario, base)]
    for step, variants in LADDER:
        for label, policy, overrides in variants:
            if overrides:
                res = run_gate(build_cases(base_scenario, overrides), policy=policy,
                               config=config, da_tie_rule=base.da_tie_rule)
            else:
                res = run_gate(base_cases, policy=policy, tol=base.tol,
                               da_tie_rule=base.da_tie_rule)
            res.name = f"{base_scenario} + {label}"
            out.append((step, label, res))
    return out


def format_ladder(runs):
    """Markdown: one line per ladder run, 20 differences from T3b."""
    head = "| Step | Switch | n | " + " | ".join(
        f"{d[:3]} {m}" for d in DOMAINS for m in METHODS) + " | domains passing |"
    out = [head, "|---|---|---|" + "---|" * 21]
    for step, label, res in runs:
        cells = [_delta(res.count(m, d) - THESIS_T3B[m][d]) for d in DOMAINS for m in METHODS]
        ok = [d for d in DOMAINS if domain_passes(res, d)]
        out.append(f"| {step} | {label} | {res.n_cases()} | " + " | ".join(cells)
                   + f" | {', '.join(ok) or 'none'} |")
    return "\n".join(out)


# ---- EXPLORATORY side computations (outside the ladder; adopt nothing) ---------

def exploratory_removal_search(result, domain, k, methods=("MR", "SP")):
    """EXPLORATORY. Can removing exactly k cases of `domain` bring the given
    methods to their thesis-reported T3b counts (exactly)?

    Each case is reduced to its vector of 0/1 credits on `methods`. The
    function returns {"needed": {method: surplus over the thesis-reported count}, "feasible":
    bool, "patterns": {credit vector: number of cases}, "solutions": list of
    {credit vector: how many cases of that pattern to remove}}. Cases with
    the same vector are interchangeable, so the solutions are counts per
    pattern, not named cases.
    """
    recs = [r for r in result.records if r["domain"] == domain]
    need = {m: result.count(m, domain) - THESIS_T3B[m][domain] for m in methods}
    patterns = {}
    for r in recs:
        key = tuple(int(r["credit"][m]) for m in methods)
        patterns[key] = patterns.get(key, 0) + 1
    keys = sorted(patterns)
    sols = []

    def rec(i, left, need_left, chosen):
        if i == len(keys):
            if left == 0 and all(v == 0 for v in need_left):
                sols.append(dict(chosen))
            return
        key = keys[i]
        for c in range(0, min(left, patterns[key]) + 1):
            nl = tuple(v - c * b for v, b in zip(need_left, key))
            if any(v < 0 for v in nl):
                break
            rec(i + 1, left - c, nl, chosen + ([(key, c)] if c else []))

    if all(v >= 0 for v in need.values()) and float(sum(need.values())).is_integer():
        rec(0, k, tuple(int(need[m]) for m in methods), [])
    return {"methods": tuple(methods), "k": k, "needed": need,
            "feasible": bool(sols), "patterns": patterns, "solutions": sols}


def without_cases(result, case_ids, name=None):
    """EXPLORATORY helper: the same gate run without the named cases."""
    drop = set(case_ids)
    missing = drop - {r["case_id"] for r in result.records}
    if missing:
        raise KeyError(f"cases not in the run: {sorted(missing)}")
    return GateResult([r for r in result.records if r["case_id"] not in drop],
                      result.policy, result.tol, name or result.name,
                      getattr(result, "da_tie_rule", A.DEFAULT_DA_TIE_RULE))


#: DA rules for a 2-2 split of the four rules on a binary case. 'not_mr' is
#: the thesis rule (TH p.35) and the gate's default; the other two are the
#: labelled sensitivity options of cdm.aggregators.DA_TIE_RULES.
DA_TIE_RULES = A.DA_TIE_RULES


def exploratory_da_count(cases, result, domain, da_tie_rule):
    """EXPLORATORY. DA success count of one domain if a tie in the DA
    counters were broken by another rule than "the answer not chosen by MR":
    'first_answer' / 'second_answer' = the first / second tied answer in
    the canonical order of the case. The four rule answers are those of
    `result` (same tie policy). `cases` must be the list the run was made
    from (same order)."""
    if da_tie_rule not in DA_TIE_RULES:
        raise ValueError(f"unknown DA tie rule {da_tie_rule!r}")
    if len(cases) != len(result.records):
        raise ValueError("cases and result do not belong together")
    n = 0
    for case, r in zip(cases, result.records):
        if case.case_id != r["case_id"]:
            raise ValueError("cases and result do not belong together")
        if case.domain != domain:
            continue
        if da_tie_rule == "not_mr":
            n += r["choice"]["DA"] == case.correct
            continue
        counts = {a: 0 for a in case.answers}
        for m in A.RULES:
            counts[r["choice"][m]] += 1
        low = min(counts.values())
        tied = [a for a in case.answers if counts[a] == low]
        pick = tied[0] if (da_tie_rule == "first_answer" or len(tied) == 1) else tied[1]
        n += pick == case.correct
    return n


def exploratory_complement_ps(cases):
    """EXPLORATORY. Rutchick's own predicted-support reading for cases with
    more than two answers: a voter who reports x for the chosen answer is
    taken to predict 1 - x for EVERY other answer (Election Analysis
    workbook). This is
    the build default (`elections_ps_other_answers: complement_each`), so
    the function changes nothing on a default build; it still converts a
    build made with `split_evenly`. Binary cases are returned unchanged.
    Returns new Case objects."""
    from cdm.schema import Case, Response
    out = []
    for c in cases:
        if len(c.answers) == 2:
            out.append(c)
            continue
        rs = []
        for r in c.responses:
            x = r.ps[c.answers.index(r.v)]
            rs.append(Response(v=r.v, c=r.c,
                               ps=tuple(x if a == r.v else 1.0 - x for a in c.answers)))
        out.append(Case(case_id=c.case_id, answers=c.answers, correct=c.correct,
                        responses=tuple(rs), domain=c.domain))
    return out


def format_delta_table(runs):
    """Markdown: one line per (label, GateResult): the domain sizes and the
    20 differences from T3b. For scenario and exploratory tables."""
    head = "| Run | n (Gen/NBA/NFL/Pol) | " + " | ".join(
        f"{d[:3]} {m}" for d in DOMAINS for m in METHODS) + " | cells within +-1 | domains passing |"
    out = [head, "|---|---|" + "---|" * 22]
    for label, res in runs:
        cells = [_delta(res.count(m, d) - THESIS_T3B[m][d]) for d in DOMAINS for m in METHODS]
        ok = [d for d in DOMAINS if domain_passes(res, d)]
        within = sum(1 for d in DOMAINS for m in METHODS
                     if abs(res.count(m, d) - THESIS_T3B[m][d]) <= TOL_T3B)
        ns = "/".join(str(res.n_cases(d)) for d in DOMAINS)
        out.append(f"| {label} | {ns} | " + " | ".join(cells)
                   + f" | {within}/20 | {', '.join(ok) or 'none'} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rule gate (plan Step D5, step A10)")
    ap.add_argument("--scenario", action="append", help="frozen scenario (repeatable)")
    ap.add_argument("--all", action="store_true", help="every frozen scenario")
    ap.add_argument("--policy", choices=POLICIES, help="tie policy (default: config OQ8)")
    ap.add_argument("--da-tie-rule", choices=A.DA_TIE_RULES,
                    help="DA tie rule (default: config da_tie_rule = thesis text "
                         "'not_mr'; the others are labelled sensitivity options)")
    ap.add_argument("--ties", action="store_true", help="also print the tie log")
    args = ap.parse_args(argv)
    names = list(args.scenario or [])
    if args.all:
        from cdm.data.worldA.build_worldA import build_root
        root = build_root() / "worldA"
        names = sorted(p.name for p in root.iterdir() if (p / "manifest.json").is_file())
    if not names:
        ap.error("give --scenario NAME or --all")
    for name in names:
        res = run_gate_on_frozen(scenario=name, policy=args.policy,
                                 da_tie_rule=args.da_tie_rule)
        print(format_table(res, title=name))
        if args.ties:
            print()
            print(format_ties(res))
        s = gate_summary(res)
        print(f"\nT3b cells within +-1: {s['cells_within_tolerance']}/20 "
              f"(exact {s['cells_exact']}); domains passing: "
              f"{', '.join(s['domains_passing']) or 'none'}; check "
              f"{'MET' if s['passes_t3b'] else 'NOT MET'}\n")


if __name__ == "__main__":
    main()
