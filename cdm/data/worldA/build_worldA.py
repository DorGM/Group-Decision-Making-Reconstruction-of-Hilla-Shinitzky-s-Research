"""World-A build (step A9 = plan §2.1.1 Steps D3, D4, D6).

raw files -> response filters -> case filters -> order -> frozen file.

Every choice is a named switch in config/worldA_scenarios.yaml. A scenario
is a named set of overrides and is a HYPOTHESIS about how the thesis data
set was built; no scenario is established as the thesis data set (every
1209-case build gives N and correct-rate SDs that differ from the printed
ones in the last digit, and on every scenario some rule-baseline counts
differ from the thesis-reported counts; docs/RULE_BASELINE_CHECK.md). The
scenario's `status` and the rule-gate result of the build are written into
the manifest by `freeze`.

Frozen format (deviation from the plan's "parquet + hash", docs/IMPLEMENTATION_DECISIONS.md
W6): JSON lines, one case per line, with SHA-256 hashes in the manifest.

Usage:
  python -m cdm.data.worldA.build_worldA --scenario h1209_nba_games
  python -m cdm.data.worldA.build_worldA --table
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import yaml

from cdm.config import CODE_ROOT, data_root as _data_root
from cdm.data.worldA import lee, martinie, rutchick
from cdm.data.worldA.common import (ANSWER_CODINGS, DOMAIN_GENERAL, DOMAIN_NBA,
                                    DOMAIN_NFL, DOMAIN_POLITICS, count_correct,
                                    filter_responses, make_case, sha256_of)
from cdm.schema import Case, Response

SCENARIOS_PATH = CODE_ROOT / "config" / "worldA_scenarios.yaml"
CASES_FILE = "worldA_cases.jsonl"
TABLE_FILE = "worldA_case_table.csv"
MANIFEST_FILE = "manifest.json"

#: TH p.43 and p.68 (plan §2.1, Step D4).
THESIS = {
    "counts": {DOMAIN_GENERAL: 500, DOMAIN_NBA: 393, DOMAIN_NFL: 212,
               DOMAIN_POLITICS: 104},
    "n_cases": 1209, "n_mean": 70, "n_sd": 35.5, "n_min": 10, "n_max": 129,
    "rate_mean": 62.0, "rate_sd": 23.8, "rate_min": 2.7, "rate_max": 98.9,
}


# ---- switches ----------------------------------------------------------------

def load_scenarios(path=None):
    with open(path or SCENARIOS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def default_switches(config=None):
    config = config if config is not None else load_scenarios()
    return {k: v["default"] for k, v in config["switches"].items()}


def scenario_names(config=None):
    """Current scenario names (deprecated aliases are not listed)."""
    config = config if config is not None else load_scenarios()
    return list(config["scenarios"])


def resolve_scenario(name, config=None):
    """Current name of a scenario. The `best_evidence_*` names used in
    earlier versions are deprecated aliases of the `h1209_*` names."""
    config = config if config is not None else load_scenarios()
    return (config.get("deprecated_aliases") or {}).get(name, name)


def scenario_info(name, config=None):
    """{"description", "status"} of a named scenario."""
    config = config if config is not None else load_scenarios()
    entry = config["scenarios"][resolve_scenario(name, config)]
    return {"description": entry.get("description", ""),
            "status": entry.get("status", DEFAULT_STATUS)}


#: Status of a scenario that does not state one. Every scenario is a hypothesis.
DEFAULT_STATUS = "hypothesis"
CUSTOM_STATUS = "custom switch set (not a named scenario); hypothesis"


def scenario_switches(name, config=None):
    """Full switch dict of a named scenario (defaults + its overrides)."""
    config = config if config is not None else load_scenarios()
    name = resolve_scenario(name, config)
    sw = default_switches(config)
    overrides = config["scenarios"][name].get("overrides") or {}
    unknown = set(overrides) - set(sw)
    if unknown:
        raise KeyError(f"scenario {name!r}: unknown switches {sorted(unknown)}")
    sw.update(overrides)
    return sw


def _check(sw, config):
    for k, v in sw.items():
        opts = config["switches"][k].get("options")
        if opts is not None and v not in opts:
            raise ValueError(f"switch {k} = {v!r} is not one of {opts}")
    by_domain = sw.get("answer_coding_by_domain") or {}
    for d, coding in by_domain.items():
        if d not in sw["domain_order"]:
            raise ValueError(f"answer_coding_by_domain: unknown domain {d!r}")
        if coding not in ANSWER_CODINGS:
            raise ValueError(f"answer_coding_by_domain[{d!r}] = {coding!r} is "
                             f"not one of {ANSWER_CODINGS}")
    k = sw.get("elections_drop_third_option_races_min_votes")
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
        raise ValueError("elections_drop_third_option_races_min_votes must be "
                         f"null or an integer >= 1, got {k!r}")


def build_root():
    """Folder for frozen builds: env CDM_BUILD_ROOT, else <repo>/build."""
    env = os.environ.get("CDM_BUILD_ROOT")
    return Path(env).expanduser().resolve() if env else (CODE_ROOT / "build").resolve()


# ---- per-domain raw loading ----------------------------------------------------

def _load_domain(domain, sw, data_root):
    """RawCases of one thesis domain, in source-file order, + input files."""
    if domain == DOMAIN_GENERAL:
        return martinie.load_raw(data_root), martinie.input_files(data_root)
    version = sw["rutchick_version"]
    if domain == DOMAIN_NBA:
        return (rutchick.load_raw_nba(version, data_root),
                [rutchick.raw_file("nba", version, data_root)])
    if domain == DOMAIN_POLITICS:
        files = [rutchick.raw_file("elections", version, data_root)]
        qsf = rutchick.election_qsf_file(data_root)
        if qsf.is_file():
            # the .qsf gives the offered answer options of every race
            files.append(qsf)
        return rutchick.load_raw_elections(version, data_root), files
    if domain == DOMAIN_NFL:
        if sw["nfl_source"] == "rutchick_s1":
            raws = rutchick.load_raw_nfl(version, data_root)
            if version == "original" and sw["nfl_original_conf"] == "recode_to_revised":
                rutchick.recode_nfl_original_confidence(raws)
            return raws, [rutchick.raw_file("nfl", version, data_root)]
        if sw["nfl_source"] == "lee2018":
            path = sw.get("lee_outcomes_file")
            if not path:
                raise lee.LeeOutcomesMissing(
                    "nfl_source = lee2018 cannot be built: the Lee 2018 folder "
                    "holds no game outcomes and lee_outcomes_file is not set.")
            with open(path, newline="", encoding="utf-8") as fh:
                outcomes = {r["case_id"]: r["winner"] for r in csv.DictReader(fh)}
            raws = []
            for raw in lee.load_raw(data_root, sw["lee_expert_only"]):
                team = outcomes.get(raw.case_id)
                if team is None:
                    continue
                raw.winner = {v: k for k, v in raw.labels.items()}[team]
                raws.append(raw)
            return raws, lee.input_files(data_root) + [Path(path)]
        raise ValueError(f"unknown nfl_source {sw['nfl_source']!r}")
    raise ValueError(f"unknown domain {domain!r}")


def _header_drop(raw, sw):
    """Reason string if the case is dropped by a column-matching hypothesis."""
    if raw.domain == DOMAIN_NFL and raw.source.startswith("Rutchick"):
        mode = sw["nfl_column_matching"]
        if mode == "token" and not raw.meta["header_tokens_match"]:
            return "nfl_header_tokens_mismatch"
        if mode == "exact_prefix" and not raw.meta["header_exact_match"]:
            return "nfl_header_prefix_mismatch"
    if raw.domain == DOMAIN_POLITICS:
        mode = sw["elections_column_matching"]
        if mode == "name_strict" and not raw.meta["name_match_strict"]:
            return "elections_header_name_mismatch"
        if mode == "name_loose" and not raw.meta["name_match_loose"]:
            return "elections_header_name_mismatch"
        if mode == "pred_agree_names" and not raw.meta["pred_agree_match"]:
            return "elections_pred_agree_name_mismatch"
    return None


# ---- the build -----------------------------------------------------------------

class Build:
    """Result of `build`.

    records : one dict per kept case, in final order (index, case_id,
              domain, n, n_correct, rate, ...).
    cases   : list of Case in the same order, or None when the scenario
              keeps incomplete responses (votes only).
    log     : per domain, the number of cases/responses removed by each
              step, with the IDs of the dropped cases.
    """

    def __init__(self, name, switches, records, cases, log, inputs,
                 description="", status=CUSTOM_STATUS):
        self.name, self.switches = name, switches
        self.records, self.cases, self.log, self.inputs = records, cases, log, inputs
        self.description, self.status = description, status

    def counts(self):
        out = {}
        for r in self.records:
            out[r["domain"]] = out.get(r["domain"], 0) + 1
        return out

    def stats(self, domain=None):
        recs = [r for r in self.records if domain is None or r["domain"] == domain]
        return summary_stats([r["n"] for r in recs], [r["rate"] for r in recs])


def _sd(xs, ddof):
    n = len(xs)
    if n - ddof <= 0:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def summary_stats(ns, rates):
    """TH p.43 statistics. Rates are in percent. Both the population and the
    sample SD are given (the thesis does not say which it reports)."""
    if not ns:
        return {"n_cases": 0}
    return {
        "n_cases": len(ns),
        "n_mean": sum(ns) / len(ns), "n_sd_pop": _sd(ns, 0), "n_sd_sample": _sd(ns, 1),
        "n_min": min(ns), "n_max": max(ns),
        "rate_mean": sum(rates) / len(rates), "rate_sd_pop": _sd(rates, 0),
        "rate_sd_sample": _sd(rates, 1), "rate_min": min(rates), "rate_max": max(rates),
    }


def build(switches=None, scenario=None, data_root=None, config=None,
          make_cases=True):
    """Run Steps D1-D3 and D6 for one set of switches.

    switches : full or partial switch dict (missing keys = defaults), or
    scenario : a name from config/worldA_scenarios.yaml.
    """
    config = config if config is not None else load_scenarios()
    if scenario:
        scenario = resolve_scenario(scenario, config)
    sw = scenario_switches(scenario, config) if scenario else default_switches(config)
    if switches:
        unknown = set(switches) - set(sw)
        if unknown:
            raise KeyError(f"unknown switches {sorted(unknown)}")
        sw.update(switches)
    _check(sw, config)
    if sw["within_domain_order"] != "source_file_order":
        raise ValueError("only source_file_order is implemented (plan Step D6)")
    data_root = Path(data_root) if data_root is not None else _data_root()

    records, cases, log, inputs = [], [], {}, []
    complete = bool(sw["complete_responses_only"])
    for domain in sw["domain_order"]:
        raws, files = _load_domain(domain, sw, data_root)
        inputs += files
        dlog = {"cases_in_file": len(raws),
                "responses_with_a_vote": sum(
                    1 for raw in raws for r in raw.responses if r.vote is not None),
                "cases_dropped": {}, "responses_dropped": {}}

        def drop(reason, raw):
            dlog["cases_dropped"].setdefault(reason, []).append(raw.case_id)

        drop_people = ()
        if domain == DOMAIN_NBA and sw["nba_extra_exclusion"] == "drop_participants":
            drop_people = tuple(sw["nba_drop_participants"])
        for raw in raws:
            reason = _header_drop(raw, sw)
            if reason:
                drop(reason, raw)
                continue
            if (domain == DOMAIN_NBA and sw["nba_extra_exclusion"] == "drop_games"
                    and raw.case_id in set(sw["nba_drop_games"])):
                drop("nba_named_game", raw)
                continue
            options = raw.options
            if domain == DOMAIN_POLITICS:
                k = sw["elections_drop_third_option_races_min_votes"]
                if k is not None and raw.meta["third_option_votes"] >= k:
                    drop("elections_third_option_votes_at_least_k", raw)
                    continue
                if sw["elections_options"] == "drop_multi_option_races" and len(options) > 2:
                    drop("elections_more_than_two_options", raw)
                    continue
                if sw["elections_options"] == "binary_drop_other_votes":
                    options = (1, 2)
                elif sw["elections_answer_set"] == "observed_codes":
                    options = raw.meta["observed_options"]
            if raw.winner not in options:
                drop("winner_not_among_answers", raw)
                continue
            kept, n = filter_responses(
                raw, complete_only=complete, drop_participants=drop_people,
                martinie_drop_vote_inconsistent=(
                    domain == DOMAIN_GENERAL and sw["martinie_drop_vote_inconsistent"]),
                options=options)
            for k, v in n.items():
                if v and k != "no_vote":
                    dlog["responses_dropped"][k] = dlog["responses_dropped"].get(k, 0) + v
            if not kept:
                drop("no_responses_left", raw)
                continue
            n_resp, n_correct = count_correct(raw, kept)
            if sw["drop_0_or_100_percent_correct"] and n_correct in (0, n_resp):
                drop("0_percent_correct" if n_correct == 0 else "100_percent_correct", raw)
                continue
            if sw["min_n"] is not None and n_resp < sw["min_n"]:
                drop("n_below_min_n", raw)
                continue
            rec = {"index": len(records), "case_id": f"{raw.domain}:{raw.case_id}",
                   "domain": domain, "source": raw.source, "n": n_resp,
                   "n_correct": n_correct, "rate": 100.0 * n_correct / n_resp,
                   "n_answers": len(options)}
            if complete and make_cases:
                formula, scale_max, clip = rutchick.likert_settings(
                    raw, sw["likert_confidence"], sw["nfl_original_conf"])
                coding = (sw["answer_coding_by_domain"] or {}).get(
                    domain, sw["answer_coding"])
                case = make_case(
                    raw, kept, martinie_conf=sw["martinie_confidence"],
                    likert_conf=formula, likert_scale_max=scale_max,
                    likert_clip=clip, answer_coding=coding,
                    coding_seed=sw["answer_coding_seed"], options=options,
                    ps_other_answers=sw["elections_ps_other_answers"])
                cases.append(case)
                rec["answers"] = list(case.answers)
                rec["correct"] = case.correct
            records.append(rec)
        dlog["cases_kept"] = sum(1 for r in records if r["domain"] == domain)
        log[domain] = dlog
    ids = [r["case_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case IDs in the build")
    info = (scenario_info(scenario, config) if scenario and not switches
            else {"description": (f"{scenario} with overrides {switches}"
                                  if scenario else "custom switch set"),
                  "status": CUSTOM_STATUS})
    return Build(scenario or "custom", sw, records,
                 cases if (complete and make_cases) else None, log, inputs,
                 info["description"], info["status"])


# ---- freeze / load -------------------------------------------------------------

def _case_line(index, case):
    return json.dumps({
        "index": index, "case_id": case.case_id, "domain": case.domain,
        "answers": list(case.answers), "correct": case.correct,
        "responses": [[r.v, r.c, list(r.ps)] for r in case.responses],
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def freeze(b, out_dir=None):
    """Write the frozen data set (plan Step D6): one JSON line per case, a
    per-case CSV table and a manifest with every switch, the counts, the
    filter log, the statistics and SHA-256 hashes of outputs and inputs.

    Format (docs/IMPLEMENTATION_DECISIONS.md W6): the plan says "parquet + hash". Neither
    pyarrow nor fastparquet is installed in this environment, so the frozen
    file is JSON lines (exact float round-trip, no extra dependency). The
    hash is of that file.

    The manifest also carries the scenario's `status` (every scenario is a
    hypothesis) and `rule_gate`: the result of plan Step
    D5 on exactly these cases (20 per-domain counts against the thesis,
    global counts, T9, pass/fail). Rules only; no classifier is involved.
    """
    if b.cases is None:
        raise ValueError("this scenario keeps incomplete responses; it has "
                         "statistics but no Case objects to freeze")
    out_dir = Path(out_dir) if out_dir is not None else build_root() / "worldA" / b.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / CASES_FILE
    with open(cases_path, "w", encoding="utf-8", newline="\n") as fh:
        for i, case in enumerate(b.cases):
            fh.write(_case_line(i, case) + "\n")
    table_path = out_dir / TABLE_FILE
    cols = ["index", "case_id", "domain", "source", "n", "n_correct", "rate",
            "n_answers", "answers", "correct"]
    with open(table_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in b.records:
            w.writerow([("|".join(r[c]) if c == "answers" else
                         repr(r[c]) if c == "rate" else r[c]) for c in cols])
    data_root = _data_root()
    manifest = {
        "scenario": b.name,
        "description": b.description,
        "status": b.status,
        "rule_gate": gate_block(b.cases),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "switches": b.switches,
        "counts": b.counts(),
        "n_cases": len(b.records),
        "n_responses": sum(r["n"] for r in b.records),
        "statistics": {"all": b.stats(),
                       **{d: b.stats(d) for d in b.switches["domain_order"]}},
        "thesis_reported": THESIS,
        "filter_log": b.log,
        "files": {CASES_FILE: sha256_of(cases_path), TABLE_FILE: sha256_of(table_path)},
        "inputs": {_rel(p, data_root): sha256_of(p) for p in b.inputs},
        "format": "JSON lines, one case per line: index, case_id, domain, "
                  "answers, correct, responses = [[v, c, [ps...]], ...]. "
                  "Deviation from the plan's parquet (docs/IMPLEMENTATION_DECISIONS.md W6).",
        "note": "No build is established as the thesis data set: every "
                "scenario is a hypothesis (docs/IMPLEMENTATION_DECISIONS.md, "
                "World-A section). The result of the rule-baseline check of "
                "this build is under rule_gate.",
    }
    with open(out_dir / MANIFEST_FILE, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    return out_dir


def gate_block(cases):
    """Rule-gate result (plan Step D5) of a list of cases, for the manifest.
    Tie policy and DA tie rule are the config defaults (thesis text)."""
    from cdm.eval import rule_gate as G
    res = G.run_gate(cases)
    summary = G.gate_summary(res)
    oq8b = res.da_oq8b()
    return {
        "tie_policy": res.policy, "tie_abs_tol": res.tol,
        "da_tie_rule": res.da_tie_rule,
        "passes": bool(summary["passes_t3b"]),
        "verdict": "MET" if summary["passes_t3b"] else "NOT MET",
        "cells_within_plus_minus_1": summary["cells_within_tolerance"],
        "cells_exact": summary["cells_exact"],
        "domains_passing": summary["domains_passing"],
        "n_cases": {d: res.n_cases(d) for d in G.DOMAINS},
        "n_cases_thesis": dict(G.THESIS_N),
        "per_domain": {m: {d: {"recomputed": res.count(m, d),
                               "thesis": G.THESIS_T3B[m][d],
                               "difference": res.count(m, d) - G.THESIS_T3B[m][d]}
                           for d in G.DOMAINS} for m in G.METHODS},
        "global": {m: {"recomputed": res.count(m), "thesis": G.THESIS_T3[m],
                       "difference": res.count(m) - G.THESIS_T3[m]}
                   for m in G.METHODS},
        "t9": {k: {"recomputed": res.t9()[k], "thesis": t,
                   "difference": res.t9()[k] - t} for k, t in G.THESIS_T9.items()},
        "da_depends_on_oq8b": oq8b,
    }


def _rel(p, root):
    try:
        return str(Path(p).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(p)


def load_frozen(path=None, scenario=None, verify=True):
    """Read a frozen build -> (list of Case, manifest dict).

    path : the build folder; default <build_root>/worldA/<scenario>.
    verify : check the SHA-256 of the cases file against the manifest.
    Cases are rebuilt through cdm.schema, so they are validated again.
    """
    if path is None:
        if scenario is None:
            raise ValueError("give a path or a scenario name")
        path = build_root() / "worldA" / resolve_scenario(scenario)
    path = Path(path)
    with open(path / MANIFEST_FILE, encoding="utf-8") as fh:
        manifest = json.load(fh)
    cases_path = path / CASES_FILE
    if verify and sha256_of(cases_path) != manifest["files"][CASES_FILE]:
        raise ValueError(f"{cases_path} does not match the hash in the manifest")
    cases = []
    with open(cases_path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            cases.append(Case(
                case_id=d["case_id"], answers=tuple(d["answers"]), correct=d["correct"],
                responses=tuple(Response(v=v, c=float(c), ps=tuple(float(x) for x in ps))
                                for v, c, ps in d["responses"]),
                domain=d["domain"]))
    return cases, manifest


# ---- scenario table ------------------------------------------------------------

def scenario_table(names=None, data_root=None, config=None):
    """One row per scenario: domain counts and the TH p.43 statistics."""
    config = config if config is not None else load_scenarios()
    rows = []
    for name in names or scenario_names(config):
        b = build(scenario=name, data_root=data_root, config=config, make_cases=False)
        c, s = b.counts(), b.stats()
        rows.append({"scenario": name,
                     **{d: c.get(d, 0) for d in b.switches["domain_order"]}, **s})
    return rows


def format_table(rows):
    head = ("| Scenario | General | NBA | NFL | Politics | Total | N mean | N SD pop/sample "
            "| N min | N max | Correct mean % | Correct SD pop/sample | min % | max % |")
    lines = [head, "|" + "---|" * 14,
             "| **Thesis p.43** | 500 | 393 | 212 | 104 | 1209 | 70 | 35.5 | 10 | 129 "
             "| 62 | 23.8 | 2.7 | 98.9 |"]
    for r in rows:
        lines.append(
            f"| {r['scenario']} | {r[DOMAIN_GENERAL]} | {r[DOMAIN_NBA]} | {r[DOMAIN_NFL]} "
            f"| {r[DOMAIN_POLITICS]} | {r['n_cases']} | {r['n_mean']:.2f} "
            f"| {r['n_sd_pop']:.2f} / {r['n_sd_sample']:.2f} | {r['n_min']} | {r['n_max']} "
            f"| {r['rate_mean']:.2f} | {r['rate_sd_pop']:.2f} / {r['rate_sd_sample']:.2f} "
            f"| {r['rate_min']:.2f} | {r['rate_max']:.2f} |")
    return "\n".join(lines)


#: Written under the scenario table.
TABLE_NOTE = """Every scenario is a hypothesis; none is established as the thesis data set.

- No 1209-case scenario gives all printed statistics. The 1209-case builds give N SD 35.6 and correct-rate SD 23.7 (thesis-reported: 35.5 and 23.8). The stated-filters build (1222 cases) gives both printed SDs; its case counts and its minimum rate of 1.64% differ from the thesis-reported ones (2.7%).
- In a random search, after the three NBA games at 1/61 are removed, 3 of 200,000 random choices of (4 NFL games, 6 races) give both printed SDs with the population SD and 0 with the sample SD. The printed minimum (2.7%) and the printed rate SD (23.8) pull in opposite directions.
- The three case-selection rules (NFL header tokens, election header names or third-option votes, NBA 1/61 games or two participants) are hypotheses; the sources do not state them. For Politics two disjoint sets of six races give 104.
- On every scenario some rule-baseline counts differ from the thesis-reported counts by more than the tolerance of the check (plan Step D5; docs/RULE_BASELINE_CHECK.md); each frozen manifest holds its own result."""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", action="append",
                    help="scenario to build and freeze (repeatable); 'all' = every "
                         "scenario that keeps complete responses")
    ap.add_argument("--table", action="store_true",
                    help="print the scenario table and write it under the build root")
    ap.add_argument("--out", help="build root (default: CDM_BUILD_ROOT or <repo>/build)")
    args = ap.parse_args(argv)
    root = Path(args.out).resolve() if args.out else build_root()
    config = load_scenarios()
    names = args.scenario or []
    if "all" in names:
        names = [n for n in scenario_names(config)
                 if scenario_switches(n, config)["complete_responses_only"]]
    for name in names:
        b = build(scenario=name, config=config)
        out = freeze(b, root / "worldA" / b.name)
        print(f"{name}: {b.counts()} total {len(b.records)} -> {out}")
    if args.table:
        rows = scenario_table(config=config)
        text = format_table(rows)
        (root / "worldA").mkdir(parents=True, exist_ok=True)
        (root / "worldA" / "scenario_table.md").write_text(
            text + "\n\n" + TABLE_NOTE + "\n", encoding="utf-8")
        with open(root / "worldA" / "scenario_table.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        print(text)


if __name__ == "__main__":
    main()
