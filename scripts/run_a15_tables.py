#!/usr/bin/env python
"""A15 driver: the AMP effectiveness tables and the per-domain success
tables, next to the thesis values (plan §9.3 row A15; TH Tables 5.4, 5.5,
A.9, A.10; plan items T8, T10-T13, T20, T21).

  python scripts/run_a15_tables.py --build waiver_main

NO MODEL IS FITTED. The script reads the per-case files that the A13 and
A14 drivers wrote:
  <runs-root>/<build>/a13_seed<k>/{amp,dap}_loo_cases.csv      (single LOO)
  <runs-root>/<build>/a14/results/{amp,dap}_nested_cases.csv   (10-fold)
A file that does not exist yet is reported as missing and its tables are
left out; the script can be run again later.

The build is still checked with the gate guard (cdm.eval.gate_guard), and
an A13 / A14 summary written for another frozen cases file is refused.

Outputs, under <runs-root>/<build>/a15/ (copied to <results-root>/<build>/a15/ on request):
  a15_tables.md          every table, this run and thesis side by side,
                         per domain; rule baselines recomputed vs printed
  a15_effectiveness.csv  the same cells in long format
  a15_summary.json       all numbers, comparisons and qualitative checks
  run.log

--loo-seeds 0,1,2,3,4 : classifier seeds of the A13 runs to tabulate; the
                        first one found is the primary table (seed 0,
                        AM18), the others are listed as seed noise.
--smoke N : end-to-end test on N synthetic cases with RANDOM selections
            (no real data, no fit); written to <runs-root>/_smoke/a15/,
            never to the results folder.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

import _common as C
from cdm.eval import effectiveness as E
from cdm.eval import loo as L
from cdm.report import tables as R

DIFFERENCE_NOTE = (
    "Note: under the documented reconstruction configuration, the values in the `run` "
    "columns differ from the thesis-reported values in several blocks; the differences "
    "are shown cell by cell. The present reconstruction does not establish the reason. "
    "Nothing is tuned.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build", default="waiver_main")
    p.add_argument("--build-root", default=str(C.DEFAULT_BUILD_ROOT))
    p.add_argument("--runs-root", default=str(C.DEFAULT_RUNS_ROOT))
    p.add_argument("--results-root", default=str(C.DEFAULT_RESULTS_ROOT),
                   help="folder to copy the small result files to (e.g. results); "
                        "default '' = do not copy")
    p.add_argument("--loo-seeds", default="0,1,2,3,4")
    p.add_argument("--a14-dir", default=None,
                   help="default <runs-root>/<build>/a14/results")
    p.add_argument("--out-dir", default=None, help="default <runs-root>/<build>/a15")
    p.add_argument("--smoke", type=int, default=0, metavar="N")
    return p.parse_args(argv)


# ---- synthetic inputs for --smoke ---------------------------------------------------

def write_smoke_inputs(root, n, seeds):
    """Per-case files of RANDOM results (no fit) in the layout of the A13 /
    A14 drivers, for an end-to-end test of this script."""
    inst, _ = C.smoke_instances(n)
    rng = np.random.default_rng(0)
    amp, dap = L.amp_inputs(inst), L.dap_inputs(inst)
    spec_a = L.make_spec("AMP", ("BR", "RF"), amp["labels"],
                         hyper=C.hyper_for_seed(0, smoke=True))
    spec_d = L.make_spec("DAP", "RF", hyper=C.hyper_for_seed(0, smoke=True))
    folds = np.repeat(np.arange(1, 11), int(np.ceil(n / 10)))[:n]

    def results(protocol):
        ra = L.amp_result(spec_a, rng.random((n, len(amp["labels"]))), amp["Y"], protocol,
                          case_ids=amp["case_ids"], domains=amp["domains"],
                          method_answers=amp["method_answers"],
                          folds=folds if protocol == "nested_cv" else None)
        p1 = rng.random(n)
        rd = L.dap_result(spec_d, np.column_stack([(p1 > 0.5).astype(float), p1]),
                          dap["y"], protocol, case_ids=dap["case_ids"],
                          domains=dap["domains"], answer_sets=dap["answer_sets"],
                          rule_outcomes=dap["rule_outcomes"],
                          folds=folds if protocol == "nested_cv" else None)
        return ra, rd

    for k in seeds:
        d = root / f"a13_seed{k}"
        d.mkdir(parents=True, exist_ok=True)
        ra, rd = results("loo")
        C.write_cases_csv(d / "amp_loo_cases.csv", ra)
        C.write_cases_csv(d / "dap_loo_cases.csv", rd)
    d = root / "a14" / "results"
    d.mkdir(parents=True, exist_ok=True)
    ra, rd = results("nested_cv")
    C.write_cases_csv(d / "amp_nested_cases.csv", ra)
    C.write_cases_csv(d / "dap_nested_cases.csv", rd)


# ---- reading -------------------------------------------------------------------------

def read_result(path, approach, protocol, missing):
    path = Path(path)
    if not path.is_file():
        missing.append(str(path))
        return None
    return E.eval_result_from_cases_csv(path, approach, protocol)


def check_same_build(summary_path, gate):
    """Refuse a summary that was written for another frozen cases file."""
    summary_path = Path(summary_path)
    if gate is None or not summary_path.is_file():
        return None
    doc = json.loads(summary_path.read_text(encoding="utf-8"))
    sha = (((doc.get("header") or {}).get("data") or {}).get("gate") or {}).get("cases_sha256")
    if sha is not None and sha != gate["cases_sha256"]:
        raise SystemExit(f"{summary_path} was written for another cases file "
                         f"({sha[:12]}... != {gate['cases_sha256'][:12]}...)")
    return bool(doc.get("complete", True))


def section(protocol, amp, dap, real, label):
    """Markdown text, CSV rows and the JSON part of one protocol."""
    out = {"label": label}
    text, rows = [], []
    name = "10-fold (leave-one-out within 10-fold CV)" if protocol == "nested_cv" \
        else "single leave-one-out"
    # success by domain (TH Table A.9 / T10)
    rep = {a: E.success_by_domain(r) for a, r in (("AMP", amp), ("DAP", dap)) if r is not None}
    th = {a: E.thesis_success_by_domain(a, protocol) for a in rep} if real else None
    if rep:
        src = ("TH Table A.9 p.115 (counts: plan T12)" if protocol == "loo" else
               "AMP: TH §5.1.0.1 p.68 (counts: plan T10); the thesis prints no DAP "
               "success per domain for the 10-fold protocol")
        text.append(R.success_by_domain_md(
            f"Success by domain, {name} [{label}]", rep, th,
            note=f"Cells: rate (successes). Thesis side: {src}."))
        out["success_by_domain"] = {"recomputed": rep, "thesis": th}
    if amp is not None:
        table = E.effectiveness_table(amp)
        th_table, sources = E.thesis_effectiveness(protocol) if real else (None, None)
        tabs = "TH Tables 5.4 and 5.5" if protocol == "nested_cv" else \
            "TH Table A.10 (and MLA Figs 2-3 for the global block)"
        text.append(R.effectiveness_md(
            f"Effectiveness of the AMP selection, {name} [{label}] - {tabs}",
            table, th_table, sources,
            note="P(success by AG) is the rule applied to every case of the block: the "
                 "recomputed rule baseline beside the thesis's. PRE from exact counts; p = "
                 "one-sample one-tailed z-test (plan §5.8 item 3), exact upper tail; the "
                 "thesis prints p for the global block only."))
        rows = R.effectiveness_csv_rows(label, table, th_table)
        out["effectiveness"] = table
        out["qualitative_checks"] = {"recomputed": E.qualitative_checks(table)}
        if real:
            out["thesis_effectiveness"] = th_table
            out["comparison"] = E.compare(table, th_table)
            out["qualitative_checks"]["thesis"] = E.qualitative_checks(th_table)
            rates = [c for c in out["comparison"] if "within_tolerance" in c
                     and c["within_tolerance"] is not None]
            out["rates_within_0.03"] = {
                "cells": len(rates), "within": sum(c["within_tolerance"] for c in rates),
                "outside": [f"{c['block']}/{c['method']}/{c['quantity']} {c['difference']:+.3f}"
                            for c in rates if not c["within_tolerance"]]}
        q = out["qualitative_checks"]
        lines = ["### Qualitative checks (plan §9.3 row A15)", "",
                 R.md_table(["check", "run", "thesis"], [
                     ["methods whose global p is not below 0.05",
                      ", ".join(q["recomputed"]["non_significant_methods"]) or "none",
                      ", ".join(q["thesis"]["non_significant_methods"]) or "none"
                      if real else R.DASH],
                     ["domain with the highest mean PRE",
                      q["recomputed"]["strongest_pre_domain"],
                      q["thesis"]["strongest_pre_domain"] if real else R.DASH],
                     ["mean PRE per domain",
                      ", ".join(f"{d} {R.rate(v)}" for d, v in
                                q["recomputed"]["domain_mean_pre"].items()),
                      ", ".join(f"{d} {R.rate(v)}" for d, v in
                                q["thesis"]["domain_mean_pre"].items()) if real else R.DASH],
                     ["methods with a negative PRE, per domain",
                      "; ".join(f"{d}: {', '.join(v) or 'none'}" for d, v in
                                q["recomputed"]["negative_pre"].items()),
                      "; ".join(f"{d}: {', '.join(v) or 'none'}" for d, v in
                                q["thesis"]["negative_pre"].items()) if real else R.DASH]])]
        if real:
            w = out["rates_within_0.03"]
            lines += ["", f"Rates within ±0.03 of the thesis: {w['within']} of {w['cells']} "
                          "cells (P(selected), P(success), P(success given selected); "
                          "all blocks)."]
        text.append("\n".join(lines) + "\n")
    return "\n".join(text), rows, out


def main(argv=None):
    args = parse_args(argv)
    real = not args.smoke
    build = args.build if real else "_smoke"
    seeds = [int(s) for s in args.loo_seeds.split(",") if s.strip() != ""]
    root = Path(args.runs_root) / build
    out_dir = Path(args.out_dir) if args.out_dir else root / "a15"
    out_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(out_dir / "run.log")
    C.LOG.info("A15 tables start: %s", vars(args))

    gate = None
    if real:
        gate = C.gate_guard.require_gate(Path(args.build_root) / args.build)
        C.LOG.info("gate guard: build %s is %s (%s)", gate["build"], gate["status"],
                   gate["reference"])
    else:
        write_smoke_inputs(root, args.smoke, seeds[:2] or [0])
        seeds = seeds[:2] or [0]
    a14 = Path(args.a14_dir) if args.a14_dir else root / "a14" / "results"

    missing, incomplete = [], []
    summary = {"step": "A15 effectiveness and per-domain tables (no model fitted)",
               "written": C.utc_now(), "argv": sys.argv, "build": build, "gate": gate,
               "not_a_result": not real, "sections": {}}
    texts = [f"# A15 tables: AMP effectiveness and success by domain - build `{build}`", "",
             f"Written {C.utc_now()} by scripts/run_a15_tables.py from the per-case files of "
             "the A13 / A14 runs; no model is fitted here."
             + ("" if real else "  **SYNTHETIC SMOKE TEST - NOT A RESULT.**"),
             "Rates are rounded half up to three decimals. \"run\" = this reconstruction, "
             "\"thesis\" = the printed value / the plan's back-solved counts.", ""]
    if real:
        texts += [f"Rule-baseline check: status `{gate['status']}` for `{gate['build']}` "
                  f"({gate['reference']}, {gate['amendment']}); frozen cases file "
                  f"{gate['cases_sha256'][:12]}....",
                  "", DIFFERENCE_NOTE, ""]
    csv_rows = []

    # -- 10-fold --
    if check_same_build(a14 / "amp_nested_summary.json", gate) is False:
        incomplete.append(str(a14 / "amp_nested_summary.json"))
    amp = read_result(a14 / "amp_nested_cases.csv", "AMP", "nested_cv", missing)
    dap = read_result(a14 / "dap_nested_cases.csv", "DAP", "nested_cv", missing)
    first = True
    if amp is not None or dap is not None:
        any_res = amp if amp is not None else dap
        texts.append(R.rule_baselines_md(E.rule_baselines_of_result(any_res, real)))
        first = False
        t, rows, part = section("nested_cv", amp, dap, real, "10-fold, seed 0")
        texts.append(t)
        csv_rows += rows
        summary["sections"]["nested_cv"] = part

    # -- single LOO, per seed --
    spread = []
    primary_done = False
    for k in seeds:
        d = root / f"a13_seed{k}"
        if check_same_build(d / "summary.json", gate) is False:
            incomplete.append(str(d / "summary.json"))
        amp = read_result(d / "amp_loo_cases.csv", "AMP", "loo", missing)
        dap = read_result(d / "dap_loo_cases.csv", "DAP", "loo", missing)
        if amp is None and dap is None:
            continue
        if first:
            any_res = amp if amp is not None else dap
            texts.append(R.rule_baselines_md(E.rule_baselines_of_result(any_res, real)))
            first = False
        t, rows, part = section("loo", amp, dap, real, f"LOO, classifier seed {k}")
        summary["sections"][f"loo_seed{k}"] = part
        csv_rows += rows[1:] if csv_rows else rows
        if not primary_done:
            texts.append(t)
            primary_done = True
        else:
            texts.append(f"<details><summary>Single LOO, classifier seed {k} "
                         f"(seed noise)</summary>\n\n{t}\n</details>\n")
        row = {"seed": k}
        for a, r in (("AMP", amp), ("DAP", dap)):
            if r is not None:
                row[a] = {d: v["successes"] for d, v in E.success_by_domain(r).items()}
        spread.append(row)
    if len(spread) > 1:
        blocks = [E.ALL] + [b for b in next(v for v in spread[0].values()
                                            if isinstance(v, dict)) if b != E.ALL]
        head = ["seed"] + [f"{a} {('all' if b == E.ALL else b)}"
                           for a in ("AMP", "DAP") for b in blocks]
        body = [[r["seed"]] + [r.get(a, {}).get(b, R.DASH) for a in ("AMP", "DAP")
                               for b in blocks] for r in spread]
        texts.append("## Single LOO: successes by classifier seed (AM18: seed noise)\n\n"
                     + R.md_table(head, body) + "\n")
    summary["loo_seed_spread"] = spread
    summary["missing_inputs"] = missing
    summary["incomplete_inputs"] = incomplete
    if missing:
        texts.append("## Inputs not found (their tables are left out)\n\n"
                     + "\n".join(f"- `{m}`" for m in missing) + "\n")
    if incomplete:
        texts.append("## Inputs from runs that had not finished\n\n"
                     + "\n".join(f"- `{m}`" for m in incomplete) + "\n")

    C.write_text(out_dir / "a15_tables.md", "\n".join(texts))
    tmp = out_dir / f".a15_effectiveness.{os.getpid()}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    os.replace(tmp, out_dir / "a15_effectiveness.csv")
    summary["complete"] = True
    C.write_json(out_dir / "a15_summary.json", summary)
    files = [out_dir / n for n in ("a15_tables.md", "a15_effectiveness.csv",
                                   "a15_summary.json", "run.log")]
    dest = str(Path(args.results_root) / build / "a15") + "/" \
        if real and args.results_root else None
    C.LOG.info("A15 tables written to %s%s; sections: %s; missing inputs: %d", out_dir,
               f"; copied to {dest}" if dest else "", list(summary["sections"]), len(missing))
    C.copy_results(files, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
