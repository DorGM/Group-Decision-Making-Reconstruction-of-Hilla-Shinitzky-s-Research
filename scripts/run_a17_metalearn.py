#!/usr/bin/env python
"""A17 driver: classic meta-learning: a decision tree on the answers of
the aggregation methods only, leave-one-out, the five rows of TH Table 5.8
(plan §5.5, §8 OQ21; plan item T16), and the rules of the trees as text.

  python scripts/run_a17_metalearn.py --build waiver_main

The build must be listed as passed or waived in config/gate_status.yaml
(cdm.eval.gate_guard); any other build is refused before it is loaded.

Outputs, under <runs-root>/<build>/a17/ (copied to <results-root>/<build>/a17/ on request):
  a17_metalearn.md        Table 5.8 beside the thesis, per domain, the
                          expected relations, rule baselines
  a17_metalearn.csv       one line per row
  a17_tree_rules.txt      per row: the tree fitted on all cases (sklearn
                          text export) and its answer-pattern table
  a17_row<r>_cases.csv    one line per case
  a17_summary.json, run.log
The run takes seconds; there is no checkpoint (a repeated run recomputes).

--seed k  : random_state of the tree (default: plan OQ21, 0); a seed other
            than the default is written to a17_seed<k>/.
--smoke N : end-to-end test on N synthetic cases; written to
            <runs-root>/_smoke/a17/, never to the results folder.
--limit N : pipeline check on a sub-sample of a real build; never written
            to the results folder.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import _common as C
from cdm.eval import effectiveness as E
from cdm.eval import metalearn as ML
from cdm.eval import stats as S
from cdm.report import tables as R


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build", default="waiver_main")
    p.add_argument("--build-root", default=str(C.DEFAULT_BUILD_ROOT))
    p.add_argument("--seed", type=int, default=None, help="tree random_state (OQ21: 0)")
    p.add_argument("--rows", default=",".join(str(r) for r in sorted(ML.ROWS)))
    p.add_argument("--runs-root", default=str(C.DEFAULT_RUNS_ROOT))
    p.add_argument("--results-root", default=str(C.DEFAULT_RESULTS_ROOT),
                   help="folder to copy the small result files to (e.g. results); "
                        "default '' = do not copy")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--smoke", type=int, default=0, metavar="N")
    p.add_argument("--limit", type=int, default=0, metavar="N")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = sorted({int(r) for r in args.rows.split(",") if r.strip()})
    if any(r not in ML.ROWS for r in rows):
        raise SystemExit("--rows: numbers in 1..5")
    real = not args.smoke
    build = args.build if real else "_smoke"
    settings = ML.settings_from_config()
    seed = settings["random_state"] if args.seed is None else args.seed
    tag = "a17" + (f"_seed{seed}" if seed != settings["random_state"] else "") \
        + (f"_limit{args.limit}" if args.limit and real else "")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.runs_root) / build / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(out_dir / "run.log")
    C.LOG.info("A17 classic meta-learning start: %s", vars(args))

    if real:
        inst, data = C.load_real_build(args.build, args.build_root, args.limit or None)
    else:
        inst, data = C.smoke_instances(args.smoke)
    full = real and not args.limit
    n = len(inst)
    t0 = time.time()
    results, rules = {}, {}
    files = []
    for row in rows:
        results[row] = ML.run_row(inst, row, seed)
        rules[row] = ML.tree_rules(inst, row, seed)
        path = out_dir / f"a17_row{row}_cases.csv"
        C.write_cases_csv(path, results[row])
        files.append(path)
        C.LOG.info("A17 row %d %s: %d/%d = %.3f (thesis %.3f%s)", row,
                   results[row].meta["features"], results[row].n_success, n,
                   results[row].success_rate, ML.thesis_rate(row),
                   "" if full else "; other data")
    expect = ML.expectations(results, inst)

    head = ["row", "f_MR", "f_WC, f_HAC", "f_SP", "f_DA", "successes", "rate: run",
            "thesis", "diff", "50/50 leaves used"]
    body, csv_rows = [], [["row", "f_MR", "f_WC_f_HAC", "f_SP", "f_DA", "successes", "n",
                           "rate", "thesis_rate", "difference", "half_rows"]]
    summary_rows = {}
    for row in rows:
        res = results[row]
        flags = ML.ROWS[row]
        rounded = S.round_half_up(res.n_success, n)
        half = res.tie_log()["half_rows"]
        body.append([row, "V"] + ["V" if f else "" for f in flags]
                    + [f"{res.n_success}/{n}", R.rate(rounded),
                       R.rate(ML.thesis_rate(row)) if full else R.DASH,
                       R.signed(rounded - ML.thesis_rate(row)) if full else R.DASH, half])
        csv_rows.append([row, 1] + list(flags) + [
            res.n_success, n, format(rounded, ".3f"),
            format(ML.thesis_rate(row), ".3f") if full else "",
            format(rounded - ML.thesis_rate(row), "+.3f") if full else "", half])
        summary_rows[row] = {
            "features": res.meta["features"], "rules_heard_by_DA": res.meta["rules_heard_by_DA"],
            "successes": res.n_success, "n": n, "rate": res.success_rate, "rounded": rounded,
            "thesis_rate": ML.thesis_rate(row) if full else None,
            "half_rows": half, "domains": E.success_by_domain(res),
            "tree": {k: rules[row][k] for k in ("depth", "n_leaves", "patterns")}}

    blocks = list(next(iter(summary_rows.values()))["domains"])
    dom_head = ["row"] + [f"{'all' if b == E.ALL else b} (n = "
                          f"{summary_rows[rows[0]]['domains'][b]['n']})" for b in blocks]
    dom_body = [[row] + [f"{summary_rows[row]['domains'][b]['successes']} "
                         f"({R.rate(summary_rows[row]['domains'][b]['rounded'])})"
                         for b in blocks] for row in rows]
    lines = []
    e = expect["row4_vs_MR"]
    if e:
        lines.append(f"- Row 4 (f_MR, f_DA) against MR applied to every case: {e['row4']} vs "
                     f"{e['MR']} successes ({'equal' if e['equal'] else 'different'}); "
                     "expected about equal (plan §5.5; thesis .687 = MR).")
    e = expect["rows_1_5_equal"]
    if e:
        lines.append(f"- Rows 1 and 5: {e['row1']} and {e['row5']} successes; same prediction "
                     f"for every case: {e['same_predictions']}; expected equal (f_DA is a "
                     "function of the other four; thesis .713 = .713).")
    e = expect["best_vs_SP"]
    if e:
        lines.append(f"- Best row ({e['best_row']}): {e['best']} successes; SP applied to "
                     f"every case: {e['SP']}; best below SP: {e['best_below_SP']} "
                     "(thesis: .713 < .715).")
    text = [f"# A17 classic meta-learning (decision tree on the rule answers) - build `{build}`",
            "", f"Written {C.utc_now()}. `DecisionTreeClassifier` with default settings, "
            f"random_state {seed}; leave-one-out over {n} cases (plan §8 OQ21: the thesis "
            "does not state the protocol). Features: the 0/1 codes of the methods' answers "
            "(1 = the first answer of the case); f_DA is recomputed over the methods of the "
            "row. Thesis side: TH Table 5.8 p.73."
            + ("" if full else "  **NOT A RESULT: smoke / limited run.**"), ""]
    if full:
        text += ["Note: the rule baselines of this build differ from the thesis-reported "
                 "counts by a few cases (below). Where the rows of this run differ from the "
                 "thesis-reported rows, the present reconstruction does not establish the "
                 "reason. Nothing is tuned.", ""]
    text += [R.rule_baselines_md(data["rule_baselines_by_domain"]),
             "## Table 5.8", "", R.md_table(head, body), "",
             "## Expected relations (plan §9.3 row A17)", ""] + lines + [
             "", "## Successes per domain and row (no thesis counterpart)", "",
             R.md_table(dom_head, dom_body), "",
             "The trees' rules are in `a17_tree_rules.txt`.", ""]
    C.write_text(out_dir / "a17_metalearn.md", "\n".join(text))

    rule_text = [f"A17 classic meta-learning: rules of the decision trees, build {build}"
                 + ("" if full else "   [NOT A RESULT: smoke / limited run]"),
                 "Each tree is fitted on ALL cases (the LOO trees differ from it by one case).",
                 "Coding: f_<method> = 1 when the method chose the first answer (a1) of the "
                 "case, 0 when it chose the other (a2); class 1 = the correct answer is a1.",
                 "weights = [cases with a* = a2, cases with a* = a1] in the leaf.", ""]
    for row in rows:
        r = rules[row]
        rule_text += [f"=== Row {row}: {', '.join(r['features'])}  (depth {r['depth']}, "
                      f"{r['n_leaves']} leaves; LOO success {results[row].n_success}/{n}) ===",
                      "", "Answer patterns that occur:", ML.format_patterns(r), "",
                      "sklearn export_text:", r["text"], ""]
    C.write_text(out_dir / "a17_tree_rules.txt", "\n".join(rule_text))

    tmp = out_dir / f".a17_metalearn.{os.getpid()}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    os.replace(tmp, out_dir / "a17_metalearn.csv")
    header = {"written": C.utc_now(), "argv": sys.argv, "data": data,
              "settings": {"OQ21": settings, "random_state_used": seed}}
    C.write_json(out_dir / "a17_summary.json", {
        "step": "A17 classic meta-learning (DecisionTree on f_AG only, LOO)",
        "header": header, "complete": True, "seconds": time.time() - t0,
        "rows": summary_rows, "expected_relations": expect})
    files += [out_dir / f for f in ("a17_metalearn.md", "a17_metalearn.csv",
                                    "a17_tree_rules.txt", "a17_summary.json", "run.log")]
    dest = str(Path(args.results_root) / build / tag) + "/" if full and args.results_root else None
    C.LOG.info("A17 finished in %.0f s; outputs in %s%s", time.time() - t0, out_dir,
               f"; copied to {dest}" if dest else "")
    C.copy_results(files, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
