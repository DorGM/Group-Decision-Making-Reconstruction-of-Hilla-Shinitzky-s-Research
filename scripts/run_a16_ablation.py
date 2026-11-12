#!/usr/bin/env python
"""A16 driver: the ablation experiments of AMP (BR+RF) and DAP (RF): a
leave-one-out per row of the 11-row matrix of plan §5.4 (TH Tables 5.6 /
5.7; plan items T14, T15), with the two-tailed proportion test of every row
against the full version (row 1).

  python scripts/run_a16_ablation.py --build waiver_main --n-jobs 4

The build must be listed as passed or waived in config/gate_status.yaml
(cdm.eval.gate_guard); any other build is refused before it is loaded.

Outputs, under <runs-root>/<build>/a16/  (seed 0, OQ41 default; otherwise
a16_seed<k>, a16_oq41alt):
  ckpt/<approach>/row_<rr>/     checkpoints of every LOO
  a16_ablation.md               both tables beside the thesis rows, the
                                significance pattern, successes per domain,
                                rule baselines recomputed vs printed
  a16_<approach>_ablation.csv   one line per row
  a16_<approach>_row<rr>_cases.csv   one line per case
  a16_summary.json, status.json, run.log
The small files (never the checkpoints) are copied to
<results-root>/<build>/<tag>/ after every finished row.

RESUME: start the same command again; finished batches are read from
ckpt/ (another build, seed, setting or code version is refused).

--rows 1,8,11 : a sub-set of rows (row 1 is needed for the tests).
--oq41 X      : plan §8 OQ41 option for features 16/17 (default: config).
--smoke N     : end-to-end test on N synthetic cases (no real data, 10
                trees); written to <runs-root>/_smoke/, never to results.
--limit N     : pipeline check on a sub-sample of a real build; never
                written to the results folder.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import _common as C
from cdm.approaches import instances as I
from cdm.eval import ablation as A
from cdm.eval import effectiveness as E
from cdm.eval import loo as L
from cdm.report import tables as R

COMPONENT_HEAD = {"conf": "Conf feats", "ps": "PS feats", "wc_hac": "WC,HAC", "sp": "SP",
                  "da": "DA"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build", default="waiver_main")
    p.add_argument("--build-root", default=str(C.DEFAULT_BUILD_ROOT))
    p.add_argument("--seed", type=int, default=0, help="classifier seed")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=L.DEFAULT_BATCH_SIZE)
    p.add_argument("--approaches", default="DAP,AMP", help="run order")
    p.add_argument("--rows", default=",".join(str(r) for r in A.ROWS))
    p.add_argument("--oq41", default=None, choices=I.OQ41_OPTIONS)
    p.add_argument("--runs-root", default=str(C.DEFAULT_RUNS_ROOT))
    p.add_argument("--results-root", default=str(C.DEFAULT_RESULTS_ROOT),
                   help="folder to copy the small result files to (e.g. results); "
                        "default '' = do not copy")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--smoke", type=int, default=0, metavar="N")
    p.add_argument("--limit", type=int, default=0, metavar="N")
    return p.parse_args(argv)


def table_md(approach, table, full):
    head = ["row"] + [COMPONENT_HEAD[c] for c in I.ABLATION_COLUMNS] + [
        "successes", "rate: run", "thesis" if full else "thesis (other data)", "diff",
        "z vs row 1", "p (two-tailed)", "significant: run", "thesis"]
    rows = []
    for r in table:
        rows.append([r["row"]] + ["V" if r["components"][c] else "" for c in I.ABLATION_COLUMNS]
                    + [f"{r['successes']}/{r['n']}", R.rate(r["rounded"]),
                       R.rate(r["thesis_rate"]) if full else R.DASH,
                       R.signed(r["difference"]) if full else R.DASH,
                       R.rate(r["z"], 2), R.pvalue(r["p"]),
                       R.DASH if r["significant"] is None else ("yes" if r["significant"] else "no"),
                       ("yes" if r["thesis_significant"] else "no") if r["row"] != 1 else R.DASH])
    return R.md_table(head, rows)


def domain_md(domains_by_row):
    blocks = list(next(iter(domains_by_row.values())))
    head = ["row"] + [f"{'all' if b == E.ALL else b} (n = "
                      f"{next(iter(domains_by_row.values()))[b]['n']})" for b in blocks]
    rows = [[row] + [f"{d[b]['successes']} ({R.rate(d[b]['rounded'])})" for b in blocks]
            for row, d in sorted(domains_by_row.items())]
    return R.md_table(head, rows)


def main(argv=None):
    args = parse_args(argv)
    approaches = [a.strip().upper() for a in args.approaches.split(",") if a.strip()]
    if not approaches or any(a not in ("AMP", "DAP") for a in approaches):
        raise SystemExit("--approaches: a comma-separated order of AMP and DAP")
    rows = sorted({int(r) for r in args.rows.split(",") if r.strip()})
    if any(r not in A.ROWS for r in rows):
        raise SystemExit(f"--rows: numbers in {A.ROWS[0]}..{A.ROWS[-1]}")
    real = not args.smoke
    build = args.build if real else "_smoke"
    oq41 = args.oq41 or I.oq41_from_config()
    tag = "a16" + (f"_seed{args.seed}" if args.seed else "") \
        + ("_oq41alt" if oq41 != I.OQ41_OPTIONS[0] else "") \
        + (f"_limit{args.limit}" if args.limit and real else "")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.runs_root) / build / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(out_dir / "run.log")
    C.LOG.info("A16 ablation start (pid %d): %s", os.getpid(), vars(args))

    if real:
        inst, data = C.load_real_build(args.build, args.build_root, args.limit or None)
    else:
        inst, data = C.smoke_instances(args.smoke)
    full = real and not args.limit
    hyper = C.hyper_for_seed(args.seed, smoke=not real)
    header = C.run_header(args, data, hyper)
    dest = str(Path(args.results_root) / build / tag) + "/" if full and args.results_root else None
    n = len(inst)
    t_start = time.time()
    counts = {a: {} for a in approaches}
    domains = {a: {} for a in approaches}
    details = {a: {} for a in approaches}
    files = set()
    status = {"state": "running", "pid": os.getpid(), "started": C.utc_now(),
              "out_dir": str(out_dir), "not_a_result": not full, "done": []}

    def a13_check(approach):
        """Row 1 is the A13 run: compare the success count when the A13
        summary of the same build and seed exists."""
        f = Path(args.runs_root) / build / f"a13_seed{args.seed}" / "summary.json"
        if not full or not f.is_file() or 1 not in counts[approach]:
            return None
        doc = json.loads(f.read_text(encoding="utf-8"))
        k = (doc.get(approach) or {}).get("successes")
        return None if k is None else {"a13_successes": k,
                                       "row1_successes": counts[approach][1],
                                       "equal": k == counts[approach][1]}

    def write_all(final=False):
        summary = {"step": "A16 ablations (LOO; AMP BR+RF, DAP RF)", "header": header,
                   "oq41": oq41, "rows_requested": rows, "complete": final,
                   "seconds": time.time() - t_start}
        texts = [f"# A16 ablation experiments - build `{build}`, classifier seed {args.seed}",
                 "", f"Written {C.utc_now()}. Leave-one-out over {n} cases per row; AMP = "
                 "BR+RF, DAP = RF (TH p.70). p = two-tailed pooled two-proportion z-test "
                 "against row 1 (TH p.70; plan §5.4); significant = p <= 0.05. The thesis "
                 "prints no p-values (AM2): its column gives the rows it calls significant. "
                 f"Features 16/17 (OQ41): `{oq41}`."
                 + ("" if full else "  **NOT A RESULT: smoke / limited run.**"), ""]
        if full:
            texts += [DIFFERENCE_NOTE, ""]
        texts.append(R.rule_baselines_md(data["rule_baselines_by_domain"]))
        for a in approaches:
            if not counts[a]:
                continue
            table = A.ablation_table(a, counts[a], n)
            pattern = A.significance_pattern(table)
            check = a13_check(a)
            summary[a] = {"table": table, "pattern": pattern, "domains": domains[a],
                          "rows": details[a], "row1_equals_a13": check}
            texts += [f"## {a} ({'TH Table 5.6' if a == 'AMP' else 'TH Table 5.7'})", "",
                      table_md(a, table, full), "",
                      f"Significant rows: run {pattern['significant_rows_run']} (below row 1: "
                      f"{pattern['significantly_below_full_run']}); thesis "
                      f"{pattern['significant_rows_thesis']}; same pattern: "
                      f"{'rows missing' if pattern['same_pattern'] is None else pattern['same_pattern']}.",
                      ""]
            if check:
                texts += [f"Row 1 against the A13 run of seed {args.seed}: "
                          f"{check['row1_successes']} vs {check['a13_successes']} "
                          f"({'equal' if check['equal'] else 'DIFFERENT'}).", ""]
            texts += [f"### {a}: successes per domain and row (no thesis counterpart)", "",
                      domain_md(domains[a]), ""]
            path = out_dir / f"a16_{a.lower()}_ablation.csv"
            tmp = out_dir / f".{path.name}.{os.getpid()}.tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["row"] + list(I.ABLATION_COLUMNS) + [
                    "successes", "n", "rate", "thesis_rate", "difference", "z", "p_two_tailed",
                    "significant", "thesis_significant"])
                for r in table:
                    w.writerow([r["row"]] + [r["components"][c] for c in I.ABLATION_COLUMNS]
                               + [r["successes"], r["n"], format(r["rounded"], ".3f"),
                                  format(r["thesis_rate"], ".3f") if full else "",
                                  format(r["difference"], "+.3f") if full else "",
                                  "" if r["z"] is None else repr(r["z"]),
                                  "" if r["p"] is None else repr(r["p"]),
                                  "" if r["significant"] is None else int(r["significant"]),
                                  int(r["thesis_significant"]) if full else ""])
            os.replace(tmp, path)
            files.add(path)
        C.write_text(out_dir / "a16_ablation.md", "\n".join(texts))
        C.write_json(out_dir / "a16_summary.json", summary)
        status.update(updated=C.utc_now(), elapsed_seconds=time.time() - t_start,
                      state="finished" if final else "running")
        C.write_json(out_dir / "status.json", status)
        files.update(out_dir / f for f in ("a16_ablation.md", "a16_summary.json",
                                           "status.json", "run.log"))
        C.copy_results(sorted(files), dest)

    for a in approaches:
        for row in rows:
            t0 = time.time()
            res, desc = A.run_row(inst, a, row, hyper=hyper, oq41=oq41, n_jobs=args.n_jobs,
                                  batch_size=args.batch_size,
                                  checkpoint_dir=out_dir / "ckpt" / a.lower() / f"row_{row:02d}")
            counts[a][row] = res.n_success
            domains[a][row] = E.success_by_domain(res)
            desc.update(seconds=time.time() - t0, ties=res.tie_log(),
                        n_from_checkpoint=res.meta.get("n_from_checkpoint"),
                        lr_not_converged=res.meta.get("lr_not_converged"))
            if a == "AMP":
                desc["selected"] = res.selection_counts()
                desc["selected_success"] = res.selected_success_counts()
            details[a][row] = desc
            path = out_dir / f"a16_{a.lower()}_row{row:02d}_cases.csv"
            C.write_cases_csv(path, res)
            files.add(path)
            status["done"].append([a, row])
            C.LOG.info("A16 %s row %d %s: %d/%d = %.3f (thesis %.3f%s), %d columns, "
                       "methods %s, %.0f s", a, row, desc["components"], res.n_success, n,
                       res.success_rate, A.thesis_rate(a, row),
                       "" if full else "; other data", desc["n_columns"], desc["methods"],
                       desc["seconds"])
            write_all()
    write_all(final=True)
    C.LOG.info("A16 ablation finished in %.0f s; outputs in %s%s", time.time() - t_start,
               out_dir, f"; copied to {dest}" if dest else "")
    return 0


DIFFERENCE_NOTE = (
    "Note: under the documented reconstruction configuration, the success rates of this "
    "run differ from the thesis-reported rates; both are shown in every row. The present "
    "reconstruction does not establish the reason. Nothing is tuned.")

if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException as exc:                   # leave a trace in the log
        if not isinstance(exc, SystemExit) or exc.code not in (0, None):
            C.LOG.exception("A16 ablation stopped: %r", exc)
            C.LOG.info("finished batches are in the checkpoint folder; start the same "
                       "command again to resume")
        raise
