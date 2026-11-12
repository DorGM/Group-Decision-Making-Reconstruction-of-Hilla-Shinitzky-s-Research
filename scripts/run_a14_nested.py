#!/usr/bin/env python
"""A14 driver: leave-one-out within 10-fold cross-validation with technique
selection (plan §5.2; TH §4.1.4, Tables 5.1 / 5.2; plan items T1, T2).

  python scripts/run_a14_nested.py --build waiver_main --n-jobs 4

DAP (KNN, LR, RF) runs first because it is short, then AMP (9 techniques).
The build must be listed as passed or waived in config/gate_status.yaml
(cdm.eval.gate_guard); any other build is refused before it is loaded.

Outputs, under <runs-root>/<build>/a14/  (seed 0; a14_seed<k> otherwise):
  ckpt/<approach>/fold_<k>/<technique>/   checkpoints of every inner LOO
  results/progress.log                    progress log (appended)
  results/status.json                     state, pid, what is running, ETA
  results/<approach>_nested_partial.json  the folds finished so far
  results/<approach>_nested_table.csv / .txt   the table (TH Table 5.1 / 5.2
                                          shape) with the thesis rows beside
  results/<approach>_nested_summary.json  final counts with the thesis-reported values
  results/<approach>_nested_cases.csv     one row per case
After every finished fold and at the end, the files in results/ (small;
never the checkpoints) are copied to <results-root>/<build>/a14/ when --results-root is given.

RESUME: start the same command again. Finished inner-LOO batches are read
from ckpt/ (a different build, seed, setting or code version is refused,
never mixed); only the missing batches are computed.

--smoke N : end-to-end test on N synthetic cases (no real data, 10 trees);
            written to <runs-root>/_smoke/a14/, never to the results folder.
--limit N : pipeline check on an evenly spaced sub-sample of a real build;
            written to a14_limit<N>/, never to the results folder.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import _common as C
from cdm.eval import loo as L
from cdm.eval import nested_cv as N
from cdm.eval.stats import round_half_up
from cdm.report import thesis_values as T


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build", default="waiver_main")
    p.add_argument("--build-root", default=str(C.DEFAULT_BUILD_ROOT))
    p.add_argument("--seed", type=int, default=0, help="classifier seed")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=L.DEFAULT_BATCH_SIZE)
    p.add_argument("--approaches", default="DAP,AMP", help="run order")
    p.add_argument("--n-splits", type=int, default=None, help="default: config OQ16 (10)")
    p.add_argument("--runs-root", default=str(C.DEFAULT_RUNS_ROOT))
    p.add_argument("--results-root", default=str(C.DEFAULT_RESULTS_ROOT),
                   help="folder to copy the small result files to (e.g. results); "
                        "default '' = do not copy")
    p.add_argument("--out-dir", default=None,
                   help="default <runs-root>/<build>/a14[_seed<k>][_limit<N>]")
    p.add_argument("--smoke", type=int, default=0, metavar="N")
    p.add_argument("--limit", type=int, default=0, metavar="N")
    return p.parse_args(argv)


def fold_dict(f):
    """Plain dict of a cdm.eval.nested_cv.FoldResult (no per-case arrays)."""
    return {"fold": f.fold, "n_train": f.n_train, "n_test": f.n_test,
            "inner_successes": dict(f.inner_successes),
            "inner_rates": f.inner_rates(),
            "selected": f.selected, "tied": list(f.tied),
            "test_successes": f.test_successes, "test_rate": f.test_rate,
            "test_successes_by_tied_technique": dict(f.tied_test_successes),
            "inner_not_converged": dict(f.inner_not_converged),
            "test_refit_not_converged": dict(f.test_not_converged),
            "inner_seconds": dict(f.inner_seconds),
            "inner_fresh_fits": dict(f.inner_fresh_fits)}


def write_table(res_dir, approach, techniques, folds, compare, total=None):
    """<approach>_nested_table.csv and .txt: one row per finished fold; the
    printed thesis row beside it when `compare` (a full real run)."""
    names_th, rows_th = T.thesis_nested_table(approach)
    same = compare and tuple(techniques) == tuple(names_th) and total == T.N_CASES
    head = ["fold", "source", "n_train", "n_test"] + list(techniques) \
        + ["test", "test_successes", "selected", "tied"]
    lines, rows = [], [head]
    w = max(8, max(len(t) for t in techniques) + 2)
    lines.append(f"{'':<16}" + "".join(f"{t:>{w}}" for t in techniques) + f"{'Test':>{w}}")
    for f in folds:
        rates = [round_half_up(f.inner_successes[t], f.n_train, 3) for t in techniques]
        test = round_half_up(f.test_successes, f.n_test, 3)
        rows.append([f.fold, "run", f.n_train, f.n_test]
                    + [format(r, ".3f") for r in rates]
                    + [format(test, ".3f"), f.test_successes, f.selected,
                       "+".join(f.tied) if f.is_tied else ""])
        lines.append(f"{'Fold ' + str(f.fold) + ' run':<16}" + "".join(
            f"{('*' if t == f.selected else '') + format(r, '.3f'):>{w}}"
            for t, r in zip(techniques, rates)) + f"{format(test, '.3f'):>{w}}"
            + (f"   TIE {list(f.tied)} -> test {f.tied_test_successes}" if f.is_tied else ""))
        if same:
            th = rows_th[f.fold - 1]
            best = max(th[:-1])
            rows.append([f.fold, "thesis", "", ""] + [format(x, ".3f") for x in th[:-1]]
                        + [format(th[-1], ".3f"), "",
                           "+".join(t for t, x in zip(names_th, th[:-1]) if x == best), ""])
            lines.append(f"{'       thesis':<16}" + "".join(
                f"{('*' if x == best else '') + format(x, '.3f'):>{w}}" for x in th[:-1])
                + f"{format(th[-1], '.3f'):>{w}}")
    lines.append("* = selected (highest inner score). Rates rounded half up to 3 decimals.")
    tmp = res_dir / f".{approach.lower()}_table.{os.getpid()}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    os.replace(tmp, res_dir / f"{approach.lower()}_nested_table.csv")
    C.write_text(res_dir / f"{approach.lower()}_nested_table.txt", "\n".join(lines) + "\n")


def main(argv=None):
    args = parse_args(argv)
    approaches = [a.strip().upper() for a in args.approaches.split(",") if a.strip()]
    if not approaches or any(a not in ("AMP", "DAP") for a in approaches):
        raise SystemExit("--approaches: a comma-separated order of AMP and DAP")
    real = not args.smoke
    build = args.build if real else "_smoke"
    tag = "a14" + (f"_seed{args.seed}" if args.seed else "") \
        + (f"_limit{args.limit}" if args.limit and real else "")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.runs_root) / build / tag
    res_dir = out_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(res_dir / "progress.log")
    C.LOG.info("A14 nested CV start (pid %d): %s", os.getpid(), vars(args))

    if real:
        inst, data = C.load_real_build(args.build, args.build_root, args.limit or None)
    else:
        inst, data = C.smoke_instances(args.smoke)
    full = real and not args.limit
    hyper = C.hyper_for_seed(args.seed, smoke=not real)
    header = C.run_header(args, data, hyper)
    dest = str(Path(args.results_root) / build / tag) + "/" if full and args.results_root else None
    t_start = time.time()
    status = {"state": "running", "pid": os.getpid(), "started": C.utc_now(),
              "out_dir": str(out_dir), "approaches": approaches, "done": [],
              "not_a_result": not full}

    def small_files():
        return sorted(p for p in res_dir.iterdir() if p.is_file() and not p.name.startswith("."))

    def save_status(**more):
        status.update(more)
        status["updated"] = C.utc_now()
        status["elapsed_seconds"] = time.time() - t_start
        C.write_json(res_dir / "status.json", status)

    save_status(current=None)
    C.copy_results(small_files(), dest)

    for approach in approaches:
        low = approach.lower()
        inp = L.amp_inputs(inst) if approach == "AMP" else L.dap_inputs(inst)
        Y = inp["Y"] if approach == "AMP" else inp["y"]
        techniques = [s for s in (N.M.AMP_TECHNIQUES_TH if approach == "AMP"
                                  else N.M.DAP_TECHNIQUES)]
        names = [N.M.technique_name(*t) if approach == "AMP" else t for t in techniques]
        n_splits = args.n_splits or N.fold_scheme_from_config()
        done, t_app = [], time.time()

        def on_fold(f):
            done.append(f)
            fresh = sum(f.inner_fresh_fits.values())
            sec = sum(f.inner_seconds.values())
            per_fit = {t: (f.inner_seconds[t] / f.inner_fresh_fits[t])
                       for t in names if f.inner_fresh_fits.get(t)}
            eta = None
            fresh_folds = [g for g in done if sum(g.inner_fresh_fits.values())]
            if fresh_folds:
                mean = sum(sum(g.inner_seconds.values()) for g in fresh_folds) / len(fresh_folds)
                eta = mean * (n_splits - len(done))
            C.LOG.info("%s fold %d/%d done: %d fresh fits in %.0f s; wall seconds per "
                       "inner-LOO case by technique: %s; about %s s left for %s",
                       approach, f.fold, n_splits, fresh, sec,
                       {t: round(v, 3) for t, v in per_fit.items()},
                       "?" if eta is None else format(eta, ".0f"), approach)
            C.write_json(res_dir / f"{low}_nested_partial.json", {
                "step": "A14 nested CV - PARTIAL (folds finished so far)",
                "approach": approach, "header": header, "techniques": names,
                "n_splits": n_splits, "folds_done": len(done),
                "test_successes_so_far": sum(g.test_successes for g in done),
                "test_cases_so_far": sum(g.n_test for g in done),
                "folds": [fold_dict(g) for g in done]})
            write_table(res_dir, approach, names, done, full, len(inst))
            save_status(current={"approach": approach, "folds_done": len(done),
                                 "n_splits": n_splits,
                                 "eta_seconds_this_approach": eta,
                                 "seconds_per_inner_case_last_fold": per_fit})
            C.copy_results(small_files(), dest)

        save_status(current={"approach": approach, "folds_done": 0, "n_splits": n_splits})
        kw = dict(techniques=techniques, n_splits=n_splits, case_ids=inp["case_ids"],
                  domains=inp["domains"], hyper=hyper, n_jobs=args.n_jobs,
                  batch_size=args.batch_size, checkpoint_dir=out_dir / "ckpt" / low,
                  on_fold=on_fold)
        if approach == "AMP":
            res = N.run_amp_nested_cv(inp["X"], Y, labels=inp["labels"],
                                      method_answers=inp["method_answers"], **kw)
        else:
            res = N.run_dap_nested_cv(inp["X"], Y, answer_sets=inp["answer_sets"],
                                      rule_outcomes=inp["rule_outcomes"], **kw)

        s = C.result_summary(
            res.result, T.NESTED_SUCCESSES[approach] if full else None,
            T.NESTED_AMP_DOMAIN_SUCCESSES if full and approach == "AMP" else None,
            T.NESTED_AMP_SELECTED if full and approach == "AMP" else None,
            T.NESTED_AMP_SELECTED_SUCCESS if full and approach == "AMP" else None)
        lo, hi = res.total_range_over_ties()
        names_th, rows_th = T.thesis_nested_table(approach)
        summary = {
            "step": "A14 leave-one-out within 10-fold CV", "approach": approach,
            "header": header, "complete": True, "finished": C.utc_now(),
            "seconds": time.time() - t_app, "techniques": list(res.techniques),
            "n_splits": n_splits, "technique_tie_rule": res.meta["technique_tie"],
            "selected_by_fold": res.selected(),
            "thesis_selected_by_fold": [
                "+".join(t for t, x in zip(names_th, r[:-1]) if x == max(r[:-1]))
                for r in rows_th] if full else None,
            "tied_folds": res.tie_report(),
            "successes_range_over_tie_resolutions": [lo, hi],
            "not_converged": res.not_converged(),
            "identical_inner_BR+KNN_LP+KNN": (res.identical_inner("BR+KNN", "LP+KNN")
                                             if approach == "AMP" else None),
            "result": s, "folds": [fold_dict(f) for f in res.folds]}
        C.write_json(res_dir / f"{low}_nested_summary.json", summary)
        C.write_cases_csv(res_dir / f"{low}_nested_cases.csv", res.result)
        write_table(res_dir, approach, list(res.techniques), res.folds, full, len(inst))
        text = C.format_summary(s, f"{approach} nested CV ({n_splits}-fold x inner LOO)"
                                + ("" if full else "   [NOT A RESULT: smoke / limited run]"))
        text += (f"\n  selected by fold: {res.selected()}\n  tied folds: {res.tie_report()}"
                 f"\n  successes over every tie resolution: {lo}..{hi}"
                 f"\n  fits not converged: {res.not_converged()}")
        C.write_text(res_dir / f"{low}_nested_summary.txt", text + "\n")
        C.LOG.info("\n%s\n%s", (res_dir / f"{low}_nested_table.txt").read_text(), text)
        status["done"].append(approach)
        save_status(current=None)
        C.copy_results(small_files(), dest)

    save_status(state="finished", finished=C.utc_now(), current=None)
    C.LOG.info("A14 nested CV finished in %.0f s; results in %s%s", time.time() - t_start,
               res_dir, f"; copied to {dest}" if dest else "")
    C.copy_results(small_files(), dest)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException as exc:                   # leave a trace in the log
        if not isinstance(exc, SystemExit) or exc.code not in (0, None):
            C.LOG.exception("A14 nested CV stopped: %r", exc)
            C.LOG.info("the finished inner loops are in the checkpoint folder; "
                       "start the same command again to resume")
        raise
