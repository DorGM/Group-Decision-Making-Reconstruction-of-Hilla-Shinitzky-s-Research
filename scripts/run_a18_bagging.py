#!/usr/bin/env python
"""A18 driver: the bagging experiments of AMP (BR+RF) and DAP (RF)
(plan §5.6, §8 OQ22 / OQ23 / OQ42; TH §4.1.7 p.50-52; TH Tables 5.9, 5.10;
plan items T17, T18).

  python scripts/run_a18_bagging.py --build waiver_main --n-jobs 4

The build must be listed as passed or waived in config/gate_status.yaml
(cdm.eval.gate_guard); any other build is refused before it is loaded.

For every F in {2, 3, 5, 10}: F sub-groups per case with at least 70
responses, each a new instance; 10 folds = the folds of the model
selection; BagTrain_K = the other originals + the new instances of the
problems that are not in Test_K; BR+RF / RF in every fold.

Outputs, under <runs-root>/<build>/<tag>/ (tag a18; a18_cochran, a18_z1.96,
a18_seed<k>, a18_bagseed<k>, a18_<undefined rule> for the switches):
  ckpt/                      per approach and F: the test outputs (a
                             repeated run reads them; other inputs or code
                             are refused, never mixed)
  a18_bagging.md             Tables 5.9 / 5.10 beside the thesis, the
                             sub-group sizes, successes per domain, rule
                             baselines recomputed vs printed
  a18_<approach>_bagging.csv one line per F and fold
  a18_<approach>_F<F>_cases.csv   one line per case
  a18_summary.json, status.json, run.log
With --results-root the small files are copied to <results-root>/<build>/<tag>/.

OPEN ITEM (plan OQ7 / amendment AM12): a sub-group in which every response
gives the same answer has no CAmin. The plan gives no rule. With the
default `--undefined-camin stop` the script writes the counts to
a18_blocked.json / a18_bagging.md and exits with code 3 WITHOUT fitting
anything when such instances exist. `drop_instance` and
`impute_subgroup_avgc` are labelled switches that are not in the plan; a run
with one of them is labelled as such and goes to its own folder.

--size-formula / --z : plan OQ22 switches (default: config = formula as
                       printed, Z = 1.645).
--describe-only      : build the new instances and report sizes and
                       undefined features; no model is fitted.
--smoke N            : end-to-end test on N synthetic cases (40-109
                       responses, 10 trees); written to <runs-root>/_smoke/,
                       never to the results folder.
--limit N            : pipeline check on a sub-sample of a real build;
                       never written to the results folder.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import _common as C
from cdm.eval import bagging as B
from cdm.eval import effectiveness as E
from cdm.eval import loo as L
from cdm.eval import stats as S
from cdm.report import tables as R
from cdm.report import thesis_values as T

BLOCKED_EXIT = 3


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build", default="waiver_main")
    p.add_argument("--build-root", default=str(C.DEFAULT_BUILD_ROOT))
    p.add_argument("--seed", type=int, default=0, help="classifier seed")
    p.add_argument("--bag-seed", type=int, default=None,
                   help="seed of the sub-group draws (default: config seed, 0)")
    p.add_argument("--factors", default="10,5,3,2")
    p.add_argument("--size-formula", default=None, choices=B.SIZE_FORMULAS)
    p.add_argument("--z", type=float, default=None)
    p.add_argument("--undefined-camin", default="stop", choices=B.UNDEFINED_POLICIES)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--approaches", default="DAP,AMP", help="run order")
    p.add_argument("--n-splits", type=int, default=None, help="default: config OQ16 (10)")
    p.add_argument("--a14-dir", default=None,
                   help="results of the A14 run for the '-' row "
                        "(default <runs-root>/<build>/a14/results)")
    p.add_argument("--runs-root", default=str(C.DEFAULT_RUNS_ROOT))
    p.add_argument("--results-root", default=str(C.DEFAULT_RESULTS_ROOT),
                   help="folder to copy the small result files to (e.g. results); "
                        "default '' = do not copy")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--describe-only", action="store_true")
    p.add_argument("--smoke", type=int, default=0, metavar="N")
    p.add_argument("--limit", type=int, default=0, metavar="N")
    return p.parse_args(argv)


def dash_row(a14_dir, approach, n_splits):
    """The "-" row of the run: test successes per fold of the A14 nested CV
    (technique selected per fold), or None when A14 is not available."""
    f = Path(a14_dir) / f"{approach.lower()}_nested_summary.json"
    if not f.is_file():
        return None
    doc = json.loads(f.read_text(encoding="utf-8"))
    folds = doc.get("folds") or []
    if not doc.get("complete") or len(folds) != n_splits:
        return None
    return [{"fold": x["fold"], "n_test": x["n_test"], "successes": x["test_successes"],
             "selected": x["selected"]} for x in folds]


def cached_run(ckpt_dir, approach, bag, inst, inst_bag, parent, hyper, n_splits, n_jobs):
    """`B.run_bagging` with the fold results kept in ckpt_dir. The cache
    key is a fingerprint of the technique, hyperparameters, library and
    code versions and of every training and test array. Returns
    {folds, cases, from_ckpt, result (the EvalResult, None from the cache)}."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    inp = L.amp_inputs(inst) if approach == "AMP" else L.dap_inputs(inst)
    b = (L.amp_inputs(inst_bag) if approach == "AMP" else L.dap_inputs(inst_bag)) \
        if inst_bag is not None else None
    Y, Yb = (inp["Y"], b["Y"] if b else None) if approach == "AMP" else \
        (inp["y"], b["y"] if b else None)
    spec = L.make_spec(approach, B.TECHNIQUE[approach],
                       **({"labels": inp["labels"]} if approach == "AMP" else {}), hyper=hyper)
    X_all = np.vstack([inp["X"], b["X"]]) if b else inp["X"]
    Y_all = np.concatenate([Y, Yb]) if b else Y
    fp = L.fingerprint(spec, X_all, Y_all, {"bagging": True, "n_original": len(inst),
                                            "parent": parent.tolist(), "n_splits": n_splits})
    path = ckpt_dir / f"{approach.lower()}_F{bag.factor}_{fp[:12]}.json"
    stale = [p.name for p in ckpt_dir.glob(f"{approach.lower()}_F{bag.factor}_*.json")
             if p != path]
    if stale:
        raise SystemExit(f"checkpoint folder {ckpt_dir} holds results of {approach} F = "
                         f"{bag.factor} for other inputs or code ({stale[0]}); use another "
                         "folder or delete them")
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
        return {"folds": doc["folds"], "cases": doc["cases"], "from_ckpt": True,
                "result": None}
    res = B.run_bagging(approach, inst, inst_bag, parent, bag.factor, hyper, n_splits,
                        n_jobs=n_jobs)
    doc = {"fingerprint": fp, "folds": res.folds,
           "cases": {"success": res.result.success.tolist(),
                     "fold": res.result.folds.tolist(),
                     "ties": res.result.tie_log(),
                     "selected": res.result.selection_counts() if approach == "AMP" else None,
                     "selected_success": res.result.selected_success_counts()
                     if approach == "AMP" else None,
                     "domains": E.success_by_domain(res.result)}}
    C.write_json(path, doc)
    return {"folds": res.folds, "cases": doc["cases"], "from_ckpt": False,
            "result": res.result}


def main(argv=None):
    args = parse_args(argv)
    approaches = [a.strip().upper() for a in args.approaches.split(",") if a.strip()]
    if not approaches or any(a not in ("AMP", "DAP") for a in approaches):
        raise SystemExit("--approaches: a comma-separated order of AMP and DAP")
    factors = [int(f) for f in args.factors.split(",") if f.strip()]
    real = not args.smoke
    build = args.build if real else "_smoke"
    settings = B.settings_from_config(size_formula=args.size_formula, z=args.z)
    default = B.settings_from_config()
    config_seed = C.load_config()["seed"]
    bag_seed = config_seed if args.bag_seed is None else args.bag_seed
    tag = "a18" + ("_cochran" if settings["size_formula"] == B.FORMULA_COCHRAN else "") \
        + (f"_z{settings['z']:g}" if settings["z"] != default["z"] else "") \
        + (f"_seed{args.seed}" if args.seed else "") \
        + (f"_bagseed{bag_seed}" if bag_seed != config_seed else "") \
        + (f"_{args.undefined_camin}" if args.undefined_camin != "stop" else "") \
        + (f"_limit{args.limit}" if args.limit and real else "")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.runs_root) / build / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(out_dir / "run.log")
    C.LOG.info("A18 bagging start (pid %d): %s", os.getpid(), vars(args))

    if real:
        inst, data, cases = C.load_real_build(args.build, args.build_root,
                                              args.limit or None, return_cases=True)
    else:
        inst, data, cases = C.smoke_instances(args.smoke, n_min=40, n_max=110,
                                              return_cases=True)
    full = real and not args.limit
    n = len(inst)
    n_splits = args.n_splits or B.N.fold_scheme_from_config()
    hyper = C.hyper_for_seed(args.seed, smoke=not real)
    header = C.run_header(args, data, hyper)
    dest = str(Path(args.results_root) / build / tag) + "/" if full and args.results_root else None
    a14 = Path(args.a14_dir) if args.a14_dir else Path(args.runs_root) / build / "a14" / "results"
    t_start = time.time()

    # ---- the new instances of every F (no model) ----
    bags, tables, parents, reports, blocked = {}, {}, {}, {}, {}
    for F in factors:
        bag = B.bag_cases(cases, F, bag_seed, settings["size_formula"], settings["z"])
        inst_bag = B.bagged_instances(bag)
        desc = bag.describe()
        if inst_bag is None:
            raise SystemExit(f"no case has at least {B.MIN_RESPONSES} responses: "
                             "nothing to bag")
        rows, counts = B.undefined_rows(inst_bag)
        desc["undefined_instances"] = int(len(rows))
        desc["undefined_by_feature"] = counts
        desc["undefined_by_domain"] = {
            d: int(sum(1 for r in rows if inst_bag.domains[r] == d))
            for d in dict.fromkeys(inst.domains)}
        desc["unanimous_subgroups"] = int(sum(
            1 for c in bag.cases if len({r.v for r in c.responses}) == 1))
        desc["instance_log"] = inst_bag.log()
        try:
            inst_bag, parent, rep = B.apply_undefined_policy(inst_bag, bag.parent,
                                                             args.undefined_camin)
            desc["policy_report"] = rep
            desc["new_instances_used"] = len(inst_bag)
        except B.UndefinedFeatureError as exc:
            blocked[F] = str(exc)
            parent = bag.parent
        bags[F], tables[F], parents[F], reports[F] = bag, inst_bag, parent, desc
        C.LOG.info("A18 F = %d: %d eligible cases (|R| >= %d), %d new instances, sizes "
                   "%s..%s, undefined CAmin in %d (%s)", F, desc["eligible_cases"],
                   B.MIN_RESPONSES, desc["new_instances"], desc["size_min"],
                   desc["size_max"], desc["undefined_instances"], desc["undefined_by_domain"])

    size_head = ["F", "cases with at least 70 responses", "new instances",
                 "sub-group size min..max (mean)", "new instances with undefined CAmin",
                 "of these per domain", "used in training"]
    size_rows = [[F, d["eligible_cases"], d["new_instances"],
                  f"{d['size_min']}..{d['size_max']} ({d['size_mean']:.1f})",
                  d["undefined_instances"],
                  ", ".join(f"{k} {v}" for k, v in d["undefined_by_domain"].items()),
                  d.get("new_instances_used", "blocked")] for F, d in reports.items()]
    rule_note = {
        "stop": "stop (plan gives no rule; OQ7 / AM12)",
        "drop_instance": "drop_instance - not in the plan (labelled switch; AM12, AM19)",
        "impute_subgroup_avgc": "impute_subgroup_avgc - not in the plan (labelled switch; "
                                "AM12, AM19)"}[args.undefined_camin]
    intro = [f"# A18 bagging experiments - build `{build}`, classifier seed {args.seed}", "",
             f"Written {C.utc_now()}. Sub-group size formula: `{settings['size_formula']}`, "
             f"Z = {settings['z']:g}, rounded up, at least {B.MIN_SIZE}; draws without "
             f"replacement, seed {bag_seed}; rule for an undefined CAmin: {rule_note}. "
             f"{n_splits} unshuffled folds (the folds of the model selection); BR+RF / RF "
             "in every fold."
             + ("" if full else "  **NOT A RESULT: smoke / limited run.**"), ""]
    if full:
        intro += ["Note: under the documented reconstruction configuration, the per-fold rates "
                  "in the `run` rows differ from the thesis-reported rows in several folds. "
                  "The present reconstruction does not establish the reason. Nothing is "
                  "tuned.", ""]
    intro += [R.rule_baselines_md(data["rule_baselines_by_domain"]),
              "## New instances", "", R.md_table(size_head, size_rows), ""]
    summary = {"step": "A18 bagging (AMP BR+RF, DAP RF)", "header": header,
               "settings": dict(settings, bag_seed=bag_seed, min_responses=B.MIN_RESPONSES,
                                min_size=B.MIN_SIZE, undefined_camin=args.undefined_camin,
                                seed_rule=B.SEED_RULE, n_splits=n_splits),
               "new_instances": reports, "complete": False}
    files = {out_dir / f for f in ("a18_bagging.md", "a18_summary.json", "run.log")}

    if blocked or args.describe_only:
        if blocked:
            intro += ["## BLOCKED: no model was fitted", "",
                      "Some new instances have an undefined CAmin and the plan gives no rule "
                      "for them (OQ7 / amendment AM12). Decide the rule, then run again with "
                      "`--undefined-camin drop_instance` or `--undefined-camin "
                      "impute_subgroup_avgc` (or after the plan names another rule).", ""]
            summary["blocked"] = blocked
            C.write_json(out_dir / "a18_blocked.json",
                         {"written": C.utc_now(), "blocked": blocked,
                          "new_instances": reports})
            files.add(out_dir / "a18_blocked.json")
            C.LOG.error("A18 BLOCKED (exit %d): %s", BLOCKED_EXIT, next(iter(blocked.values())))
        C.write_text(out_dir / "a18_bagging.md", "\n".join(intro))
        C.write_json(out_dir / "a18_summary.json", summary)
        C.copy_results(sorted(files), dest)
        return BLOCKED_EXIT if blocked else 0

    # ---- the runs ----
    results = {a: {} for a in approaches}

    def write_all(final=False):
        texts = list(intro)
        for a in approaches:
            if not results[a]:
                continue
            th, th_dash = T.thesis_bagging_table(a)
            comparable = full and n_splits == 10
            dash = dash_row(a14, a, n_splits)
            head = ["F", "source"] + [f"fold {k}" for k in range(1, n_splits + 1)] + ["all"]
            rows, csv_rows = [], [["F", "fold", "n_test", "successes", "rate", "thesis_rate",
                                   "n_train_original", "n_train_bagged",
                                   "n_bagged_excluded"]]
            for F in factors:
                if F not in results[a]:
                    continue
                folds = results[a][F]["folds"]
                tot = sum(f["successes"] for f in folds)
                rows.append([F, "run"] + [R.rate(S.round_half_up(f["successes"], f["n_test"]))
                                          for f in folds]
                            + [f"{R.rate(S.round_half_up(tot, n))} ({tot})"])
                if comparable and F in th:
                    rows.append([F, "thesis"] + [R.rate(x) for x in th[F]] + [R.DASH])
                for f in folds:
                    csv_rows.append([F, f["fold"], f["n_test"], f["successes"],
                                     format(S.round_half_up(f["successes"], f["n_test"]), ".3f"),
                                     format(th[F][f["fold"] - 1], ".3f")
                                     if comparable and F in th else "",
                                     f["n_train_original"], f["n_train_bagged"],
                                     f["n_bagged_excluded"]])
            if dash:
                tot = sum(f["successes"] for f in dash)
                rows.append(["–", "run (A14 test column)"]
                            + [R.rate(S.round_half_up(f["successes"], f["n_test"]))
                               for f in dash] + [f"{R.rate(S.round_half_up(tot, n))} ({tot})"])
                rows.append(["", "technique selected in A14"] + [f["selected"] for f in dash]
                            + [""])
            else:
                rows.append(["–", "run (A14 test column)"] + ["not available"] * n_splits + [""])
            if comparable:
                rows.append(["–", "thesis"] + [R.rate(x) for x in th_dash]
                            + [R.rate(S.round_half_up(T.NESTED_SUCCESSES[a], T.N_CASES))
                               + f" ({T.NESTED_SUCCESSES[a]})"])
            texts += [f"## {a} ({'TH Table 5.9' if a == 'AMP' else 'TH Table 5.10'})", "",
                      R.md_table(head, rows), ""]
            blocks = list(next(iter(results[a].values()))["cases"]["domains"])
            texts += [f"### {a}: successes per domain (no thesis counterpart)", "",
                      R.md_table(["F"] + [f"{'all' if b == E.ALL else b} (n = "
                                          f"{next(iter(results[a].values()))['cases']['domains'][b]['n']})"
                                          for b in blocks],
                                 [[F] + [f"{r['cases']['domains'][b]['successes']} "
                                         f"({R.rate(r['cases']['domains'][b]['rounded'])})"
                                         for b in blocks]
                                  for F, r in results[a].items()]), "",
                      f"### {a}: training-set sizes per fold (originals + new instances; "
                      "new instances of the test problems left out)", "",
                      R.md_table(["F"] + [f"fold {k}" for k in range(1, n_splits + 1)],
                                 [[F] + [f"{f['n_train_original']} + {f['n_train_bagged']} "
                                         f"(− {f['n_bagged_excluded']})" for f in r["folds"]]
                                  for F, r in results[a].items()]), ""]
            path = out_dir / f"a18_{a.lower()}_bagging.csv"
            tmp = out_dir / f".{path.name}.{os.getpid()}.tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(csv_rows)
            os.replace(tmp, path)
            files.add(path)
            summary[a] = {str(F): r for F, r in results[a].items()}
            summary[a]["dash_row_a14"] = dash
        summary["complete"] = final
        summary["seconds"] = time.time() - t_start
        C.write_text(out_dir / "a18_bagging.md", "\n".join(texts))
        C.write_json(out_dir / "a18_summary.json", summary)
        C.write_json(out_dir / "status.json", {
            "state": "finished" if final else "running", "pid": os.getpid(),
            "updated": C.utc_now(), "elapsed_seconds": time.time() - t_start,
            "done": {a: list(results[a]) for a in approaches}, "not_a_result": not full})
        files.add(out_dir / "status.json")
        C.copy_results(sorted(files), dest)

    for a in approaches:
        for F in factors:
            t0 = time.time()
            got = cached_run(out_dir / "ckpt", a, bags[F], inst, tables[F], parents[F],
                             hyper, n_splits, args.n_jobs)
            folds, cases_doc, from_ckpt = got["folds"], got["cases"], got["from_ckpt"]
            results[a][F] = {"folds": folds, "cases": {k: v for k, v in cases_doc.items()
                                                       if k not in ("success", "fold")},
                             "successes": int(sum(f["successes"] for f in folds)),
                             "from_checkpoint": from_ckpt, "seconds": time.time() - t0}
            if not from_ckpt:
                path = out_dir / f"a18_{a.lower()}_F{F}_cases.csv"
                C.write_cases_csv(path, got["result"])
            path = out_dir / f"a18_{a.lower()}_F{F}_cases.csv"
            if path.is_file():
                files.add(path)
            C.LOG.info("A18 %s F = %d: %d/%d successes; folds %s; %.0f s%s", a, F,
                       results[a][F]["successes"], n,
                       [format(S.round_half_up(f["successes"], f["n_test"]), ".3f")
                        for f in folds], time.time() - t0,
                       " (from the checkpoint)" if from_ckpt else "")
            write_all()
    write_all(final=True)
    C.LOG.info("A18 bagging finished in %.0f s; outputs in %s%s", time.time() - t_start,
               out_dir, f"; copied to {dest}" if dest else "")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException as exc:                   # leave a trace in the log
        if not isinstance(exc, SystemExit) or exc.code not in (0, None, BLOCKED_EXIT):
            C.LOG.exception("A18 bagging stopped: %r", exc)
        raise
