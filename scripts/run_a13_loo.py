#!/usr/bin/env python
"""A13 driver: single leave-one-out of AMP (BR+RF) and DAP (RF) on a frozen
build (plan §5.3; MLA §5.5; plan items T12, T19-T22).

  python scripts/run_a13_loo.py --build waiver_main --seed 0 --n-jobs 4

The build must be listed as passed or waived in config/gate_status.yaml
(cdm.eval.gate_guard); any other build is refused before it is loaded.

Outputs, under <runs-root>/<build>/a13_seed<k>/:
  ckpt/amp_BR_RF/, ckpt/dap_RF/  checkpoints (the run can be started again
                                  with the same command and resumes)
  amp_loo_cases.csv, dap_loo_cases.csv   one row per case
  summary.json                    all counts, with the thesis-reported values beside them
  summary.txt                     the same as a short text table
  run.log                         progress log
The small files (not the checkpoints) are also copied to
<results-root>/<build>/a13_seed<k>/ when --results-root is given.

--seed is the classifier seed (RandomForest random_state). The features
are built with the config seed (see scripts/_common.py).
--smoke N : end-to-end test on N synthetic cases (no real data, 10 trees);
            written to <runs-root>/_smoke/, never to the results folder.
--limit N : pipeline check on an evenly spaced sub-sample of a real build;
            written to a13_seed<k>_limit<N>/, never to the results folder.
"""
import argparse
import sys
import time
from pathlib import Path

import _common as C
from cdm.eval import loo as L
from cdm.report import thesis_values as T


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build", default="waiver_main")
    p.add_argument("--build-root", default=str(C.DEFAULT_BUILD_ROOT))
    p.add_argument("--seed", type=int, default=0, help="classifier seed")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=L.DEFAULT_BATCH_SIZE)
    p.add_argument("--approaches", default="AMP,DAP")
    p.add_argument("--runs-root", default=str(C.DEFAULT_RUNS_ROOT))
    p.add_argument("--results-root", default=str(C.DEFAULT_RESULTS_ROOT),
                   help="folder to copy the small result files to (e.g. results); "
                        "default '' = do not copy")
    p.add_argument("--out-dir", default=None,
                   help="default <runs-root>/<build>/a13_seed<k>[_limit<N>]")
    p.add_argument("--smoke", type=int, default=0, metavar="N")
    p.add_argument("--limit", type=int, default=0, metavar="N")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    approaches = [a.strip().upper() for a in args.approaches.split(",") if a.strip()]
    if not approaches or any(a not in ("AMP", "DAP") for a in approaches):
        raise SystemExit("--approaches: AMP, DAP or AMP,DAP")
    real = not args.smoke
    build = args.build if real else "_smoke"
    tag = f"a13_seed{args.seed}" + (f"_limit{args.limit}" if args.limit and real else "")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.runs_root) / build / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(out_dir / "run.log")
    C.LOG.info("A13 LOO start: %s", vars(args))

    if real:
        inst, data = C.load_real_build(args.build, args.build_root, args.limit or None)
    else:
        inst, data = C.smoke_instances(args.smoke)
    full = real and not args.limit
    hyper = C.hyper_for_seed(args.seed, smoke=not real)
    header = C.run_header(args, data, hyper)
    summary = {"step": "A13 single LOO (AMP BR+RF, DAP RF)", "header": header,
               "complete": False}
    files = [out_dir / "summary.json", out_dir / "summary.txt", out_dir / "run.log"]
    dest = None
    if full and args.results_root:
        dest = str(Path(args.results_root) / build / tag) + "/"
    texts = [f"A13 LOO, build {build}, classifier seed {args.seed}, {len(inst)} cases"
             + ("" if full else "   [NOT A RESULT: smoke / limited run]"),
             "rule baselines (recomputed / thesis): "
             + ", ".join(f"{m} {r['recomputed']}"
                         + (f"/{r['thesis']}" if "thesis" in r else "")
                         for m, r in data["rule_baselines"].items())]

    for approach in approaches:
        t0 = time.time()
        if approach == "AMP":
            inp = L.amp_inputs(inst)
            res = L.run_amp_loo(inp["X"], inp["Y"], labels=inp["labels"],
                                case_ids=inp["case_ids"], domains=inp["domains"],
                                method_answers=inp["method_answers"], hyper=hyper,
                                n_jobs=args.n_jobs, batch_size=args.batch_size,
                                checkpoint_dir=out_dir / "ckpt" / "amp_BR_RF")
            s = C.result_summary(
                res, T.LOO_SUCCESSES["AMP"] if full else None,
                T.LOO_DOMAIN_SUCCESSES["AMP"] if full else None,
                T.LOO_AMP_SELECTED if full else None,
                T.LOO_AMP_SELECTED_SUCCESS if full else None,
                T.LOO_DISCORDANT["AMP"] if full else None)
            csv_path = out_dir / "amp_loo_cases.csv"
        else:
            inp = L.dap_inputs(inst)
            res = L.run_dap_loo(inp["X"], inp["y"], case_ids=inp["case_ids"],
                                domains=inp["domains"], answer_sets=inp["answer_sets"],
                                rule_outcomes=inp["rule_outcomes"], hyper=hyper,
                                n_jobs=args.n_jobs, batch_size=args.batch_size,
                                checkpoint_dir=out_dir / "ckpt" / "dap_RF")
            s = C.result_summary(
                res, T.LOO_SUCCESSES["DAP"] if full else None,
                T.LOO_DOMAIN_SUCCESSES["DAP"] if full else None,
                thesis_discordant=T.LOO_DISCORDANT["DAP"] if full else None)
            s["target_rate_code1"] = float(res.target.mean())
            s["predicted_rate_code1"] = float(res.predicted.mean())
            csv_path = out_dir / "dap_loo_cases.csv"
        s["technique"] = res.meta["technique"]
        s["seconds"] = time.time() - t0
        s["fit_seconds"] = res.meta.get("fit_seconds")
        s["n_from_checkpoint"] = res.meta.get("n_from_checkpoint")
        s["n_features"] = int(inp["X"].shape[1])
        summary[approach] = s
        C.write_cases_csv(csv_path, res)
        files.append(csv_path)
        text = C.format_summary(s, f"{approach} LOO ({s['technique']}), "
                                   f"{s['n_features']} features")
        texts.append(text)
        C.LOG.info("\n%s", text)
        C.write_json(out_dir / "summary.json", summary)
        C.write_text(out_dir / "summary.txt", "\n\n".join(texts) + "\n")
        C.copy_results(files, dest)

    summary["complete"] = True
    summary["finished"] = C.utc_now()
    C.write_json(out_dir / "summary.json", summary)
    C.write_text(out_dir / "summary.txt", "\n\n".join(texts) + "\n")
    C.LOG.info("A13 LOO finished; outputs in %s%s", out_dir,
               f"; copied to {dest}" if dest else "")
    C.copy_results(files, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
