"""Step B3 (plan §2.3, §9.3): the SYNTHETIC World-B stand-in.

    SYNTHETIC — not a reproduction. Every file this script writes says so.

  python scripts/run_b3_synthetic.py --calibrate
      seeded grid search of the INVENTED-CALIBRATED parameters on the four
      rule counts of T24 (plan §2.3); writes config/synthetic_worldB.yaml
      and the search tables.
  python scripts/run_b3_synthetic.py
      generates the data with the parameters of config/synthetic_worldB.yaml
      and writes it in the loader layout, with the checks of plan §9.3 row
      B3: the task design of TH Table 4.1 matched exactly, 77 sub-sets with the printed
      average sizes, predicted supports summing to 1, the feature oracle.
  --first-grid   (with --calibrate) the search on the first grid only; the
      result is written to the run folder, not to config/ (decision B3-9).

Outputs: runs/worldB_synthetic/{b3_calibration,b3_data}/; with
--copy-results the small files are also copied to results/worldB_synthetic/.
"""
import argparse
import csv
import sys
from fractions import Fraction
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np                                             # noqa: E402

from _common import copy_results, setup_logging, utc_now, write_json, write_text, LOG  # noqa: E402
from cdm.data.worldB import loader as L                        # noqa: E402
from cdm.data.worldB import partition as WP                    # noqa: E402
from cdm.data.worldB import synthetic as S                     # noqa: E402
from cdm.data.worldB import table41 as T41                     # noqa: E402

RUN_ROOT = CODE_ROOT / "runs" / "worldB_synthetic"
RESULTS_DIR = str(CODE_ROOT / "results" / "worldB_synthetic") + "/"
MARK = L.SYNTHETIC_MARKER


def _copy_wanted(args):
    """Copy to results/ only on request and only from the default runs folder."""
    return (args.copy_results and not args.no_copy
            and Path(args.runs_root).resolve() == RUN_ROOT.resolve())


def write_marked_csv(path, header, rows):
    """CSV whose first line is the synthetic marker (a "#" comment)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {MARK}\n")
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def do_calibrate(args):
    out = Path(args.runs_root) / ("b3_calibration_first_grid" if args.first_grid
                                  else "b3_calibration")
    setup_logging(out / "calibration.log")
    LOG.info("%s -- calibration of the synthetic World-B generator", MARK)
    grid = ({"ps_solver_lure_shift": S.FIRST_GRID_PS_SOLVER_LURE_SHIFT}
            if args.first_grid else None)

    def progress(block, row):
        LOG.info("%s %s -> %s loss %.4f", block, row["values"],
                 {k: round(v, 3) for k, v in row["means"].items()}, row["loss"])

    res = S.calibrate(seed=args.seed, grid=grid, progress=progress)
    files = []
    for st in res["stages"]:
        header = st["names"] + [f"mean_{r}" for r in st["rules"]] + ["loss", "chosen"]
        rows = [[row["values"][k] for k in st["names"]]
                + [row["means"][r] for r in st["rules"]]
                + [row["loss"], int(row is st["chosen"])] for row in st["table"]]
        files.append(write_marked_csv(out / f"calibration_grid_{st['block']}.csv",
                                      header, rows))
    summary = {
        "_marker": MARK, "written": utc_now(), "plan": "§2.3, §9.3 row B3, OQ43",
        "first_grid_only": bool(args.first_grid),
        "seed": res["seed"], "partition_seeds": res["partition_seeds"],
        "targets": dict(S.CALIBRATION_TARGETS),
        "parameters": res["parameters"],
        "tags": {k: v["tag"] for k, v in S.PARAMETERS.items()},
        "mean_counts_reached": res["means"], "loss": res["loss"],
        "within_2_of_target": {r: abs(res["means"][r] - t) <= 2
                               for r, t in S.CALIBRATION_TARGETS.items()},
        "stages": [{"block": st["block"], "rules": st["rules"],
                    "combinations": len(st["table"]), "chosen": st["chosen"]}
                   for st in res["stages"]],
        "note": "Rule counts are the calibration targets: agreement with TH "
                "Fig 5.3 is by construction. RP/AP outcomes were not looked at.",
    }
    files.append(write_json(out / "calibration_summary.json", summary) or
                 out / "calibration_summary.json")
    if not args.first_grid:
        S.PARAMETERS_PATH.write_text(S.parameters_document(res), encoding="utf-8")
        LOG.info("wrote %s", S.PARAMETERS_PATH)
        S.load_parameters()                          # must read back
    LOG.info("calibrated: %s", res["parameters"])
    LOG.info("means %s loss %.4f", res["means"], res["loss"])
    if _copy_wanted(args):
        copy_results([Path(f) for f in files],
                     RESULTS_DIR + out.name + "/")
    return 0


def do_generate(args):
    out = Path(args.runs_root) / "b3_data"
    setup_logging(out / "generate.log")
    LOG.info("%s -- generating the synthetic World-B data", MARK)
    params, doc = S.load_parameters()
    data = S.generate(params, seed=args.seed)
    paths = L.write_world_b(data, out, manifest_extra={
        "generator": "cdm.data.worldB.synthetic", "seed": args.seed,
        "parameters": params,
        "parameter_tags": {k: v["tag"] for k, v in S.PARAMETERS.items()},
        "layout_invented": data.log["layout"], "written": utc_now()})
    back = L.load_written(out)
    round_trip = back == data
    rep = L.table_4_1_report(back)

    # 77 sub-sets with the printed average sizes, for the 30 seeds
    sizes_ok, n_sub = True, set()
    oracle = []
    for seed in S.CALIBRATION_SEEDS:
        subs = WP.subset_cases(back, seed=seed)
        n_sub.add(len(subs))
        by_task = {}
        for tid, case in zip(subs.task_ids, subs.cases):
            by_task.setdefault(tid, []).append(case.n)
        for task, printed in zip(back.tasks, T41.TABLE_4_1):
            ss = by_task[task.case_id]
            # |R_P| / k rounded half up to the one printed decimal
            avg = float(Fraction(int(Fraction(sum(ss), len(ss)) * 10
                                     + Fraction(1, 2)), 10))
            if (avg != printed.avg_subset_size or max(ss) - min(ss) > 1
                    or sum(ss) != printed.n_responses):
                sizes_ok = False
        oracle += [(seed,) + x for x in S.a4_like_instances(subs)]
    sums = [sum(r.ps) for c in back.tasks for r in c.responses]
    conf = [r.c for c in back.tasks for r in c.responses]
    checks = {
        "_marker": MARK, "written": utc_now(),
        "round_trip_exact": round_trip,
        "table_4_1_all_match": rep["all_match"],
        "n_tasks": rep["n_tasks"], "n_responses": rep["n_responses"],
        "oq34": rep["oq34"],
        "n_subsets_over_30_seeds": sorted(n_sub),
        "subset_sizes_match_printed_averages": sizes_ok,
        "ps_sum_max_abs_deviation_from_1": float(max(abs(s - 1) for s in sums)),
        "ps_min": float(min(min(r.ps) for c in back.tasks for r in c.responses)),
        "confidence_min_max_mean": [float(min(conf)), float(max(conf)),
                                    float(np.mean(conf))],
        "unweighted_mean_solver_share": float(np.mean(
            [r["s_correct"] for r in rep["rows"]])),
        "weighted_mean_solver_share": sum(r["n_solvers"] for r in rep["rows"])
        / rep["n_responses"],
        "feature_oracle_a4_like": {
            "n_instances_over_30_seeds": len(oracle),
            "n_seeds_with_one": len({o[0] for o in oracle}),
            "examples": [list(o) for o in oracle[:10]]},
        "parameters": params,
    }
    write_json(out / "b3_checks.json", checks)
    header = ["task_id", "name", "n_answers", "printed_n_answers", "n_responses",
              "printed_n_responses", "n_solvers", "s_correct_2dp",
              "printed_s_correct", "n_subsets", "printed_avg_subset_size",
              "invented_correct", "invented_lure", "invented_lure_share"]
    rows = []
    for r, lay in zip(rep["rows"], data.log["layout"]):
        p = r["printed"]
        rows.append([r["task_id"], r["name"], r["n_answers"], p["n_answers"],
                     r["n_responses"], p["n_responses"], r["n_solvers"],
                     T41.printed_share(r["n_solvers"], r["n_responses"]),
                     p["s_correct"], r["n_subsets"], p["avg_subset_size"],
                     lay["correct"], lay["lure"], round(lay["lure_share"], 4)])
    write_marked_csv(out / "b3_table_4_1_check.csv", header, rows)

    lines = [f"**{MARK}**", "",
             f"# Synthetic World-B stand-in (plan §2.3, step B3) — {MARK}", "",
             "Invented responses for the 22 tasks of TH Table 4.1. Only the "
             "table's |A_P|, |R_P| and S(a*) are evidence; everything else is "
             "invented (tags in config/synthetic_worldB.yaml).", "",
             f"- written and re-read through the loader, identical: {round_trip}",
             f"- task design of TH Table 4.1 matched exactly (|A_P|, |R_P|, S(a*) of 22 "
             f"tasks): {rep['all_match']}",
             f"- responses: {rep['n_responses']} ({rep['oq34']})",
             f"- sub-sets for partition seeds 0..29: {sorted(n_sub)}; sizes equal "
             f"the printed averages: {sizes_ok}",
             f"- predicted supports: largest |sum − 1| = "
             f"{checks['ps_sum_max_abs_deviation_from_1']:.2e}, smallest value "
             f"{checks['ps_min']:.3g}",
             f"- confidence: min {checks['confidence_min_max_mean'][0]:.3f}, max "
             f"{checks['confidence_min_max_mean'][1]:.3f}, mean "
             f"{checks['confidence_min_max_mean'][2]:.3f}",
             f"- mean solver share: unweighted "
             f"{checks['unweighted_mean_solver_share']:.3f}, response-weighted "
             f"{checks['weighted_mean_solver_share']:.3f} (plan §2.2: .303 / .280)",
             f"- feature oracle (a sub-set of 33 with a non-majority answer at "
             f"8/33 and a supporter predicting 100% for it): "
             f"{len(oracle)} instances in {len({o[0] for o in oracle})} of 30 "
             "partition seeds", "",
             "| Task | \\|A\\| | \\|R\\| | solvers | S(a*) | printed S(a*) | sub-sets "
             "| printed avg size | correct* | lure* | lure share* |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row[1]} | {row[2]} | {row[4]} | {row[6]} | {row[7]:.2f} "
                     f"| {row[8]:.2f} | {row[9]} | {row[10]} | {row[11]} | {row[12]} "
                     f"| {row[13]:.3f} |")
    lines += ["", "\\* invented by the generator.", "", f"**{MARK}**"]
    write_text(out / "b3_report.md", "\n".join(lines) + "\n")
    LOG.info("checks: %s", {k: v for k, v in checks.items()
                            if k not in ("parameters", "_marker")})
    if _copy_wanted(args):
        copy_results([paths["tasks"], paths["responses"], paths["manifest"],
                      out / "b3_checks.json", out / "b3_table_4_1_check.csv",
                      out / "b3_report.md"], RESULTS_DIR + "b3_data/")
    ok = (round_trip and rep["all_match"] and sizes_ok and sorted(n_sub) == [77])
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--first-grid", action="store_true")
    ap.add_argument("--seed", type=int, default=S.DEFAULT_SEED)
    ap.add_argument("--copy-results", action="store_true",
                    help="also copy the small files to results/ (off by default)")
    ap.add_argument("--no-copy", action="store_true",
                    help="never copy to results/ (the default)")
    ap.add_argument("--runs-root", default=str(RUN_ROOT),
                    help="output folder (default runs/worldB_synthetic)")
    args = ap.parse_args(argv)
    if args.first_grid and not args.calibrate:
        ap.error("--first-grid needs --calibrate")
    return do_calibrate(args) if args.calibrate else do_generate(args)


if __name__ == "__main__":
    sys.exit(main())
