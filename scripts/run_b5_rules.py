"""Step B5 (plan §5.7, §9.3): the rule baselines MR, WC, SP, HAC on the 77
World-B sub-sets, over 30 partition seeds (plan §8 OQ25), beside T24.

  python scripts/run_b5_rules.py            data = config/worldB.yaml
                                            (shipped: the synthetic stand-in)

On the synthetic stand-in every output is marked "SYNTHETIC — not a
reproduction", the run and the thesis are in separate columns, and the
files go to runs/worldB_synthetic/b5_rules/. The four rule counts are the
generator's calibration targets (plan §2.3), so their agreement with the
thesis is by construction.

A real file (config/worldB.yaml `source: real`) would be written to
runs/worldB_real/b5_rules/; that path is UNTESTED ON REAL DATA.
"""
import argparse
import csv
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from _common import LOG, copy_results, setup_logging, utc_now, write_json, write_text  # noqa: E402
from cdm import aggregators as AG                              # noqa: E402
from cdm.config import get_switch, load_config                 # noqa: E402
from cdm.data.worldB import loader as L                        # noqa: E402
from cdm.data.worldB import partition as WP                    # noqa: E402
from cdm.data.worldB import synthetic as S                     # noqa: E402
from cdm.eval import worldB_rules as WR                        # noqa: E402

RUNS_ROOT = CODE_ROOT / "runs"
RESULTS_ROOT = str(CODE_ROOT / "results") + "/"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None, help="World-B source file "
                    "(default config/worldB.yaml)")
    ap.add_argument("--copy-results", action="store_true",
                    help="also copy the result files to results/ (off by default)")
    ap.add_argument("--no-copy", action="store_true",
                    help="never copy to results/ (the default)")
    ap.add_argument("--runs-root", default=str(RUNS_ROOT),
                    help="parent of the output folder (default runs/)")
    args = ap.parse_args(argv)

    config = load_config()
    data = L.load_from_config(args.config)
    name = "worldB_synthetic" if data.synthetic else "worldB_real"
    out = Path(args.runs_root) / name / "b5_rules"
    setup_logging(out / "b5_rules.log")
    mark = data.marker if data.synthetic else L.REAL_DATA_STATUS
    LOG.info("%s -- World-B rule baselines (step B5)", mark)

    oq25 = get_switch("OQ25", config)
    seeds = tuple(range(int(oq25["n_seeds"])))
    assignment = "random"
    if oq25["use_original_assignment_if_available"] and data.original_subsets is not None:
        assignment = "original"
    opts = AG.options_from_config(config)
    run = WR.evaluate_seeds(data, seeds=seeds, assignment=assignment, **opts)
    summary = run.summary()
    rep = L.table_4_1_report(data)
    part = WP.partition_report(WP.subset_cases(
        data, seed=0, assignment=assignment))

    first = {"_marker": mark} if data.synthetic else {"_status": mark}
    payload = dict(first)
    payload.update({
        "written": utc_now(), "step": "B5", "plan": "§5.7, §9.3 row B5, OQ25, T24",
        "synthetic": data.synthetic,
        "data": {"n_tasks": rep["n_tasks"], "n_responses": rep["n_responses"],
                 "oq34": rep["oq34"], "table_4_1_all_match": rep["all_match"],
                 "generator_seed": data.log.get("seed"),
                 "parameters": data.log.get("parameters")},
        "partition": {"assignment": assignment, "seeds": list(run.seeds),
                      "n_subsets": run.n_subsets, **part},
        "tie_policy": opts,
        "rules": WR.RULES_B,
        # no thesis value in the run block; for synthetic data also none of
        # the "seeds within k of the thesis" counts (they measure a distance
        # to the thesis, which decision R2 does not give for synthetic data)
        "run": {r: {k: v for k, v in summary[r].items()
                    if not k.startswith("thesis")
                    and not (data.synthetic and k.endswith("_of_thesis"))}
                for r in WR.RULES_B},
        "thesis": {"source": "TH Fig 5.3 p.76 (rates); counts of 77 = plan T24",
                   "count": WR.THESIS_COUNT, "rate": WR.THESIS_RATE},
        "counts_per_seed": {r: run.counts(r) for r in WR.RULES_B},
    })
    if data.synthetic:
        payload["note"] = (
            "The synthetic generator was calibrated on these four rule counts "
            "over these partition seeds (plan §2.3). Agreement of 'run' with "
            "'thesis' is BY CONSTRUCTION and is not a reproduction of T24.")
        payload["generator_seed_note"] = S.GENERATOR_SEED_NOTE
    write_json(out / "rules_summary.json", payload)

    rows = WR.per_seed_rows(run)
    with open(out / "rules_per_seed.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {mark}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    text = WR.format_summary(run)
    text += "\n#### Success counts per partition seed" \
            + (f" — {mark}" if data.synthetic else "") + "\n\n"
    text += "| seed | " + " | ".join(WR.RULES_B) + " |\n|---|" + "---|" * 4 + "\n"
    for row in rows:
        text += f"| {row['partition_seed']} | " + " | ".join(
            str(row[f"{r}_correct"]) for r in WR.RULES_B) + " |\n"
    # the thesis counts are not a row of the run's table
    text += "\nthesis (count of 77; TH Fig 5.3, not part of the run): " + " / ".join(
        f"{r} {WR.THESIS_COUNT[r]}" for r in WR.RULES_B) + "\n"
    if data.synthetic:
        text += f"\n**{mark}**\n"
    write_text(out / "rules_summary.md", text)
    LOG.info("\n%s", text)
    if (args.copy_results and not args.no_copy
            and Path(args.runs_root).resolve() == RUNS_ROOT.resolve()):
        copy_results([out / "rules_summary.md", out / "rules_summary.json",
                      out / "rules_per_seed.csv"],
                     RESULTS_ROOT + name + "/b5_rules/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
