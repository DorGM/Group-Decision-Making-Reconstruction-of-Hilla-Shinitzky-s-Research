"""Step B6 (plan §4.3, §5.7, §9.3): the RP pipeline — RCR instances, the
soft-voting ensemble, leave-one-group-out over the sub-sets, the nine
classification-aggregation strategies — in the shape of TH Table 4.2 (T26).

  python scripts/run_b6_rp.py            data = config/worldB.yaml
                                         (shipped: the synthetic stand-in)

On the synthetic stand-in every output is marked "SYNTHETIC — not a
reproduction"; the run and the thesis are in separate columns; files go to
runs/worldB_synthetic/b6_rp/. RP outcomes are not calibration targets of
the generator (plan §2.3): the numbers show that the pipeline runs end to
end and nothing else.

A real file (config/worldB.yaml `source: real`) is written to
runs/worldB_real/b6_rp/; that path is UNTESTED ON REAL DATA.

Outputs: rp_table_4_2.md, rp_summary.json, rp_per_seed.csv,
rp_outcomes.json (read by steps B8, B9, B11), rp_predictions.csv, b6_rp.log.
"""
import sys

import _worldb as W
from _worldb import LOG, write_json, write_text
from cdm.approaches import rp
from cdm.eval import worldB_ml as ML
from cdm.eval import worldB_thesis_values as T

CLASSIFIER = "soft_vote"


def per_seed_rows(runs):
    for r in runs:
        for name in rp.STRATEGY_NAMES:
            s = r["summary"][name]
            yield {"partition_seed": W.seed_key(r["seed"]), "strategy": name,
                   "correct": s["correct"], "n_subsets": s["n"],
                   "primary_ties": s["primary_ties"], "final_ties": s["final_ties"],
                   "low": s["low"], "high": s["high"]}


def main(argv=None):
    args = W.parser(__doc__).parse_args(argv)
    run = W.Run(args, "b6_rp", "RP pipeline (step B6)")
    runs = W.run_seeds(run, "RP", CLASSIFIER)
    n = runs[0]["n_subsets"]
    summary = ML.summarise_rp(runs)
    diag = ML.diagnostics(runs)

    doc = run.head("B6", "§4.3, §5.7, §9.3 row B6, OQ25-OQ28, T26")
    doc.update({
        "approach": "RP", "classifier": CLASSIFIER, "n_subsets": n,
        "group": "sub-set (TH p.57: each group is the set of instances of a single set "
                 "of responses)",
        "final_strategy": rp.FINAL_NAME,
        "run": summary, "diagnostics": diag,
        "thesis": {"source": "TH Table 4.2 p.57 = MCF Table 4 (success %, tie %)",
                   "table_4_2": {k: list(v) for k, v in T.TABLE_4_2.items()}},
        "minutes": run.minutes(),
    })
    if run.synthetic:
        doc["note"] = ("SYNTHETIC stand-in: RP outcomes are not calibration targets "
                       "(plan §2.3). 'run' is not a reproduction of 'thesis' and no "
                       "difference is computed.")
    write_json(run.out / "rp_summary.json", doc)
    write_json(run.out / "rp_outcomes.json", W.outcomes_document(run, runs, "RP", CLASSIFIER))
    run.csv(run.out / "rp_per_seed.csv",
            ("partition_seed", "strategy", "correct", "n_subsets", "primary_ties",
             "final_ties", "low", "high"), per_seed_rows(runs))
    run.csv(run.out / "rp_predictions.csv", W.PREDICTION_FIELDS, W.prediction_rows(runs))

    extra = [
        "", "#### Run diagnostics" + (f" — {run.mark}" if run.synthetic else ""), "",
        f"- Instances per partition: {diag['instances_min']}–{diag['instances_max']} "
        f"(share of correct responses {diag['positive_share_mean']:.3f}); model fits: "
        f"{diag['fits']}; logistic regressions stopped at max_iter: {diag['lr_not_converged']}.",
        f"- Out-of-group accuracy of the response classifier: "
        f"{diag['instance_accuracy_mean']:.3f}; share classified \"True\": "
        f"{diag['predicted_true_share_mean']:.3f}.",
        f"- Sub-sets in which every response gives the same answer (empty out-group, "
        f"decision B1-2 / B6-4): {diag['unanimous_subsets_total']} over all partitions "
        f"({diag['undefined_instances_total']} instances without out-group features).",
        f"- Sub-sets whose correct answer has no vote (RP cannot succeed there): "
        f"{diag['subsets_without_vote_for_correct_mean']:.2f} per partition; sub-sets with "
        f"a tied majority (OQ28): {diag['majority_tie_subsets_mean']:.2f} per partition.",
        f"- Library warnings during the fits: {diag['warnings'] or 'none'}.",
    ]
    text = ML.format_table_4_2(summary, run.seeds, run.assignment, n, run.synthetic,
                               extra=extra)
    write_text(run.out / "rp_table_4_2.md", text)
    LOG.info("\n%s", text)
    run.copy([run.out / f for f in ("rp_table_4_2.md", "rp_summary.json", "rp_per_seed.csv",
                                    "rp_outcomes.json", "rp_predictions.csv")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
