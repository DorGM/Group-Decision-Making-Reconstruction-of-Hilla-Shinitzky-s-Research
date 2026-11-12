"""Step B7 (plan §4.4, §5.7, §9.3): the AP pipeline — ACR instances (one
per answer with a vote), the soft-voting ensemble, leave-one-group-out, the
thesis's three-case rule (asserted equal to argmax P(True)) — and the
headline table in the shape of TH Fig 5.3 (T23, with T24 and T25).

  python scripts/run_b7_ap.py            data = config/worldB.yaml

The RP row of the headline table is read from step B6's output
(<run>/b6_rp/rp_outcomes.json, strategy prop + avgp); run B6 first.

On the synthetic stand-in every output is marked "SYNTHETIC — not a
reproduction" and goes to runs/worldB_synthetic/b7_ap/; AP / RP outcomes
are not calibration targets (plan §2.3). A real file is written to
runs/worldB_real/b7_ap/: UNTESTED ON REAL DATA.

Outputs: ap_t23.md, ap_summary.json, ap_per_seed.csv, ap_outcomes.json
(read by steps B8, B9, B11), ap_predictions.csv, b7_ap.log.
"""
import sys

import _worldb as W
from _worldb import LOG, write_json, write_text
from cdm.approaches import rp
from cdm.eval import worldB_ml as ML
from cdm.eval import worldB_rules as WR
from cdm.eval import worldB_thesis_values as T

CLASSIFIER = "soft_vote"
AP_FIELDS = ("partition_seed", "correct", "n_subsets", "ties", "low", "high",
             "argmax_correct", "three_case_differs_from_argmax", "case_one_true",
             "case_several_true", "case_none_true", "case_unanimous")


def per_seed_rows(runs):
    for r in runs:
        s = r["summary"]
        row = {"partition_seed": W.seed_key(r["seed"]), "n_subsets": s["n"]}
        row.update({k: s[k] for k in AP_FIELDS if k in s})
        yield row


def rp_headline(run):
    """(cell, counts) of RP prop + avgp from step B6, or (None, None)."""
    doc = W.load_outcomes(run, run.sibling("b6_rp") / "rp_outcomes.json", "RP", CLASSIFIER)
    if doc is None:
        return None, None
    rp_runs = W.runs_from_outcomes(doc)
    counts = [int(r["correct"][rp.FINAL_NAME].sum()) for r in rp_runs]
    return ML._cell(counts, doc["n_subsets"]), counts


def main(argv=None):
    args = W.parser(__doc__).parse_args(argv)
    run = W.Run(args, "b7_ap", "AP pipeline (step B7)")
    runs = W.run_seeds(run, "AP", CLASSIFIER)
    n = runs[0]["n_subsets"]
    cell = ML.summarise_ap(runs)
    rules = ML.summarise_rules(runs)
    diag = ML.diagnostics(runs)
    # plan §4.4: the three-case rule equals argmax P(True) (asserted per
    # sub-set inside ap.evaluate for the soft vote; re-checked here)
    if cell["total_three_case_differs_from_argmax"] != 0:
        raise AssertionError("the three-case rule and argmax P(True) differ (plan §4.4)")
    rp_cell, rp_counts = rp_headline(run)
    if rp_cell is None:
        LOG.warning("step B6 output not found: the RP row of the T23 table is empty")
    mr = rules["MR"]["counts"]
    pre = {"AP": ML.pre_vs_mr(cell["counts"], mr, n),
           "RP": ML.pre_vs_mr(rp_counts, mr, n) if rp_counts else None}

    doc = run.head("B7", "§4.4, §5.7, §9.3 row B7, T23, T24, T25")
    doc.update({
        "approach": "AP", "classifier": CLASSIFIER, "n_subsets": n,
        "rule": "three cases of TH p.41; asserted equal to argmax P(True) in every "
                "sub-set of every partition (plan §4.4)",
        "three_case_equals_argmax_in_every_subset": True,
        "run": {"AP": cell, "RP_prop_avgp_from_b6": rp_cell, "rules": rules,
                "pre_vs_MR": pre},
        "diagnostics": diag,
        "thesis": {"source": "TH Fig 5.3 p.76; MCF §5; plan T23-T25",
                   "T23_rate": T.T23_RATE, "T23_count": T.T23_COUNT,
                   "T24_rate": WR.THESIS_RATE, "T24_count": WR.THESIS_COUNT,
                   "T25_pre_vs_MR": T.T25_PRE_VS_MR},
        "minutes": run.minutes(),
    })
    if run.synthetic:
        doc["note"] = ("SYNTHETIC stand-in: the rule rows agree with the thesis BY "
                       "CONSTRUCTION (calibration targets, plan §2.3); the RP and AP "
                       "rows are not a reproduction of T23 and no difference is computed.")
    write_json(run.out / "ap_summary.json", doc)
    write_json(run.out / "ap_outcomes.json", W.outcomes_document(run, runs, "AP", CLASSIFIER))
    run.csv(run.out / "ap_per_seed.csv", AP_FIELDS, per_seed_rows(runs))
    run.csv(run.out / "ap_predictions.csv", W.PREDICTION_FIELDS, W.prediction_rows(runs))

    extra = [
        "", "#### AP run diagnostics" + (f" — {run.mark}" if run.synthetic else ""), "",
        f"- The three-case rule (TH p.41) and argmax P(True) return the same answer in "
        f"every sub-set of every partition (plan §4.4; asserted): "
        f"{len(runs) * n} sub-set evaluations, 0 differences.",
        f"- Which case applied (mean sub-sets per partition): exactly one answer \"True\" "
        f"{cell['mean_case_one_true']:.2f}; several \"True\" {cell['mean_case_several_true']:.2f}; "
        f"none \"True\" {cell['mean_case_none_true']:.2f}; single voted answer "
        f"{cell['mean_case_unanimous']:.2f}.",
        f"- Ties of the deciding probability (canonical order decides; decision B7-2): "
        f"{cell['mean_ties']:.2f} sub-sets per partition.",
        f"- Instances per partition: {diag['instances_min']}–{diag['instances_max']} "
        f"(share of correct answers {diag['positive_share_mean']:.3f}); model fits: "
        f"{diag['fits']}; logistic regressions stopped at max_iter: {diag['lr_not_converged']}.",
        f"- Sub-sets with a single voted answer (empty out-group, decision B1-2 / B6-4): "
        f"{diag['unanimous_subsets_total']} over all partitions; sub-sets whose correct "
        f"answer has no vote (no correct instance, B2-3): "
        f"{diag['subsets_without_vote_for_correct_mean']:.2f} per partition.",
        f"- Library warnings during the fits: {diag['warnings'] or 'none'}.",
    ]
    text = ML.format_t23(rp_cell, cell, rules, pre, run.seeds, run.assignment, n,
                         run.synthetic, extra=extra)
    write_text(run.out / "ap_t23.md", text)
    LOG.info("\n%s", text)
    run.copy([run.out / f for f in ("ap_t23.md", "ap_summary.json", "ap_per_seed.csv",
                                    "ap_outcomes.json", "ap_predictions.csv")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
