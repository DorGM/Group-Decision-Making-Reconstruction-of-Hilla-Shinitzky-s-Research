"""Step B8 (plan §5.7, §8 OQ35, §9.3): the alternative classifiers Bag-KNN
and SVM for RP (nine strategies each) and AP, in the shape of TH Table A.7
(T27) and Table A.8 (T28).

  python scripts/run_b8_alt.py           data = config/worldB.yaml

The "Vote soft" rows are read from steps B6 and B7
(<run>/b6_rp/rp_outcomes.json, <run>/b7_ap/ap_outcomes.json); run them
first, otherwise those rows say "not run".

On the synthetic stand-in every output is marked "SYNTHETIC — not a
reproduction" and goes to runs/worldB_synthetic/b8_alt/. A real file is
written to runs/worldB_real/b8_alt/: UNTESTED ON REAL DATA.

Outputs: alt_tables_a7_a8.md, alt_summary.json, alt_per_seed.csv,
{rp,ap}_outcomes_<classifier>.json, {rp,ap}_predictions_<classifier>.csv,
b8_alt.log.
"""
import sys

import _worldb as W
from _worldb import LOG, write_json, write_text
from cdm.approaches import rp
from cdm.eval import worldB_ml as ML
from cdm.eval import worldB_thesis_values as T
from cdm.models import worldB_classifiers as WC

ALTERNATIVES = ("bag_knn", "svm")
FIELDS = ("approach", "classifier", "partition_seed", "strategy", "correct", "n_subsets",
          "primary_ties", "final_ties", "low", "high", "three_case_differs_from_argmax")


def per_seed_rows(approach, classifier, runs):
    for r in runs:
        base = {"approach": approach, "classifier": classifier,
                "partition_seed": W.seed_key(r["seed"])}
        if approach == "RP":
            for name in rp.STRATEGY_NAMES:
                s = r["summary"][name]
                yield dict(base, strategy=name, correct=s["correct"], n_subsets=s["n"],
                           primary_ties=s["primary_ties"], final_ties=s["final_ties"],
                           low=s["low"], high=s["high"])
        else:
            s = r["summary"]
            yield dict(base, strategy="three_case", correct=s["correct"], n_subsets=s["n"],
                       primary_ties=s["ties"], low=s["low"], high=s["high"],
                       three_case_differs_from_argmax=s["three_case_differs_from_argmax"])


def main(argv=None):
    args = W.parser(__doc__).parse_args(argv)
    run = W.Run(args, "b8_alt", "RP / AP with Bag-KNN and SVM (step B8)")
    rp_sum, ap_sum, diag, rows, files = {}, {}, {}, [], []
    n = None
    for clf in ALTERNATIVES:
        for approach in ("RP", "AP"):
            runs = W.run_seeds(run, approach, clf)
            n = runs[0]["n_subsets"]
            (rp_sum if approach == "RP" else ap_sum)[clf] = (
                ML.summarise_rp(runs) if approach == "RP" else ML.summarise_ap(runs))
            diag[f"{approach}/{clf}"] = ML.diagnostics(runs)
            rows += list(per_seed_rows(approach, clf, runs))
            low = approach.lower()
            write_json(run.out / f"{low}_outcomes_{clf}.json",
                       W.outcomes_document(run, runs, approach, clf))
            run.csv(run.out / f"{low}_predictions_{clf}.csv", W.PREDICTION_FIELDS,
                    W.prediction_rows(runs))
            files += [run.out / f"{low}_outcomes_{clf}.json",
                      run.out / f"{low}_predictions_{clf}.csv"]
    # the "Vote soft" rows from steps B6 / B7
    for approach, step, store in (("RP", "b6_rp", rp_sum), ("AP", "b7_ap", ap_sum)):
        doc = W.load_outcomes(run, run.sibling(step) / f"{approach.lower()}_outcomes.json",
                              approach, "soft_vote")
        if doc is None:
            LOG.warning("step %s output not found: the Vote soft rows are empty", step)
            continue
        runs = W.runs_from_outcomes(doc)
        store["soft_vote"] = ML.summarise_rp(runs) if approach == "RP" else ML.summarise_ap(runs)
        diag[f"{approach}/soft_vote"] = ML.diagnostics(runs)
        rows += list(per_seed_rows(approach, "soft_vote", runs))

    doc = run.head("B8", "§5.7, §8 OQ35, §9.3 row B8, T27, T28")
    doc.update({
        "n_subsets": n,
        "classifiers": {c: WC.THESIS_LABEL[c] for c in WC.CLASSIFIERS},
        "run": {"RP": rp_sum, "AP": ap_sum}, "diagnostics": diag,
        "thesis": {"source": "TH Table A.7, Table A.8 p.114 (success %, tie %)",
                   "table_a_7": {c: {k: list(v) for k, v in t.items()}
                                 for c, t in T.TABLE_A_7.items()},
                   "table_a_8": T.TABLE_A_8},
        "minutes": run.minutes(),
    })
    if run.synthetic:
        doc["note"] = ("SYNTHETIC stand-in: not a reproduction of Tables A.7 / A.8; no "
                       "difference is computed.")
    write_json(run.out / "alt_summary.json", doc)
    run.csv(run.out / "alt_per_seed.csv", FIELDS, rows)

    def line(key):
        d = diag.get(key)
        if d is None:
            return f"- {key}: not run."
        return (f"- {key}: share classified \"True\" {d['predicted_true_share_mean']:.3f} "
                f"(share of correct instances {d['positive_share_mean']:.3f}); instances "
                f"classified \"False\" although P(True) > 0.5: "
                f"{d['label_false_but_p_true_above_half']}, classified \"True\" although "
                f"P(True) <= 0.5: {d['label_true_but_p_true_not_above_half']} (over all "
                f"partitions); library warnings: {d['warnings'] or 'none'}.")

    a7 = ML.format_a7(rp_sum, run.seeds, run.assignment, n, run.synthetic)
    extra = ["", "#### Classifier diagnostics" + (f" — {run.mark}" if run.synthetic else ""),
             ""] + [line(f"{a}/{c}") for a in ("RP", "AP") for c in T.A7_CLASSIFIER_ORDER]
    a8 = ML.format_a8(ap_sum, run.seeds, run.assignment, n, run.synthetic, extra=extra)
    text = a7.rstrip("\n") + "\n\n" + a8
    write_text(run.out / "alt_tables_a7_a8.md", text)
    LOG.info("\n%s", text)
    run.copy([run.out / "alt_tables_a7_a8.md", run.out / "alt_summary.json",
              run.out / "alt_per_seed.csv"] + files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
