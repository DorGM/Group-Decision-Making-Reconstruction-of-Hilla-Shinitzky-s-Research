"""Step B9 (plan §5.8, §9.3): McNemar (chi-square with continuity
correction) and one-tailed pooled proportion tests of RP and AP against
SP, MR, WC and HAC, in the shape of TH Table 5.11 (T29).

  python scripts/run_b9_stats.py         data = config/worldB.yaml

Reads the per-sub-set outcomes of steps B6 and B7
(<run>/b6_rp/rp_outcomes.json: strategy prop + avgp;
<run>/b7_ap/ap_outcomes.json); run them first. No model is fitted here.

On the synthetic stand-in every output is marked "SYNTHETIC — not a
reproduction" and goes to runs/worldB_synthetic/b9_stats/. A real file is
written to runs/worldB_real/b9_stats/: UNTESTED ON REAL DATA.

Outputs: stats_table_5_11.md, stats_summary.json, stats_per_seed.csv,
b9_stats.log.
"""
import sys

import numpy as np

import _worldb as W
from _worldb import LOG, write_json, write_text
from cdm.approaches import rp
from cdm.eval import worldB_ml as ML
from cdm.eval import worldB_stats as WS
from cdm.eval import worldB_thesis_values as T

FIELDS = ("partition_seed", "approach", "rule", "n", "x_approach", "x_rule", "b", "c",
          "mcnemar_chi2", "mcnemar_p", "z", "proportion_p")


def main(argv=None):
    args = W.parser(__doc__).parse_args(argv)
    run = W.Run(args, "b9_stats", "significance tests of RP / AP (step B9)")
    docs = {"RP": W.load_outcomes(run, run.sibling("b6_rp") / "rp_outcomes.json",
                                  "RP", "soft_vote"),
            "AP": W.load_outcomes(run, run.sibling("b7_ap") / "ap_outcomes.json",
                                  "AP", "soft_vote")}
    if all(d is None for d in docs.values()):
        raise SystemExit("no output of step B6 or B7 found for this data and these "
                         "seeds; run scripts/run_b6_rp.py and scripts/run_b7_ap.py first")
    outcomes, rules, n = {}, None, None
    for approach, doc in docs.items():
        if doc is None:
            LOG.warning("%s outcomes not found: its columns are empty", approach)
            continue
        runs = W.runs_from_outcomes(doc)
        n = doc["n_subsets"]
        outcomes[approach] = [r["correct"][rp.FINAL_NAME] if approach == "RP" else r["correct"]
                              for r in runs]
        these = [r["rules"] for r in runs]
        if rules is not None:           # same partitions -> same rule outcomes
            for x, y in zip(rules, these):
                if any(not np.array_equal(x[k], y[k]) for k in x):
                    raise AssertionError("steps B6 and B7 disagree on the rule outcomes")
        rules = these
    tab = WS.table(outcomes, rules)

    doc = run.head("B9", "§5.8 items 2 and 4, §9.3 row B9, T29")
    doc.update({
        "n_subsets": n, "alpha": T.ALPHA,
        "tests": {"mcnemar": "two-sided chi-square with continuity correction "
                             "(cdm.eval.stats.mcnemar_test)",
                  "proportion": "pooled two-sample z, one-tailed H1 approach > rule, exact "
                                "upper tail (cdm.eval.stats.two_proportion_ztest)"},
        "rp_strategy": rp.FINAL_NAME,
        "run": tab,
        "thesis": {"source": "TH Table 5.11 p.76 = MCF Table 5 (printed p-values: "
                             "[McNemar, proportion])",
                   "table_5_11": {r: {a: list(v) for a, v in c.items()}
                                  for r, c in T.TABLE_5_11.items()}},
        "thesis_recomputed_from_published_counts": WS.thesis_recomputed(),
        "minutes": run.minutes(),
    })
    if run.synthetic:
        doc["note"] = ("SYNTHETIC stand-in: 'run' is not a reproduction of Table 5.11. "
                       "'thesis_recomputed_from_published_counts' uses no data.")
    write_json(run.out / "stats_summary.json", doc)
    rows = []
    for rule in WS.RULES:
        for approach, cell in tab[rule].items():
            for seed, p in zip(run.seeds, cell["per_partition"]):
                row = {"partition_seed": W.seed_key(seed), "approach": approach, "rule": rule}
                row.update({k: ("" if p[k] is None else p[k]) for k in FIELDS[3:]})
                rows.append(row)
    run.csv(run.out / "stats_per_seed.csv", FIELDS, rows)
    text = WS.format_table(tab, run.synthetic, run.mark,
                           ML.partition_line(run.seeds, run.assignment, n), ML.wrap)
    write_text(run.out / "stats_table_5_11.md", text)
    LOG.info("\n%s", text)
    run.copy([run.out / f for f in ("stats_table_5_11.md", "stats_summary.json",
                                    "stats_per_seed.csv")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
