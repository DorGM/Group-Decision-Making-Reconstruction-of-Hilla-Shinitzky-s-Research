"""Step B11 (plan §9.3): the World-B report. Collects the outputs of steps
B5-B10 and writes every thesis-reported World-B item T23-T30 side by side: the thesis
value, the run's value, and whether a comparison of the two means anything.

  python scripts/run_b11_report.py       data = config/worldB.yaml

Synthetic stand-in -> results_worldB_synthetic.md (in
runs/worldB_synthetic/b11_report/; with --copy-results also in
results/worldB_synthetic/b11_report/). The report states at its top that it shows the pipeline running
end to end and nothing about the thesis's RP / AP results. A real file -> results_worldB_real.md: UNTESTED ON REAL DATA.

No model is fitted here; run steps B5-B10 first (a missing step is
reported as "not run").
"""
import json
import sys

import _worldb as W
from _worldb import LOG, write_json, write_text
from cdm.approaches import rp
from cdm.eval import worldB_ml as ML
from cdm.eval import worldB_rules as WR
from cdm.eval import worldB_stats as WS
from cdm.eval import worldB_thesis_values as T
from cdm.models import worldB_classifiers as WC

NOT_RUN = "not run"

#: Note on reading the synthetic rates (decision D3). The AUC figures were
#: computed once on partition seed 0 of the shipped synthetic data; they are
#: quoted, not recomputed here.
SYNTHETIC_RATES_NOTE = (
    "**Why the synthetic RP / AP rates must not be read against the thesis.** "
    "On the shipped stand-in (generator "
    "seed 0, `config/synthetic_worldB.yaml`) solvers are easy to identify: the "
    "single feature dPSv separates correct from wrong responses with an AUC of "
    "about 0.83 on partition seed 0 (computed once, not by these drivers; "
    "ChiSq alone 0.75, ChosenMajAns alone 0.69). This is a side effect of the "
    "invented prediction mechanism (a solver moves half of the predicted mass to "
    "the lure, the value at which the SP baseline reached its target of 33), not of "
    "any tuning towards RP / AP. The synthetic RP / AP success rates therefore "
    "reflect that invented mechanism and are NOT comparable with the thesis's "
    "0.636 / 0.545. In addition, the calibrated rule baselines hold for generator "
    "seed 0 only (decision B3-10).")


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _body(path, synthetic, mark):
    """A step's Markdown without its first / last marker lines."""
    if not path.is_file():
        return [f"_{path.name}: {NOT_RUN}._"]
    lines = path.read_text(encoding="utf-8").rstrip("\n").splitlines()
    if synthetic:
        while lines and (not lines[0].strip() or lines[0].strip() == f"**{mark}**"):
            lines.pop(0)
        while lines and (not lines[-1].strip() or lines[-1].strip() == f"**{mark}**"):
            lines.pop()
    return [("#" + ln if ln.startswith("###") else ln) for ln in lines]


def _pm(cell, d=3, scale=1.0, unit=""):
    if cell is None:
        return NOT_RUN
    m, sd = scale * cell["mean_rate"], scale * cell["sd_rate"]
    text = f"{m:.{d}f}{unit}" if sd != sd or sd is None else f"{m:.{d}f} ± {sd:.{d}f}{unit}"
    return f"{text} (count {cell['mean_count']:.2f}, range {cell['min_count']}–{cell['max_count']})"


def target_rows(run, docs):
    """Rows of the T23-T30 table: (id, quantity, thesis, run, meaningful, why)."""
    syn = run.synthetic
    no = "NO" if syn else "yes (UNTESTED ON REAL DATA)"
    why_ml = ("RP / AP outcomes are not calibration targets; the response model is invented "
              "(plan §2.3)" if syn else "real file")
    b5, b6, b7, b8, b9, b10 = (docs[k] for k in ("b5", "b6", "b7", "b8", "b9", "b10"))
    rows = []
    apc = b7["run"]["AP"] if b7 else None
    rpc = (b7["run"]["RP_prop_avgp_from_b6"] if b7 else None) or (
        b6["run"][rp.FINAL_NAME] if b6 else None)
    rows.append(("T23", "RP success rate (soft vote, prop + avgp), leave-one-group-out over "
                 "the 77 sub-sets", f"{T.T23_RATE['RP']:.3f} (49/77)", _pm(rpc), no, why_ml))
    rows.append(("T23", "AP success rate (soft vote, three-case rule)",
                 f"{T.T23_RATE['AP']:.3f} (42/77)", _pm(apc), no, why_ml))
    for rule in WR.RULES_B:
        cell = None
        if b7:
            cell = b7["run"]["rules"][rule]
        elif b5:
            cell = dict(b5["run"][rule])
        rows.append(("T24", f"{rule} success rate on the sub-sets",
                     f"{WR.THESIS_RATE[rule]:.3f} ({WR.THESIS_COUNT[rule]}/77)", _pm(cell),
                     "NO — agreement is BY CONSTRUCTION" if syn else no,
                     "the generator was calibrated on exactly these four counts over "
                     "these seeds (plan §2.3), for generator seed 0 only" if syn else "real file"))
    for name in ("RP", "AP"):
        p = b7["run"]["pre_vs_MR"].get(name) if b7 else None
        rows.append(("T25", f"PRE of {name} against MR", f"{T.T25_PRE_VS_MR[name]:.3f}",
                     NOT_RUN if not p else f"{p['from_mean_rates']:.3f} (from the mean rates)",
                     no, why_ml))
    for name in rp.STRATEGY_NAMES:
        cell = b6["run"][name] if b6 else None
        t_s, t_t = T.TABLE_4_2[name]
        run_text = NOT_RUN if cell is None else (
            f"{100 * cell['mean_rate']:.1f} ± {100 * cell['sd_rate']:.1f}% success; "
            f"{cell['mean_tie_pct']:.1f}% tie breakers")
        rows.append(("T26", f"RP strategy {name} (Table 4.2)",
                     f"{t_s}% success; {t_t}% tie breakers", run_text, no, why_ml))
    for clf in ("bag_knn", "svm"):
        cells = b8["run"]["RP"].get(clf) if b8 else None
        lo = min(v[0] for v in T.TABLE_A_7[clf].values())
        hi = max(v[0] for v in T.TABLE_A_7[clf].values())
        tl = min(v[1] for k, v in T.TABLE_A_7[clf].items() if "+" in k)
        th = max(v[1] for k, v in T.TABLE_A_7[clf].items() if "+" in k)
        if cells:
            rates = [100 * c["mean_rate"] for c in cells.values()]
            ties = [c["mean_tie_pct"] for k, c in cells.items() if "+" in k]
            run_text = (f"{min(rates):.1f}–{max(rates):.1f}% success over the 9 strategies; "
                        f"{min(ties):.1f}–{max(ties):.1f}% tie breakers (maj / prop rows)")
        else:
            run_text = NOT_RUN
        rows.append(("T27", f"RP with {WC.THESIS_LABEL[clf]}, 9 strategies (Table A.7)",
                     f"{lo}–{hi}% success; {tl}–{th}% tie breakers (maj / prop rows)",
                     run_text, no, why_ml))
    for clf in T.A8_CLASSIFIER_ORDER:
        cell = b8["run"]["AP"].get(clf) if b8 else None
        rows.append(("T28", f"AP with {WC.THESIS_LABEL[clf]} (Table A.8)",
                     f"{T.TABLE_A_8[clf]}%",
                     NOT_RUN if cell is None else
                     f"{100 * cell['mean_rate']:.1f} ± {100 * cell['sd_rate']:.1f}%",
                     no, why_ml))
    for approach in WS.APPROACHES:
        for rule in WS.RULES:
            cell = b9["run"][rule].get(approach) if b9 else None
            mc, pr = T.TABLE_5_11[rule][approach]
            if cell is None:
                run_text = NOT_RUN
            else:
                n = cell["n_partitions"]
                m, p = cell["mcnemar"], cell["proportion"]
                run_text = (f"McNemar median {WS.fmt_p(m['median'])}, p < 0.05 in "
                            f"{m['n_below_alpha']} of {n}; proportion median "
                            f"{WS.fmt_p(p['median'])}, p < 0.05 in {p['n_below_alpha']} of {n}")
            rows.append(("T29", f"{approach} vs {rule}: McNemar / one-tailed proportion "
                         "p-value (Table 5.11)", f"{mc} / {pr}", run_text, no, why_ml))
    rec = WS.thesis_recomputed()
    ok = all(_printed(rec[r][a]["proportion"], T.TABLE_5_11[r][a][1])
             and all(_printed(p, T.TABLE_5_11[r][a][0]) for p in rec[r][a]["mcnemar"])
             for r in WS.RULES for a in WS.APPROACHES)
    rows.append(("T29", "Test conventions: Table 5.11 recomputed from the published counts "
                 "(49, 42 vs 33, 34, 34, 22 of 77; pairs of plan §5.8)",
                 "16 printed p-values",
                 "all 16 printed values are matched after rounding to three decimals: the 8 "
                 "proportion-test values follow from the published counts alone; the 8 McNemar "
                 "values are matched by discordant pairs that the plan inferred because they fit "
                 "(they are not published). AP vs HAC, proportion: 0.00054 against a printed "
                 "0.001" if ok else "not all 16 printed values matched",
                 "YES — no data involved",
                 "identifies the test variants from the thesis's own numbers; it is not a "
                 "recomputation on data"))
    for approach, feats in (("RP", "RCR"), ("AP", "ACR")):
        cell = b10["run"][approach]["models"].get("xgboost_tree") if b10 else None
        rows.append(("T30", f"SHAP rank order of the {feats} features, top 5 "
                     f"({T.SHAP_FIGURE[approach]})",
                     ", ".join(T.SHAP_RANKING[approach][:5]),
                     NOT_RUN if cell is None else
                     ", ".join(cell["run_ranking"][:5]) + " (XGBoost, TreeExplainer)",
                     "NO — smoke run only" if syn else no,
                     ("plan row B10: acceptance on real data only; which invented features "
                      "carry information is a property of the generator."
                      + (" That dPSv is first in both columns is by design of the generator "
                         "(its prediction model is the SP mechanism) and is not evidence"
                         if approach == "RP" else "")) if syn else "real file"))
    return rows


def _printed(p, printed):
    if "E" in printed:
        mant, exp = f"{p:.1e}".split("e")
        return f"{mant}E{int(exp):+03d}" == printed
    return f"{p:.3f}" == printed


SPECIAL_CASES = [
    "### 3. Special cases and tie rules",
    "",
    "| Case | Rule in use | Basis | How often in this run |",
    "|---|---|---|---|",
]


def special_rows(docs):
    b6, b7, b8 = docs["b6"], docs["b7"], docs["b8"]
    d6 = b6["diagnostics"] if b6 else None
    d7 = b7["diagnostics"] if b7 else None
    ap = b7["run"]["AP"] if b7 else None

    def f(x, fmt="{:.2f}"):
        return NOT_RUN if x is None else fmt.format(x)
    svm_diff = (b8["run"]["AP"]["svm"]["total_three_case_differs_from_argmax"]
                if b8 and "svm" in b8["run"]["AP"] else None)
    rows = [
        ("Empty out-group: every response of a sub-set gives the same answer, so RCR "
         "features 5, 10, 14 and ACR features 5, 8 are undefined (decision B1-2, B2-1)",
         "The instances are left out of every training set and get no prediction; the "
         "sub-set's answer is its only voted answer (the only candidate for RP and AP). "
         "Switch: `raise` (decision B6-4)",
         "[INVENTED] labelled default; the thesis and the plan give no rule. Open question "
         "(decision B6-4)",
         f"RP: {f(d6 and d6['unanimous_subsets_total'], '{}')} sub-sets over all partitions; "
         f"AP: {f(d7 and d7['unanimous_subsets_total'], '{}')}"),
        ("Correct answer without a vote in the sub-set",
         "RP and AP can only return a voted answer, so the sub-set counts as a failure",
         "[EVIDENCE] TH p.41 (AP: only answers with at least one vote); implicit for RP",
         f"{f(d6 and d6['subsets_without_vote_for_correct_mean'])} sub-sets per partition"),
        ("Tied majority answer (ChosenMajAns / IsMajority)",
         "1 for every answer tied for the most votes", "plan OQ28 default [OPEN]",
         f"{f(d6 and d6['majority_tie_subsets_mean'])} sub-sets per partition"),
        ("RP, maj / prop: tie on the primary criterion (including: no supporter of any "
         "answer classified \"True\")",
         "the strategy's tie breaker (avgp, maxp or wm) among the tied answers; counted as "
         "a \"tie breaker\" case", "[EVIDENCE] TH p.40; counting rule plan OQ27 [OPEN]",
         "see the tie-breaker column of section 2.1"),
        ("RP, maj / prop: still tied after the tie breaker", "first answer in the order of A_P",
         "plan OQ27 [OPEN]",
         f"{f(b6 and b6['run'][rp.FINAL_NAME]['mean_final_ties'])} sub-sets per partition "
         "(prop + avgp)"),
        ("RP, maxp / avgp / wm alone: tie", "seeded random choice among the tied answers; "
         "switch `canonical_first`",
         "[EVIDENCE] TH Table A.7 prints the tie breaker \"rnd\"; reading it as random is "
         "plan §5.7 [INFERRED]",
         "maxp " + f(b6 and b6['run']['maxp']['mean_final_ties']) + ", avgp "
         + f(b6 and b6['run']['avgp']['mean_final_ties']) + ", wm "
         + f(b6 and b6['run']['wm']['mean_final_ties']) + " sub-sets per partition"),
        ("AP: tie of the deciding probability", "first answer in the order of A_P",
         "[INVENTED] by analogy with plan OQ27 (decision B7-2)",
         f"{f(ap and ap['mean_ties'])} sub-sets per partition"),
        ("AP: three-case rule vs argmax P(True)",
         "the literal three-case rule gives the answer; equality with argmax P(True) is "
         "asserted for the soft vote and Bag KNN, counted for SVC (its classification and "
         "probability can disagree)",
         "[EVIDENCE] TH p.41; [INFERRED] plan §4.4; decision B7-1, B8-1",
         f"soft vote: {f(ap and ap['total_three_case_differs_from_argmax'], '{}')} "
         f"differences; SVM: {f(svm_diff, '{}')} (all partitions)"),
        ("Float comparisons (probabilities, proportions)", "two values within 1e-9 are tied",
         "AM3 tolerance, extended to RP / AP [INVENTED] (decision B6-6)", "—"),
        ("Rule baselines MR / WC / SP / HAC: ties", "first answer in the order of A_P",
         "plan OQ8, AM3 (step B5, decision R3)", "see the B5 output"),
    ]
    return [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]


def main(argv=None):
    args = W.parser(__doc__).parse_args(argv)
    run = W.Run(args, "b11_report", "World-B report (step B11)")
    paths = {"b5": run.root / "b5_rules" / "rules_summary.json",
             "b6": run.sibling("b6_rp") / "rp_summary.json",
             "b7": run.sibling("b7_ap") / "ap_summary.json",
             "b8": run.sibling("b8_alt") / "alt_summary.json",
             "b9": run.sibling("b9_stats") / "stats_summary.json",
             "b10": run.root / "b10_shap" / "shap_summary.json"}
    docs = {k: _load(p) for k, p in paths.items()}
    for k in ("b6", "b7", "b8", "b9", "b10"):
        if docs[k] is not None and docs[k]["data"]["fingerprint"] != run.fingerprint:
            raise SystemExit(f"{paths[k]}: written for other data; re-run its step")
    syn, mark = run.synthetic, run.mark
    name = "results_worldB_synthetic.md" if syn else "results_worldB_real.md"
    n_seeds = len(run.seeds)
    minutes = {k: docs[k].get("minutes") for k in ("b6", "b7", "b8", "b9", "b10") if docs[k]}

    L = []
    L += ["# World B (RP / AP) — results on the "
          + ("SYNTHETIC stand-in" if syn else "real file") + (f" — {mark}" if syn else ""), ""]
    if syn:
        L += [
            "**What this report is.** The MTurk data of the thesis (22 multiple-choice "
            "tasks, 77 sub-sets) is not public (available on request according to the 2024 "
            "article) and has not been supplied. Everything below "
            "was computed on a synthetic stand-in: invented responses that copy only the "
            "published task sizes and solver shares (TH Table 4.1) and were calibrated on "
            "the four rule baselines of TH Fig 5.3.",
            "",
            "**What it shows.** The RP / AP code (steps B6–B10) runs end to end: "
            "instances, classifiers, leave-one-group-out, aggregation strategies, "
            "significance tests, SHAP, tables in the shape of the thesis's tables.",
            "",
            "**What it does not show.** The numbers in the \"run (synthetic)\" columns say "
            "carry NO information about the thesis's RP / AP results. Whether a run "
            "value lies near a thesis value or far from it carries no information: no "
            "synthetic parameter was tuned towards an RP / AP result (plan §2.3), and "
            "none should be. No difference between a run value and a thesis value is computed "
            "anywhere in this report.",
            "",
            SYNTHETIC_RATES_NOTE,
            "",
            "**Status of the real-data path.** Every driver takes the real file through "
            "`config/worldB.yaml` (`source: real`) with the same commands (section 5). "
            "That path is UNTESTED ON REAL DATA: the file's format is an assumption "
            "(decision L1–L12).", ""]
    else:
        L += [f"Loader status: {mark}. The file's format was assumed (decision L1–L12); "
              "check the loader's report before reading any number below.", ""]
    L += ["## 1. Thesis-reported items T23–T30 side by side" + (f" — {mark}" if syn else ""), "",
          f"Data: {len(run.data.tasks)} tasks, {run.data.n_responses} responses"
          + (f" (generator seed {run.data.log.get('seed')})" if syn else "") + ". "
          + ML.partition_line(run.seeds, run.assignment, T.N_SUBSETS)
          + " Run values are mean ± SD over the partitions with the range of the count of "
          "77 in brackets.", "",
          f"| Item | Quantity | thesis | {ML.run_label(syn)} | Is the comparison "
          "meaningful? | Why |", "|---|---|---|---|---|---|"]
    rows = target_rows(run, docs)
    L += [f"| {a} | {b} | {c} | {d} | {e} | {g} |" for a, b, c, d, e, g in rows]
    L += [""]
    if syn:
        b6 = docs["b6"]
        acc = b6["diagnostics"]["instance_accuracy_mean"] if b6 else None
        b7 = docs["b7"]

        def rate(name):
            cell = None
            if b7:
                cell = b7["run"]["AP"] if name == "AP" else b7["run"]["RP_prop_avgp_from_b6"]
            if cell is None and name == "RP" and b6:
                cell = b6["run"][rp.FINAL_NAME]
            return NOT_RUN if cell is None else f"{100 * cell['mean_rate']:.1f}%"
        L += ["Reading aid, not a result: on this stand-in the response classifier's "
              "out-of-group accuracy is " + (f"{acc:.3f}" if acc else NOT_RUN)
              + f", and RP / AP succeed on {rate('RP')} / {rate('AP')} of the sub-sets. "
              "These levels are a property "
              "of the generator (plan §2.3: its confidence and prediction models are "
              "invented and were calibrated on the rule baselines only), not a finding. "
              "The generator was left as it is: adjusting it until the RP / AP columns "
              "looked like the thesis's would be exactly the tuning the plan forbids, and "
              "would make the columns meaningless in a less visible way. One practical "
              "consequence: this run exercises the tie rules and the empty-out-group rule "
              "rarely or never (section 3, last column); they are covered by unit tests.", ""]
    L += ["## 2. Tables in the shape of the thesis's tables" + (f" — {mark}" if syn else ""), ""]
    for title, path in (
            ("2.1 RP strategies (TH Table 4.2)", run.sibling("b6_rp") / "rp_table_4_2.md"),
            ("2.2 Headline (TH Fig 5.3)", run.sibling("b7_ap") / "ap_t23.md"),
            ("2.3 Other classifiers (TH Tables A.7, A.8)",
             run.sibling("b8_alt") / "alt_tables_a7_a8.md"),
            ("2.4 Significance tests (TH Table 5.11)",
             run.sibling("b9_stats") / "stats_table_5_11.md"),
            ("2.5 SHAP (TH Figs 5.8, 5.9)", run.root / "b10_shap" / "shap_rcr_acr.md")):
        L += [f"### {title}" + (f" — {mark}" if syn else ""), ""] + _body(path, syn, mark) + [""]
    L += SPECIAL_CASES + special_rows(docs) + [""]
    L += [
        "Leave-one-group-out: a group is the set of instances of one sub-set (TH p.57; "
        "77 groups), as in the thesis. This implementation also contains a "
        "leave-one-task-out option (`group_by=\"task\"`, cdm/eval/worldB_logo.py); it is a "
        "controlled variant of this implementation, is off in every stored run, and is "
        "planned to be reported beside the thesis procedure on the real data.",
        "",
        "Strategy selection: as in the thesis (TH p.58), prop + avgp is the RP strategy "
        "used for the headline result; all nine strategies are reported.", "",
        "## 4. Open points raised by these steps" + (f" — {mark}" if syn else ""), "",
        "Full entries with tags: `docs/IMPLEMENTATION_DECISIONS.md`, section \"Phase B — World B "
        "(steps B6–B11)\". The open questions:", "",
        "- B6-2: weights, hyperparameters, scaling and seeds of the soft-voting ensemble "
        "are not stated (plan OQ26 defaults in use).",
        "- B6-4: rule for a sub-set in which every response gives the same answer.",
        "- B6-6: tie rules beyond the stated tie breaker.",
        "- B6-7: the thesis's tie percentages differ between tie breakers of the same "
        "strategy (16 / 17 / 16; Bag KNN 8 / 16 / 12), which the plan's definition cannot "
        "produce from one set of predictions.",
        "- B8-1: SVC's classification and probability can disagree; sklearn 1.9 "
        "deprecates `probability=True`.",
        "- B9-2: how to report significance over 30 partitions.",
        "- B10-1..3: model, partition and sample of the SHAP analysis.", "",
        "## 5. Commands (the same for the real file)" + (f" — {mark}" if syn else ""), "",
        "```",
        "python scripts/run_b5_rules.py     # rule baselines (T24)",
        "python scripts/run_b6_rp.py        # RP, Table 4.2 (T26)",
        "python scripts/run_b7_ap.py        # AP, Fig 5.3 (T23, T25)",
        "python scripts/run_b8_alt.py       # Bag-KNN, SVM: Tables A.7, A.8 (T27, T28)",
        "python scripts/run_b9_stats.py     # Table 5.11 (T29)",
        "python scripts/run_b10_shap.py     # Figs 5.8, 5.9 (T30)",
        "python scripts/run_b11_report.py   # this report",
        "```", "",
        "For the real file set `source: real` and the `real:` block of "
        "`config/worldB.yaml` (file paths, confidence scale, prediction scale). Outputs "
        "then go to `runs/worldB_real/` and the tables are labelled \"run (real file)\". "
        "UNTESTED ON REAL DATA.", "",
        f"Partitions: {n_seeds}. Wall-clock minutes of the steps (3 processes): "
        + ", ".join(f"{k} {v:.1f}" for k, v in minutes.items() if v is not None) + ".",
        f"Library versions: {W.versions()}."]
    text = ML.wrap(L, syn)
    write_text(run.out / name, text)
    doc = run.head("B11", "§9.3 row B11")
    doc.update({"report": name,
                "steps_found": {k: v is not None for k, v in docs.items()},
                "items": [dict(zip(("item", "quantity", "thesis", "run", "meaningful",
                                      "why"), r)) for r in rows]})
    write_json(run.out / "report_summary.json", doc)
    LOG.info("report written: %s (%d lines)", run.out / name, len(text.splitlines()))
    run.copy([run.out / name, run.out / "report_summary.json"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
