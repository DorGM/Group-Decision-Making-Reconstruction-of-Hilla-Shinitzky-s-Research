"""RP / AP evaluation over the partition seeds, and the tables in the shape
of TH Table 4.2, Fig 5.3 (T23), Table A.7 and Table A.8 (plan §5.7, §9.3
rows B6-B8; plan items T23, T25-T28).

    UNTESTED ON REAL DATA.

One partition = the 77 sub-sets of one seed (plan §8 OQ25: the original
assignment is unknown, so "random with a seed, repeated over 30 seeds;
report mean ± SD"). For every partition: instances (RCR / ACR), leave-one-
group-out predictions, the aggregation into one answer per sub-set, and
the four rule baselines on the same sub-sets (plan §5.7).

Fidelity note (plan §5.7): as in the thesis, all nine strategies are
reported and prop + avgp is the RP headline.

On the SYNTHETIC stand-in (plan §2.3) nothing in these tables is a
reproduction: RP / AP outcomes are not calibration targets and the
invented response model need not resemble the MTurk data. The run and the
thesis are in separate columns and no difference column is given.
"""
import math

import numpy as np

from cdm import aggregators as AG
from cdm.approaches import ap, rp
from cdm.approaches import worldB_instances as WI
from cdm.data.worldB import partition as WP
from cdm.data.worldB.loader import REAL_DATA_STATUS, SYNTHETIC_MARKER
from cdm.eval import stats as ST
from cdm.eval import worldB_logo as LG
from cdm.eval import worldB_rules as WR
from cdm.eval import worldB_thesis_values as T
from cdm.features import rcr
from cdm.models import worldB_classifiers as WC


# ---- options ----------------------------------------------------------------------------

def options_from_config(config=None):
    """Everything a run needs from config/defaults.yaml:
    OQ15 (hyper), OQ26 / OQ33 / OQ35 (checked), OQ27 (checked), OQ28
    (majority_rule), OQ8 + AM3 (rule tie policy and tolerance)."""
    from cdm.config import get_switch, load_config
    config = config if config is not None else load_config()
    oq27 = get_switch("OQ27", config)
    want = {"prop": "n_true_over_n_supporters",
            "ties": "chosen_tie_breaker_then_canonical_order",
            "tie_breaker_percentage": "share_of_77_subsets_where_primary_criterion_tied",
            "no_supporter_predicted_true": "prop_zero_for_all_then_tie_breaker"}
    if dict(oq27) != want:
        raise NotImplementedError(f"OQ27 = {oq27!r} is not implemented (only {want})")
    rules = AG.options_from_config(config)
    return {"hyper": WC.hyperparameters_from_config(config),
            "checked": dict(WC.oq26_from_config(config), OQ27=dict(oq27)),
            "majority_rule": rcr.options_from_config(config)["majority_rule"],
            "rule_policy": rules["policy"], "tol": float(rules["tol"]),
            "group_by": "subset",
            "empty_out_group": LG.DEFAULT_EMPTY_OUT_GROUP_RULE,
            "standalone_ties": "random"}


def seeds_from_config(data, config=None):
    """(seeds, assignment) of plan §8 OQ25: the file's original assignment
    if it has one (single "seed" None), else seeds 0 .. n_seeds - 1."""
    from cdm.config import get_switch, load_config
    oq25 = get_switch("OQ25", config if config is not None else load_config())
    if oq25["use_original_assignment_if_available"] and data.original_subsets is not None:
        return (None,), "original"
    return tuple(range(int(oq25["n_seeds"]))), "random"


# ---- one partition ----------------------------------------------------------------------

def rule_outcomes(subsets, policy="canonical_first", tol=AG.P.DEFAULT_TIE_ABS_TOL):
    """{rule: bool array over the sub-sets} for MR, WC, SP, HAC (plan §5.7)."""
    out = {r: np.zeros(len(subsets), dtype=bool) for r in WR.RULES_B}
    for g, case in enumerate(subsets.cases):
        res = WR.subset_outcome(case, policy, tol)
        for r in WR.RULES_B:
            out[r][g] = res[r]["correct"]
    return out


def run_partition(data, seed, assignment, approach, classifier, opts,
                  keep_predictions=True):
    """RP or AP with one classifier on one partition. Returns a plain dict
    (picklable; the drivers run the seeds in separate processes)."""
    subsets = WP.subset_cases(data, seed=0 if seed is None else seed, assignment=assignment)
    inst = WI.build_instances(subsets, approach, opts["majority_rule"])
    pred = LG.logo_predict(inst, classifier, opts["hyper"], opts["group_by"],
                           opts["empty_out_group"])
    out = {"seed": seed, "approach": approach, "classifier": classifier,
           "n_subsets": len(subsets), "case_ids": list(inst.case_ids),
           "task_ids": list(inst.task_ids), "sizes": subsets.sizes(),
           "instances": inst.info, "predictions_log": pred.log,
           "rules": rule_outcomes(subsets, opts["rule_policy"], opts["tol"])}
    if approach == "RP":
        res = rp.evaluate(inst, pred, opts["tol"], opts["standalone_ties"],
                          tie_seed=0 if seed is None else seed)
        out.update(summary=res.summary(), correct=res.correct, answer=res.answer,
                   primary_tie=res.primary_tie, final_tie=res.final_tie,
                   unanimous=res.unanimous)
    else:
        # plan §4.4: assert three-case == argmax P(True). Guaranteed only
        # when the label is the arg-max of the probability (not for SVC).
        res = ap.evaluate(inst, pred, opts["tol"],
                          assert_equivalence=classifier != "svm")
        out.update(summary=res.summary(), correct=res.correct, answer=res.answer,
                   argmax_correct=res.argmax_correct, case=list(res.case), tie=res.tie)
    if keep_predictions:
        out["predictions"] = {"group": inst.group, "member": inst.member,
                              "answer_index": inst.answer_index, "y": inst.y,
                              "label": pred.label, "p_true": pred.p_true,
                              "p_false": pred.p_false}
    return out


# ---- summaries over the partitions ------------------------------------------------------

def mean_sd(xs):
    """(mean, sample SD ddof 1) — the convention of step B5 (R2); SD is
    NaN for a single partition."""
    xs = [float(x) for x in xs]
    m = math.fsum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _cell(xs, n):
    m, sd = mean_sd(xs)
    return {"mean_count": m, "sd_count": sd, "min_count": min(xs), "max_count": max(xs),
            "mean_rate": m / n, "sd_rate": sd / n, "counts": [int(x) for x in xs]}


def summarise_rp(runs):
    """{strategy: mean / SD / range of the success count, tie percentages}."""
    n = runs[0]["n_subsets"]
    out = {}
    for name in rp.STRATEGY_NAMES:
        cell = _cell([r["summary"][name]["correct"] for r in runs], n)
        ties = [r["summary"][name]["primary_ties"] for r in runs]
        final = [r["summary"][name]["final_ties"] for r in runs]
        cell.update(
            mean_tie_pct=100.0 * mean_sd(ties)[0] / n,
            min_tie_pct=100.0 * min(ties) / n, max_tie_pct=100.0 * max(ties) / n,
            primary_ties=[int(t) for t in ties],
            mean_final_ties=mean_sd(final)[0],
            mean_low=mean_sd([r["summary"][name]["low"] for r in runs])[0],
            mean_high=mean_sd([r["summary"][name]["high"] for r in runs])[0])
        out[name] = cell
    return out


def summarise_ap(runs):
    n = runs[0]["n_subsets"]
    cell = _cell([r["summary"]["correct"] for r in runs], n)
    for key in ("ties", "low", "high", "argmax_correct", "three_case_differs_from_argmax",
                "case_one_true", "case_several_true", "case_none_true", "case_unanimous"):
        cell[f"mean_{key}"] = mean_sd([r["summary"][key] for r in runs])[0]
    cell["total_three_case_differs_from_argmax"] = int(sum(
        r["summary"]["three_case_differs_from_argmax"] for r in runs))
    return cell


def summarise_rules(runs):
    n = runs[0]["n_subsets"]
    return {rule: _cell([int(r["rules"][rule].sum()) for r in runs], n)
            for rule in WR.RULES_B}


def pre_vs_mr(approach_counts, mr_counts, n):
    """T25: PRE of an approach against MR (plan §5.1; TH §6.1 computes it
    from the rates). Returns the PRE of the mean rates and mean / SD of the
    per-partition PREs."""
    # stats.pre(reference, compared): the rule is the reference
    per = [ST.pre(m / n, a / n) for a, m in zip(approach_counts, mr_counts)]
    m, sd = mean_sd(per)
    return {"from_mean_rates": ST.pre(mean_sd(mr_counts)[0] / n,
                                      mean_sd(approach_counts)[0] / n),
            "mean_of_partitions": m, "sd_of_partitions": sd}


def diagnostics(runs):
    """Sums / ranges of the run logs over the partitions."""
    logs = [r["predictions_log"] for r in runs]
    inst = [r["instances"] for r in runs]
    warn = {}
    for lg in logs:
        for k, v in lg["warnings"].items():
            warn[k] = warn.get(k, 0) + v
    return {
        "n_partitions": len(runs), "fits": int(sum(lg["n_fits"] for lg in logs)),
        "lr_not_converged": int(sum(lg["lr_not_converged"] for lg in logs)),
        "instances_min": min(i["n_instances"] for i in inst),
        "instances_max": max(i["n_instances"] for i in inst),
        "positive_share_mean": mean_sd([i["positive_share"] for i in inst])[0],
        "predicted_true_share_mean": mean_sd([lg["predicted_true_share"] for lg in logs])[0],
        "instance_accuracy_mean": mean_sd([lg["instance_accuracy"] for lg in logs])[0],
        "undefined_instances_total": int(sum(i["undefined_instances"] for i in inst)),
        "unanimous_subsets_total": int(sum(i["unanimous_subsets"] for i in inst)),
        "majority_tie_subsets_mean": mean_sd([i["majority_tie_subsets"] for i in inst])[0],
        "subsets_without_vote_for_correct_mean": mean_sd(
            [i["subsets_without_vote_for_correct"] for i in inst])[0],
        "label_false_but_p_true_above_half": int(sum(
            lg["label_false_but_p_true_above_half"] for lg in logs)),
        "label_true_but_p_true_not_above_half": int(sum(
            lg["label_true_but_p_true_not_above_half"] for lg in logs)),
        "warnings": warn}


# ---- rendering --------------------------------------------------------------------------

def run_label(synthetic):
    return "run (synthetic)" if synthetic else "run (real file)"


def wrap(lines, synthetic):
    """Markdown text whose first and last non-empty lines carry the marker
    for synthetic data (decision B3-13); for a real file the loader
    status is stated at the end."""
    if synthetic:
        return "\n".join([f"**{SYNTHETIC_MARKER}**", ""] + list(lines)
                         + ["", f"**{SYNTHETIC_MARKER}**"]) + "\n"
    return "\n".join(list(lines) + ["", f"Loader status: {REAL_DATA_STATUS}."]) + "\n"


def partition_line(seeds, assignment, n_subsets):
    if assignment == "original":
        return f"Sub-sets: {n_subsets}; partition: the file's original assignment."
    return (f"Sub-sets per partition: {n_subsets}; partition: random, {len(seeds)} seeds "
            f"({seeds[0]}..{seeds[-1]}) (plan OQ25); SD = sample SD over the seeds (ddof 1).")


def _pm(m, sd, d=3):
    return f"{m:.{d}f}" if sd != sd else f"{m:.{d}f} ± {sd:.{d}f}"


def _tb(name):
    primary, _, breaker = name.partition("+")
    return primary, breaker or "–"


def strategy_rows_md(summary, thesis, label, first_col=None):
    """Rows in the shape of TH Table 4.2 (one per strategy)."""
    rows = []
    for name in rp.STRATEGY_NAMES:
        c = summary[name]
        primary, breaker = _tb(name)
        t_succ, t_tie = thesis[name]
        lead = f"| {first_col} " if first_col else ""
        rows.append(
            f"{lead}| {primary} | {breaker} "
            f"| {_pm(100 * c['mean_rate'], 100 * c['sd_rate'], 1)}% "
            f"| {c['min_count']}–{c['max_count']} "
            f"| {c['mean_tie_pct']:.1f}% ({c['min_tie_pct']:.0f}–{c['max_tie_pct']:.0f}) "
            f"| {c['mean_final_ties']:.2f} "
            f"| {c['mean_low']:.2f} – {c['mean_high']:.2f} "
            f"| {t_succ}% | {t_tie}% |")
    return rows


def strategy_header_md(label, first_col=None, n=T.N_SUBSETS):
    lead = f"| {first_col} " if first_col else ""
    sep = "|---" if first_col else ""
    return [
        f"{lead}| Strategy | Tie breaker | {label}: success rate, mean ± SD "
        f"| {label}: count of {n}, min–max | {label}: tie breakers, mean % (min–max) "
        f"| {label}: ties left after the tie breaker (mean sub-sets) "
        f"| {label}: mean count if every remaining tie went wrong – right "
        f"| thesis: success rate | thesis: tie breakers |",
        f"{sep}|---|---|---|---|---|---|---|---|---|"]


TIE_NOTES = [
    "- \"Tie breakers\" = share of the sub-sets in which the primary criterion is tied "
    "between two or more answers (plan OQ27), including the case that no supporter of "
    "any answer is classified \"True\" (maj and prop are then 0 for all).",
    "- A tie left after the tie breaker goes to the first answer in the order of A_P "
    "(plan OQ27). maxp / avgp / wm alone have no tie breaker in Table 4.2 (\"–\"); "
    "Table A.7 prints \"rnd\" for them: a tie is resolved by a seeded random choice.",
    "- With this definition the tie percentage cannot depend on the tie breaker, so the "
    "three maj rows (and the three prop rows) are equal in the run by construction. The "
    "thesis prints 16 / 17 / 16 and 18 / 16 / 17; the reason is not stated (decision B6-7).",
]


def format_table_4_2(summary, seeds, assignment, n_subsets, synthetic,
                     classifier="soft_vote", extra=()):
    """Table in the shape of TH Table 4.2 with separate run / thesis columns."""
    label = run_label(synthetic)
    lines = ["### Response-Prediction (RP): classification-aggregation strategies "
             "(shape of TH Table 4.2, T26)"
             + (f" — {SYNTHETIC_MARKER}" if synthetic else ""), "",
             f"Classifier: {WC.THESIS_LABEL[classifier]} (soft voting over LDA, RF, XGBoost, "
             "LR, KNN; plan OQ26). Leave-one-group-out, group = sub-set (TH p.57). "
             + partition_line(seeds, assignment, n_subsets), ""]
    lines += strategy_header_md(label, n=n_subsets)
    lines += strategy_rows_md(summary, T.TABLE_4_2, label)
    lines += [""] + TIE_NOTES + [
        "", "As in the thesis (TH p.58), prop + avgp is the RP strategy used for the "
        "headline result; all nine strategies are reported."]
    if synthetic:
        lines += ["", "The thesis columns are printed for the table's shape only. The "
                  "synthetic stand-in was calibrated on the four rule baselines and on "
                  "nothing else (plan §2.3); its RP numbers carry no information about the "
                  "thesis's Table 4.2. No difference is computed."]
    return wrap(lines + list(extra), synthetic)


def format_t23(rp_cell, ap_cell, rules, pre, seeds, assignment, n_subsets, synthetic,
               extra=()):
    """Fig 5.3 as a table (T23 with the rule baselines T24) and T25."""
    label = run_label(synthetic)
    lines = ["### RP, AP and the rule baselines on the sub-sets (shape of TH Fig 5.3; "
             "T23, T24, T25)" + (f" — {SYNTHETIC_MARKER}" if synthetic else ""), "",
             "RP = soft vote, prop + avgp (TH p.57); AP = soft vote, three-case rule "
             "(TH p.41). " + partition_line(seeds, assignment, n_subsets), "",
             f"| Method | {label}: mean count ± SD | {label}: min–max | "
             f"{label}: mean rate ± SD | thesis: count of 77 | thesis: rate |",
             "|---|---|---|---|---|---|"]
    cells = [("RP", rp_cell, T.T23_COUNT["RP"], T.T23_RATE["RP"]),
             ("AP", ap_cell, T.T23_COUNT["AP"], T.T23_RATE["AP"])]
    cells += [(r, rules[r], WR.THESIS_COUNT[r], WR.THESIS_RATE[r]) for r in WR.RULES_B]
    for name, c, t_count, t_rate in cells:
        if c is None:
            lines.append(f"| {name} | not run | | | {t_count} | {t_rate:.3f} |")
            continue
        lines.append(
            f"| {name} | {_pm(c['mean_count'], c['sd_count'], 2)} "
            f"| {c['min_count']}–{c['max_count']} "
            f"| {_pm(c['mean_rate'], c['sd_rate'])} | {t_count} | {t_rate:.3f} |")
    # On the synthetic stand-in the PRE table has no thesis column: the synthetic
    # PRE is a within-synthetic quantity (plan §2.3, audit S16).
    thesis_col = not synthetic
    lines += ["", f"| PRE against MR (T25) | {label}: from the mean rates | "
              f"{label}: mean ± SD over the partitions |" + (" thesis |" if thesis_col else ""),
              "|---|---|---|" + ("---|" if thesis_col else "")]
    for name in ("RP", "AP"):
        p = pre.get(name)
        t_cell = f" {T.T25_PRE_VS_MR[name]:.3f} |" if thesis_col else ""
        if p is None:
            lines.append(f"| {name} | not run | |" + t_cell)
        else:
            lines.append(f"| {name} | {p['from_mean_rates']:.3f} "
                         f"| {_pm(p['mean_of_partitions'], p['sd_of_partitions'])} |"
                         + t_cell)
    lines += ["", "Thesis counts are the only integers of 77 that round to the printed "
              "rates (plan T23, T24). "
              + ("The synthetic PRE is a within-synthetic quantity (a property of the "
                 "invented response model) and is not comparable with the PRE reported "
                 "in the thesis; no thesis PRE is printed beside it." if synthetic else
                 "The PRE printed in the thesis is computed from rounded rates "
                 "(plan T25).")]
    if synthetic:
        lines += ["", "The four rule rows agree with the thesis BY CONSTRUCTION (the "
                  "generator was calibrated on them, plan §2.3). The RP and AP rows are "
                  "NOT a reproduction of T23: they show that the pipeline runs end to "
                  "end, and nothing about the thesis's result."]
    return wrap(lines + list(extra), synthetic)


def format_a7(summaries, seeds, assignment, n_subsets, synthetic, extra=()):
    """Shape of TH Table A.7: classifier x strategy."""
    label = run_label(synthetic)
    lines = ["### RP: classifiers and classification-aggregation strategies (shape of TH "
             "Table A.7, T27)" + (f" — {SYNTHETIC_MARKER}" if synthetic else ""), "",
             "Bag KNN = BaggingClassifier(KNeighborsClassifier()), SVM = SVC(probability="
             "True), library defaults (plan OQ35). "
             + partition_line(seeds, assignment, n_subsets), ""]
    lines += strategy_header_md(label, "Classifier", n_subsets)
    for clf in T.A7_CLASSIFIER_ORDER:
        if clf in summaries:
            lines += strategy_rows_md(summaries[clf], T.TABLE_A_7[clf], label,
                                      WC.THESIS_LABEL[clf])
        else:
            lines.append(f"| {WC.THESIS_LABEL[clf]} | not run | | | | | | | | |")
    lines += [""] + TIE_NOTES
    if synthetic:
        lines += ["", "Printed for the table's shape only; no difference is computed and "
                  "the synthetic values carry no information about the thesis's Table A.7."]
    return wrap(lines + list(extra), synthetic)


def format_a8(cells, seeds, assignment, n_subsets, synthetic, extra=()):
    """Shape of TH Table A.8: AP success per classifier."""
    label = run_label(synthetic)
    lines = ["### AP: classifiers (shape of TH Table A.8, T28)"
             + (f" — {SYNTHETIC_MARKER}" if synthetic else ""), "",
             partition_line(seeds, assignment, n_subsets), "",
             f"| Classifier | {label}: success rate, mean ± SD | {label}: count of {n_subsets}, "
             f"min–max | {label}: tied sub-sets (mean) | {label}: sub-sets where the "
             "three-case rule and argmax P(True) differ (total over the partitions) "
             "| thesis: success rate |", "|---|---|---|---|---|---|"]
    for clf in T.A8_CLASSIFIER_ORDER:
        c = cells.get(clf)
        if c is None:
            lines.append(f"| {WC.THESIS_LABEL[clf]} | not run | | | | {T.TABLE_A_8[clf]}% |")
            continue
        lines.append(
            f"| {WC.THESIS_LABEL[clf]} | {_pm(100 * c['mean_rate'], 100 * c['sd_rate'], 1)}% "
            f"| {c['min_count']}–{c['max_count']} | {c['mean_ties']:.2f} "
            f"| {c['total_three_case_differs_from_argmax']} | {T.TABLE_A_8[clf]}% |")
    lines += ["", "AP answer = the thesis's three-case rule (TH p.41). For Vote soft and "
              "Bag KNN it must equal argmax P(True) (plan §4.4; asserted in the run). "
              "For SVC the classification and the probability can disagree, so the two "
              "are compared and counted, not asserted (decision B8-1)."]
    if synthetic:
        lines += ["", "Printed for the table's shape only; no difference is computed and "
                  "the synthetic values carry no information about the thesis's Table A.8."]
    return wrap(lines + list(extra), synthetic)
