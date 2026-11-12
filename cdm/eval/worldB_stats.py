"""Significance tests of RP and AP against the rule baselines, in the shape
of TH Table 5.11 = MCF Table 5 (plan §5.8 items 2 and 4, §9.3 row B9; T29).

    UNTESTED ON REAL DATA.

TH p.75: "we performed McNemar matched-pair sign tests and one-tailed
proportion tests, comparing the results of RP and AP to each rule-based
method". The variants are not stated; plan §5.8 fixes them by recomputing
the printed p-values [INFERRED, strong]:
  * McNemar: two-sided chi-square with continuity correction,
    (|b - c| - 1)^2 / (b + c)  -> cdm.eval.stats.mcnemar_test;
  * proportion: pooled two-sample z-test, one-tailed (H1: approach > rule),
    n1 = n2 = 77 -> cdm.eval.stats.two_proportion_ztest (exact upper tail,
    AM1).
Nothing is re-implemented here; this module pairs the per-sub-set outcomes,
calls cdm.eval.stats and tabulates.

Choices the plan does not cover (decision B9-1, B9-2):
  * b + c = 0 (the two methods never disagree): McNemar is undefined
    (cdm.eval.stats rejects it) -> reported as "undefined", not as p = 1;
    a pooled standard error of 0 -> the proportion test is "undefined";
  * with 30 partition seeds there are 30 tables. The tests are run per
    partition; reported are the median p-value, its range, and the number
    of partitions with p < 0.05 (the thesis's level, TH p.65). p-values of
    different partitions of the same responses are not independent and are
    not pooled.

`thesis_recomputed()` recomputes Table 5.11 from the published counts
(plan §5.8: 49 / 42 of 77 for RP / AP; 33 / 34 / 34 / 22 for SP / MR / WC /
HAC; the discordant pairs back-solved in §5.8 item 4). That is a check of
the test conventions and needs no data.
"""
import statistics

import numpy as np

from cdm.eval import stats as ST
from cdm.eval import worldB_thesis_values as T

APPROACHES = ("RP", "AP")
RULES = T.TABLE_5_11_ORDER

#: Plan §5.8 item 4 [INFERRED]: discordant pairs (approach only, rule only)
#: that reproduce the printed McNemar p-values. Where two pairs fit, both
#: are listed. WC is printed with the same values as MR.
PLAN_PAIRS = {
    ("RP", "SP"): ((18, 2), (19, 3)), ("RP", "MR"): ((15, 0),), ("RP", "WC"): ((15, 0),),
    ("RP", "HAC"): ((34, 7),),
    ("AP", "SP"): ((11, 2),), ("AP", "MR"): ((9, 1),), ("AP", "WC"): ((9, 1),),
    ("AP", "HAC"): ((30, 10), (31, 11)),
}
PLAN_COUNTS = {"RP": 49, "AP": 42, "SP": 33, "MR": 34, "WC": 34, "HAC": 22}


def compare(approach_correct, rule_correct):
    """Both tests for one approach against one rule on one partition.

    Returns {n, x_approach, x_rule, b, c, mcnemar_chi2, mcnemar_p, z,
    proportion_p}; a value that is undefined is None.
    """
    a = np.asarray(approach_correct, dtype=bool)
    r = np.asarray(rule_correct, dtype=bool)
    if a.shape != r.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("two outcome vectors of the same, non-zero length are needed")
    n = int(a.size)
    b, c = ST.discordant_counts(a, r)
    out = {"n": n, "x_approach": int(a.sum()), "x_rule": int(r.sum()), "b": b, "c": c,
           "mcnemar_chi2": None, "mcnemar_p": None, "z": None, "proportion_p": None}
    if b + c > 0:
        out["mcnemar_chi2"], out["mcnemar_p"] = ST.mcnemar_test(b, c)
    try:
        out["z"], out["proportion_p"] = ST.two_proportion_ztest(
            out["x_approach"], n, out["x_rule"], n, alternative="larger")
    except ZeroDivisionError:
        pass
    return out


def _summary(values, alpha):
    defined = [v for v in values if v is not None]
    if not defined:
        return {"median": None, "min": None, "max": None, "n_defined": 0,
                "n_below_alpha": 0}
    return {"median": statistics.median(defined), "min": min(defined),
            "max": max(defined), "n_defined": len(defined),
            "n_below_alpha": sum(1 for v in defined if v < alpha)}


def table(approach_outcomes, rule_outcomes, alpha=T.ALPHA):
    """Table 5.11 over the partitions.

    approach_outcomes : {"RP": [bool array per partition], "AP": [...]}
                        (an approach may be missing)
    rule_outcomes     : [{rule: bool array} per partition]
    Returns {rule: {approach: {"per_partition": [compare(...)], "mcnemar":
    summary, "proportion": summary, "n_approach_ahead": partitions in
    which the approach has more successes than the rule}}}.
    """
    out = {}
    for rule in RULES:
        out[rule] = {}
        for approach in APPROACHES:
            if approach not in approach_outcomes:
                continue
            series = approach_outcomes[approach]
            if len(series) != len(rule_outcomes):
                raise ValueError("one outcome vector per partition is needed")
            per = [compare(a, r[rule]) for a, r in zip(series, rule_outcomes)]
            out[rule][approach] = {
                "per_partition": per,
                "mcnemar": _summary([p["mcnemar_p"] for p in per], alpha),
                "proportion": _summary([p["proportion_p"] for p in per], alpha),
                "n_approach_ahead": sum(1 for p in per if p["x_approach"] > p["x_rule"]),
                "n_partitions": len(per)}
    return out


def thesis_recomputed():
    """Table 5.11 recomputed from the published counts (no data).
    {rule: {approach: {"mcnemar": [p per fitting pair], "proportion": p}}}"""
    out = {}
    for rule in RULES:
        out[rule] = {}
        for approach in APPROACHES:
            pairs = PLAN_PAIRS[(approach, rule)]
            for b, c in pairs:
                assert b - c == PLAN_COUNTS[approach] - PLAN_COUNTS[rule]
            out[rule][approach] = {
                "pairs": [list(p) for p in pairs],
                "mcnemar": [ST.mcnemar_test(b, c)[1] for b, c in pairs],
                "proportion": ST.two_proportion_ztest(
                    PLAN_COUNTS[approach], T.N_SUBSETS, PLAN_COUNTS[rule], T.N_SUBSETS)[1]}
    return out


# ---- rendering ---------------------------------------------------------------------------

def fmt_p(p):
    if p is None:
        return "undefined"
    if p != p:
        return "nan"
    if p < 0.001:
        mant, exp = f"{p:.1e}".split("e")
        return f"{mant}E{int(exp):+03d}"
    return f"{p:.3f}"


def _run_cell(s, n):
    if s["n_defined"] == 0:
        return "undefined in every partition"
    if n == 1:
        return fmt_p(s["median"])
    text = (f"median {fmt_p(s['median'])} ({fmt_p(s['min'])} – {fmt_p(s['max'])}); "
            f"p < 0.05 in {s['n_below_alpha']} of {n}")
    if s["n_defined"] < n:
        text += f"; undefined in {n - s['n_defined']}"
    return text


def format_table(tab, synthetic, marker, partition_text, wrap):
    """Markdown in the shape of TH Table 5.11 with separate run / thesis
    columns. `wrap` = cdm.eval.worldB_ml.wrap."""
    label = "run (synthetic)" if synthetic else "run (real file)"
    rec = thesis_recomputed()
    present = [a for a in APPROACHES if any(a in tab[r] for r in RULES)]
    lines = ["### RP and AP against the rule baselines: p-values (shape of TH Table 5.11, "
             "T29)" + (f" — {marker}" if synthetic else ""), "",
             "McNemar = two-sided chi-square with continuity correction; proportion = "
             "pooled two-sample z-test, one-tailed, H1: approach > rule (plan §5.8). "
             + partition_text, ""]
    head = "| Rule "
    for a in APPROACHES:
        head += f"| {label}: vs {a}, McNemar | {label}: vs {a}, proportion "
    for a in APPROACHES:
        head += f"| thesis: vs {a}, McNemar | thesis: vs {a}, proportion "
    lines += [head + "|", "|---" * 9 + "|"]
    for rule in RULES:
        row = f"| {rule} "
        for a in APPROACHES:
            if a not in present:
                row += "| not run | not run "
                continue
            cell = tab[rule][a]
            n = cell["n_partitions"]
            row += f"| {_run_cell(cell['mcnemar'], n)} | {_run_cell(cell['proportion'], n)} "
        for a in APPROACHES:
            m, p = T.TABLE_5_11[rule][a]
            row += f"| {m} | {p} "
        lines.append(row + "|")
    lines += ["", "Success counts behind the tests (mean over the partitions; partitions "
              "in which the approach has more successes than the rule):", "",
              "| Rule | " + " | ".join(f"{label}: {a} mean count / rule mean count / "
                                       f"approach ahead" for a in APPROACHES) + " |",
              "|---" * 3 + "|"]
    for rule in RULES:
        row = f"| {rule} "
        for a in APPROACHES:
            if a not in present:
                row += "| not run "
                continue
            per = tab[rule][a]["per_partition"]
            n = len(per)
            row += (f"| {sum(p['x_approach'] for p in per) / n:.2f} / "
                    f"{sum(p['x_rule'] for p in per) / n:.2f} / "
                    f"{tab[rule][a]['n_approach_ahead']} of {n} ")
        lines.append(row + "|")
    lines += ["", "#### Check of the test conventions on the thesis's own counts (no data "
              "involved)", "",
              "Recomputed with cdm.eval.stats from the published counts (RP 49, AP 42, SP "
              "33, MR 34, WC 34, HAC 22 of 77) and the discordant pairs of plan §5.8 "
              "[INFERRED]; beside the printed values of TH Table 5.11.", "",
              "| Rule | vs RP, McNemar: recomputed (pair) / printed | vs RP, proportion: "
              "recomputed / printed | vs AP, McNemar: recomputed (pair) / printed | vs AP, "
              "proportion: recomputed / printed |", "|---" * 5 + "|"]
    for rule in RULES:
        row = f"| {rule} "
        for a in APPROACHES:
            r = rec[rule][a]
            mc = "; ".join(f"{fmt_p(p)} ({b}, {c})" for p, (b, c) in zip(r["mcnemar"], r["pairs"]))
            row += (f"| {mc} / {T.TABLE_5_11[rule][a][0]} "
                    f"| {fmt_p(r['proportion'])} / {T.TABLE_5_11[rule][a][1]} ")
        lines.append(row + "|")
    lines += ["", "A p-value is \"undefined\" when the two methods never disagree (McNemar, "
              "b + c = 0) or the pooled standard error is 0 (decision B9-1). The "
              "p-values of different partitions are not independent and are not pooled "
              "(decision B9-2)."]
    if synthetic:
        lines += ["", "The run columns are NOT a reproduction of Table 5.11: on the "
                  "synthetic stand-in they only show that the tests run on the pipeline's "
                  "output. The convention check above is independent of any data."]
    return wrap(lines, synthetic)
