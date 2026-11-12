"""Effectiveness of the AMP selection, globally and per domain.

Step A15 of plan §9.3 ("AMP effectiveness + per-domain (10-fold and LOO)"
-> TH Tables 5.4, 5.5, A.9, A.10; plan items T8, T10-T13, T20, T21).

For every method AG_i and every block of cases (all cases, or one domain;
plan §8 OQ40: the per-domain tables slice the global predictions by
domain, TH p.69):

  P(AMP selected AG_i)            = n_selected / n
  P(success by AG_i)              = n_success / n        (the rule applied
                                    uniformly to the block)
  P(success by AG_i | selected)   = n_selected_success / n_selected
  PRE = (E1 - E2) / E1, E1 = 1 - P(success), E2 = 1 - P(success | selected)
        (TH p.65; plan §5.1), from the EXACT counts (plan §5.8 item 3)
  p   = one-sample z-test of P(success | selected), n = n_selected, against
        the fixed p0 = P(success), one-tailed (plan §5.8 item 3). The exact
        upper tail is reported and 1 - Phi(z) in double precision beside it
        (amendment AM1: the printed cells are of the second form).

The thesis prints p for the global 10-fold table only (Table 5.4); MLA
Fig 2 prints it for the global LOO. Tables 5.5 and A.10 have no p row; the
per-domain p-values computed here (p0 = the rule's success in the domain)
have no printed counterpart.

Rounding: every displayed rate is the exact ratio rounded half UP to three
decimals (cdm.eval.stats.round_half_up; the thesis rounds half up, TH
Table 5.5 Politics SP 13/16 = 0.813). PRE is rounded from its exact
fraction.

Undefined cells are None: P(success | selected), PRE and p when a method
was never selected in the block; PRE when the rule solves every case of
the block (E1 = 0); p when the rule's P(success) is 0 or 1.

Nothing here fits a model or reads a data build: the input is an
`cdm.eval.loo.EvalResult` (or the per-case CSV a driver wrote from one).
"""
import csv
from fractions import Fraction

import numpy as np

from cdm.eval import stats as S
from cdm.eval.loo import EvalResult
from cdm.report import thesis_values as T

#: Column order of TH Tables 5.4 / 5.5 / A.10.
METHODS = T.METHODS
ALL = "all"
RATE_KEYS = ("p_selected", "p_success", "p_success_given_selected", "pre")
#: Plan §9.3 row A15: "+-0.03 on rates".
RATE_TOLERANCE = 0.03
PROTOCOLS = ("nested_cv", "loo")


# ---- one cell block -------------------------------------------------------------

def effectiveness_row(n, n_selected, n_success, n_selected_success):
    """The statistics of one method in one block, from its four counts.

    Returns a dict with the counts, the float values `p_selected`,
    `p_success`, `p_success_given_selected`, `pre`, `z`, `p` (exact upper
    tail), `p_one_minus_cdf` (1 - Phi(z), amendment AM1), and `rounded`:
    the first four rounded half up to 3 decimals from the exact fractions.
    Undefined values are None (see the module docstring).
    """
    n, sel, tot, ok = int(n), int(n_selected), int(n_success), int(n_selected_success)
    if n <= 0 or not 0 <= sel <= n or not 0 <= tot <= n or not 0 <= ok <= min(sel, tot):
        raise ValueError(f"inconsistent counts: n={n}, selected={sel}, "
                         f"successes={tot}, correct when selected={ok}")
    exact = {"p_selected": Fraction(sel, n), "p_success": Fraction(tot, n),
             "p_success_given_selected": Fraction(ok, sel) if sel else None,
             "pre": None}
    if sel and tot < n:
        # (E1 - E2) / E1 with exact fractions (plan §5.8 item 3)
        e1, e2 = 1 - exact["p_success"], 1 - exact["p_success_given_selected"]
        exact["pre"] = (e1 - e2) / e1
    row = {"n": n, "n_selected": sel, "n_success": tot, "n_selected_success": ok,
           "z": None, "p": None, "p_one_minus_cdf": None}
    for k in RATE_KEYS:
        row[k] = None if exact[k] is None else float(exact[k])
    row["rounded"] = {k: None if exact[k] is None else S.round_half_up(exact[k])
                      for k in RATE_KEYS}
    if sel and 0 < tot < n:
        row["z"], row["p"] = S.one_sample_proportion_ztest(ok, sel, tot / n)
        row["p_one_minus_cdf"] = S.one_sample_proportion_ztest(
            ok, sel, tot / n, tail="one_minus_cdf")[1]
    return row


def block_rows(n, selected, success, selected_success, methods=METHODS):
    """{method: effectiveness_row} of one block from three count dicts."""
    return {m: effectiveness_row(n, selected[m], success[m], selected_success[m])
            for m in methods}


# ---- from an evaluation result ------------------------------------------------------

def _check_amp(res):
    if res.approach != "AMP":
        raise ValueError("the effectiveness tables are defined for AMP only")
    if res.outcomes is None or res.selected is None:
        raise ValueError("the result holds no method outcomes / selections")


def effectiveness_table(res, methods=None):
    """{block: {method: effectiveness_row}} of an AMP EvalResult; blocks
    are "all" and, when the result stores domains, every domain in data
    order. `methods`: the display order (default: SP, MR, WC, HAC, DA,
    restricted to the labels of the result)."""
    _check_amp(res)
    methods = tuple(methods) if methods is not None else tuple(
        m for m in METHODS if m in res.labels)
    blocks = {ALL: res}
    if res.domains is not None:
        blocks.update(res.by_domain())
    out = {}
    for name, r in blocks.items():
        out[name] = block_rows(r.n, r.selection_counts(), r.method_success_counts(),
                               r.selected_success_counts(), methods)
    return out


def success_by_domain(res):
    """{domain: {n, successes, rate, rounded}} plus "all" (TH Table A.9;
    plan T10, T12), for AMP or DAP."""
    out = {}
    for d, row in res.domain_table().items():
        row = dict(row)
        row["rounded"] = S.round_half_up(row["successes"], row["n"])
        out[d] = row
    return out


def rule_baselines(outcomes, domains=None, with_thesis=True, methods=METHODS):
    """Correct cases of every rule-based method on the data of a run, next
    to the thesis's counts, globally and per domain (the rule baselines
    recomputed vs printed; docs/RULE_BASELINE_CHECK.md).

    outcomes : {method: (n,) 0/1 O_AG}; methods not in it are skipped.
    domains  : per-case domain names, or None (global block only).
    Returns {block: {method: {recomputed, n, thesis, thesis_n}}}; `thesis`
    is None when `with_thesis` is false or the block is not a thesis
    domain (T3 globally, T3b per domain).
    """
    methods = [m for m in methods if m in outcomes]
    cols = {m: np.asarray(outcomes[m]).astype(int) for m in methods}
    n = len(next(iter(cols.values())))
    masks = {ALL: np.ones(n, dtype=bool)}
    if domains is not None:
        d = np.array(list(domains), dtype=object)
        for name in dict.fromkeys(domains):
            masks[name] = d == name
    out = {}
    for block, mask in masks.items():
        if block == ALL:
            th, th_n = T.RULES_THESIS, T.N_CASES
        else:
            th, th_n = T.RULES_THESIS_DOMAIN.get(block), T.DOMAIN_N.get(block)
        out[block] = {m: {"recomputed": int(cols[m][mask].sum()), "n": int(mask.sum()),
                          "thesis": th[m] if with_thesis and th else None,
                          "thesis_n": th_n if with_thesis and th else None}
                      for m in methods}
    return out


def rule_baselines_of_result(res, with_thesis=True):
    """`rule_baselines` of the method outcomes stored in an EvalResult."""
    if res.outcomes is None:
        raise ValueError("the result holds no method outcomes")
    return rule_baselines({m: res.outcomes[:, j] for j, m in enumerate(res.labels)},
                          res.domains, with_thesis)


# ---- the thesis side ------------------------------------------------------------------

def _printed(table):
    """{row key: {method: printed value}} from a table of thesis-reported values."""
    return {k: dict(zip(METHODS, v)) for k, v in table.items()}


def thesis_effectiveness(protocol):
    """The thesis tables in the same shape as `effectiveness_table`, built
    from the plan's back-solved counts (T3, T3b, T8, T11, T13, T20, T21),
    with the printed cells attached as row["printed"] where the thesis
    (or MLA, for the global LOO) prints them.

    protocol "nested_cv": global = TH Table 5.4, domains = TH Table 5.5.
    protocol "loo"      : global = MLA Figs 2-3 (the thesis has no global
                          LOO table), domains = TH Table A.10.
    Returns (table, sources) with sources = {block: text}.
    """
    if protocol == "nested_cv":
        g_sel, g_ok = T.NESTED_AMP_SELECTED, T.NESTED_AMP_SELECTED_SUCCESS
        d_sel, d_ok = T.NESTED_AMP_DOMAIN_SELECTED, T.NESTED_AMP_DOMAIN_SELECTED_SUCCESS
        g_print, d_print = _printed(T.TH_TABLE_5_4), T.TH_TABLE_5_5
        src_g, src_d = "TH Table 5.4 p.66 (counts: plan T8)", "TH Table 5.5 p.69 (counts: plan T11)"
    elif protocol == "loo":
        g_sel, g_ok = T.LOO_AMP_SELECTED, T.LOO_AMP_SELECTED_SUCCESS
        d_sel, d_ok = T.LOO_AMP_DOMAIN_SELECTED, T.LOO_AMP_DOMAIN_SELECTED_SUCCESS
        g_print = {k: dict(zip(METHODS, T.MLA_LOO_PRINTED[k]))
                   for k in ("p_success_given_selected", "p")}
        d_print = T.TH_TABLE_A_10
        src_g = "MLA Figs 2-3 (counts: plan T20/T21; no global LOO table in the thesis)"
        src_d = "TH Table A.10 p.115 (counts: plan T13)"
    else:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    table = {ALL: block_rows(T.N_CASES, g_sel, T.RULES_THESIS, g_ok)}
    sources = {ALL: src_g}
    for m, row in table[ALL].items():
        row["printed"] = {k: v[m] for k, v in g_print.items()}
    for d in T.DOMAINS:
        table[d] = block_rows(T.DOMAIN_N[d], d_sel[d], T.RULES_THESIS_DOMAIN[d], d_ok[d])
        pr = _printed(d_print[d])
        for m, row in table[d].items():
            row["printed"] = {k: v[m] for k, v in pr.items()}
        sources[d] = src_d
    return table, sources


def thesis_success_by_domain(approach, protocol):
    """{domain: {n, successes, rate, rounded, printed}} of the thesis, or
    None where the thesis prints nothing (DAP per domain under the
    10-fold protocol). LOO: TH Table A.9 / plan T12; 10-fold AMP: plan T10."""
    if protocol == "loo":
        counts, printed = T.LOO_DOMAIN_SUCCESSES[approach], T.TH_TABLE_A_9[approach]
        total = T.LOO_SUCCESSES[approach]
    elif protocol == "nested_cv":
        total = T.NESTED_SUCCESSES[approach]
        if approach != "AMP":
            return {ALL: {"n": T.N_CASES, "successes": total, "rate": total / T.N_CASES,
                          "rounded": S.round_half_up(total, T.N_CASES), "printed": None}}
        counts, printed = T.NESTED_AMP_DOMAIN_SUCCESSES, T.TH_NESTED_AMP_DOMAIN_RATES
    else:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    out = {d: {"n": T.DOMAIN_N[d], "successes": counts[d],
               "rate": counts[d] / T.DOMAIN_N[d],
               "rounded": S.round_half_up(counts[d], T.DOMAIN_N[d]),
               "printed": printed[d]} for d in T.DOMAINS}
    out[ALL] = {"n": T.N_CASES, "successes": total, "rate": total / T.N_CASES,
                "rounded": S.round_half_up(total, T.N_CASES), "printed": None}
    return out


# ---- comparison -------------------------------------------------------------------------

def _diff(a, b):
    return None if a is None or b is None else a - b


def compare(reproduced, thesis):
    """Cell-by-cell comparison of two effectiveness tables (same blocks and
    methods are compared; a block or method missing on either side is
    skipped). Returns a list of dicts: block, method, quantity, reproduced,
    thesis, difference, and for the three rates `within_tolerance`
    (|difference| <= 0.03, plan §9.3 row A15). Unrounded values are
    compared."""
    rows = []
    for block, methods in reproduced.items():
        if block not in thesis:
            continue
        for m, r in methods.items():
            t = thesis[block].get(m)
            if t is None:
                continue
            for q in ("n", "n_selected", "n_success", "n_selected_success") + RATE_KEYS \
                    + ("p", "p_one_minus_cdf"):
                d = _diff(r[q], t[q])
                row = {"block": block, "method": m, "quantity": q,
                       "recomputed": r[q], "thesis": t[q], "difference": d}
                if q in RATE_KEYS[:3]:
                    row["within_tolerance"] = None if d is None else abs(d) <= RATE_TOLERANCE
                rows.append(row)
    return rows


def qualitative_checks(table, alpha=0.05):
    """The qualitative statements of plan §9.3 row A15, evaluated on one
    effectiveness table (the reproduced one or the thesis one):

      non_significant_methods : methods of the global block whose p is
          >= alpha or undefined (thesis 10-fold: HAC only, TH p.67).
      domain_mean_pre         : {domain: mean PRE over its defined cells}.
      strongest_pre_domain    : the domain with the highest mean PRE
          (thesis: NBA, TH p.70).
      negative_pre            : {domain: methods with PRE < 0}
          (thesis 10-fold NFL: SP, MR, HAC, DA).
    Nothing is asserted; the driver prints both sides.
    """
    out = {"non_significant_methods": [m for m, r in table[ALL].items()
                                       if r["p"] is None or r["p"] >= alpha]}
    means, neg = {}, {}
    for block, methods in table.items():
        if block == ALL:
            continue
        pres = [r["pre"] for r in methods.values() if r["pre"] is not None]
        means[block] = float(np.mean(pres)) if pres else None
        neg[block] = [m for m, r in methods.items() if r["pre"] is not None and r["pre"] < 0]
    out["domain_mean_pre"] = means
    known = {d: v for d, v in means.items() if v is not None}
    out["strongest_pre_domain"] = max(known, key=known.get) if known else None
    out["negative_pre"] = neg
    return out


# ---- reading a driver's per-case file ---------------------------------------------------

def eval_result_from_cases_csv(path, approach, protocol):
    """Rebuild an EvalResult from the per-case CSV written by
    scripts/_common.write_cases_csv (A13: `*_loo_cases.csv`, A14:
    `*_nested_cases.csv`). No model is fitted.

    AMP: `selected` is the method name stored in the file (the selection
    made by the run, plan §8 OQ17); the P_<method> and O_<method> columns
    give the label order. DAP: predicted / target codes, P(code = 1) and
    the O_<method> columns of the rules.
    Raises ValueError when the file does not have the expected columns or
    when its `success` column contradicts the other columns.
    """
    if approach not in ("AMP", "DAP"):
        raise ValueError(f"unknown approach {approach!r}")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} holds no cases")
    head = list(rows[0])
    labels = tuple(c[2:] for c in head if c.startswith("O_"))
    need = ["case_id", "domain", "success"] + (
        ["selected"] if approach == "AMP" else ["predicted", "target", "p_code1"])
    missing = [c for c in need if c not in head]
    if missing or (approach == "AMP" and not labels):
        raise ValueError(f"{path} is not a per-case file of {approach}: "
                         f"missing columns {missing or ['O_<method>']}")
    success = np.array([int(r["success"]) for r in rows])
    outcomes = (np.array([[int(r["O_" + m]) for m in labels] for r in rows])
                if labels else None)
    folds = np.array([int(r["fold"]) for r in rows]) if "fold" in head else None
    techniques = [r["technique"] for r in rows] if "technique" in head else None
    domains = [r["domain"] for r in rows]
    if all(d == "" for d in domains):
        domains = None
    kw = dict(approach=approach, protocol=protocol, case_ids=[r["case_id"] for r in rows],
              domains=domains, success=success, labels=labels, outcomes=outcomes,
              folds=folds, meta={"source_file": str(path)})
    if approach == "AMP":
        if any(r["selected"] not in labels for r in rows):
            raise ValueError(f"{path}: a selected method is not one of {labels}")
        selected = np.array([labels.index(r["selected"]) for r in rows])
        if not np.array_equal(success, outcomes[np.arange(len(rows)), selected]):
            raise ValueError(f"{path}: `success` is not the outcome of the selected method")
        proba = np.array([[float(r["P_" + m]) for m in labels] for r in rows])
        return EvalResult(techniques=techniques or ["BR+RF"] * len(rows),
                          selected=selected, proba=proba,
                          answers=[r.get("answer", "") for r in rows], **kw)
    predicted = np.array([int(r["predicted"]) for r in rows])
    target = np.array([int(r["target"]) for r in rows])
    if not np.array_equal(success, (predicted == target).astype(int)):
        raise ValueError(f"{path}: `success` is not (predicted == target)")
    return EvalResult(techniques=techniques or ["RF"] * len(rows), predicted=predicted,
                      target=target, proba=np.array([float(r["p_code1"]) for r in rows]),
                      answers=[r.get("answer", "") for r in rows], **kw)
