"""Classic meta-learning: a decision tree on the rule answers only.

Step A17 of plan §9.3 ("Classic meta-learning (DecisionTree)" -> TH Table
5.8; plan item T16). Plan §5.5 (TH §4.1.6 p.49-50, §5.1.2 p.71-73).

  * "A basic, stripped-down implementation of the DAP approach, without
    using the CDMCR features" (TH p.49): the instance of a case is the
    answers chosen by the aggregation methods, (f_MR, f_HAC, f_WC, f_SP,
    f_DA) in the thesis order (TH p.38), coded as in DAP (plan §4.2 / OQ5:
    1 = the first answer of the case, 0 = the other); the target is the
    code of a*. Binary cases only (cdm.approaches.dap).
  * Five configurations (TH Table 5.8, rendered p.73; plan §5.5):
      1  f_MR, f_WC + f_HAC, f_SP, f_DA     .713
      2  f_MR, f_SP, f_DA                   .708
      3  f_MR, f_WC + f_HAC, f_DA           .695
      4  f_MR, f_DA                         .687
      5  f_MR, f_WC + f_HAC, f_SP           .713
    When WC/HAC or SP is removed, f_DA is recomputed over the methods kept
    (TH p.50), e.g. row 4: DA = "not MR".
  * Classifier: "a basic decision-tree classifier" (TH p.49) = sklearn
    `DecisionTreeClassifier` with default settings, random_state = 0
    (plan §8 OQ21).
  * Protocol: not stated in the thesis; plan §8 OQ21 default: leave-one-
    out over all cases, as in the ablations. Only that protocol is
    implemented.
  * Decision: the tree's `predict` (the class with the larger share in
    the leaf; an exact 50/50 leaf gives class 0 = the answer coded 0, as
    for DAP, docs/IMPLEMENTATION_DECISIONS.md M7). Such cases are counted (`half_rows`).

Expected (plan §5.5, §9.3 row A17; reported by the driver, not asserted):
row 4 ~ MR; rows 1 and 5 equal (DA is a function of the other four);
best row below SP.

`tree_rules` exports the rules of the tree fitted on ALL cases (TH p.49:
the tree "provides an easy-to-understand set of rules"), as sklearn's
text export and as a table of the answer patterns.

Nothing here reads a data build; the caller passes the instance table.
"""
from itertools import product

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

from cdm.approaches import dap as D
from cdm.approaches import instances as I
from cdm.eval.loo import EvalResult
from cdm.report import thesis_values as T

TECHNIQUE = "DT"
#: Row -> AblationConfig flags (wc_hac, sp, da) of TH Table 5.8; f_MR is
#: always included; no CDMCR feature is used in any row.
ROWS = {row: flags for row, (flags, _) in T.TH_TABLE_5_8.items()}
PROTOCOLS = ("loo",)


def settings_from_config(config=None):
    """Plan §8 OQ21, validated: {"protocol": "loo", "random_state": 0}.
    Any other protocol or model raises (the plan names no alternative)."""
    from cdm.config import get_switch
    oq21 = get_switch("OQ21", config)
    if oq21["protocol"] not in PROTOCOLS:
        raise NotImplementedError(f"OQ21 protocol {oq21['protocol']!r} is not implemented; "
                                  f"only {PROTOCOLS} (plan §5.5)")
    if oq21["model"] != "DecisionTreeClassifier_defaults":
        raise NotImplementedError(f"OQ21 model {oq21['model']!r} is not implemented")
    return {"protocol": oq21["protocol"], "random_state": int(oq21["random_state"])}


def row_config(row):
    """The AblationConfig of a Table 5.8 row: no Confidence / PS feature
    group, the methods of the row."""
    if row not in ROWS:
        raise ValueError(f"unknown meta-learning row {row!r}; TH Table 5.8 has rows 1..5")
    wc_hac, sp, da = (bool(x) for x in ROWS[row])
    return I.AblationConfig(row=f"ML{row}", conf=False, ps=False, wc_hac=wc_hac, sp=sp, da=da)


def thesis_rate(row):
    return T.TH_TABLE_5_8[row][1]


def metalearn_xy(inst, row):
    """(X, y, methods, rules) of a row: X (n, number of methods) = the 0/1
    codes of f_AG in the order MR, HAC, WC, SP, DA (those kept); y = the
    code of a*; rules = the rule-based methods f_DA listens to."""
    bad = D.non_binary_positions(inst)
    if bad:
        raise NotImplementedError(
            f"{len(bad)} case(s) do not have exactly two answers; the answer coding "
            "is defined for binary cases only (plan §4.2, OQ5)")
    cfg = row_config(row)
    methods, rules = I.methods_kept(cfg), I.rules_kept(cfg)
    cols = [[D.encode_answer(a, ans) for a, ans in
             zip(inst.method_answers(m, rules), inst.answers)] for m in methods]
    X = np.array(cols, dtype=float).T.reshape(len(inst), len(methods))
    y = np.array([D.encode_answer(c, ans) for c, ans in zip(inst.correct, inst.answers)],
                 dtype=int)
    return X, y, methods, rules


def make_tree(random_state=0):
    """A fresh `DecisionTreeClassifier` with default settings (OQ21)."""
    return DecisionTreeClassifier(random_state=random_state)


def _predict(tree, x):
    """(predicted code, P(code = 1)) of one row."""
    proba = tree.predict_proba(x)[0]
    classes = list(tree.classes_)
    p1 = float(proba[classes.index(1)]) if 1 in classes else 0.0
    return int(tree.predict(x)[0]), p1


def loo_predictions(X, y, random_state=0):
    """Leave-one-out predictions of the tree: (pred (n,), p1 (n,)). Every
    case is a real refit on the other n - 1 cases."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    n = len(y)
    if n < 2:
        raise ValueError("leave-one-out needs at least 2 cases")
    pred = np.empty(n, dtype=int)
    p1 = np.empty(n)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        keep[i] = False
        tree = make_tree(random_state).fit(X[keep], y[keep])
        pred[i], p1[i] = _predict(tree, X[i:i + 1])
        keep[i] = True
    return pred, p1


def run_row(inst, row, random_state=None, config=None):
    """LOO of one Table 5.8 row. Returns an EvalResult (approach "DAP",
    protocol "loo", technique "DT") whose `outcomes` are the O_AG of the
    methods of the row (f_DA recomputed), so that the result can be
    compared with the rules of the row."""
    settings = settings_from_config(config)
    rs = settings["random_state"] if random_state is None else int(random_state)
    X, y, methods, rules = metalearn_xy(inst, row)
    pred, p1 = loo_predictions(X, y, rs)
    return EvalResult(
        approach="DAP", protocol=settings["protocol"], techniques=[TECHNIQUE] * len(y),
        case_ids=list(inst.case_ids), domains=list(inst.domains),
        success=(pred == y).astype(int), labels=tuple(methods),
        outcomes=np.column_stack([inst.outcomes(m, rules) for m in methods]).astype(int),
        proba=p1, predicted=pred, target=y,
        answers=[D.decode_answer(int(c), a) for c, a in zip(pred, inst.answers)],
        meta={"row": row, "methods": list(methods), "rules_heard_by_DA": list(rules),
              "random_state": rs, "model": "DecisionTreeClassifier (sklearn defaults)",
              "features": ["f_" + m for m in methods]})


def run_all(inst, rows=None, random_state=None, config=None):
    """{row: EvalResult} for the rows of TH Table 5.8 (default: all)."""
    return {row: run_row(inst, row, random_state, config)
            for row in (sorted(ROWS) if rows is None else rows)}


def tree_rules(inst, row, random_state=None, config=None):
    """The rules of the tree of a row fitted on ALL cases.

    Returns {"text": sklearn export_text of the tree (features f_<method>,
    value 1 = the method chose the first answer a1 of the case; class 1 =
    "a* is a1"), "patterns": one dict per combination of method answers
    that occurs: the codes, n cases, n with a* = a1, the tree's prediction
    and its P(a1), "depth", "n_leaves"}.
    """
    settings = settings_from_config(config)
    rs = settings["random_state"] if random_state is None else int(random_state)
    X, y, methods, _ = metalearn_xy(inst, row)
    tree = make_tree(rs).fit(X, y)
    names = ["f_" + m for m in methods]
    patterns = []
    for combo in product((1, 0), repeat=len(methods)):
        mask = np.all(X == np.array(combo, dtype=float), axis=1)
        if not mask.any():
            continue
        pred, p1 = _predict(tree, np.array([combo], dtype=float))
        patterns.append({"codes": dict(zip(names, combo)), "n": int(mask.sum()),
                         "n_correct_is_a1": int(y[mask].sum()),
                         "predicted": "a1" if pred == 1 else "a2", "p_a1": p1,
                         "agrees_with": [m for m, c in zip(methods, combo)
                                         if (c == 1) == (pred == 1)]})
    return {"text": export_text(tree, feature_names=names, show_weights=True),
            "patterns": patterns, "depth": int(tree.get_depth()),
            "n_leaves": int(tree.get_n_leaves()), "features": names}


def format_patterns(rules):
    """Text table of `tree_rules(...)["patterns"]`: one line per answer
    pattern (a1 / a2 per method), the number of cases, how often a* = a1,
    and what the tree returns."""
    names = rules["features"]
    lines = ["  ".join(f"{n:<6}" for n in names) + "  cases  a*=a1  tree returns  "
             "(= the answer of)"]
    for p in rules["patterns"]:
        lines.append("  ".join(f"{'a1' if p['codes'][n] else 'a2':<6}" for n in names)
                     + f"  {p['n']:>5}  {p['n_correct_is_a1']:>5}  {p['predicted']:<12}  "
                     + (", ".join(p["agrees_with"]) or "no method"))
    return "\n".join(lines)


def expectations(results, inst):
    """The expected relations of plan §5.5 / §9.3 row A17, evaluated on a
    set of row results (nothing is asserted):
      row4_vs_MR        : successes of row 4 and of MR applied uniformly;
      rows_1_5_equal    : rows 1 and 5 have the same predictions;
      best_vs_SP        : the best row's successes and SP's.
    Entries whose rows are missing are None."""
    out = {"row4_vs_MR": None, "rows_1_5_equal": None, "best_vs_SP": None}
    mr, sp = int(inst.outcomes("MR").sum()), int(inst.outcomes("SP").sum())
    if 4 in results:
        k = results[4].n_success
        out["row4_vs_MR"] = {"row4": k, "MR": mr, "equal": k == mr}
    if 1 in results and 5 in results:
        out["rows_1_5_equal"] = {
            "row1": results[1].n_success, "row5": results[5].n_success,
            "same_predictions": bool(np.array_equal(results[1].predicted,
                                                    results[5].predicted))}
    if results:
        best = max(results, key=lambda r: results[r].n_success)
        out["best_vs_SP"] = {"best_row": best, "best": results[best].n_success, "SP": sp,
                             "best_below_SP": results[best].n_success < sp}
    return out
