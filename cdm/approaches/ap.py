"""AP: from the answers' predictions to the returned answer (plan §4.4;
TH §3.3.2.2 p.40-41, §4.2.4 p.58-59; MCF §3.2.2). Step B7.

    UNTESTED ON REAL DATA.

One instance per answer with at least one vote in the sub-set. The thesis's
rule, "three mutually exclusive cases" (TH p.41, p.59):

  1. "If only one answer is classified as 'True', this answer is selected."
  2. "If multiple answers are classified as 'True', the answer with the
     highest prediction probability is selected."
  3. "If all answers are classified as 'False', the answer with the lowest
     predicted probability of being False is selected."

`three_case` implements this literally (case 3 uses the classifier's
P(False), not 1 - P(True)). Plan §4.4: "In every case this reduces to
argmax P(True) [INFERRED]. Implement it literally anyway and assert the two
agree." The reduction needs the classification to be the arg-max of the
probability ("True" iff P(True) > 0.5): true for the soft-vote ensemble
and for bagging, NOT guaranteed for sklearn's SVC, whose `predict` and
`predict_proba` can disagree. `evaluate(..., assert_equivalence=True)`
raises on a disagreement; with False the disagreements are counted and the
literal rule gives the answer (decision B7-1, B8-1).

Ties: the thesis gives no rule. Two probabilities within `tol` (1e-9, the
AM3 tolerance) are tied; a tie -> canonical order of A_P (as plan OQ27
does for RP). Ties are counted (decision B7-2).
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np

DEFAULT_TOL = 1e-9
CASES = ("one_true", "several_true", "none_true")


def _best(values, positions, largest, tol):
    vals = [values[i] for i in positions]
    ref = max(vals) if largest else min(vals)
    if largest:
        return [i for i in positions if values[i] >= ref - tol]
    return [i for i in positions if values[i] <= ref + tol]


def three_case(label, p_true, p_false, tol=DEFAULT_TOL):
    """The literal rule of TH p.41. Returns (position, case, tied positions).

    label, p_true, p_false: one entry per ACR instance of the sub-set, in
    canonical order of the answers.
    """
    label = np.asarray(label, dtype=int)
    p_true = np.asarray(p_true, dtype=float)
    p_false = np.asarray(p_false, dtype=float)
    if not (label.shape == p_true.shape == p_false.shape) or label.ndim != 1 or label.size == 0:
        raise ValueError("one label, P(True) and P(False) per answer (at least one answer)")
    if not np.isin(label, (0, 1)).all() or not (np.isfinite(p_true).all()
                                                and np.isfinite(p_false).all()):
        raise ValueError("labels must be 0/1 and probabilities finite")
    trues = [int(i) for i in np.flatnonzero(label == 1)]
    if len(trues) == 1:
        return trues[0], "one_true", (trues[0],)
    if len(trues) > 1:
        tied = _best(p_true, trues, True, tol)
        return tied[0], "several_true", tuple(tied)
    tied = _best(p_false, list(range(label.size)), False, tol)
    return tied[0], "none_true", tuple(tied)


def argmax_true(p_true, tol=DEFAULT_TOL):
    """argmax P(True) (plan §4.4). Returns (position, tied positions)."""
    p_true = np.asarray(p_true, dtype=float)
    if p_true.ndim != 1 or p_true.size == 0 or not np.isfinite(p_true).all():
        raise ValueError("finite probabilities of at least one answer are needed")
    tied = _best(p_true, list(range(p_true.size)), True, tol)
    return tied[0], tuple(tied)


@dataclass
class APResult:
    """AP on every sub-set of one partition (arrays over the sub-sets).

    answer  : position in A_P of the returned answer (three-case rule)
    correct : it equals a*
    case    : which of the three cases applied ("unanimous" = one voted
              answer, returned without a classifier)
    tie     : the deciding probability was tied (canonical order decided)
    correct_possible : a* is among the tied answers
    argmax_answer    : the answer of argmax P(True)
    agree   : three-case answer == argmax answer (or tied with it within
              the tolerance)
    """
    answer: np.ndarray
    correct: np.ndarray
    case: Tuple[str, ...]
    tie: np.ndarray
    correct_possible: np.ndarray
    argmax_answer: np.ndarray
    argmax_correct: np.ndarray
    agree: np.ndarray
    n: int
    tol: float

    def count(self):
        return int(self.correct.sum())

    def summary(self):
        c, t, cp = self.correct, self.tie, self.correct_possible
        out = {"correct": int(c.sum()), "n": self.n, "ties": int(t.sum()),
               "low": int((c & ~t).sum()), "high": int((c & ~t).sum() + (cp & t).sum()),
               "argmax_correct": int(self.argmax_correct.sum()),
               "three_case_differs_from_argmax": int((~self.agree).sum())}
        for name in CASES + ("unanimous",):
            out[f"case_{name}"] = sum(1 for k in self.case if k == name)
        return out


def evaluate(inst, pred, tol=DEFAULT_TOL, assert_equivalence=True):
    """Apply the three-case rule to the out-of-group predictions of an AP
    InstanceSet and compare it with argmax P(True)."""
    if inst.approach != "AP":
        raise ValueError("the AP rule needs AP instances")
    G = inst.n_groups
    answer = np.zeros(G, dtype=int)
    arg_answer = np.zeros(G, dtype=int)
    correct = np.zeros(G, dtype=bool)
    arg_correct = np.zeros(G, dtype=bool)
    tie = np.zeros(G, dtype=bool)
    possible = np.zeros(G, dtype=bool)
    agree = np.ones(G, dtype=bool)
    cases = []
    for g in range(G):
        rows = inst.rows_of(g)
        idx = inst.answer_index[rows]
        correct_idx = inst.answers[g].index(inst.correct[g])
        if rows.size == 1:
            answer[g] = arg_answer[g] = idx[0]
            correct[g] = arg_correct[g] = possible[g] = idx[0] == correct_idx
            cases.append("unanimous")
            continue
        if (pred.label[rows] < 0).any():
            raise ValueError(f"sub-set {inst.case_ids[g]!r}: an instance has no prediction")
        pos, case, tied = three_case(pred.label[rows], pred.p_true[rows],
                                     pred.p_false[rows], tol)
        apos, atied = argmax_true(pred.p_true[rows], tol)
        answer[g], arg_answer[g] = idx[pos], idx[apos]
        correct[g] = idx[pos] == correct_idx
        arg_correct[g] = idx[apos] == correct_idx
        tie[g] = len(tied) > 1
        possible[g] = correct_idx in [int(idx[i]) for i in tied]
        # equal up to a tie of P(True) within the tolerance
        agree[g] = pos == apos or pos in atied
        cases.append(case)
        if assert_equivalence and not agree[g]:
            raise AssertionError(
                f"sub-set {inst.case_ids[g]!r}: the three-case rule returns answer "
                f"{pos} and argmax P(True) answer {apos} (plan §4.4 expects them equal); "
                f"labels {pred.label[rows].tolist()}, P(True) {pred.p_true[rows].tolist()}")
    return APResult(answer=answer, correct=correct, case=tuple(cases), tie=tie,
                    correct_possible=possible, argmax_answer=arg_answer,
                    argmax_correct=arg_correct, agree=agree, n=G, tol=float(tol))
