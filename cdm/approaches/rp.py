"""RP: the classification-aggregation strategies (plan §4.3, §8 OQ27;
TH §3.3.2.1 p.39-40, §4.2.4 p.56-58, Table 4.2, Table A.7; MCF §3.2.1,
Table 4). Step B6.

    UNTESTED ON REAL DATA.

Input per sub-set: for every response its classification ("True"/"False")
and its prediction probability ("the probability of the classified instance
being True", TH p.39). The five strategies (TH p.39-40), computed for every
answer over its supporters in the sub-set:

  maj   number of supporters classified "True"
  prop  "highest proportion of 'True' versus 'False' classified supporters"
        = #True / #supporters (plan OQ27). The other reading, the ratio
        #True / #False, orders the answers identically and has the same
        ties, so the choice cannot change a result.
  wm    sum of the supporters' prediction probabilities
  avgp  average prediction probability of the supporters
  maxp  maximum prediction probability of the supporters

"for the methods maj and prop, we use one of the other three methods (wm,
avgp and maxp) as a tie-breaker, if needed ... nine classification-
aggregation strategies in total" (TH p.40). Final choice of the thesis:
prop with avgp as tie-breaker (TH p.57).

Only answers with at least one supporter are candidates (implicit: an
answer without a supporter has no instance; plan §4.3).

Ties (the thesis gives no rule beyond the tie-breaker):
  * "Percentage of Tie Breakers" (Table 4.2) = share of the sub-sets in
    which the PRIMARY criterion is tied between two or more answers (plan
    OQ27). If no supporter is classified "True", maj and prop are 0 for
    every answer, which is a tie. [INFERRED support: the 91% of the SVM
    rows of Table A.7 fit a classifier that calls almost every response
    "False".]
  * maj / prop: a tie that remains after the tie-breaker -> canonical order
    of A_P (plan OQ27).
  * maxp / avgp / wm alone: TH Table A.7 prints the tie-breaker "rnd" for
    these three rows; plan §5.7 reads it as random tie-breaking. A seeded
    random choice among the tied answers is the default here
    (`standalone_ties="random"`); `"canonical_first"` is the switch
    (decision B6-6).
  * float scores (prop, wm, avgp, maxp) are tied if they differ by at most
    `tol` (the AM3 tolerance, 1e-9); maj is an integer.
Every decision also records whether the correct answer is among the finally
tied answers, so that the range of the success count over every resolution
of the remaining ties can be reported (as for the rules in step B5).
"""
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

CRITERIA = ("maj", "prop", "wm", "avgp", "maxp")
TIE_BREAKERS = ("avgp", "maxp", "wm")
#: The nine strategies in the row order of TH Table 4.2: (primary, tie-breaker).
STRATEGIES = (
    ("maxp", None), ("avgp", None), ("wm", None),
    ("maj", "avgp"), ("maj", "maxp"), ("maj", "wm"),
    ("prop", "avgp"), ("prop", "maxp"), ("prop", "wm"),
)
#: TH p.57: "we chose the prop strategy, with avgp as a tie-breaker".
FINAL_STRATEGY = ("prop", "avgp")
STANDALONE_TIE_RULES = ("random", "canonical_first")
DEFAULT_TOL = 1e-9


def strategy_name(strategy):
    primary, breaker = strategy
    return primary if breaker is None else f"{primary}+{breaker}"


STRATEGY_NAMES = tuple(strategy_name(s) for s in STRATEGIES)
FINAL_NAME = strategy_name(FINAL_STRATEGY)


def answer_scores(answer_index, label, p_true):
    """The five criteria of every answer with at least one supporter.

    answer_index : per response of the sub-set, the position of its answer
                   in A_P
    label        : per response, 1 = classified "True", 0 = "False"
    p_true       : per response, the predicted probability of "True"

    Returns (candidates, scores): candidates = the voted answers' positions
    in ascending (canonical) order; scores = {criterion: list aligned with
    candidates}.
    """
    answer_index = np.asarray(answer_index, dtype=int)
    label = np.asarray(label, dtype=int)
    p_true = np.asarray(p_true, dtype=float)
    if not (answer_index.shape == label.shape == p_true.shape) or answer_index.ndim != 1:
        raise ValueError("one answer, label and probability per response")
    if answer_index.size == 0:
        raise ValueError("a sub-set without responses has no answer")
    if not np.isin(label, (0, 1)).all() or not np.isfinite(p_true).all():
        raise ValueError("labels must be 0/1 and probabilities finite")
    candidates = sorted(set(int(a) for a in answer_index))
    scores = {c: [] for c in CRITERIA}
    for a in candidates:
        sup = answer_index == a
        k = int(label[sup].sum())
        n = int(sup.sum())
        p = p_true[sup]
        scores["maj"].append(k)
        scores["prop"].append(k / n)
        scores["wm"].append(float(p.sum()))
        scores["avgp"].append(float(p.sum()) / n)
        scores["maxp"].append(float(p.max()))
    return candidates, scores


def _top(values, positions, criterion, tol):
    """Positions (into the candidates) with the best value of a criterion."""
    best = max(values[i] for i in positions)
    if criterion == "maj":
        return [i for i in positions if values[i] == best]
    return [i for i in positions if values[i] >= best - tol]


@dataclass(frozen=True)
class Decision:
    """The answer of one strategy on one sub-set.

    answer        : position in A_P of the returned answer
    primary_tied  : candidates tied on the primary criterion (len > 1 = a
                    "tie breaker" case of Table 4.2)
    final_tied    : candidates still tied after the tie-breaker (for a
                    standalone strategy = primary_tied); len > 1 means the
                    answer was picked by `resolved_by`
    resolved_by   : "primary", the tie-breaker's name, "canonical_first" or
                    "random"
    """
    answer: int
    primary_tied: Tuple[int, ...]
    final_tied: Tuple[int, ...]
    resolved_by: str


def decide(candidates, scores, strategy, tol=DEFAULT_TOL,
           standalone_ties="random", rng=None):
    """Apply one of the nine strategies to the scores of a sub-set."""
    primary, breaker = strategy
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    if standalone_ties not in STANDALONE_TIE_RULES:
        raise ValueError(f"standalone_ties must be one of {STANDALONE_TIE_RULES}")
    pos = list(range(len(candidates)))
    tied = _top(scores[primary], pos, primary, tol)
    primary_tied = tuple(candidates[i] for i in tied)
    if len(tied) == 1:
        return Decision(candidates[tied[0]], primary_tied, primary_tied, "primary")
    if breaker is None:
        # TH Table A.7: tie-breaker "rnd" for maxp / avgp / wm
        if standalone_ties == "random":
            if rng is None:
                raise ValueError("a random generator is needed for a random tie")
            pick = tied[int(rng.integers(len(tied)))]
            return Decision(candidates[pick], primary_tied, primary_tied, "random")
        return Decision(candidates[tied[0]], primary_tied, primary_tied, "canonical_first")
    tied2 = _top(scores[breaker], tied, breaker, tol)
    final = tuple(candidates[i] for i in tied2)
    if len(tied2) == 1:
        return Decision(candidates[tied2[0]], primary_tied, final, breaker)
    # plan OQ27: "a further tie -> canonical order"
    return Decision(candidates[tied2[0]], primary_tied, final, "canonical_first")


def tie_rng(seed, case_id, name):
    """Generator of the random tie-break of one sub-set and strategy:
    first 8 bytes of SHA-256("<seed>|rp_rnd_tie|<case_id>|<strategy>")
    (the pattern of decision B3-11 / L12)."""
    digest = hashlib.sha256(f"{int(seed)}|rp_rnd_tie|{case_id}|{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


@dataclass
class RPResult:
    """The nine strategies on every sub-set of one partition.

    Per strategy name, arrays over the sub-sets:
      answer (position in A_P), correct, primary_tie, final_tie,
      correct_possible (the correct answer is among the finally tied ones)
    unanimous : sub-sets with one voted answer (returned without a
                classifier; decision B6-4)
    """
    names: Tuple[str, ...]
    answer: dict
    correct: dict
    primary_tie: dict
    final_tie: dict
    correct_possible: dict
    unanimous: np.ndarray
    n: int
    tol: float
    standalone_ties: str
    tie_seed: Optional[int]

    def count(self, name):
        return int(self.correct[name].sum())

    def summary(self):
        out = {}
        for name in self.names:
            c, ft, cp = self.correct[name], self.final_tie[name], self.correct_possible[name]
            out[name] = {
                "correct": int(c.sum()), "n": self.n,
                "primary_ties": int(self.primary_tie[name].sum()),
                "final_ties": int(ft.sum()),
                # count if every remaining tie went wrong / right
                "low": int((c & ~ft).sum()),
                "high": int((c & ~ft).sum() + (cp & ft).sum()),
            }
        return out


def evaluate(inst, pred, tol=DEFAULT_TOL, standalone_ties="random", tie_seed=0):
    """Apply the nine strategies to the out-of-group predictions of an RP
    InstanceSet (cdm.eval.worldB_logo.logo_predict).

    A sub-set with a single voted answer returns that answer under every
    strategy (it is the only candidate); its instances have no prediction
    when their out-group is empty.
    """
    if inst.approach != "RP":
        raise ValueError("RP strategies need RP instances")
    G = inst.n_groups
    res = {k: {name: np.zeros(G, dtype=t) for name in STRATEGY_NAMES}
           for k, t in (("answer", int), ("correct", bool), ("primary_tie", bool),
                        ("final_tie", bool), ("correct_possible", bool))}
    unanimous = np.zeros(G, dtype=bool)
    for g in range(G):
        rows = inst.rows_of(g)
        answers = inst.answers[g]
        correct_idx = answers.index(inst.correct[g])
        voted = sorted(set(int(a) for a in inst.answer_index[rows]))
        if len(voted) == 1:
            unanimous[g] = True
            for name in STRATEGY_NAMES:
                res["answer"][name][g] = voted[0]
                res["correct"][name][g] = voted[0] == correct_idx
                res["correct_possible"][name][g] = voted[0] == correct_idx
            continue
        if (pred.label[rows] < 0).any():
            raise ValueError(f"sub-set {inst.case_ids[g]!r}: an instance has no prediction")
        candidates, scores = answer_scores(inst.answer_index[rows], pred.label[rows],
                                           pred.p_true[rows])
        for strategy, name in zip(STRATEGIES, STRATEGY_NAMES):
            rng = None
            if strategy[1] is None and standalone_ties == "random":
                rng = tie_rng(tie_seed, inst.case_ids[g], name)
            d = decide(candidates, scores, strategy, tol, standalone_ties, rng)
            res["answer"][name][g] = d.answer
            res["correct"][name][g] = d.answer == correct_idx
            res["primary_tie"][name][g] = len(d.primary_tied) > 1
            res["final_tie"][name][g] = len(d.final_tied) > 1
            res["correct_possible"][name][g] = correct_idx in d.final_tied
    return RPResult(names=STRATEGY_NAMES, answer=res["answer"], correct=res["correct"],
                    primary_tie=res["primary_tie"], final_tie=res["final_tie"],
                    correct_possible=res["correct_possible"], unanimous=unanimous,
                    n=G, tol=float(tol), standalone_ties=standalone_ties,
                    tie_seed=None if standalone_ties != "random" else int(tie_seed))
