"""RP / AP instances of the World-B sub-sets (plan §4.3, §4.4; steps B6, B7).

    UNTESTED ON REAL DATA (the MTurk file has not been supplied; see
    cdm/data/worldB/loader.py).

RP (TH §3.3.2.1 p.39; §4.2.2 p.55; MCF §3.2.1): "each response ... is
transformed into one instance" with the 14 RCR features of TH Table 3.3,
computed within its sub-set (plan §4.3), and the binary target "whether the
response is correct" (r.v = a*).

AP (TH §3.3.2.2 p.40-41; §4.2.2 p.56; MCF §3.2.2): "each possible answer
... which had at least one vote in R_Pj, is transformed into one instance"
with the 10 ACR features of TH Table 3.4 and the binary target "whether
the answer is correct" (a = a*).

"Each 'group' is the set of instances generated from a single set of
responses" (TH p.58; MCF §4.4): `group` holds the sub-set of every
instance. `task` holds the sub-set's task (used only by the optional
leave-one-task-out sensitivity run, which is not in the papers).

Empty out-group (decision B1-2 / B2-1): in a sub-set where every response
gives the same answer the out-group features are undefined (NaN). Such
instances are flagged in `undefined`; what the evaluation does with them is
decided in cdm/eval/worldB_logo.py (decision B6-4).
"""
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from cdm.features import acr, rcr

APPROACHES = ("RP", "AP")
FEATURE_NAMES = {"RP": rcr.FEATURE_NAMES, "AP": acr.FEATURE_NAMES}


@dataclass(frozen=True)
class InstanceSet:
    """The instances of one partition.

    approach : "RP" or "AP"
    X        : (n, d) features, columns in the order of `feature_names`
    y        : (n,) 1 = the response / answer is correct
    group    : (n,) position of the instance's sub-set in `case_ids`
    answer_index : (n,) position, in the sub-set's A_P, of the answer the
               response voted for (RP) / the instance stands for (AP)
    member   : (n,) RP: position of the response within its sub-set;
               AP: equal to answer_index
    undefined: (n,) True if a feature of the instance is NaN (empty
               out-group)
    case_ids, task_ids, answers, correct : per sub-set
    info     : counts (majority ties, undefined instances, sub-sets whose
               correct answer has no vote)
    """
    approach: str
    X: np.ndarray
    y: np.ndarray
    group: np.ndarray
    answer_index: np.ndarray
    member: np.ndarray
    undefined: np.ndarray
    feature_names: Tuple[str, ...]
    case_ids: Tuple[str, ...]
    task_ids: Tuple[str, ...]
    answers: Tuple[tuple, ...]
    correct: Tuple[object, ...]
    synthetic: bool
    marker: str
    info: dict = field(default_factory=dict, compare=False)

    @property
    def n(self):
        return len(self.y)

    @property
    def n_groups(self):
        return len(self.case_ids)

    def rows_of(self, g):
        """Positions of the instances of sub-set g, in instance order."""
        return np.flatnonzero(self.group == g)


def build_instances(subsets, approach, majority_rule=rcr.DEFAULT_MAJORITY_RULE):
    """The RP or AP instances of every sub-set of a WorldBSubsets.

    RP: one instance per response, in the order of the sub-set's responses.
    AP: one instance per answer with at least one vote, canonical order.
    majority_rule : plan §8 OQ28 (cdm.features.rcr.majority_answers).
    """
    if approach not in APPROACHES:
        raise ValueError(f"approach must be one of {APPROACHES}")
    names = FEATURE_NAMES[approach]
    X, y, group, a_idx, member = [], [], [], [], []
    majority_ties = 0
    no_vote_for_correct = 0
    unanimous = 0
    for g, case in enumerate(subsets.cases):
        index = {a: i for i, a in enumerate(case.answers)}
        voted = acr.voted_answers(case.responses, case.answers)
        no_vote_for_correct += case.correct not in voted
        unanimous += len(voted) == 1
        if approach == "RP":
            rows, info = rcr.rcr_features(case.responses, case.answers,
                                          majority_rule, return_info=True)
            for j, (r, feats) in enumerate(zip(case.responses, rows)):
                X.append([feats[k] for k in names])
                y.append(int(r.v == case.correct))
                group.append(g)
                a_idx.append(index[r.v])
                member.append(j)
        else:
            rows, info = acr.acr_features(case.responses, case.answers,
                                          majority_rule, return_info=True)
            for a, feats in rows:
                X.append([feats[k] for k in names])
                y.append(int(a == case.correct))
                group.append(g)
                a_idx.append(index[a])
                member.append(index[a])
        majority_ties += bool(info["majority_tie"])
    X = np.asarray(X, dtype=float).reshape(len(y), len(names))
    undefined = np.isnan(X).any(axis=1)
    return InstanceSet(
        approach=approach, X=X, y=np.asarray(y, dtype=int),
        group=np.asarray(group, dtype=int),
        answer_index=np.asarray(a_idx, dtype=int),
        member=np.asarray(member, dtype=int), undefined=undefined,
        feature_names=tuple(names),
        case_ids=tuple(c.case_id for c in subsets.cases),
        task_ids=tuple(subsets.task_ids),
        answers=tuple(tuple(c.answers) for c in subsets.cases),
        correct=tuple(c.correct for c in subsets.cases),
        synthetic=subsets.synthetic, marker=subsets.marker,
        info={"n_instances": len(y), "n_subsets": len(subsets),
              "positive_share": float(np.mean(y)) if len(y) else float("nan"),
              "majority_tie_subsets": int(majority_ties),
              "undefined_instances": int(undefined.sum()),
              "unanimous_subsets": int(unanimous),
              "subsets_without_vote_for_correct": int(no_vote_for_correct),
              "majority_rule": majority_rule})
