"""Leave-one-group-out predictions for RP and AP (plan §5.7; TH §4.2.4
p.57-59; MCF §4.4; steps B6-B8).

    UNTESTED ON REAL DATA.

TH p.57: "we performed a leave-one-group-out procedure for extracting the
model's predictions ... For each group of instances, train the
classification model over the entire data set, excluding the instances in
that group, and then extract a prediction from the trained model for each
instance in the group. In our implementation, each 'group' is the set of
instances generated from a single set of responses" — i.e. one of the 77
sub-sets [EVIDENCE]. So the other sub-sets of the SAME task stay in the
training data of a held-out sub-set. That is the thesis's procedure and
the default here (`group_by="subset"`). `group_by="task"` leaves out every
sub-set of the held-out sub-set's task; it is NOT in the papers and exists
only as a sensitivity switch (decision B6-3).

Empty out-group (decision B1-2, B6-4): instances with an undefined
feature come only from a sub-set in which every response gives the same
answer. Rule in use [INVENTED, labelled default]
`drop_from_training_answer_is_unanimous`: such instances are left out of
every training set and get no prediction (label -1, probabilities NaN);
the sub-set's answer is its only voted answer whatever a classifier would
say, so no prediction is needed. Alternative: `raise`.
"""
import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from cdm.models import worldB_classifiers as WC

GROUP_BY = ("subset", "task")
EMPTY_OUT_GROUP_RULES = ("drop_from_training_answer_is_unanimous", "raise")
DEFAULT_EMPTY_OUT_GROUP_RULE = EMPTY_OUT_GROUP_RULES[0]


@dataclass
class Predictions:
    """Out-of-group predictions of every instance of an InstanceSet.

    label   : (n,) the classifier's classification, 1 = "True", 0 = "False";
              -1 for an undefined instance (no prediction)
    p_true, p_false : (n,) predicted probabilities; NaN if undefined
    """
    classifier: str
    group_by: str
    label: np.ndarray
    p_true: np.ndarray
    p_false: np.ndarray
    log: dict = field(default_factory=dict)


def held_out_units(inst, group_by="subset"):
    """(n,) the held-out unit of every instance: its sub-set (thesis) or
    the task of its sub-set (sensitivity)."""
    if group_by not in GROUP_BY:
        raise ValueError(f"group_by must be one of {GROUP_BY}")
    if group_by == "subset":
        return inst.group.copy()
    code = {}
    for t in inst.task_ids:
        code.setdefault(t, len(code))
    per_group = np.asarray([code[t] for t in inst.task_ids], dtype=int)
    return per_group[inst.group]


def logo_predict(inst, classifier="soft_vote", hyper=None, group_by="subset",
                 empty_out_group=DEFAULT_EMPTY_OUT_GROUP_RULE, make=None):
    """Leave-one-group-out predictions (TH p.57-59).

    inst       : cdm.approaches.worldB_instances.InstanceSet
    classifier : a name of cdm.models.worldB_classifiers.CLASSIFIERS
    hyper      : OQ15 hyperparameters (default: the config's)
    make       : optional factory () -> unfitted classifier (tests)
    """
    if empty_out_group not in EMPTY_OUT_GROUP_RULES:
        raise ValueError(f"empty_out_group must be one of {EMPTY_OUT_GROUP_RULES}")
    hyper = dict(hyper if hyper is not None else WC.hyperparameters_from_config())
    if inst.undefined.any() and empty_out_group == "raise":
        raise ValueError(
            f"{int(inst.undefined.sum())} instances have an undefined feature "
            "(empty out-group, decision B1-2) and empty_out_group = 'raise'")
    units = held_out_units(inst, group_by)
    defined = ~inst.undefined
    n = inst.n
    label = np.full(n, -1, dtype=int)
    p_true = np.full(n, np.nan)
    p_false = np.full(n, np.nan)
    seen = Counter()
    not_converged = 0
    n_fits = 0
    sizes = []
    for u in np.unique(units):
        test = np.flatnonzero((units == u) & defined)
        if test.size == 0:
            continue
        train = np.flatnonzero((units != u) & defined)
        y = inst.y[train]
        if len(np.unique(y)) < 2:
            raise ValueError(f"held-out unit {u}: the training data has one class only")
        model = make() if make is not None else WC.make_classifier(classifier, hyper)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(inst.X[train], y)
            lab, pt, pf = WC.predict_both(model, inst.X[test])
        for w in caught:
            seen[f"{w.category.__name__}: {str(w.message).splitlines()[0][:120]}"] += 1
        label[test], p_true[test], p_false[test] = lab, pt, pf
        not_converged += WC.lr_not_converged(model, hyper)
        n_fits += 1
        sizes.append(int(train.size))
    done = label >= 0
    return Predictions(
        classifier=classifier, group_by=group_by, label=label, p_true=p_true,
        p_false=p_false,
        log={"n_fits": n_fits, "lr_not_converged": int(not_converged),
             "train_size_min": min(sizes) if sizes else None,
             "train_size_max": max(sizes) if sizes else None,
             "undefined_instances": int(inst.undefined.sum()),
             "empty_out_group_rule": empty_out_group,
             "predicted_true_share": float(np.mean(label[done] == 1)) if done.any() else None,
             # label vs probability (they can disagree for SVC; decision B8-1)
             "label_true_but_p_true_not_above_half": int(np.sum((label == 1) & ~(p_true > 0.5))),
             "label_false_but_p_true_above_half": int(np.sum((label == 0) & (p_true > 0.5))),
             "instance_accuracy": float(np.mean(label[done] == inst.y[done])) if done.any() else None,
             "warnings": dict(seen)})
