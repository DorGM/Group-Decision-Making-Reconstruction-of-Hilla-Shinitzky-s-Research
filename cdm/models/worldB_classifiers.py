"""Classifiers of the RP / AP approaches (plan §4.3, §8 OQ26, OQ35;
steps B6-B8).

What the thesis says (TH §4.2.4 p.56; MCF §4.4), and all it says:
  "The chosen classification technique, for the final evaluation of both RP
  and AP, is a Weighted Average Probability voting ensemble technique
  (i.e., Soft Voting), which is composed of five classifiers: Linear
  Discriminant Analysis, Random Forest, XGboost, Logistic Regression, and
  K-Nearest Neighbor."
  Tables A.7 / A.8 (TH p.114) add "Bag KNN" and "SVM".
Library, weights, hyperparameters, scaling and seeds are not stated.

Plan defaults (labelled, open):
  OQ26  soft_vote = sklearn VotingClassifier(voting="soft"), equal weights,
        members with library defaults, no scaling.
  OQ35  bag_knn = BaggingClassifier(KNeighborsClassifier()) defaults;
        svm = SVC(probability=True) defaults.
  OQ15  RF n_estimators 100, KNN k 5, LR lbfgs with max_iter raised
        (AM18: 10000), seed 0. RF / LR / KNN are the `make_base` objects of
        cdm.models.multilabel (the Phase-A classifiers).
  OQ33  no treatment of class imbalance.
Choices beyond the plan (decision B6-2): XGBClassifier, BaggingClassifier
and SVC get random_state = the OQ15 seed; XGBoost runs on one thread
(n_jobs = 1).

"Predicted True" (plan §4.3: "p_r > 0.5 (sklearn `predict`)") is the
classifier's own `predict`; the probability is `predict_proba`. For
soft_vote and bag_knn `predict` is the arg-max of that probability; for
SVC it is the sign of the decision function, which can disagree with the
Platt-scaled probability (sklearn documentation). Both are kept
(decision B8-1).
"""
import numpy as np

from cdm.models import multilabel as M

#: Names used in the code -> the label of TH Tables A.7 / A.8.
CLASSIFIERS = ("soft_vote", "bag_knn", "svm")
THESIS_LABEL = {"soft_vote": "Vote soft", "bag_knn": "Bag KNN", "svm": "SVM"}
#: Members of the ensemble in the order of TH p.56.
SOFT_VOTE_MEMBERS = ("LDA", "RF", "XGB", "LR", "KNN")


def oq26_from_config(config=None):
    """The OQ26 / OQ35 / OQ33 defaults of the config; raise if they are not
    the ones implemented here (a changed switch is never ignored)."""
    from cdm.config import get_switch, load_config
    config = config if config is not None else load_config()
    oq26 = get_switch("OQ26", config)
    want26 = {"ensemble": "VotingClassifier_soft_equal_weights",
              "members": ["LDA", "RF", "XGBClassifier", "LR", "KNN"],
              "member_hyperparameters": "defaults", "scaling": "none"}
    if dict(oq26) != want26:
        raise NotImplementedError(f"OQ26 = {oq26!r} is not implemented (only {want26})")
    oq35 = get_switch("OQ35", config)
    want35 = {"bag_knn": "BaggingClassifier(KNeighborsClassifier())_defaults",
              "svm": "SVC(probability=True)_defaults"}
    if dict(oq35) != want35:
        raise NotImplementedError(f"OQ35 = {oq35!r} is not implemented (only {want35})")
    if get_switch("OQ33", config) != "none":
        raise NotImplementedError("OQ33: only 'none' is implemented")
    return {"OQ26": dict(oq26), "OQ35": dict(oq35), "OQ33": "none"}


def hyperparameters_from_config(config=None):
    """OQ15 hyperparameters (as Phase A). OQ26 fixes "no scaling" for the
    ensemble; the OQ15 scaling switch must agree."""
    hyper = M.hyperparameters_from_config(config)
    if hyper["scaling"] != "none":
        raise NotImplementedError(
            "World B: plan OQ26 fixes 'no scaling'; OQ15 scaling must be 'none'")
    return hyper


def make_soft_vote(hyper):
    """VotingClassifier(voting="soft") over LDA, RF, XGBoost, LR, KNN with
    equal weights (plan OQ26; TH p.56)."""
    import xgboost
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import VotingClassifier
    seed = int(hyper["seed"])
    members = [
        ("LDA", LinearDiscriminantAnalysis()),
        ("RF", M.make_base("RF", **hyper)),
        ("XGB", xgboost.XGBClassifier(random_state=seed, n_jobs=1)),
        ("LR", M.make_base("LR", **hyper)),
        ("KNN", M.make_base("KNN", **hyper)),
    ]
    assert tuple(n for n, _ in members) == SOFT_VOTE_MEMBERS
    return VotingClassifier(estimators=members, voting="soft")


def make_classifier(name, hyper=None):
    """An unfitted classifier: "soft_vote", "bag_knn" or "svm"."""
    hyper = dict(hyper if hyper is not None else hyperparameters_from_config())
    seed = int(hyper["seed"])
    if name == "soft_vote":
        return make_soft_vote(hyper)
    if name == "bag_knn":
        from sklearn.ensemble import BaggingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        return BaggingClassifier(KNeighborsClassifier(), random_state=seed)
    if name == "svm":
        from sklearn.svm import SVC
        return SVC(probability=True, random_state=seed)
    raise ValueError(f"unknown classifier {name!r}; expected one of {CLASSIFIERS}")


def lr_not_converged(model, hyper):
    """1 if the fitted model holds a logistic regression that stopped at
    max_iter (as Phase A counts `lr_not_converged`), else 0."""
    named = getattr(model, "named_estimators_", None)
    if not named or "LR" not in named:
        return 0
    n_iter = np.max(np.atleast_1d(named["LR"].n_iter_))
    return int(n_iter >= int(hyper["lr_max_iter"]))


def predict_both(model, X):
    """(label, p_true, p_false): the classifier's binary classification
    (`predict`, 1 = "True") and its probabilities (`predict_proba`) of the
    classes 1 and 0 (TH p.39: "a binary classification as well as a
    prediction probability")."""
    classes = list(model.classes_)
    if sorted(classes) != [0, 1]:
        raise ValueError(f"the model was fitted on the classes {classes}, not on 0 and 1")
    proba = np.asarray(model.predict_proba(X), dtype=float)
    label = np.asarray(model.predict(X)).astype(int)
    return label, proba[:, classes.index(1)], proba[:, classes.index(0)]
