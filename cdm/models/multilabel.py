"""Multi-label factories for AMP, the AMP decision protocol, and the DAP
classifier factories.

Step A12 of plan §9.3 ("Multi-label factories + AMP protocol";
acceptance: BR+KNN == LP+KNN choices on random data (TH Table 5.1
fingerprint); argmax tie order, OQ17).

Techniques (plan §5.2; TH §4.1.4 p.45-46; MLA §5.4):
  AMP : {BR, CC, LP} x {KNN, LR, RF} = 9 (thesis, in the column order of
        TH Table 5.1); MLA adds Bernoulli NB -> 12 (MLA Table 5).
  DAP : KNN, LR, RF.

Library (plan §9.1, §5.2 "Fingerprint", §8 OQ19): scikit-multilearn
`BinaryRelevance`, `ClassifierChain`, `LabelPowerset` with their defaults.
scikit-multilearn 0.2.0 works under scikit-learn 1.9.1 / Python 3.13, so
the plan's fallback (sklearn MultiOutputClassifier + ClassifierChain + a
custom LP wrapper) is NOT used.

Per-label probability P(O_AG = 1 | x) of each wrapper (plan §8 OQ19:
"scikit-multilearn defaults (order = label order; LP marginalization)"):
  BR : one copy of the base classifier per label; its predict_proba[:, 1].
  CC : chain in the label order (MR, HAC, WC, SP, DA for the full label
       set); classifier j sees X plus the labels 0..j-1 (true labels when
       fitting, the chain's own 0/1 PREDICTIONS when predicting); the
       probability of label j is predict_proba[:, 1] of classifier j.
  LP : one multi-class classifier over the label combinations seen in the
       training set; the probability of label j is the sum of the class
       probabilities of the combinations that contain j (marginal).

Hyperparameters (plan §8 OQ15): sklearn defaults, set explicitly:
RF n_estimators=100, KNN n_neighbors=5, LR solver lbfgs; no scaling
(StandardScaler is the plan's sensitivity option); seed 0 (random_state
of RF and LR). LR `max_iter`: the plan says "raised to avoid
non-convergence" and gives NO number (config OQ15 `lr_max_iter: null`,
docs/IMPLEMENTATION_DECISIONS.md C4). This module does not pick one: an LR technique raises
until a number is given (argument `lr_max_iter` or the config).

AMP decision protocol (plan §4.1; TH p.37): the selected method is the
label with the highest probability; the 0.5 threshold of the text does
not affect the argmax. Ties (plan §8 OQ17): the first tied label in the
order of the label columns (thesis order MR, HAC, WC, SP, DA). The
comparison is plain `argmax` on the floats the library returns; no
tolerance is applied (the plan names none). What that means per technique
(decision M8):
  * KNN (k = 5), all three wrappers: a true tie is an exact float tie. BR
    and CC return n/5. LP sums the class probabilities n_c/5 in class
    order, and equal neighbour counts give the same float WITHIN ONE
    WRAPPER (checked exhaustively in tests/test_a12_multilabel.py). The
    floats of BR and LP are not bit-identical to each other (e.g. 0.6
    against 0.2 + 0.4); their CHOICES agree, which is the Table 5.1
    fingerprint.
  * RF with BR or CC: probabilities are multiples of 1/100; exact ties.
  * RF with LP: NOT exact. The marginals are sums of k/100 values, and
    rounding noise can break a true tie, so the selected label is then
    not always the first in the order (in a check about 6% of rows had a
    true tie, and in about 0.2% of rows the noise selected another label).
    This is what `argmax` on scikit-multilearn's output does; it is
    faithful to the library but it is not the documented OQ17 rule.
    `rounding_tie_rows` counts such rows so that a run can log them. No
    rounding is applied by default (docs/IMPLEMENTATION_DECISIONS.md M8).

Constant labels: BR, CC and LP all refuse a label column that is constant
in the training set (one rule for the three wrappers; docs/IMPLEMENTATION_DECISIONS.md M5).

DAP: `make_dap_model` wraps the classifier with a finite-input check on
every fit and predict (sklearn's RF accepts NaN silently).

Nothing in this module reads data or runs an experiment.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TRANSFORMS = ("BR", "CC", "LP")
BASES_TH = ("KNN", "LR", "RF")
#: MLA §5.4 also tests Bernoulli NB (plan §5.2; MLA Table 5).
BASES_MLA = ("BNB", "KNN", "LR", "RF")

#: The 9 thesis techniques in the column order of TH Table 5.1 (p.63).
AMP_TECHNIQUES_TH = tuple((t, b) for t in TRANSFORMS for b in BASES_TH)
#: The 12 MLA techniques (MLA Table 5).
AMP_TECHNIQUES_MLA = tuple((t, b) for t in TRANSFORMS for b in BASES_MLA)
#: DAP classifiers in the column order of TH Table 5.2 (p.63).
DAP_TECHNIQUES = ("KNN", "LR", "RF")

SCALINGS = ("none", "standard_scaler")
OQ17_IMPLEMENTED = "first_label_in_order"
OQ19_IMPLEMENTED = "scikit_multilearn_defaults"


def technique_name(transform, base):
    """'BR+RF' etc., as in TH Table 5.1."""
    return f"{transform}+{base}"


# ---- hyperparameters (OQ15) -----------------------------------------------------

def hyperparameters_from_config(config=None):
    """Keyword arguments {rf_n_estimators, knn_k, lr_solver, lr_max_iter,
    scaling, seed} of `make_base` / `make_amp_model` / `make_dap_classifier`
    from the config (plan §8 OQ15). `hyperparameters` must be
    'sklearn_defaults' (the only value the plan defines)."""
    from cdm.config import get_switch
    oq15 = get_switch("OQ15", config)
    if oq15["hyperparameters"] != "sklearn_defaults":
        raise NotImplementedError(
            f"OQ15 hyperparameters = {oq15['hyperparameters']!r} is not implemented")
    if oq15["scaling"] not in SCALINGS:
        raise ValueError(f"OQ15 scaling = {oq15['scaling']!r} is not one of {SCALINGS}")
    return {"rf_n_estimators": oq15["rf_n_estimators"], "knn_k": oq15["knn_k"],
            "lr_solver": oq15["lr_solver"], "lr_max_iter": oq15["lr_max_iter"],
            "scaling": oq15["scaling"], "seed": oq15["seed"]}


def make_base(name, rf_n_estimators=100, knn_k=5, lr_solver="lbfgs",
              lr_max_iter=None, scaling="none", seed=0):
    """An unfitted single-label sklearn classifier. Plan §8 OQ15.

    name : "KNN" -> KNeighborsClassifier(n_neighbors=knn_k)
           "LR"  -> LogisticRegression(solver=lr_solver, max_iter=lr_max_iter,
                                       random_state=seed)
           "RF"  -> RandomForestClassifier(n_estimators=rf_n_estimators,
                                           random_state=seed)
           "BNB" -> BernoulliNB()   (MLA variant only)
           Every other parameter is the sklearn default.
    lr_max_iter : REQUIRED for "LR". The plan gives no number ("raised to
           avoid non-convergence"), so None raises ValueError.
    scaling : "none" (plan default) or "standard_scaler" (plan sensitivity
           run): StandardScaler -> classifier in a Pipeline, fitted on the
           training data of each fit. Inside CC the scaler then also sees
           the appended label columns, and in DAP the f_AG columns.
    """
    if scaling not in SCALINGS:
        raise ValueError(f"scaling = {scaling!r} is not one of {SCALINGS}")
    if name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=knn_k)
    elif name == "LR":
        if lr_max_iter is None:
            raise ValueError(
                "LR needs lr_max_iter: plan §8 OQ15 says 'max_iter raised to "
                "avoid non-convergence' without a number (config OQ15 "
                "lr_max_iter is null; docs/IMPLEMENTATION_DECISIONS.md C4). Give the number "
                "explicitly or set it in the config")
        clf = LogisticRegression(solver=lr_solver, max_iter=int(lr_max_iter),
                                 random_state=seed)
    elif name == "RF":
        clf = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=seed)
    elif name == "BNB":
        clf = BernoulliNB()
    else:
        raise ValueError(f"unknown classifier {name!r}; expected one of {BASES_MLA}")
    return make_pipeline(StandardScaler(), clf) if scaling == "standard_scaler" else clf


def make_dap_classifier(name, **hyper):
    """An unfitted, BARE sklearn DAP classifier: KNN, LR or RF (plan §5.2;
    TH p.46), with the hyperparameters of `make_base`. DAP returns
    `predict(X)` (plan §4.2: "return the predicted class").

    The bare classifier has no input check: sklearn's RandomForest accepts
    NaN silently. Experiment code must use `make_dap_model`, which wraps
    this classifier with a finite-input guard."""
    if name not in DAP_TECHNIQUES:
        raise ValueError(f"unknown DAP classifier {name!r}; expected one of {DAP_TECHNIQUES}")
    return make_base(name, **hyper)


def _finite_2d(X, what="X"):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{what} must be (n, d)")
    if not np.isfinite(X).all():
        raise ValueError(f"{what} holds NaN or inf")
    return X


class DAPModel:
    """A DAP classifier (KNN, LR or RF) with input checks on every fit and
    predict (decision M6).

    fit(X, y)         : X (n, d) finite, y (n,) 0/1 codes of a*.
    predict(X)        : the predicted 0/1 code (plan §4.2).
    predict_proba(X)  : P(code = 1 | x), shape (n,).

    Every `fit` builds a fresh classifier (no state carried over).

    Unstated tie rule (docs/IMPLEMENTATION_DECISIONS.md M7): sklearn's `predict` returns the
    first class at equal probability, so P = 0.5 exactly gives code 0 (the
    answer a2). `half_rows` counts such rows so that a run can log them.

    A target that is constant in the training set raises (one class only;
    the plan gives no rule; docs/IMPLEMENTATION_DECISIONS.md M5).
    """

    def __init__(self, name, **hyper):
        make_dap_classifier(name, **hyper)            # validate now
        self.name, self.hyper = name, dict(hyper)
        self.model_ = None

    def fit(self, X, y):
        X = _finite_2d(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be (n,) with the n of X")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("y must hold 0/1 only")
        y = y.astype(int)
        if y.min() == y.max():
            raise ValueError("the target is constant in the training set; "
                             "DAP is undefined for it")
        self.model_ = make_dap_classifier(self.name, **self.hyper).fit(X, y)
        self.n_features_ = X.shape[1]
        return self

    def _check(self, X):
        if self.model_ is None:
            raise ValueError("the model is not fitted")
        X = _finite_2d(X)
        if X.shape[1] != self.n_features_:
            raise ValueError("X must be (n, d) with the d of the training set")
        return X

    def predict(self, X):
        X = self._check(X)
        return np.asarray(self.model_.predict(X), dtype=int)

    def predict_proba(self, X):
        X = self._check(X)
        classes = list(self.model_.classes_)
        return np.asarray(self.model_.predict_proba(X))[:, classes.index(1)]

    def half_rows(self, X):
        """Number of rows with P(code = 1) exactly 0.5 (decided by sklearn's
        class order, i.e. code 0)."""
        return int((self.predict_proba(X) == 0.5).sum())


def make_dap_model(name, **hyper):
    """Factory: DAPModel(name, **hyper), the guarded DAP classifier."""
    return DAPModel(name, **hyper)


# ---- multi-label wrappers ---------------------------------------------------------

def _wrapper_class(transform):
    from skmultilearn.problem_transform import (BinaryRelevance, ClassifierChain,
                                                LabelPowerset)
    classes = {"BR": BinaryRelevance, "CC": ClassifierChain, "LP": LabelPowerset}
    if transform not in classes:
        raise ValueError(f"unknown transformation {transform!r}; expected one of {TRANSFORMS}")
    return classes[transform]


def _dense(m):
    return np.asarray(m.todense() if hasattr(m, "todense") else m, dtype=float)


class AMPModel:
    """A multi-label technique (BR / CC / LP around a base classifier) with
    per-label probabilities and the AMP decision rule.

    transform : "BR", "CC" or "LP".
    base      : "KNN", "LR", "RF" or "BNB".
    labels    : names of the label columns of Y, in column order. The
                column order is the classifier-chain order (OQ19) and the
                argmax tie order (OQ17). Default: the thesis order
                (MR, HAC, WC, SP, DA); in an ablation pass the kept labels
                (cdm.approaches.instances.methods_kept).
    hyper     : keyword arguments of `make_base`.

    Every `fit` builds a fresh scikit-multilearn object from a fresh base
    classifier, so a model can be refitted (leave-one-out) without state
    carried over, and results depend only on the data and the seed.
    """

    def __init__(self, transform, base, labels=("MR", "HAC", "WC", "SP", "DA"),
                 **hyper):
        _wrapper_class(transform)
        make_base(base, **hyper)                     # validate now
        self.transform, self.base, self.hyper = transform, base, dict(hyper)
        self.labels = tuple(labels)
        if len(set(self.labels)) != len(self.labels) or not self.labels:
            raise ValueError("labels must be unique and non-empty")
        self.model_ = None

    @property
    def name(self):
        return technique_name(self.transform, self.base)

    def fit(self, X, Y):
        """Fit on X (n, d) and the 0/1 label matrix Y (n, q).

        Raises ValueError for non-finite X (sklearn's RF would accept NaN
        silently; how an undefined feature is filled is not decided, see
        docs/IMPLEMENTATION_DECISIONS.md F2 / AM12), for Y not in {0,1}, and for a label
        column that is constant in the training set. For BR and CC the
        binary classifier of that label would see one class, for which
        scikit-multilearn's `predict_proba[:, 1]` is undefined. LP could
        handle it (the marginal is 0 or 1), but the same rule is applied
        to all three wrappers so that the techniques of one table are
        always defined on the same training sets (docs/IMPLEMENTATION_DECISIONS.md M5; the
        plan gives no rule; it cannot occur in LOO over 1209 cases).
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y)
        if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
            raise ValueError("X must be (n, d) and Y (n, q) with the same n")
        if Y.shape[1] != len(self.labels):
            raise ValueError(f"Y has {Y.shape[1]} columns for {len(self.labels)} labels")
        if not np.isfinite(X).all():
            raise ValueError("X holds NaN or inf")
        if not np.isin(Y, (0, 1)).all():
            raise ValueError("Y must hold 0/1 only")
        Y = Y.astype(int)
        constant = [self.labels[j] for j in range(Y.shape[1])
                    if Y[:, j].min() == Y[:, j].max()]
        if constant:
            raise ValueError(
                f"label(s) {constant} are constant in the training set; "
                f"no rule is defined for them (applies to BR, CC and LP alike)")
        model = _wrapper_class(self.transform)(
            classifier=make_base(self.base, **self.hyper), require_dense=[True, True])
        model.fit(X, Y)
        self.model_, self.n_features_ = model, X.shape[1]
        return self

    def predict_proba(self, X):
        """P(O_AG = 1 | x) for every label: array (n, q), columns in the
        order of `labels`. See the module docstring for each wrapper."""
        if self.model_ is None:
            raise ValueError("the model is not fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError("X must be (n, d) with the d of the training set")
        if not np.isfinite(X).all():
            raise ValueError("X holds NaN or inf")
        return _dense(self.model_.predict_proba(X))

    def select(self, X, tie="first_label_in_order"):
        """AMP decision: index of the selected label for every row."""
        return select_methods(self.predict_proba(X), tie)

    def select_names(self, X, tie="first_label_in_order"):
        """AMP decision: name of the selected method for every row."""
        return [self.labels[j] for j in self.select(X, tie)]


def make_amp_model(transform, base, labels=("MR", "HAC", "WC", "SP", "DA"), **hyper):
    """Factory: AMPModel(transform, base, labels, **hyper)."""
    return AMPModel(transform, base, labels, **hyper)


def amp_techniques(variant="TH"):
    """The (transform, base) pairs of a variant: "TH" (9, TH Table 5.1) or
    "MLA" (12, MLA Table 5)."""
    if variant == "TH":
        return AMP_TECHNIQUES_TH
    if variant == "MLA":
        return AMP_TECHNIQUES_MLA
    raise ValueError(f"unknown variant {variant!r}; expected 'TH' or 'MLA'")


# ---- AMP decision protocol ---------------------------------------------------------

def tie_rule_from_config(config=None):
    """The OQ17 tie rule, validated. Only 'first_label_in_order' is
    implemented. The plan's alternative 'prefer_mr' says what to do only
    when MR is among the tied labels; with MR first in the label order it
    then equals the default, and for a tie without MR the plan gives no
    rule, so it raises."""
    from cdm.config import get_switch
    rule = get_switch("OQ17", config)
    _check_tie_rule(rule)
    return rule


def _check_tie_rule(rule):
    if rule == "prefer_mr":
        raise NotImplementedError(
            "OQ17 = 'prefer_mr' is not implemented: the plan does not say "
            "which label wins a tie that does not include MR")
    if rule != OQ17_IMPLEMENTED:
        raise ValueError(f"unknown OQ17 tie rule {rule!r}")


def select_methods(proba, tie="first_label_in_order"):
    """AMP decision (plan §4.1; TH p.37): for every row of the per-label
    probability matrix, the index of the label with the highest
    probability. Ties (plan §8 OQ17): the first tied label in column
    order. Plain float comparison (no tolerance): exact for KNN and for
    BR/CC with RF; for LP+RF rounding noise in the summed marginals can
    break a true tie (see the module docstring and `rounding_tie_rows`)."""
    _check_tie_rule(tie)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] == 0:
        raise ValueError("proba must be (n, q) with q >= 1")
    if np.isnan(proba).any():
        raise ValueError("proba holds NaN")
    return np.argmax(proba, axis=1)          # numpy: first occurrence of the maximum


def tied_labels(proba):
    """Boolean (n, q): the labels sharing the row maximum (for logging how
    often OQ17 decides)."""
    proba = np.asarray(proba, dtype=float)
    return proba == proba.max(axis=1, keepdims=True)


def rounding_tie_rows(proba, decimals=9):
    """Number of rows where `argmax(proba)` differs from
    `argmax(round(proba, decimals))`, i.e. where rounding noise, not the
    OQ17 label order, decided a tie (LP+RF; decision M8). For
    logging only; the selection itself is never rounded."""
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] == 0:
        raise ValueError("proba must be (n, q) with q >= 1")
    return int((np.argmax(proba, axis=1)
                != np.argmax(np.round(proba, decimals), axis=1)).sum())


def amp_success(Y, selected):
    """Outcome of AMP per case: O of the selected method, i.e. 1 iff the
    answer f_AG(R) of the selected method is a* (plan §4.1, §5.1).
    Y : (n, q) 0/1 outcomes of the methods; selected : (n,) label indices."""
    Y = np.asarray(Y)
    selected = np.asarray(selected, dtype=int)
    if Y.ndim != 2 or selected.shape != (len(Y),):
        raise ValueError("Y must be (n, q) and selected (n,)")
    return Y[np.arange(len(Y)), selected]


def check_oq19(config=None):
    """Plan §8 OQ19 has one implemented value (scikit-multilearn defaults:
    chain order = label order; LP marginals); any other raises."""
    from cdm.config import get_switch
    value = get_switch("OQ19", config)
    if value != OQ19_IMPLEMENTED:
        raise NotImplementedError(
            f"OQ19 = {value!r} is not implemented; only {OQ19_IMPLEMENTED!r} is")
    return value
