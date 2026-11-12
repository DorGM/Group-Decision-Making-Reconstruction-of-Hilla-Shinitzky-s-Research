"""A12 acceptance (plan §9.3 row A12): multi-label factories + AMP protocol.

  "BR+KNN == LP+KNN choices on random data (Table 5.1 fingerprint);
   argmax tie order (OQ17)"

SYNTHETIC / RANDOM DATA ONLY. The real runs are made by the driver scripts in
scripts/; no model is fitted on a World-A build here.

LR needs a `max_iter` that the plan does not give (OQ15); the tests pass
LR_MAX_ITER explicitly. It is a test value, not a project default.
"""
import itertools
import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from cdm.config import load_config
from cdm.models import multilabel as M

LR_MAX_ITER = 2000          # test value only (plan OQ15 gives no number)
LABELS = ("MR", "HAC", "WC", "SP", "DA")


def synthetic(n=160, d=8, seed=0, n_test=60):
    """Random features; five correlated 0/1 labels with rates like the
    methods' success rates (about .7, .65, .7, .7, .3)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n + n_test, d))
    z = X[:, 0] + 0.5 * rng.normal(size=n + n_test)
    Y = np.column_stack([
        z + 0.8 * rng.normal(size=n + n_test) > -0.6,
        z + 1.5 * rng.normal(size=n + n_test) > -0.6,
        z + 0.9 * rng.normal(size=n + n_test) > -0.6,
        X[:, 1] + z > -0.8,
        z + 0.8 * rng.normal(size=n + n_test) < -0.6,
    ]).astype(int)
    return X[:n], Y[:n], X[n:], Y[n:]


def model(t, b, **kw):
    kw.setdefault("lr_max_iter", LR_MAX_ITER)
    return M.make_amp_model(t, b, **kw)


# ---- technique lists ---------------------------------------------------------------

def test_technique_lists():
    # TH Table 5.1 column order
    assert [M.technique_name(*t) for t in M.AMP_TECHNIQUES_TH] == [
        "BR+KNN", "BR+LR", "BR+RF", "CC+KNN", "CC+LR", "CC+RF",
        "LP+KNN", "LP+LR", "LP+RF"]
    assert len(M.amp_techniques("TH")) == 9                 # TH: 3 x 3
    assert len(M.amp_techniques("MLA")) == 12               # MLA adds BNB
    assert {b for _, b in M.amp_techniques("MLA")} == {"BNB", "KNN", "LR", "RF"}
    assert M.DAP_TECHNIQUES == ("KNN", "LR", "RF")          # TH Table 5.2
    with pytest.raises(ValueError):
        M.amp_techniques("other")


# ---- hyperparameters (OQ15) --------------------------------------------------------

def test_base_classifiers_have_the_plan_defaults():
    knn = M.make_base("KNN")
    assert isinstance(knn, KNeighborsClassifier) and knn.n_neighbors == 5
    assert knn.weights == "uniform" and knn.metric == "minkowski" and knn.p == 2
    rf = M.make_base("RF")
    assert isinstance(rf, RandomForestClassifier)
    assert rf.n_estimators == 100 and rf.random_state == 0
    lr = M.make_base("LR", lr_max_iter=LR_MAX_ITER)
    assert isinstance(lr, LogisticRegression)
    assert lr.solver == "lbfgs" and lr.max_iter == LR_MAX_ITER and lr.random_state == 0
    assert isinstance(M.make_base("BNB"), BernoulliNB)
    # everything not named by the plan is the sklearn default
    for mine, ref in ((knn, KNeighborsClassifier()), (rf, RandomForestClassifier()),
                      (lr, LogisticRegression())):
        a, b = mine.get_params(), ref.get_params()
        diff = {k for k in a if a[k] != b[k]}
        assert diff <= {"random_state", "max_iter"}, diff
    assert M.make_base("RF", seed=7).random_state == 7
    with pytest.raises(ValueError):
        M.make_base("SVM")


def test_lr_max_iter_is_not_guessed():
    """Plan OQ15: 'max_iter raised' without a number. The config value is
    10000 (amendment AM18); the model code itself still has no default:
    LR raises when `lr_max_iter` is None."""
    hyper = M.hyperparameters_from_config()
    assert hyper == {"rf_n_estimators": 100, "knn_k": 5, "lr_solver": "lbfgs",
                     "lr_max_iter": 10000, "scaling": "none", "seed": 0}
    assert M.make_base("LR", **hyper).max_iter == 10000
    hyper["lr_max_iter"] = None
    with pytest.raises(ValueError, match="lr_max_iter"):
        M.make_base("LR", **hyper)
    with pytest.raises(ValueError, match="lr_max_iter"):
        M.make_amp_model("BR", "LR", **hyper)
    with pytest.raises(ValueError, match="lr_max_iter"):
        M.make_dap_classifier("LR")
    M.make_base("KNN", **hyper)
    M.make_base("RF", **hyper)
    cfg = load_config()
    cfg["open_questions"]["OQ15"]["default"]["lr_max_iter"] = 500
    assert M.make_base("LR", **M.hyperparameters_from_config(cfg)).max_iter == 500
    cfg["open_questions"]["OQ15"]["default"]["scaling"] = "minmax"
    with pytest.raises(ValueError):
        M.hyperparameters_from_config(cfg)


def test_scaling_option():
    assert not isinstance(M.make_base("KNN"), Pipeline)                 # default: none
    p = M.make_base("KNN", scaling="standard_scaler")
    assert isinstance(p, Pipeline) and [s[0] for s in p.steps] == [
        "standardscaler", "kneighborsclassifier"]
    with pytest.raises(ValueError):
        M.make_base("KNN", scaling="minmax")


def test_dap_factories():
    X, Y, Xt, _ = synthetic()
    y = Y[:, 0]
    for name in M.DAP_TECHNIQUES:
        clf = M.make_dap_classifier(name, lr_max_iter=LR_MAX_ITER)
        pred = clf.fit(X, y).predict(Xt)
        again = M.make_dap_classifier(name, lr_max_iter=LR_MAX_ITER).fit(X, y).predict(Xt)
        np.testing.assert_array_equal(pred, again)                      # deterministic
        assert set(np.unique(pred)) <= {0, 1}
    with pytest.raises(ValueError):
        M.make_dap_classifier("BNB")                                    # not a DAP technique


def test_dap_model_guards_fit_and_predict():
    """Sklearn's RF accepts NaN silently; the DAP
    wrapper refuses non-finite input before every fit and predict."""
    X, Y, Xt, _ = synthetic()
    y = Y[:, 0]
    Xn, Xi, Xtn = X.copy(), X.copy(), Xt.copy()
    Xn[0, 0], Xi[1, 1], Xtn[0, 0] = np.nan, np.inf, np.nan
    # the bare RF takes NaN without a word: this is what the guard is for
    M.make_dap_classifier("RF").fit(Xn, y)
    for name in M.DAP_TECHNIQUES:
        m = M.make_dap_model(name, lr_max_iter=LR_MAX_ITER)
        assert isinstance(m, M.DAPModel)
        with pytest.raises(ValueError, match="not fitted"):
            m.predict(Xt)
        for bad in (Xn, Xi):
            with pytest.raises(ValueError, match="NaN or inf"):
                m.fit(bad, y)
        m.fit(X, y)
        for call in (m.predict, m.predict_proba, m.half_rows):
            with pytest.raises(ValueError, match="NaN or inf"):
                call(Xtn)
        with pytest.raises(ValueError):
            m.predict(Xt[:, :3])
        pred = m.predict(Xt)
        bare = M.make_dap_classifier(name, lr_max_iter=LR_MAX_ITER).fit(X, y).predict(Xt)
        np.testing.assert_array_equal(pred, bare)             # same model, only guarded
        p = m.predict_proba(Xt)
        assert p.shape == (len(Xt),) and p.min() >= 0 and p.max() <= 1
        # unstated tie rule: P = 0.5 exactly -> code 0 (sklearn class order)
        np.testing.assert_array_equal(pred[p == 0.5], 0)
        np.testing.assert_array_equal(pred[p > 0.5], 1)
        assert m.half_rows(Xt) == int((p == 0.5).sum())
        with pytest.raises(ValueError):
            m.fit(X, y * 2)
        with pytest.raises(ValueError, match="constant"):
            m.fit(X, np.ones(len(X), dtype=int))
        with pytest.raises(ValueError):
            m.fit(X, Y)                                       # y must be 1-d
    with pytest.raises(ValueError):
        M.make_dap_model("BNB")


def test_rounding_tie_rows_counts_ties_decided_by_noise():
    """With LP+RF the marginals are sums of k/100 and a
    true tie can be broken by rounding noise instead of the label order."""
    a = 0.1 + 0.2                                   # 0.30000000000000004
    proba = np.array([[0.3, a, 0.1],                # true tie; noise picks label 1
                      [a, 0.3, 0.1],                # true tie; label 0 either way
                      [0.5, 0.2, 0.1]])             # no tie
    assert list(M.select_methods(proba)) == [1, 0, 0]
    assert M.rounding_tie_rows(proba) == 1
    assert M.rounding_tie_rows(np.array([[.2, .2], [.4, .1]])) == 0
    with pytest.raises(ValueError):
        M.rounding_tie_rows(np.array([.1, .2]))


# ---- wrappers: scikit-multilearn is used -------------------------------------------

def test_scikit_multilearn_is_used_not_the_fallback():
    from skmultilearn.problem_transform import (BinaryRelevance, ClassifierChain,
                                                LabelPowerset)
    X, Y, _, _ = synthetic()
    for t, cls in (("BR", BinaryRelevance), ("CC", ClassifierChain), ("LP", LabelPowerset)):
        assert isinstance(model(t, "KNN").fit(X, Y).model_, cls)


@pytest.mark.parametrize("t,b", M.AMP_TECHNIQUES_MLA)
def test_every_technique_gives_per_label_probabilities(t, b):
    X, Y, Xt, _ = synthetic()
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # e.g. LR non-convergence
        m = model(t, b).fit(X, Y)
        p = m.predict_proba(Xt)
    assert m.name == f"{t}+{b}"
    assert isinstance(p, np.ndarray) and p.shape == (len(Xt), 5)
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1 + 1e-12
    sel = m.select(Xt)
    assert sel.shape == (len(Xt),) and set(sel) <= set(range(5))
    assert m.select_names(Xt) == [LABELS[j] for j in sel]
    np.testing.assert_array_equal(sel, np.argmax(p, axis=1))


# ---- how each wrapper yields the per-label probability (OQ19) ----------------------

def test_br_probability_is_one_classifier_per_label():
    X, Y, Xt, _ = synthetic()
    p = model("BR", "RF").fit(X, Y).predict_proba(Xt)
    for j in range(5):
        ref = M.make_base("RF").fit(X, Y[:, j]).predict_proba(Xt)[:, 1]
        np.testing.assert_allclose(p[:, j], ref, atol=1e-12)


def test_cc_chain_is_in_label_order_and_feeds_predictions():
    X, Y, Xt, _ = synthetic()
    p = model("CC", "LR").fit(X, Y).predict_proba(Xt)
    Xtr, Xte = X.copy(), Xt.copy()
    for j in range(5):                      # label order = column order (OQ19)
        clf = M.make_base("LR", lr_max_iter=LR_MAX_ITER).fit(Xtr, Y[:, j])
        np.testing.assert_allclose(p[:, j], clf.predict_proba(Xte)[:, 1], atol=1e-9)
        Xtr = np.hstack([Xtr, Y[:, [j]]])                      # true labels in training
        Xte = np.hstack([Xte, clf.predict(Xte)[:, None]])      # own predictions at test
    # the first label of the chain equals BR; a later one does not
    br = model("BR", "LR").fit(X, Y).predict_proba(Xt)
    np.testing.assert_allclose(p[:, 0], br[:, 0], atol=1e-9)
    assert not np.allclose(p[:, 4], br[:, 4], atol=1e-6)
    # the chain order follows the column order: permuting the labels changes it
    perm = [4, 3, 2, 1, 0]
    q = model("CC", "LR", labels=tuple(LABELS[k] for k in perm)).fit(
        X, Y[:, perm]).predict_proba(Xt)
    np.testing.assert_allclose(q[:, 0], br[:, 4], atol=1e-9)


def test_lp_probability_is_the_marginal_over_label_combinations():
    X, Y, Xt, _ = synthetic()
    p = model("LP", "RF").fit(X, Y).predict_proba(Xt)
    combos, cls = [], []
    for row in Y:                           # classes in order of first appearance
        key = tuple(row)
        if key not in combos:
            combos.append(key)
        cls.append(combos.index(key))
    ref_clf = M.make_base("RF").fit(X, np.array(cls))
    cp = ref_clf.predict_proba(Xt)
    ref = cp @ np.array(combos, dtype=float)
    np.testing.assert_allclose(p, ref, atol=1e-12)
    np.testing.assert_allclose(cp.sum(axis=1), 1.0)


# ---- the Table 5.1 fingerprint -----------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_br_knn_equals_lp_knn_choices(seed):
    """TH Table 5.1: BR+KNN and LP+KNN have identical scores in every fold
    (plan §5.2 'Fingerprint'). On random data the two give the same
    per-label probabilities, hence the same AMP choices."""
    X, Y, Xt, Yt = synthetic(n=200, seed=seed, n_test=150)
    br, lp = model("BR", "KNN").fit(X, Y), model("LP", "KNN").fit(X, Y)
    pb, pl = br.predict_proba(Xt), lp.predict_proba(Xt)
    np.testing.assert_allclose(pb, pl, atol=1e-12)
    np.testing.assert_array_equal(br.select(Xt), lp.select(Xt))
    assert br.select_names(Xt) == lp.select_names(Xt)
    # so the AMP success rate is identical too
    assert M.amp_success(Yt, br.select(Xt)).sum() == M.amp_success(Yt, lp.select(Xt)).sum()
    # there ARE ties here (steps of 0.2), so the tie order is exercised
    assert (M.tied_labels(pb).sum(axis=1) > 1).any()
    # BR+KNN probabilities are multiples of 1/5
    np.testing.assert_allclose(pb * 5, np.round(pb * 5), atol=1e-12)


def test_fingerprint_also_holds_on_random_labels_and_with_scaling():
    rng = np.random.default_rng(42)
    X, Xt = rng.normal(size=(150, 6)), rng.normal(size=(100, 6))
    Y = (rng.random((150, 5)) < 0.5).astype(int)
    for scaling in ("none", "standard_scaler"):     # plan OQ15: "holds either way"
        a = model("BR", "KNN", scaling=scaling).fit(X, Y).select(Xt)
        b = model("LP", "KNN", scaling=scaling).fit(X, Y).select(Xt)
        np.testing.assert_array_equal(a, b)
    # not a property of every technique: BR+RF and LP+RF differ
    pa = model("BR", "RF").fit(X, Y).predict_proba(Xt)
    pb = model("LP", "RF").fit(X, Y).predict_proba(Xt)
    assert not np.allclose(pa, pb)


def _compositions(n):
    if n == 0:
        yield ()
        return
    for first in range(1, n + 1):
        for rest in _compositions(n - first):
            yield (first,) + rest


def test_lp_knn_ties_are_exact_float_ties():
    """Why no tolerance is needed for OQ17 with KNN (k = 5): LP adds the
    class probabilities n_c/5 of the classes that contain a label, in
    class order, starting from 0.0. For every way of splitting 5
    neighbours into classes and every two labels (= two sets of classes)
    with the same neighbour count, the two float sums are bit-identical,
    so a true tie is never broken by rounding noise."""
    checked = 0
    for comp in _compositions(5):
        probs = [c / 5 for c in comp]
        sums = {}
        for mask in range(1, 2 ** len(comp)):
            total, count = 0.0, 0
            for i in range(len(comp)):
                if mask >> i & 1:
                    total += probs[i]
                    count += comp[i]
            sums.setdefault(count, set()).add(total)
            checked += 1
        assert all(len(v) == 1 for v in sums.values()), comp
    assert checked > 100


# ---- AMP decision protocol and tie order (OQ17) ------------------------------------

def test_select_highest_probability():
    p = np.array([[.1, .9, .3, .2, .0],
                  [.6, .2, .3, .7, .1],
                  [.2, .2, .2, .2, .9]])
    np.testing.assert_array_equal(M.select_methods(p), [1, 3, 4])
    # TH p.37: the 0.5 threshold does not matter; also when no label is > 0.5
    np.testing.assert_array_equal(M.select_methods(p * 0.4), [1, 3, 4])


def test_argmax_tie_order_is_the_label_order():
    """Plan OQ17: first label in the order (MR, HAC, WC, SP, DA)."""
    order = ("MR", "HAC", "WC", "SP", "DA")
    assert tuple(load_config()["open_questions"]["OQ17"]["label_order"]) == order
    assert M.tie_rule_from_config() == "first_label_in_order"
    for k in range(2, 6):
        for tied in itertools.combinations(range(5), k):
            p = np.full((1, 5), 0.4)
            p[0, list(tied)] = 0.8                       # KNN-like steps of 0.2
            assert M.select_methods(p)[0] == min(tied)
            assert tuple(np.flatnonzero(M.tied_labels(p)[0])) == tied
    assert order[M.select_methods(np.full((1, 5), 0.6))[0]] == "MR"
    assert order[M.select_methods(np.array([[.2, .8, .4, .8, .8]]))[0]] == "HAC"
    assert order[M.select_methods(np.array([[.2, .4, .8, .8, .8]]))[0]] == "WC"
    assert order[M.select_methods(np.array([[.2, .4, .6, 1., 1.]]))[0]] == "SP"


def test_tie_order_on_a_fitted_knn_model():
    """Two labels that are identical in the training set always tie under
    KNN; the earlier column wins, whatever the labels are called."""
    X, Y, Xt, _ = synthetic()
    Y2 = Y.copy()
    Y2[:, 3] = Y2[:, 1]                                   # SP := HAC
    for t in ("BR", "LP"):
        m = model(t, "KNN").fit(X, Y2)
        sel = m.select(Xt)
        assert 3 not in sel                               # SP never beats HAC
        p = m.predict_proba(Xt)
        np.testing.assert_array_equal(p[:, 1], p[:, 3])
    # label order of an ablation row (e.g. row 5: MR, SP, DA)
    m = M.make_amp_model("BR", "KNN", labels=("MR", "SP", "DA")).fit(X, Y[:, [0, 3, 4]])
    assert set(m.select_names(Xt)) <= {"MR", "SP", "DA"}


def test_other_tie_rules_and_bad_input():
    with pytest.raises(NotImplementedError):
        M.select_methods(np.ones((1, 5)), tie="prefer_mr")     # undefined without MR
    with pytest.raises(ValueError):
        M.select_methods(np.ones((1, 5)), tie="random")
    cfg = load_config()
    cfg["open_questions"]["OQ17"]["default"] = "prefer_mr"
    with pytest.raises(NotImplementedError):
        M.tie_rule_from_config(cfg)
    with pytest.raises(ValueError):
        M.select_methods(np.array([[.1, np.nan]]))
    with pytest.raises(ValueError):
        M.select_methods(np.ones(5))
    assert M.check_oq19() == "scikit_multilearn_defaults"
    cfg["open_questions"]["OQ19"]["default"] = "other"
    with pytest.raises(NotImplementedError):
        M.check_oq19(cfg)


def test_amp_success():
    Y = np.array([[1, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 0]])
    np.testing.assert_array_equal(M.amp_success(Y, [0, 0, 4]), [1, 0, 0])
    np.testing.assert_array_equal(M.amp_success(Y, [3, 4, 1]), [1, 1, 1])
    with pytest.raises(ValueError):
        M.amp_success(Y, [0, 1])


# ---- determinism -------------------------------------------------------------------

@pytest.mark.parametrize("t,b", M.AMP_TECHNIQUES_TH)
def test_determinism_under_a_fixed_seed(t, b):
    X, Y, Xt, _ = synthetic(seed=3)
    p1 = model(t, b).fit(X, Y).predict_proba(Xt)
    p2 = model(t, b).fit(X, Y).predict_proba(Xt)
    np.testing.assert_array_equal(p1, p2)
    m = model(t, b)
    np.testing.assert_array_equal(m.fit(X, Y).predict_proba(Xt), p1)
    np.testing.assert_array_equal(m.fit(X, Y).predict_proba(Xt), p1)    # refit: no state
    np.testing.assert_array_equal(model(t, b).fit(X, Y).select(Xt), np.argmax(p1, axis=1))


def test_rf_seed_matters_and_is_explicit():
    X, Y, Xt, _ = synthetic(seed=3)
    p0 = model("BR", "RF", seed=0).fit(X, Y).predict_proba(Xt)
    p1 = model("BR", "RF", seed=1).fit(X, Y).predict_proba(Xt)
    assert not np.array_equal(p0, p1)
    np.testing.assert_array_equal(p0, model("BR", "RF").fit(X, Y).predict_proba(Xt))


# ---- input checks ------------------------------------------------------------------

def test_fit_input_checks():
    X, Y, Xt, _ = synthetic()
    with pytest.raises(ValueError):
        M.make_amp_model("XX", "KNN")
    with pytest.raises(ValueError):
        M.make_amp_model("BR", "SVM")
    with pytest.raises(ValueError):
        M.make_amp_model("BR", "KNN", labels=("MR", "MR"))
    with pytest.raises(ValueError):
        model("BR", "KNN").predict_proba(Xt)                      # not fitted
    with pytest.raises(ValueError):
        model("BR", "KNN").fit(X, Y[:, :4])                       # 4 columns, 5 labels
    Xn = X.copy()
    Xn[0, 0] = np.nan
    for b in ("KNN", "RF"):
        with pytest.raises(ValueError, match="NaN"):
            model("BR", b).fit(Xn, Y)                             # RF would accept NaN silently
    with pytest.raises(ValueError):
        model("BR", "KNN").fit(X, Y * 2)
    Yc = Y.copy()
    Yc[:, 2] = 1
    # one rule for the three wrappers
    for t in ("BR", "CC", "LP"):
        with pytest.raises(ValueError, match="constant"):
            model(t, "KNN").fit(X, Yc)
    m = model("BR", "KNN").fit(X, Y)
    with pytest.raises(ValueError):
        m.predict_proba(Xt[:, :3])
