"""SHAP analysis of the CDMCR features (step A19 of plan §9.3; plan §5.9;
TH §4.3.1 p.60-61 and §5.3.1 p.77-82: Figs 5.4-5.7 and A.1; plan item T30).

What the thesis says (and all it says):
  * "we evaluated each target label separately, as if it was the sole
    target label of a binary classification model" (TH p.60);
  * "The classification models were trained using the same data sets used
    for evaluating the AMP and DAP aggregation approaches" (TH p.60-61);
  * the plots show "the SHAP values of every feature for every sample",
    features "sorted by their importance" (TH p.77);
  * SHAP = Lundberg & Lee 2017 (ref [105]), cited in general terms only.
The model, the explainer, the library versions and the plotting call are
not stated. Plan §5.9 / §8 OQ30 (config OQ30) therefore fix:
  (i)  log-odds analysis: one XGBoost classifier per label on the 27
       CDMCR features, `shap.TreeExplainer`, log-odds output, beeswarm
       plot with max_display = 20 (the figures end with "Sum of 8 other
       features" = 27 - 19 [INFERRED]; their x-axes span about -5 .. +5,
       and SHAP values of a probability output lie within -1 .. +1);
  (ii) comparison: `shap.KernelExplainer` on the actual model. For CDMCR
       the actual model is the Random Forest of a label inside BR+RF
       (BR = one independent classifier per label; TH p.70);
  both on models trained on ALL cases, SHAP values for all cases.
Everything else in this module is a recorded choice (docs/IMPLEMENTATION_DECISIONS.md SH1-SH10).

Importance of a feature = mean |SHAP| over the cases (what a beeswarm plot
sorts by; TH p.77 words it as the "sum of its SHAP values", plan
§5.9). The thesis's rank orders below were read from the rendered figures.

Nothing here is tuned towards the thesis's rankings.
"""
import warnings

import numpy as np
from scipy import stats as sps

from cdm.approaches import instances as I
from cdm.features import cdmcr
from cdm.models import multilabel as M

#: Labels in the order the thesis presents them (Figs 5.4, 5.5, 5.6, 5.7, A.1).
LABELS_REPORT_ORDER = ("MR", "DA", "SP", "HAC", "WC")
THESIS_FIGURE = {"MR": "TH Fig. 5.4 (p.80)", "DA": "TH Fig. 5.5 (p.80)",
                 "SP": "TH Fig. 5.6 (p.81)", "HAC": "TH Fig. 5.7 (p.81)",
                 "WC": "TH Fig. A.1 (p.113)"}
#: plan OQ30 (i): beeswarm max_display = 20 -> 19 features + "Sum of 8 other".
MAX_DISPLAY = 20
N_SHOWN = MAX_DISPLAY - 1
MODELS = ("xgboost_tree", "rf_kernel")
MODEL_TITLE = {
    "xgboost_tree": "XGBoost per label, TreeExplainer, log-odds (plan OQ30 (i))",
    "rf_kernel": "Random Forest of the label (as in BR+RF), KernelExplainer, "
                 "probability (plan OQ30 (ii))"}

#: The figures write "AvgPSv"; TH Table 3.2 and the code write "AvgPSV".
_FIGURE_NAME = {"AvgPSv": "AvgPSV"}


def _names(*figure_names):
    return tuple(_FIGURE_NAME.get(n, n) for n in figure_names)


#: T30: the 19 features each figure lists, top to bottom, as read from the
#: rendered thesis pages (PDF pages 101, 102, 134). The other 8 features of
#: a figure are merged into "Sum of 8 other features" (no order known).
THESIS_RANKING = {
    "MR": _names(
        "SG_B_Amax", "SG_D_MaxES_MinES", "MaxPSa", "AvgPSv", "D_Smax_Smin", "CAmin",
        "SG_VarSAmax", "N", "SG_VarCAmin", "SG_VarCAmax", "D_MaxC_AvgC", "VarC",
        "P_lowPSv_highC", "SG_D_MaxAvgC_MinAvgC", "CAmax", "SG_D_MaxVarS_MinVarS",
        "AvgC", "SG_AvgCAmax", "SG_AvgCAmin"),
    "DA": _names(
        "SG_B_Amax", "SG_D_MaxES_MinES", "MaxPSa", "D_Smax_Smin", "CAmin", "AvgPSv",
        "SG_VarSAmax", "D_MaxC_AvgC", "N", "SG_VarCAmax", "SG_D_MaxAvgC_MinAvgC",
        "VarC", "SG_VarCAmin", "P_lowPSv_highC", "SG_D_MaxVarS_MinVarS", "CAmax",
        "SG_AvgCAmax", "SG_AvgCAmin", "SG_D_MaxVarC_MinVarC"),
    "SP": _names(
        "SG_B_Amax", "D_Smax_Smin", "SG_D_MaxES_MinES", "MaxPSa", "SG_VarSAmax", "N",
        "D_MaxC_AvgC", "CAmax", "AvgPSv", "CAmin", "VarC", "SG_VarCAmin",
        "SG_VarCAmax", "SG_D_MaxAvgC_MinAvgC", "SG_D_MaxVarS_MinVarS",
        "P_lowPSv_highC", "AvgC", "SG_D_MaxVarC_MinVarC", "P_lowC_highPSv"),
    "HAC": _names(
        "CAmin", "B_Amax_MaxCa", "AvgPSv", "SG_B_Amax", "SG_D_MaxES_MinES",
        "D_Smax_Smin", "SG_AvgCAmax", "MaxPSa", "N", "SG_VarCAmin", "VarC",
        "SG_VarSAmax", "SG_AvgCAmin", "SG_D_MaxAvgC_MinAvgC", "SG_VarCAmax", "MinC",
        "P_lowPSv_highC", "SG_D_MaxVarC_MinVarC", "P_lowC_highPSv"),
    "WC": _names(
        "SG_B_Amax", "SG_D_MaxES_MinES", "CAmin", "MaxPSa", "D_Smax_Smin", "AvgPSv",
        "SG_VarSAmax", "N", "D_MaxC_AvgC", "SG_VarCAmin", "SG_AvgCAmax",
        "SG_VarCAmax", "SG_AvgCAmin", "SG_D_MaxVarC_MinVarC", "VarC", "AvgC",
        "SG_D_MaxAvgC_MinAvgC", "P_lowPSv_highC", "SG_D_MaxVarS_MinVarS"),
}

#: Directions the thesis TEXT states (TH p.78-79): +1 = high values of the
#: feature go with a positive classification of the label, -1 = with a
#: negative one. Only what the text says; nothing read off the dot colours.
THESIS_DIRECTION = {
    ("MR", "SG_B_Amax"): -1,      # "MR and SP are classified as correct for low values (0)"
    ("SP", "SG_B_Amax"): -1,
    ("DA", "SG_B_Amax"): +1,      # "an opposite trend for DA"
    ("MR", "AvgPSV"): +1,         # "a positive label value is associated with high values"
    ("MR", "D_Smax_Smin"): +1,    # "lower values ... negative class for MR and positive
    ("DA", "D_Smax_Smin"): -1,    #  class for DA"
    ("HAC", "CAmin"): -1,         # "Lower values of CAmin ... positive classifications for HAC"
    ("MR", "CAmin"): -1,          # "this trend is also observed for MR"
    ("SP", "CAmin"): -1,          # "a similar but less coherent trend for SP"
    ("DA", "CAmin"): +1,          # "an opposite trend for DA"
    ("HAC", "B_Amax_MaxCa"): +1,  # "higher values ... correct result by HAC"
    ("SP", "N"): -1,              # "low values of N are associated with a correct result by SP"
}

#: Plan §9.3 row A19: the acceptance statements (reported, never asserted).
ACCEPTANCE_TOP = {"MR": "SG_B_Amax", "DA": "SG_B_Amax", "SP": "SG_B_Amax", "HAC": "CAmin"}


def oq30_from_config(config=None):
    """The OQ30 default of the config; raises if it is not the one this
    module implements."""
    from cdm.config import get_switch
    oq30 = get_switch("OQ30", config)
    want = {"faithful": "xgboost_per_label_TreeExplainer_log_odds_beeswarm_max_display_20",
            "comparison": "KernelExplainer_on_actual_model", "trained_on": "all_data"}
    if dict(oq30) != want:
        raise NotImplementedError(f"OQ30 = {oq30!r} is not implemented (only {want})")
    return dict(oq30)


# ---- the two models and their SHAP values ------------------------------------------

def _check(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    if X.ndim != 2 or y.shape != (len(X),):
        raise ValueError("X must be (n, d) and y (n,)")
    if not np.isfinite(X).all():
        raise ValueError("X has non-finite values (docs/IMPLEMENTATION_DECISIONS.md M6: no imputation rule)")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("the label must take both values 0 and 1")
    return X, y


def fit_xgboost(X, y, seed=0, n_jobs=1):
    """XGBClassifier with the library defaults, `random_state` = seed
    (plan OQ30 (i) names the library only; docs/IMPLEMENTATION_DECISIONS.md SH2)."""
    import xgboost
    X, y = _check(X, y)
    return xgboost.XGBClassifier(random_state=int(seed), n_jobs=int(n_jobs)).fit(X, y)


def xgboost_shap(model, X):
    """(values (n, d), base value): TreeSHAP of the fitted XGBoost model in
    log-odds units (`shap.TreeExplainer` defaults for XGBoost: path-
    dependent, margin output). The values of a case plus the base value
    equal the model's margin for the case."""
    import shap
    X = np.asarray(X, dtype=float)
    ex = shap.TreeExplainer(model)
    values = np.asarray(ex.shap_values(X), dtype=float)
    base = float(np.ravel(ex.expected_value)[0])
    if values.shape != X.shape:
        raise RuntimeError(f"unexpected SHAP shape {values.shape}")
    return values, base


def fit_rf(X, y, hyper=None):
    """The Random Forest of one label exactly as BR+RF fits it: the base
    classifier of cdm.models.multilabel.make_base("RF", ...) with the OQ15
    hyperparameters (BR fits one such forest per label on the same X)."""
    X, y = _check(X, y)
    hyper = dict(hyper if hyper is not None else M.hyperparameters_from_config())
    return M.make_base("RF", **hyper).fit(X, y)


def _p1(model):
    k = list(model.classes_).index(1)
    return lambda Z: model.predict_proba(Z)[:, k]


def kernel_shap(model, X, background_n=50, nsamples="auto", seed=0, rows=None):
    """(values (len(rows), d), base value): KernelSHAP of P(label = 1) of a
    fitted sklearn classifier (plan OQ30 (ii)).

    background_n : number of cases drawn without replacement from X as the
                   background (`shap.sample`, random_state = seed); all of
                   X if X has no more rows. docs/IMPLEMENTATION_DECISIONS.md SH4.
    nsamples     : coalitions per case; "auto" = shap's 2 d + 2048.
    seed         : also seeds numpy's global generator, which KernelSHAP
                   draws its coalitions from (restored afterwards).
    rows         : positions of the cases to explain; default all.
    """
    import shap
    X = np.asarray(X, dtype=float)
    rows = np.arange(len(X)) if rows is None else np.asarray(rows, dtype=int)
    bg = X if len(X) <= background_n else shap.sample(X, int(background_n),
                                                      random_state=int(seed))
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ex = shap.KernelExplainer(_p1(model), bg)
            values = np.asarray(ex.shap_values(X[rows], nsamples=nsamples, silent=True),
                                dtype=float)
    finally:
        np.random.set_state(state)
    return values.reshape(len(rows), X.shape[1]), float(np.ravel(ex.expected_value)[0])


def rf_kernel_label(X, y, hyper=None, background_n=50, nsamples="auto", seed=0):
    """Fit the label's forest and return its KernelSHAP (values, base).
    A top-level function so that labels can run in separate processes."""
    return kernel_shap(fit_rf(X, y, hyper), X, background_n, nsamples, seed)


# ---- importance, ranks, directions ---------------------------------------------------

def mean_abs(values):
    """Importance of every feature: mean |SHAP| over the cases."""
    return np.abs(np.asarray(values, dtype=float)).mean(axis=0)


def ranking(importance):
    """Positions of the features from most to least important. Equal
    importances keep the feature order (TH Table 3.2 order)."""
    imp = np.asarray(importance, dtype=float)
    return [int(i) for i in np.argsort(-imp, kind="stable")]


def ranks(importance):
    """Rank (1 = most important) of every feature, by `ranking`."""
    out = np.empty(len(importance), dtype=int)
    out[ranking(importance)] = np.arange(1, len(importance) + 1)
    return out


def directions(values, X):
    """Per feature: Spearman correlation between the feature's value and
    its SHAP value over the cases (> 0: high values push towards "the
    method is correct"). NaN for a feature that is constant or whose SHAP
    values are constant."""
    values, X = np.asarray(values, dtype=float), np.asarray(X, dtype=float)
    out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        if np.ptp(X[:, j]) > 0 and np.ptp(values[:, j]) > 0:
            out[j] = float(sps.spearmanr(X[:, j], values[:, j]).statistic)
    return out


def _sign(x):
    return 0 if x != x or x == 0 else (1 if x > 0 else -1)


def thesis_ranks(label, names=cdmcr.FEATURE_NAMES):
    """Thesis rank of every feature for a label: 1..19 for the features
    the figure lists, None for the 8 it merges."""
    order = THESIS_RANKING[label]
    return [order.index(n) + 1 if n in order else None for n in names]


def compare_with_thesis(label, importance, names=cdmcr.FEATURE_NAMES):
    """Rank order of a run against the thesis figure of a label.

    spearman_all    : Spearman over all features; the thesis's 8 unlisted
                      features share the average of ranks 20..27 (ties).
    spearman_listed : Spearman over the 19 listed features only (the run's
                      importances restricted to them).
    top1, top5, top10, top19 : the run's top feature and the number of the
                      thesis's top-k features among the run's top k.
    rank_of_thesis_top : the run's rank of the thesis's top feature.
    docs/IMPLEMENTATION_DECISIONS.md SH6: the plan says "Spearman" without the tie handling.
    """
    names = tuple(names)
    imp = np.asarray(importance, dtype=float)
    th = thesis_ranks(label, names)
    run = ranks(imp)
    order = [names[i] for i in ranking(imp)]
    th_order = THESIS_RANKING[label]
    rest = (N_SHOWN + 1 + len(names)) / 2.0
    th_all = np.array([rest if r is None else r for r in th], dtype=float)
    listed = [i for i, r in enumerate(th) if r is not None]
    out = {"label": label, "figure": THESIS_FIGURE[label],
           "top1_run": order[0], "top1_thesis": th_order[0],
           "top1_equal": order[0] == th_order[0],
           "rank_of_thesis_top": int(run[names.index(th_order[0])]),
           "spearman_all": float(sps.spearmanr(run, th_all).statistic),
           "spearman_listed": float(sps.spearmanr(
               -imp[listed], [th[i] for i in listed]).statistic) if len(listed) > 2
           else float("nan")}
    for k in (5, 10, N_SHOWN):
        out[f"top{k}_overlap"] = len(set(order[:k]) & set(th_order[:k]))
    return out


def direction_checks(label, direction, names=cdmcr.FEATURE_NAMES):
    """The text-stated directions of a label against the run's:
    [{feature, thesis (+1/-1), run (Spearman), agrees (None if undefined)}]."""
    names = tuple(names)
    out = []
    for (lab, feat), sign in THESIS_DIRECTION.items():
        if lab != label or feat not in names:
            continue
        d = float(direction[names.index(feat)])
        out.append({"feature": feat, "thesis": sign, "run": d,
                    "agrees": None if _sign(d) == 0 else _sign(d) == sign})
    return out


def mirror(values_a, values_b, X, top=10):
    """How far label b is the "mirror image" of label a (TH p.78: "similar
    ranking features with the opposite trends"):
      spearman_importance : Spearman of the two importance vectors;
      shap_correlation    : Pearson correlation of all SHAP values (cases x
                            features), -1 for an exact mirror;
      top / opposite      : among a's `top` features with a defined
                            direction in both, how many have opposite signs.
    """
    va, vb = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    ia, ib = mean_abs(va), mean_abs(vb)
    da, db = directions(va, X), directions(vb, X)
    idx = [i for i in ranking(ia)[:top] if _sign(da[i]) and _sign(db[i])]
    flat = np.corrcoef(va.ravel(), vb.ravel())[0, 1] if va.std() and vb.std() else np.nan
    return {"spearman_importance": float(sps.spearmanr(ia, ib).statistic),
            "shap_correlation": float(flat),
            "top": int(top), "top_defined": len(idx),
            "opposite": int(sum(_sign(da[i]) != _sign(db[i]) for i in idx))}


# ---- one analysis ----------------------------------------------------------------------

def label_rows(model, label, values, X, names=cdmcr.FEATURE_NAMES):
    """One dict per feature for the CSV: importance, rank and direction of
    the run, with the thesis rank in its own column (None = not listed)."""
    imp, run, d, th = mean_abs(values), ranks(mean_abs(values)), directions(values, X), \
        thesis_ranks(label, names)
    return [{"model": model, "label": label, "feature": n, "feature_number": j + 1,
             "mean_abs_shap": float(imp[j]), "rank_run": int(run[j]),
             "direction_run": float(d[j]), "shap_min": float(np.min(values[:, j])),
             "shap_max": float(np.max(values[:, j])), "rank_thesis": th[j],
             "direction_thesis": THESIS_DIRECTION.get((label, n)),
             "thesis_figure": THESIS_FIGURE.get(label)}
            for j, n in enumerate(names)]


def analyse(X, Y, labels=I.LABEL_ORDER, names=cdmcr.FEATURE_NAMES, models=MODELS,
            seed=0, hyper=None, n_jobs=1, background_n=50, nsamples="auto",
            with_thesis=True):
    """SHAP values of every label under every requested model.

    X, Y   : the AMP inputs (cdm.approaches.amp.amp_xy, row 1): features
             (n, 27) and 0/1 outcomes (n, q) of the methods `labels`.
    models : sub-set of MODELS.
    Returns {"values": {model: {label: (n, d) array}},
             "base": {model: {label: float}},
             "rows": [label_rows ...],
             "comparison": {model: {label: compare_with_thesis}},
             "directions": {model: {label: direction_checks}},
             "mirror_MR_DA": {model: mirror}, "acceptance": {model: {...}}}.
    `with_thesis` False (synthetic data) leaves the thesis parts out.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y)
    labels, names = tuple(labels), tuple(names)
    if Y.shape != (len(X), len(labels)) or X.shape[1] != len(names):
        raise ValueError("shapes of X, Y, labels and names do not agree")
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; expected {MODELS}")
    values, base = {m: {} for m in models}, {m: {} for m in models}
    if "xgboost_tree" in models:
        for j, lab in enumerate(labels):
            v, b = xgboost_shap(fit_xgboost(X, Y[:, j], seed, n_jobs), X)
            values["xgboost_tree"][lab], base["xgboost_tree"][lab] = v, b
    if "rf_kernel" in models:
        from joblib import Parallel, delayed
        h = dict(hyper if hyper is not None else M.hyperparameters_from_config())
        h["seed"] = int(seed)
        res = Parallel(n_jobs=min(int(n_jobs), len(labels)))(
            delayed(rf_kernel_label)(X, Y[:, j], h, background_n, nsamples, seed)
            for j in range(len(labels)))
        for lab, (v, b) in zip(labels, res):
            values["rf_kernel"][lab], base["rf_kernel"][lab] = v, b
    out = {"values": values, "base": base, "rows": [], "comparison": {}, "directions": {},
           "mirror_MR_DA": {}, "acceptance": {}, "labels": labels, "names": names}
    for m in models:
        for lab in labels:
            out["rows"] += label_rows(m, lab, values[m][lab], X, names)
        if with_thesis:
            out["comparison"][m] = {
                lab: compare_with_thesis(lab, mean_abs(values[m][lab]), names)
                for lab in labels if lab in THESIS_RANKING}
            out["directions"][m] = {
                lab: direction_checks(lab, directions(values[m][lab], X), names)
                for lab in labels}
        if "MR" in labels and "DA" in labels:
            out["mirror_MR_DA"][m] = mirror(values[m]["MR"], values[m]["DA"], X)
        if with_thesis:
            out["acceptance"][m] = acceptance(out["comparison"][m], out["mirror_MR_DA"].get(m))
    return out


def acceptance(comparison, mirror_mr_da):
    """The statements of plan §9.3 row A19 for one model, as observed
    (reported in the output, never asserted, never tuned):
      top feature for MR / DA / SP = SG_B_Amax; HAC top = CAmin;
      MR vs DA mirror image. "Mirror image" is not defined numerically in
      the plan; here: every one of MR's top-10 features with a defined
      direction has the opposite sign for DA, and the two importance
      orders correlate positively (docs/IMPLEMENTATION_DECISIONS.md SH7)."""
    tops = {lab: {"expected": feat, "run": comparison[lab]["top1_run"],
                  "rank_of_expected": comparison[lab]["rank_of_thesis_top"],
                  "met": comparison[lab]["top1_run"] == feat}
            for lab, feat in ACCEPTANCE_TOP.items() if lab in comparison}
    mir = None
    if mirror_mr_da is not None:
        mir = dict(mirror_mr_da)
        mir["met"] = bool(mir["top_defined"] > 0 and mir["opposite"] == mir["top_defined"]
                          and mir["spearman_importance"] > 0)
    return {"top_feature": tops, "mirror_MR_DA": mir,
            "all_met": bool(all(t["met"] for t in tops.values())
                            and (mir is None or mir["met"]))}


# ---- figure ---------------------------------------------------------------------------

def beeswarm_png(values, base, X, names, path, title=None, max_display=MAX_DISPLAY):
    """`shap.plots.beeswarm` of one label (plan §5.9 [INFERRED]: the plot
    type of the thesis figures), saved as a PNG. Features sorted by mean
    |SHAP|; max_display = 20 gives 19 features and "Sum of ... other
    features" as in the thesis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    X = np.asarray(X, dtype=float)
    exp = shap.Explanation(values=np.asarray(values, dtype=float),
                           base_values=np.full(len(X), float(base)), data=X,
                           feature_names=list(names))
    plt.figure()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.plots.beeswarm(exp, max_display=max_display, show=False)
    if title:
        plt.title(title, fontsize=10)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path
