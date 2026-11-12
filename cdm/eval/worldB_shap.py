"""SHAP analysis of the RCR and ACR features (step B10 of plan §9.3; plan
§5.9, §8 OQ30; TH §4.3.2 p.61, §5.3.2 p.82-85, Figs 5.8 and 5.9; MCF §4.5,
Figs 2 and 3; plan item T30, World-B part).

    UNTESTED ON REAL DATA.

What the thesis says: "The features constructed by the RCR and ACR
approaches were evaluated by assessing their influence on the outputs of
the response-based and answer-based classification models ... The
classification models were trained using the same data sets used for
evaluating the RP and AP aggregation approaches" (TH p.61). The model, the
explainer and the plotting call are not stated; the figures' x-axes run to
about ±4 (RCR) and ±3 (ACR); SHAP values of a probability output lie
within ±1 (plan §5.9).

Plan OQ30, as for the CDMCR features in step A19 (cdm/eval/
shap_analysis.py, whose functions are reused here):
  (i)  xgboost_tree     : XGBoost (defaults) on the RCR / ACR instances,
                          shap.TreeExplainer, log-odds, beeswarm;
  (ii) soft_vote_kernel : shap.KernelExplainer on P("True") of the actual
                          model, the soft-voting ensemble of RP / AP;
both on a model trained on ALL instances of one partition (the 77
sub-sets), no hold-out.

Choices beyond the plan (decision B10-1 .. B10-3): the partition (the
first seed, or the file's original assignment); instances with an undefined
feature are left out (B6-4); KernelSHAP explains a seeded sub-sample of the
instances (it costs about 10^5 ensemble predictions per instance).

Plan row B10: "Top-3 RCR features PS-based (dPSv first); top ACR dPS_OG
(real data only)". `compare_with_thesis` therefore REFUSES synthetic data:
on the synthetic stand-in the driver runs end to end and prints the two
rank orders in separate columns, without any agreement measure.
"""
import numpy as np
from scipy import stats as sps

from cdm.eval import shap_analysis as SH
from cdm.eval import worldB_thesis_values as T
from cdm.models import worldB_classifiers as WC

MODELS = ("xgboost_tree", "soft_vote_kernel")
MODEL_TITLE = {
    "xgboost_tree": "XGBoost, TreeExplainer, log-odds (plan OQ30 (i))",
    "soft_vote_kernel": "soft-voting ensemble of RP / AP, KernelExplainer, P(True) "
                        "(plan OQ30 (ii))"}
#: The model whose output unit matches the figures' axes (log-odds).
FAITHFUL_MODEL = "xgboost_tree"
#: RCR features "originated from the predicted support of v" (TH p.82).
PS_V_FEATURES = ("PSv", "dPSv", "PSvD", "PSvD_IG", "PSvD_OG")
DEFAULT_KERNEL_ROWS = 150


def defined_instances(inst):
    """(X, y) of the instances without an undefined feature."""
    keep = ~inst.undefined
    return inst.X[keep], inst.y[keep]


def kernel_rows(n, n_rows, seed=0):
    """Positions of the instances KernelSHAP explains: all if n_rows is 0
    or >= n, else a seeded sample without replacement, ascending."""
    if not n_rows or n_rows >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(int(seed)).choice(n, int(n_rows), replace=False))


def soft_vote_kernel_chunk(X, y, hyper, rows, background_n, nsamples, seed):
    """Fit the soft-voting ensemble on all of (X, y) and return its
    KernelSHAP (values, base) for `rows`. A top-level function so that
    chunks of rows can run in separate processes (the fit is deterministic,
    so every chunk explains the same model)."""
    model = WC.make_classifier("soft_vote", hyper).fit(X, y)
    return SH.kernel_shap(model, X, background_n, nsamples, seed, rows=rows)


def analyse(inst, hyper=None, models=MODELS, seed=0, n_kernel_rows=DEFAULT_KERNEL_ROWS,
            background_n=50, nsamples="auto", n_jobs=1):
    """SHAP values of the RCR (RP) or ACR (AP) features of one partition.

    Returns {model: {"values" (rows, d), "base", "rows" (positions in the
    defined instances), "importance" (mean |SHAP|), "ranking" (feature
    names, most important first), "direction" (Spearman of feature value
    and SHAP value)}} and the (X, y) used.
    """
    hyper = dict(hyper if hyper is not None else WC.hyperparameters_from_config())
    X, y = defined_instances(inst)
    names = list(inst.feature_names)
    out = {}
    for model in models:
        if model == "xgboost_tree":
            fitted = SH.fit_xgboost(X, y, seed=seed, n_jobs=1)
            values, base = SH.xgboost_shap(fitted, X)
            rows = np.arange(len(X))
        elif model == "soft_vote_kernel":
            rows = kernel_rows(len(X), n_kernel_rows, seed)
            chunks = [c for c in np.array_split(rows, max(1, int(n_jobs))) if c.size]
            if len(chunks) == 1:
                parts = [soft_vote_kernel_chunk(X, y, hyper, chunks[0], background_n,
                                                nsamples, seed)]
            else:
                from joblib import Parallel, delayed
                parts = Parallel(n_jobs=len(chunks))(
                    delayed(soft_vote_kernel_chunk)(X, y, hyper, c, background_n,
                                                    nsamples, seed) for c in chunks)
            values = np.vstack([p[0] for p in parts])
            base = float(parts[0][1])
        else:
            raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
        imp = SH.mean_abs(values)
        out[model] = {"values": values, "base": base, "rows": rows, "importance": imp,
                      "ranking": [names[i] for i in SH.ranking(imp)],
                      "direction": SH.directions(values, X[rows])}
    return out, X, y


def thesis_ranks(approach, names):
    """Rank (1 = top) of every feature in the thesis figure."""
    order = T.SHAP_RANKING[approach]
    return np.asarray([order.index(n) + 1 for n in names], dtype=int)


def compare_with_thesis(approach, importance, names, synthetic):
    """Agreement of a run's rank order with TH Fig 5.8 / 5.9 and the
    acceptance statements of plan row B10. REAL DATA ONLY."""
    if synthetic:
        raise ValueError("no comparison with the thesis's SHAP ranking is made on "
                         "synthetic data (plan §9.3 row B10: real data only)")
    names = list(names)
    run_ranks = SH.ranks(importance)
    rho = float(sps.spearmanr(run_ranks, thesis_ranks(approach, names)).statistic)
    order = [names[i] for i in SH.ranking(importance)]
    out = {"spearman_with_thesis": rho, "run_top": order[:5],
           "thesis_top": list(T.SHAP_RANKING[approach][:5])}
    if approach == "RP":
        out["acceptance"] = {
            "top_feature_is_dPSv": order[0] == "dPSv",
            "top_3_are_based_on_the_predicted_support_of_v":
                all(n in PS_V_FEATURES for n in order[:3])}
    else:
        out["acceptance"] = {"top_feature_is_dPS_OG": order[0] == "dPS_OG"}
    return out
