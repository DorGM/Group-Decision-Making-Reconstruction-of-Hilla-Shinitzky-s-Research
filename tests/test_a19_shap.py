"""Step A19: SHAP values of the CDMCR features (TH Figs 5.4-5.7, A.1; T30).

SYNTHETIC DATA ONLY: random features with a planted signal, and the driver
run with `--smoke`. No build is read.

Tested:
  * the thesis rank orders typed from the figures: 19 distinct CDMCR
    features per label, the tops named in plan §9.3 row A19 and in the
    thesis text, the T30 prefix of the plan;
  * XGBoost + TreeExplainer: values + base = the model's log-odds; a
    planted feature comes first with the planted direction;
  * the label's forest is the BR+RF base classifier; KernelSHAP: values +
    base = P(label = 1), same seed -> same values;
  * ranks, directions, the comparison with the thesis, the mirror measure;
  * the driver: smoke run (files, separate run / thesis columns, marked
    NOT A RESULT, nothing copied), refusal of an unlisted build.
"""
import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from cdm.config import CODE_ROOT
from cdm.eval import shap_analysis as SH
from cdm.features import cdmcr
from cdm.models import multilabel as M

SCRIPTS = CODE_ROOT / "scripts"
NAMES = cdmcr.FEATURE_NAMES


def run_script(*args):
    return subprocess.run([sys.executable, str(SCRIPTS / "run_a19_shap.py"), *args],
                          capture_output=True, text=True, cwd=str(SCRIPTS))


@pytest.fixture(scope="module")
def planted():
    """300 cases, 6 features; y depends on feature 2 (up) and a 0/1 flag,
    feature 4 (down)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    X[:, 4] = rng.integers(0, 2, 300)
    logit = 3.0 * X[:, 2] - 1.5 * X[:, 4]
    y = (logit + rng.normal(scale=0.3, size=300) > 0).astype(int)
    return X, y


# ---- the thesis side --------------------------------------------------------------------

def test_thesis_rankings_as_read_from_the_figures():
    assert set(SH.THESIS_RANKING) == set(SH.LABELS_REPORT_ORDER) == set(SH.THESIS_FIGURE)
    for lab, order in SH.THESIS_RANKING.items():
        assert len(order) == SH.N_SHOWN == 19 and len(set(order)) == 19, lab
        assert set(order) <= set(NAMES), lab
        assert len(NAMES) - len(order) == 8          # "Sum of 8 other features"
    # plan §9.3 row A19 / TH p.78-79
    for lab in ("MR", "DA", "SP", "WC"):
        assert SH.THESIS_RANKING[lab][0] == "SG_B_Amax"
    assert SH.THESIS_RANKING["HAC"][:2] == ("CAmin", "B_Amax_MaxCa")
    assert SH.THESIS_RANKING["HAC"].index("SG_B_Amax") == 3
    # plan §1.1 T30: "top CDMCR-MR: SG_B_Amax, SG_D_MaxES_MinES, MaxPSa, AvgPSv,
    # D_Smax_Smin, CAmin"
    assert SH.THESIS_RANKING["MR"][:6] == ("SG_B_Amax", "SG_D_MaxES_MinES", "MaxPSa",
                                           "AvgPSV", "D_Smax_Smin", "CAmin")
    # ES, VarS, D_S_Uniform are in no figure
    for f in ("ES", "VarS", "D_S_Uniform"):
        assert all(f not in order for order in SH.THESIS_RANKING.values())
    assert SH.ACCEPTANCE_TOP == {"MR": "SG_B_Amax", "DA": "SG_B_Amax", "SP": "SG_B_Amax",
                                 "HAC": "CAmin"}
    for (lab, feat), sign in SH.THESIS_DIRECTION.items():
        assert lab in SH.THESIS_RANKING and feat in NAMES and sign in (-1, 1)
    assert SH.THESIS_DIRECTION[("MR", "SG_B_Amax")] == -SH.THESIS_DIRECTION[("DA", "SG_B_Amax")]


def test_thesis_ranks_and_oq30():
    r = SH.thesis_ranks("HAC")
    assert r[NAMES.index("CAmin")] == 1 and r[NAMES.index("ES")] is None
    assert sorted(x for x in r if x is not None) == list(range(1, 20))
    assert SH.oq30_from_config()["trained_on"] == "all_data"
    bad = {"open_questions": {"OQ30": {"default": {"faithful": "x"}}}}
    with pytest.raises(NotImplementedError):
        SH.oq30_from_config(bad)


# ---- XGBoost + TreeExplainer --------------------------------------------------------------

def test_xgboost_shap_is_additive_in_log_odds(planted):
    X, y = planted
    model = SH.fit_xgboost(X, y, seed=0)
    values, base = SH.xgboost_shap(model, X)
    assert values.shape == X.shape
    margin = model.predict(X, output_margin=True)
    assert np.allclose(values.sum(axis=1) + base, margin, atol=1e-3)
    assert np.abs(values).max() > 1                      # log-odds, not probabilities
    imp = SH.mean_abs(values)
    assert SH.ranking(imp)[:2] == [2, 4]
    d = SH.directions(values, X)
    assert d[2] > 0.8 and d[4] < -0.8
    v2, b2 = SH.xgboost_shap(SH.fit_xgboost(X, y, seed=0), X)
    assert np.array_equal(values, v2) and base == b2


def test_models_refuse_bad_input(planted):
    X, y = planted
    with pytest.raises(ValueError):
        SH.fit_xgboost(X, np.ones(len(X), dtype=int))
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        SH.fit_rf(Xn, y)


# ---- the label's forest + KernelExplainer -----------------------------------------------

def test_rf_is_the_br_rf_base_classifier(planted):
    X, y = planted
    hyper = dict(M.hyperparameters_from_config(), rf_n_estimators=10)
    rf = SH.fit_rf(X, y, hyper)
    ref = M.make_base("RF", **hyper).fit(X, y)
    assert type(rf) is type(ref) and rf.n_estimators == 10
    assert np.array_equal(rf.predict_proba(X), ref.predict_proba(X))
    # BR+RF gives the same per-label probability as the stand-alone forest
    Y = np.column_stack([y, 1 - y])
    br = M.make_amp_model("BR", "RF", labels=("a", "b"), **hyper).fit(X, Y)
    assert np.allclose(np.asarray(br.predict_proba(X))[:, 0], rf.predict_proba(X)[:, 1])


def test_kernel_shap_is_additive_and_seeded(planted):
    X, y = planted
    X, y = X[:60], y[:60]
    rf = SH.fit_rf(X, y, dict(M.hyperparameters_from_config(), rf_n_estimators=10))
    state = np.random.get_state()[1][:5].copy()
    values, base = SH.kernel_shap(rf, X, background_n=20, nsamples=200, seed=3)
    assert np.array_equal(np.random.get_state()[1][:5], state)   # global state restored
    assert values.shape == X.shape
    p1 = rf.predict_proba(X)[:, 1]
    assert np.allclose(values.sum(axis=1) + base, p1, atol=1e-6)
    assert np.abs(values).max() <= 1
    again, base2 = SH.kernel_shap(rf, X, background_n=20, nsamples=200, seed=3)
    assert np.allclose(values, again) and base == base2
    part, _ = SH.kernel_shap(rf, X, background_n=20, nsamples=200, seed=3, rows=[4, 7])
    assert part.shape == (2, X.shape[1])
    assert SH.ranking(SH.mean_abs(values))[0] == 2


# ---- ranks and comparisons ----------------------------------------------------------------

def test_ranking_ties_keep_the_feature_order():
    imp = [0.2, 0.5, 0.2, 0.0, 0.5]
    assert SH.ranking(imp) == [1, 4, 0, 2, 3]
    assert list(SH.ranks(imp)) == [3, 1, 4, 5, 2]


def test_directions_undefined_for_constant_columns():
    X = np.column_stack([np.arange(10.0), np.ones(10), np.arange(10.0)])
    v = np.column_stack([np.arange(10.0), np.arange(10.0), np.zeros(10)])
    d = SH.directions(v, X)
    assert d[0] == pytest.approx(1.0) and np.isnan(d[1]) and np.isnan(d[2])


def test_compare_with_thesis_on_the_thesis_order_itself():
    for lab, order in SH.THESIS_RANKING.items():
        imp = np.array([float(20 - order.index(n)) if n in order else 0.0 for n in NAMES])
        c = SH.compare_with_thesis(lab, imp)
        assert c["top1_equal"] and c["rank_of_thesis_top"] == 1
        assert c["spearman_listed"] == pytest.approx(1.0)
        assert c["spearman_all"] > 0.98                      # ties among the 8 unlisted
        assert (c["top5_overlap"], c["top10_overlap"], c["top19_overlap"]) == (5, 10, 19)
    rev = np.array([float(order.index(n)) if n in order else 30.0 for n in NAMES])
    c = SH.compare_with_thesis(lab, rev)
    assert not c["top1_equal"] and c["spearman_listed"] == pytest.approx(-1.0)
    assert c["rank_of_thesis_top"] == 27


def test_direction_checks():
    d = np.zeros(len(NAMES))
    d[NAMES.index("SG_B_Amax")] = -0.4
    d[NAMES.index("CAmin")] = 0.2
    d[NAMES.index("D_Smax_Smin")] = np.nan
    rows = {r["feature"]: r for r in SH.direction_checks("MR", d)}
    assert set(rows) == {"SG_B_Amax", "AvgPSV", "D_Smax_Smin", "CAmin"}
    assert rows["SG_B_Amax"]["agrees"] is True and rows["CAmin"]["agrees"] is False
    assert rows["D_Smax_Smin"]["agrees"] is None and rows["AvgPSV"]["agrees"] is None


def test_mirror_and_acceptance(planted):
    X, y = planted
    Y = np.column_stack([y, 1 - y])                       # an exact mirror pair
    res = SH.analyse(X, Y, labels=("MR", "DA"), names=tuple("abcdef"),
                     models=("xgboost_tree",), with_thesis=False)
    m = res["mirror_MR_DA"]["xgboost_tree"]
    assert m["shap_correlation"] == pytest.approx(-1.0, abs=1e-3)
    assert m["spearman_importance"] == pytest.approx(1.0)
    assert m["opposite"] == m["top_defined"] > 0
    assert res["comparison"] == {} and res["acceptance"] == {}
    assert len(res["rows"]) == 2 * 6
    assert {r["rank_thesis"] for r in res["rows"]} == {None}
    same = SH.mirror(res["values"]["xgboost_tree"]["MR"], res["values"]["xgboost_tree"]["MR"], X)
    assert same["opposite"] == 0 and same["shap_correlation"] == pytest.approx(1.0)
    comp = {lab: {"top1_run": top, "rank_of_thesis_top": 1} for lab, top in
            SH.ACCEPTANCE_TOP.items()}
    assert SH.acceptance(comp, dict(m))["all_met"]
    comp["HAC"]["top1_run"] = "N"
    acc = SH.acceptance(comp, dict(m))
    assert not acc["all_met"] and not acc["top_feature"]["HAC"]["met"]
    assert not SH.acceptance({k: v for k, v in comp.items() if k != "HAC"}, same)["all_met"]


def test_analyse_checks_shapes(planted):
    X, y = planted
    with pytest.raises(ValueError):
        SH.analyse(X, y.reshape(-1, 1), labels=("MR",), names=tuple("abc"))
    with pytest.raises(ValueError):
        SH.analyse(X, y.reshape(-1, 1), labels=("MR",), names=tuple("abcdef"), models=("x",))


def test_beeswarm_png(planted, tmp_path):
    X, y = planted
    values, base = SH.xgboost_shap(SH.fit_xgboost(X, y), X)
    path = SH.beeswarm_png(values, base, X, tuple("abcdef"), tmp_path / "b.png", title="t")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---- the driver ---------------------------------------------------------------------------

def test_driver_smoke(tmp_path):
    p = run_script("--smoke", "60", "--n-jobs", "2", "--kernel-nsamples", "200",
                   "--kernel-background", "15", "--runs-root", str(tmp_path),
                   "--results-root", str(tmp_path / "results"))
    assert p.returncode == 0, p.stderr
    out = tmp_path / "_smoke" / "a19"
    doc = json.loads((out / "a19_summary.json").read_text())
    assert doc["complete"] and doc["header"]["data"]["kind"].startswith("SYNTHETIC")
    assert doc["models"] == list(SH.MODELS) and not doc["thesis_columns_are_comparable"]
    assert set(doc["acceptance"]) == set(SH.MODELS)
    for m in SH.MODELS:
        for lab in SH.LABELS_REPORT_ORDER:
            assert (out / f"a19_{m}_{lab}.png").stat().st_size > 1000
    with open(out / "a19_mean_abs_shap.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 * 5 * 27
    assert {"rank_run", "rank_thesis", "mean_abs_shap"} <= set(rows[0])
    for m in SH.MODELS:
        for lab in SH.LABELS_REPORT_ORDER:
            mine = [r for r in rows if r["model"] == m and r["label"] == lab]
            assert sorted(int(r["rank_run"]) for r in mine) == list(range(1, 28))
            assert sorted(int(r["rank_thesis"]) for r in mine if r["rank_thesis"]) \
                == list(range(1, 20))
    text = (out / "a19_shap.md").read_text()
    assert "NOT A RESULT" in text and "| top feature: run | thesis |" in text
    assert "## MR label (TH Fig. 5.4 (p.80))" in text
    z = np.load(out / "a19_shap_values.npz")
    assert z["xgboost_tree__MR"].shape == (60, 27) == z["rf_kernel__DA"].shape
    assert json.loads((out / "status.json").read_text())["state"] == "finished"
    assert not (tmp_path / "results").exists()


def test_driver_refuses_unlisted_build(tmp_path):
    p = run_script("--build", "h1209_nba_games", "--runs-root", str(tmp_path),
                   "--results-root", "")
    assert p.returncode != 0 and "GateError" in p.stderr
    assert not list(tmp_path.rglob("*.png"))
