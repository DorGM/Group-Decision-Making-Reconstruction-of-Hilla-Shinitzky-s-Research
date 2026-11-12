"""B10 (plan §9.3): SHAP of the RCR / ACR features.

Plan row B10: acceptance on REAL DATA ONLY. These tests check the driver
end to end on the synthetic stand-in (a smoke run) and that NO comparison
with the thesis's rank order is produced for synthetic data.

    UNTESTED ON REAL DATA.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import b_small
from cdm.approaches import worldB_instances as WI
from cdm.data.worldB import loader as L
from cdm.data.worldB import partition as WP
from cdm.eval import shap_analysis as SH
from cdm.eval import worldB_shap as WS
from cdm.eval import worldB_thesis_values as T
from cdm.features import acr, rcr
from test_b6_rp import _marked

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MARK = L.SYNTHETIC_MARKER


def test_thesis_rank_orders_as_read_from_the_figures():
    # plan T30: "Top RCR: dPSv, PSvD_OG, PSvD, ConfD_IG, PSv ...; top ACR:
    # dPS_OG, AvgConf, AvgPS_OG, AvgPS_IG, dPS ..."
    assert T.SHAP_RANKING["RP"][:5] == ("dPSv", "PSvD_OG", "PSvD", "ConfD_IG", "PSv")
    assert T.SHAP_RANKING["AP"][:5] == ("dPS_OG", "AvgConf", "AvgPS_OG", "AvgPS_IG", "dPS")
    assert len(T.SHAP_RANKING["RP"]) == 14 and len(T.SHAP_RANKING["AP"]) == 10
    # TH p.82: "the top three features ... all originated from the predicted support of v"
    assert all(n in WS.PS_V_FEATURES for n in T.SHAP_RANKING["RP"][:3])
    assert list(WS.thesis_ranks("AP", acr.FEATURE_NAMES))[acr.FEATURE_NAMES.index("dPS_OG")] == 1


def test_no_comparison_with_the_thesis_on_synthetic_data():
    imp = np.arange(14, 0, -1.0)
    with pytest.raises(ValueError, match="real data only"):
        WS.compare_with_thesis("RP", imp, rcr.FEATURE_NAMES, synthetic=True)


def test_comparison_for_real_data_measures_rank_agreement():
    names = list(rcr.FEATURE_NAMES)
    order = T.SHAP_RANKING["RP"]
    imp = np.asarray([14.0 - order.index(n) for n in names])      # exactly the thesis order
    c = WS.compare_with_thesis("RP", imp, names, synthetic=False)
    assert c["spearman_with_thesis"] == pytest.approx(1.0)
    assert c["acceptance"] == {"top_feature_is_dPSv": True,
                               "top_3_are_based_on_the_predicted_support_of_v": True}
    c = WS.compare_with_thesis("RP", imp[::-1].copy(), names, synthetic=False)
    assert c["spearman_with_thesis"] < 1.0
    names = list(acr.FEATURE_NAMES)
    imp = np.asarray([10.0 - T.SHAP_RANKING["AP"].index(n) for n in names])
    c = WS.compare_with_thesis("AP", imp, names, synthetic=False)
    assert c["acceptance"] == {"top_feature_is_dPS_OG": True}


def test_kernel_rows_are_a_seeded_sample():
    assert WS.kernel_rows(10, 0).tolist() == list(range(10))
    assert WS.kernel_rows(10, 50).tolist() == list(range(10))
    a, b = WS.kernel_rows(1000, 20, seed=0), WS.kernel_rows(1000, 20, seed=0)
    assert a.tolist() == b.tolist() == sorted(set(a.tolist())) and len(a) == 20
    assert WS.kernel_rows(1000, 20, seed=1).tolist() != a.tolist()


def test_xgboost_tree_shap_is_additive_in_log_odds():
    data = b_small.small_synthetic()
    inst = WI.build_instances(WP.subset_cases(data, seed=0), "AP")
    out, X, y = WS.analyse(inst, models=("xgboost_tree",))
    r = out["xgboost_tree"]
    assert r["values"].shape == X.shape == (inst.n, 10)
    model = SH.fit_xgboost(X, y, seed=0)
    margin = model.predict(X, output_margin=True)
    assert np.allclose(r["values"].sum(axis=1) + r["base"], margin, atol=1e-3)
    assert sorted(r["ranking"]) == sorted(acr.FEATURE_NAMES)
    with pytest.raises(ValueError):
        WS.analyse(inst, models=("lime",))


def _run(tmp_path, data, extra=()):
    import run_b10_shap as B10
    cfg = b_small.write_config(tmp_path, data)
    assert B10.main(["--config", str(cfg), "--runs-root", str(tmp_path / "runs"),
                     "--n-jobs", "2", "--kernel-rows", "6", "--kernel-background", "10",
                     "--kernel-nsamples", "60", *extra]) == 0


def test_b10_driver_smoke_run_on_synthetic_data_makes_no_comparison(tmp_path):
    _run(tmp_path, b_small.small_synthetic())
    out = tmp_path / "runs" / "worldB_synthetic" / "b10_shap"
    names = sorted(p.name for p in out.iterdir())
    assert names == ["b10_shap.log", "b10_soft_vote_kernel_AP.png",
                     "b10_soft_vote_kernel_RP.png", "b10_xgboost_tree_AP.png",
                     "b10_xgboost_tree_RP.png", "shap_mean_abs.csv", "shap_rcr_acr.md",
                     "shap_summary.json", "shap_values.npz"]
    for p in out.iterdir():
        if p.suffix in (".json", ".csv", ".md", ".log"):
            assert _marked(p), p.name
    assert str(np.load(out / "shap_values.npz")["marker"]) == MARK
    doc = json.loads((out / "shap_summary.json").read_text(encoding="utf-8"))
    assert "SMOKE RUN" in doc["note"]
    text = json.dumps(doc)
    assert "comparison_with_thesis" not in text and "spearman" not in text.lower()
    md = (out / "shap_rcr_acr.md").read_text(encoding="utf-8")
    assert "smoke run" in md.lower() and "Spearman" not in md and "acceptance (" not in md
    assert "run (synthetic): xgboost_tree" in md and "thesis: figure order" in md
    for approach, d in (("RP", 14), ("AP", 10)):
        cell = doc["run"][approach]
        assert set(cell["models"]) == {"xgboost_tree", "soft_vote_kernel"}
        assert cell["models"]["soft_vote_kernel"]["n_explained"] == 6
        assert len(cell["models"]["xgboost_tree"]["run_ranking"]) == d
        assert cell["thesis_ranking"] == list(T.SHAP_RANKING[approach])
    rows = (out / "shap_mean_abs.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 1 + 2 * (14 + 10)


def test_b10_driver_on_an_unmarked_file_adds_the_comparison(tmp_path):
    _run(tmp_path, b_small.small_unmarked(), extra=("--models", "xgboost_tree"))
    out = tmp_path / "runs" / "worldB_real" / "b10_shap"
    doc = json.loads((out / "shap_summary.json").read_text(encoding="utf-8"))
    assert list(doc)[0] == "_status" and "note" not in doc
    c = doc["run"]["RP"]["models"]["xgboost_tree"]["comparison_with_thesis"]
    assert -1.0 <= c["spearman_with_thesis"] <= 1.0 and "top_feature_is_dPSv" in c["acceptance"]
    md = (out / "shap_rcr_acr.md").read_text(encoding="utf-8")
    assert MARK not in md and "Spearman with the thesis order" in md
    assert "UNTESTED ON REAL DATA" in md
