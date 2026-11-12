"""B8 (plan §9.3): RP / AP with Bag-KNN and SVM, shape of TH Tables A.7 and
A.8.

    UNTESTED ON REAL DATA (synthetic stand-in only; plan row B8's
    acceptance "SVM ≈ .43–.45 with ~91% ties" is for real data and is not
    assessed).
"""
import json
import sys
from pathlib import Path

import pytest

import b_small
from cdm.approaches import rp
from cdm.data.worldB import loader as L
from cdm.eval import worldB_ml as ML
from cdm.eval import worldB_thesis_values as T
from cdm.models import worldB_classifiers as WC
from test_b6_rp import _marked

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MARK = L.SYNTHETIC_MARKER


def test_bag_knn_and_svm_are_the_library_defaults_of_oq35():
    bag = WC.make_classifier("bag_knn")
    assert type(bag).__name__ == "BaggingClassifier"
    assert type(bag.estimator).__name__ == "KNeighborsClassifier"
    assert bag.estimator.n_neighbors == 5 and bag.n_estimators == 10
    assert bag.max_samples in (1.0, None) and bag.bootstrap is True
    svm = WC.make_classifier("svm")
    assert type(svm).__name__ == "SVC" and svm.probability is True
    assert svm.kernel == "rbf" and svm.C == 1.0 and svm.gamma == "scale"
    assert svm.class_weight is None                               # OQ33: no re-weighting
    assert WC.THESIS_LABEL == {"soft_vote": "Vote soft", "bag_knn": "Bag KNN", "svm": "SVM"}


def test_tables_a7_and_a8_as_printed():
    assert tuple(T.TABLE_A_7) == T.A7_CLASSIFIER_ORDER == ("soft_vote", "bag_knn", "svm")
    assert all(tuple(t) == rp.STRATEGY_NAMES for t in T.TABLE_A_7.values())
    assert sum(len(t) for t in T.TABLE_A_7.values()) == 27
    assert T.TABLE_A_7["bag_knn"]["avgp"] == (56, 0)
    assert T.TABLE_A_7["bag_knn"]["maj+maxp"] == (49, 16)
    assert T.TABLE_A_7["bag_knn"]["prop+wm"] == (52, 5)
    assert {v[1] for k, v in T.TABLE_A_7["svm"].items() if "+" in k} == {91}
    assert {v[0] for v in T.TABLE_A_7["svm"].values()} == {43, 45}
    assert T.TABLE_A_8 == {"bag_knn": 44, "svm": 43, "soft_vote": 55}
    # the soft-vote row of A.8 is the AP headline (T23)
    assert round(100 * T.T23_RATE["AP"]) == T.TABLE_A_8["soft_vote"]


@pytest.mark.parametrize("clf", ["bag_knn", "svm"])
def test_alternatives_run_end_to_end_for_rp_and_ap(clf):
    data = b_small.small_synthetic()
    opts = ML.options_from_config()
    out = ML.run_partition(data, 0, "random", "RP", clf, opts, keep_predictions=False)
    assert set(out["summary"]) == set(rp.STRATEGY_NAMES) and "predictions" not in out
    # AP: for SVC the three-case rule is compared with argmax, not asserted
    out = ML.run_partition(data, 0, "random", "AP", clf, opts, keep_predictions=False)
    assert out["summary"]["three_case_differs_from_argmax"] >= 0
    if clf == "bag_knn":
        assert out["summary"]["three_case_differs_from_argmax"] == 0
        assert out["predictions_log"]["label_false_but_p_true_above_half"] == 0


def test_b8_driver_tables_and_markers(tmp_path):
    import run_b6_rp as B6
    import run_b7_ap as B7
    import run_b8_alt as B8
    cfg = b_small.write_config(tmp_path, b_small.small_synthetic())
    common = ["--config", str(cfg), "--runs-root", str(tmp_path / "runs"), "--seeds", "1",
              "--n-jobs", "1"]
    assert B8.main(common) == 0
    out = tmp_path / "runs" / "worldB_synthetic" / "b8_alt_seeds1"
    md = (out / "alt_tables_a7_a8.md").read_text(encoding="utf-8")
    assert "| Vote soft | not run |" in md                       # B6 / B7 not run yet
    assert B6.main(common) == 0 and B7.main(common) == 0 and B8.main(common) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["alt_per_seed.csv", "alt_summary.json", "alt_tables_a7_a8.md",
                     "ap_outcomes_bag_knn.json", "ap_outcomes_svm.json",
                     "ap_predictions_bag_knn.csv", "ap_predictions_svm.csv", "b8_alt.log",
                     "rp_outcomes_bag_knn.json", "rp_outcomes_svm.json",
                     "rp_predictions_bag_knn.csv", "rp_predictions_svm.csv"]
    for p in out.iterdir():
        assert _marked(p), p.name
    md = (out / "alt_tables_a7_a8.md").read_text(encoding="utf-8")
    assert "not run" not in md
    a7, a8 = md.split("### AP: classifiers")
    for label in ("Vote soft", "Bag KNN", "SVM"):
        assert sum(1 for ln in a7.splitlines() if ln.startswith(f"| {label} |")) == 9
        assert sum(1 for ln in a8.splitlines() if ln.startswith(f"| {label} |")) == 1
    assert "run (synthetic):" in md and "thesis:" in md
    doc = json.loads((out / "alt_summary.json").read_text(encoding="utf-8"))
    assert set(doc["run"]["RP"]) == set(doc["run"]["AP"]) == {"soft_vote", "bag_knn", "svm"}
    assert doc["thesis"]["table_a_8"] == {"bag_knn": 44, "svm": 43, "soft_vote": 55}
    rows = (out / "alt_per_seed.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 1 + 3 * (9 + 1)
