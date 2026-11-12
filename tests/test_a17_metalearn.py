"""Step A17: classic meta-learning (TH Table 5.8).

SYNTHETIC DATA ONLY: small synthetic instance tables and the driver run
with `--smoke`. No build is read.

Tested:
  * the five rows of TH Table 5.8 (methods per row, printed rates);
  * the instances: only the 0/1 codes of the kept methods' answers, in
    the thesis order; f_DA recomputed over the methods kept (row 4:
    DA = "not MR");
  * the leave-one-out loop against sklearn's own LOO; the 50/50 rule;
  * rows 1 and 5 give the same prediction for every case (f_DA is a
    function of the other four; plan §5.5);
  * the exported rules; the OQ21 switch; the driver.
"""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from cdm.config import CODE_ROOT, load_config
from cdm.eval import metalearn as ML
from cdm.report import thesis_values as T

SCRIPTS = CODE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="module")
def inst():
    import _common as C
    return C.smoke_instances(90)[0]


def test_rows_of_table_5_8():
    # plan §5.5 (rendered TH p.73): methods and printed success rate
    want = {1: (("MR", "HAC", "WC", "SP", "DA"), 0.713),
            2: (("MR", "SP", "DA"), 0.708),
            3: (("MR", "HAC", "WC", "DA"), 0.695),
            4: (("MR", "DA"), 0.687),
            5: (("MR", "HAC", "WC", "SP"), 0.713)}
    from cdm.approaches import instances as I
    for row, (methods, rate) in want.items():
        assert I.methods_kept(ML.row_config(row)) == methods
        assert ML.thesis_rate(row) == rate
    assert sorted(ML.ROWS) == [1, 2, 3, 4, 5] == sorted(T.TH_TABLE_5_8)
    with pytest.raises(ValueError):
        ML.row_config(6)


def test_instances_hold_only_the_method_answers(inst):
    X, y, methods, rules = ML.metalearn_xy(inst, 1)
    assert X.shape == (90, 5) and methods == ("MR", "HAC", "WC", "SP", "DA")
    assert set(np.unique(X)) <= {0.0, 1.0} and set(np.unique(y)) <= {0, 1}
    for j, m in enumerate(methods):          # code 1 = the first answer of the case
        want = [int(a == ans[0]) for a, ans in zip(inst.method_answers(m, rules), inst.answers)]
        assert X[:, j].tolist() == want
    assert y.tolist() == [int(c == a[0]) for c, a in zip(inst.correct, inst.answers)]
    # a method's column is right exactly when its O_AG is 1
    assert np.array_equal((X[:, 0] == y).astype(int), inst.outcomes("MR"))


def test_da_is_recomputed_over_the_methods_kept(inst):
    X4, _, methods, rules = ML.metalearn_xy(inst, 4)
    assert methods == ("MR", "DA") and rules == ("MR",)
    assert np.array_equal(X4[:, 1], 1 - X4[:, 0])           # DA = "not MR" (TH p.48)
    X2, _, m2, r2 = ML.metalearn_xy(inst, 2)
    assert m2 == ("MR", "SP", "DA") and r2 == ("MR", "SP")
    X1, _, _, _ = ML.metalearn_xy(inst, 1)
    assert not np.array_equal(X1[:, 4], X4[:, 1])           # DA differs between rows
    X5, _, m5, r5 = ML.metalearn_xy(inst, 5)
    assert m5 == ("MR", "HAC", "WC", "SP") and r5 == ("MR", "WC", "HAC", "SP")
    assert np.array_equal(X5, X1[:, :4])


def test_loo_equals_sklearn_loo(inst):
    X, y, _, _ = ML.metalearn_xy(inst, 3)
    pred, p1 = ML.loo_predictions(X, y, random_state=0)
    want = cross_val_predict(DecisionTreeClassifier(random_state=0), X, y, cv=LeaveOneOut())
    assert np.array_equal(pred, want)
    assert np.all((p1 > 0.5) == (pred == 1))                # 0.5 -> class 0
    res = ML.run_row(inst, 3)
    assert np.array_equal(res.predicted, pred) and res.n_success == int((pred == y).sum())
    assert res.approach == "DAP" and set(res.techniques) == {"DT"}
    assert res.labels == ("MR", "HAC", "WC", "DA")
    assert res.meta["model"].startswith("DecisionTreeClassifier")


def test_half_leaf_returns_code_0_and_is_counted():
    X = np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
    y = np.array([1, 0, 1, 0, 0, 1, 1])
    pred, p1 = ML.loo_predictions(X, y)
    # case 0 left out: the x = 1 leaf holds one 0 and one 1 -> P = 0.5 -> code 0
    assert p1[0] == 0.5 and pred[0] == 0
    assert p1[1] == 1.0 and pred[1] == 1
    with pytest.raises(ValueError):
        ML.loo_predictions(X[:1], y[:1])


def test_rows_1_and_5_coincide(inst):
    r1, r5 = ML.run_row(inst, 1), ML.run_row(inst, 5)
    assert np.array_equal(r1.predicted, r5.predicted)
    e = ML.expectations({1: r1, 5: r5, 4: ML.run_row(inst, 4)}, inst)
    assert e["rows_1_5_equal"]["same_predictions"] is True
    assert e["row4_vs_MR"]["MR"] == int(inst.outcomes("MR").sum())
    assert e["best_vs_SP"]["SP"] == int(inst.outcomes("SP").sum())
    assert ML.expectations({}, inst) == {"row4_vs_MR": None, "rows_1_5_equal": None,
                                         "best_vs_SP": None}


def test_row_4_tree_is_a_function_of_mr(inst):
    rules = ML.tree_rules(inst, 4)
    assert rules["features"] == ["f_MR", "f_DA"]
    assert len(rules["patterns"]) == 2                       # (a1, a2) and (a2, a1) only
    assert sum(p["n"] for p in rules["patterns"]) == len(inst)
    assert rules["n_leaves"] <= 2


def test_tree_rules_export(inst):
    rules = ML.tree_rules(inst, 1)
    assert "f_" in rules["text"] and "class:" in rules["text"]
    assert sum(p["n"] for p in rules["patterns"]) == len(inst)
    X, y, _, _ = ML.metalearn_xy(inst, 1)
    for p in rules["patterns"]:
        codes = np.array([p["codes"][f] for f in rules["features"]], dtype=float)
        mask = np.all(X == codes, axis=1)
        assert p["n"] == mask.sum() and p["n_correct_is_a1"] == y[mask].sum()
        # the full tree returns the majority answer of the pattern (ties -> a2)
        assert p["predicted"] == ("a1" if 2 * p["n_correct_is_a1"] > p["n"] else "a2")
    text = ML.format_patterns(rules)
    assert text.count("\n") == len(rules["patterns"])


def test_oq21_switch():
    assert ML.settings_from_config() == {"protocol": "loo", "random_state": 0}
    cfg = copy.deepcopy(load_config())
    cfg["open_questions"]["OQ21"]["default"]["protocol"] = "10fold"
    with pytest.raises(NotImplementedError):
        ML.settings_from_config(cfg)
    cfg = copy.deepcopy(load_config())
    cfg["open_questions"]["OQ21"]["default"]["model"] = "RandomForest"
    with pytest.raises(NotImplementedError):
        ML.settings_from_config(cfg)


def test_non_binary_cases_are_refused():
    from cdm.approaches import instances as I
    from cdm.schema import Case, Response
    rs = tuple(Response(v=v, c=0.5, ps=(0.4, 0.3, 0.3)) for v in "aabbc" * 4)
    table = I.build_instances([Case(case_id=k, answers=("a", "b", "c"), correct="a",
                                    responses=rs) for k in range(3)])
    with pytest.raises(NotImplementedError):
        ML.metalearn_xy(table, 1)


# ---- the driver -----------------------------------------------------------------------------

def run_script(*args):
    return subprocess.run([sys.executable, str(SCRIPTS / "run_a17_metalearn.py"), *args],
                          capture_output=True, text=True, cwd=str(SCRIPTS))


def test_driver_smoke(tmp_path):
    p = run_script("--smoke", "60", "--runs-root", str(tmp_path),
                   "--results-root", str(tmp_path / "results"))
    assert p.returncode == 0, p.stderr
    out = tmp_path / "_smoke" / "a17"
    doc = json.loads((out / "a17_summary.json").read_text())
    assert doc["complete"] and sorted(doc["rows"]) == ["1", "2", "3", "4", "5"]
    assert doc["rows"]["1"]["thesis_rate"] is None                     # synthetic data
    assert doc["rows"]["4"]["features"] == ["f_MR", "f_DA"]
    assert doc["expected_relations"]["rows_1_5_equal"]["same_predictions"] is True
    assert set(doc["rows"]["2"]["domains"]) == {"General", "NBA", "NFL", "Politics", "all"}
    text = (out / "a17_metalearn.md").read_text()
    assert "NOT A RESULT" in text and "## Table 5.8" in text
    rules = (out / "a17_tree_rules.txt").read_text()
    assert rules.count("=== Row") == 5 and "sklearn export_text" in rules
    assert (out / "a17_row5_cases.csv").is_file()
    assert not (tmp_path / "results").exists()


def test_driver_refuses_unlisted_build(tmp_path):
    p = run_script("--build", "h1209_nba_games", "--runs-root", str(tmp_path),
                   "--results-root", "")
    assert p.returncode != 0 and "GateError" in p.stderr
    assert not (tmp_path / "h1209_nba_games" / "a17" / "a17_summary.json").exists()
