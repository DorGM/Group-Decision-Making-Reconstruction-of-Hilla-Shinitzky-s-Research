"""B7 (plan §9.3): the AP pipeline — ACR instances, the three-case rule of
TH p.41, its equality with argmax P(True) (plan §4.4), the T23 table.

    UNTESTED ON REAL DATA (synthetic stand-in and hand-made numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import b_small
from cdm.approaches import ap
from cdm.approaches import worldB_instances as WI
from cdm.data.worldB import loader as L
from cdm.data.worldB import partition as WP
from cdm.eval import worldB_logo as LG
from cdm.eval import worldB_ml as ML
from cdm.eval import worldB_thesis_values as T
from cdm.features import acr
from test_b6_rp import _case, _marked, _subsets

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MARK = L.SYNTHETIC_MARKER


@pytest.fixture(scope="module")
def small():
    data = b_small.small_synthetic()
    return data, WP.subset_cases(data, seed=0)


# ---- instances (TH p.41, p.56: one per answer with at least one vote) -------------------

def test_ap_instances_are_the_acr_features_of_each_voted_answer(small):
    _, subsets = small
    inst = WI.build_instances(subsets, "AP")
    assert inst.feature_names == acr.FEATURE_NAMES and inst.X.shape[1] == 10
    n = 0
    for g, case in enumerate(subsets.cases):
        want = acr.acr_features(case.responses, case.answers)
        rows = inst.rows_of(g)
        assert rows.size == len(want) <= len(case.answers)       # TH p.56: |AI| <= |A_P|
        for i, (a, feats) in zip(rows, want):
            assert list(inst.X[i]) == [feats[k] for k in acr.FEATURE_NAMES]
            assert case.answers[inst.answer_index[i]] == a
            assert inst.y[i] == int(a == case.correct)           # "whether the answer is correct"
        # at most one correct instance; none if a* has no vote (B2-3)
        assert inst.y[rows].sum() == int(any(r.v == case.correct for r in case.responses))
        n += len(want)
    assert inst.n == n


# ---- the three cases (TH p.41) -----------------------------------------------------------

def test_case_1_only_one_answer_classified_true():
    pos, case, tied = ap.three_case([0, 1, 0], [0.2, 0.6, 0.4], [0.8, 0.4, 0.6])
    assert (pos, case, tied) == (1, "one_true", (1,))


def test_case_2_several_true_highest_probability():
    pos, case, tied = ap.three_case([1, 0, 1], [0.7, 0.3, 0.9], [0.3, 0.7, 0.1])
    assert (pos, case, tied) == (2, "several_true", (2,))


def test_case_3_all_false_lowest_probability_of_false():
    pos, case, tied = ap.three_case([0, 0, 0], [0.1, 0.45, 0.3], [0.9, 0.55, 0.7])
    assert (pos, case, tied) == (1, "none_true", (1,))
    # literal: P(False) is used, not 1 - P(True)
    pos, _, _ = ap.three_case([0, 0, 0], [0.1, 0.45, 0.3], [0.9, 0.8, 0.7])
    assert pos == 2


def test_ties_go_to_canonical_order_and_are_reported():
    pos, case, tied = ap.three_case([1, 1, 0], [0.8, 0.8, 0.1], [0.2, 0.2, 0.9])
    assert (pos, case, tied) == (0, "several_true", (0, 1))
    pos, case, tied = ap.three_case([0, 0, 0], [0.2, 0.3, 0.3], [0.8, 0.7, 0.7])
    assert (pos, case, tied) == (1, "none_true", (1, 2))
    assert ap.argmax_true([0.2, 0.3, 0.3]) == (1, (1, 2))


def test_three_case_rule_equals_argmax_whenever_true_means_above_half():
    # plan §4.4 [INFERRED]: "In every case this reduces to argmax P(True)"
    rng = np.random.default_rng(1)
    for _ in range(3000):
        k = int(rng.integers(1, 8))
        p = np.round(rng.random(k), 2)                           # ties do occur
        label = (p > 0.5).astype(int)
        pos, _, _ = ap.three_case(label, p, 1.0 - p)
        assert pos == ap.argmax_true(p)[0]


def test_the_two_can_differ_when_label_and_probability_disagree():
    # e.g. sklearn's SVC: predict (decision function) vs Platt probability
    label, p = [1, 0], [0.40, 0.45]
    assert ap.three_case(label, p, [0.60, 0.55])[0] == 0
    assert ap.argmax_true(p)[0] == 1


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        ap.three_case([], [], [])
    with pytest.raises(ValueError):
        ap.three_case([0, 2], [0.1, 0.2], [0.9, 0.8])
    with pytest.raises(ValueError):
        ap.three_case([0, 1], [0.1, float("nan")], [0.9, 0.8])
    with pytest.raises(ValueError):
        ap.argmax_true([])


# ---- evaluate ----------------------------------------------------------------------------

class Fixed:
    """Predicts from a table {rounded Support: (label, P(True))}."""
    def __init__(self, label, p):
        self.label, self.p = label, p

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.p)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return np.full(len(X), self.label, dtype=int)


def test_evaluate_asserts_the_equivalence_and_handles_a_single_voted_answer():
    cases = [_case("u", ["b"] * 6, "b"), _case("v", list("aabbbc"), "b"),
             _case("w", list("abccca"), "c"), _case("x", list("aaaabc"), "a")]
    inst = WI.build_instances(_subsets(cases), "AP")
    assert inst.n == 1 + 3 + 3 + 3 and inst.undefined.tolist() == [True] + [False] * 9
    pred = LG.logo_predict(inst, "fixed", make=lambda: Fixed(0, 0.3))
    assert pred.log["n_fits"] == 3 and pred.label[0] == -1
    res = ap.evaluate(inst, pred)
    assert res.case == ("unanimous", "none_true", "none_true", "none_true")
    assert res.correct[0] and res.answer[0] == 1                 # the only voted answer
    # equal probabilities: tie -> canonical first = answer "a"
    assert res.answer.tolist() == [1, 0, 0, 0] and res.tie.tolist() == [False, True, True, True]
    s = res.summary()
    assert s["correct"] == 2 and s["low"] == 1 and s["high"] == 4
    assert s["three_case_differs_from_argmax"] == 0
    # labels that contradict the probabilities: asserted by default
    pred.label[inst.rows_of(1)] = [0, 0, 1]
    pred.p_true[inst.rows_of(1)] = [0.2, 0.9, 0.4]
    pred.p_false[inst.rows_of(1)] = [0.8, 0.1, 0.6]
    with pytest.raises(AssertionError, match="plan §4.4"):
        ap.evaluate(inst, pred)
    res = ap.evaluate(inst, pred, assert_equivalence=False)
    assert res.agree.tolist() == [True, False, True, True]
    assert res.answer[1] == 2 and res.argmax_answer[1] == 1      # literal rule gives the answer
    assert res.summary()["three_case_differs_from_argmax"] == 1
    with pytest.raises(ValueError):
        ap.evaluate(WI.build_instances(_subsets(cases), "RP"), pred)


def test_run_partition_ap_end_to_end_and_equivalence_holds_for_the_soft_vote(small):
    data, subsets = small
    out = ML.run_partition(data, 0, "random", "AP", "soft_vote", ML.options_from_config())
    assert out["summary"]["three_case_differs_from_argmax"] == 0
    assert out["summary"]["argmax_correct"] == out["summary"]["correct"]
    assert out["predictions_log"]["label_false_but_p_true_above_half"] == 0
    assert sum(out["summary"][f"case_{c}"] for c in ap.CASES + ("unanimous",)) == len(subsets)
    cell = ML.summarise_ap([out])
    assert cell["mean_count"] == out["summary"]["correct"] and cell["sd_count"] != cell["sd_count"]


def test_thesis_values_of_t23_and_t25():
    assert T.T23_COUNT == {"RP": 49, "AP": 42}
    assert round(49 / 77, 3) == 0.636 and round(42 / 77, 3) == 0.545
    from cdm.eval import stats as ST
    # T25 is computed from the rounded rates (plan T25)
    assert round(ST.pre(0.442, 0.636), 3) == T.T25_PRE_VS_MR["RP"]
    assert round(ST.pre(0.442, 0.545), 3) == T.T25_PRE_VS_MR["AP"]
    p = ML.pre_vs_mr([49, 49], [34, 34], 77)
    assert p["from_mean_rates"] == pytest.approx((49 - 34) / (77 - 34))
    assert p["sd_of_partitions"] == 0.0


# ---- the driver --------------------------------------------------------------------------

def test_b7_driver_reads_b6_and_marks_every_file(tmp_path):
    import run_b6_rp as B6
    import run_b7_ap as B7
    cfg = b_small.write_config(tmp_path, b_small.small_synthetic())
    common = ["--config", str(cfg), "--runs-root", str(tmp_path / "runs"), "--seeds", "2",
              "--n-jobs", "2"]
    # without B6: the RP row says so
    assert B7.main(common) == 0
    out = tmp_path / "runs" / "worldB_synthetic" / "b7_ap_seeds2"
    md = (out / "ap_t23.md").read_text(encoding="utf-8")
    assert "| RP | not run |" in md
    assert B6.main(common) == 0 and B7.main(common) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["ap_outcomes.json", "ap_per_seed.csv", "ap_predictions.csv",
                     "ap_summary.json", "ap_t23.md", "b7_ap.log"]
    for p in out.iterdir():
        assert _marked(p), p.name
    md = (out / "ap_t23.md").read_text(encoding="utf-8")
    assert "| RP | not run |" not in md
    assert "run (synthetic):" in md and "thesis:" in md and "BY CONSTRUCTION" in md
    assert "NOT a reproduction of T23" in md
    # synthetic PRE: within-synthetic, no thesis column beside it (audit S16)
    pre_head = [ln for ln in md.splitlines() if ln.startswith("| PRE against MR")][0]
    assert "thesis" not in pre_head
    assert "not comparable with the PRE reported in the thesis" in md
    doc = json.loads((out / "ap_summary.json").read_text(encoding="utf-8"))
    assert doc["three_case_equals_argmax_in_every_subset"] is True
    assert doc["thesis"]["T23_count"] == {"RP": 49, "AP": 42}
    assert set(doc["run"]["rules"]) == {"MR", "WC", "SP", "HAC"}
    assert doc["run"]["RP_prop_avgp_from_b6"]["counts"] and doc["run"]["pre_vs_MR"]["RP"]
    # an outcomes file of other seeds is refused, not silently used
    with pytest.raises(SystemExit, match="re-run"):
        import _worldb as W
        args = W.parser("x").parse_args(common[:4] + ["--seeds", "1"])
        run = W.Run(args, "b7_ap", "test")
        W.load_outcomes(run, out.parent / "b6_rp_seeds2" / "rp_outcomes.json", "RP", "soft_vote")
