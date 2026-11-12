"""B6 (plan §9.3): the RP pipeline — instances, soft vote, leave-one-group-
out, the nine classification-aggregation strategies.

    UNTESTED ON REAL DATA. The data used here is the synthetic stand-in
    (plan §2.3) or hand-made numbers. Nothing below compares a result with
    the thesis: plan row B6 gives "Synthetic: runs end to end only".
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import b_small
from cdm.approaches import rp
from cdm.approaches import worldB_instances as WI
from cdm.data.worldB import loader as L
from cdm.data.worldB import partition as WP
from cdm.eval import worldB_logo as LG
from cdm.eval import worldB_ml as ML
from cdm.eval import worldB_thesis_values as T
from cdm.features import rcr
from cdm.models import worldB_classifiers as WC
from cdm.schema import Case, Response

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MARK = L.SYNTHETIC_MARKER


@pytest.fixture(scope="module")
def small():
    data = b_small.small_synthetic()
    return data, WP.subset_cases(data, seed=0)


# ---- instances (TH p.55: one instance per response; plan §4.3) -------------------------

def test_rp_instances_are_the_rcr_features_of_each_response(small):
    data, subsets = small
    inst = WI.build_instances(subsets, "RP")
    assert inst.feature_names == rcr.FEATURE_NAMES and inst.X.shape[1] == 14
    assert inst.n == sum(c.n for c in data.tasks) == sum(subsets.sizes())
    assert inst.n_groups == len(subsets)
    for g in (0, len(subsets) - 1):
        case = subsets.cases[g]
        rows = inst.rows_of(g)
        want = rcr.rcr_features(case.responses, case.answers)
        assert rows.size == case.n and list(inst.member[rows]) == list(range(case.n))
        for i, (r, feats) in zip(rows, zip(case.responses, want)):
            assert list(inst.X[i]) == [feats[k] for k in rcr.FEATURE_NAMES]
            # target: "whether the response is correct" (TH p.39)
            assert inst.y[i] == int(r.v == case.correct)
            assert case.answers[inst.answer_index[i]] == r.v
    assert inst.synthetic and inst.marker == MARK
    assert inst.task_ids == subsets.task_ids


def test_instances_reject_an_unknown_approach(small):
    with pytest.raises(ValueError):
        WI.build_instances(small[1], "DAP")


# ---- the classifier (TH p.56; plan OQ26) -----------------------------------------------

def test_soft_vote_is_the_five_classifiers_of_th_p56_with_equal_weights():
    hyper = WC.hyperparameters_from_config()
    clf = WC.make_classifier("soft_vote", hyper)
    assert type(clf).__name__ == "VotingClassifier" and clf.voting == "soft"
    assert clf.weights is None                                   # equal weights
    names = [type(e).__name__ for _, e in clf.estimators]
    assert names == ["LinearDiscriminantAnalysis", "RandomForestClassifier",
                     "XGBClassifier", "LogisticRegression", "KNeighborsClassifier"]
    assert [n for n, _ in clf.estimators] == list(WC.SOFT_VOTE_MEMBERS)
    by = dict(clf.estimators)
    assert by["RF"].n_estimators == 100 and by["KNN"].n_neighbors == 5       # OQ15
    assert by["LR"].max_iter == 10000                                        # AM18
    assert WC.oq26_from_config()["OQ26"]["scaling"] == "none"


def test_soft_vote_probability_is_the_mean_of_the_members_and_true_means_above_half():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(int)
    clf = WC.make_classifier("soft_vote").fit(X[:150], y[:150])
    label, p_true, p_false = WC.predict_both(clf, X[150:])
    members = np.mean([e.predict_proba(X[150:])[:, 1] for e in clf.estimators_], axis=0)
    assert np.allclose(p_true, members, atol=1e-12)              # "Weighted Average Probability"
    assert np.allclose(p_true + p_false, 1.0)
    assert np.array_equal(label == 1, p_true > 0.5)              # plan §4.3
    assert WC.lr_not_converged(clf, WC.hyperparameters_from_config()) == 0


def test_unknown_classifier_and_changed_switches_are_refused():
    with pytest.raises(ValueError):
        WC.make_classifier("naive_bayes")
    from cdm.config import load_config
    cfg = load_config()
    cfg["open_questions"]["OQ26"]["default"]["scaling"] = "standard_scaler"
    with pytest.raises(NotImplementedError):
        WC.oq26_from_config(cfg)
    cfg = load_config()
    cfg["open_questions"]["OQ27"]["default"]["prop"] = "something_else"
    with pytest.raises(NotImplementedError):
        ML.options_from_config(cfg)


# ---- leave-one-group-out (TH p.57: group = one set of responses) ------------------------

class Spy:
    """Records the training sets; predicts P(True) = 0.25 for everything."""
    seen = []

    def fit(self, X, y):
        Spy.seen.append(X.copy())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.tile([0.75, 0.25], (len(X), 1))

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_logo_trains_on_all_other_subsets_and_never_on_the_held_out_one(small):
    _, subsets = small
    inst = WI.build_instances(subsets, "RP")
    Spy.seen = []
    pred = LG.logo_predict(inst, "spy", make=Spy)
    assert pred.log["n_fits"] == len(subsets) == len(Spy.seen)
    for g, Xtrain in enumerate(Spy.seen):
        rows = inst.rows_of(g)
        assert len(Xtrain) == inst.n - rows.size
        assert np.array_equal(Xtrain, np.delete(inst.X, rows, axis=0))
    # the other sub-sets of the same task ARE in the training data (thesis)
    assert len(set(subsets.task_ids)) < len(subsets)
    assert (pred.label == 0).all() and np.allclose(pred.p_true, 0.25)


def test_leave_one_task_out_is_a_switch_and_not_the_default(small):
    _, subsets = small
    inst = WI.build_instances(subsets, "RP")
    assert ML.options_from_config()["group_by"] == "subset"
    Spy.seen = []
    pred = LG.logo_predict(inst, "spy", make=Spy, group_by="task")
    tasks = list(dict.fromkeys(subsets.task_ids))
    assert pred.log["n_fits"] == len(tasks)
    for t, Xtrain in zip(tasks, Spy.seen):
        n_task = sum(c.n for c, tid in zip(subsets.cases, subsets.task_ids) if tid == t)
        assert len(Xtrain) == inst.n - n_task
    with pytest.raises(ValueError):
        LG.logo_predict(inst, "spy", make=Spy, group_by="fold")


# ---- empty out-group (decision B1-2, B6-4) --------------------------------------------

def _case(cid, votes, correct, answers=("a", "b", "c")):
    rs = []
    for j, v in enumerate(votes):
        ps = [0.2, 0.3, 0.5] if j % 2 else [0.5, 0.3, 0.2]
        rs.append(Response(v=v, c=0.4 + 0.05 * (j % 7), ps=tuple(ps)))
    return Case(case_id=cid, answers=answers, correct=correct, responses=tuple(rs))


def _subsets(cases):
    return WP.WorldBSubsets(cases=tuple(cases), task_ids=tuple(c.case_id for c in cases),
                            response_indices=tuple(tuple(range(c.n)) for c in cases),
                            seed=0, assignment="random", synthetic=False, marker="")


def test_unanimous_subset_is_left_out_of_training_and_returns_its_only_answer():
    cases = [_case("u", ["b"] * 6, "b"), _case("v", list("aabbbc"), "b"),
             _case("w", list("abccca"), "c"), _case("x", list("aaaabc"), "a")]
    inst = WI.build_instances(_subsets(cases), "RP")
    assert inst.info["unanimous_subsets"] == 1 and inst.info["undefined_instances"] == 6
    assert inst.undefined[inst.rows_of(0)].all() and not inst.undefined[6:].any()
    Spy.seen = []
    pred = LG.logo_predict(inst, "spy", make=Spy)
    assert pred.log["n_fits"] == 3                               # no fit for the unanimous one
    assert all(np.isfinite(X).all() for X in Spy.seen)           # never trained on NaN
    assert [len(X) for X in Spy.seen] == [12, 12, 12]
    assert (pred.label[inst.rows_of(0)] == -1).all()
    assert np.isnan(pred.p_true[inst.rows_of(0)]).all()
    res = rp.evaluate(inst, pred)
    assert res.unanimous.tolist() == [True, False, False, False]
    for name in rp.STRATEGY_NAMES:
        assert res.answer[name][0] == 1 and res.correct[name][0]
        assert not res.primary_tie[name][0]
    with pytest.raises(ValueError, match="empty out-group"):
        LG.logo_predict(inst, "spy", make=Spy, empty_out_group="raise")
    with pytest.raises(ValueError):
        LG.logo_predict(inst, "spy", make=Spy, empty_out_group="impute_zero")


# ---- the strategies (TH p.39-40; plan OQ27) ---------------------------------------------

def _scores(idx, label, p):
    return rp.answer_scores(idx, label, p)


def test_the_five_criteria_on_a_hand_example():
    #          answer 0        answer 2          answer 3
    idx = [0, 0, 0,            2, 2,             3]
    lab = [1, 0, 0,            1, 1,             0]
    p = [0.9, 0.2, 0.1,        0.6, 0.7,         0.45]
    cand, s = _scores(idx, lab, p)
    assert cand == [0, 2, 3]                                     # only voted answers
    assert s["maj"] == [1, 2, 0]
    assert s["prop"] == pytest.approx([1 / 3, 1.0, 0.0])
    assert s["wm"] == pytest.approx([1.2, 1.3, 0.45])
    assert s["avgp"] == pytest.approx([0.4, 0.65, 0.45])
    assert s["maxp"] == pytest.approx([0.9, 0.7, 0.45])
    pick = {name: rp.decide(cand, s, st, standalone_ties="canonical_first").answer
            for st, name in zip(rp.STRATEGIES, rp.STRATEGY_NAMES)}
    assert pick == {"maxp": 0, "avgp": 2, "wm": 2, "maj+avgp": 2, "maj+maxp": 2,
                    "maj+wm": 2, "prop+avgp": 2, "prop+maxp": 2, "prop+wm": 2}


def test_nine_strategies_in_the_order_of_table_4_2():
    assert rp.STRATEGY_NAMES == ("maxp", "avgp", "wm", "maj+avgp", "maj+maxp", "maj+wm",
                                 "prop+avgp", "prop+maxp", "prop+wm")
    assert rp.FINAL_NAME == "prop+avgp"                          # TH p.57
    assert tuple(T.TABLE_4_2) == rp.STRATEGY_NAMES


def test_tie_on_the_primary_criterion_goes_to_the_tie_breaker():
    # maj: answers 0 and 1 both have 2 "True" supporters; prop: 2/3 vs 2/2
    idx = [0, 0, 0, 1, 1, 2]
    lab = [1, 1, 0, 1, 1, 0]
    p = [0.95, 0.6, 0.3, 0.7, 0.7, 0.2]
    cand, s = _scores(idx, lab, p)
    d = rp.decide(cand, s, ("maj", "avgp"))
    assert d.primary_tied == (0, 1) and d.answer == 1 and d.resolved_by == "avgp"
    d = rp.decide(cand, s, ("maj", "maxp"))
    assert d.answer == 0 and d.resolved_by == "maxp"
    d = rp.decide(cand, s, ("maj", "wm"))
    assert d.answer == 0 and d.final_tied == (0,)                # 1.85 vs 1.4
    d = rp.decide(cand, s, ("prop", "avgp"))
    assert d.primary_tied == (1,) and d.answer == 1 and d.resolved_by == "primary"


def test_no_supporter_classified_true_is_a_tie_decided_by_the_tie_breaker():
    # plan OQ27: "When no supporter is predicted True, prop is 0 for all
    # answers -> tie -> avgp decides."
    idx = [0, 0, 1, 2, 2]
    lab = [0, 0, 0, 0, 0]
    p = [0.1, 0.2, 0.45, 0.3, 0.4]
    cand, s = _scores(idx, lab, p)
    for primary in ("maj", "prop"):
        d = rp.decide(cand, s, (primary, "avgp"))
        assert d.primary_tied == (0, 1, 2) and d.answer == 1
        assert rp.decide(cand, s, (primary, "wm")).answer == 2   # 0.7 > 0.45 > 0.3


def test_a_tie_left_after_the_tie_breaker_goes_to_canonical_order():
    idx = [1, 1, 3, 3]
    lab = [1, 0, 1, 0]
    p = [0.8, 0.2, 0.8, 0.2]
    cand, s = _scores(idx, lab, p)
    for st in rp.STRATEGIES[3:]:
        d = rp.decide(cand, s, st)
        assert d.final_tied == (1, 3) and d.answer == 1 and d.resolved_by == "canonical_first"


def test_standalone_ties_are_random_as_in_table_a7_and_seeded():
    idx = [1, 1, 3, 3]
    lab = [1, 0, 1, 0]
    p = [0.8, 0.2, 0.8, 0.2]
    cand, s = _scores(idx, lab, p)
    with pytest.raises(ValueError, match="random generator"):
        rp.decide(cand, s, ("maxp", None))
    picks = [rp.decide(cand, s, ("maxp", None), rng=rp.tie_rng(0, f"case{k}", "maxp")).answer
             for k in range(200)]
    assert set(picks) == {1, 3} and 60 < picks.count(1) < 140
    again = [rp.decide(cand, s, ("maxp", None), rng=rp.tie_rng(0, f"case{k}", "maxp")).answer
             for k in range(200)]
    assert picks == again
    d = rp.decide(cand, s, ("maxp", None), standalone_ties="canonical_first")
    assert d.answer == 1 and d.resolved_by == "canonical_first"


def test_float_scores_within_the_am3_tolerance_are_tied():
    idx = [0, 0, 0, 1, 1, 1]
    lab = [1, 1, 1, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.3, 0.2, 0.1]                           # the same numbers, other order
    cand, s = _scores(idx, lab, p)
    d = rp.decide(cand, s, ("avgp", None), standalone_ties="canonical_first")
    assert d.primary_tied == (0, 1)
    d = rp.decide(cand, s, ("avgp", None), tol=0.0, standalone_ties="canonical_first")
    assert len(d.primary_tied) >= 1                              # exact comparison: no guarantee


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        rp.answer_scores([], [], [])
    with pytest.raises(ValueError):
        rp.answer_scores([0, 1], [1, 2], [0.5, 0.5])
    with pytest.raises(ValueError):
        rp.answer_scores([0, 1], [1, 0], [0.5, float("nan")])
    cand, s = _scores([0, 1], [1, 0], [0.6, 0.4])
    with pytest.raises(ValueError):
        rp.decide(cand, s, ("maj", None))
    with pytest.raises(ValueError):
        rp.decide(cand, s, ("maxp", None), standalone_ties="first")


def test_evaluate_counts_ties_and_bounds(small):
    _, subsets = small
    inst = WI.build_instances(subsets, "RP")
    Spy.seen = []
    pred = LG.logo_predict(inst, "spy", make=Spy)      # every response "False", p = 0.25
    res = rp.evaluate(inst, pred, tie_seed=0)
    G = len(subsets)
    s = res.summary()
    for name in ("maj+avgp", "prop+maxp"):
        # nothing "True": the primary criterion ties everywhere, the tie
        # breaker ties too -> canonical order
        assert s[name]["primary_ties"] == G and s[name]["final_ties"] == G
        assert s[name]["low"] == 0 and s[name]["high"] == G - inst.info[
            "subsets_without_vote_for_correct"]
    # wm = 0.25 * supporters: the answer with most supporters (= MR)
    assert s["wm"]["correct"] <= G
    assert s["maxp"]["primary_ties"] == G


# ---- the printed table ------------------------------------------------------------------

def test_every_percentage_of_tables_4_2_and_a7_is_a_share_of_77_subsets():
    # support for "Percentage of Tie Breakers = share of the 77 sub-sets"
    possible = {int(100 * k / 77 + 0.5) for k in range(78)}
    for table in T.TABLE_A_7.values():
        for succ, tie in table.values():
            assert succ in possible and tie in possible
    assert T.TABLE_A_7["soft_vote"] == T.TABLE_4_2
    assert T.TABLE_4_2["prop+avgp"] == (64, 18)
    assert round(T.T23_COUNT["RP"] / 77, 3) == T.T23_RATE["RP"]
    # the three maj rows / prop rows print different tie percentages
    assert {T.TABLE_4_2[n][1] for n in ("maj+avgp", "maj+maxp", "maj+wm")} == {16, 17}
    assert {T.TABLE_4_2[n][1] for n in ("prop+avgp", "prop+maxp", "prop+wm")} == {16, 17, 18}


# ---- the pipeline on one small partition ------------------------------------------------

def test_run_partition_end_to_end_on_the_small_synthetic_set(small):
    data, subsets = small
    opts = ML.options_from_config()
    out = ML.run_partition(data, 0, "random", "RP", "soft_vote", opts)
    G = len(subsets)
    assert out["n_subsets"] == G and out["predictions_log"]["n_fits"] == G
    assert set(out["summary"]) == set(rp.STRATEGY_NAMES)
    assert set(out["rules"]) == {"MR", "WC", "SP", "HAC"}
    assert out["predictions"]["label"].shape == (data.n_responses,)
    assert out["predictions_log"]["label_false_but_p_true_above_half"] == 0
    assert out["predictions_log"]["label_true_but_p_true_not_above_half"] == 0
    # deterministic (seeds fixed, OQ15)
    again = ML.run_partition(data, 0, "random", "RP", "soft_vote", opts)
    assert np.array_equal(out["predictions"]["p_true"], again["predictions"]["p_true"])
    assert out["summary"] == again["summary"]
    s = ML.summarise_rp([out, again])
    assert s["prop+avgp"]["sd_count"] == 0.0
    assert s["maj+avgp"]["mean_tie_pct"] == s["maj+wm"]["mean_tie_pct"]


def test_mean_sd_convention():
    assert ML.mean_sd([1, 2, 3]) == (2.0, 1.0)                   # sample SD, ddof 1 (R2)
    m, sd = ML.mean_sd([5])
    assert m == 5.0 and sd != sd


# ---- the driver ---------------------------------------------------------------------------

def _marked(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        doc = json.loads(text)
        return list(doc)[0] == "_marker" and doc["_marker"] == MARK
    if path.suffix == ".csv":
        first = text.splitlines()[0]
        return first.startswith("#") and MARK in first
    if path.suffix == ".md":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return MARK in lines[0] and MARK in lines[-1]
    return MARK in text.splitlines()[0]                         # .log


def test_b6_driver_marks_every_file_and_separates_run_from_thesis(tmp_path):
    import run_b6_rp as B6
    cfg = b_small.write_config(tmp_path, b_small.small_synthetic())
    assert B6.main(["--config", str(cfg), "--runs-root", str(tmp_path / "runs"),
                    "--seeds", "2", "--n-jobs", "2"]) == 0
    out = tmp_path / "runs" / "worldB_synthetic" / "b6_rp_seeds2"   # not the full run's folder
    names = sorted(p.name for p in out.iterdir())
    assert names == ["b6_rp.log", "rp_outcomes.json", "rp_per_seed.csv",
                     "rp_predictions.csv", "rp_summary.json", "rp_table_4_2.md"]
    for p in out.iterdir():
        assert _marked(p), p.name
    doc = json.loads((out / "rp_summary.json").read_text(encoding="utf-8"))
    assert doc["synthetic"] is True and "not a reproduction" in doc["note"]
    assert doc["partition"]["seeds"] == [0, 1] and doc["partition"]["full_set_of_seeds"] is False
    assert set(doc["run"]) == set(doc["thesis"]["table_4_2"]) == set(rp.STRATEGY_NAMES)
    assert not any("thesis" in k for cell in doc["run"].values() for k in cell)
    md = (out / "rp_table_4_2.md").read_text(encoding="utf-8")
    assert "run (synthetic):" in md and "thesis:" in md
    assert "difference" not in md.split("|---")[0]              # no difference column
    assert md.count("\n| ") >= 9
    rows = (out / "rp_per_seed.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 1 + 2 * 9


def test_b6_driver_on_an_unmarked_file_goes_to_worldB_real_with_the_status(tmp_path):
    import run_b6_rp as B6
    cfg = b_small.write_config(tmp_path, b_small.small_unmarked())
    assert B6.main(["--config", str(cfg), "--runs-root", str(tmp_path / "runs"),
                    "--seeds", "1", "--n-jobs", "1"]) == 0
    assert not (tmp_path / "runs" / "worldB_synthetic").exists()
    out = tmp_path / "runs" / "worldB_real" / "b6_rp_seeds1"
    doc = json.loads((out / "rp_summary.json").read_text(encoding="utf-8"))
    assert list(doc)[0] == "_status" and doc["_status"] == "UNTESTED ON REAL DATA"
    assert doc["synthetic"] is False and "note" not in doc
    md = (out / "rp_table_4_2.md").read_text(encoding="utf-8")
    assert MARK not in md and "run (real file):" in md
    assert "UNTESTED ON REAL DATA" in md
    assert (out / "rp_per_seed.csv").read_text(encoding="utf-8").startswith(
        "# UNTESTED ON REAL DATA")
