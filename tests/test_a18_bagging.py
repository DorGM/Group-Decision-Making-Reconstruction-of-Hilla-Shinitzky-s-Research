"""Step A18: bagging experiments (TH Tables 5.9, 5.10).

SYNTHETIC DATA ONLY: hand-built and synthetic cases, and the driver run
with `--smoke`. No build is read.

Tested (plan §9.3 row A18):
  * the size formula: as printed with Z = 1.645 gives 35 <= size <= 42
    (rounded up); the Cochran option gives 35 <= size <= 68; both <= 70
    (TH p.50); exact rounding; the OQ22 switches;
  * the new instances: only cases with |R| >= 70, F per case, drawn
    without replacement, a* / answers / domain kept, features and labels
    recomputed on the sub-group, reproducible draws;
  * no leakage across problems: the new instances of the problems in
    Test_K are not in BagTrain_K, all others are (asserted in the run and
    checked here on the rows the classifier actually receives);
  * counts: training-set sizes, per-fold successes, BR+RF / RF in every
    fold, the folds of the model selection;
  * the open item: an undefined CAmin stops the run by default;
  * the thesis-reported tables; the driver.
"""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from cdm import aggregators as AG
from cdm.config import CODE_ROOT, load_config
from cdm.eval import bagging as B
from cdm.eval import loo as L
from cdm.eval import nested_cv as N
from cdm.report import thesis_values as T
from cdm.schema import Case, Response

SCRIPTS = CODE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
PRINTED, COCHRAN = B.FORMULA_PRINTED, B.FORMULA_COCHRAN


# ---- size formula (TH p.50; plan §5.6, OQ22) ----------------------------------------

def test_size_formula_values():
    # as printed: max(35, Z * S (1 - S) / 0.1^2), Z = 1.645
    assert B.subgroup_size(50, 100) == 42                    # 41.125 -> 42
    assert B.subgroup_size(45, 100) == 41                    # 40.71 -> 41
    assert B.subgroup_size(30, 100) == 35                    # 34.5 -> 35 (= the floor)
    assert B.subgroup_size(90, 100) == 35                    # 14.8 -> floor 35
    assert B.subgroup_size(1, 37) == 35
    # Cochran: max(35, Z^2 * S (1 - S) / 0.1^2)
    assert B.subgroup_size(50, 100, COCHRAN) == 68           # 67.65 -> 68
    assert B.subgroup_size(80, 100, COCHRAN) == 44           # 43.3 -> 44
    assert B.subgroup_size(95, 100, COCHRAN) == 35           # 12.9 -> floor 35
    # Z = 1.96
    assert B.subgroup_size(50, 100, PRINTED, 1.96) == 49     # exactly 49.0: not 50
    assert B.subgroup_size(50, 100, COCHRAN, 1.96) == 97     # 96.04 -> 97 (> 70)


def test_size_ranges_of_plan_row_a18():
    sizes = {f: [B.subgroup_size(k, n, f) for n in range(70, 130) for k in range(n + 1)]
             for f in (PRINTED, COCHRAN)}
    assert (min(sizes[PRINTED]), max(sizes[PRINTED])) == (35, 42)
    assert (min(sizes[COCHRAN]), max(sizes[COCHRAN])) == (35, 68)
    assert max(sizes[COCHRAN]) <= 70 == B.MIN_RESPONSES
    z196 = [B.subgroup_size(k, n, PRINTED, 1.96) for n in range(70, 130) for k in range(n + 1)]
    assert (min(z196), max(z196)) == (35, 49)
    assert B.MIN_SIZE == 35 and B.FACTORS == (2, 3, 5, 10)


def test_size_formula_rejects_bad_input():
    with pytest.raises(ValueError):
        B.subgroup_size(5, 100, "cochran")
    with pytest.raises(ValueError):
        B.subgroup_size(5, 100, PRINTED, 2.58)
    with pytest.raises(ValueError):
        B.subgroup_size(101, 100)
    with pytest.raises(ValueError, match="cannot be drawn"):
        B.draw_subgroup(70, B.subgroup_size(35, 70, COCHRAN, 1.96), seed=1)


def test_settings_from_config():
    assert B.settings_from_config() == {"size_formula": PRINTED, "z": 1.645}
    assert B.settings_from_config(size_formula=COCHRAN, z=1.96) == {
        "size_formula": COCHRAN, "z": 1.96}
    cfg = copy.deepcopy(load_config())
    cfg["open_questions"]["OQ22"]["default"]["size_rounding"] = "round"
    with pytest.raises(NotImplementedError):
        B.settings_from_config(cfg)
    cfg = copy.deepcopy(load_config())
    cfg["open_questions"]["OQ42"]["default"]["within_case"] = "with_replacement"
    with pytest.raises(NotImplementedError):
        B.settings_from_config(cfg)
    with pytest.raises(ValueError):
        B.settings_from_config(z=2.0)


# ---- the new instances ------------------------------------------------------------------

def make_case(case_id, n, n_correct, domain="General", seed=0):
    rng = np.random.default_rng(seed)
    votes = ["a1"] * n_correct + ["a2"] * (n - n_correct)
    rs = tuple(Response(v=v, c=float(round(rng.uniform(0.1, 1.0), 2)),
                        ps=(p, round(1 - p, 2)))
               for v, p in zip(votes, np.round(rng.uniform(0.1, 0.9, n), 2).tolist()))
    return Case(case_id=case_id, answers=("a1", "a2"), correct="a1", responses=rs,
                domain=domain)


def test_only_cases_with_70_responses_are_bagged():
    cases = [make_case("small", 69, 40), make_case("edge", 70, 35), make_case("big", 120, 90)]
    bag = B.bag_cases(cases, 3, seed=0)
    assert bag.eligible == [1, 2] and len(bag) == 6
    assert bag.parent.tolist() == [1, 1, 1, 2, 2, 2]
    assert bag.parent_ids == ["edge"] * 3 + ["big"] * 3
    assert bag.sizes.tolist() == [42] * 3 + [35] * 3          # S = .5 -> 42; S = .75 -> 35
    assert len({c.case_id for c in bag.cases}) == 6
    d = bag.describe()
    assert (d["eligible_cases"], d["new_instances"], d["size_min"], d["size_max"]) == (2, 6, 35, 42)
    assert len(B.bag_cases(cases[:1], 10)) == 0 and B.bagged_instances(
        B.bag_cases(cases[:1], 10)) is None
    with pytest.raises(ValueError):
        B.bag_cases(cases, 0)


def test_subgroups_are_drawn_without_replacement_from_the_parent():
    case = make_case("p", 100, 50)
    bag = B.bag_cases([case], 10, seed=0)
    for new, idx in zip(bag.cases, bag.draws):
        assert len(idx) == 42 == len(set(idx.tolist())) and idx.max() < 100
        assert new.responses == tuple(case.responses[i] for i in idx)
        assert (new.answers, new.correct, new.domain) == (case.answers, case.correct,
                                                          case.domain)
    # sub-groups are independent draws: they differ, and they may overlap
    assert len({tuple(i.tolist()) for i in bag.draws}) == 10
    assert set(bag.draws[0].tolist()) & set(bag.draws[1].tolist())
    # reproducible; another seed or another F gives other draws
    again = B.bag_cases([case], 10, seed=0)
    assert all(np.array_equal(a, b) for a, b in zip(bag.draws, again.draws))
    other = B.bag_cases([case], 10, seed=1)
    assert not all(np.array_equal(a, b) for a, b in zip(bag.draws, other.draws))
    f5 = B.bag_cases([case], 5, seed=0)
    assert not np.array_equal(f5.draws[0], bag.draws[0])
    # a case's draws do not depend on the other cases
    two = B.bag_cases([make_case("q", 80, 30), case], 10, seed=0)
    assert all(np.array_equal(a, b) for a, b in zip(bag.draws, two.draws[10:]))
    assert B.draw_seed(0, "p", 10, 0) != B.draw_seed(0, "p", 10, 1)


def test_features_and_labels_are_recomputed_on_the_subgroup():
    case = make_case("p", 100, 50, seed=3)
    bag = B.bag_cases([case], 5, seed=0)
    inst = B.bagged_instances(bag)
    assert inst.F[:, 0].tolist() == [42.0] * 5                # feature 1: N of the sub-group
    assert inst.correct == ["a1"] * 5 and len(inst) == 5
    for j, new in enumerate(bag.cases):
        for m in AG.RULES:
            assert inst.choices[m][j] == AG.choose(m, new.responses, new.answers)
        share = sum(r.v == "a1" for r in new.responses) / 42
        assert inst.outcomes("MR")[j] == int(share > 0.5) or share == 0.5
    # the sub-groups differ, so the recomputed features differ from the parent's
    from cdm.approaches import instances as I
    parent = I.build_instances([case])
    assert not np.allclose(inst.F[:, 8], parent.F[0, 8])      # AvgC


# ---- no leakage ---------------------------------------------------------------------------

def test_bag_train_mask_and_leak_check():
    parent = np.array([0, 0, 3, 3, 5, 5])
    tr, te = np.array([0, 1, 2, 4, 5]), np.array([3])
    mask = B.bag_train_mask(parent, te)
    assert mask.tolist() == [True, True, False, False, True, True]
    B.check_no_leakage(tr, te, parent, mask, 6)
    leak = mask.copy()
    leak[2] = True                                             # a test problem's instance
    with pytest.raises(AssertionError, match="test problem"):
        B.check_no_leakage(tr, te, parent, leak, 6)
    lost = mask.copy()
    lost[0] = False                                            # a training instance dropped
    with pytest.raises(AssertionError, match="left out"):
        B.check_no_leakage(tr, te, parent, lost, 6)
    with pytest.raises(AssertionError, match="overlap"):
        B.check_no_leakage(np.array([0, 1, 2, 3, 4, 5]), te, parent, mask, 6)
    with pytest.raises(AssertionError, match="cover"):
        B.check_no_leakage(np.array([0, 1, 2]), te, parent, mask, 6)
    ids = list("abcdef")
    with pytest.raises(AssertionError):
        B.check_no_leakage(tr, te, parent, leak, 6, ids, [ids[p] for p in parent])


@pytest.fixture(scope="module")
def smoke():
    import _common as C
    inst, _, cases = C.smoke_instances(40, n_min=50, n_max=100, return_cases=True)
    return inst, cases, C.hyper_for_seed(0, smoke=True)


@pytest.mark.parametrize("approach", ["AMP", "DAP"])
def test_training_rows_never_come_from_test_problems(smoke, approach, monkeypatch):
    inst, cases, hyper = smoke
    bag = B.bag_cases(cases, 3, seed=0)
    inst_bag, parent, report = B.apply_undefined_policy(B.bagged_instances(bag), bag.parent)
    assert report["undefined_instances"] == 0 and 0 < len(bag.eligible) < len(cases)
    inp = (L.amp_inputs if approach == "AMP" else L.dap_inputs)
    X, Xb = inp(inst)["X"], inp(inst_bag)["X"]
    seen = []
    real = L.fit_predict_counted

    def spy(spec, X_train, Y_train, X_test):
        seen.append((spec.name, np.array(X_train), np.array(X_test)))
        return real(spec, X_train, Y_train, X_test)

    monkeypatch.setattr(L, "fit_predict_counted", spy)
    res = B.run_bagging(approach, inst, inst_bag, parent, 3, hyper, n_splits=5)
    splits = N.outer_folds(len(inst), 5)
    assert len(seen) == 5
    rows = lambda A: {tuple(r) for r in np.round(A, 12).tolist()}       # noqa: E731
    for (name, X_train, X_test), (tr, te), f in zip(seen, splits, res.folds):
        assert name == ("BR+RF" if approach == "AMP" else "RF")          # every fold
        assert np.array_equal(X_test, X[te])                             # originals only
        from_test = np.isin(parent, te)
        assert len(X_train) == len(tr) + int((~from_test).sum())
        assert np.array_equal(X_train[:len(tr)], X[tr])
        assert np.array_equal(X_train[len(tr):], Xb[~from_test])
        # no row of a test problem (original or new) is in the training set
        assert not rows(X_train) & rows(X[te])
        assert not rows(X_train) & rows(Xb[from_test])
        assert f["n_train_original"] == len(tr) and f["n_train_bagged"] == (~from_test).sum()
        assert f["n_bagged_excluded"] == 3 * len(set(te.tolist()) & set(bag.eligible))
        assert f["n_train_bagged"] + f["n_bagged_excluded"] == len(inst_bag)


def test_run_counts_and_folds(smoke):
    inst, cases, hyper = smoke
    bag = B.bag_cases(cases, 2, seed=0)
    inst_bag = B.bagged_instances(bag)
    res = B.run_bagging("AMP", inst, inst_bag, bag.parent, 2, hyper, n_splits=5)
    assert res.approach == "AMP" and res.factor == 2 and res.result.protocol == "bagging_cv"
    assert [f["n_test"] for f in res.folds] == [8] * 5
    assert sum(f["successes"] for f in res.folds) == res.result.n_success
    for k, (tr, te) in enumerate(N.outer_folds(len(inst), 5), start=1):  # OQ23: same folds
        assert set(res.result.folds[te].tolist()) == {k}
        assert res.folds[k - 1]["successes"] == int(res.result.success[te].sum())
    assert len(res.fold_rates()) == 5 and set(res.result.techniques) == {"BR+RF"}
    assert res.result.case_ids == inst.case_ids
    # deterministic, also with joblib
    again = B.run_bagging("AMP", inst, inst_bag, bag.parent, 2, hyper, n_splits=5, n_jobs=2)
    assert np.array_equal(again.result.proba, res.result.proba)
    # no new instance at all = the plain 5-fold run of the technique
    plain = B.run_bagging("DAP", inst, None, np.array([], dtype=int), None, hyper, n_splits=5)
    assert [f["n_train_bagged"] for f in plain.folds] == [0] * 5
    with pytest.raises(ValueError):
        B.run_bagging("DAP", inst, inst_bag, bag.parent[:-1], 2, hyper, n_splits=5)
    with pytest.raises(ValueError):
        B.run_bagging("RP", inst, inst_bag, bag.parent, 2, hyper, n_splits=5)


# ---- the open item: undefined CAmin (OQ7 / AM12) ----------------------------------------------

def test_undefined_camin_stops_by_default():
    # 79 of 80 responses give a1: most sub-groups of 35 are unanimous
    cases = [make_case("u", 80, 79), make_case("v", 80, 40, seed=1)]
    bag = B.bag_cases(cases, 10, seed=0)
    inst_bag = B.bagged_instances(bag)
    rows, counts = B.undefined_rows(inst_bag)
    unanimous = [j for j, c in enumerate(bag.cases) if len({r.v for r in c.responses}) == 1]
    assert rows.tolist() == unanimous and 0 < len(rows) < 10
    assert counts == {"CAmin": len(rows)}
    with pytest.raises(B.UndefinedFeatureError, match="plan gives no rule") as err:
        B.apply_undefined_policy(inst_bag, bag.parent)
    assert err.value.counts == counts and err.value.rows == unanimous
    # the two labelled candidates (not in the plan)
    kept, parent, rep = B.apply_undefined_policy(B.bagged_instances(bag), bag.parent,
                                                 "drop_instance")
    assert len(kept) == 20 - len(rows) == len(parent) and np.isfinite(kept.F).all()
    assert rep["undefined_instances"] == len(rows) and rep["policy"] == "drop_instance"
    imp, parent, _ = B.apply_undefined_policy(B.bagged_instances(bag), bag.parent,
                                              "impute_subgroup_avgc")
    names = list(imp.feature_names)
    assert len(imp) == 20 and np.isfinite(imp.F).all()
    assert np.array_equal(imp.F[rows, names.index("CAmin")], imp.F[rows, names.index("AvgC")])
    with pytest.raises(ValueError):
        B.apply_undefined_policy(inst_bag, bag.parent, "zero")
    # a NaN anywhere else is never accepted
    broken = B.bagged_instances(bag)
    broken.F[0, names.index("AvgC")] = np.nan
    with pytest.raises(B.UndefinedFeatureError, match="other than CAmin"):
        B.apply_undefined_policy(broken, bag.parent, "drop_instance")
    # and the run itself refuses NaN
    import _common as C
    with pytest.raises(B.UndefinedFeatureError, match="NaN"):
        B.run_bagging("AMP", C.smoke_instances(20)[0], inst_bag, np.zeros(20, dtype=int),
                      10, C.hyper_for_seed(0, smoke=True), n_splits=2)


# ---- thesis-reported values ----------------------------------------------------------------------------------

def test_thesis_tables():
    from cdm.eval.stats import round_half_up
    for approach, total in (("AMP", 935), ("DAP", 937)):
        table, dash = T.thesis_bagging_table(approach)
        assert sorted(table) == [2, 3, 5, 10] and all(len(r) == 10 for r in table.values())
        # the "-" row is the Test column of TH Table 5.1 / 5.2 and sums to T1 / T2
        counts = [round(x * (121 if k < 9 else 120)) for k, x in enumerate(dash)]
        assert sum(counts) == total
        assert all(round_half_up(c, 121 if k < 9 else 120) == x
                   for k, (c, x) in enumerate(zip(counts, dash)))
    assert T.thesis_bagging_table("AMP")[1][5] == 0.917 and T.TH_TABLE_5_9[10][0] == 0.868
    assert T.TH_TABLE_5_10[10][4] == 0.512 and T.TH_TABLE_5_10[2][9] == 0.592
    with pytest.raises(ValueError):
        T.thesis_bagging_table("RP")


# ---- the driver ------------------------------------------------------------------------------

def run_script(*args):
    return subprocess.run([sys.executable, str(SCRIPTS / "run_a18_bagging.py"), *args],
                          capture_output=True, text=True, cwd=str(SCRIPTS))


def test_driver_smoke_and_cache(tmp_path):
    args = ["--smoke", "40", "--factors", "3,2", "--n-jobs", "1", "--runs-root",
            str(tmp_path), "--results-root", str(tmp_path / "results")]
    p = run_script(*args)
    assert p.returncode == 0, p.stderr
    out = tmp_path / "_smoke" / "a18"
    doc = json.loads((out / "a18_summary.json").read_text())
    assert doc["complete"] and doc["settings"]["size_formula"] == B.FORMULA_PRINTED
    assert doc["settings"]["undefined_camin"] == "stop"
    for a in ("AMP", "DAP"):
        for F in ("3", "2"):
            r = doc[a][F]
            assert len(r["folds"]) == 10 and not r["from_checkpoint"]
            assert r["successes"] == sum(f["successes"] for f in r["folds"])
            assert all(f["n_train_bagged"] + f["n_bagged_excluded"]
                       == doc["new_instances"][F]["new_instances"] for f in r["folds"])
        assert doc[a]["dash_row_a14"] is None
        assert (out / f"a18_{a.lower()}_F3_cases.csv").is_file()
    assert doc["new_instances"]["3"]["new_instances"] == \
        3 * doc["new_instances"]["3"]["eligible_cases"]
    text = (out / "a18_bagging.md").read_text()
    assert "NOT A RESULT" in text and "## AMP (TH Table 5.9)" in text
    assert not (tmp_path / "results").exists()
    first = {a: doc[a]["3"]["successes"] for a in ("AMP", "DAP")}
    p = run_script(*args)                                    # second run: from the cache
    assert p.returncode == 0, p.stderr
    doc = json.loads((out / "a18_summary.json").read_text())
    assert all(doc[a]["3"]["from_checkpoint"] and doc[a]["3"]["successes"] == first[a]
               for a in ("AMP", "DAP"))
    # the Cochran switch goes to its own folder; --describe-only fits nothing
    p = run_script("--smoke", "40", "--describe-only", "--size-formula", B.FORMULA_COCHRAN,
                   "--runs-root", str(tmp_path), "--results-root", "")
    assert p.returncode == 0, p.stderr
    d = json.loads((tmp_path / "_smoke" / "a18_cochran" / "a18_summary.json").read_text())
    assert d["new_instances"]["10"]["size_max"] > 42 and "AMP" not in d
    assert not (tmp_path / "_smoke" / "a18_cochran" / "ckpt").exists()


def test_driver_refuses_unlisted_build(tmp_path):
    p = run_script("--build", "h1209_nba_games", "--runs-root", str(tmp_path),
                   "--results-root", "")
    assert p.returncode != 0 and "GateError" in p.stderr
    assert not list(tmp_path.rglob("ckpt"))
