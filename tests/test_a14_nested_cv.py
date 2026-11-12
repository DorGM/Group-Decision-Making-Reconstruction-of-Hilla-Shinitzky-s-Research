"""A14 (plan §9.3 row A14; §5.2): leave-one-out within 10-fold CV.

SYNTHETIC DATA ONLY. Some rule-baseline counts of the real data differ from the
thesis-reported counts; the project owner authorised proceeding
on 2026-11-11 with the builds `waiver_main` / `waiver_politics_alt`
(docs/IMPLEMENTATION_DECISIONS.md G6), and the real runs are made by the driver scripts in
scripts/, not by pytest: no classifier is fitted on a World-A build here.
The plan's "own run" criteria (inner scores +-0.02, an RF technique wins
every fold, fold-test pattern, AMP ~935, DAP ~937) need a real run and are
NOT tested. Tested:
  * the plan's deterministic selection-logic test: fed the 10 x 9
    numbers of TH Table 5.1 (p.63) the selection returns BR+RF in folds
    1-5, 7, 8, 10 and CC+RF in folds 6 and 9; fed Table 5.2, RF in all;
  * fold sizes 121 x 9 + 120 for n = 1209, unshuffled, in case order;
  * on synthetic data: the run equals a hand-written nested loop; the
    BR+KNN == LP+KNN fingerprint; independence of n_jobs; checkpoints.

LR needs a `max_iter` that the plan does not give (OQ15; the config value
is 10000, amendment AM18); LR_MAX_ITER and the small forests here are test
values only, not project defaults.
"""
import numpy as np
import pytest

from cdm.config import load_config
from cdm.eval import loo as L
from cdm.eval import nested_cv as N
from cdm.models import multilabel as M

LR_MAX_ITER = 500           # test value only (plan OQ15 gives no number)
LABELS = ("MR", "HAC", "WC", "SP", "DA")
HYPER = dict(rf_n_estimators=5, knn_k=5, lr_solver="lbfgs", lr_max_iter=LR_MAX_ITER,
             scaling="none", seed=0)

AMP_NAMES = ("BR+KNN", "BR+LR", "BR+RF", "CC+KNN", "CC+LR", "CC+RF",
             "LP+KNN", "LP+LR", "LP+RF")
#: TH Table 5.1 p.63 (= MLA Table 5 without the BNB rows): per fold the
#: inner-LOO success rate of the 9 techniques, then the test score.
TH_TABLE_5_1 = [
    (0.776, 0.759, 0.800, 0.774, 0.749, 0.790, 0.776, 0.767, 0.789, 0.851),
    (0.783, 0.772, 0.812, 0.789, 0.767, 0.809, 0.783, 0.784, 0.806, 0.793),
    (0.778, 0.767, 0.808, 0.774, 0.754, 0.796, 0.778, 0.772, 0.792, 0.826),
    (0.788, 0.767, 0.803, 0.784, 0.748, 0.792, 0.788, 0.777, 0.792, 0.835),
    (0.769, 0.748, 0.793, 0.768, 0.731, 0.777, 0.769, 0.756, 0.789, 0.934),
    (0.768, 0.750, 0.789, 0.763, 0.726, 0.790, 0.768, 0.757, 0.789, 0.917),
    (0.779, 0.767, 0.805, 0.778, 0.754, 0.801, 0.779, 0.774, 0.797, 0.860),
    (0.791, 0.779, 0.827, 0.788, 0.778, 0.808, 0.791, 0.798, 0.823, 0.645),
    (0.813, 0.788, 0.830, 0.810, 0.768, 0.836, 0.813, 0.799, 0.824, 0.479),
    (0.796, 0.781, 0.817, 0.789, 0.764, 0.809, 0.796, 0.788, 0.813, 0.592),
]
#: TH Table 5.2 p.63 (= MLA Table 6): KNN, LR, RF, then the test score.
TH_TABLE_5_2 = [
    (0.763, 0.695, 0.793, 0.810), (0.767, 0.708, 0.804, 0.826),
    (0.757, 0.707, 0.802, 0.901), (0.760, 0.699, 0.792, 0.868),
    (0.757, 0.730, 0.792, 0.826), (0.759, 0.733, 0.781, 0.826),
    (0.765, 0.720, 0.802, 0.835), (0.790, 0.737, 0.824, 0.727),
    (0.797, 0.720, 0.827, 0.579), (0.782, 0.714, 0.820, 0.550),
]


def synthetic(n=40, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    z = X[:, 0] + 0.5 * rng.normal(size=n)
    Y = np.column_stack([
        z + 0.8 * rng.normal(size=n) > -0.6,
        z + 1.5 * rng.normal(size=n) > -0.6,
        z + 0.9 * rng.normal(size=n) > -0.6,
        X[:, 1] + z > -0.8,
        -z + 0.7 * rng.normal(size=n) > 0.5,
    ]).astype(int)
    y = (X[:, 2] + z > 0).astype(int)
    domains = [("General", "NBA", "NFL", "Politics")[min(3, 4 * i // n)] for i in range(n)]
    return X, Y, y, domains


# ---- the plan's selection-logic unit test --------------------------------------------

def test_technique_order_is_the_table_order():
    assert tuple(M.technique_name(*t) for t in M.AMP_TECHNIQUES_TH) == AMP_NAMES
    assert M.DAP_TECHNIQUES == ("KNN", "LR", "RF")


def test_selection_on_th_table_5_1():
    """Bold cells of TH Table 5.1 (plan §5.2): BR+RF in folds 1-5, 7,
    8 and 10; CC+RF in folds 6 (0.790) and 9 (0.836)."""
    chosen = [N.select_technique(row[:9], AMP_NAMES) for row in TH_TABLE_5_1]
    want = ["BR+RF"] * 10
    want[5] = want[8] = "CC+RF"
    assert chosen == want
    assert TH_TABLE_5_1[5][5] == 0.790 and TH_TABLE_5_1[8][5] == 0.836
    # the default (no tie rule) never had to decide: no printed row has a
    # tie for the maximum
    for row in TH_TABLE_5_1:
        assert sorted(row[:9])[-1] > sorted(row[:9])[-2]


def test_selection_on_th_table_5_2():
    assert [N.select_technique(row[:3], M.DAP_TECHNIQUES)
            for row in TH_TABLE_5_2] == ["RF"] * 10


def test_th_tables_fingerprint_and_count_closure():
    """Checks that the numbers typed above are the thesis's: BR+KNN and
    LP+KNN are equal in every fold (plan §5.2 fingerprint), and the test
    scores times the fold sizes (121 x 9, 120) are whole counts that sum
    to 935 (T1) and 937 (T2) (plan §2.1)."""
    for row in TH_TABLE_5_1:
        assert row[0] == row[6]
    sizes = [len(te) for _, te in N.outer_folds(1209, 10)]
    for table, total in ((TH_TABLE_5_1, 935), (TH_TABLE_5_2, 937)):
        counts = []
        for row, m in zip(table, sizes):
            k = round(row[-1] * m)
            assert abs(k / m - row[-1]) < 5e-4
            counts.append(k)
        assert sum(counts) == total
    assert round(TH_TABLE_5_1[0][-1] * 121) == 103 and round(TH_TABLE_5_1[9][-1] * 120) == 71


def test_tie_between_techniques_raises_unless_a_rule_is_named():
    """The plan gives no rule for two techniques with the same highest
    score; the default is to stop."""
    names = ("KNN", "LR", "RF")
    with pytest.raises(N.TechniqueTieError) as err:
        N.select_technique([700, 650, 700], names)
    assert err.value.tied == ("KNN", "RF")
    assert N.select_technique([700, 650, 700], names, tie="first_in_table_order") == "KNN"
    assert N.select_technique([650, 700, 700], names, tie="first_in_table_order") == "LR"
    # a tie below the maximum is no tie
    assert N.select_technique([650, 650, 700], names) == "RF"
    with pytest.raises(ValueError):
        N.select_technique([1, 2, 3], names, tie="random")
    with pytest.raises(ValueError):
        N.select_technique([1, float("nan"), 3], names)
    with pytest.raises(ValueError):
        N.select_technique([1, 2], names)


# ---- outer folds (plan §5.2, OQ16) --------------------------------------------------------

def test_fold_sizes_for_1209_cases():
    folds = N.outer_folds(1209, 10)
    assert [len(te) for _, te in folds] == [121] * 9 + [120]
    assert [len(tr) for tr, _ in folds] == [1088] * 9 + [1089]
    # unshuffled, consecutive blocks in case order, a partition of 0..1208
    assert np.array_equal(np.concatenate([te for _, te in folds]), np.arange(1209))
    for tr, te in folds:
        assert np.array_equal(te, np.arange(te[0], te[-1] + 1))
        assert np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(1209))
        assert np.all(np.diff(tr) > 0)             # train rows keep the case order
    # plan §2.1: fold 5 = rows 484-604 (16 General + 105 NBA); fold 8 = 847-967
    assert (folds[4][1][0], folds[4][1][-1]) == (484, 604)
    assert (folds[7][1][0], folds[7][1][-1]) == (847, 967)
    assert (folds[9][1][0], folds[9][1][-1]) == (1089, 1208)


def test_fold_scheme_comes_from_the_config():
    assert N.fold_scheme_from_config() == 10
    cfg = load_config()
    cfg["open_questions"]["OQ16"]["default"]["scheme"] = "shuffled_stratified"
    with pytest.raises(NotImplementedError):
        N.fold_scheme_from_config(cfg)
    with pytest.raises(ValueError):
        N.outer_folds(5, 10)


# ---- the run on synthetic data ---------------------------------------------------------------

@pytest.fixture(scope="module")
def amp_run():
    X, Y, _, domains = synthetic()
    res = N.run_amp_nested_cv(X, Y, hyper=HYPER, n_splits=4, domains=domains,
                              technique_tie="first_in_table_order", n_jobs=2)
    return X, Y, domains, res


def test_amp_run_equals_a_hand_written_nested_loop(amp_run):
    """Inner score = success rate of a LOO over the 90% (OQ18); the best
    technique is refitted on the 90% and applied to the 10%."""
    X, Y, _, res = amp_run
    assert res.techniques == AMP_NAMES and len(res.folds) == 4
    folds = N.outer_folds(len(X), 4)
    total = 0
    for f, (tr, te) in zip(res.folds, folds):
        assert np.array_equal(f.train_idx, tr) and np.array_equal(f.test_idx, te)
        counts = {}
        for t in M.AMP_TECHNIQUES_TH:
            inner = L.run_amp_loo(X[tr], Y[tr], t, hyper=HYPER)
            name = M.technique_name(*t)
            counts[name] = inner.n_success
            assert np.array_equal(f.inner_success[name], inner.success)
        assert f.inner_successes == counts
        best = max(counts.values())
        tied = tuple(n for n in AMP_NAMES if counts[n] == best)
        assert f.tied == tied and f.selected == tied[0]
        model = M.make_amp_model(*f.selected.split("+"), LABELS, **HYPER).fit(X[tr], Y[tr])
        sel = model.select(X[te])
        ok = Y[te, sel]
        assert f.test_successes == ok.sum()
        assert np.array_equal(res.result.selected[te], sel)
        assert np.array_equal(res.result.success[te], ok)
        assert set(res.result.techniques[i] for i in te) == {f.selected}
        assert np.all(res.result.folds[te] == f.fold)
        total += int(ok.sum())
    assert res.n_success == total == res.result.n_success
    assert res.success_rate == total / len(X)
    assert res.result.protocol == "nested_cv"


def test_br_knn_equals_lp_knn_in_every_fold(amp_run):
    """Plan §5.2 fingerprint (TH Table 5.1): identical inner outcomes."""
    res = amp_run[3]
    assert res.identical_inner("BR+KNN", "LP+KNN")
    for f in res.folds:
        assert f.inner_successes["BR+KNN"] == f.inner_successes["LP+KNN"]


def test_table_has_the_shape_of_th_table_5_1(amp_run):
    res = amp_run[3]
    frame = res.to_frame()
    assert list(frame.index) == ["Fold 1", "Fold 2", "Fold 3", "Fold 4"]
    assert list(frame.columns[:10]) == list(AMP_NAMES) + ["Test"]
    for f, (_, row) in zip(res.folds, frame.iterrows()):
        assert row["BR+RF"] == f.inner_successes["BR+RF"] / f.n_train
        assert row["Test"] == f.test_successes / f.n_test
        assert row["Selected"] == f.selected == max(
            AMP_NAMES, key=lambda t: (row[t], -AMP_NAMES.index(t)))
    assert frame["Test successes"].sum() == res.n_success
    text = res.format_table().splitlines()
    assert len(text) == 6 and text[0].split() == ["Classifier", *AMP_NAMES, "Test"]
    assert all(line.count("*") == 1 for line in text[1:5])
    rows = res.table_rows()
    assert rows[0]["n_train"] == 30 and rows[0]["n_test"] == 10


def test_result_feeds_the_effectiveness_counts(amp_run):
    """The accumulated test outcomes are an EvalResult (A15's input)."""
    _, Y, _, res = amp_run
    r = res.result
    assert sum(r.selection_counts().values()) == r.n
    assert sum(r.selected_success_counts().values()) == r.n_success
    assert sum(p.n_success for p in r.by_domain().values()) == r.n_success
    assert np.array_equal(r.outcomes, Y)
    assert r.meta["selected"] == res.selected() and r.meta["n_splits"] == 4


def test_dap_run_parallel_equals_serial_and_manual():
    X, _, y, domains = synthetic()
    kw = dict(hyper=HYPER, n_splits=4, domains=domains,
              technique_tie="first_in_table_order")
    a = N.run_dap_nested_cv(X, y, n_jobs=1, **kw)
    b = N.run_dap_nested_cv(X, y, n_jobs=2, batch_size=3, **kw)
    assert a.techniques == ("KNN", "LR", "RF")
    assert a.selected() == b.selected()
    assert np.array_equal(a.result.predicted, b.result.predicted)
    for fa, fb in zip(a.folds, b.folds):
        assert fa.inner_successes == fb.inner_successes
        tr, te = fa.train_idx, fa.test_idx
        for t in M.DAP_TECHNIQUES:
            assert fa.inner_successes[t] == L.run_dap_loo(X[tr], y[tr], t, hyper=HYPER).n_success
        clf = M.make_dap_classifier(fa.selected, **HYPER).fit(X[tr], y[tr])
        assert fa.test_successes == int((clf.predict(X[te]) == y[te]).sum())
    assert a.n_success == sum(f.test_successes for f in a.folds)


def test_selection_really_follows_the_inner_scores():
    """A data set on which KNN is clearly the best inner technique in one
    part and RF elsewhere is not needed: it is enough that the selected
    technique is the argmax of the inner counts in every fold, for two
    different technique lists (a list without the overall winner must
    select another technique)."""
    X, _, y, _ = synthetic()
    kw = dict(hyper=HYPER, n_splits=4, technique_tie="first_in_table_order")
    full = N.run_dap_nested_cv(X, y, **kw)
    for f in full.folds:
        assert f.inner_successes[f.selected] == max(f.inner_successes.values())
    rest = tuple(t for t in M.DAP_TECHNIQUES if t != full.folds[0].selected)
    part = N.run_dap_nested_cv(X, y, techniques=rest, **kw)
    assert part.techniques == rest and part.folds[0].selected in rest
    for f, g in zip(full.folds, part.folds):
        assert all(g.inner_successes[t] == f.inner_successes[t] for t in rest)


def test_tie_in_a_run_raises_after_the_inner_loops(tmp_path):
    """Without a rule a tie stops the run, and the inner loops are on disk
    so that nothing is lost. (The same technique twice is refused; two
    KNN wrappers with identical outcomes give a guaranteed tie.)"""
    X, Y, _, _ = synthetic()
    kw = dict(techniques=[("BR", "KNN"), ("LP", "KNN")], hyper=HYPER, n_splits=4,
              checkpoint_dir=tmp_path / "ck")
    with pytest.raises(N.TechniqueTieError, match="fold 1") as err:
        N.run_amp_nested_cv(X, Y, technique_tie=None, **kw)
    assert len(list((tmp_path / "ck").glob("fold_*/*/batch_*.npz"))) == 4 * 2 * 2
    # N1: the error carries the finished inner loops
    assert len(err.value.counts) == 4 and len(err.value.inner_success) == 4
    assert all(c["BR+KNN"] == c["LP+KNN"] for c in err.value.counts)
    assert all(v["BR+KNN"].shape == (30,) for v in err.value.inner_success)
    res = N.run_amp_nested_cv(X, Y, technique_tie="first_in_table_order", **kw)
    assert res.selected() == ["BR+KNN"] * 4
    assert all(f.tied == ("BR+KNN", "LP+KNN") for f in res.folds)
    assert [f.inner_successes for f in res.folds] == err.value.counts
    with pytest.raises(ValueError):
        N.run_amp_nested_cv(X, Y, techniques=[("BR", "KNN"), ("BR", "KNN")], hyper=HYPER,
                            technique_tie="first_in_table_order")


def test_n1_no_rule_without_checkpoint_is_refused_at_the_start(monkeypatch):
    """With the rule None (stop on a tie) and no
    checkpoint folder, a tie would lose every inner loop, so the run is
    refused before any fit."""
    X, Y, y, _ = synthetic()
    monkeypatch.setattr(L, "fit_predict_counted",
                        lambda *a, **k: pytest.fail("a fit was started"))
    with pytest.raises(ValueError, match="checkpoint_dir"):
        N.run_amp_nested_cv(X, Y, hyper=HYPER, n_splits=4, technique_tie=None)
    with pytest.raises(ValueError, match="checkpoint_dir"):
        N.run_dap_nested_cv(X, y, hyper=HYPER, n_splits=4, technique_tie=None)


def test_t1_no_rule_run_completes_when_there_is_no_tie(tmp_path):
    """The default-less run (rule None) finishes when no
    fold is tied, and equals the run with a rule."""
    X, _, y, _ = synthetic()
    kw = dict(hyper=HYPER, n_splits=4)
    ref = N.run_dap_nested_cv(X, y, technique_tie="first_in_table_order", **kw)
    assert ref.tied_folds() == [], "the synthetic data must give no tie for this test"
    res = N.run_dap_nested_cv(X, y, technique_tie=None, checkpoint_dir=tmp_path / "ck", **kw)
    assert res.selected() == ref.selected()
    assert np.array_equal(res.result.predicted, ref.result.predicted)
    assert res.meta["technique_tie"] is None and res.meta["tied_folds"] == []
    assert all(f.tied == (f.selected,) for f in res.folds)
    assert all(f.tied_test_successes == {f.selected: f.test_successes} for f in res.folds)
    assert res.total_range_over_ties() == (res.n_success, res.n_success)


def test_t1_inner_loop_of_a_fold_ignores_its_test_fold():
    """Corrupting X and y of test fold 1 leaves every
    inner outcome of fold 1 unchanged and changes fold 2 (whose training
    set holds those rows)."""
    X, Y, _, _ = synthetic()
    kw = dict(hyper=HYPER, n_splits=4, technique_tie="first_in_table_order",
              techniques=[("BR", "RF"), ("BR", "KNN")])
    a = N.run_amp_nested_cv(X, Y, **kw)
    te = a.folds[0].test_idx
    X2, Y2 = X.copy(), Y.copy()
    X2[te] = X2[te][::-1] * 7.0 + 3.0
    Y2[te] = 1 - Y2[te]
    b = N.run_amp_nested_cv(X2, Y2, **kw)
    for t in a.techniques:
        assert np.array_equal(a.folds[0].inner_success[t], b.folds[0].inner_success[t])
    assert a.folds[0].selected == b.folds[0].selected
    assert any(not np.array_equal(a.folds[1].inner_success[t], b.folds[1].inner_success[t])
               for t in a.techniques)


def test_am18_tie_rule_default_is_the_config_and_tied_outcomes_are_stored():
    """AM18: the default rule is the config's `technique_tie_rule`
    (first_in_table_order); for a tied fold the test outcome under each
    tied technique is stored."""
    X, Y, _, _ = synthetic()
    assert N.technique_tie_from_config() == "first_in_table_order"
    cfg = load_config()
    cfg["technique_tie_rule"] = "stop"
    assert N.technique_tie_from_config(cfg) is None
    cfg["technique_tie_rule"] = "coin"
    with pytest.raises(ValueError):
        N.technique_tie_from_config(cfg)
    # BR+KNN and CC+KNN: tied in some folds or not; BR+KNN == LP+KNN always
    techs = [("LP", "KNN"), ("CC", "RF"), ("BR", "KNN")]
    res = N.run_amp_nested_cv(X, Y, techniques=techs, hyper=HYPER, n_splits=4)
    assert res.meta["technique_tie"] == "first_in_table_order"
    seen_tie = False
    for f in res.folds:
        tr, te = f.train_idx, f.test_idx
        best = max(f.inner_successes.values())
        assert f.tied == tuple(t for t in res.techniques if f.inner_successes[t] == best)
        assert f.selected == f.tied[0] and set(f.tied_test_successes) == set(f.tied)
        assert f.test_successes == f.tied_test_successes[f.selected]
        for t in f.tied:                           # independent refit per tied technique
            tb = tuple(t.split("+"))
            p = M.make_amp_model(*tb, **HYPER).fit(X[tr], Y[tr]).predict_proba(X[te])
            ok = M.amp_success(Y[te], M.select_methods(p))
            assert f.tied_test_successes[t] == int(np.sum(ok))
            assert np.array_equal(f.tied_test_success[t], np.asarray(ok, dtype=int))
        seen_tie |= f.is_tied
    assert seen_tie, "the synthetic data must give at least one tie for this test"
    assert res.meta["tied_folds"] == [f.fold for f in res.tied_folds()]
    rep = res.tie_report()
    assert [r["fold"] for r in rep] == res.meta["tied_folds"]
    assert all(set(r["test_successes_by_technique"]) == set(r["tied"]) for r in rep)
    lo, hi = res.total_range_over_ties()
    assert lo <= res.n_success <= hi


def test_on_fold_callback_and_convergence_counts():
    """`on_fold` gets every finished fold in order; the convergence counts
    of the inner loops and the refits are stored per fold (N2)."""
    X, _, y, _ = synthetic()
    Xbig = X * np.array([1e4, 1, 1, 1, 1, 1])
    seen = []
    res = N.run_dap_nested_cv(Xbig, y, techniques=["LR", "KNN"], hyper=HYPER,
                              lr_max_iter=2, n_splits=4, on_fold=seen.append)
    assert [f.fold for f in seen] == [1, 2, 3, 4] and seen == res.folds
    for f in res.folds:
        assert f.inner_not_converged == {"LR": 30, "KNN": 0}
        assert f.test_not_converged.get("LR", 1) == 1 and f.test_not_converged.get("KNN", 0) == 0
    nc = res.not_converged()
    assert nc["LR"] == 120 and nc["KNN"] == 0
    assert res.meta["lr_not_converged_by_technique"] == {"LR": 120, "KNN": 0}
    assert res.meta["lr_not_converged"] == 120 + nc["test_refits"]


def test_checkpoint_resume(tmp_path, monkeypatch):
    X, _, y, _ = synthetic()
    kw = dict(hyper=HYPER, n_splits=4, technique_tie="first_in_table_order",
              batch_size=10, checkpoint_dir=tmp_path / "ck")
    full = N.run_dap_nested_cv(X, y, **kw)
    files = sorted((tmp_path / "ck").glob("fold_*/*/batch_*.npz"))
    assert len(files) == 4 * 3 * 3                 # folds x techniques x batches
    assert (tmp_path / "ck" / "fold_03" / "RF" / "meta.json").exists()

    calls = []
    real = L.fit_predict_counted
    monkeypatch.setattr(L, "fit_predict_counted",
                        lambda *a, **k: (calls.append(len(a[1])), real(*a, **k))[1])
    again = N.run_dap_nested_cv(X, y, **kw)
    # only the refits on the 90% are done again (30 training cases each):
    # one per fold, plus one per further tied technique (AM18)
    n_refits = sum(len(f.tied) for f in full.folds)
    assert calls == [30] * n_refits and n_refits >= 4
    assert again.selected() == full.selected()
    assert np.array_equal(again.result.predicted, full.result.predicted)

    calls.clear()
    files[0].unlink()                              # one batch of 10 inner cases lost
    again = N.run_dap_nested_cv(X, y, **kw)
    assert sorted(calls) == [29] * 10 + [30] * n_refits
    assert np.array_equal(again.result.predicted, full.result.predicted)
    for f, g in zip(full.folds, again.folds):
        assert f.inner_successes == g.inner_successes


def test_lr_needs_max_iter():
    X, Y, y, _ = synthetic()
    hyper = dict(HYPER, lr_max_iter=None)
    with pytest.raises(ValueError, match="lr_max_iter"):
        N.run_amp_nested_cv(X, Y, hyper=hyper, n_splits=4)
    with pytest.raises(ValueError, match="lr_max_iter"):
        N.run_dap_nested_cv(X, y, hyper=hyper, n_splits=4)
    # the config value (AM18: 10000) is used when no `hyper` is given
    assert dict(L.make_spec("DAP", "LR").hyper)["lr_max_iter"] == 10000
    # the explicit argument is enough
    res = N.run_dap_nested_cv(X, y, techniques=["LR"], hyper=hyper,
                              lr_max_iter=LR_MAX_ITER, n_splits=4)
    assert res.selected() == ["LR"] * 4
