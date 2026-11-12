"""A13 (plan §9.3 row A13; §5.3): leave-one-out evaluation of AMP and DAP.

SYNTHETIC DATA ONLY. Some rule-baseline counts of the real data differ from the
thesis-reported counts; the project owner authorised proceeding
on 2026-11-11 with the builds `waiver_main` / `waiver_politics_alt`
(docs/IMPLEMENTATION_DECISIONS.md G6), and the real runs are made by the driver scripts in
scripts/, not by pytest: no classifier is fitted on a World-A build here.
The plan's numeric acceptance (AMP ~975/1209, DAP ~982, selection counts,
P(success | selected), McNemar pairs) needs a real run and is NOT tested;
what is tested instead:
  * the LOO loop equals a hand-written loop (fit on all other cases);
  * the numbers do not depend on n_jobs or batch size; checkpoints resume;
  * the result object reproduces the published arithmetic of T19-T22 when
    it is fed arrays built from the back-solved counts of the plan (no
    model is involved).

LR needs a `max_iter` that the plan does not give (OQ15; the config value
is 10000, amendment AM18); LR_MAX_ITER and the small forests here are test
values only, not project defaults.
"""
import json

import numpy as np
import pytest

from cdm.eval import loo as L
from cdm.eval import stats as S
from cdm.models import multilabel as M

LR_MAX_ITER = 500           # test value only (plan OQ15 gives no number)
LABELS = ("MR", "HAC", "WC", "SP", "DA")
#: small forests keep the tests fast; a test value, not the plan's 100.
HYPER = dict(rf_n_estimators=7, knn_k=5, lr_solver="lbfgs", lr_max_iter=None,
             scaling="none", seed=0)


def synthetic(n=36, d=6, seed=0):
    """Random features, five correlated 0/1 labels, a binary target, four
    'domains' in blocks."""
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


@pytest.fixture(scope="module")
def data():
    return synthetic()


# ---- the loop equals a hand-written leave-one-out ------------------------------------

@pytest.mark.parametrize("technique", [("BR", "RF"), ("CC", "KNN"), ("LP", "LR")])
def test_amp_loo_equals_manual_loop(data, technique):
    X, Y, _, _ = data
    res = L.run_amp_loo(X, Y, technique, hyper=HYPER, lr_max_iter=LR_MAX_ITER)
    hyper = dict(HYPER, lr_max_iter=LR_MAX_ITER)
    n = len(X)
    for i in range(n):
        keep = np.arange(n) != i
        model = M.make_amp_model(*technique, LABELS, **hyper).fit(X[keep], Y[keep])
        p = model.predict_proba(X[i:i + 1])[0]
        assert np.array_equal(res.proba[i], p)
        j = int(np.argmax(p))                     # first maximum: OQ17
        assert res.selected[i] == j
        assert res.success[i] == Y[i, j]
    assert res.n_success == int(res.success.sum())
    assert res.success_rate == res.n_success / n
    assert res.protocol == "loo" and set(res.techniques) == {M.technique_name(*technique)}


@pytest.mark.parametrize("technique", ["RF", "KNN", "LR"])
def test_dap_loo_equals_manual_loop(data, technique):
    X, _, y, _ = data
    res = L.run_dap_loo(X, y, technique, hyper=HYPER, lr_max_iter=LR_MAX_ITER)
    hyper = dict(HYPER, lr_max_iter=LR_MAX_ITER)
    n = len(X)
    for i in range(n):
        keep = np.arange(n) != i
        clf = M.make_dap_classifier(technique, **hyper).fit(X[keep], y[keep])
        assert res.predicted[i] == clf.predict(X[i:i + 1])[0]
        assert res.success[i] == int(res.predicted[i] == y[i])
    assert np.array_equal(res.target, y)


def test_defaults_are_br_rf_and_rf(data):
    """Plan §5.3: the LOO runs use BR+RF (AMP) and RF (DAP)."""
    X, Y, y, _ = data
    assert set(L.run_amp_loo(X[:12], Y[:12], hyper=HYPER).techniques) == {"BR+RF"}
    assert set(L.run_dap_loo(X[:12], y[:12], hyper=HYPER).techniques) == {"RF"}


def test_hyperparameters_default_to_the_config(data):
    """Without `hyper` the plan's OQ15 values are used (RF 100 trees,
    KNN k = 5, no scaling, seed 0, LR max_iter 10000 = AM18); a `hyper`
    without an LR max_iter still raises (the code never picks a number)."""
    spec = L.make_spec("AMP", ("BR", "RF"))
    assert dict(spec.hyper) == M.hyperparameters_from_config()
    assert dict(spec.hyper)["rf_n_estimators"] == 100 and dict(spec.hyper)["seed"] == 0
    assert dict(L.make_spec("AMP", ("BR", "LR")).hyper)["lr_max_iter"] == 10000
    with pytest.raises(ValueError, match="lr_max_iter"):
        L.make_spec("AMP", ("BR", "LR"), hyper=HYPER)
    with pytest.raises(ValueError, match="lr_max_iter"):
        L.run_dap_loo(data[0], data[2], "LR", hyper=HYPER)
    assert dict(L.make_spec("DAP", "LR", lr_max_iter=123).hyper)["lr_max_iter"] == 123


# ---- parallelism, batches, checkpoints ---------------------------------------------

def test_result_independent_of_jobs_and_batches(data):
    X, Y, y, _ = data
    a = L.run_amp_loo(X, Y, hyper=HYPER, n_jobs=1, batch_size=36)
    b = L.run_amp_loo(X, Y, hyper=HYPER, n_jobs=2, batch_size=5)
    c = L.run_amp_loo(X, Y, hyper=HYPER, n_jobs=1, batch_size=1)
    assert np.array_equal(a.proba, b.proba) and np.array_equal(a.proba, c.proba)
    assert np.array_equal(a.selected, b.selected)
    d1 = L.run_dap_loo(X, y, hyper=HYPER, n_jobs=1)
    d2 = L.run_dap_loo(X, y, hyper=HYPER, n_jobs=2, batch_size=7)
    assert np.array_equal(d1.predicted, d2.predicted)
    assert np.array_equal(d1.proba, d2.proba)


def test_seed_is_the_only_source_of_randomness(data):
    X, Y, _, _ = data
    a = L.run_amp_loo(X, Y, hyper=HYPER)
    b = L.run_amp_loo(X, Y, hyper=HYPER)
    c = L.run_amp_loo(X, Y, hyper=dict(HYPER, seed=1))
    assert np.array_equal(a.proba, b.proba)
    assert not np.array_equal(a.proba, c.proba)
    assert a.meta["hyper"]["seed"] == 0 and c.meta["hyper"]["seed"] == 1


def test_checkpoint_resume(data, tmp_path, monkeypatch):
    X, Y, _, _ = data
    ck = tmp_path / "amp"
    full = L.run_amp_loo(X, Y, hyper=HYPER, batch_size=6, checkpoint_dir=ck)
    files = sorted(ck.glob("batch_*.npz"))
    assert len(files) == 6
    meta = json.loads((ck / "meta.json").read_text())
    assert meta["n"] == 36 and meta["spec"]["technique"] == "BR+RF"
    # C1: the batch files carry the first 12 hex digits of the fingerprint
    assert all(f.name.startswith("batch_" + meta["fingerprint"][:12] + "_") for f in files)
    assert meta["code_fingerprint"] == L.code_fingerprint()

    calls = []
    real = L.fit_predict_counted
    monkeypatch.setattr(L, "fit_predict_counted",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    # everything is on disk: no fit at all
    again = L.run_amp_loo(X, Y, hyper=HYPER, batch_size=6, checkpoint_dir=ck)
    assert calls == [] and np.array_equal(again.proba, full.proba)
    # two batches lost (as after an interrupted run): only they are refitted
    files[1].unlink()
    files[4].unlink()
    again = L.run_amp_loo(X, Y, hyper=HYPER, batch_size=6, checkpoint_dir=ck)
    assert len(calls) == 12 and np.array_equal(again.proba, full.proba)
    assert len(list(ck.glob("batch_*.npz"))) == 6
    assert not list(ck.glob(".*tmp"))


def test_c1_batches_without_meta_or_from_another_run_are_refused(data, tmp_path):
    """A folder with batch files but no meta.json, and a
    same-shape batch file of another run, must not be reused."""
    X, Y, _, _ = data
    ck = tmp_path / "amp"
    L.run_amp_loo(X[:12], Y[:12], hyper=HYPER, batch_size=6, checkpoint_dir=ck)
    (ck / "meta.json").unlink()
    Y2 = 1 - Y[:12]                                # other inputs, same shapes
    with pytest.raises(ValueError, match="no meta.json"):
        L.run_amp_loo(X[:12], Y2, hyper=HYPER, batch_size=6, checkpoint_dir=ck)
    with pytest.raises(ValueError, match="no meta.json"):
        L.run_amp_loo(X[:12], Y[:12], hyper=HYPER, batch_size=6, checkpoint_dir=ck)
    # a batch file of another run (other fingerprint in its name) in a valid folder
    ck2, ck3 = tmp_path / "a", tmp_path / "b"
    L.run_amp_loo(X[:12], Y[:12], hyper=HYPER, batch_size=6, checkpoint_dir=ck2)
    L.run_amp_loo(X[:12], Y2, hyper=HYPER, batch_size=6, checkpoint_dir=ck3)
    f = sorted(ck3.glob("batch_*.npz"))[0]
    f.rename(ck2 / f.name)
    with pytest.raises(ValueError, match="do not belong"):
        L.run_amp_loo(X[:12], Y[:12], hyper=HYPER, batch_size=6, checkpoint_dir=ck2)
    # a truncated batch file raises instead of being used
    g = sorted(ck3.glob("batch_*.npz"))[0]
    g.write_bytes(g.read_bytes()[:40])
    with pytest.raises(ValueError, match="cannot be read"):
        L.run_amp_loo(X[:12], Y2, hyper=HYPER, batch_size=6, checkpoint_dir=ck3)


def test_c2_fingerprint_covers_the_project_code(data, tmp_path, monkeypatch):
    """A change of the model code (or a raised
    CHECKPOINT_VERSION) invalidates old checkpoints."""
    X, Y, _, _ = data
    spec = L.make_spec("AMP", ("BR", "RF"), hyper=HYPER)
    fp, code = L.fingerprint(spec, X, Y), L.code_fingerprint()
    ck = tmp_path / "amp"
    L.run_amp_loo(X[:12], Y[:12], hyper=HYPER, checkpoint_dir=ck)
    monkeypatch.setattr(L, "CHECKPOINT_VERSION", L.CHECKPOINT_VERSION + 1)
    assert L.code_fingerprint() != code and L.fingerprint(spec, X, Y) != fp
    with pytest.raises(ValueError, match="other inputs"):
        L.run_amp_loo(X[:12], Y[:12], hyper=HYPER, checkpoint_dir=ck)
    monkeypatch.undo()
    # the source of cdm/models/multilabel.py is part of it
    copy = tmp_path / "multilabel.py"
    copy.write_bytes(open(M.__file__, "rb").read() + b"\n# changed\n")
    monkeypatch.setattr(M, "__file__", str(copy))
    assert L.code_fingerprint() != code
    with pytest.raises(ValueError, match="other inputs"):
        L.run_amp_loo(X[:12], Y[:12], hyper=HYPER, checkpoint_dir=ck)


def test_n2_non_convergence_is_counted(data, tmp_path, caplog):
    """LR fits that stop at max_iter are counted, logged,
    stored in the result and kept in the checkpoint; the outputs are those
    of the plain fit."""
    import logging
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    X, Y, y, _ = data
    Xbig = X * np.array([1e4, 1, 1, 1, 1, 1])      # one large-scale column, like N
    ck = tmp_path / "lr"
    with caplog.at_level(logging.WARNING, logger="cdm.eval"):
        bad = L.run_dap_loo(Xbig, y, "LR", hyper=HYPER, lr_max_iter=2, checkpoint_dir=ck)
    assert bad.meta["lr_not_converged"] == len(y) == bad.meta["n_fits"]
    assert bad.summary()["lr_not_converged"] == len(y)
    assert any("did NOT converge" in r.message for r in caplog.records)
    # the checkpoint keeps the count (no refit, same number)
    again = L.run_dap_loo(Xbig, y, "LR", hyper=HYPER, lr_max_iter=2, checkpoint_dir=ck)
    assert again.meta["lr_not_converged"] == len(y)
    assert again.meta["n_from_checkpoint"] == len(y)
    # same through joblib workers
    par = L.run_dap_loo(Xbig, y, "LR", hyper=HYPER, lr_max_iter=2, n_jobs=2, batch_size=5)
    assert par.meta["lr_not_converged"] == len(y)
    assert np.array_equal(par.predicted, bad.predicted)
    # the fit is unchanged: equal to a bare fit with the warning ignored
    spec = L.make_spec("DAP", "LR", hyper=HYPER, lr_max_iter=2)
    out, n_warn = L.fit_predict_counted(spec, Xbig[1:], y[1:], Xbig[:1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = M.make_dap_classifier("LR", **dict(HYPER, lr_max_iter=2)).fit(Xbig[1:], y[1:])
    assert n_warn == 1 and out[0, 0] == clf.predict(Xbig[:1])[0]
    assert np.array_equal(L.fit_predict(spec, Xbig[1:], y[1:], Xbig[:1]), out)
    # BR + LR: one warning per label; the case is counted once
    amp = L.run_amp_loo(Xbig, Y, ("BR", "LR"), hyper=HYPER, lr_max_iter=2)
    assert amp.meta["lr_not_converged"] == len(Y)
    spec = L.make_spec("AMP", ("BR", "LR"), hyper=HYPER, lr_max_iter=2)
    assert L.fit_predict_counted(spec, Xbig[1:], Y[1:], Xbig[:1])[1] == Y.shape[1]
    # a converged run, and techniques without the warning, report 0
    assert L.run_dap_loo(X, y, "LR", hyper=HYPER, lr_max_iter=5000).meta["lr_not_converged"] == 0
    assert L.run_amp_loo(X, Y, hyper=HYPER).meta["lr_not_converged"] == 0


def test_n2_other_warnings_are_passed_on(data, monkeypatch):
    """Only convergence warnings are counted; any other warning of a fit
    still reaches the caller."""
    X, Y, _, _ = data
    spec = L.make_spec("AMP", ("BR", "RF"), hyper=HYPER)
    real = spec.make

    class Noisy:
        def fit(self, X, Y):
            import warnings
            warnings.warn("something else", UserWarning)
            return real().fit(X, Y)

    monkeypatch.setattr(L.Spec, "make", lambda self: Noisy())
    with pytest.warns(UserWarning, match="something else"):
        out, n_warn = L.fit_predict_counted(spec, X[1:], Y[1:], X[:1])
    assert n_warn == 0 and out.shape == (1, 5)


def test_checkpoint_of_other_inputs_is_refused(data, tmp_path):
    X, Y, _, _ = data
    ck = tmp_path / "amp"
    L.run_amp_loo(X[:12], Y[:12], hyper=HYPER, checkpoint_dir=ck)
    X2 = X[:12].copy()
    X2[0, 0] += 1e-9
    for kwargs in (dict(X=X2, Y=Y[:12], hyper=HYPER),
                   dict(X=X[:12], Y=Y[:12], hyper=dict(HYPER, seed=1)),
                   dict(X=X[:12], Y=Y[:12], hyper=HYPER, technique=("CC", "RF")),
                   dict(X=X[:12], Y=Y[:12], hyper=HYPER, batch_size=5)):
        with pytest.raises(ValueError, match="other inputs"):
            L.run_amp_loo(checkpoint_dir=ck, **kwargs)


# ---- input checks ------------------------------------------------------------------------

def test_bad_inputs_raise(data):
    X, Y, y, _ = data
    Xn = X.copy()
    Xn[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        L.run_amp_loo(Xn, Y, hyper=HYPER)
    with pytest.raises(ValueError):
        L.run_amp_loo(X, Y[:, :4], hyper=HYPER)
    with pytest.raises(ValueError):
        L.run_dap_loo(X, Y, hyper=HYPER)
    with pytest.raises(ValueError, match="approach"):
        L.make_spec("RP", "RF", hyper=HYPER)
    with pytest.raises(ValueError):
        L.make_spec("DAP", "SVM", hyper=HYPER)
    with pytest.raises(NotImplementedError):
        L.make_spec("AMP", ("BR", "RF"), hyper=HYPER, tie="prefer_mr")


def test_label_constant_in_a_training_set_raises(data):
    """docs/IMPLEMENTATION_DECISIONS.md M5: a label with a single positive case is constant
    once that case is left out; no rule is defined, so the run stops."""
    X, Y, _, _ = data
    Y2 = Y.copy()
    Y2[:, 4] = 0
    Y2[5, 4] = 1
    with pytest.raises(ValueError, match="constant"):
        L.run_amp_loo(X, Y2, hyper=HYPER)


# ---- result object: counts, per-method, per-domain ----------------------------------------

def test_counts_and_domains_are_consistent(data):
    X, Y, _, domains = data
    answers = [tuple(f"{m}{i}" for m in LABELS) for i in range(len(X))]
    res = L.run_amp_loo(X, Y, hyper=HYPER, domains=domains,
                        case_ids=[f"c{i}" for i in range(len(X))],
                        method_answers=answers)
    sel, ok = res.selection_counts(), res.selected_success_counts()
    assert list(sel) == list(LABELS)
    assert sum(sel.values()) == res.n and sum(ok.values()) == res.n_success
    for j, m in enumerate(LABELS):
        mask = res.selected == j
        assert sel[m] == mask.sum() and ok[m] == Y[mask, j].sum()
        want = Y[mask, j].mean() if mask.any() else float("nan")
        got = res.success_given_selected()[m]
        assert (np.isnan(got) and np.isnan(want)) or got == want
    assert res.method_success_counts() == {m: int(Y[:, j].sum()) for j, m in enumerate(LABELS)}
    # the returned answer is the answer of the selected method
    assert res.answers == [answers[i][j] for i, j in enumerate(res.selected)]
    assert res.selected_names == [LABELS[j] for j in res.selected]
    # per-domain breakdown
    parts = res.by_domain()
    assert list(parts) == ["General", "NBA", "NFL", "Politics"]
    assert sum(p.n for p in parts.values()) == res.n
    assert sum(p.n_success for p in parts.values()) == res.n_success
    for m in LABELS:
        assert sum(p.selection_counts()[m] for p in parts.values()) == sel[m]
        assert sum(p.selected_success_counts()[m] for p in parts.values()) == ok[m]
    table = res.domain_table()
    assert table["all"] == {"n": res.n, "successes": res.n_success, "rate": res.success_rate}
    assert table["NBA"]["n"] == 9
    assert parts["NFL"].case_ids == [f"c{i}" for i in range(18, 27)]
    s = res.summary()
    json.dumps(s)                                  # plain types only
    assert s["successes"] == res.n_success and s["techniques"] == {"BR+RF": res.n}


def test_dap_result_answers_and_rule_outcomes(data):
    X, Y, y, domains = data
    sets = [("yes", "no")] * len(X)
    res = L.run_dap_loo(X, y, hyper=HYPER, domains=domains, answer_sets=sets,
                        rule_outcomes={m: Y[:, j] for j, m in enumerate(LABELS)})
    assert res.answers == ["yes" if c == 1 else "no" for c in res.predicted]
    assert res.labels == LABELS and np.array_equal(res.outcomes, Y)
    b, c = res.discordant("SP")
    assert b == int(((res.success == 1) & (Y[:, 3] == 0)).sum())
    assert c == int(((res.success == 0) & (Y[:, 3] == 1)).sum())
    assert b - c == res.n_success - int(Y[:, 3].sum())
    with pytest.raises(ValueError):
        res.selection_counts()                     # AMP only
    assert set(res.tie_log()) == {"half_rows"}
    assert sum(p.n_success for p in res.by_domain().values()) == res.n_success


def test_tie_log_counts_argmax_ties():
    """KNN probabilities are multiples of 0.2, so ties for the maximum are
    frequent; the first label in the order wins (OQ17) and the log counts
    the rows."""
    X, Y, _, _ = synthetic(n=30, seed=3)
    res = L.run_amp_loo(X, Y, ("BR", "KNN"), hyper=HYPER)
    tied = (res.proba == res.proba.max(axis=1, keepdims=True)).sum(axis=1) > 1
    assert res.tie_log()["tied_rows"] == int(tied.sum()) > 0
    assert np.array_equal(res.selected, np.argmax(res.proba, axis=1))


# ---- the published arithmetic, from the plan's back-solved counts (no model) --------------

def result_from_counts(selected, selected_ok, method_totals, n=1209, seed=0):
    """An AMP EvalResult whose arrays realise given counts: per label the
    number of cases selected, selected-and-correct, and correct overall."""
    rng = np.random.default_rng(seed)
    sel = rng.permutation(np.repeat(np.arange(5), selected))
    assert len(sel) == n
    Y = np.zeros((n, 5), dtype=int)
    for j in range(5):
        inside = np.flatnonzero(sel == j)
        outside = np.flatnonzero(sel != j)
        Y[inside[:selected_ok[j]], j] = 1
        Y[outside[:method_totals[j] - selected_ok[j]], j] = 1
    proba = np.zeros((n, 5))
    proba[np.arange(n), sel] = 1.0
    spec = L.make_spec("AMP", ("BR", "RF"), hyper=HYPER)
    return L.amp_result(spec, proba, Y, "loo")


def test_result_object_reproduces_mla_figures_from_plan_counts():
    """Plan T19-T21 (MLA Figs 1-3; TH Table A.10): label order MR, HAC,
    WC, SP, DA. Selected 268/180/185/422/154; correct when selected
    222/127/149/351/126 (sum 975); rule totals 831/774/830/864/372 (T3)."""
    res = result_from_counts(selected=[268, 180, 185, 422, 154],
                             selected_ok=[222, 127, 149, 351, 126],
                             method_totals=[831, 774, 830, 864, 372])
    assert res.n == 1209 and res.n_success == 975
    assert S.round_half_up(res.n_success, res.n) == 0.806                      # T19
    assert res.selection_counts() == dict(MR=268, HAC=180, WC=185, SP=422, DA=154)
    shares = {m: round(100 * c / 1209) for m, c in res.selection_counts().items()}
    assert shares == dict(SP=35, MR=22, WC=15, HAC=15, DA=13)                  # T20
    eff = res.effectiveness()
    printed = dict(SP=(.832, 5.0e-8), MR=(.828, 3.2e-7), WC=(.805, 2.5e-4),
                   HAC=(.706, 3.4e-2), DA=(.818, None))                       # T21
    for m, (rate, p) in printed.items():
        assert round(eff[m]["p_success_given_selected"], 3) == rate
        if p is None:
            assert eff[m]["p"] < 1e-30
        else:
            assert float(f"{eff[m]['p']:.1e}") == p
    # proportion column of MLA Table 7 (plan §5.8 item 5): AMP vs SP 6.13e-8
    assert float(f"{res.proportion_test('SP')[1]:.2e}") == 6.13e-8
    # b - c of a McNemar pair is fixed by the two success counts (T22)
    b, c = res.discordant("SP")
    assert b - c == 975 - 864


def test_result_object_reproduces_th_table_5_4_from_plan_counts():
    """Plan T8 (TH Table 5.4): selected/correct SP 349/287, MR 244/200,
    WC 241/201, HAC 132/93, DA 243/154; sum 1209 / 935."""
    res = result_from_counts(selected=[244, 132, 241, 349, 243],
                             selected_ok=[200, 93, 201, 287, 154],
                             method_totals=[831, 774, 830, 864, 372])
    assert res.n_success == 935 and S.round_half_up(935, 1209) == 0.773       # T1
    eff = res.effectiveness()
    printed = dict(SP=(.289, .715, .822, .377, 4.2e-6), MR=(.202, .687, .820, .423, 4.1e-6),
                   WC=(.199, .687, .834, .471, 4.0e-7), HAC=(.109, .640, .705, .179, .062),
                   DA=(.201, .308, .634, .471, None))
    for m, (psel, psucc, pgiven, pre, p) in printed.items():
        e = eff[m]
        assert round(e["p_selected"], 3) == psel
        assert round(e["p_success"], 3) == psucc
        assert round(e["p_success_given_selected"], 3) == pgiven
        assert round(e["pre"], 3) == pre
        if p is None:
            assert e["p"] < 1e-20
        elif p == .062:
            assert round(e["p"], 3) == p
        else:
            assert float(f"{e['p']:.1e}") == p


def test_never_selected_method_gives_nan():
    spec = L.make_spec("AMP", ("BR", "RF"), hyper=HYPER)
    proba = np.array([[.9, .1, 0, 0, 0], [.2, .8, 0, 0, 0], [.7, .7, 0, 0, 0]])
    Y = np.array([[1, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 1, 1, 0, 1]])
    res = L.amp_result(spec, proba, Y, "loo")
    assert list(res.selected) == [0, 1, 0]          # row 3: tie -> MR (OQ17)
    assert list(res.success) == [1, 0, 0]
    assert res.selection_counts() == dict(MR=2, HAC=1, WC=0, SP=0, DA=0)
    assert np.isnan(res.success_given_selected()["WC"])
    eff = res.effectiveness()
    assert np.isnan(eff["WC"]["pre"]) and np.isnan(eff["WC"]["p"])
    assert res.tie_log()["tied_rows"] == 1


# ---- instance-table adapters (synthetic cases) ---------------------------------------------

def test_inputs_from_an_instance_table():
    """`amp_inputs` / `dap_inputs` on a small table of synthetic cases:
    shapes 27 / 32 columns, and the returned answer of AMP is the answer
    of the selected method."""
    from cdm.approaches.instances import build_instances
    from cdm.schema import Case, Response
    rng = np.random.default_rng(0)
    cases = []
    for k in range(14):
        rs = []
        for _ in range(24):
            v = "A" if rng.random() < 0.3 + 0.04 * k else "B"
            x = float(np.round(rng.uniform(0.2, 0.9), 2))
            rs.append(Response(v=v, c=float(np.round(rng.random(), 2)),
                               ps=(x, round(1 - x, 2)) if v == "A" else (round(1 - x, 2), x)))
        cases.append(Case(case_id=f"s{k}", domain="General" if k < 7 else "NBA",
                          answers=("A", "B"), correct="A" if k % 2 else "B",
                          responses=rs))
    inst = build_instances(cases)
    a = L.amp_inputs(inst)
    d = L.dap_inputs(inst)
    assert a["X"].shape == (14, 27) and a["Y"].shape == (14, 5)
    assert a["labels"] == LABELS and d["X"].shape == (14, 32)
    for i in range(14):
        for j, m in enumerate(LABELS):
            assert int(a["method_answers"][i][j] == inst.correct[i]) == a["Y"][i, j]
    assert set(d["rule_outcomes"]) == set(LABELS)
