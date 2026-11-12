"""Leave-one-out within 10-fold cross-validation, with model selection.

Step A14 of plan §9.3 ("Nested CV (10-fold unshuffled x inner LOO, 9
techniques)" -> TH Tables 5.1 / 5.2, T1 / T2).

Procedure (plan §5.2; TH §4.1.4 p.45-46; MLA §5.4):
  "a leave-one-out within a 10-fold cross-validation. In each iteration of
   the 10-fold cross-validation, we tested several classification
   techniques, each in leave-one-out cross-validation using only 90% of
   the data. Then, the technique that achieved the highest score (the
   highest rate of successful aggregation outcomes resulting from its
   predictions) is used to train a new classifier over the same 90% of the
   data and then tested using the remaining 10% of the data."

  * Outer folds (plan §5.2, §8 OQ16 [INFERRED]): unshuffled KFold(10) over
    the cases in the order given (General -> NBA -> NFL -> Politics in the
    frozen data). sklearn's rule: the first n mod 10 folds get one case
    more, so 1209 cases give 121 x 9 + 120 (plan §2.1: fold 1 test
    0.851 = 103/121, fold 10 test 0.592 = 71/120).
  * Inner score (plan §8 OQ18): the success rate of the AMP / DAP protocol
    in a leave-one-out over the 90% (cdm.eval.loo).
  * Techniques: AMP {BR, CC, LP} x {KNN, LR, RF}; DAP KNN, LR, RF, in the
    column order of TH Tables 5.1 / 5.2 (cdm.models.multilabel).
  * Headline (T1 / T2; TH p.64): the accumulated test successes of the 10
    folds over all cases.

Ties between techniques. The best technique is the one with the highest
inner score. The scores of one fold share the denominator, so they are
compared as integer success counts (exact). The plan and the thesis give
NO rule for two techniques with the same highest score (in TH Table 5.1
BR+KNN and LP+KNN are equal in every fold but never the best, and no
printed row has a tie for the maximum). `select_technique` therefore
raises `TechniqueTieError` unless the caller names a rule. The only rule
offered, "first_in_table_order", is a labelled option that is NOT in the
plan: the tied technique that comes first in the column order of TH
Table 5.1 / 5.2. It is the config value `technique_tie_rule` (amendment
AM18: a default of this implementation), which
`run_nested_cv` uses unless the caller passes another value; for every
tied fold the test outcome under EACH tied technique is also computed and
stored (`FoldResult.tied_test_successes`), so the effect of the rule is
visible. With the rule None (stop on a tie) a run needs a
`checkpoint_dir`: all inner loops are then finished and on disk before
the tie is raised, so a tie loses no work; without a checkpoint folder
such a run is refused at the start.

Checkpoints: `checkpoint_dir/fold_<k>/<technique>/` holds the batch files
of that inner LOO (cdm.eval.loo.loo_outputs). A run that is started again
with the same inputs reuses them. Seeds: see cdm.eval.loo (every fit is
seeded by the `seed` hyperparameter only, so the numbers do not depend on
the number of jobs or their order).

Nothing in this module reads a data build. The caller passes the arrays.
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold

from cdm.eval import loo as L
from cdm.models import multilabel as M

LOG = logging.getLogger("cdm.eval")

TECHNIQUE_TIE_RULES = ("first_in_table_order",)
#: Value of the config entry `technique_tie_rule` that means "no rule:
#: stop on a tie" (= technique_tie None).
TECHNIQUE_TIE_STOP = "stop"
#: Default of `run_nested_cv(technique_tie=...)`: read the config.
FROM_CONFIG = "from_config"


class TechniqueTieError(ValueError):
    """Two or more techniques share the highest inner score and no rule
    was named (the plan gives none). `tied` holds their names (of the
    first tied fold). Raised by a run, it also carries
      counts        : per fold {technique: inner success count},
      inner_success : per fold {technique: (n_train,) 0/1 array},
    so that the finished inner loops are not lost."""

    def __init__(self, message, tied=(), counts=None, inner_success=None):
        super().__init__(message)
        self.tied = tuple(tied)
        self.counts = counts
        self.inner_success = inner_success


def technique_tie_from_config(config=None):
    """The technique tie rule of the config (top-level `technique_tie_rule`,
    amendment AM18): one of TECHNIQUE_TIE_RULES, or None for 'stop'."""
    from cdm.config import load_config
    config = config if config is not None else load_config()
    if "technique_tie_rule" not in config:
        raise ValueError("config has no 'technique_tie_rule' entry "
                         f"(use {TECHNIQUE_TIE_STOP!r} or one of {TECHNIQUE_TIE_RULES})")
    rule = config["technique_tie_rule"]
    if rule == TECHNIQUE_TIE_STOP:
        return None
    if rule not in TECHNIQUE_TIE_RULES:
        raise ValueError(f"technique_tie_rule = {rule!r} is not {TECHNIQUE_TIE_STOP!r} "
                         f"or one of {TECHNIQUE_TIE_RULES}")
    return rule


# ---- outer folds (plan §5.2, OQ16) -------------------------------------------------

def fold_scheme_from_config(config=None):
    """n_splits of plan §8 OQ16. Only the plan default 'unshuffled_kfold'
    is implemented; the alternative 'shuffled_stratified' (a robustness
    check, for which the plan defines neither the strata nor the seed)
    raises."""
    from cdm.config import get_switch
    oq16 = get_switch("OQ16", config)
    if oq16["scheme"] != "unshuffled_kfold":
        raise NotImplementedError(
            f"OQ16 scheme = {oq16['scheme']!r} is not implemented; only "
            "'unshuffled_kfold' is (the plan does not define the strata or "
            "the seed of the shuffled alternative)")
    return int(oq16["n_splits"])


def outer_folds(n, n_splits=10):
    """[(train_idx, test_idx)] of the unshuffled K-fold split over the case
    order: `sklearn.model_selection.KFold(n_splits, shuffle=False)`.
    Test folds are consecutive blocks; the first n mod n_splits blocks
    have one case more (n = 1209: 121 x 9 + 120)."""
    if n_splits < 2 or n < n_splits:
        raise ValueError("need 2 <= n_splits <= n")
    return [(tr, te) for tr, te in
            KFold(n_splits=n_splits, shuffle=False).split(np.zeros((n, 1)))]


# ---- selection (TH p.45) --------------------------------------------------------------

def select_technique(scores, names, tie=None):
    """Name of the technique with the highest inner score (TH p.45-46).

    scores : one number per technique: success counts (a run) or printed
             rates (the unit test on TH Table 5.1). Compared exactly.
    names  : the technique names, in the column order of the table.
    tie    : what to do when several techniques share the maximum.
             None (default): raise TechniqueTieError (the plan gives no
             rule). "first_in_table_order": the first tied technique in
             `names` (an option that is not in the plan).
    """
    scores = list(scores)
    names = list(names)
    if not scores or len(scores) != len(names):
        raise ValueError("scores and names must be non-empty and of equal length")
    if any(s != s for s in scores):
        raise ValueError("scores hold NaN")
    if tie is not None and tie not in TECHNIQUE_TIE_RULES:
        raise ValueError(f"unknown technique tie rule {tie!r}; expected None "
                         f"or one of {TECHNIQUE_TIE_RULES}")
    best = max(scores)
    tied = [n for n, s in zip(names, scores) if s == best]
    if len(tied) > 1 and tie is None:
        raise TechniqueTieError(
            f"techniques {tied} share the highest inner score ({best}); the "
            "plan and the thesis give no rule for this. Name a rule "
            f"(technique_tie, one of {TECHNIQUE_TIE_RULES}) once it is decided",
            tied)
    return tied[0]


# ---- results --------------------------------------------------------------------------

@dataclass
class FoldResult:
    """One outer fold: a row of TH Table 5.1 / 5.2.

    fold            : 1..K.
    train_idx, test_idx : case positions.
    inner_successes : {technique: successes of its LOO over the train set}.
    inner_success   : {technique: (n_train,) 0/1 per training case}.
    selected        : the technique with the highest inner score.
    tied            : the techniques sharing that score (one name unless a
                      tie rule decided).
    test_successes  : successes of the selected technique, refitted on the
                      train set, on the test set.
    tied_test_successes : {technique: test successes} for EVERY technique
                      in `tied` (amendment AM18); one entry, the selected
                      technique, when there is no tie.
    tied_test_success : {technique: (n_test,) 0/1 per test case}, same keys.
    inner_not_converged : {technique: number of inner-LOO fits with a
                      ConvergenceWarning}.
    test_not_converged : {technique: 1 if its refit on the train set raised
                      a ConvergenceWarning else 0}, keys as `tied`.
    inner_seconds, inner_fresh_fits : {technique: wall seconds and number
                      of the inner-LOO fits made in this run} (0 fits when
                      the loop was read from a checkpoint). Timing only.
    """
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    inner_successes: Dict[str, int]
    inner_success: Dict[str, np.ndarray]
    selected: str
    tied: Tuple[str, ...]
    test_successes: int
    tied_test_successes: Dict[str, int] = field(default_factory=dict)
    tied_test_success: Dict[str, np.ndarray] = field(default_factory=dict)
    inner_not_converged: Dict[str, int] = field(default_factory=dict)
    test_not_converged: Dict[str, int] = field(default_factory=dict)
    inner_seconds: Dict[str, float] = field(default_factory=dict)
    inner_fresh_fits: Dict[str, int] = field(default_factory=dict)

    @property
    def n_train(self):
        return len(self.train_idx)

    @property
    def n_test(self):
        return len(self.test_idx)

    def inner_rates(self):
        return {t: k / self.n_train for t, k in self.inner_successes.items()}

    @property
    def test_rate(self):
        return self.test_successes / self.n_test

    @property
    def is_tied(self):
        return len(self.tied) > 1

    @property
    def lr_not_converged(self):
        """Fits of this fold with a ConvergenceWarning: all inner loops
        plus the refit of the selected technique."""
        return int(sum(self.inner_not_converged.values())
                   + self.test_not_converged.get(self.selected, 0))


@dataclass
class NestedCVResult:
    """Result of the nested cross-validation of one approach.

    techniques : technique names in table order.
    folds      : the FoldResult of every outer fold.
    result     : cdm.eval.loo.EvalResult over all cases (protocol
                 "nested_cv"): the accumulated test outcomes (T1 / T2),
                 with the fold and the selected technique of every case.
    """
    approach: str
    techniques: Tuple[str, ...]
    folds: List[FoldResult]
    result: L.EvalResult
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_success(self):
        return self.result.n_success

    @property
    def success_rate(self):
        return self.result.success_rate

    def selected(self):
        """[selected technique of fold 1, fold 2, ...]."""
        return [f.selected for f in self.folds]

    def tied_folds(self):
        """The folds in which a tie rule decided the selection."""
        return [f for f in self.folds if f.is_tied]

    def tie_report(self):
        """One dict per tied fold: fold, tied techniques, inner count,
        selected, and the test successes under each tied technique
        (amendment AM18)."""
        return [{"fold": f.fold, "tied": list(f.tied),
                 "inner_successes": f.inner_successes[f.selected],
                 "selected": f.selected, "n_test": f.n_test,
                 "test_successes_by_technique": dict(f.tied_test_successes)}
                for f in self.tied_folds()]

    def total_range_over_ties(self):
        """(lowest, highest) accumulated test successes over every way of
        resolving the ties; both equal `n_success` when no fold is tied."""
        lo = sum(min(f.tied_test_successes.values()) for f in self.folds)
        hi = sum(max(f.tied_test_successes.values()) for f in self.folds)
        return lo, hi

    def not_converged(self):
        """{technique: fits with a ConvergenceWarning over all inner loops}
        and "test_refits": the same for the refits of the selected
        techniques."""
        out = {t: int(sum(f.inner_not_converged.get(t, 0) for f in self.folds))
               for t in self.techniques}
        out["test_refits"] = int(sum(f.test_not_converged.get(f.selected, 0)
                                     for f in self.folds))
        return out

    def table_rows(self):
        """One dict per fold: fold, n_train, n_test, the inner success
        rate of every technique, `selected`, `test` (rate) and
        `test_successes`: the content of TH Table 5.1 / 5.2."""
        rows = []
        for f in self.folds:
            row = {"fold": f.fold, "n_train": f.n_train, "n_test": f.n_test}
            row.update(f.inner_rates())
            row.update({"selected": f.selected, "test": f.test_rate,
                        "test_successes": f.test_successes})
            rows.append(row)
        return rows

    def to_frame(self):
        """The table as a pandas DataFrame indexed by "Fold k"; columns:
        the techniques, then "Test" (as in TH Table 5.1 / 5.2), then
        "Selected", "n_train", "n_test", "Test successes"."""
        import pandas as pd
        rows = self.table_rows()
        data = {t: [r[t] for r in rows] for t in self.techniques}
        data["Test"] = [r["test"] for r in rows]
        data["Selected"] = [r["selected"] for r in rows]
        data["n_train"] = [r["n_train"] for r in rows]
        data["n_test"] = [r["n_test"] for r in rows]
        data["Test successes"] = [r["test_successes"] for r in rows]
        return pd.DataFrame(data, index=[f"Fold {r['fold']}" for r in rows])

    def format_table(self, decimals=3):
        """Text table shaped like TH Table 5.1 / 5.2. The best technique
        of a fold (bold in the thesis) is marked with '*'. Rates are
        rounded half up, as in the thesis (cdm.eval.stats.round_half_up)."""
        from cdm.eval.stats import round_half_up
        w = max(8, max(len(t) for t in self.techniques) + 2)
        head = f"{'Classifier':<10}" + "".join(f"{t:>{w}}" for t in self.techniques) \
            + f"{'Test':>{w}}"
        lines = [head]
        for f in self.folds:
            cells = []
            for t in self.techniques:
                v = round_half_up(f.inner_successes[t], f.n_train, decimals)
                mark = "*" if t == f.selected else ""
                cells.append(f"{mark + format(v, f'.{decimals}f'):>{w}}")
            test = round_half_up(f.test_successes, f.n_test, decimals)
            lines.append(f"{'Fold ' + str(f.fold):<10}" + "".join(cells)
                         + f"{format(test, f'.{decimals}f'):>{w}}")
        lines.append(f"Accumulated test successes: {self.n_success}/{self.result.n}"
                     f" = {self.success_rate:.3f}")
        return "\n".join(lines)

    def identical_inner(self, a, b):
        """True iff the techniques a and b succeed on exactly the same
        training cases in every fold (the plan's fingerprint: BR+KNN and
        LP+KNN, plan §5.2)."""
        return all(np.array_equal(f.inner_success[a], f.inner_success[b])
                   for f in self.folds)


# ---- the run ----------------------------------------------------------------------------

def _success_from_outputs(spec, outputs, Y):
    """(n,) 0/1 success of the AMP / DAP protocol from model outputs."""
    if spec.approach == "AMP":
        return np.asarray(M.amp_success(Y, M.select_methods(outputs, spec.tie)), dtype=int)
    return (np.asarray(outputs)[:, 0].astype(int) == np.asarray(Y)).astype(int)


def _safe(name):
    return name.replace("+", "_")


def run_nested_cv(approach, X, Y, techniques=None, labels=L.LABELS, n_splits=None,
                  technique_tie=FROM_CONFIG, case_ids=None, domains=None,
                  method_answers=None, answer_sets=None, rule_outcomes=None,
                  hyper=None, lr_max_iter=None, tie=None, config=None, n_jobs=1,
                  batch_size=L.DEFAULT_BATCH_SIZE, checkpoint_dir=None,
                  on_fold=None):
    """Leave-one-out within K-fold cross-validation (plan §5.2).

    approach   : "AMP" or "DAP".
    X, Y       : AMP: features (n, d) and method outcomes (n, q); DAP:
                 features (n, d) and the 0/1 code of a* (n,). The row
                 order IS the fold order (plan OQ16).
    techniques : AMP: (transform, base) pairs, default the 9 of TH Table
                 5.1; DAP: names, default KNN, LR, RF. Table order.
    n_splits   : default: plan §8 OQ16 (10).
    technique_tie : rule for two techniques with the same highest inner
                 score (see `select_technique`). Default: the config's
                 `technique_tie_rule` (AM18: "first_in_table_order").
                 None: raise TechniqueTieError on a tie; this needs a
                 `checkpoint_dir`, else ValueError.
    hyper, lr_max_iter, config : see cdm.eval.loo.resolve_hyper.
    tie        : AMP argmax ties, plan §8 OQ17; default: the config value.
    n_jobs, batch_size : joblib workers and cases per job for the inner
                 LOO loops (cdm.eval.loo.loo_outputs).
    checkpoint_dir : see the module docstring.
    on_fold    : optional callable(FoldResult), called when a fold is
                 complete (for progress files). Not called when the rule
                 is None (the folds are then completed only at the end).
    Other arguments: stored in the EvalResult (see cdm.eval.loo).

    Order of work, per fold: (1) the inner LOO loops of all techniques on
    the 90%; (2) selection; (3) refit of the selected technique (and of
    every technique tied with it) on the 90% and test on the 10%. With
    the rule None, (2) and (3) are done for all folds after all inner
    loops, so that every tie is reported at once. The numbers are the
    same either way (every fit depends only on its data and the seed).
    Returns a NestedCVResult.
    """
    if n_splits is None:
        n_splits = fold_scheme_from_config(config)
    if technique_tie == FROM_CONFIG:
        technique_tie = technique_tie_from_config(config)
    if technique_tie is not None and technique_tie not in TECHNIQUE_TIE_RULES:
        raise ValueError(f"unknown technique tie rule {technique_tie!r}")
    if technique_tie is None and checkpoint_dir is None:
        raise ValueError(
            "technique_tie is None (stop on a tie) and no checkpoint_dir is "
            "given: a tie would raise after all inner loops and lose them. "
            f"Give a checkpoint_dir or name a rule (one of {TECHNIQUE_TIE_RULES})")
    if techniques is None:
        techniques = M.AMP_TECHNIQUES_TH if approach == "AMP" else M.DAP_TECHNIQUES
    specs = [L.make_spec(approach, t, labels, hyper, lr_max_iter, tie, config)
             for t in techniques]
    names = [s.name for s in specs]
    if len(set(names)) != len(names) or not names:
        raise ValueError("techniques must be unique and non-empty")
    by_name = dict(zip(names, specs))
    X, Y = L._check_xy(specs[0], X, Y)
    n = len(X)
    splits = outer_folds(n, n_splits)
    t0 = time.time()
    outputs = np.full((n, specs[0].n_out), np.nan)
    fold_of = np.zeros(n, dtype=int)
    tech_of = [None] * n
    folds = []

    def finish_fold(k, tr, te, per_tech, bad, timing):
        """(2) + (3) for fold k; returns its FoldResult."""
        c = {t: int(v.sum()) for t, v in per_tech.items()}
        best = max(c.values())
        tied = tuple(t for t in names if c[t] == best)
        chosen = select_technique([c[t] for t in names], names, technique_tie)
        tied_ok, tied_n, test_bad = {}, {}, {}
        for t in tied:
            out, n_warn = L.fit_predict_counted(by_name[t], X[tr], Y[tr], X[te])
            ok = _success_from_outputs(by_name[t], out, Y[te])
            tied_ok[t], tied_n[t], test_bad[t] = ok, int(ok.sum()), int(n_warn > 0)
            if t == chosen:
                outputs[te] = out
        fold_of[te] = k
        for i in te:
            tech_of[i] = chosen
        f = FoldResult(fold=k, train_idx=tr, test_idx=te, inner_successes=c,
                       inner_success=per_tech, selected=chosen, tied=tied,
                       test_successes=tied_n[chosen], tied_test_successes=tied_n,
                       tied_test_success=tied_ok, inner_not_converged=bad,
                       test_not_converged=test_bad,
                       inner_seconds={t: v[0] for t, v in timing.items()},
                       inner_fresh_fits={t: v[1] for t, v in timing.items()})
        LOG.info("%s fold %d/%d: selected %s, test %d/%d = %.3f%s", approach, k,
                 n_splits, chosen, f.test_successes, len(te), f.test_rate,
                 "" if len(tied) == 1 else
                 f" (TIE at {best} between {list(tied)}, rule {technique_tie}; "
                 f"test successes under each: {tied_n})")
        return f

    inner, inner_bad, inner_time = [], [], []
    for k, (tr, te) in enumerate(splits, start=1):
        # (1) inner leave-one-out of every technique on this 90%
        per_tech, bad, timing = {}, {}, {}
        for spec in specs:
            ck = None
            if checkpoint_dir is not None:
                ck = Path(checkpoint_dir) / f"fold_{k:02d}" / _safe(spec.name)
            out, info = L.loo_outputs(
                spec, X[tr], Y[tr], n_jobs, batch_size, ck,
                label=f"{approach} fold {k}/{n_splits} {spec.name} inner LOO",
                return_info=True)
            per_tech[spec.name] = _success_from_outputs(spec, out, Y[tr])
            bad[spec.name] = int(info["n_not_converged"])
            timing[spec.name] = (float(info["seconds"]), int(info["n_fresh"]))
            LOG.info("%s fold %d/%d %s: inner LOO %d/%d = %.3f, %d fits not "
                     "converged (%.0f s since start)",
                     approach, k, n_splits, spec.name, int(per_tech[spec.name].sum()),
                     len(tr), per_tech[spec.name].mean(), bad[spec.name],
                     time.time() - t0)
        inner.append(per_tech)
        inner_bad.append(bad)
        inner_time.append(timing)
        if technique_tie is not None:
            folds.append(finish_fold(k, tr, te, per_tech, bad, timing))
            if on_fold is not None:
                on_fold(folds[-1])

    if technique_tie is None:
        counts = [{t: int(v.sum()) for t, v in per_tech.items()} for per_tech in inner]
        problems = []
        for k, c in enumerate(counts, start=1):
            best = max(c.values())
            tied = tuple(t for t in names if c[t] == best)
            if len(tied) > 1:
                problems.append((k, tied, c))
        if problems:
            text = "; ".join(f"fold {k}: {list(t)} at {c[t[0]]}" for k, t, c in problems)
            raise TechniqueTieError(
                "tie for the highest inner score and no rule named (the plan gives "
                f"none): {text}. All inner scores: {counts}", problems[0][1],
                counts=counts, inner_success=inner)
        for k, (tr, te) in enumerate(splits, start=1):
            folds.append(finish_fold(k, tr, te, inner[k - 1], inner_bad[k - 1],
                                     inner_time[k - 1]))

    chosen = [f.selected for f in folds]
    lr_bad = int(sum(f.lr_not_converged for f in folds))
    meta = L._meta(specs[0], techniques=names, n_splits=n_splits,
                   technique_tie=technique_tie, selected=chosen,
                   tied_folds=[f.fold for f in folds if f.is_tied],
                   lr_not_converged=lr_bad,
                   lr_not_converged_by_technique={
                       t: int(sum(f.inner_not_converged[t] for f in folds))
                       for t in names})
    del meta["technique"]
    if approach == "AMP":
        result = L.amp_result(specs[0], outputs, Y, "nested_cv", techniques=tech_of,
                              case_ids=case_ids, domains=domains,
                              method_answers=method_answers, folds=fold_of, meta=meta)
    else:
        result = L.dap_result(specs[0], outputs, Y, "nested_cv", techniques=tech_of,
                              case_ids=case_ids, domains=domains,
                              answer_sets=answer_sets, rule_outcomes=rule_outcomes,
                              folds=fold_of, meta=meta)
    assert result.n_success == sum(f.test_successes for f in folds)
    return NestedCVResult(approach=approach, techniques=tuple(names), folds=folds,
                          result=result, meta=meta)


def run_amp_nested_cv(X, Y, **kwargs):
    """`run_nested_cv("AMP", X, Y, ...)`: TH Table 5.1 and T1."""
    return run_nested_cv("AMP", X, Y, **kwargs)


def run_dap_nested_cv(X, y, **kwargs):
    """`run_nested_cv("DAP", X, y, ...)`: TH Table 5.2 and T2."""
    return run_nested_cv("DAP", X, y, **kwargs)
