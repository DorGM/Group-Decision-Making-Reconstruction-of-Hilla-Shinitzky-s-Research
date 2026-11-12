"""Leave-one-out evaluation of AMP and DAP, and the result object shared
with the nested cross-validation (cdm.eval.nested_cv).

Step A13 of plan §9.3 ("MLA LOO evaluation (BR+RF / RF)" -> T19-T22).
Protocol (plan §5.3; MLA §5.5; TH §4.1.5 p.48): a single leave-one-out
over all cases with one technique (AMP: BR+RF, DAP: RF). For every case i
the model is fitted on all other cases and applied to case i:
  AMP (plan §4.1): the selected method is the label with the highest
      P(O_AG = 1 | x) (ties: plan §8 OQ17, first label in the order);
      AMP succeeds iff the answer of the selected method is a*, i.e. iff
      O of the selected label is 1 (cdm.models.multilabel.amp_success).
  DAP (plan §4.2): the predicted class is the returned answer; DAP
      succeeds iff it is a*.
Success rate = share of successful cases (plan §5.1).

What is here
  * `Spec`           : approach + technique + hyperparameters (picklable).
  * `loo_outputs`    : the generic LOO loop (joblib, checkpoints, log); also
                       used for the inner loops of the nested CV.
  * `run_amp_loo`, `run_dap_loo` : the LOO runs, returning an `EvalResult`.
  * `EvalResult`     : per-case outcomes plus the counts that plan T19-T22
                       (LOO) and T1/T2/T8/T10-T13 (10-fold) are made of:
                       selection counts, P(success | selected), per-domain
                       breakdown, discordant pairs against a rule. The
                       tests and tables themselves are cdm.eval.stats (A3)
                       and step A15.

Seeds (plan §8 OQ15 "Seeds fixed and recorded (0)"): every fit builds a
fresh model whose `random_state` is the `seed` hyperparameter
(cdm.models.multilabel.make_base). The result of the fit for case i
therefore depends only on the data, the technique and that seed, not on
the number of jobs, the batch size or the order in which jobs finish.

Refitting: every case is a real refit, for every technique including KNN.
No shortcut is used (a KNN shortcut would need a proof of equivalence
under distance ties; KNN fits are cheap).

Checkpoints: with `checkpoint_dir`, the outputs of every finished batch of
cases are written to `batch_<fp12>_<start>_<stop>.npz` (atomic rename;
<fp12> = the first 12 hex digits of the fingerprint; arrays `out` and
`not_converged`), and `meta.json` stores a SHA-256 fingerprint of X, Y, the
Spec, the library versions and the project's own model code
(`code_fingerprint`, CHECKPOINT_VERSION). A later call with the same inputs
reuses the finished batches; a different fingerprint raises instead of
mixing runs, and so does a folder that holds batch files without a
`meta.json` (decision E3).

Convergence (decision E2; plan OQ15 "max_iter raised to avoid
non-convergence"): every fit is run with the warnings recorded;
sklearn `ConvergenceWarning`s are counted, never silenced into nothing: the
count of fits that did not converge is returned with the outputs, logged,
kept in the checkpoint and stored in `EvalResult.meta["lr_not_converged"]`.
The fit itself is not changed.

Nothing in this module reads a data build. The caller passes the arrays.
"""
import hashlib
import inspect
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cdm.eval import stats as S
from cdm.models import multilabel as M

LOG = logging.getLogger("cdm.eval")

APPROACHES = ("AMP", "DAP")
#: Thesis label order (TH p.37), = cdm.approaches.instances.LABEL_ORDER.
LABELS = ("MR", "HAC", "WC", "SP", "DA")
DEFAULT_BATCH_SIZE = 16
#: Raise by hand when a change of the project code makes old checkpoint
#: batches invalid in a way the source hash below does not see.
CHECKPOINT_VERSION = 2


# ---- technique specification ------------------------------------------------------

def _freeze(hyper):
    return tuple(sorted(dict(hyper).items()))


@dataclass(frozen=True)
class Spec:
    """One technique of one approach, with everything a fit needs.

    approach  : "AMP" or "DAP".
    technique : AMP: (transform, base), e.g. ("BR", "RF"); DAP: the
                classifier name, e.g. "RF".
    labels    : AMP only: names of the label columns (chain order OQ19 and
                tie order OQ17).
    hyper     : sorted tuple of (name, value): the keyword arguments of
                cdm.models.multilabel.make_base.
    tie       : AMP only: plan §8 OQ17 rule.
    """
    approach: str
    technique: Any
    labels: Tuple[str, ...] = ()
    hyper: Tuple[Tuple[str, Any], ...] = ()
    tie: str = M.OQ17_IMPLEMENTED

    @property
    def name(self):
        if self.approach == "AMP":
            return M.technique_name(*self.technique)
        return str(self.technique)

    @property
    def n_out(self):
        """Columns of the per-case output: AMP q probabilities; DAP
        (predicted code, P(code = 1))."""
        return len(self.labels) if self.approach == "AMP" else 2

    def make(self):
        """A fresh unfitted model."""
        if self.approach == "AMP":
            return M.make_amp_model(self.technique[0], self.technique[1],
                                    self.labels, **dict(self.hyper))
        return M.make_dap_model(self.technique, **dict(self.hyper))

    def describe(self):
        return {"approach": self.approach, "technique": self.name,
                "labels": list(self.labels),
                "hyper": {k: v for k, v in self.hyper}, "tie": self.tie}


def resolve_hyper(hyper=None, lr_max_iter=None, config=None):
    """Hyperparameters of a run: `hyper` (default: the config's plan §8
    OQ15 values, cdm.models.multilabel.hyperparameters_from_config), with
    `lr_max_iter` replacing the entry of the same name when it is given.

    The config's `lr_max_iter` is 10000 (amendment AM18, docs/IMPLEMENTATION_DECISIONS.md M9;
    the plan gives no number). A `hyper` dict whose `lr_max_iter` is None
    still makes an LR technique raise: this module never picks a number.
    """
    hyper = dict(hyper) if hyper is not None else M.hyperparameters_from_config(config)
    if lr_max_iter is not None:
        hyper["lr_max_iter"] = int(lr_max_iter)
    return hyper


def make_spec(approach, technique, labels=LABELS, hyper=None, lr_max_iter=None,
              tie=None, config=None):
    """Build and validate a Spec.

    approach  : "AMP" / "DAP".
    technique : AMP: (transform, base) or "BR+RF"; DAP: "KNN" / "LR" / "RF".
    tie       : AMP: plan §8 OQ17 rule; default: the config value.
    The model is built once here so that a bad technique or a missing LR
    `max_iter` fails before any job starts.
    """
    if approach not in APPROACHES:
        raise ValueError(f"unknown approach {approach!r}; expected one of {APPROACHES}")
    hyper = resolve_hyper(hyper, lr_max_iter, config)
    if approach == "AMP":
        if isinstance(technique, str):
            technique = tuple(technique.split("+"))
        technique = tuple(technique)
        if len(technique) != 2:
            raise ValueError(f"an AMP technique is (transform, base), got {technique!r}")
        tie = M.tie_rule_from_config(config) if tie is None else tie
        M._check_tie_rule(tie)
        M.check_oq19(config)
        spec = Spec("AMP", technique, tuple(labels), _freeze(hyper), tie)
    else:
        if technique not in M.DAP_TECHNIQUES:
            raise ValueError(f"unknown DAP classifier {technique!r}; "
                             f"expected one of {M.DAP_TECHNIQUES}")
        spec = Spec("DAP", technique, (), _freeze(hyper), "")
    spec.make()                                    # validate now (LR max_iter etc.)
    return spec


def fit_predict_counted(spec, X_train, Y_train, X_test):
    """Fit a fresh model on the training part and return
    (outputs for the test rows: array (n_test, spec.n_out),
     number of sklearn ConvergenceWarnings raised by the fit).

    AMP: the per-label probabilities P(O_AG = 1 | x), columns as
         spec.labels. The selection is made later from these numbers with
         cdm.models.multilabel.select_methods (plain argmax, OQ17).
    DAP: column 0 = the predicted 0/1 code (sklearn `predict`, plan §4.2),
         column 1 = P(code = 1) (for the tie log only, docs/IMPLEMENTATION_DECISIONS.md M7).

    The warnings of the fit are recorded (decision E2): convergence
    warnings are counted (one per base classifier that stopped at
    `max_iter`: up to q for BR / CC + LR, 1 for LP + LR and for DAP LR);
    every other warning is passed on unchanged. The fit is not changed.
    """
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = spec.make().fit(X_train, Y_train)
    n_warn = 0
    for w in caught:
        if issubclass(w.category, ConvergenceWarning):
            n_warn += 1
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if spec.approach == "AMP":
        return np.asarray(model.predict_proba(X_test), dtype=float), n_warn
    return np.column_stack([model.predict(X_test).astype(float),
                            model.predict_proba(X_test)]), n_warn


def fit_predict(spec, X_train, Y_train, X_test):
    """`fit_predict_counted` without the warning count: the outputs only."""
    return fit_predict_counted(spec, X_train, Y_train, X_test)[0]


# ---- the leave-one-out loop ------------------------------------------------------

def _loo_batch(spec, X, Y, start, stop, path):
    """LOO outputs of the cases start..stop-1 (each fitted on all other
    rows of X, Y), and per case 1 if its fit raised a ConvergenceWarning,
    else 0. Written to `path` (atomic) when a path is given.
    Returns (start, stop, out, not_converged)."""
    n = len(X)
    out = np.empty((stop - start, spec.n_out))
    bad = np.zeros(stop - start, dtype=np.int64)
    keep = np.ones(n, dtype=bool)
    for i in range(start, stop):
        keep[i] = False
        o, n_warn = fit_predict_counted(spec, X[keep], Y[keep], X[i:i + 1])
        out[i - start] = o[0]
        bad[i - start] = 1 if n_warn else 0
        keep[i] = True
    if path is not None:
        _atomic_save(path, out, bad)
    return start, stop, out, bad


def _atomic_save(path, out, not_converged):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, out=out, not_converged=not_converged)
    os.replace(tmp, path)


def _load_batch(path, shape):
    """(out, not_converged) of a batch file; raises ValueError when the
    file is unreadable or does not have the expected shape."""
    try:
        with np.load(path) as z:
            out, bad = np.asarray(z["out"], dtype=float), np.asarray(z["not_converged"])
    except Exception as exc:                      # truncated / foreign file
        raise ValueError(f"checkpoint file {path} cannot be read ({exc}); "
                         "delete it to have the batch recomputed") from exc
    if out.shape != shape or bad.shape != (shape[0],):
        raise ValueError(f"checkpoint file {path} has shape {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"checkpoint file {path} holds NaN or inf")
    return out, bad.astype(np.int64)


def code_fingerprint():
    """SHA-256 over CHECKPOINT_VERSION and the source of the project code
    that decides what a fit returns: the whole of
    cdm/models/multilabel.py, and `fit_predict_counted` and `_loo_batch`
    of this module (decision E3)."""
    h = hashlib.sha256()
    h.update(str(CHECKPOINT_VERSION).encode())
    h.update(Path(M.__file__).read_bytes())
    h.update(_OWN_SOURCE.encode())
    return h.hexdigest()


#: Source of the two functions of this module that produce a batch, read
#: once at import (so a test double put in their place does not change it).
_OWN_SOURCE = inspect.getsource(fit_predict_counted) + inspect.getsource(_loo_batch)


def _versions():
    from importlib.metadata import version
    return {p: version(p) for p in ("numpy", "scikit-learn", "scikit-multilearn")}


def fingerprint(spec, X, Y, extra=None):
    """SHA-256 over the Spec, the library versions, the project's model
    code (`code_fingerprint`), `extra`, and the bytes of X and Y.
    Identifies what a checkpoint was computed from."""
    h = hashlib.sha256()
    h.update(json.dumps({"spec": spec.describe(), "versions": _versions(),
                         "code": code_fingerprint(), "extra": extra},
                        sort_keys=True, default=str).encode())
    for a in (np.ascontiguousarray(X, dtype=float), np.ascontiguousarray(Y, dtype=np.int64)):
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _open_checkpoint(checkpoint_dir, meta):
    """Create the folder and `meta.json`, or check an existing one.
    A folder written for other inputs raises ValueError; so does a folder
    that holds batch files but no `meta.json` (their origin is unknown;
    decision E3)."""
    d = Path(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / "meta.json"
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old.get("fingerprint") != meta["fingerprint"]:
            raise ValueError(
                f"checkpoint folder {d} was written for other inputs "
                f"(fingerprint {old.get('fingerprint')!r:.20} != {meta['fingerprint']!r:.20}); "
                "use another folder or delete it")
    else:
        stale = sorted(p.name for p in d.glob("batch_*"))
        if stale:
            raise ValueError(
                f"checkpoint folder {d} holds {len(stale)} batch file(s) "
                f"(first: {stale[0]}) but no meta.json, so what they were "
                "computed from is unknown; use another folder or delete them")
        tmp = d / f".meta.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(meta, indent=1, sort_keys=True, default=str),
                       encoding="utf-8")
        os.replace(tmp, f)
    return d


def _check_xy(spec, X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y)
    if X.ndim != 2:
        raise ValueError("X must be (n, d)")
    if spec.approach == "AMP":
        if Y.ndim != 2 or Y.shape != (len(X), len(spec.labels)):
            raise ValueError(f"Y must be (n, {len(spec.labels)}) for the labels {spec.labels}")
    elif Y.ndim != 1 or len(Y) != len(X):
        raise ValueError("y must be (n,)")
    if not np.isfinite(X).all():
        raise ValueError("X holds NaN or inf")
    if not np.isin(Y, (0, 1)).all():
        raise ValueError("Y must hold 0/1 only")
    return X, Y.astype(int)


def loo_outputs(spec, X, Y, n_jobs=1, batch_size=DEFAULT_BATCH_SIZE,
                checkpoint_dir=None, label=None, return_info=False):
    """Leave-one-out outputs of a technique: array (n, spec.n_out), row i
    from a model fitted on all rows except i (see `fit_predict`).

    n_jobs         : joblib workers (processes); 1 = no joblib.
    batch_size     : cases per job and per checkpoint file. Affects speed
                     and the checkpoint grain only, never the numbers.
    checkpoint_dir : folder for the batch files (see the module docstring);
                     None = no checkpoint.
    label          : text for the progress log (logger "cdm.eval", INFO).
    return_info    : True: return (outputs, info) with info =
                     {"not_converged": (n,) 0/1, the fits that raised a
                      ConvergenceWarning; "n_not_converged": their number;
                      "n_from_checkpoint": cases read from the checkpoint;
                      "seconds": wall time of the fresh fits;
                      "n_fresh": number of fresh fits}.
    """
    X, Y = _check_xy(spec, X, Y)
    n = len(X)
    if n < 2:
        raise ValueError("leave-one-out needs at least 2 cases")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    label = label or f"{spec.approach} {spec.name} LOO"
    out = np.full((n, spec.n_out), np.nan)
    bad = np.zeros(n, dtype=np.int64)
    batches = [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]
    paths = {}
    todo = []
    if checkpoint_dir is not None:
        fp = fingerprint(spec, X, Y, {"batch_size": batch_size})
        d = _open_checkpoint(checkpoint_dir, {
            "fingerprint": fp, "spec": spec.describe(), "n": n,
            "batch_size": batch_size, "versions": _versions(),
            "code_fingerprint": code_fingerprint(),
            "checkpoint_version": CHECKPOINT_VERSION})
        expected = {f"batch_{fp[:12]}_{s:06d}_{e:06d}.npz" for s, e in batches}
        foreign = sorted(p.name for p in d.glob("batch_*") if p.name not in expected)
        if foreign:
            raise ValueError(
                f"checkpoint folder {d} holds {len(foreign)} batch file(s) that "
                f"do not belong to this run (first: {foreign[0]}); use another "
                "folder or delete them")
        for s, e in batches:
            p = d / f"batch_{fp[:12]}_{s:06d}_{e:06d}.npz"
            paths[(s, e)] = p
            if p.exists():
                out[s:e], bad[s:e] = _load_batch(p, (e - s, spec.n_out))
            else:
                todo.append((s, e))
    else:
        todo = list(batches)
    done = n - sum(e - s for s, e in todo)
    if done:
        LOG.info("%s: %d/%d cases read from the checkpoint", label, done, n)

    t0 = time.time()
    n_todo = n - done

    def record(s, e, a, b):
        nonlocal done
        out[s:e] = a
        bad[s:e] = b
        done += e - s
        el = time.time() - t0
        fresh = done - (n - n_todo)
        eta = el / fresh * (n - done) if fresh else float("nan")
        LOG.info("%s: %d/%d cases, %.0f s elapsed, about %.0f s left",
                 label, done, n, el, eta)

    if todo and (n_jobs == 1 or len(todo) == 1):
        for s, e in todo:
            record(*_loo_batch(spec, X, Y, s, e, paths.get((s, e))))
    elif todo:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_loo_batch)(spec, X, Y, s, e, paths.get((s, e))) for s, e in todo)
        for s, e, a, b in results:
            record(s, e, a, b)
    if np.isnan(out).any():
        raise RuntimeError("leave-one-out outputs are incomplete")
    n_bad = int(bad.sum())
    if n_bad:
        LOG.warning("%s: %d of %d fits did NOT converge (ConvergenceWarning; "
                    "max_iter = %s)", label, n_bad, n,
                    dict(spec.hyper).get("lr_max_iter"))
    if return_info:
        return out, {"not_converged": bad, "n_not_converged": n_bad,
                     "n_from_checkpoint": n - n_todo, "n_fresh": n_todo,
                     "seconds": time.time() - t0}
    return out


# ---- result object -----------------------------------------------------------------

def _counts_row(k, n):
    return {"n": int(n), "successes": int(k),
            "rate": (k / n) if n else float("nan")}


@dataclass
class EvalResult:
    """Per-case outcomes of one evaluation run of AMP or DAP.

    Shared by the LOO (plan §5.3) and the 10-fold nested CV (plan §5.2).

    approach   : "AMP" / "DAP".
    protocol   : "loo" or "nested_cv".
    techniques : per case, the name of the technique that produced the
                 outcome ("BR+RF" for every case of a LOO; in the nested
                 CV the technique selected in the case's fold).
    case_ids, domains : per case, as given by the caller (domains may be
                 None).
    success    : (n,) 0/1, the returned answer is a*.
    labels     : names of the methods. AMP: the label columns. DAP: the
                 methods of `outcomes` (may be empty).
    outcomes   : (n, q) 0/1, O_AG of every method (f_AG(R) = a*). AMP: the
                 label matrix Y. DAP: optional, given by the caller, for
                 the comparisons with the rules (plan T22).
    selected   : AMP: (n,) index of the selected label; DAP: None.
    proba      : AMP: (n, q) P(O_AG = 1 | x); DAP: (n,) P(code = 1).
    predicted  : DAP: (n,) predicted 0/1 code; AMP: None.
    target     : DAP: (n,) 0/1 code of a*; AMP: None.
    answers    : optional per-case returned answer (AMP: the answer of the
                 selected method, if the method answers were given; DAP:
                 the decoded answer, if the answer sets were given).
    folds      : nested CV: (n,) outer fold number 1..K; LOO: None.
    meta       : run description (hyperparameters, seed, tie rule, ...).
    """
    approach: str
    protocol: str
    techniques: List[str]
    case_ids: List[Any]
    domains: Optional[List[Any]]
    success: np.ndarray
    labels: Tuple[str, ...] = ()
    outcomes: Optional[np.ndarray] = None
    selected: Optional[np.ndarray] = None
    proba: Optional[np.ndarray] = None
    predicted: Optional[np.ndarray] = None
    target: Optional[np.ndarray] = None
    answers: Optional[List[Any]] = None
    folds: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- basic counts (plan §5.1) --
    def __len__(self):
        return len(self.success)

    @property
    def n(self):
        return len(self.success)

    @property
    def n_success(self):
        return int(np.sum(self.success))

    @property
    def success_rate(self):
        return S.success_rate(self.n_success, self.n)

    @property
    def selected_names(self):
        """AMP: name of the selected method per case."""
        self._amp_only()
        return [self.labels[j] for j in self.selected]

    def _amp_only(self):
        if self.approach != "AMP":
            raise ValueError("defined for AMP only")

    def _method(self, method):
        if self.outcomes is None or method not in self.labels:
            raise ValueError(f"no outcomes stored for the method {method!r}")
        return self.labels.index(method)

    # -- per-method counts (plan T8 / T20 / T21) --
    def method_success_counts(self):
        """{method: number of cases where f_AG(R) = a*} over all cases
        (the "P(success)" row of TH Table 5.4, as a count)."""
        if self.outcomes is None:
            raise ValueError("no method outcomes stored")
        return {m: int(self.outcomes[:, j].sum()) for j, m in enumerate(self.labels)}

    def selection_counts(self):
        """AMP: {method: number of cases where AMP selected it}, in label
        order (plan T20; "P(selected)" of TH Table 5.4 as a count)."""
        self._amp_only()
        c = np.bincount(self.selected, minlength=len(self.labels))
        return {m: int(c[j]) for j, m in enumerate(self.labels)}

    def selected_success_counts(self):
        """AMP: {method: number of cases where AMP selected it AND it was
        correct} (plan T21). Sums to `n_success`."""
        self._amp_only()
        c = np.bincount(self.selected, weights=self.success,
                        minlength=len(self.labels))
        return {m: int(round(c[j])) for j, m in enumerate(self.labels)}

    def success_given_selected(self):
        """AMP: {method: P(success | selected)} = selected-and-correct /
        selected (plan T8, T21). NaN for a method that was never selected
        (the ratio 0/0 is undefined)."""
        sel, ok = self.selection_counts(), self.selected_success_counts()
        return {m: (ok[m] / sel[m]) if sel[m] else float("nan") for m in self.labels}

    def effectiveness(self, tail="sf"):
        """AMP: per method, the counts and statistics of TH Table 5.4 /
        A.10 and MLA Figs 2-3:
          n_selected, p_selected, n_success, p_success (over all cases),
          n_selected_success, p_success_given_selected,
          pre (from exact counts, plan §5.8 item 3),
          z, p (one-sample z-test of P(success | selected) against the
          fixed p0 = P(success), one-tailed; plan §5.8 item 3).
        pre / z / p are NaN where undefined (method never selected, or
        P(success) equal to 0 or 1)."""
        sel, ok = self.selection_counts(), self.selected_success_counts()
        tot = self.method_success_counts()
        rows = {}
        nan = float("nan")
        for m in self.labels:
            row = {"n_selected": sel[m], "p_selected": sel[m] / self.n,
                   "n_success": tot[m], "p_success": tot[m] / self.n,
                   "n_selected_success": ok[m],
                   "p_success_given_selected": ok[m] / sel[m] if sel[m] else nan,
                   "pre": nan, "z": nan, "p": nan}
            if sel[m] and 0 < tot[m] < self.n:
                row["pre"] = S.pre_from_counts(tot[m], self.n, ok[m], sel[m])
                row["z"], row["p"] = S.one_sample_proportion_ztest(
                    ok[m], sel[m], tot[m] / self.n, tail=tail)
            rows[m] = row
        return rows

    # -- comparisons with a rule (plan T22, §5.8) --
    def discordant(self, method):
        """(b, c): cases where only the approach is correct, cases where
        only the method is correct (the order of plan T22)."""
        j = self._method(method)
        return S.discordant_counts(self.success, self.outcomes[:, j])

    def mcnemar(self, method):
        """McNemar (chi2, p) against a method (plan §5.8 item 4)."""
        return S.mcnemar_test(*self.discordant(method))

    def proportion_test(self, method, tail="sf"):
        """Pooled one-tailed two-proportion z-test (z, p), approach >
        method, n1 = n2 = n (plan §5.8 items 1 and 5)."""
        j = self._method(method)
        return S.two_proportion_ztest(self.n_success, self.n,
                                      int(self.outcomes[:, j].sum()), self.n,
                                      alternative="larger", tail=tail)

    # -- sub-sets --
    def subset(self, mask):
        """The result restricted to the cases of a boolean mask."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n,):
            raise ValueError("mask must be (n,)")
        idx = np.flatnonzero(mask)
        pick = lambda xs: None if xs is None else [xs[i] for i in idx]  # noqa: E731
        arr = lambda a: None if a is None else a[idx]                   # noqa: E731
        return EvalResult(
            approach=self.approach, protocol=self.protocol,
            techniques=pick(self.techniques), case_ids=pick(self.case_ids),
            domains=pick(self.domains), success=self.success[idx],
            labels=self.labels, outcomes=arr(self.outcomes),
            selected=arr(self.selected), proba=arr(self.proba),
            predicted=arr(self.predicted), target=arr(self.target),
            answers=pick(self.answers), folds=arr(self.folds), meta=self.meta)

    def domain_names(self):
        """Domains in order of first appearance (the data order)."""
        if self.domains is None:
            raise ValueError("no domains stored")
        return list(dict.fromkeys(self.domains))

    def by_domain(self):
        """{domain: EvalResult of its cases} (plan T10-T13)."""
        dom = np.array(self.domain_names(), dtype=object)
        d = np.array(self.domains, dtype=object)
        return {x: self.subset(d == x) for x in dom}

    def domain_table(self):
        """{domain: {n, successes, rate}} plus "all" (plan T10, T12)."""
        t = {x: _counts_row(r.n_success, r.n) for x, r in self.by_domain().items()}
        t["all"] = _counts_row(self.n_success, self.n)
        return t

    # -- logs --
    def tie_log(self):
        """How often an unstated tie rule decided a case.
        AMP: `tied_rows` = rows where several labels share the maximum
             probability (plan §8 OQ17 decides); `rounding_tie_rows` =
             rows where float noise, not the label order, decided
             (LP+RF, docs/IMPLEMENTATION_DECISIONS.md M8).
        DAP: `half_rows` = rows with P(code = 1) exactly 0.5 (sklearn
             returns code 0; docs/IMPLEMENTATION_DECISIONS.md M7)."""
        if self.approach == "AMP":
            return {"tied_rows": int((M.tied_labels(self.proba).sum(axis=1) > 1).sum()),
                    "rounding_tie_rows": M.rounding_tie_rows(self.proba)}
        return {"half_rows": int((self.proba == 0.5).sum())}

    def summary(self):
        """Counts for a log or a report, as plain Python types."""
        out = {"approach": self.approach, "protocol": self.protocol,
               "n": self.n, "successes": self.n_success, "rate": self.success_rate,
               "techniques": dict(zip(*np.unique(self.techniques, return_counts=True))),
               "ties": self.tie_log()}
        out["techniques"] = {str(k): int(v) for k, v in out["techniques"].items()}
        if self.approach == "AMP":
            out["selected"] = self.selection_counts()
            out["selected_success"] = self.selected_success_counts()
        if self.domains is not None:
            out["domains"] = self.domain_table()
        if "lr_not_converged" in self.meta:
            out["lr_not_converged"] = self.meta["lr_not_converged"]
        return out


# ---- building results from model outputs -------------------------------------------

def _list_or_default(xs, n, what):
    if xs is None:
        return None
    xs = list(xs)
    if len(xs) != n:
        raise ValueError(f"{what} must have one entry per case")
    return xs


def amp_result(spec, outputs, Y, protocol, techniques=None, case_ids=None,
               domains=None, method_answers=None, folds=None, meta=None):
    """EvalResult of AMP from the per-label probabilities `outputs` (n, q)
    and the outcomes Y (n, q). Selection: plan §4.1 with the OQ17 rule of
    the Spec; success: O of the selected label.

    method_answers : optional (n, q) answers f_AG(R) in label order; the
                     returned answer of a case is then stored.
    """
    Y = np.asarray(Y).astype(int)
    proba = np.asarray(outputs, dtype=float)
    selected = M.select_methods(proba, spec.tie)
    n = len(Y)
    answers = None
    if method_answers is not None:
        if len(method_answers) != n or any(len(r) != Y.shape[1] for r in method_answers):
            raise ValueError("method_answers must be (n, q)")
        answers = [method_answers[i][j] for i, j in enumerate(selected)]
    return EvalResult(
        approach="AMP", protocol=protocol,
        techniques=list(techniques) if techniques is not None else [spec.name] * n,
        case_ids=_list_or_default(case_ids, n, "case_ids") or list(range(n)),
        domains=_list_or_default(domains, n, "domains"),
        success=np.asarray(M.amp_success(Y, selected), dtype=int),
        labels=tuple(spec.labels), outcomes=Y, selected=np.asarray(selected, dtype=int),
        proba=proba, answers=answers,
        folds=None if folds is None else np.asarray(folds, dtype=int),
        meta=dict(meta or {}))


def dap_result(spec, outputs, y, protocol, techniques=None, case_ids=None,
               domains=None, answer_sets=None, rule_outcomes=None, folds=None,
               meta=None):
    """EvalResult of DAP from `outputs` (n, 2) = (predicted code,
    P(code = 1)) and the target codes y. Success: predicted code = code of
    a* (plan §4.2).

    answer_sets   : optional per-case (a1, a2); the returned answer is
                    then decoded (cdm.approaches.dap.decode_answer).
    rule_outcomes : optional {method: (n,) 0/1 O_AG}, stored for the
                    comparisons with the rules (plan T22).
    """
    y = np.asarray(y).astype(int)
    outputs = np.asarray(outputs, dtype=float)
    pred = outputs[:, 0].astype(int)
    n = len(y)
    answers = None
    if answer_sets is not None:
        from cdm.approaches.dap import decode_answer
        if len(answer_sets) != n:
            raise ValueError("answer_sets must have one entry per case")
        answers = [decode_answer(int(c), a) for c, a in zip(pred, answer_sets)]
    labels, outcomes = (), None
    if rule_outcomes:
        labels = tuple(rule_outcomes)
        outcomes = np.column_stack([np.asarray(rule_outcomes[m]).astype(int)
                                    for m in labels])
        if outcomes.shape[0] != n:
            raise ValueError("rule_outcomes must have one entry per case")
    return EvalResult(
        approach="DAP", protocol=protocol,
        techniques=list(techniques) if techniques is not None else [spec.name] * n,
        case_ids=_list_or_default(case_ids, n, "case_ids") or list(range(n)),
        domains=_list_or_default(domains, n, "domains"),
        success=(pred == y).astype(int), labels=labels, outcomes=outcomes,
        proba=outputs[:, 1], predicted=pred, target=y, answers=answers,
        folds=None if folds is None else np.asarray(folds, dtype=int),
        meta=dict(meta or {}))


def _meta(spec, **more):
    out = spec.describe()
    out.update(more)
    out["versions"] = _versions()
    return out


def _run_info(info):
    """Meta entries of a LOO run from the `loo_outputs` info:
    `lr_not_converged` = number of fits with a ConvergenceWarning (0 for
    KNN / RF, which have no such warning), `n_fits`, `fit_seconds`,
    `n_from_checkpoint`."""
    return {"lr_not_converged": int(info["n_not_converged"]),
            "n_fits": int(len(info["not_converged"])),
            "n_from_checkpoint": int(info["n_from_checkpoint"]),
            "fit_seconds": float(info["seconds"])}


# ---- the LOO runs (plan §5.3) ---------------------------------------------------------

def run_amp_loo(X, Y, technique=("BR", "RF"), labels=LABELS,
                case_ids=None, domains=None, method_answers=None, hyper=None,
                lr_max_iter=None, tie=None, config=None, n_jobs=1,
                batch_size=DEFAULT_BATCH_SIZE, checkpoint_dir=None):
    """Leave-one-out evaluation of AMP with one technique (plan §5.3;
    default BR+RF, the technique of MLA §5.5 and TH Tables 5.6 / A.9 /
    A.10). Returns an EvalResult (protocol "loo").

    X, Y    : features (n, d) and the 0/1 outcomes of the methods (n, q),
              e.g. from cdm.approaches.amp.amp_xy.
    labels  : names of the columns of Y (for an ablation row:
              cdm.approaches.instances.methods_kept).
    hyper, lr_max_iter, config : see `resolve_hyper`.
    tie     : plan §8 OQ17; default: the config value.
    n_jobs, batch_size, checkpoint_dir : see `loo_outputs`.
    """
    spec = make_spec("AMP", technique, labels, hyper, lr_max_iter, tie, config)
    out, info = loo_outputs(spec, X, Y, n_jobs, batch_size, checkpoint_dir,
                            return_info=True)
    return amp_result(spec, out, Y, "loo", case_ids=case_ids, domains=domains,
                      method_answers=method_answers,
                      meta=_meta(spec, **_run_info(info)))


def run_dap_loo(X, y, technique="RF", case_ids=None, domains=None,
                answer_sets=None, rule_outcomes=None, hyper=None, lr_max_iter=None,
                config=None, n_jobs=1, batch_size=DEFAULT_BATCH_SIZE,
                checkpoint_dir=None):
    """Leave-one-out evaluation of DAP with one classifier (plan §5.3;
    default RF). Returns an EvalResult (protocol "loo").

    X, y : features (n, d) (27 CDMCR + the f_AG codes) and the 0/1 code of
           a*, e.g. from cdm.approaches.dap.dap_xy.
    """
    spec = make_spec("DAP", technique, hyper=hyper, lr_max_iter=lr_max_iter,
                     config=config)
    out, info = loo_outputs(spec, X, y, n_jobs, batch_size, checkpoint_dir,
                            return_info=True)
    return dap_result(spec, out, y, "loo", case_ids=case_ids, domains=domains,
                      answer_sets=answer_sets, rule_outcomes=rule_outcomes,
                      meta=_meta(spec, **_run_info(info)))


# ---- convenience for an instance table -------------------------------------------------

def amp_inputs(inst, row=1, oq41=None, config=None):
    """Arguments of `run_amp_loo` / nested CV for an instance table
    (cdm.approaches.instances.Instances) and an ablation row:
    dict(X, Y, labels, case_ids, domains, method_answers)."""
    from cdm.approaches import amp, instances as I
    xy = amp.amp_xy(inst, row, oq41, config)
    cfg = I.ablation_config(row)
    methods, rules = I.methods_kept(cfg), I.rules_kept(cfg)
    cols = [inst.method_answers(m, rules) for m in methods]
    return {"X": xy.X, "Y": xy.Y, "labels": methods, "case_ids": xy.case_ids,
            "domains": xy.domains,
            "method_answers": [tuple(c[i] for c in cols) for i in range(len(inst))]}


def dap_inputs(inst, row=1, oq41=None, config=None):
    """Arguments of `run_dap_loo` / nested CV for an instance table:
    dict(X, y, case_ids, domains, answer_sets, rule_outcomes). The rule
    outcomes are those of the methods kept in the row."""
    from cdm.approaches import dap, instances as I
    xy = dap.dap_xy(inst, row, oq41, config)
    cfg = I.ablation_config(row)
    methods, rules = I.methods_kept(cfg), I.rules_kept(cfg)
    return {"X": xy.X, "y": xy.Y, "case_ids": xy.case_ids, "domains": xy.domains,
            "answer_sets": list(inst.answers),
            "rule_outcomes": {m: inst.outcomes(m, rules) for m in methods}}
