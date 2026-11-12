"""Bagging experiments of AMP and DAP.

Step A18 of plan §9.3 ("Bagging" -> TH Tables 5.9, 5.10; plan items T17,
T18). Plan §5.6 (TH §4.1.7 p.50-52, §5.1.3 p.73-74), §8 OQ22, OQ23, OQ42.

Procedure (TH p.50-52):
  For each case P_j with |R_Pj| >= 70:
    1. draw F sub-groups of responses at random from R_Pj. Size of a
       sub-group = max(35, Z_0.95 * S(a*) * (1 - S(a*)) / 0.1^2), S(a*) =
       the share of R_Pj that voted for the correct answer. This is the
       formula AS PRINTED (rendered PDF p.50: no square on Z; plan OQ22
       option (a), the default). Switch: option (b), Cochran's formula
       with Z^2. Z = 1.645 (default) or 1.96. The size is rounded up
       (OQ22 "ceil"; 41.125 -> 42).
    2. every sub-group R'_Pj becomes a new instance; BI_j = the new
       instances of problem P_j. A new instance is a full case: its CDMCR
       features (including the SG features, on its own sub-set partition),
       its rule answers, DA and labels are recomputed on the sub-group;
       a* is unchanged (plan §5.6).
  Sampling (plan §8 OQ42): without replacement inside a sub-group;
  sub-groups are drawn independently of each other (they may overlap).

  For K = 1..10 (the SAME folds as the model selection, plan OQ23):
    Test_K     = fold K of the original data, original instances only;
    BagTrain_K = the other 90 % of the originals + every new instance
                 EXCEPT those of the problems in Test_K;
    train BR+RF (AMP) / RF (DAP) on BagTrain_K, predict Test_K, success
    rate of the fold. F in {2, 3, 5, 10}.

Random draws: the plan fixes the global seed (OQ15: 0) but not how the
seed of one draw is derived. `draw_seed` is AN IMPLEMENTATION CHOICE, in the
style of cdm.approaches.instances.case_seed: the first 8 bytes of
SHA-256("<seed>|bagging|<case_id>|F=<F>|<j>"). The draws of different F
are therefore independent ("for each value of F" a new data set is
created, TH p.51), and a case's draws do not depend on the other cases.

Undefined feature (OPEN, plan §8 OQ7 / amendment AM12; docs/IMPLEMENTATION_DECISIONS.md F2,
M6): in a sub-group in which every response votes for the same answer,
a_min has no supporter and CAmin is undefined (NaN). The plan gives no
rule for a bagged instance. `apply_undefined_policy` therefore STOPS by
default ("stop": raises `UndefinedFeatureError` with the counts). Two
rules exist as labelled options that are NOT in the plan and
must be chosen by the user before they are used:
  "drop_instance"        : such new instances are left out of BagTrain;
  "impute_subgroup_avgc" : CAmin := AvgC of the sub-group (the analogue
                           of the OQ7 rule for sub-sets, docs/IMPLEMENTATION_DECISIONS.md F5).

Nothing here reads a data build; the caller passes the cases.
"""
import hashlib
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List

import numpy as np

from cdm.approaches import instances as I
from cdm.eval import loo as L
from cdm.eval import nested_cv as N
from cdm.eval import stats as S
from cdm.schema import Case

#: TH p.50: "For each set of responses R_Pj such that |R_Pj| >= 70".
MIN_RESPONSES = 70
#: TH p.50: "the maximal value in (35, ...)".
MIN_SIZE = 35
#: TH p.50: the denominator 0.1^2.
MARGIN = Fraction(1, 10)
#: TH p.52: "four different values of F (re-sampling factor): 2, 3, 5, and 10".
FACTORS = (2, 3, 5, 10)
FORMULA_PRINTED = "a_as_printed_Z_S_1_minus_S_over_0.01"
FORMULA_COCHRAN = "b_cochran_Z2_S_1_minus_S_over_0.01"
SIZE_FORMULAS = (FORMULA_PRINTED, FORMULA_COCHRAN)
Z_VALUES = (1.645, 1.96)
TECHNIQUE = {"AMP": ("BR", "RF"), "DAP": "RF"}
PROTOCOL = "bagging_cv"
UNDEFINED_POLICIES = ("stop", "drop_instance", "impute_subgroup_avgc")
SEED_RULE = "int(sha256('<seed>|bagging|<case_id>|F=<F>|<j>')[:8], big-endian)"


class UndefinedFeatureError(ValueError):
    """New instances have an undefined feature and no rule is chosen
    (plan OQ7 / AM12). `rows` = their positions, `counts` = {feature: n}."""

    def __init__(self, message, rows=(), counts=None):
        super().__init__(message)
        self.rows = list(rows)
        self.counts = dict(counts or {})


# ---- settings (plan §8 OQ22, OQ42) ------------------------------------------------

def settings_from_config(config=None, size_formula=None, z=None):
    """{"size_formula", "z"} from plan §8 OQ22, with optional overrides;
    validated. The parts of OQ22 / OQ42 for which only the plan default is
    implemented (ceil rounding, sampling without replacement, independent
    sub-groups, recomputed labels and SG features) must have their default
    values, otherwise NotImplementedError."""
    from cdm.config import get_switch
    oq22, oq42 = get_switch("OQ22", config), get_switch("OQ42", config)
    fixed = {"size_rounding": "ceil", "within_subgroup": "without_replacement",
             "subgroups_independent": True,
             "recompute_labels_and_f_ag_per_subgroup": True,
             "recompute_sg_features_per_subgroup": True}
    for key, want in fixed.items():
        if oq22.get(key) != want:
            raise NotImplementedError(f"OQ22 {key} = {oq22.get(key)!r}: only {want!r} "
                                      "is implemented (plan §5.6)")
    if oq42.get("within_case") != "without_replacement" or oq42.get(
            "subgroups_independent") is not True:
        raise NotImplementedError(f"OQ42 = {oq42!r}: only the plan default is implemented")
    out = {"size_formula": oq22["size_formula"] if size_formula is None else size_formula,
           "z": float(oq22["z"] if z is None else z)}
    _check(out["size_formula"], out["z"])
    return out


def _check(size_formula, z):
    if size_formula not in SIZE_FORMULAS:
        raise ValueError(f"unknown size formula {size_formula!r}; expected one of "
                         f"{SIZE_FORMULAS}")
    if not any(math.isclose(z, v) for v in Z_VALUES):
        raise ValueError(f"Z = {z!r} is not one of the plan's values {Z_VALUES} (OQ22)")


# ---- size of a sub-group (TH p.50; plan §5.6, OQ22) ---------------------------------

def subgroup_size(n_correct, n, size_formula=FORMULA_PRINTED, z=1.645):
    """Size of a sampled sub-group of a case with `n` responses of which
    `n_correct` voted for a*.

    printed (default) : max(35, ceil(Z * S (1 - S) / 0.1^2))
    cochran           : max(35, ceil(Z^2 * S (1 - S) / 0.1^2))
    with S = n_correct / n. Exact rational arithmetic, so the rounding up
    is not affected by float noise. Range: printed, Z = 1.645: 35..42;
    Z = 1.96: 35..49; cochran, Z = 1.645: 35..68; Z = 1.96: 35..97 (which
    exceeds a case of 70..96 responses: `draw_subgroup` then raises).
    """
    _check(size_formula, z)
    if n <= 0 or not 0 <= n_correct <= n:
        raise ValueError("need 0 <= n_correct <= n and n > 0")
    s = Fraction(int(n_correct), int(n))
    zf = Fraction(str(z))
    if size_formula == FORMULA_COCHRAN:
        zf = zf * zf
    raw = zf * s * (1 - s) / (MARGIN * MARGIN)
    return max(MIN_SIZE, math.ceil(raw))


def draw_seed(seed, case_id, factor, j):
    """Seed of the j-th sub-group (j = 0..F-1) of a case for one F (see
    the module docstring; the derivation is an implementation choice)."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"the global seed must be an integer, got {seed!r}")
    digest = hashlib.sha256(
        f"{int(seed)}|bagging|{case_id}|F={int(factor)}|{int(j)}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def draw_subgroup(n, size, seed):
    """Positions of a sub-group of `size` responses out of `n`, drawn
    without replacement (plan OQ42), in increasing order."""
    if size > n:
        raise ValueError(f"a sub-group of {size} responses cannot be drawn without "
                         f"replacement from {n} responses (formula / Z combination "
                         "impossible for this case, plan §5.6)")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=size, replace=False))


def bagged_case_id(case_id, factor, j):
    """ID of the j-th new instance (j = 0..F-1) of a case."""
    return f"{case_id}#bagF{int(factor)}#{int(j) + 1}"


# ---- the new instances -----------------------------------------------------------------

@dataclass
class BaggedSet:
    """The new cases of one value of F.

    cases      : the new cases (same answers, a*, domain as the parent;
                 responses = the sampled sub-group).
    parent     : (m,) position of the parent case in the original list.
    parent_ids : the parent's case_id, per new case.
    draws      : per new case, the sampled response positions.
    sizes      : (m,) sub-group sizes; seeds: (m,) draw seeds.
    eligible   : positions of the original cases with |R| >= 70.
    """
    factor: int
    cases: List[Case]
    parent: np.ndarray
    parent_ids: List[Any]
    draws: List[np.ndarray]
    sizes: np.ndarray
    seeds: List[int]
    eligible: List[int]
    settings: Dict[str, Any] = field(default_factory=dict)

    def __len__(self):
        return len(self.cases)

    def describe(self):
        s = self.sizes
        return {"F": self.factor, "eligible_cases": len(self.eligible),
                "new_instances": len(self.cases),
                "size_min": int(s.min()) if len(s) else None,
                "size_max": int(s.max()) if len(s) else None,
                "size_mean": float(s.mean()) if len(s) else None,
                "settings": dict(self.settings), "seed_rule": SEED_RULE}


def bag_cases(cases, factor, seed=0, size_formula=FORMULA_PRINTED, z=1.645,
              min_responses=MIN_RESPONSES):
    """The new cases of one F: for every case with at least
    `min_responses` responses, `factor` sub-groups (TH p.50 steps 1-2).

    cases : the original cases, in data order (the positions are the
            parents). seed : the global seed of the draws.
    """
    if isinstance(factor, bool) or int(factor) != factor or factor < 1:
        raise ValueError("F must be a positive integer")
    _check(size_formula, z)
    out, parent, parent_ids, draws, sizes, seeds, eligible = [], [], [], [], [], [], []
    for pos, case in enumerate(cases):
        n = len(case.responses)
        if n < min_responses:
            continue
        eligible.append(pos)
        n_correct = sum(1 for r in case.responses if r.v == case.correct)
        size = subgroup_size(n_correct, n, size_formula, z)
        for j in range(int(factor)):
            s = draw_seed(seed, case.case_id, factor, j)
            idx = draw_subgroup(n, size, s)
            out.append(Case(case_id=bagged_case_id(case.case_id, factor, j),
                            answers=case.answers, correct=case.correct,
                            responses=tuple(case.responses[i] for i in idx),
                            domain=case.domain))
            parent.append(pos)
            parent_ids.append(case.case_id)
            draws.append(idx)
            sizes.append(size)
            seeds.append(s)
    return BaggedSet(int(factor), out, np.array(parent, dtype=int), parent_ids, draws,
                     np.array(sizes, dtype=int), seeds, eligible,
                     {"size_formula": size_formula, "z": float(z), "seed": int(seed),
                      "min_responses": int(min_responses), "min_size": MIN_SIZE})


def bagged_instances(bag, config=None, seed=None):
    """Instance table of the new cases (cdm.approaches.instances.
    build_instances): every feature, rule answer and label is recomputed
    on the sub-group (plan §5.6). An empty set gives None."""
    if not len(bag):
        return None
    return I.build_instances(bag.cases, seed=seed, config=config)


def undefined_rows(inst_bag):
    """(positions of the new instances with a non-finite feature,
    {feature name: number of such instances})."""
    bad = ~np.isfinite(inst_bag.F)
    rows = np.flatnonzero(bad.any(axis=1))
    counts = {n: int(c) for n, c in zip(inst_bag.feature_names, bad.sum(axis=0)) if c}
    return rows, counts


def apply_undefined_policy(inst_bag, parent, policy="stop"):
    """Apply the rule for new instances with an undefined feature (see the
    module docstring; OPEN, AM12). Returns (inst_bag, parent, report).

    "stop" (default)       : raise UndefinedFeatureError if there is any.
    "drop_instance"        : leave those new instances out. NOT IN THE PLAN.
    "impute_subgroup_avgc" : CAmin := AvgC of the sub-group. NOT IN THE PLAN.
    A non-finite value in any feature other than CAmin always raises.
    """
    if policy not in UNDEFINED_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {UNDEFINED_POLICIES}")
    parent = np.asarray(parent, dtype=int)
    rows, counts = undefined_rows(inst_bag)
    report = {"policy": policy, "undefined_instances": int(len(rows)),
              "undefined_by_feature": counts, "new_instances": len(inst_bag)}
    if not len(rows):
        return inst_bag, parent, report
    other = {k: v for k, v in counts.items() if k != "CAmin"}
    if other:
        raise UndefinedFeatureError(
            f"new instances have non-finite features other than CAmin: {other}",
            rows, counts)
    if policy == "stop":
        raise UndefinedFeatureError(
            f"{len(rows)} of {len(inst_bag)} new instances have an undefined CAmin "
            "(a sub-group in which every response votes for the same answer). The "
            "plan gives no rule for it (OQ7 / amendment AM12). Choose one of "
            f"{UNDEFINED_POLICIES[1:]} explicitly once it is decided", rows, counts)
    if policy == "drop_instance":
        keep = np.setdiff1d(np.arange(len(inst_bag)), rows)
        return inst_bag.select(keep), parent[keep], report
    names = list(inst_bag.feature_names)
    inst_bag.F[rows, names.index("CAmin")] = inst_bag.F[rows, names.index("AvgC")]
    return inst_bag, parent, report


# ---- BagTrain_K / Test_K (TH p.51) ---------------------------------------------------------

def bag_train_mask(parent, test_idx):
    """Boolean mask over the new instances: True = belongs to BagTrain_K,
    i.e. its problem is NOT in Test_K (TH p.51)."""
    return ~np.isin(np.asarray(parent, dtype=int), np.asarray(test_idx, dtype=int))


def check_no_leakage(train_idx, test_idx, parent, mask, n, case_ids=None,
                     bag_parent_ids=None):
    """Assert the definition of the fold (TH p.51):
      * Test_K and the original training cases are disjoint and together
        are all n originals;
      * no new instance in BagTrain_K comes from a problem in Test_K;
      * every new instance of a training problem IS in BagTrain_K;
      * with the IDs given: the same, checked on the case IDs (independent
        of the positions).
    Raises AssertionError otherwise."""
    train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
    parent, mask = np.asarray(parent, dtype=int), np.asarray(mask, dtype=bool)
    assert not set(train_idx.tolist()) & set(test_idx.tolist()), "train and test overlap"
    assert len(train_idx) + len(test_idx) == n and \
        len(set(train_idx.tolist()) | set(test_idx.tolist())) == n, "folds do not cover the data"
    test = set(test_idx.tolist())
    assert not test & set(parent[mask].tolist()), \
        "a new instance of a test problem is in BagTrain"
    assert all(p in test for p in parent[~mask].tolist()), \
        "a new instance of a training problem was left out of BagTrain"
    if case_ids is not None and bag_parent_ids is not None:
        test_ids = {case_ids[i] for i in test_idx.tolist()}
        used = {pid for pid, m in zip(bag_parent_ids, mask) if m}
        assert not test_ids & used, "a new instance of a test problem is in BagTrain (IDs)"


@dataclass
class BaggingResult:
    """One approach, one F.

    result : cdm.eval.loo.EvalResult over all original cases (each tested
             once, protocol "bagging_cv", `folds` = the fold of each case).
    folds  : per fold {fold, n_test, successes, rate, n_train_original,
             n_train_bagged, n_bagged_excluded, not_converged}.
    """
    approach: str
    factor: int
    result: L.EvalResult
    folds: List[Dict[str, Any]]
    meta: Dict[str, Any] = field(default_factory=dict)

    def fold_rates(self, rounded=True):
        return [S.round_half_up(f["successes"], f["n_test"]) if rounded else f["rate"]
                for f in self.folds]


def _fold(spec, X, Y, Xb, Yb, tr, te, mask):
    Xtr = np.vstack([X[tr], Xb[mask]]) if Xb is not None else X[tr]
    Ytr = np.concatenate([Y[tr], Yb[mask]]) if Xb is not None else Y[tr]
    return L.fit_predict_counted(spec, Xtr, Ytr, X[te])


def run_bagging(approach, inst, inst_bag, parent, factor=None, hyper=None, n_splits=None,
                config=None, n_jobs=1):
    """The cross-validation of TH p.51-52 for one approach and one set of
    new instances. Returns a BaggingResult.

    inst     : instance table of the ORIGINAL cases, in data order.
    inst_bag : instance table of the new instances (after
               `apply_undefined_policy`), or None for no new instance.
    parent   : (m,) position in `inst` of the parent of every new instance.
    n_splits : default plan §8 OQ16 / OQ23: the 10 unshuffled folds of the
               model selection (cdm.eval.nested_cv.outer_folds).
    The technique is BR+RF (AMP) / RF (DAP) in every fold (TH p.73).
    """
    if approach not in TECHNIQUE:
        raise ValueError(f"unknown approach {approach!r}")
    n = len(inst)
    n_splits = N.fold_scheme_from_config(config) if n_splits is None else int(n_splits)
    parent = np.asarray(parent, dtype=int)
    m = 0 if inst_bag is None else len(inst_bag)
    if len(parent) != m:
        raise ValueError("`parent` must have one entry per new instance")
    if m and (parent.min() < 0 or parent.max() >= n):
        raise ValueError("`parent` holds positions outside the original data")
    if approach == "AMP":
        inp = L.amp_inputs(inst, config=config)
        X, Y = inp["X"], inp["Y"]
        spec = L.make_spec("AMP", TECHNIQUE["AMP"], inp["labels"], hyper, config=config)
        b = L.amp_inputs(inst_bag, config=config) if m else None
        Xb, Yb = (b["X"], b["Y"]) if m else (None, None)
    else:
        inp = L.dap_inputs(inst, config=config)
        X, Y = inp["X"], inp["y"]
        spec = L.make_spec("DAP", TECHNIQUE["DAP"], hyper=hyper, config=config)
        b = L.dap_inputs(inst_bag, config=config) if m else None
        Xb, Yb = (b["X"], b["y"]) if m else (None, None)
    for name, arr in (("original", X), ("new", Xb)):
        if arr is not None and not np.isfinite(arr).all():
            raise UndefinedFeatureError(f"the {name} instances hold NaN or inf")
    bag_ids = None if not m else [inst.case_ids[p] for p in parent]
    splits = N.outer_folds(n, n_splits)
    masks = []
    for tr, te in splits:
        mask = bag_train_mask(parent, te)
        check_no_leakage(tr, te, parent, mask, n, inst.case_ids, bag_ids)
        masks.append(mask)
    if n_jobs == 1:
        outs = [_fold(spec, X, Y, Xb, Yb, tr, te, mk) for (tr, te), mk in zip(splits, masks)]
    else:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_jobs)(delayed(_fold)(spec, X, Y, Xb, Yb, tr, te, mk)
                                       for (tr, te), mk in zip(splits, masks))
    out = np.full((n, spec.n_out), np.nan)
    fold_of = np.zeros(n, dtype=int)
    for k, ((tr, te), (o, _)) in enumerate(zip(splits, outs), start=1):
        out[te] = o
        fold_of[te] = k
    if np.isnan(out).any() or (fold_of == 0).any():
        raise RuntimeError("bagging outputs are incomplete")
    meta = L._meta(spec, F=factor, n_splits=n_splits, new_instances=m,
                   lr_not_converged=int(sum(w for _, w in outs)))
    if approach == "AMP":
        res = L.amp_result(spec, out, Y, PROTOCOL, case_ids=inp["case_ids"],
                           domains=inp["domains"], method_answers=inp["method_answers"],
                           folds=fold_of, meta=meta)
    else:
        res = L.dap_result(spec, out, Y, PROTOCOL, case_ids=inp["case_ids"],
                           domains=inp["domains"], answer_sets=inp["answer_sets"],
                           rule_outcomes=inp["rule_outcomes"], folds=fold_of, meta=meta)
    folds = []
    for k, ((tr, te), mk, (_, n_warn)) in enumerate(zip(splits, masks, outs), start=1):
        succ = int(res.success[te].sum())
        folds.append({"fold": k, "n_test": int(len(te)), "successes": succ,
                      "rate": succ / len(te), "n_train_original": int(len(tr)),
                      "n_train_bagged": int(mk.sum()),
                      "n_bagged_excluded": int((~mk).sum()), "not_converged": int(n_warn)})
    return BaggingResult(approach, factor, res, folds, meta)
