"""Case-level instance table shared by AMP and DAP, and the ablation masks.

Step A11 of plan §9.3 ("AMP/DAP instance builder + ablation masks";
acceptance: shapes 1209x27 (AMP) and 1209x32 (DAP), label means = T3, the
§5.4 matrix encoded, groups per TH p.47).

One case gives (plan §4.1, §4.2; TH §3.3.1.2 p.36-37, §3.3.1.3 p.38):
  * the 27 CDMCR features (cdm.features.cdmcr; TH Table 3.2 p.28);
  * the answers of the rule-based methods f_MR, f_WC, f_HAC, f_SP
    (cdm.aggregators) and f_DA built on the methods kept (TH p.35, p.48);
  * the correct answer a*.
`cdm.approaches.amp` and `cdm.approaches.dap` turn this table into X / Y.
Nothing here trains or evaluates a model.

Ablations (plan §5.4; TH §4.1.5 p.46-49; matrix from the rendered TH p.72,
identical in MLA Tables 3/4): `ABLATION_MATRIX` holds the 11 rows exactly
as in the plan. The ablation is component-wise: a row drops exactly the
columns marked 0 and nothing else.
  * Voting features, O_MR and f_MR are always kept (TH p.48).
  * Features 16 and 17 belong to both Conf and PS (TH p.47): plan §8 OQ41,
    default "dropped whenever either group is dropped".
  * When WC/HAC or SP is removed, DA is recomputed from the remaining
    methods (TH p.48; plan §5.4), e.g. with only MR, DA = "not MR".

Per-case seed of the sub-set partition (plan §3.3.1, OQ6 "Store the
seed"): the plan fixes the global seed (OQ15: 0) but does NOT say how the
seed of one case is derived from it (docs/IMPLEMENTATION_DECISIONS.md PT1: "How a per-case
seed is derived is not defined yet"). `case_seed` is AN IMPLEMENTATION CHOICE
(open question, decision M1): the first 8 bytes of
SHA-256("<global seed>|cdmcr_partition|<case_id>"). It depends only on the
global seed and the case ID, so the partition of a case does not change
when other cases are added, removed or reordered (scenarios, ablations,
cross-validation), and it is the same on every platform. The seeds and the
partitions used are stored in the table.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from cdm import aggregators as AG
from cdm import primitives as P
from cdm.features import cdmcr

#: Label / method order of the thesis: {O_MR, O_HAC, O_WC, O_SP, O_DA}
#: (TH p.37) and {f_MR, f_HAC, f_WC, f_SP, f_DA} (TH p.38). Also the
#: argmax tie order of plan §8 OQ17 and the classifier-chain order (OQ19).
LABEL_ORDER = ("MR", "HAC", "WC", "SP", "DA")

#: How `case_seed` derives a seed; stored with every table.
SEED_RULE = "int(sha256('<seed>|cdmcr_partition|<case_id>')[:8], big-endian)"


def case_seed(seed, case_id):
    """Seed of the sub-set partition of one case (see the module docstring).

    The derivation is an implementation choice: the plan is silent (plan §3.3.1,
    §8 OQ6/OQ15; docs/IMPLEMENTATION_DECISIONS.md PT1). `seed` is the global seed (an int).
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"the global seed must be an integer, got {seed!r}")
    digest = hashlib.sha256(f"{int(seed)}|cdmcr_partition|{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


# ---- ablation matrix (plan §5.4) ----------------------------------------------

#: Column names of the matrix, as in plan §5.4 (TH Tables 5.6/5.7 p.72).
ABLATION_COLUMNS = ("conf", "ps", "wc_hac", "sp", "da")

#: Plan §5.4 table, rows 1-11: (Conf feats, PS feats, WC+HAC, SP, DA),
#: 1 = included. Voting features and O_MR / f_MR are always included.
ABLATION_MATRIX = {
    1: (1, 1, 1, 1, 1),
    2: (0, 1, 1, 1, 1),
    3: (1, 0, 1, 1, 1),
    4: (0, 0, 1, 1, 1),
    5: (1, 1, 0, 1, 1),
    6: (1, 1, 1, 0, 1),
    7: (1, 1, 0, 0, 1),
    8: (1, 1, 1, 1, 0),
    9: (0, 1, 0, 1, 1),
    10: (1, 0, 1, 0, 1),
    11: (0, 0, 0, 0, 1),
}

#: Thesis results of the rows (TH Tables 5.6 / 5.7; plan §5.4), for the
#: report of step A16 only. Not used by any computation here.
ABLATION_THESIS = {
    1: (.806, .812), 2: (.800, .812), 3: (.816, .825), 4: (.795, .803),
    5: (.808, .804), 6: (.797, .805), 7: (.791, .806), 8: (.754, .813),
    9: (.806, .811), 10: (.799, .815), 11: (.763, .798),
}

OQ41_OPTIONS = ("dropped_whenever_either_group_dropped",
                "kept_unless_both_groups_dropped")


@dataclass(frozen=True)
class AblationConfig:
    """One row of the ablation matrix (plan §5.4). True = included."""
    row: Any
    conf: bool
    ps: bool
    wc_hac: bool
    sp: bool
    da: bool

    def as_tuple(self):
        return tuple(int(getattr(self, c)) for c in ABLATION_COLUMNS)


def ablation_config(row=1):
    """The AblationConfig of a row number 1..11 of plan §5.4 (row 1 = the
    full version). An AblationConfig is returned unchanged."""
    if isinstance(row, AblationConfig):
        return row
    if row not in ABLATION_MATRIX:
        raise ValueError(f"unknown ablation row {row!r}; plan §5.4 has rows 1..11")
    return AblationConfig(row, *(bool(x) for x in ABLATION_MATRIX[row]))


def all_ablation_configs():
    """The 11 configurations in the order of plan §5.4."""
    return [ablation_config(r) for r in sorted(ABLATION_MATRIX)]


def rules_kept(cfg):
    """The rule-based methods kept in a row, in the order of
    cdm.aggregators.RULES (MR, WC, HAC, SP). DA listens to exactly these
    (TH p.48: "if a method's label was excluded, the DA did not consider
    this method's choice"). MR is always kept (TH p.48)."""
    cfg = ablation_config(cfg)
    keep = {"MR": True, "WC": cfg.wc_hac, "HAC": cfg.wc_hac, "SP": cfg.sp}
    return tuple(m for m in AG.RULES if keep[m])


def methods_kept(cfg):
    """The labels (AMP) / f_AG features (DAP) kept in a row, in the thesis
    order MR, HAC, WC, SP, DA (TH p.37, p.38)."""
    cfg = ablation_config(cfg)
    rules = rules_kept(cfg)
    return tuple(m for m in LABEL_ORDER
                 if (m == "DA" and cfg.da) or m in rules)


def oq41_from_config(config=None):
    """Plan §8 OQ41 switch, validated."""
    from cdm.config import get_switch
    value = get_switch("OQ41", config)
    if value not in OQ41_OPTIONS:
        raise ValueError(f"OQ41 = {value!r} is not one of {OQ41_OPTIONS}")
    return value


def feature_numbers_kept(cfg, oq41="dropped_whenever_either_group_dropped"):
    """Feature numbers (1..27) kept in a row. Plan §5.4, §3.3, OQ41.

    Vote features are always kept. The Conf group is kept iff cfg.conf, the
    PS group iff cfg.ps. Features 16 and 17 are in both groups (TH p.47):

    oq41 = "dropped_whenever_either_group_dropped" (plan default): 16/17
        are kept only when both groups are kept.
    oq41 = "kept_unless_both_groups_dropped" (plan alternative, "rows 2-4
        only"): 16/17 are kept when at least one of the two groups is kept,
        EXCEPT when the row removes all confidence information (Conf
        features and WC/HAC both dropped: row 9) or all predicted-support
        information (PS features and SP both dropped: row 10), or both
        (row 11): there one of the two inputs of 16/17 is missing and they
        cannot be computed (plan §5.4). So the alternative differs from
        the default in rows 2 and 3 only.
    """
    cfg = ablation_config(cfg)
    if oq41 not in OQ41_OPTIONS:
        raise ValueError(f"OQ41 = {oq41!r} is not one of {OQ41_OPTIONS}")
    both = set(cdmcr.GROUPS["Conf"]) & set(cdmcr.GROUPS["PS"])
    keep = set(cdmcr.GROUPS["Vote"])
    if cfg.conf:
        keep |= set(cdmcr.GROUPS["Conf"]) - both
    if cfg.ps:
        keep |= set(cdmcr.GROUPS["PS"]) - both
    if oq41 == "dropped_whenever_either_group_dropped":
        keep_both = cfg.conf and cfg.ps
    else:
        no_conf_info = not cfg.conf and not cfg.wc_hac
        no_ps_info = not cfg.ps and not cfg.sp
        keep_both = (cfg.conf or cfg.ps) and not no_conf_info and not no_ps_info
    if keep_both:
        keep |= both
    return tuple(sorted(keep))


def feature_mask(cfg, oq41="dropped_whenever_either_group_dropped"):
    """Boolean mask of length 27 over cdmcr.FEATURE_NAMES for a row."""
    kept = set(feature_numbers_kept(cfg, oq41))
    return np.array([i + 1 in kept for i in range(len(cdmcr.FEATURE_NAMES))])


# ---- the instance table -------------------------------------------------------

@dataclass
class Instances:
    """Features, rule answers and correct answers of a list of cases.

    F        : array (n_cases, 27), the CDMCR features in the order of
               cdmcr.FEATURE_NAMES (float; NaN where a feature is
               undefined, see `nan_counts`).
    choices  : {rule: list of answers} for MR, WC, HAC, SP.
    supports : per case {answer: S(a, R)} (needed by DA only for more than
               two answers, plan §8 OQ8b).
    seeds    : per-case partition seeds (`case_seed`); partitions: the
               index lists used (plan OQ6 "Store the seed"; PT1).
    info     : per-case tie / imputation flags of cdmcr_features
               (plan §8 OQ7/OQ8 "log the frequency").
    """
    case_ids: List[Any]
    domains: List[Any]
    answers: List[Tuple[Any, ...]]
    correct: List[Any]
    n_responses: List[int]
    F: np.ndarray
    choices: Dict[str, List[Any]]
    supports: List[Dict[Any, float]]
    seeds: List[int]
    partitions: List[List[List[int]]]
    info: List[Dict[str, Any]]
    seed: int
    options: Dict[str, Any] = field(default_factory=dict)
    seed_rule: str = SEED_RULE
    feature_names: Tuple[str, ...] = cdmcr.FEATURE_NAMES

    def __len__(self):
        return len(self.case_ids)

    def da(self, rules=AG.RULES):
        """f_DA of every case over the given rule-based methods (TH p.35;
        ablation sub-sets TH p.48). Uses cdm.aggregators.da_from_choices on
        the stored rule answers, with the DA tie rule the table was built
        with (options["da_tie_rule"]; default the thesis text, 'not_mr')."""
        rules = tuple(rules)
        tie_rule = self.options.get("da_tie_rule", AG.DEFAULT_DA_TIE_RULE)
        return [AG.da_from_choices({m: self.choices[m][i] for m in rules},
                                   self.answers[i], rules, self.supports[i],
                                   tie_rule)
                for i in range(len(self))]

    def method_answers(self, method, rules=AG.RULES):
        """f_AG of every case; for DA, built on `rules`."""
        if method == "DA":
            return self.da(rules)
        if method not in self.choices:
            raise ValueError(f"unknown method {method!r}")
        return list(self.choices[method])

    def outcomes(self, method, rules=AG.RULES):
        """O_AG of every case: 1 iff f_AG(R) = a* (plan §4.1; TH p.37)."""
        return np.array([int(a == c) for a, c in
                         zip(self.method_answers(method, rules), self.correct)])

    def select(self, indices):
        """A new table with the given case positions (in the given order)."""
        idx = [int(i) for i in indices]
        pick = lambda xs: [xs[i] for i in idx]
        return Instances(
            case_ids=pick(self.case_ids), domains=pick(self.domains),
            answers=pick(self.answers), correct=pick(self.correct),
            n_responses=pick(self.n_responses), F=self.F[idx].copy(),
            choices={m: pick(v) for m, v in self.choices.items()},
            supports=pick(self.supports), seeds=pick(self.seeds),
            partitions=pick(self.partitions), info=pick(self.info),
            seed=self.seed, options=self.options, seed_rule=self.seed_rule,
            feature_names=self.feature_names)

    def nan_counts(self):
        """{feature name: number of cases where it is NaN}, non-zero only.
        (CAmin is NaN when a_min has no supporter in the full set: plan §8
        OQ7 "CAmin undefined"; docs/IMPLEMENTATION_DECISIONS.md F2 / AM12.)"""
        counts = np.isnan(self.F).sum(axis=0)
        return {n: int(c) for n, c in zip(self.feature_names, counts) if c}

    def log(self):
        """Frequencies that plan §8 OQ7/OQ8 ask to log, over the table."""
        keys = ("amax_tie", "amin_tie", "hac_tie", "single_subset")
        out = {k: sum(1 for i in self.info if i.get(k)) for k in keys}
        for k in ("imputed_CAmin", "imputed_CAmax", "subset_amax_ties",
                  "subset_amin_ties"):
            out["cases_with_" + k] = sum(1 for i in self.info if i.get(k, 0) > 0)
        out["nan_features"] = self.nan_counts()
        out["non_binary_cases"] = sum(1 for a in self.answers if len(a) != 2)
        return out


def build_instances(cases, seed=None, config=None, feature_options=None,
                    aggregator_options=None, da_tie_rule=None):
    """Compute the instance table of a list of Case. No machine learning.

    seed   : the global seed; default: the config's top-level `seed`
             (plan §8 OQ15: 0). The partition of each case is drawn with
             `case_seed(seed, case.case_id)`.
    config : config dict (default: config/defaults.yaml).
    feature_options    : keyword arguments of cdmcr.cdmcr_features;
                         default cdmcr.options_from_config(config).
    aggregator_options : {policy, tol} of cdm.aggregators.choose;
                         default aggregators.options_from_config(config).
    da_tie_rule        : cdm.aggregators.DA_TIE_RULES; default: the config's
                         `da_tie_rule` (the thesis text, 'not_mr').

    The four rules are computed with cdm.aggregators.choose, i.e. exactly
    as in the rule gate (cdm.eval.rule_gate) for an answer-level tie
    policy. With the scoring policy 'sp_tie_half_credit' a tied SP case has
    no single answer and therefore no 0/1 label: it raises (plan §8 OQ8;
    docs/IMPLEMENTATION_DECISIONS.md A6).

    Case IDs must be unique (they define the per-case seeds).
    """
    from cdm.config import load_config
    config = config if config is not None else load_config()
    if seed is None:
        seed = config["seed"]
    fopts = dict(feature_options if feature_options is not None
                 else cdmcr.options_from_config(config))
    aopts = dict(aggregator_options if aggregator_options is not None
                 else AG.options_from_config(config))
    if da_tie_rule is None:
        da_tie_rule = AG.da_tie_rule_from_config(config)
    if da_tie_rule not in AG.DA_TIE_RULES:
        raise ValueError(f"unknown DA tie rule {da_tie_rule!r}")
    for forbidden in ("seed", "subsets", "return_info"):
        if forbidden in fopts:
            raise ValueError(f"feature_options must not contain {forbidden!r}")

    cases = list(cases)
    ids = [c.case_id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("case IDs must be unique: they define the per-case seeds")

    F = np.full((len(cases), len(cdmcr.FEATURE_NAMES)), np.nan)
    choices = {m: [] for m in AG.RULES}
    supports, seeds, partitions, infos = [], [], [], []
    for i, case in enumerate(cases):
        R, answers = case.responses, case.answers
        s = case_seed(seed, case.case_id)
        f, info = cdmcr.cdmcr_features(R, answers, seed=s, return_info=True, **fopts)
        F[i] = [f[name] for name in cdmcr.FEATURE_NAMES]
        partitions.append(info.pop("subsets"))
        infos.append(info)
        seeds.append(s)
        for m in AG.RULES:
            choices[m].append(AG.choose(m, R, answers, **aopts))
        supports.append(dict(zip(answers, P.support_vector(R, answers))))

    return Instances(
        case_ids=ids, domains=[c.domain for c in cases],
        answers=[tuple(c.answers) for c in cases],
        correct=[c.correct for c in cases],
        n_responses=[len(c.responses) for c in cases], F=F, choices=choices,
        supports=supports, seeds=seeds, partitions=partitions, info=infos,
        seed=int(seed),
        options={"features": fopts, "aggregators": aopts,
                 "da_tie_rule": da_tie_rule})


@dataclass
class XY:
    """A learning problem: X (n, d), Y (n, q) or y (n,), and the names."""
    X: np.ndarray
    Y: np.ndarray
    feature_names: Tuple[str, ...]
    target_names: Tuple[str, ...]
    config: AblationConfig
    case_ids: List[Any]
    domains: List[Any]


def masked_features(inst, cfg, oq41):
    """(X, names): the CDMCR columns of `inst` kept in an ablation row."""
    mask = feature_mask(cfg, oq41)
    names = tuple(n for n, k in zip(inst.feature_names, mask) if k)
    return inst.F[:, mask].copy(), names
