"""CDMCR: the 27 features of a collective decision-making case.

Plan §3.3 (TH Table 3.2 p.28, Table A.1 p.109; MLA Table 2). Steps A6
(features 1-17) and A7 (sub-group features 18-27) of plan §9.3.

Notation: R = R_P, the responses of the case; {R'} = the sub-sets of R
given by the partition of plan §3.3.1 (cdm.partition).

Conventions (each is a switch; defaults = plan §8):
  OQ7   S(R) is over all of A_P; a_max/a_min ties -> canonical first.
  OQ9   "Var" = population standard deviation.
  OQ10  Entropy base 2.
  OQ11  D_S_Uniform = Euclidean distance.
  OQ12  SG features use the a_max/a_min of each sub-set.
  OQ13  strict inequalities in features 16/17.
  OQ14  B_Amax_MaxCa uses f_HAC with the OQ8 tie rule.

Tie policies are separate arguments: `amax_policy` (OQ7, a_max/a_min) and
`hac_policy` (OQ8 through OQ14, f_HAC in feature 12). The older single
`policy` argument still sets both when they are not given.

Float comparisons (amendments AM3/AM9): `tol`
(default primitives.DEFAULT_TIE_ABS_TOL = config `tie_abs_tol`) is used for
HAC ties in feature 12 and for the comparisons with AvgC / AvgPSV in
features 16/17. tol=0 gives exact float comparison, under which features
16/17 depend on the order of the responses. Vote-count and label
comparisons (a_max, a_min, SG_B_Amax) never use it.
"""
import math

from cdm import aggregators as AG
from cdm import primitives as P
from cdm.partition import partition

NAN = float("nan")

#: Feature names in the order of TH Table 3.2 (plan §3.3 table).
FEATURES_1_17 = (
    "N", "D_Smax_Smin", "ES", "VarS", "D_S_Uniform",
    "CAmin", "CAmax", "MinC", "AvgC", "VarC", "D_MaxC_AvgC", "B_Amax_MaxCa",
    "MaxPSa", "MinPSa", "AvgPSV", "P_lowC_highPSv", "P_lowPSv_highC",
)
FEATURES_18_27 = (
    "SG_B_Amax", "SG_VarSAmax", "SG_D_MaxVarS_MinVarS", "SG_D_MaxES_MinES",
    "SG_D_MaxAvgC_MinAvgC", "SG_D_MaxVarC_MinVarC",
    "SG_VarCAmin", "SG_VarCAmax", "SG_AvgCAmin", "SG_AvgCAmax",
)
FEATURE_NAMES = FEATURES_1_17 + FEATURES_18_27

#: Feature groups of TH §4.1.5 p.47 (plan §3.3), by feature number 1..27.
#: 16 and 17 belong to both Conf and PS.
GROUPS = {
    "Vote": (1, 2, 3, 4, 5, 18, 19, 20, 21),
    "Conf": (6, 7, 8, 9, 10, 11, 12, 16, 17, 22, 23, 24, 25, 26, 27),
    "PS": (13, 14, 15, 16, 17),
}


def _avg_conf_of(a, R):
    """Average(C(R(a))); NaN when a has no supporter in R."""
    return P.average(P.confidences(P.supporters(a, R)))


# ---- features 1-17 (step A6) ----------------------------------------------

def _policies(policy, amax_policy, hac_policy):
    return (policy if amax_policy is None else amax_policy,
            policy if hac_policy is None else hac_policy)


def features_1_17(R, answers, var_kind="population_sd", strict=True,
                  policy="canonical_first", amax_policy=None, hac_policy=None,
                  tol=P.DEFAULT_TIE_ABS_TOL, return_info=False):
    """CDMCR features 1-17 of one set of responses.

    TH Table 3.2 p.28 rows 1-17 (descriptions in TH Table A.1 p.109);
    plan §3.3 table.

    var_kind    : plan §8 OQ9 ("Var" = population SD; switch
                  "population_variance").
    strict      : plan §8 OQ13 (True: features 16/17 use < and >, as
                  printed; False: <= and >=). Must be a bool.
    amax_policy : tie policy for a_max and a_min (OQ7).
    hac_policy  : tie policy for f_HAC in feature 12 (OQ8, by OQ14).
    policy      : sets both policies above when they are not given.
    tol         : absolute tolerance for float comparisons (AM3/AM9): HAC ties in feature 12; in features 16/17 a
                  value within tol of the average counts as EQUAL to it
                  (strict: c < AvgC - tol and ps[v] > AvgPSV + tol;
                  non-strict: c <= AvgC + tol and ps[v] >= AvgPSV - tol).

    CAmin is NaN if a_min has no supporter in R (plan §8 OQ7: "CAmin
    undefined"; it cannot happen at the full-set level in World A).

    Returns a dict {feature name: value} in the order of FEATURES_1_17;
    with return_info=True, (dict, info) where info holds the full-set tie
    flags that OQ7/OQ8 ask to log: amax_tied / amin_tied / hac_tied (the
    tied answers, canonical order) and amax_tie / amin_tie / hac_tie.
    """
    if not isinstance(strict, bool):
        raise ValueError(f"strict must be True or False, got {strict!r}")
    tol = P.check_tol(tol)
    amax_policy, hac_policy = _policies(policy, amax_policy, hac_policy)
    answers = list(answers)
    n = len(R)
    if n == 0:
        raise ValueError("the features need at least one response")
    s = P.support_vector(R, answers)
    amax = P.a_max(R, answers, amax_policy)
    amin = P.a_min(R, answers, amax_policy)
    c = P.confidences(R)
    avg_c = P.average(c)
    psv = P.predicted_support_of_vote(R, answers)
    avg_psv = P.average(psv)
    avg_ps = [P.average(P.predicted_support(a, R, answers)) for a in answers]
    uniform = 1.0 / len(answers)

    # Features 16/17: a value within tol of the average is "equal" to it.
    if strict:
        low_c_high_ps = sum(1 for ci, xi in zip(c, psv)
                            if ci < avg_c - tol and xi > avg_psv + tol)
        low_ps_high_c = sum(1 for ci, xi in zip(c, psv)
                            if ci > avg_c + tol and xi < avg_psv - tol)
    else:
        low_c_high_ps = sum(1 for ci, xi in zip(c, psv)
                            if ci <= avg_c + tol and xi >= avg_psv - tol)
        low_ps_high_c = sum(1 for ci, xi in zip(c, psv)
                            if ci >= avg_c - tol and xi <= avg_psv + tol)

    amax_tied = P.a_max_ties(R, answers)
    amin_tied = P.a_min_ties(R, answers)
    hac_tied = AG.tied_answers("HAC", R, answers, tol)
    info = {"amax_tied": amax_tied, "amin_tied": amin_tied, "hac_tied": hac_tied,
            "amax_tie": len(amax_tied) > 1, "amin_tie": len(amin_tied) > 1,
            "hac_tie": len(hac_tied) > 1}

    f = {
        # 1  |R_P|
        "N": n,
        # 2  S(a_max, R_P) - S(a_min, R_P)
        "D_Smax_Smin": P.support(amax, R) - P.support(amin, R),
        # 3  Entropy(S(R_P)), base 2 (OQ10)
        "ES": P.entropy(s, base=2),
        # 4  Variance(S(R_P)) -> population SD over the |A| shares (OQ9)
        "VarS": P.var(s, var_kind),
        # 5  Euclidean distance of S(R_P) from the uniform vector (OQ11)
        "D_S_Uniform": math.sqrt(sum((x - uniform) ** 2 for x in s)),
        # 6  Average(C(R_P(a_min)))
        "CAmin": _avg_conf_of(amin, R),
        # 7  Average(C(R_P(a_max)))
        "CAmax": _avg_conf_of(amax, R),
        # 8  Min(C(R_P))
        "MinC": P.minimum(c),
        # 9  Average(C(R_P))
        "AvgC": avg_c,
        # 10 Variance(C(R_P)) -> population SD (OQ9)
        "VarC": P.var(c, var_kind),
        # 11 Max(C(R_P)) - Average(C(R_P))
        "D_MaxC_AvgC": P.maximum(c) - avg_c,
        # 12 1 if f_HAC(R_P) == a_max, else 0 (TH Table 3.2; OQ14)
        "B_Amax_MaxCa": int(AG.hac(R, answers, hac_policy, tol) == amax),
        # 13 Max over a_i of Average(PS(a_i, R_P))
        "MaxPSa": max(avg_ps),
        # 14 Min over a_i of Average(PS(a_i, R_P))
        "MinPSa": min(avg_ps),
        # 15 Average(PSV(R_P))
        "AvgPSV": avg_psv,
        # 16 share of r with r.c < AvgC and r.ps[r.v] > AvgPSV
        "P_lowC_highPSv": low_c_high_ps / n,
        # 17 share of r with r.c > AvgC and r.ps[r.v] < AvgPSV
        "P_lowPSv_highC": low_ps_high_c / n,
    }
    assert tuple(f) == FEATURES_1_17
    return (f, info) if return_info else f


# ---- features 18-27, sub-group features (step A7) -------------------------

def features_18_27(R, answers, subsets, var_kind="population_sd",
                   amax_mode="per_subset",
                   empty_supporters="nan_then_impute_subset_avgc",
                   policy="canonical_first", return_info=False,
                   amax_policy=None, allow_single_subset=False):
    """CDMCR sub-group (SG) features 18-27.

    TH Table 3.2 p.28 rows 18-27 (TH Table A.1 p.109); plan §3.3 table and
    §3.3.1. Each value is first extracted from every sub-set R' and then
    combined across the sub-sets (TH p.27, steps 1-3).

    subsets   : list of index lists into R (from cdm.partition.partition).
    var_kind  : plan §8 OQ9; used for every "Variance", both inside a
                sub-set and across the sub-sets.
    amax_mode : plan §8 OQ12. "per_subset" (default): a_max and a_min are
                taken inside each sub-set. "global": the a_max/a_min of the
                full set R are used in features 19 and 24-27. Feature 18
                compares the sub-sets' own a_max in both modes (it has no
                meaning otherwise).
    empty_supporters : plan §8 OQ7. When a_min (or, in global mode, a_max)
                has no supporter inside a sub-set, Average(C(R'(a))) is
                undefined. "nan_then_impute_subset_avgc" (default) uses
                Average(C(R')) of that sub-set; "zero" uses 0.
    policy, amax_policy : tie policy for a_max/a_min (OQ7); `amax_policy`
                wins when given. All comparisons here are on vote counts
                and labels, so no float tolerance is involved.

    allow_single_subset : False (default) raises when only one sub-set is
                given. True (used by `cdmcr_features` for OQ6 with
                halves_if_k_below_2 = false and 30 <= N < 60) returns the
                degenerate values: SG_B_Amax = 0, every spread and
                "Variance" = 0, the averages = the values of that sub-set;
                info["single_subset"] is True.

    Feature 22 uses confidence, Average(C(R')), as in TH Table A.1 and MLA
    (TH Table 3.2 prints "Average(S(R'))"; TH Table A.1 and MLA give C; this
    implementation uses C: plan §3.3 row 22, D7, OQ36).

    Returns a dict in the order of FEATURES_18_27; with return_info=True,
    (dict, info) where info counts the imputations and the sub-sets with
    an a_max tie / an a_min tie (OQ7 "flag", "log the frequency").
    """
    answers = list(answers)
    if amax_mode not in ("per_subset", "global"):
        raise ValueError(f"unknown amax_mode {amax_mode!r}")
    if empty_supporters not in ("nan_then_impute_subset_avgc", "zero"):
        raise ValueError(f"unknown empty_supporters {empty_supporters!r}")
    if amax_policy is not None:
        policy = amax_policy
    if len(subsets) < (1 if allow_single_subset else 2):
        raise ValueError("the SG features need at least two sub-sets "
                         "(one with allow_single_subset=True)")
    subs = []
    for idx in subsets:
        if len(idx) == 0:
            raise ValueError("empty sub-set")
        subs.append([R[i] for i in idx])

    g_amax = P.a_max(R, answers, policy)
    g_amin = P.a_min(R, answers, policy)
    info = {"imputed_CAmin": 0, "imputed_CAmax": 0, "subset_amax_ties": 0,
            "subset_amin_ties": 0, "n_subsets": len(subs),
            "single_subset": len(subs) == 1}

    def conf_of(a, Rs, key):
        v = _avg_conf_of(a, Rs)
        if math.isnan(v):
            info[key] += 1
            v = P.average(P.confidences(Rs)) if empty_supporters != "zero" else 0.0
        return v

    own_amax, s_amax, var_s, es, avg_c, var_c, ca_min, ca_max = ([] for _ in range(8))
    for Rs in subs:
        if len(P.a_max_ties(Rs, answers)) > 1:
            info["subset_amax_ties"] += 1
        if len(P.a_min_ties(Rs, answers)) > 1:
            info["subset_amin_ties"] += 1
        amax_s = P.a_max(Rs, answers, policy)
        own_amax.append(amax_s)
        if amax_mode == "per_subset":
            amax, amin = amax_s, P.a_min(Rs, answers, policy)
        else:
            amax, amin = g_amax, g_amin
        s = P.support_vector(Rs, answers)
        c = P.confidences(Rs)
        s_amax.append(P.support(amax, Rs))
        var_s.append(P.var(s, var_kind))
        es.append(P.entropy(s, base=2))
        avg_c.append(P.average(c))
        var_c.append(P.var(c, var_kind))
        ca_min.append(conf_of(amin, Rs, "imputed_CAmin"))
        ca_max.append(conf_of(amax, Rs, "imputed_CAmax"))

    f = {
        # 18 1 if a_max differs across the sub-sets, else 0
        "SG_B_Amax": int(len(set(own_amax)) > 1),
        # 19 Variance({S(a_max, R')})
        "SG_VarSAmax": P.var(s_amax, var_kind),
        # 20 Max({Variance(S(R'))}) - Min({Variance(S(R'))})
        "SG_D_MaxVarS_MinVarS": max(var_s) - min(var_s),
        # 21 Max({Entropy(S(R'))}) - Min({Entropy(S(R'))})
        "SG_D_MaxES_MinES": max(es) - min(es),
        # 22 Max({Average(C(R'))}) - Min({Average(C(R'))})
        "SG_D_MaxAvgC_MinAvgC": max(avg_c) - min(avg_c),
        # 23 Max({Variance(C(R'))}) - Min({Variance(C(R'))})
        "SG_D_MaxVarC_MinVarC": max(var_c) - min(var_c),
        # 24 Variance({Average(C(R'(a_min)))})
        "SG_VarCAmin": P.var(ca_min, var_kind),
        # 25 Variance({Average(C(R'(a_max)))})
        "SG_VarCAmax": P.var(ca_max, var_kind),
        # 26 Average({Average(C(R'(a_min)))})
        "SG_AvgCAmin": P.average(ca_min),
        # 27 Average({Average(C(R'(a_max)))})
        "SG_AvgCAmax": P.average(ca_max),
    }
    assert tuple(f) == FEATURES_18_27
    return (f, info) if return_info else f


# ---- all 27 ------------------------------------------------------------------

def cdmcr_features(R, answers, subsets=None, seed=None, partition_kwargs=None,
                   var_kind="population_sd", strict=True,
                   amax_mode="per_subset",
                   empty_supporters="nan_then_impute_subset_avgc",
                   policy="canonical_first", amax_policy=None, hac_policy=None,
                   tol=P.DEFAULT_TIE_ABS_TOL, return_info=False):
    """The 27 CDMCR features of one case (TH Table 3.2; plan §3.3).

    subsets : explicit sub-sets (index lists). If None, the partition of
              plan §3.3.1 / OQ6 is drawn with `seed` (which must then be
              given, so that it can be stored) and `partition_kwargs`.
    amax_policy, hac_policy, policy, tol : see `features_1_17`.

    Returns a dict {name: value} in the order of FEATURE_NAMES; with
    return_info=True, (dict, info): the tie and imputation flags of
    `features_1_17` and `features_18_27` merged, plus "subsets" (the index
    lists used), so the caller can log tie frequency (OQ7/OQ8) and store
    the partition.
    """
    single_ok = False
    if subsets is None:
        if seed is None:
            raise ValueError("give `subsets` or a `seed` for the partition")
        subsets = partition(len(R), seed, **(partition_kwargs or {}))
        # OQ6 with the halves fallback switched off: one sub-set for
        # 30 <= N < 60, degenerate SG features (decision PT3).
        single_ok = (partition_kwargs or {}).get("halves_if_k_below_2", True) is False
    amax_policy, hac_policy = _policies(policy, amax_policy, hac_policy)
    f, info = features_1_17(R, answers, var_kind=var_kind, strict=strict,
                            amax_policy=amax_policy, hac_policy=hac_policy,
                            tol=tol, return_info=True)
    g, sg_info = features_18_27(R, answers, subsets, var_kind=var_kind,
                                amax_mode=amax_mode,
                                empty_supporters=empty_supporters,
                                amax_policy=amax_policy, return_info=True,
                                allow_single_subset=single_ok)
    f.update(g)
    info.update(sg_info)
    info["subsets"] = [list(idx) for idx in subsets]
    assert tuple(f) == FEATURE_NAMES
    return (f, info) if return_info else f


#: Implemented values of the switches read by `options_from_config`.
_IMPLEMENTED_SINGLE = {
    "OQ10": 2,                              # entropy base
    "OQ11": "euclidean",                    # D_S_Uniform distance
    "OQ14": "f_hac_with_oq8_tie_rule",      # feature 12
    "OQ36": "use_confidence",               # feature 22
}
_CHOICES = {
    "OQ9": ("population_sd", "population_variance"),
    "OQ12": ("per_subset", "global"),
    "OQ13": ("strict", "non_strict"),
}


def options_from_config(config=None):
    """Keyword arguments of `cdmcr_features` taken from config/defaults.yaml.

    Used switches: OQ6 (partition), OQ7 (empty supporters; a_max/a_min tie
    policy), OQ8 (HAC tie policy of feature 12, by OQ14), OQ9, OQ12, OQ13,
    and `tie_abs_tol` (AM3/AM9).

    Checked switches: OQ10, OQ11, OQ14, OQ36 and OQ7
    `support_vector_includes_zero_vote_answers` have one implemented value
    each; OQ7 `ties` supports canonical_first only. Any other value raises
    (NotImplementedError, or ValueError for a value that is not an option),
    so a changed switch is never silently ignored.
    """
    from cdm.config import get_switch, load_config, tie_abs_tol
    config = config if config is not None else load_config()
    for oq, implemented in _IMPLEMENTED_SINGLE.items():
        value = get_switch(oq, config)
        if isinstance(value, bool) or value != implemented:
            raise NotImplementedError(
                f"{oq} = {value!r} is not implemented; only {implemented!r} is")
    for oq, choices in _CHOICES.items():
        value = get_switch(oq, config)
        if value not in choices:
            raise ValueError(f"{oq} = {value!r} is not one of {choices}")
    oq6 = get_switch("OQ6", config)
    oq7 = get_switch("OQ7", config)
    if oq7["support_vector_includes_zero_vote_answers"] is not True:
        raise NotImplementedError(
            "OQ7 support_vector_includes_zero_vote_answers must be true: "
            "S(R) over the voted answers only is not implemented")
    if oq7["ties"] != "canonical_first":
        raise NotImplementedError(
            f"OQ7 ties = {oq7['ties']!r} is not implemented for a_max/a_min; "
            "the plan defines canonical_first only")
    if oq7["empty_supporters_in_subset"] not in ("nan_then_impute_subset_avgc", "zero"):
        raise ValueError(
            f"OQ7 empty_supporters_in_subset = {oq7['empty_supporters_in_subset']!r} "
            "is not an option")
    if not isinstance(oq6["halves_if_k_below_2"], bool):
        raise ValueError("OQ6 halves_if_k_below_2 must be true or false")
    return {
        "partition_kwargs": {
            "subset_size": oq6["subset_size"],
            "scheme": oq6["scheme"],
            "halves_if_k_below_2": oq6["halves_if_k_below_2"],
        },
        "var_kind": get_switch("OQ9", config),
        "strict": get_switch("OQ13", config) == "strict",
        "amax_mode": get_switch("OQ12", config),
        "empty_supporters": oq7["empty_supporters_in_subset"],
        "amax_policy": oq7["ties"],
        "hac_policy": AG.oq8_policy_from_config(config),
        "tol": tie_abs_tol(config),
    }
