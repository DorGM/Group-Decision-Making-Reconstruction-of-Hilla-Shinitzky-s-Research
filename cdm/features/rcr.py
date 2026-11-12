"""RCR: the 14 features of one response within its set of responses.

Plan §3.4 (TH Table 3.3 p.30, Table A.3 p.111; MCF Table 1). Step B1 of
plan §9.3.

Notation: R = the set of responses the response r belongs to (in RP/AP this
is one of the 77 sub-sets, plan §4.3: "computed within its sub-set");
in-group of r = R(r.v), which includes r itself; out-group = R minus R(r.v)
(MCF §3; plan §3.1).

  1  ChosenMajAns  1 if r.v is the majority answer, else 0
  2  Conf          r.c
  3  ConfD         r.c - Average(C(R))
  4  ConfD_IG      r.c - Average(C(in-group))
  5  ConfD_OG      r.c - Average(C(out-group))
  6  PSv           r.ps[r.v]
  7  dPSv          S(r.v, R) - r.ps[r.v]
  8  PSvD          r.ps[r.v] - Average(PS(r.v, R))
  9  PSvD_IG       r.ps[r.v] - Average(PS(r.v, in-group))
  10 PSvD_OG       r.ps[r.v] - Average(PS(r.v, out-group))
  11 ChiSq         chi2(r.ps, S(R))
  12 PChiSq        chi2(r.ps, Average PS(., R))
  13 PChiSq_IG     chi2(r.ps, Average PS(., in-group))
  14 PChiSq_OG     chi2(r.ps, Average PS(., out-group))

chi2(x, y) = 1/2 * sum over all a in A_P of (x[a] - y[a])^2 / (x[a] + y[a])
(TH Table 3.3 rows 11-14, rendered page). A term with x[a] = y[a] = 0 is
taken as 0 [INFERRED, plan §3.4: TH Table A.4 needs answers with ps = 0 and
S = 0 to contribute nothing].

Conventions:
  OQ28  majority answer under a vote tie: 1 for every tied top answer
        (default, flagged in info); alternative "canonical_first".
  Empty out-group (every response of R gives the same answer): features
        5, 10 and 14 are undefined. The thesis and the plan give no rule;
        they are returned as NaN and counted in info["empty_out_group"]
        (decision B1-2). The in-group is never empty (it holds r).

No float tolerance is used: the only comparison is between integer vote
counts (the majority answer).
"""
import math

from cdm import primitives as P

NAN = float("nan")

#: Feature names in the order of TH Table 3.3 p.30 (plan §3.4 table).
FEATURE_NAMES = (
    "ChosenMajAns", "Conf", "ConfD", "ConfD_IG", "ConfD_OG",
    "PSv", "dPSv", "PSvD", "PSvD_IG", "PSvD_OG",
    "ChiSq", "PChiSq", "PChiSq_IG", "PChiSq_OG",
)

#: OQ28 options (plan §8).
MAJORITY_RULES = ("one_for_all_tied_top_answers", "canonical_first")
DEFAULT_MAJORITY_RULE = "one_for_all_tied_top_answers"


def chi_sq_distance(x, y):
    """1/2 * sum_a (x[a] - y[a])^2 / (x[a] + y[a]).

    TH Table 3.3 rows 11-14 (the symmetric chi-square distance between two
    share vectors; plan §3.4). A term whose denominator is 0 (both shares
    are 0) contributes 0 [INFERRED from TH Table A.4]. NaN if any value is
    NaN (an undefined reference, e.g. an empty out-group). Negative values
    are rejected: the formula is for shares.
    """
    x, y = list(x), list(y)
    if len(x) != len(y):
        raise ValueError(f"vectors of different length: {len(x)} and {len(y)}")
    if any(P._isnan(v) for v in x) or any(P._isnan(v) for v in y):
        return NAN
    terms = []
    for xi, yi in zip(x, y):
        if xi < 0 or yi < 0:
            raise ValueError(f"negative share: {xi!r}, {yi!r}")
        den = xi + yi
        if den == 0:
            continue                      # 0/0 := 0
        terms.append((xi - yi) ** 2 / den)
    return 0.5 * math.fsum(terms)


def majority_answers(R, answers, majority_rule=DEFAULT_MAJORITY_RULE):
    """The answers that count as "the majority answer" (features RCR 1 and
    ACR 1), in canonical order. Plan §8 OQ28.

    one_for_all_tied_top_answers : every answer with the largest number of
                                   votes (default).
    canonical_first              : the first of them in the order of A_P
                                   (= a_max under OQ7).
    """
    if majority_rule not in MAJORITY_RULES:
        raise ValueError(f"unknown majority rule {majority_rule!r}; "
                         f"expected one of {MAJORITY_RULES}")
    tied = P.a_max_ties(R, answers)
    if not tied:
        raise ValueError("the majority answer is undefined for an empty set")
    return tied if majority_rule == DEFAULT_MAJORITY_RULE else tied[:1]


def _mean(xs):
    """Average with math.fsum (independent of the order); NaN if empty."""
    xs = list(xs)
    return math.fsum(xs) / len(xs) if xs else NAN


def group_statistics(R, answers):
    """Per-answer quantities shared by RCR and ACR (plan §3.4, §3.5).

    Returns a dict:
      n, counts {a: votes}, support {a: S(a, R)}, support_vector,
      avg_c            Average(C(R))
      avg_c_in {a}     Average(C(R(a)))            NaN if R(a) is empty
      avg_c_out {a}    Average(C(R minus R(a)))    NaN if that set is empty
      avg_ps           [Average(PS(b, R)) for b in A_P]
      avg_ps_in {a}    the same vector over R(a)          (NaN entries if empty)
      avg_ps_out {a}   the same vector over R minus R(a)  (NaN entries if empty)
    """
    answers = list(answers)
    n = len(R)
    if n == 0:
        raise ValueError("the features need at least one response")
    m = len(answers)
    counts = dict(zip(answers, P.vote_counts(R, answers)))

    def ps_means(rs):
        return [_mean(r.ps[i] for r in rs) for i in range(m)]

    out = {
        "n": n, "counts": counts,
        "support": {a: counts[a] / n for a in answers},
        "avg_c": _mean(r.c for r in R),
        "avg_ps": ps_means(R),
        "avg_c_in": {}, "avg_c_out": {}, "avg_ps_in": {}, "avg_ps_out": {},
    }
    out["support_vector"] = [out["support"][a] for a in answers]
    for a in answers:
        inside = [r for r in R if r.v == a]
        outside = [r for r in R if r.v != a]
        out["avg_c_in"][a] = _mean(r.c for r in inside)
        out["avg_c_out"][a] = _mean(r.c for r in outside)
        out["avg_ps_in"][a] = ps_means(inside)
        out["avg_ps_out"][a] = ps_means(outside)
    return out


def rcr_features(R, answers, majority_rule=DEFAULT_MAJORITY_RULE,
                 return_info=False):
    """The 14 RCR features of every response of R.

    TH Table 3.3 p.30 (descriptions in Table A.3 p.111); plan §3.4.

    R        : the set of responses (a sub-set in RP/AP, plan §4.3).
    answers  : A_P in its canonical order; the chi-square sums run over all
               of A_P, zero-vote answers included (TH Table 3.3: "a_i in
               A_P").
    majority_rule : plan §8 OQ28 (see `majority_answers`).

    Returns a list with one dict {feature name: value} per response, in the
    order of R and with the keys in the order of FEATURE_NAMES. With
    return_info=True, (list, info) where info holds
      majority_answers : the answers counted as majority (canonical order)
      majority_tie     : True if several answers share the top vote count
      empty_out_group  : number of responses whose out-group is empty
                         (features 5, 10, 14 are NaN for them)
    """
    answers = list(answers)
    g = group_statistics(R, answers)
    index = {a: i for i, a in enumerate(answers)}
    top = P.a_max_ties(R, answers)
    majority = majority_answers(R, answers, majority_rule)
    rows = []
    empty_out = 0
    for r in R:
        v = r.v
        i = index[v]
        psv = r.ps[i]
        has_out = g["counts"][v] < g["n"]
        if not has_out:
            empty_out += 1
        rows.append({
            # 1  1 if r.v is the majority answer; 0 else
            "ChosenMajAns": 1 if v in majority else 0,
            # 2  r.c
            "Conf": r.c,
            # 3  r.c - Average(C(R_P))
            "ConfD": r.c - g["avg_c"],
            # 4  r.c - Average(C(R_P(r.v)))            in-group, includes r
            "ConfD_IG": r.c - g["avg_c_in"][v],
            # 5  r.c - Average(C(out-group))
            "ConfD_OG": r.c - g["avg_c_out"][v] if has_out else NAN,
            # 6  r.ps[r.v]
            "PSv": psv,
            # 7  S(r.v, R_P) - r.ps[r.v]
            "dPSv": g["support"][v] - psv,
            # 8  r.ps[r.v] - Average(PS(r.v, R_P))
            "PSvD": psv - g["avg_ps"][i],
            # 9  r.ps[r.v] - Average(PS(r.v, in-group))
            "PSvD_IG": psv - g["avg_ps_in"][v][i],
            # 10 r.ps[r.v] - Average(PS(r.v, out-group))
            "PSvD_OG": psv - g["avg_ps_out"][v][i] if has_out else NAN,
            # 11 chi2(r.ps, S(R_P))
            "ChiSq": chi_sq_distance(r.ps, g["support_vector"]),
            # 12 chi2(r.ps, Average PS(., R_P))
            "PChiSq": chi_sq_distance(r.ps, g["avg_ps"]),
            # 13 chi2(r.ps, Average PS(., in-group))
            "PChiSq_IG": chi_sq_distance(r.ps, g["avg_ps_in"][v]),
            # 14 chi2(r.ps, Average PS(., out-group))
            "PChiSq_OG": (chi_sq_distance(r.ps, g["avg_ps_out"][v])
                          if has_out else NAN),
        })
    if not return_info:
        return rows
    info = {"majority_answers": list(majority), "majority_tie": len(top) > 1,
            "empty_out_group": empty_out}
    return rows, info


def options_from_config(config=None):
    """Keyword arguments of `rcr_features` / `acr_features` from the config.

    OQ28 -> majority_rule. A value that is not one of the plan's two options
    raises, so a changed switch is never silently ignored.
    """
    from cdm.config import get_switch, load_config
    config = config if config is not None else load_config()
    rule = get_switch("OQ28", config)
    if rule not in MAJORITY_RULES:
        raise ValueError(f"OQ28 = {rule!r} is not one of {MAJORITY_RULES}")
    return {"majority_rule": rule}
