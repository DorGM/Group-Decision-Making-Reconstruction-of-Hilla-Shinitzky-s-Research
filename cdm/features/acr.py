"""ACR: the 10 features of one answer within a set of responses.

Plan §3.5 (TH Table 3.4 p.31, Table A.5 p.112; MCF Table 2). Step B2 of
plan §9.3.

One instance per answer a in A_P "which has at least one vote in the given
set of responses" (TH p.30, p.41; MCF §3.1.2). Supporters of a = R(a)
("in-group"), non-supporters = R minus R(a) ("out-group") (TH p.31).

  1  IsMajority  1 if a is the majority answer, else 0
  2  Support     S(a, R)
  3  AvgPS       Average(PS(a, R))
  4  AvgPS_IG    Average(PS(a, R(a)))
  5  AvgPS_OG    Average(PS(a, R minus R(a)))
  6  dPS         S(a, R) - AvgPS
  7  dPS_IG      S(a, R) - AvgPS_IG
  8  dPS_OG      S(a, R) - AvgPS_OG
  9  AvgChiSq    mean over r in R(a) of ChiSq(r)      (RCR feature 11)
  10 AvgConf     Average(C(R(a)))

Conventions:
  OQ28  majority answer under a vote tie: 1 for every tied top answer
        (default, flagged); alternative "canonical_first".
  Empty out-group (every response gives answer a): features 5 and 8 are
        undefined; the thesis and the plan give no rule. They are NaN and
        the case is flagged in info["empty_out_group"] (decision B1-2).
"""
import math

from cdm import primitives as P
from cdm.features import rcr

NAN = float("nan")

#: Feature names in the order of TH Table 3.4 p.31 (plan §3.5 table).
FEATURE_NAMES = (
    "IsMajority", "Support", "AvgPS", "AvgPS_IG", "AvgPS_OG",
    "dPS", "dPS_IG", "dPS_OG", "AvgChiSq", "AvgConf",
)


def voted_answers(R, answers):
    """The answers with at least one vote in R, in the canonical order of
    A_P: the ACR instances of R (TH p.30; plan §3.5)."""
    counts = P.vote_counts(R, answers)
    return [a for a, k in zip(answers, counts) if k > 0]


def acr_features(R, answers, majority_rule=rcr.DEFAULT_MAJORITY_RULE,
                 return_info=False):
    """The 10 ACR features of every answer with at least one vote in R.

    TH Table 3.4 p.31 (descriptions in Table A.5 p.112); plan §3.5.

    Returns a list of (answer, {feature name: value}) in the canonical order
    of A_P, zero-vote answers left out; the dict keys are in the order of
    FEATURE_NAMES. With return_info=True, (list, info) where info holds
      majority_answers : the answers counted as majority (canonical order)
      majority_tie     : True if several answers share the top vote count
      empty_out_group  : True if one answer has every vote (features 5 and
                         8 of that answer are NaN)
    """
    answers = list(answers)
    g = rcr.group_statistics(R, answers)
    index = {a: i for i, a in enumerate(answers)}
    top = P.a_max_ties(R, answers)
    majority = rcr.majority_answers(R, answers, majority_rule)
    sv = g["support_vector"]
    out = []
    empty_out = False
    for a in voted_answers(R, answers):
        i = index[a]
        s = g["support"][a]
        has_out = g["counts"][a] < g["n"]
        empty_out = empty_out or not has_out
        avg_ps = g["avg_ps"][i]
        avg_ps_in = g["avg_ps_in"][a][i]
        avg_ps_out = g["avg_ps_out"][a][i] if has_out else NAN
        chis = [rcr.chi_sq_distance(r.ps, sv) for r in R if r.v == a]
        out.append((a, {
            # 1  1 if a is the majority answer; 0 else
            "IsMajority": 1 if a in majority else 0,
            # 2  S(a, R_P)
            "Support": s,
            # 3  Average(PS(a, R_P))
            "AvgPS": avg_ps,
            # 4  Average(PS(a, R_P(a)))               supporters
            "AvgPS_IG": avg_ps_in,
            # 5  Average(PS(a, non-supporters))
            "AvgPS_OG": avg_ps_out,
            # 6  S(a, R_P) - AvgPS
            "dPS": s - avg_ps,
            # 7  S(a, R_P) - AvgPS_IG
            "dPS_IG": s - avg_ps_in,
            # 8  S(a, R_P) - AvgPS_OG
            "dPS_OG": s - avg_ps_out if has_out else NAN,
            # 9  sum_{r in R_P(a)} ChiSq(r) / |R_P(a)|
            "AvgChiSq": math.fsum(chis) / len(chis),
            # 10 Average(C(R_P(a)))
            "AvgConf": g["avg_c_in"][a],
        }))
    if not return_info:
        return out
    info = {"majority_answers": list(majority), "majority_tie": len(top) > 1,
            "empty_out_group": empty_out}
    return out, info


#: Same switch as RCR (OQ28).
options_from_config = rcr.options_from_config
