"""Rule-based aggregators MR, WC, HAC, SP and the Devil's Advocate (DA).

Plan §3.2 (TH §3.1 p.24-25 for MR/WC/HAC/SP; TH §3.3.1.1 p.34-36 and
MLA §3.1 for DA). Step A4 of plan §9.3.

All functions take a set of responses R (a sequence of Response; the full
R_P or a sub-set) and the answer list A_P in its canonical order.

Scores (one per answer of A_P, in the order of A_P):
  MR  : S(a, R)                                  -> number of votes / |R|
  WC  : S(a, R) * Average(C(R(a)))               =  sum_{r in R(a)} r.c / |R|
  HAC : Average(C(R(a)))                         (NaN when R(a) is empty)
  SP  : S(a, R) - Average(PS(a, R))
Each method returns the answer with the largest score.

Ties (plan §8 OQ8): the first tied answer in the canonical order of A_P by
default; `policy` selects the other rules of the Step D5 ladder (item 1).
The tied set of every method is exposed by `tied_answers` so that the
caller can log tie frequency (OQ8 "logged").

Tie detection: MR compares integer vote counts exactly. WC, HAC and SP
are float scores; two scores within `tol` (absolute, default
primitives.DEFAULT_TIE_ABS_TOL = 1e-9 = config `tie_abs_tol`) are a tie.
This is amendment AM3; tol=0 gives exact float
equality, under which true ties are decided by rounding noise (e.g. WC
0.12 vs 0.12000000000000002). The tolerance is not used for MR, the DA
counters or the OQ8b support comparison.

Config: `options_from_config()` reads OQ8 (policy), `tie_abs_tol`, and
checks OQ8b and OQ38, so a changed switch is used or raises.
`da_tie_rule_from_config()` reads the DA tie-rule switch (DA_TIE_RULES;
default = the thesis text).

Numerical note: sums use math.fsum (correctly rounded, independent of the
order of the responses), so a score does not depend on how R is ordered.
"""
import math

from cdm import primitives as P

NAN = float("nan")

#: The four rule-based methods that DA listens to, in the thesis order of
#: TH p.35 ("{f_MR, f_WC, f_HAC, f_SP}").
RULES = ("MR", "WC", "HAC", "SP")


# ---- scores --------------------------------------------------------------

def mr_scores(R, answers):
    """MR criterion: S(a, R) for every a in A_P. TH p.24; plan §3.2.

    Returned as vote counts divided by |R|; ties are decided on the integer
    counts (see `tied_answers`), so no float issue can arise for MR.
    """
    return P.support_vector(R, answers)


def wc_scores(R, answers):
    """WC criterion: S(a, R) * Average(C(R(a))). TH p.25; plan §3.2.

    Computed in the equivalent form given by the plan (§3.2, WC row):
    sum_{r in R(a)} r.c / |R|. An answer with no supporter scores 0, which
    is what that form gives for an empty sum.
    """
    n = len(R)
    if n == 0:
        return [NAN for _ in answers]
    return [math.fsum(r.c for r in R if r.v == a) / n for a in answers]


def hac_scores(R, answers):
    """HAC criterion: Average(C(R(a))). TH p.25; plan §3.2.

    NaN for an answer with no supporter (its average confidence is
    undefined); such an answer can never be chosen by HAC.
    """
    out = []
    for a in answers:
        cs = [r.c for r in R if r.v == a]
        out.append(math.fsum(cs) / len(cs) if cs else NAN)
    return out


def sp_scores(R, answers):
    """SP criterion: S(a, R) - Average(PS(a, R)). TH p.25; plan §3.2.

    The thesis difference form, for any number of answers (plan §8 OQ38).
    Computed over all of A_P, zero-vote answers included (their S is 0).
    """
    n = len(R)
    if n == 0:
        return [NAN for _ in answers]
    out = []
    for i, a in enumerate(answers):
        s = sum(1 for r in R if r.v == a) / n
        avg_ps = math.fsum(r.ps[i] for r in R) / n
        out.append(s - avg_ps)
    return out


_SCORES = {"MR": mr_scores, "WC": wc_scores, "HAC": hac_scores, "SP": sp_scores}


def scores(method, R, answers):
    """Scores of one of MR/WC/HAC/SP for every answer, in the order of A_P."""
    if method not in _SCORES:
        raise ValueError(f"unknown method {method!r}; expected one of {RULES}")
    return _SCORES[method](R, answers)


# ---- ties and choices ----------------------------------------------------

def tied_answers(method, R, answers, tol=P.DEFAULT_TIE_ABS_TOL):
    """All answers sharing the best score of a method, in canonical order.

    MR: integer vote counts, exact (tol is not used). WC/HAC/SP: scores
    within the absolute tolerance `tol` of the best one (AM3;
    tol=0 = exact float equality). A list of length > 1 is a tie that plan
    §8 OQ8 asks to log.
    """
    tol = P.check_tol(tol)
    if method == "MR":
        return P.a_max_ties(R, answers)
    return P.tied_best(scores(method, R, answers), answers, largest=True, tol=tol)


def choose(method, R, answers, policy="canonical_first",
           tol=P.DEFAULT_TIE_ABS_TOL):
    """f_AG(R) for AG in MR/WC/HAC/SP. Ties by plan §8 OQ8 (`policy`).

    canonical_first    : default.
    canonical_second   : Step D5 item 1; the second of two tied answers.
    sp_tie_half_credit : Step D5 item 1 / OQ8 "SP tie = 0.5 (Rutchick)".
        This is a scoring rule (see `success_credit`). MR/WC/HAC keep the
        canonical-first answer. For SP an untied case returns its answer;
        a tied case raises NotImplementedError, because the plan does not
        say which single SP answer (needed by DA) a half-credit tie has.
    random             : raises NotImplementedError (not defined by the plan).
    """
    tied = tied_answers(method, R, answers, tol)
    if policy == "sp_tie_half_credit":
        if method != "SP" or len(tied) == 1:
            return P.break_tie(tied, "canonical_first")
        if len(tied) == 0:
            raise ValueError("no answer to choose from")
        raise NotImplementedError(
            "SP is tied and the policy is 'sp_tie_half_credit': the plan "
            "(§8 OQ8, Step D5 item 1) scores the tie as 0.5 correct but does "
            "not define the single SP answer, so f_SP and DA are undefined "
            "for this case. Use success_credit('SP', ...) for the score")
    return P.break_tie(tied, policy)


def success_credit(method, R, answers, correct, policy="canonical_first",
                   tol=P.DEFAULT_TIE_ABS_TOL):
    """Credit of one rule on one case: 1.0 correct, 0.0 wrong, 0.5 half.

    For every policy except 'sp_tie_half_credit' this is
    1.0 if choose(...) == correct else 0.0.

    'sp_tie_half_credit' (plan §8 OQ8: "SP tie = 0.5 (Rutchick)"; Step D5
    item 1): a tied SP case of a BINARY question scores 0.5. MR/WC/HAC and
    untied SP cases are scored on their canonical-first answer. For a tied
    SP case with more than two answers the plan gives no rule
    (NotImplementedError).
    """
    if method not in RULES:
        raise ValueError(f"unknown method {method!r}; expected one of {RULES}")
    if correct not in list(answers):
        raise ValueError(f"correct answer {correct!r} is not one of the answers")
    if policy == "sp_tie_half_credit" and method == "SP":
        tied = tied_answers("SP", R, answers, tol)
        if len(tied) > 1:
            if len(list(answers)) != 2:
                raise NotImplementedError(
                    "half credit for an SP tie is defined for binary cases "
                    "only (Rutchick S1; plan §8 OQ8)")
            return 0.5
    return 1.0 if choose(method, R, answers, policy, tol) == correct else 0.0


def mr(R, answers, policy="canonical_first", tol=P.DEFAULT_TIE_ABS_TOL):
    """Majority Rule: the answer with the highest support, a_max. TH p.24."""
    return choose("MR", R, answers, policy, tol)


def wc(R, answers, policy="canonical_first", tol=P.DEFAULT_TIE_ABS_TOL):
    """Weighted Confidence: argmax S(a)*Average(C(R(a))). TH p.25."""
    return choose("WC", R, answers, policy, tol)


def hac(R, answers, policy="canonical_first", tol=P.DEFAULT_TIE_ABS_TOL):
    """Highest Average Confidence: argmax Average(C(R(a))). TH p.25."""
    return choose("HAC", R, answers, policy, tol)


def sp(R, answers, policy="canonical_first", tol=P.DEFAULT_TIE_ABS_TOL):
    """Surprisingly Popular: argmax S(a) - Average(PS(a, R)). TH p.25."""
    return choose("SP", R, answers, policy, tol)


# ---- Devil's Advocate ----------------------------------------------------

def _check_methods(methods):
    methods = tuple(methods)
    for m in methods:
        if m not in RULES:
            raise ValueError(f"unknown method {m!r}; expected one of {RULES}")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must be unique")
    if "MR" not in methods:
        # The DA tie-break is "the answer not chosen by MR" (TH p.35) and
        # MR is kept in every ablation (plan §5.4). The plan does not
        # define DA without MR.
        raise ValueError("DA needs MR among its methods (plan §5.4: MR is always kept)")
    return methods


#: DA rule when several answers share the lowest counter.
#:   not_mr        : THESIS TEXT (TH p.35-36, MLA §3.1): "the answer that
#:                   was not chosen by MR". The default.
#:   first_answer  : SENSITIVITY OPTION, not in the thesis text: the first
#:                   tied answer in the canonical order of A_P.
#:   second_answer : SENSITIVITY OPTION: the second tied answer in the
#:                   canonical order.
#: Decision G2: under the text rule this implementation obtains General DA
#: 111; the thesis reports 108. The positional options are controlled
#: variants of this implementation, kept as labelled sensitivity options
#: only and not adopted.
DA_TIE_RULES = ("not_mr", "first_answer", "second_answer")
DEFAULT_DA_TIE_RULE = "not_mr"


def da_detail(choices, answers, methods=RULES, support=None,
              tie_rule=DEFAULT_DA_TIE_RULE):
    """Devil's Advocate from the choices of the other methods, with details.

    TH §3.3.1.1 p.35 (plan §3.2, DA row): for each a in A_P (zero-vote
    answers included) count how many of the methods chose it; return the
    answer with the lowest count. Tie-break (`tie_rule`, see DA_TIE_RULES):
    the answer not chosen by MR (thesis text, default), or a positional
    rule (labelled sensitivity option).

    choices : mapping method name -> chosen answer; must hold every method
              in `methods`.
    methods : the methods DA listens to. Default: MR, WC, HAC, SP. In the
              ablations it is the sub-set kept in that experiment (TH p.48;
              plan §5.4). MR must be included.
    support : mapping answer -> S(a, R). Needed only when several non-MR
              answers stay tied (possible only for |A_P| > 2, plan §8
              OQ8b): the tied answer with the lowest support is chosen
              (and a further tie goes to the canonical order, OQ7).
              Supports are k/|R| with one common |R|, so they are compared
              exactly (no tolerance).

    Returns a dict:
      answer        : the DA answer
      counts        : {answer: number of methods that chose it}
      tied_lowest   : answers sharing the lowest count, canonical order
      mr_tie_break  : True if the "not chosen by MR" rule was used
      oq8b_used     : True if the OQ8b lowest-support rule was used (flag)
      oq8b_candidates : the tied non-MR answers OQ8b had to choose from
                      (the thesis leaves this choice open; any of them is a
                      possible DA answer)
      oq8b_tied     : answers still tied after OQ8b (canonical first taken)
      tie_rule      : the rule used
      positional_tie_break : True if a positional rule decided
    """
    if tie_rule not in DA_TIE_RULES:
        raise ValueError(f"unknown DA tie rule {tie_rule!r}; expected one of "
                         f"{DA_TIE_RULES}")
    methods = _check_methods(methods)
    answers = list(answers)
    counts = {a: 0 for a in answers}
    for m in methods:
        if m not in choices:
            raise ValueError(f"no choice given for method {m!r}")
        if choices[m] not in counts:
            raise ValueError(f"choice {choices[m]!r} of {m} is not one of the answers")
        counts[choices[m]] += 1

    lowest = min(counts.values())
    tied = [a for a in answers if counts[a] == lowest]
    detail = {"counts": counts, "tied_lowest": list(tied),
              "mr_tie_break": False, "oq8b_used": False, "oq8b_tied": [],
              "oq8b_candidates": [], "tie_rule": tie_rule,
              "positional_tie_break": False}

    if len(tied) > 1 and tie_rule != "not_mr":
        # Sensitivity options (not the thesis text): position in A_P.
        detail["positional_tie_break"] = True
        tied = [tied[0] if tie_rule == "first_answer" else tied[1]]
    elif len(tied) > 1:
        # TH p.35: "selecting the answer that was not chosen by MR".
        candidates = [a for a in tied if a != choices["MR"]]
        detail["mr_tie_break"] = True
        if len(candidates) > 1:
            # Plan §8 OQ8b (|A_P| > 2 only): lowest support among the tied
            # non-MR answers; flagged.
            if support is None:
                raise ValueError(
                    "several non-MR answers are tied for DA (|A_P| > 2); "
                    "pass `support` to apply plan §8 OQ8b")
            detail["oq8b_used"] = True
            detail["oq8b_candidates"] = list(candidates)
            low = min(support[a] for a in candidates)
            candidates = [a for a in candidates if support[a] == low]
            detail["oq8b_tied"] = list(candidates)
        tied = candidates

    detail["answer"] = tied[0]
    return detail


def da_from_choices(choices, answers, methods=RULES, support=None,
                    tie_rule=DEFAULT_DA_TIE_RULE):
    """The DA answer given the other methods' choices. See `da_detail`."""
    return da_detail(choices, answers, methods, support, tie_rule)["answer"]


def aggregate_all(R, answers, methods=RULES, policy="canonical_first",
                  tol=P.DEFAULT_TIE_ABS_TOL, da_tie_rule=DEFAULT_DA_TIE_RULE):
    """Choices of the kept rule-based methods and of DA built on them.

    methods : the rule-based methods kept (ablation sub-set, plan §5.4;
              MR must be included). DA uses exactly these (TH p.48).
    policy  : tie policy of MR/WC/HAC/SP (plan §8 OQ8).
    tol     : tie tolerance of the float scores WC/HAC/SP (AM3).
    da_tie_rule : DA_TIE_RULES; default the thesis text ("not_mr").

    Returns {method: answer for each kept method, "DA": answer}.
    """
    methods = _check_methods(methods)
    out = {m: choose(m, R, answers, policy, tol) for m in methods}
    support = dict(zip(answers, P.support_vector(R, answers)))
    out["DA"] = da_from_choices(out, answers, methods, support, da_tie_rule)
    return out


def da(R, answers, methods=RULES, policy="canonical_first",
       tol=P.DEFAULT_TIE_ABS_TOL, da_tie_rule=DEFAULT_DA_TIE_RULE):
    """f_DA(R): Devil's Advocate over the given methods. TH p.35; plan §3.2."""
    return aggregate_all(R, answers, methods, policy, tol, da_tie_rule)["DA"]


def da_tie_rule_from_config(config=None):
    """The DA tie rule (top-level config key `da_tie_rule`), validated.
    A config without the key means the thesis text rule."""
    from cdm.config import load_config
    config = config if config is not None else load_config()
    rule = config.get("da_tie_rule", DEFAULT_DA_TIE_RULE)
    if rule not in DA_TIE_RULES:
        raise ValueError(f"da_tie_rule = {rule!r} is not one of {DA_TIE_RULES}")
    return rule


# ---- config wiring ---------------------------------------------------------

#: OQ8 values that the code implements (see `choose`, `success_credit`).
IMPLEMENTED_OQ8 = ("canonical_first", "canonical_second", "sp_tie_half_credit")
#: The only implemented values of the single-valued aggregator switches.
IMPLEMENTED_OQ8B = "lowest_support_among_tied_non_mr"
IMPLEMENTED_OQ38 = "thesis_difference_form"


def oq8_policy_from_config(config=None):
    """The OQ8 tie policy of MR/WC/HAC/SP, validated.

    Raises NotImplementedError for a value the plan names but the code does
    not implement ('random'), ValueError for an unknown value.
    """
    from cdm.config import get_switch, load_config
    config = config if config is not None else load_config()
    policy = get_switch("OQ8", config)
    if policy in IMPLEMENTED_OQ8:
        return policy
    if policy in P.TIE_POLICIES:
        raise NotImplementedError(
            f"OQ8 = {policy!r} is not implemented (plan §8 OQ8 names it but "
            f"defines no generator/seed convention); implemented: {IMPLEMENTED_OQ8}")
    raise ValueError(f"OQ8 = {policy!r} is unknown; expected one of {P.TIE_POLICIES}")


def options_from_config(config=None):
    """Keyword arguments {policy, tol} for choose / mr / wc / hac / sp /
    aggregate_all / da / success_credit, read from the config.

    OQ8          -> policy
    tie_abs_tol  -> tol (AM3)
    OQ8b, OQ38   -> checked: only one value of each is implemented, and any
                    other value raises, so a changed switch is never ignored.
    """
    from cdm.config import get_switch, load_config, tie_abs_tol
    config = config if config is not None else load_config()
    oq8b = get_switch("OQ8b", config)
    if oq8b != IMPLEMENTED_OQ8B:
        raise NotImplementedError(
            f"OQ8b = {oq8b!r} is not implemented; only {IMPLEMENTED_OQ8B!r} is")
    oq38 = get_switch("OQ38", config)
    if oq38 != IMPLEMENTED_OQ38:
        raise NotImplementedError(
            f"OQ38 = {oq38!r} is not implemented; only {IMPLEMENTED_OQ38!r} is "
            "(Prelec's normalized form is an optional extra of the plan)")
    return {"policy": oq8_policy_from_config(config), "tol": tie_abs_tol(config)}
