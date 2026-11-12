"""Shared pieces of the World-A loaders (step A8 = plan §2.1.1 Steps D1-D2).

A loader first reads a source into `RawCase` objects: the raw answer codes
and raw numbers exactly as stored in the file, with nothing dropped. The
functions here then apply the response-level switches and turn a RawCase
into a validated `cdm.schema.Case`.

Every choice is an argument. The values the plan names are:
  OQ2  confidence normalization  (MARTINIE_CONF, LIKERT_CONF)
  OQ3  predicted-support construction (`predicted_support`)
  OQ4  Martinie vote-inconsistency exclusion, complete responses (Step D2)
  OQ5  answer coding (ANSWER_CODINGS)
"""
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cdm.schema import Case, Response

DOMAIN_GENERAL = "General"
DOMAIN_NBA = "NBA"
DOMAIN_NFL = "NFL"
DOMAIN_POLITICS = "Politics"

#: plan §8 OQ2, Martinie options (p = stated probability that the statement
#: is true, 0-100).
MARTINIE_CONF = (
    "a_abs_p_minus_50_over_50",
    "b_max_p_100_minus_p_over_100",
    "c_p_if_true_else_100_minus_p",
)
#: plan §8 OQ2, Likert options (raw scale 1..5).
LIKERT_CONF = ("a_c_minus_1_over_4", "b_c_over_5")
#: plan §8 OQ5.
ANSWER_CODINGS = (
    "a_natural_source_coding",
    "b_raw_file_option_order",
    "c_random_per_case_seeded",
)


@dataclass
class RawResponse:
    """One raw response. Missing fields are None; nothing is normalized."""
    participant: Any
    vote: Any       # raw answer code
    conf: Any       # raw confidence number (Likert 1..5, or Martinie p 0..100)
    others: Any     # raw predicted-support number, in percent
    extra: Any = None


@dataclass
class RawCase:
    """One problem as stored in a source file.

    options       : raw answer codes offered, in raw order.
    labels        : raw code -> label used in Case.answers.
    natural_first : the code that is answer a1 under OQ5 option (a)
                    (plan §2.1.1 D1: "a1 = True / home team / first-listed
                    candidate"). For NFL it is the SECOND-NAMED team of the
                    header, which is usually but not always the home team
                    (decision W7).
    winner        : raw code of the correct answer, None if the source has
                    no outcome.
    conf_kind     : "likert" or "probability_true".
    others_kind   : "agree_with_me" (percent of others agreeing with the
                    respondent's own pick) or "share_true" (percent of
                    others answering True).
    meta          : free-form facts about the columns (headers, flags).
    """
    case_id: str
    domain: str
    source: str
    options: Tuple[Any, ...]
    labels: Dict[Any, str]
    natural_first: Any
    winner: Any
    responses: List[RawResponse]
    conf_kind: str
    others_kind: str
    meta: Dict[str, Any] = field(default_factory=dict)


# ---- small helpers -----------------------------------------------------------

def require_file(path):
    """Raise FileNotFoundError with a pointer to docs/DATA.md if `path` is absent."""
    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"raw data file not found: {path}. The data sets are not part of this "
            "repository; see docs/DATA.md for where to obtain them and for the "
            "expected layout of the data folder (default data/, or env CDM_DATA_ROOT).")
    return path


def is_missing(x):
    """True for an empty cell: None, NaN, or a blank string."""
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == ""
    try:
        return x != x          # NaN
    except Exception:          # pragma: no cover
        return False


def clean_number(x):
    """Cell -> int (if integral), float, or None for a missing cell.

    Non-numeric strings are returned unchanged (the caller decides)."""
    if is_missing(x):
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        f = float(x)
        return int(f) if f == int(f) else f
    if isinstance(x, str):
        try:
            f = float(x.strip())
        except ValueError:
            return x.strip()
        return int(f) if f == int(f) else f
    try:                        # numpy scalars
        f = float(x)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return x


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clip01(x):
    """Plan §8 OQ3: "Clip to [0,1]"."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


# ---- response-level switches -------------------------------------------------

def martinie_vote_inconsistent(r, true_code=2, false_code=1):
    """Martinie's own exclusion (preprocessData.m): voted True with p < 50,
    or voted False with p > 50. p = 50 is never inconsistent. Plan §8 OQ4."""
    if r.vote == true_code:
        return r.conf < 50
    if r.vote == false_code:
        return r.conf > 50
    return False


def filter_responses(raw, complete_only=True, drop_participants=(),
                     martinie_drop_vote_inconsistent=False, options=None):
    """Apply the response-level filters to one RawCase.

    Order: (1) named participants, (2) no vote, (3) vote that is not one of
    the answer codes, (4) Step D2 completeness: vote, confidence and
    predicted support all present (TH p.43), (5) Martinie's
    vote-inconsistency rule (OQ4).

    options : the answer codes of the case; default raw.options. Passing a
              shorter tuple drops the votes for the other codes.

    Returns (kept responses, counts dict of how many each step removed).
    """
    options = tuple(raw.options if options is None else options)
    drop_participants = set(drop_participants)
    kept = []
    n = {"participant_dropped": 0, "no_vote": 0, "vote_not_an_answer": 0,
         "incomplete": 0, "martinie_vote_inconsistent": 0}
    for r in raw.responses:
        if r.participant in drop_participants:
            if not is_missing(r.vote):
                n["participant_dropped"] += 1
            continue
        if is_missing(r.vote):
            n["no_vote"] += 1
            continue
        if r.vote not in options:
            n["vote_not_an_answer"] += 1
            continue
        if complete_only and (is_missing(r.conf) or is_missing(r.others)):
            n["incomplete"] += 1
            continue
        if (martinie_drop_vote_inconsistent and raw.conf_kind == "probability_true"
                and not is_missing(r.conf) and martinie_vote_inconsistent(r)):
            n["martinie_vote_inconsistent"] += 1
            continue
        kept.append(r)
    return kept, n


# ---- confidence (OQ2) --------------------------------------------------------

def martinie_confidence(vote_is_true, p, option):
    """Plan §8 OQ2, Martinie. p = probability (0-100) that the statement is
    true. (a) |p-50|/50; (b) max(p,100-p)/100; (c) p/100 for a True vote and
    (100-p)/100 for a False vote."""
    p = float(p)
    if option == "a_abs_p_minus_50_over_50":
        return abs(p - 50.0) / 50.0
    if option == "b_max_p_100_minus_p_over_100":
        return max(p, 100.0 - p) / 100.0
    if option == "c_p_if_true_else_100_minus_p":
        return (p if vote_is_true else 100.0 - p) / 100.0
    raise ValueError(f"unknown Martinie confidence option {option!r}")


def likert_confidence(x, option, scale_max=5, clip=False):
    """Plan §8 OQ2, Likert. (a) (x-1)/(scale_max-1); (b) x/scale_max.

    scale_max is 5 for every source as elicited. It is 7 only for the
    out-of-scale codes of the original Rutchick NFL file under a min-max
    reading (see rutchick.NFL_ORIGINAL_CONF). clip=True cuts values above 1
    (the "clip" reading of the same file)."""
    x = float(x)
    if option == "a_c_minus_1_over_4":
        c = (x - 1.0) / (scale_max - 1.0)
    elif option == "b_c_over_5":
        c = x / float(scale_max)
    else:
        raise ValueError(f"unknown Likert confidence option {option!r}")
    return clip01(c) if clip else c


# ---- predicted support (OQ3) -------------------------------------------------

#: Predicted support of the answers a respondent did NOT choose, when a case
#: has more than two answers (46 election races; the plan is silent, AM15).
#:   complement_each : 1 - x for EVERY other answer. This is the formula of
#:                     Rutchick's own SP analysis (Election Analysis
#:                     workbook, rows 318-351). The vector then does not sum to 1 (allowed:
#:                     docs/IMPLEMENTATION_DECISIONS.md S4).
#:   split_evenly    : the remainder 1 - x split evenly over the other
#:                     answers. invented for this implementation; kept as a switch only.
#: For a binary case both give ps[other] = 1 - x.
PS_OTHER_ANSWERS = ("complement_each", "split_evenly")


def predicted_support(raw, r, codes, other_answers="complement_each"):
    """Predicted-support vector from the single reported number.

    Plan §2.1 step 2 and §8 OQ3:
      agree_with_me : ps[v] = x/100, ps[other] = 1 - x/100   (Rutchick, Lee)
      share_true    : ps[True] = y/100, ps[False] = 1 - y/100 (Martinie)
    Clipped to [0,1].

    The plan is silent for more than two answers (it assumes every World-A
    case is binary; 46 election races are not). `other_answers` selects the
    reading, see PS_OTHER_ANSWERS. Neither is a plan rule (AM15).
    """
    if other_answers not in PS_OTHER_ANSWERS:
        raise ValueError(f"unknown ps rule {other_answers!r}; expected one of "
                         f"{PS_OTHER_ANSWERS}")
    x = clip01(float(r.others) / 100.0)
    if raw.others_kind == "share_true":
        if len(codes) != 2:
            raise ValueError("share_true needs a binary case")
        true_code = raw.meta["true_code"]
        return tuple(x if code == true_code else 1.0 - x for code in codes)
    if raw.others_kind == "agree_with_me":
        if other_answers == "split_evenly":
            rest = (1.0 - x) / (len(codes) - 1)
        else:
            rest = 1.0 - x
        return tuple(x if code == r.vote else rest for code in codes)
    raise ValueError(f"unknown others_kind {raw.others_kind!r}")


# ---- answer coding (OQ5) -----------------------------------------------------

def ordered_codes(raw, coding="a_natural_source_coding", seed=0, options=None):
    """Order of the raw answer codes in Case.answers. Plan §8 OQ5.

    (a) natural source coding: raw.natural_first first, the rest in raw order.
    (b) the raw file option order.
    (c) random per case: a shuffle seeded by (seed, case_id), so it does not
        depend on which other cases are in the build.
    """
    codes = list(raw.options if options is None else options)
    if coding == "b_raw_file_option_order":
        return tuple(codes)
    if coding == "a_natural_source_coding":
        if raw.natural_first in codes:
            codes.remove(raw.natural_first)
            codes.insert(0, raw.natural_first)
        return tuple(codes)
    if coding == "c_random_per_case_seeded":
        import numpy as np
        digest = hashlib.sha256(f"{seed}|{raw.case_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        return tuple(codes[i] for i in rng.permutation(len(codes)))
    raise ValueError(f"unknown answer coding {coding!r}")


# ---- RawCase -> Case ---------------------------------------------------------

def make_case(raw, responses=None, martinie_conf="a_abs_p_minus_50_over_50",
              likert_conf="a_c_minus_1_over_4", likert_scale_max=5,
              likert_clip=False, answer_coding="a_natural_source_coding",
              coding_seed=0, options=None, ps_other_answers="complement_each"):
    """Build a validated Case from a RawCase and its kept responses.

    responses : complete responses (vote, conf, others all present), e.g.
                from `filter_responses`; default: that filter with its
                defaults.
    ps_other_answers : PS_OTHER_ANSWERS; matters only for more than two
                answers.
    Raises ValueError if the source has no outcome for the case (Lee) or the
    winner is not one of the answer codes.
    """
    if responses is None:
        responses, _ = filter_responses(raw, options=options)
    codes = ordered_codes(raw, answer_coding, coding_seed, options)
    if raw.winner is None:
        raise ValueError(f"case {raw.case_id!r}: the source holds no outcome")
    if raw.winner not in codes:
        raise ValueError(f"case {raw.case_id!r}: winner {raw.winner!r} is not "
                         f"one of the answers {codes!r}")
    out = []
    for r in responses:
        if raw.conf_kind == "probability_true":
            c = martinie_confidence(r.vote == raw.meta["true_code"], r.conf,
                                    martinie_conf)
        elif raw.conf_kind == "likert":
            c = likert_confidence(r.conf, likert_conf, likert_scale_max,
                                  likert_clip)
        else:
            raise ValueError(f"unknown conf_kind {raw.conf_kind!r}")
        ps = predicted_support(raw, r, codes, ps_other_answers)
        out.append(Response(v=raw.labels[r.vote], c=float(c),
                            ps=tuple(float(x) for x in ps)))
    return Case(case_id=f"{raw.domain}:{raw.case_id}",
                answers=tuple(raw.labels[c] for c in codes),
                correct=raw.labels[raw.winner],
                responses=tuple(out), domain=raw.domain)


def count_correct(raw, responses):
    """(N, number of responses that chose the correct answer)."""
    return len(responses), sum(1 for r in responses if r.vote == raw.winner)
