"""Response and Case data classes with validation.

Plan §3.1 (TH §3.1 p.21-24, Table 3.1):
  case     = (P, A_P, a*, R_P)
  response = (v in A_P, c in [0,1], ps in [0,1]^|A_P|)
"""
import math
from dataclasses import dataclass
from typing import Hashable, Optional, Tuple


class ValidationError(ValueError):
    """Raised when a response or a case breaks the TH Table 3.1 definitions."""


_NOT_NUMBERS = (bool, str, bytes, bytearray)


def _is_bool(x):
    return isinstance(x, bool) or type(x).__name__ in ("bool_", "bool")


def _in_unit_interval(x):
    """True if x is a real number in [0, 1].

    NaN and inf are rejected, and so are text and booleans (they convert
    to float but are not confidences or shares).
    """
    if isinstance(x, _NOT_NUMBERS) or _is_bool(x):
        return False
    try:
        x = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and 0.0 <= x <= 1.0


@dataclass(frozen=True)
class Response:
    """One response r = (v, c, ps). TH Table 3.1; plan §3.1.

    v  : the chosen answer (must be one of the case's answers).
    c  : confidence in [0, 1].
    ps : predicted support, one value in [0, 1] per answer, in the same
         order as Case.answers.
    """
    v: Hashable
    c: float
    ps: Tuple[float, ...]

    def __post_init__(self):
        # The values are stored as converted floats, so what was checked is
        # what is kept. Text, booleans and mappings are rejected: a dict
        # keyed by answer would otherwise be stored as its KEYS.
        c = self.c
        if isinstance(c, _NOT_NUMBERS) or _is_bool(c):
            raise ValidationError(f"confidence {c!r} is not a number")
        try:
            c = float(c)
        except (TypeError, ValueError):
            raise ValidationError(f"confidence {self.c!r} is not a number") from None
        ps = self.ps
        if isinstance(ps, _NOT_NUMBERS + (dict,)) or _is_bool(ps):
            raise ValidationError(
                f"ps must be a sequence of numbers in the order of the answers, "
                f"got {type(ps).__name__} {ps!r}")
        try:
            ps = tuple(ps)
        except TypeError:
            raise ValidationError(
                f"ps must be a sequence of numbers, got {self.ps!r}") from None
        out = []
        for p in ps:
            if isinstance(p, _NOT_NUMBERS) or _is_bool(p):
                raise ValidationError(f"predicted support {p!r} is not a number")
            try:
                out.append(float(p))
            except (TypeError, ValueError):
                raise ValidationError(
                    f"predicted support {p!r} is not a number") from None
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "ps", tuple(out))


@dataclass(frozen=True)
class Case:
    """One case (P, A_P, a*, R_P). TH Table 3.1; plan §3.1.

    case_id   : identifier of the problem P.
    answers   : A_P, in a fixed canonical order (this order is used by the
                tie policy, plan §8 OQ7/OQ8).
    correct   : a*, the correct answer.
    responses : R_P.
    domain    : optional label (General/NBA/NFL/Politics), plan §2.1.1 D1.
    """
    case_id: Hashable
    answers: Tuple[Hashable, ...]
    correct: Hashable
    responses: Tuple[Response, ...]
    domain: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.answers, (str, bytes)):
            # a plain string would be split into its characters
            raise ValidationError(
                f"answers must be a list of answers, got the text {self.answers!r}")
        try:
            object.__setattr__(self, "answers", tuple(self.answers))
        except TypeError:
            raise ValidationError(
                f"answers must be a list of answers, got {self.answers!r}") from None
        if isinstance(self.responses, (str, bytes, dict)):
            raise ValidationError("responses must be a list of Response objects")
        try:
            object.__setattr__(self, "responses", tuple(self.responses))
        except TypeError:
            raise ValidationError(
                f"responses must be a list of Response objects, "
                f"got {self.responses!r}") from None
        validate_case(self)

    @property
    def n(self):
        """|R_P|, the number of responses."""
        return len(self.responses)


def validate_response(r, answers):
    """Check one response against A_P. Plan §9.3 A1 (TH Table 3.1).

    Rejects: v not in A_P, c outside [0,1], ps length != |A_P|,
    ps values outside [0,1].
    """
    if r.v not in answers:
        raise ValidationError(f"vote {r.v!r} is not one of the answers {answers!r}")
    if not _in_unit_interval(r.c):
        raise ValidationError(f"confidence {r.c!r} is not in [0, 1]")
    if len(r.ps) != len(answers):
        raise ValidationError(
            f"ps has {len(r.ps)} values but the case has {len(answers)} answers")
    for p in r.ps:
        if not _in_unit_interval(p):
            raise ValidationError(f"predicted support {p!r} is not in [0, 1]")


def validate_case(case):
    """Check a whole case. Plan §9.3 A1 (TH Table 3.1)."""
    if len(case.answers) < 2:
        raise ValidationError("a case needs at least two answers")
    try:
        unique = len(set(case.answers)) == len(case.answers)
    except TypeError:
        raise ValidationError("answers must be hashable values") from None
    if not unique:
        raise ValidationError("answers must be unique")
    if case.correct not in case.answers:
        raise ValidationError(
            f"correct answer {case.correct!r} is not one of the answers")
    if len(case.responses) == 0:
        raise ValidationError("a case needs at least one response")
    for i, r in enumerate(case.responses):
        if not isinstance(r, Response):
            raise ValidationError(
                f"case {case.case_id!r}, response {i}: expected a Response, "
                f"got {type(r).__name__}")
        try:
            validate_response(r, case.answers)
        except ValidationError as err:
            raise ValidationError(f"case {case.case_id!r}, response {i}: {err}") from None


def case_from_dict(raw):
    """Build a validated Case from a raw dict (plan §9.3 A1: raw dict -> Case).

    Expected keys:
      "case_id", "answers" (list), "correct",
      "responses": list of {"v": ..., "c": ..., "ps": [...]},
      optional "domain".
    """
    if not isinstance(raw, dict):
        raise ValidationError(f"a raw case must be a dict, got {type(raw).__name__}")
    for key in ("case_id", "answers", "correct", "responses"):
        if key not in raw:
            raise ValidationError(f"missing key {key!r}")
    if isinstance(raw["responses"], (str, bytes, dict)) or raw["responses"] is None:
        raise ValidationError("'responses' must be a list of response dicts")
    try:
        raw_responses = list(raw["responses"])
    except TypeError:
        raise ValidationError("'responses' must be a list of response dicts") from None
    responses = []
    for i, d in enumerate(raw_responses):
        if not isinstance(d, dict):
            raise ValidationError(
                f"response {i}: expected a dict with keys v, c, ps, "
                f"got {type(d).__name__}")
        for key in ("v", "c", "ps"):
            if key not in d:
                raise ValidationError(f"response {i}: missing key {key!r}")
        try:
            responses.append(Response(v=d["v"], c=d["c"], ps=d["ps"]))
        except ValidationError as err:
            raise ValidationError(
                f"case {raw['case_id']!r}, response {i}: {err}") from None
    return Case(
        case_id=raw["case_id"],
        answers=raw["answers"],
        correct=raw["correct"],
        responses=tuple(responses),
        domain=raw.get("domain"),
    )
