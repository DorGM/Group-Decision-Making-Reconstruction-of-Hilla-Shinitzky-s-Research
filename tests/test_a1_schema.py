"""A1 acceptance: rejects c not in [0,1], v not in A, ps length != |A| (TH Table 3.1)."""
import copy

import pytest

from cdm.schema import Case, Response, ValidationError, case_from_dict

GOOD = {
    "case_id": "q1",
    "answers": ["a1", "a2"],
    "correct": "a1",
    "domain": "General",
    "responses": [
        {"v": "a1", "c": 0.8, "ps": [0.7, 0.3]},
        {"v": "a2", "c": 0.0, "ps": [0.4, 0.6]},
        {"v": "a1", "c": 1.0, "ps": [1.0, 0.0]},
    ],
}


def _bad(**change):
    raw = copy.deepcopy(GOOD)
    raw["responses"][0].update(change)
    return raw


def test_good_case_is_accepted():
    case = case_from_dict(GOOD)
    assert case.n == 3
    assert case.answers == ("a1", "a2")
    assert case.correct == "a1"
    assert case.responses[0] == Response("a1", 0.8, (0.7, 0.3))
    assert case.domain == "General"


@pytest.mark.parametrize("c", [-0.01, 1.01, 5, float("nan"), float("inf"), None, "high"])
def test_rejects_confidence_outside_unit_interval(c):
    with pytest.raises(ValidationError, match="confidence"):
        case_from_dict(_bad(c=c))


def test_rejects_vote_not_in_answers():
    with pytest.raises(ValidationError, match="vote"):
        case_from_dict(_bad(v="a3"))


@pytest.mark.parametrize("ps", [[0.7], [0.5, 0.3, 0.2], []])
def test_rejects_wrong_ps_length(ps):
    with pytest.raises(ValidationError, match="ps has"):
        case_from_dict(_bad(ps=ps))


@pytest.mark.parametrize("ps", [[1.2, 0.3], [-0.1, 0.3], [float("nan"), 0.3]])
def test_rejects_ps_value_outside_unit_interval(ps):
    with pytest.raises(ValidationError, match="predicted support"):
        case_from_dict(_bad(ps=ps))


def test_rejects_correct_answer_not_in_answers():
    raw = copy.deepcopy(GOOD)
    raw["correct"] = "a9"
    with pytest.raises(ValidationError, match="correct answer"):
        case_from_dict(raw)


def test_rejects_bad_answer_sets_and_empty_case():
    for answers in (["a1"], ["a1", "a1"]):
        raw = copy.deepcopy(GOOD)
        raw["answers"] = answers
        with pytest.raises(ValidationError):
            case_from_dict(raw)
    raw = copy.deepcopy(GOOD)
    raw["responses"] = []
    with pytest.raises(ValidationError, match="at least one response"):
        case_from_dict(raw)


def test_rejects_missing_keys():
    raw = copy.deepcopy(GOOD)
    del raw["correct"]
    with pytest.raises(ValidationError, match="missing key"):
        case_from_dict(raw)
    raw = copy.deepcopy(GOOD)
    del raw["responses"][1]["ps"]
    with pytest.raises(ValidationError, match="missing key"):
        case_from_dict(raw)


def test_direct_construction_is_validated_too():
    with pytest.raises(ValidationError):
        Case("q", ("a1", "a2"), "a1", (Response("a1", 2.0, (0.5, 0.5)),))


def test_multiple_choice_case():
    case = Case("q", ("a", "b", "c", "d"), "c",
                (Response("a", 0.5, (0.25, 0.25, 0.25, 0.25)),))
    assert len(case.answers) == 4


# ---- store what was checked; error type ---------------

@pytest.mark.parametrize("c", ["0.5", b"0.5", True, False])
def test_f4_rejects_text_and_bool_confidence(c):
    with pytest.raises(ValidationError, match="confidence"):
        Response("a1", c, (0.7, 0.3))
    with pytest.raises(ValidationError, match="confidence"):
        case_from_dict(_bad(c=c))


@pytest.mark.parametrize("ps", ["01", b"01", {0: 0.7, 1: 0.3},
                                {"a1": 0.7, "a2": 0.3}, True,
                                ["0.7", "0.3"], [True, False], [0.7, None]])
def test_f4_rejects_text_dict_and_bool_ps(ps):
    with pytest.raises(ValidationError):
        Response("a1", 0.5, ps)
    with pytest.raises(ValidationError):
        case_from_dict(_bad(ps=ps))


def test_f4_values_are_stored_as_converted_floats():
    import numpy as np
    r = Response("a1", 1, [np.float32(0.5), 0])
    assert type(r.c) is float and r.c == 1.0
    assert r.ps == (0.5, 0.0) and all(type(p) is float for p in r.ps)
    r = Response("a1", np.float64(0.25), np.array([0.25, 0.75]))
    assert type(r.c) is float and r.ps == (0.25, 0.75)
    assert Response("a1", 0.8, [0.7, 0.3]) == Response("a1", 0.8, (0.7, 0.3))


@pytest.mark.parametrize("change", [
    {"ps": None}, {"ps": 0.5},
])
def test_f5_malformed_response_raises_validation_error(change):
    with pytest.raises(ValidationError):
        case_from_dict(_bad(**change))
    with pytest.raises(ValidationError):
        Response("a1", 0.5, change["ps"])


@pytest.mark.parametrize("key,value", [
    ("responses", None), ("responses", 5), ("responses", "abc"),
    ("responses", [None]), ("responses", [[0.1, 0.2]]),
    ("answers", None), ("answers", 7), ("answers", "ab"),
    ("answers", [["a1"], ["a2"]]),
])
def test_f5_malformed_case_raises_validation_error(key, value):
    raw = copy.deepcopy(GOOD)
    raw[key] = value
    with pytest.raises(ValidationError):
        case_from_dict(raw)


def test_f5_direct_construction_with_malformed_structure():
    good = Response("a1", 0.5, (0.5, 0.5))
    with pytest.raises(ValidationError):
        Case("q", ("a1", "a2"), "a1", None)
    with pytest.raises(ValidationError):
        Case("q", ("a1", "a2"), "a1", ({"v": "a1", "c": 0.5, "ps": [0.5, 0.5]},))
    with pytest.raises(ValidationError):
        Case("q", None, "a1", (good,))
    with pytest.raises(ValidationError):
        Case("q", "ab", "a", (Response("a", 0.5, (0.5, 0.5)),))
    with pytest.raises(ValidationError):
        Case("q", (["a1"], ["a2"]), "a1", (good,))
    with pytest.raises(ValidationError):
        case_from_dict(None)
