"""Loaders for Rutchick, Ross, Calvillo & Mesick 2020 (OSF u9k72):
Study 1 NFL, Study 2 Elections ("Politics"), Study 3 NBA.

Step A8, plan §2.1.1 Step D1. File layout as found in the
public files (docs/DATA.md). Each workbook has one sheet; one
case = one column triple; one response = one participant row.

  Study 1 NFL   : triples  X_PRED, X_CONF, X_OTHERS  (vote, conf, others).
                  Codes 1 = first-named team, 2 = second-named team of the
                  header W{week}_{T1}_{T2}. Row "WINNER".
  Study 3 NBA   : triples  X_PRED, X_OTHERS, X_CONF  (vote, others, conf).
                  Codes 1 = away team, 2 = home team (.qsf). Row "winner".
  Study 2 Elec. : blocks of 11 columns starting X_pred, X_conf, X_agree_1.
                  Row 2 is the question text. Codes 1..4 = the listed
                  options; 46 of 110 races offered 3 or 4 options. Row
                  "actual winner".

Both the original (2020) and the REVISED (July 2022) workbooks are present
(plan §2.1.1 OQ44). The loaders read either; nothing is dropped here.
Columns are always read by position. Whether the three header names of a
triple agree is recorded in meta (the "header" hypotheses of the build).
"""
import html
import json
import re
from functools import lru_cache
from pathlib import Path

import openpyxl

from cdm.config import data_root as _data_root
from cdm.data.worldA import common
from cdm.data.worldA.common import (DOMAIN_NBA, DOMAIN_NFL, DOMAIN_POLITICS,
                                    RawCase, RawResponse, clean_number,
                                    filter_responses, is_missing, make_case)

SOURCE = "Rutchick2020"
VERSIONS = ("original", "revised")

_FILES = {
    "nfl": ("Study 1", "NFL Data Raw (Study 1){}.xlsx"),
    "elections": ("Study 2", "Election Data Raw (Study 2){}.xlsx"),
    "nba": ("Study 3", "NBA Data Raw (Study 3){}.xlsx"),
}
_ELECTION_QSF = ("Study 2", ".qsf file", "Real_SP_Election.qsf")

#: How to read the confidence of the ORIGINAL NFL file, where 25 "_CONF"
#: columns are coded {1,4,5,6,7} instead of {1,2,3,4,5}.
#: The plan does not cover this; each value is a named alternative.
#:   recode_to_revised  : map 1,4,5,6,7 -> 1,2,3,4,5 in those columns (this
#:                        is what the REVISED file contains)
#:   domain_minmax_1_7  : treat the whole NFL domain as a 1..7 scale
#:   column_minmax_1_7  : treat only the 25 affected columns as 1..7
#:   clip               : normalize as 1..5 and cut at 1
#:   all_likert_minmax_1_7 : one 1..7 min-max over ALL Rutchick confidences
#:                        (NBA and Politics too)
#:   c_over_7           : c/7 for the whole NFL domain, whatever the Likert
#:                        formula of the other domains
#: The default everywhere is recode_to_revised: it equals the REVISED file
#: and is the only treatment that needs no reading of the odd codes. No
#: treatment is supported over another by the rule gate (NFL HAC 131 or 132
#: against 131; WC 117 against 114 under every treatment).
NFL_ORIGINAL_CONF = ("recode_to_revised", "domain_minmax_1_7",
                     "column_minmax_1_7", "clip", "all_likert_minmax_1_7",
                     "c_over_7")
_RECODE = {1: 1, 4: 2, 5: 3, 6: 4, 7: 5}


def extracted_dir(data_root=None):
    root = Path(data_root) if data_root is not None else _data_root()
    return root / "Rutchick2020_NBA_NFL_Politics" / "extracted"


def raw_file(study, version="revised", data_root=None):
    """Path of the Raw workbook of one study ("nfl", "elections", "nba")."""
    if version not in VERSIONS:
        raise ValueError(f"unknown Rutchick version {version!r}")
    folder, pattern = _FILES[study]
    suffix = " REVISED" if version == "revised" else ""
    return extracted_dir(data_root) / folder / pattern.format(suffix)


@lru_cache(maxsize=None)
def _grid(path):
    ws = openpyxl.load_workbook(common.require_file(path), read_only=True, data_only=True).worksheets[0]
    return [tuple(row) for row in ws.iter_rows(values_only=True)]


def _find_row(grid, label, n_cols=2):
    """Index of the row whose first cells hold `label` (case-insensitive)."""
    for i, row in enumerate(grid):
        for x in row[:n_cols]:
            if isinstance(x, str) and x.strip().lower() == label:
                return i
    raise ValueError(f"row {label!r} not found")


def _participant_rows(grid, first, id_col, stop):
    rows = []
    for i in range(first, stop):
        pid = clean_number(grid[i][id_col])
        if isinstance(pid, int):
            rows.append((pid, grid[i]))
    return rows


def _tokens(header, n_suffix_tokens=1):
    """Header name -> its "_"-separated tokens without the suffix; empty
    tokens (doubled underscores) are ignored."""
    parts = [t for t in str(header).split("_") if t]
    return tuple(parts[:-n_suffix_tokens]) if len(parts) > n_suffix_tokens else ()


# ---- Study 1: NFL ------------------------------------------------------------

def load_raw_nfl(version="revised", data_root=None):
    """216 NFL games (Rutchick S1) in file column order (weeks 1-15).

    meta keys:
      headers             : the three header names (PRED, CONF, OTHERS)
      header_tokens_match : True if (week, team1, team2) agree across the
                            three names, ignoring doubled underscores
      header_exact_match  : True if the three names share the exact prefix
      conf_out_of_scale   : True if the confidence column holds codes > 5
                            (25 columns of the original file)
    natural_first = 2: the SECOND-NAMED team of the header. It is usually
    but not always the home team: the file and the .qsf only say
    "first-named / second-named", and no schedule file is part of the data,
    so this is not checked (decision W7). So OQ5 (a) for NFL is
    "second-named team first", not a true home coding.
    """
    grid = _grid(str(raw_file("nfl", version, data_root)))
    hdr = grid[0]
    win_row = grid[_find_row(grid, "winner")]
    id_col = next(j for j, h in enumerate(hdr) if h == "Participant")
    stop = _find_row(grid, "fivethirtyeight")
    people = _participant_rows(grid, 1, id_col, stop)
    cases = []
    for j, h in enumerate(hdr):
        if not (isinstance(h, str) and h.endswith("_PRED")):
            continue
        h_conf, h_oth = str(hdr[j + 1]), str(hdr[j + 2])
        if not (h_conf.endswith("CONF") and h_oth.endswith("OTHERS")):
            raise ValueError(f"unexpected NFL column order at {h!r}")
        tok = _tokens(h)
        if len(tok) != 3:
            raise ValueError(f"cannot parse NFL header {h!r}")
        week, t1, t2 = tok
        responses = [RawResponse(pid, clean_number(row[j]),
                                 clean_number(row[j + 1]),
                                 clean_number(row[j + 2]))
                     for pid, row in people]
        conf_values = {r.conf for r in responses if r.conf is not None}
        cases.append(RawCase(
            case_id="_".join(tok), domain=DOMAIN_NFL, source=SOURCE + "_S1",
            options=(1, 2), labels={1: t1, 2: t2}, natural_first=2,
            winner=clean_number(win_row[j]), responses=responses,
            conf_kind="likert", others_kind="agree_with_me",
            meta={"version": version, "week": int(week[1:]),
                  "headers": (h, h_conf, h_oth),
                  "header_tokens_match": tok == _tokens(h_conf) == _tokens(h_oth),
                  "header_exact_match": (h[:-len("_PRED")] == h_conf[:-len("_CONF")]
                                         == h_oth[:-len("_OTHERS")]),
                  "conf_out_of_scale": bool(conf_values - {1, 2, 3, 4, 5})},
        ))
    return cases


def recode_nfl_original_confidence(cases):
    """Apply 1,4,5,6,7 -> 1,2,3,4,5 in the out-of-scale confidence columns
    of the original NFL file (in place). The result equals the REVISED
    file's confidence cells (tested in tests/test_a8_loaders.py)."""
    for raw in cases:
        if raw.meta.get("conf_out_of_scale"):
            for r in raw.responses:
                if r.conf is not None:
                    r.conf = _RECODE[r.conf]
            raw.meta["conf_out_of_scale"] = False
            raw.meta["conf_recoded"] = True
    return cases


# ---- Study 3: NBA ------------------------------------------------------------

def load_raw_nba(version="revised", data_root=None):
    """465 NBA games (Rutchick S3) in file column order (weeks 9, 8, ..., 2,
    12, 11, 10; within a week by a text sort of the game index).

    Codes: 1 = away team, 2 = home team (NBA_Week_1.qsf); natural_first = 2.
    The original file holds one vote coded 0 (participant 25, game W3_2);
    it is kept here and falls to the "vote is not an answer" filter.
    """
    grid = _grid(str(raw_file("nba", version, data_root)))
    hdr = grid[0]
    win_i = _find_row(grid, "winner")
    win_row = grid[win_i]
    people = _participant_rows(grid, 1, 0, win_i)
    cases = []
    for j, h in enumerate(hdr):
        if not (isinstance(h, str) and h.endswith("_PRED")):
            continue
        h_oth, h_conf = str(hdr[j + 1]), str(hdr[j + 2])
        if not (h_oth.endswith("OTHERS") and h_conf.endswith("CONF")):
            raise ValueError(f"unexpected NBA column order at {h!r}")
        tok = _tokens(h)
        responses = [RawResponse(pid, clean_number(row[j]),
                                 clean_number(row[j + 2]),
                                 clean_number(row[j + 1]))
                     for pid, row in people]
        cases.append(RawCase(
            case_id="_".join(tok), domain=DOMAIN_NBA, source=SOURCE + "_S3",
            options=(1, 2), labels={1: "away", 2: "home"}, natural_first=2,
            winner=clean_number(win_row[j]), responses=responses,
            conf_kind="likert", others_kind="agree_with_me",
            meta={"version": version, "week": int(tok[0][1:]),
                  "headers": (h, h_conf, h_oth),
                  "header_tokens_match": tok == _tokens(h_conf) == _tokens(h_oth),
                  "header_exact_match": (h[:-len("_PRED")] == h_conf[:-len("_CONF")]
                                         == h_oth[:-len("_OTHERS")])},
        ))
    return cases


# ---- Study 2: Elections ------------------------------------------------------

def _norm_text(t):
    t = html.unescape(re.sub(r"<[^>]+>", "", str(t))).replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip().lower()


@lru_cache(maxsize=None)
def _qsf_choices(path):
    """Question text (normalized) -> list of choice labels, from the .qsf."""
    try:
        with open(path, encoding="utf-8") as fh:
            q = json.load(fh)
    except (OSError, ValueError):
        return {}
    out = {}
    for e in q.get("SurveyElements", []):
        if e.get("Element") != "SQ":
            continue
        p = e.get("Payload") or {}
        if not str(p.get("DataExportTag", "")).endswith("_pred"):
            continue
        ch = p.get("Choices") or {}
        labels = [ch[k]["Display"] for k in sorted(ch, key=int)]
        key = _norm_text(p.get("QuestionText", ""))
        if key in out and out[key] != labels:
            out[key] = None            # ambiguous
        else:
            out[key] = labels
    return out


def load_raw_elections(version="revised", data_root=None):
    """110 election races (Rutchick S2) in file column order.

    options : the answer codes OFFERED in the survey (2, 3 or 4), from the
              .qsf matched by question text; if the .qsf cannot be matched,
              the codes observed in votes and winner (always at least 1, 2).
              meta["options_source"] says which. Which answer set a Case
              gets is a build switch (`elections_answer_set`):
              meta["observed_options"] holds the codes that occur in the
              workbook (votes and winner, always with 1 and 2), which is
              all that a loader working from the xlsx alone can see.
    The header name AK_gub is used for two races (Alaska and Arkansas
    governor); the second gets case_id "AK_gub#2".

    meta keys:
      headers             : (pred, conf, agree) header names
      name_match_strict   : conf == X_conf and agree == X_agree_1
      name_match_loose    : as strict, but X_agree is also accepted
                            (REVISED renamed Q58_1 to TX_sen_agree)
      pred_agree_match    : agree == X_agree_1 (the vote and the "others"
                            column share the race name; conf not checked)
      pred_conf_match     : conf == X_conf
      observed_options    : sorted codes found in votes and winner, + {1, 2}
      third_option_votes  : number of votes for a code other than 1 or 2
      n_options_offered, choice_labels, question
    natural_first = 1 (plan D1: "first-listed candidate").
    """
    grid = _grid(str(raw_file("elections", version, data_root)))
    hdr, qtext = grid[0], grid[1]
    win_i = _find_row(grid, "actual winner")
    win_row = grid[win_i]
    people = _participant_rows(grid, 2, 0, _find_row(grid, "fivethirtyeight"))
    qsf = _qsf_choices(str(extracted_dir(data_root).joinpath(*_ELECTION_QSF)))
    cases, seen = [], {}
    for j, h in enumerate(hdr):
        if not (isinstance(h, str) and h.endswith("_pred")):
            continue
        race = h[:-len("_pred")]
        h_conf, h_agree = str(hdr[j + 1]), str(hdr[j + 2])
        responses = [RawResponse(pid, clean_number(row[j]),
                                 clean_number(row[j + 1]),
                                 clean_number(row[j + 2]))
                     for pid, row in people]
        winner = clean_number(win_row[j])
        observed = {r.vote for r in responses if r.vote is not None} | {winner, 1, 2}
        choice_labels = qsf.get(_norm_text(qtext[j]))
        if choice_labels:
            options = tuple(range(1, len(choice_labels) + 1))
            source = "qsf"
            if not observed <= set(options):
                raise ValueError(f"{race}: observed codes {observed} exceed the "
                                 f".qsf choices {options}")
        else:
            options = tuple(sorted(observed))
            source = "observed"
        seen[race] = seen.get(race, 0) + 1
        case_id = race if seen[race] == 1 else f"{race}#{seen[race]}"
        cases.append(RawCase(
            case_id=case_id, domain=DOMAIN_POLITICS, source=SOURCE + "_S2",
            options=options, labels={c: f"option{c}" for c in options},
            natural_first=1, winner=winner, responses=responses,
            conf_kind="likert", others_kind="agree_with_me",
            meta={"version": version, "headers": (h, h_conf, h_agree),
                  "name_match_strict": (h_conf == race + "_conf"
                                        and h_agree == race + "_agree_1"),
                  "name_match_loose": (h_conf == race + "_conf" and h_agree in
                                       (race + "_agree_1", race + "_agree")),
                  "pred_agree_match": h_agree == race + "_agree_1",
                  "pred_conf_match": h_conf == race + "_conf",
                  "observed_options": tuple(sorted(observed)),
                  "third_option_votes": sum(
                      1 for r in responses
                      if r.vote is not None and r.vote not in (1, 2)),
                  "n_options_offered": len(options), "options_source": source,
                  "choice_labels": tuple(choice_labels) if choice_labels else None,
                  "question": None if is_missing(qtext[j]) else str(qtext[j])},
        ))
    return cases


# ---- Case-level convenience --------------------------------------------------

_LOADERS = {"nfl": load_raw_nfl, "nba": load_raw_nba,
            "elections": load_raw_elections}


def load_raw(study, version="revised", data_root=None):
    return _LOADERS[study](version, data_root)


def input_files(version="revised", data_root=None):
    return [raw_file(s, version, data_root) for s in ("nba", "nfl", "elections")]


def election_qsf_file(data_root=None):
    """Path of the Study 2 .qsf (the source of the offered answer options)."""
    return extracted_dir(data_root).joinpath(*_ELECTION_QSF)


def likert_settings(raw, likert_conf="a_c_minus_1_over_4",
                    nfl_original_conf="recode_to_revised"):
    """(likert formula, scale_max, clip) for one case; see NFL_ORIGINAL_CONF.

    Only the ORIGINAL Rutchick files are affected; with the REVISED files
    (and for every other source) the result is (likert_conf, 5, False).
    """
    if nfl_original_conf not in NFL_ORIGINAL_CONF:
        raise ValueError(f"unknown nfl_original_conf {nfl_original_conf!r}")
    if raw.meta.get("version") != "original":
        return likert_conf, 5, False
    if nfl_original_conf == "all_likert_minmax_1_7":
        return likert_conf, 7, False
    if raw.domain != DOMAIN_NFL:
        return likert_conf, 5, False
    if nfl_original_conf == "domain_minmax_1_7":
        return likert_conf, 7, False
    if nfl_original_conf == "column_minmax_1_7":
        return likert_conf, (7 if raw.meta.get("conf_out_of_scale") else 5), False
    if nfl_original_conf == "clip":
        return likert_conf, 5, True
    if nfl_original_conf == "c_over_7":
        return "b_c_over_5", 7, False
    return likert_conf, 5, False                      # recode_to_revised


def likert_scale_max(raw, nfl_original_conf="recode_to_revised"):
    """(scale_max, clip) of `likert_settings` (older two-value form)."""
    _, scale_max, clip = likert_settings(raw, "a_c_minus_1_over_4", nfl_original_conf)
    return scale_max, clip


ELECTIONS_ANSWER_SETS = ("observed_codes", "offered_qsf")


def load_cases(study, version="revised", data_root=None,
               likert_conf="a_c_minus_1_over_4",
               answer_coding="a_natural_source_coding", coding_seed=0,
               complete_only=True, nfl_original_conf="recode_to_revised",
               elections_answer_set="observed_codes",
               ps_other_answers="complement_each"):
    """One study as Case objects: complete responses only (Step D2), no
    case-level filter. Cases that cannot be a valid Case are skipped (none
    in these files: every game/race has responses and a winner among its
    answers).

    elections_answer_set : "observed_codes" (the codes that occur in the
        workbook) or "offered_qsf" (the options offered in the survey).
    ps_other_answers : common.PS_OTHER_ANSWERS (more than two answers only).
    """
    if elections_answer_set not in ELECTIONS_ANSWER_SETS:
        raise ValueError(f"unknown elections_answer_set {elections_answer_set!r}")
    raws = load_raw(study, version, data_root)
    if study == "nfl" and version == "original" and nfl_original_conf == "recode_to_revised":
        recode_nfl_original_confidence(raws)
    out = []
    for raw in raws:
        options = None
        if raw.domain == DOMAIN_POLITICS and elections_answer_set == "observed_codes":
            options = raw.meta["observed_options"]
        kept, _ = filter_responses(raw, complete_only=complete_only, options=options)
        formula, scale_max, clip = likert_settings(raw, likert_conf, nfl_original_conf)
        out.append(make_case(raw, kept, likert_conf=formula,
                             likert_scale_max=scale_max, likert_clip=clip,
                             answer_coding=answer_coding, coding_seed=coding_seed,
                             options=options, ps_other_answers=ps_other_answers))
    return out
