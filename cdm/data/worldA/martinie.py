"""Loader for Martinie, Wilkening & Howe 2020 (PLOS ONE) -> "General" domain.

Step A8, plan §2.1.1 Step D1. File layout as found in the
public files (docs/DATA.md):
  Raws/r2a..r2e_rawdata.xlsx : no header; one row per worker; 100 questions
      x 4 columns: (1) vote, 1 = False, 2 = True; (2) predicted % of others
      voting True; (3) own probability that the statement is true;
      (4) predicted average probability of others (not used).
  File a holds question IDs 1-100, b 101-200, ..., e 401-500.
  Raws/r2_outcomes.xlsx : no header; ID, outcome (1 true / 0 false),
      difficulty, statement text.
All percentages are whole numbers 0-100. There is no confidence question:
confidence is derived from column 3 (plan §8 OQ2).

The 41 self-reported cheaters are already absent from the files and cannot
be identified, so the plan's "drop cheaters" flag (OQ4) has nothing to act
on.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

from cdm.config import data_root as _data_root
from cdm.data.worldA import common
from cdm.data.worldA.common import (DOMAIN_GENERAL, RawCase, RawResponse,
                                    clean_number, filter_responses, make_case)

SOURCE = "Martinie2020"
FILES = ("r2a", "r2b", "r2c", "r2d", "r2e")
TRUE_CODE, FALSE_CODE = 2, 1


def raws_dir(data_root=None):
    root = Path(data_root) if data_root is not None else _data_root()
    return root / "Martinie2020_General" / "extracted" / "AnalysisCode" / "Raws"


def input_files(data_root=None):
    d = raws_dir(data_root)
    return [d / f"{f}_rawdata.xlsx" for f in FILES] + [d / "r2_outcomes.xlsx"]


@lru_cache(maxsize=None)
def _read(path):
    return pd.read_excel(common.require_file(path), header=None).values


def load_raw(data_root=None):
    """All 500 statements as RawCase objects, in source order: file a..e,
    then column-block order = question ID 1..500 (plan §8 OQ16, Step D6).

    Raw answer codes: 1 = False, 2 = True. `conf` holds the probability
    (0-100) that the statement is true; `others` the predicted percentage
    of others voting True.
    """
    d = raws_dir(data_root)
    outcomes = _read(str(d / "r2_outcomes.xlsx"))
    if outcomes.shape[0] != 500:
        raise ValueError(f"r2_outcomes has {outcomes.shape[0]} rows, expected 500")
    cases = []
    qid = 0
    for fi, f in enumerate(FILES):
        grid = _read(str(d / f"{f}_rawdata.xlsx"))
        if grid.shape[1] != 400:
            raise ValueError(f"{f}: {grid.shape[1]} columns, expected 400")
        for q in range(100):
            qid += 1
            row = outcomes[qid - 1]
            if int(row[0]) != qid:
                raise ValueError(f"r2_outcomes row {qid} has ID {row[0]}")
            responses = [
                RawResponse(participant=f"{f}:{w + 1}",
                            vote=clean_number(grid[w, 4 * q]),
                            conf=clean_number(grid[w, 4 * q + 2]),
                            others=clean_number(grid[w, 4 * q + 1]))
                for w in range(grid.shape[0])
            ]
            cases.append(RawCase(
                case_id=str(qid), domain=DOMAIN_GENERAL, source=SOURCE,
                options=(FALSE_CODE, TRUE_CODE),
                labels={TRUE_CODE: "True", FALSE_CODE: "False"},
                natural_first=TRUE_CODE,
                winner=TRUE_CODE if int(row[1]) == 1 else FALSE_CODE,
                responses=responses,
                conf_kind="probability_true", others_kind="share_true",
                meta={"true_code": TRUE_CODE, "file": f, "file_index": fi,
                      "question_id": qid, "difficulty": int(row[2]),
                      "statement": str(row[3])},
            ))
    return cases


def load_cases(data_root=None, confidence="a_abs_p_minus_50_over_50",
               drop_vote_inconsistent=True,
               answer_coding="a_natural_source_coding", coding_seed=0):
    """The 500 Martinie items as Case objects (no case-level filter).

    Defaults are the plan's (OQ2 option a; OQ4 "apply Martinie's
    exclusions"). The combination that reproduces TH Table A.2 on item 1 is
    confidence="c_p_if_true_else_100_minus_p", drop_vote_inconsistent=False.
    """
    out = []
    for raw in load_raw(data_root):
        kept, _ = filter_responses(
            raw, martinie_drop_vote_inconsistent=drop_vote_inconsistent)
        out.append(make_case(raw, kept, martinie_conf=confidence,
                             answer_coding=answer_coding,
                             coding_seed=coding_seed))
    return out
