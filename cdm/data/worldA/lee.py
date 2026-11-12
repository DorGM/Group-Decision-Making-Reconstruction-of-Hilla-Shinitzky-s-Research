"""Loader for Lee, Danileiko & Vi 2018 (OSF 3kjmu), NFL 2017 season.

Step A8, plan §2.1.1 Step D1 and OQ1/OQ1b/D21. File layout as found
in the public files (docs/DATA.md):
  NFLSP_week1.csv ... NFLSP_week17.csv, header + 100 rows (one MTurk worker
  per row, new workers each week). Columns: gender, age, self-rated
  knowledge, then one triple per game (the question text is the header):
    "Who do you think will win the <date> game between the A and the B?"
        -> a full team-name string
    "How confident are you that your selection is correct?"
        -> Guess / Low / Moderate / High / Very High Confidence
    "What percentage of other people do you think will agree with your
     selection? - Percentage agreeing" -> integer 0-100

THE FOLDER HOLDS NO GAME OUTCOMES. This loader therefore returns responses
only (RawCase.winner is None). `load_cases` needs an outcomes mapping
supplied by the caller and raises `LeeOutcomesMissing` without one. No
outcome is invented here.

257 game columns: the pair Tampa Bay Buccaneers - Miami Dolphins appears in
week 1 and again in week 11. Both are loaded; the week-1 column is flagged
meta["same_pair_appears_later"] = True (256 distinct games).
"""
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from cdm.config import data_root as _data_root
from cdm.data.worldA import common
from cdm.data.worldA.common import (DOMAIN_NFL, RawCase, RawResponse,
                                    clean_number, filter_responses, is_missing,
                                    make_case)

SOURCE = "Lee2018"
N_WEEKS = 17
CONFIDENCE_CODES = {
    "Guess": 1, "Low Confidence": 2, "Moderate Confidence": 3,
    "High Confidence": 4, "Very High Confidence": 5,
}
EXPERT_LABEL = "Extremely knowledgeable"
_VOTE = "Who do you think will win"
_HEADER = re.compile(r"win the (.*?) game between the (.*?) and (?:the )?(.*?)\??$")


class LeeOutcomesMissing(RuntimeError):
    """The Lee 2018 folder has no game results; Cases cannot be built."""


def extracted_dir(data_root=None):
    root = Path(data_root) if data_root is not None else _data_root()
    return root / "Lee2018_NFL" / "extracted"


def input_files(data_root=None):
    d = extracted_dir(data_root)
    return [d / f"NFLSP_week{w}.csv" for w in range(1, N_WEEKS + 1)]


@lru_cache(maxsize=None)
def _read(path):
    return pd.read_csv(common.require_file(path))


def load_raw(data_root=None, expert_only=False):
    """All 257 game columns as RawCase objects (weeks 1-17, column order).

    Codes: 1 = first-named team, 2 = second-named team. natural_first = 2,
    on the same reading as Rutchick NFL (second-named = home team); the
    files do not say which team is at home.
    expert_only : keep only "Extremely knowledgeable" workers (the paper's
                  headline subset; plan §8 OQ4, default off).
    Each response's `extra` is the worker's self-rated knowledge string.
    """
    cases = []
    for week, path in enumerate(input_files(data_root), start=1):
        df = _read(str(path))
        cols = list(df.columns)
        know = df.iloc[:, 2].tolist()
        g = 0
        for j, col in enumerate(cols):
            if not col.startswith(_VOTE):
                continue
            g += 1
            m = _HEADER.search(col.strip())
            if not m:
                raise ValueError(f"week {week}: cannot parse header {col!r}")
            date, t1, t2 = (s.strip() for s in m.groups())
            if not (cols[j + 1].startswith("How confident")
                    and cols[j + 2].startswith("What percentage")):
                raise ValueError(f"week {week}: unexpected column order at {col!r}")
            codes = {t1: 1, t2: 2}
            responses = []
            for i in range(len(df)):
                if expert_only and know[i] != EXPERT_LABEL:
                    continue
                v, c, o = df.iat[i, j], df.iat[i, j + 1], df.iat[i, j + 2]
                responses.append(RawResponse(
                    participant=f"W{week}:{i + 1}",
                    vote=None if is_missing(v) else codes.get(str(v).strip(), str(v).strip()),
                    conf=None if is_missing(c) else CONFIDENCE_CODES[str(c).strip()],
                    others=clean_number(o),
                    extra=None if is_missing(know[i]) else str(know[i])))
            cases.append(RawCase(
                case_id=f"W{week}_{g}", domain=DOMAIN_NFL, source=SOURCE,
                options=(1, 2), labels={1: t1, 2: t2}, natural_first=2,
                winner=None, responses=responses,
                conf_kind="likert", others_kind="agree_with_me",
                meta={"week": week, "date": date, "teams": (t1, t2),
                      "header": col, "same_pair_appears_later": False},
            ))
    last = {}
    for k, raw in enumerate(cases):
        last[raw.meta["teams"]] = k          # ordered pair (first, second)
    for k, raw in enumerate(cases):
        raw.meta["same_pair_appears_later"] = last[raw.meta["teams"]] != k
    return cases


def load_cases(data_root=None, outcomes=None, expert_only=False,
               likert_conf="a_c_minus_1_over_4",
               answer_coding="a_natural_source_coding", coding_seed=0):
    """Lee games as Case objects. Needs `outcomes`: {case_id: winning team
    name}, obtained outside the folder (the folder has none). Games missing
    from `outcomes` are skipped."""
    if not outcomes:
        raise LeeOutcomesMissing(
            "The Lee 2018 folder holds no game outcomes (docs/DATA.md). "
            "Pass outcomes={case_id: winning team name} from an external "
            "results table for the 2017 NFL season.")
    out = []
    for raw in load_raw(data_root, expert_only):
        team = outcomes.get(raw.case_id)
        if team is None:
            continue
        code = {v: k for k, v in raw.labels.items()}.get(team)
        if code is None:
            raise ValueError(f"{raw.case_id}: {team!r} is not one of {raw.meta['teams']}")
        raw.winner = code
        kept, _ = filter_responses(raw)
        out.append(make_case(raw, kept, likert_conf=likert_conf,
                             answer_coding=answer_coding, coding_seed=coding_seed))
    return out
