"""World-B loader: the per-response table of the 22 multiple-choice tasks
(plan §2.2, §9.3 row B4) -> one validated Case per task.

    *********************************************************************
    *  UNTESTED ON REAL DATA.  The real MTurk file has not been supplied *
    *  (plan §2.2: "not public"; §10 item 2).  Its format is unknown.    *
    *  Everything below about the file layout is an ASSUMPTION, listed   *
    *  in docs/IMPLEMENTATION_DECISIONS.md (section "World B", entries L1-L12).  The loader *
    *  is exercised only on files written by this project: a tiny        *
    *  fabricated fixture and the synthetic stand-in of plan §2.3.       *
    *********************************************************************

The schema is the response of plan §3.1 plus `task_id` (plan §2.3: "exactly
the loader schema of §3.1 plus task_id"), which is what plan §2.2 lists as
the needed input: "task id, response id; vote, confidence (raw scale), predicted %
per option; the correct answer per task; ideally the original 77-subset
assignment".

Assumed layout: two CSV files (UTF-8, comma separated, one header row). A
UTF-8 byte-order mark at the start of a file (usual in Excel exports) is
tolerated and ignored.

  tasks file      one row per task
      task_id     identifier, unique
      name        task name; used to match TH Table 4.1 (optional column)
      type        task type as in Table 4.1 (optional column)
      answers     the options of the task in a fixed order, separated by "|"
      correct     the correct option (one of `answers`)

  responses file  one row per response
      task_id     a task of the tasks file
      response_id identifier, unique within the task
      vote        the chosen option (one of the task's `answers`)
      confidence  number on the raw scale `confidence_range` = (low, high)
      ps_1..ps_K  predicted support of the task's options, in the order of
                  `answers`; K = the largest number of options of any task;
                  the cells beyond a task's number of options are empty
      subset_id   optional: the original sub-set of the response (plan
                  §2.2 "ideally the original 77-subset assignment")

A first line that starts with "#" is a comment. Files written for the
synthetic stand-in carry the marker "SYNTHETIC — not a reproduction" there
(and in every task_id), and the loaded object then has `synthetic = True`.

Transformations (each is an argument, none has a silent default for the
real file):
  confidence_range : (low, high) of the raw scale; c = (raw - low) /
                     (high - low). The scale of the MTurk form is not
                     reported (TH p.53 gives only the question text), so the
                     caller must state it. The rule mirrors the min-max
                     reading of World A (plan §8 OQ2); it is an assumption.
  ps_scale         : 100 if the predictions are percentages, 1 if shares.
                     A single prediction above 1 after division by
                     `ps_scale` (beyond `ps_sum_tol`) is an error under
                     every policy: one option cannot have more than 100%,
                     so the stated scale is wrong. It is never repaired
                     by renormalising (decision L7).
  ps_sum_policy    : plan §8 OQ39. "renormalize_if_violated" (default):
                     a vector whose total differs from 1 by more than
                     `ps_sum_tol` is divided by its total. "raw": used as
                     given. Violations are counted either way.
  incomplete       : "raise" (default) or "drop": a response with an empty
                     vote, confidence or prediction. The plan gives no rule
                     for World B ("drop" is the World-A choice of OQ4).

Checks (plan §9.3 row B4): `table_4_1_report` compares |A_P|, |R_P| and
S(a*) of every task with TH Table 4.1 and logs whether the file has 2490 or
2514 rows (OQ34). It reports; it does not raise.
"""
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from cdm.data.worldB import table41 as T41
from cdm.schema import Case, Response, ValidationError

#: Carried by the data object and by every file of the synthetic stand-in
#: (plan §2.3: "Synthetic results are never reported as reproductions";
#: §9.3 row B11).
SYNTHETIC_MARKER = "SYNTHETIC — not a reproduction"
#: Status of this module, written into every loaded object's log.
REAL_DATA_STATUS = "UNTESTED ON REAL DATA"

RESPONSES_FILE = "worldB_responses.csv"
TASKS_FILE = "worldB_tasks.csv"
MANIFEST_FILE = "worldB_manifest.json"

ANSWER_SEPARATOR = "|"
PS_SUM_POLICIES = ("renormalize_if_violated", "raw")
INCOMPLETE_POLICIES = ("raise", "drop")
#: A predicted-support vector whose total differs from 1 by more than this
#: (after division by `ps_scale`) is a violation of OQ39. [ASSUMPTION]
DEFAULT_PS_SUM_TOL = 1e-6


class WorldBFormatError(ValueError):
    """The file does not have the assumed layout."""


@dataclass(frozen=True)
class WorldBData:
    """The World-B tasks, one validated Case per task (all its responses).

    tasks        : Case per task, in the order of the tasks file;
                   case_id = task_id, answers in the file's order.
    names, types : per task (may be None).
    response_ids : per task, the ids of its responses in the order of
                   Case.responses.
    original_subsets : per task, the `subset_id` of every response, or None
                   if the file has no such column.
    synthetic    : True for the stand-in of plan §2.3.
    marker       : SYNTHETIC_MARKER for synthetic data, "" otherwise.
    log          : counts and options of the load (rows, dropped responses,
                   OQ39 violations, confidence range ...).
    """
    tasks: Tuple[Case, ...]
    names: Tuple[Optional[str], ...]
    types: Tuple[Optional[str], ...]
    response_ids: Tuple[Tuple[str, ...], ...]
    original_subsets: Optional[Tuple[Tuple[str, ...], ...]]
    synthetic: bool
    marker: str
    log: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        n = len(self.tasks)
        if not (len(self.names) == len(self.types) == len(self.response_ids) == n):
            raise ValueError("names, types and response_ids need one entry per task")
        if self.original_subsets is not None and len(self.original_subsets) != n:
            raise ValueError("original_subsets needs one entry per task")
        for case, ids in zip(self.tasks, self.response_ids):
            if len(ids) != case.n or len(set(ids)) != len(ids):
                raise ValueError(f"task {case.case_id!r}: response ids must be "
                                 "unique, one per response")
        if self.synthetic != (self.marker == SYNTHETIC_MARKER):
            raise ValueError("synthetic data must carry SYNTHETIC_MARKER, "
                             "and only synthetic data may")
        ids = [c.case_id for c in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task ids must be unique")

    @property
    def n_responses(self):
        return sum(c.n for c in self.tasks)


# ---- reading ---------------------------------------------------------------------

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_csv(path):
    """(comment lines, header, rows as dicts). Leading "#" lines are
    comments; a "#" line anywhere else is an error. A UTF-8 byte-order
    mark at the start of the file is dropped ("utf-8-sig")."""
    path = Path(path)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        lines = fh.read().splitlines()
    comments = []
    while lines and lines[0].startswith("#"):
        comments.append(lines.pop(0))
    if not lines:
        raise WorldBFormatError(f"{path.name}: no header row")
    reader = csv.reader(lines)
    header = [h.strip() for h in next(reader)]
    if len(set(header)) != len(header):
        raise WorldBFormatError(f"{path.name}: duplicate column names")
    rows = []
    for lineno, cells in enumerate(reader, start=len(comments) + 2):
        if not cells or all(c.strip() == "" for c in cells):
            continue
        if cells[0].startswith("#"):
            raise WorldBFormatError(
                f"{path.name}, line {lineno}: comment line after the header")
        if len(cells) != len(header):
            raise WorldBFormatError(
                f"{path.name}, line {lineno}: {len(cells)} cells, "
                f"{len(header)} columns")
        row = {h: c.strip() for h, c in zip(header, cells)}
        row["_line"] = lineno
        rows.append(row)
    return comments, header, rows


def _number(text, what, where):
    try:
        x = float(text)
    except ValueError:
        raise WorldBFormatError(f"{where}: {what} {text!r} is not a number") from None
    if not math.isfinite(x):
        raise WorldBFormatError(f"{where}: {what} {text!r} is not finite")
    return x


def _check_options(confidence_range, ps_scale, ps_sum_policy, ps_sum_tol, incomplete):
    if confidence_range is None:
        raise ValueError(
            "confidence_range=(low, high) must be given: the raw confidence "
            "scale of the MTurk form is not reported (TH p.53; plan §2.2)")
    try:
        low, high = (float(x) for x in confidence_range)
    except (TypeError, ValueError):
        raise ValueError(f"confidence_range must be (low, high), got "
                         f"{confidence_range!r}") from None
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        raise ValueError(f"confidence_range must be finite with high > low, got "
                         f"{confidence_range!r}")
    if ps_scale is None or isinstance(ps_scale, bool) or ps_scale not in (1, 100):
        raise ValueError("ps_scale must be given: 100 (percentages) or 1 (shares)")
    if ps_sum_policy not in PS_SUM_POLICIES:
        raise ValueError(f"ps_sum_policy must be one of {PS_SUM_POLICIES}")
    if incomplete not in INCOMPLETE_POLICIES:
        raise ValueError(f"incomplete must be one of {INCOMPLETE_POLICIES}")
    if isinstance(ps_sum_tol, bool) or not (float(ps_sum_tol) >= 0):
        raise ValueError("ps_sum_tol must be a number >= 0")
    return low, high


def load_world_b(responses_path, tasks_path, *, confidence_range, ps_scale,
                 ps_sum_policy="renormalize_if_violated",
                 ps_sum_tol=DEFAULT_PS_SUM_TOL, incomplete="raise"):
    """Read the two CSV files of the assumed layout (module docstring).

    UNTESTED ON REAL DATA. Returns a WorldBData; `log` holds the counts
    that plan §9.3 row B4 asks to log.
    """
    low, high = _check_options(confidence_range, ps_scale, ps_sum_policy,
                               ps_sum_tol, incomplete)
    responses_path, tasks_path = Path(responses_path), Path(tasks_path)
    t_comments, t_header, t_rows = _read_csv(tasks_path)
    r_comments, r_header, r_rows = _read_csv(responses_path)

    # ---- synthetic marker: both files or neither
    marks = [any(SYNTHETIC_MARKER in c for c in cs) for cs in (t_comments, r_comments)]
    if marks[0] != marks[1]:
        raise WorldBFormatError(
            "one file carries the synthetic marker and the other does not")
    synthetic = marks[0]

    # ---- tasks
    for col in ("task_id", "answers", "correct"):
        if col not in t_header:
            raise WorldBFormatError(f"{tasks_path.name}: missing column {col!r}")
    task_order, task_info = [], {}
    for row in t_rows:
        where = f"{tasks_path.name}, line {row['_line']}"
        tid = row["task_id"]
        if tid == "":
            raise WorldBFormatError(f"{where}: empty task_id")
        if tid in task_info:
            raise WorldBFormatError(f"{where}: duplicate task_id {tid!r}")
        answers = tuple(a.strip() for a in row["answers"].split(ANSWER_SEPARATOR))
        if len(answers) < 2 or any(a == "" for a in answers) \
                or len(set(answers)) != len(answers):
            raise WorldBFormatError(
                f"{where}: 'answers' must be at least two distinct options "
                f"separated by {ANSWER_SEPARATOR!r}")
        if row["correct"] not in answers:
            raise WorldBFormatError(
                f"{where}: correct answer {row['correct']!r} is not an option")
        task_order.append(tid)
        task_info[tid] = {"answers": answers, "correct": row["correct"],
                          "name": row.get("name") or None,
                          "type": row.get("type") or None}
    if not task_order:
        raise WorldBFormatError(f"{tasks_path.name}: no task")
    if not synthetic and any(SYNTHETIC_MARKER.split()[0] in t.upper()
                             for t in task_order):
        raise WorldBFormatError(
            "a task_id says SYNTHETIC but the files carry no synthetic marker")
    k_max = max(len(task_info[t]["answers"]) for t in task_order)

    # ---- responses
    ps_cols = [f"ps_{i}" for i in range(1, k_max + 1)]
    for col in ["task_id", "response_id", "vote", "confidence"] + ps_cols:
        if col not in r_header:
            raise WorldBFormatError(f"{responses_path.name}: missing column {col!r}")
    extra_ps = [h for h in r_header if h.startswith("ps_") and h not in ps_cols]
    if extra_ps:
        raise WorldBFormatError(
            f"{responses_path.name}: prediction columns {extra_ps} beyond the "
            f"largest number of options ({k_max})")
    has_subsets = "subset_id" in r_header
    per_task = {t: {"responses": [], "ids": [], "subsets": []} for t in task_order}
    n_rows = len(r_rows)
    dropped, violations, max_dev = 0, 0, 0.0
    for row in r_rows:
        where = f"{responses_path.name}, line {row['_line']}"
        tid = row["task_id"]
        if tid not in task_info:
            raise WorldBFormatError(f"{where}: unknown task_id {tid!r}")
        info = task_info[tid]
        m = len(info["answers"])
        cells = [row[c] for c in ps_cols]
        if any(c != "" for c in cells[m:]):
            raise WorldBFormatError(
                f"{where}: prediction given beyond the task's {m} options")
        if row["response_id"] == "":
            raise WorldBFormatError(f"{where}: empty response_id")
        if row["vote"] == "" or row["confidence"] == "" or any(c == "" for c in cells[:m]):
            if incomplete == "raise":
                raise WorldBFormatError(
                    f"{where}: incomplete response (empty vote, confidence or "
                    "prediction); the plan gives no rule for World B -- pass "
                    "incomplete='drop' to leave such responses out")
            dropped += 1
            continue
        if row["vote"] not in info["answers"]:
            raise WorldBFormatError(
                f"{where}: vote {row['vote']!r} is not an option of task {tid!r}")
        raw_c = _number(row["confidence"], "confidence", where)
        if not low <= raw_c <= high:
            raise WorldBFormatError(
                f"{where}: confidence {raw_c!r} is outside the stated raw "
                f"scale [{low}, {high}]")
        c = (raw_c - low) / (high - low)
        ps = [_number(x, "prediction", where) / ps_scale for x in cells[:m]]
        if any(p < 0 for p in ps):
            raise WorldBFormatError(f"{where}: negative prediction")
        # a wrong ps_scale must not be repaired by the OQ39 renormalisation
        if any(p > 1.0 + ps_sum_tol for p in ps):
            raise WorldBFormatError(
                f"{where}: prediction {max(ps)!r} is above 1 after division "
                f"by ps_scale={ps_scale}; a single option cannot have more "
                "than 100%. Is ps_scale wrong (100 = percentages, 1 = "
                "shares)? Nothing is renormalised in this case")
        total = math.fsum(ps)
        dev = abs(total - 1.0)
        if dev > ps_sum_tol:
            violations += 1
            max_dev = max(max_dev, dev)
            if ps_sum_policy == "renormalize_if_violated":
                if total == 0:
                    raise WorldBFormatError(
                        f"{where}: all predictions are 0; OQ39 "
                        "(renormalize) is undefined for it")
                ps = [p / total for p in ps]
        try:
            response = Response(v=row["vote"], c=c, ps=tuple(ps))
        except ValidationError as err:
            raise WorldBFormatError(f"{where}: {err}") from None
        per_task[tid]["responses"].append(response)
        per_task[tid]["ids"].append(row["response_id"])
        per_task[tid]["subsets"].append(row["subset_id"] if has_subsets else None)

    if has_subsets:
        blank = sum(1 for t in task_order for s in per_task[t]["subsets"] if s == "")
        if blank:
            raise WorldBFormatError(
                f"{responses_path.name}: {blank} responses have an empty subset_id")
    cases = []
    for tid in task_order:
        info = task_info[tid]
        if not per_task[tid]["responses"]:
            raise WorldBFormatError(f"task {tid!r} has no response")
        try:
            cases.append(Case(case_id=tid, answers=info["answers"],
                              correct=info["correct"],
                              responses=tuple(per_task[tid]["responses"]),
                              domain=SYNTHETIC_MARKER if synthetic else None))
        except ValidationError as err:
            raise WorldBFormatError(f"task {tid!r}: {err}") from None
    n_kept = sum(c.n for c in cases)
    log = {
        "status": REAL_DATA_STATUS if not synthetic else SYNTHETIC_MARKER,
        "responses_file": str(responses_path), "tasks_file": str(tasks_path),
        "sha256": {"responses": sha256_of(responses_path),
                   "tasks": sha256_of(tasks_path)},
        "n_rows_in_file": n_rows, "n_responses": n_kept, "n_tasks": len(cases),
        "incomplete_policy": incomplete, "incomplete_dropped": dropped,
        "confidence_range": [low, high], "ps_scale": ps_scale,
        "ps_sum_policy": ps_sum_policy, "ps_sum_tol": float(ps_sum_tol),
        "ps_sum_violations": violations, "ps_sum_max_deviation": max_dev,
        "has_original_subsets": has_subsets,
        "oq34": oq34_note(n_kept),
    }
    return WorldBData(
        tasks=tuple(cases),
        names=tuple(task_info[t]["name"] for t in task_order),
        types=tuple(task_info[t]["type"] for t in task_order),
        response_ids=tuple(tuple(per_task[t]["ids"]) for t in task_order),
        original_subsets=(tuple(tuple(per_task[t]["subsets"]) for t in task_order)
                          if has_subsets else None),
        synthetic=synthetic,
        marker=SYNTHETIC_MARKER if synthetic else "",
        log=log)


def oq34_note(n_responses):
    """Plan §8 OQ34 / D9: which of the two published totals the data has."""
    if n_responses == T41.N_RESPONSES_TABLE:
        return "2490 responses = the sum of TH Table 4.1 (and MCF-arXiv)"
    if n_responses == T41.N_RESPONSES_TEXT:
        return "2514 responses = the total stated in TH p.53 and MCF"
    return (f"{n_responses} responses: neither 2490 (TH Table 4.1) nor 2514 "
            "(TH p.53, MCF)")


# ---- checks against TH Table 4.1 (plan §9.3 row B4) ------------------------------------

def task_stats(case):
    """|A_P|, |R_P|, number of solvers, S(a*) and the number of sub-sets
    k = floor(|R_P| / 30) of one task."""
    solvers = sum(1 for r in case.responses if r.v == case.correct)
    return {"n_answers": len(case.answers), "n_responses": case.n,
            "n_solvers": solvers, "s_correct": solvers / case.n,
            "n_subsets": T41.n_subsets(case.n)}


def table_4_1_report(data):
    """Compare every task with its row of TH Table 4.1 (matched by name).

    Returns a dict:
      rows   : per task: the file's statistics, the printed row (or None if
               the name is not in the table) and the three match flags
               (|A_P|, |R_P|, S(a*) rounded half up to 2 d.p.)
      unmatched_tasks / missing_table_rows : names without a partner
      n_tasks, n_responses, n_subsets (sum of floor(|R_P| / 30)), oq34
      all_match : True if the 22 rows match in all three columns and the
               totals are 22 tasks and 77 sub-sets
    Nothing is raised: for the real file a mismatch is a finding to report.
    """
    rows, unmatched = [], []
    seen = set()
    for case, name in zip(data.tasks, data.names):
        st = task_stats(case)
        printed = T41.BY_NAME.get(name)
        row = {"task_id": case.case_id, "name": name, **st, "printed": None,
               "match_n_answers": False, "match_n_responses": False,
               "match_s_correct": False}
        if printed is None or name in seen:
            unmatched.append(name)
        else:
            seen.add(name)
            row["printed"] = printed._asdict()
            row["match_n_answers"] = st["n_answers"] == printed.n_answers
            row["match_n_responses"] = st["n_responses"] == printed.n_responses
            row["match_s_correct"] = (
                T41.printed_share(st["n_solvers"], st["n_responses"])
                == printed.s_correct)
        rows.append(row)
    missing = [t.name for t in T41.TABLE_4_1 if t.name not in seen]
    n_subsets = sum(r["n_subsets"] for r in rows)
    all_match = (not unmatched and not missing
                 and all(r["match_n_answers"] and r["match_n_responses"]
                         and r["match_s_correct"] for r in rows)
                 and len(rows) == T41.N_TASKS and n_subsets == T41.N_SUBSETS)
    return {"rows": rows, "unmatched_tasks": unmatched,
            "missing_table_rows": missing, "n_tasks": len(rows),
            "n_responses": data.n_responses, "n_subsets": n_subsets,
            "oq34": oq34_note(data.n_responses), "all_match": all_match}


# ---- writing (the same layout; used by the synthetic stand-in and the tests) -----------

def _comment(data):
    return ("# " + SYNTHETIC_MARKER + " (stand-in of plan §2.3; invented "
            "responses, NOT the MTurk data of the thesis)")


def write_world_b(data, out_dir, manifest_extra=None):
    """Write `data` in the assumed layout: confidences in [0, 1] and
    predictions as shares, with full float precision (the round trip through
    `load_world_b(..., confidence_range=(0, 1), ps_scale=1)` is exact).

    Synthetic data gets the marker as the first line of both CSV files, and
    as the first key of the manifest. Returns the three paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_max = max(len(c.answers) for c in data.tasks)
    for case in data.tasks:
        for a in case.answers:
            if ANSWER_SEPARATOR in str(a) or str(a) != str(a).strip() or str(a) == "":
                raise ValueError(f"answer label {a!r} cannot be written")
    tasks_path = out_dir / TASKS_FILE
    with open(tasks_path, "w", encoding="utf-8", newline="") as fh:
        if data.synthetic:
            fh.write(_comment(data) + "\n")
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["task_id", "name", "type", "answers", "correct"])
        for case, name, typ in zip(data.tasks, data.names, data.types):
            w.writerow([case.case_id, name or "", typ or "",
                        ANSWER_SEPARATOR.join(str(a) for a in case.answers),
                        case.correct])
    responses_path = out_dir / RESPONSES_FILE
    with open(responses_path, "w", encoding="utf-8", newline="") as fh:
        if data.synthetic:
            fh.write(_comment(data) + "\n")
        w = csv.writer(fh, lineterminator="\n")
        header = ["task_id", "response_id", "vote", "confidence"] \
            + [f"ps_{i}" for i in range(1, k_max + 1)]
        if data.original_subsets is not None:
            header.append("subset_id")
        w.writerow(header)
        for t, case in enumerate(data.tasks):
            pad = [""] * (k_max - len(case.answers))
            for i, r in enumerate(case.responses):
                row = [case.case_id, data.response_ids[t][i], r.v, repr(r.c)] \
                    + [repr(p) for p in r.ps] + pad
                if data.original_subsets is not None:
                    row.append(data.original_subsets[t][i])
                w.writerow(row)
    manifest = {}
    if data.synthetic:
        manifest["_marker"] = SYNTHETIC_MARKER
    manifest.update({
        "layout": "cdm.data.worldB.loader (assumed layout; " + REAL_DATA_STATUS + ")",
        "synthetic": data.synthetic,
        "n_tasks": len(data.tasks), "n_responses": data.n_responses,
        "confidence_range": [0, 1], "ps_scale": 1,
        "files": {TASKS_FILE: sha256_of(tasks_path),
                  RESPONSES_FILE: sha256_of(responses_path)},
    })
    manifest.update(manifest_extra or {})
    manifest_path = out_dir / MANIFEST_FILE
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return {"tasks": tasks_path, "responses": responses_path,
            "manifest": manifest_path}


def load_written(folder, verify=True):
    """Read a folder written by `write_world_b` (shares, confidences in
    [0, 1]); with verify, the SHA-256 of both CSV files must equal the
    manifest's."""
    folder = Path(folder)
    with open(folder / MANIFEST_FILE, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if verify:
        for name in (TASKS_FILE, RESPONSES_FILE):
            if sha256_of(folder / name) != manifest["files"][name]:
                raise WorldBFormatError(f"{name}: hash differs from the manifest")
    data = load_world_b(folder / RESPONSES_FILE, folder / TASKS_FILE,
                        confidence_range=tuple(manifest["confidence_range"]),
                        ps_scale=manifest["ps_scale"], ps_sum_policy="raw")
    if data.synthetic != bool(manifest.get("synthetic")):
        raise WorldBFormatError("the manifest and the CSV files disagree on "
                                "whether the data is synthetic")
    return data


# ---- config switch (plan §2.3) ----------------------------------------------------------

SOURCES = ("synthetic", "real")


def load_from_config(path=None, defaults=None):
    """The World-B data named by config/worldB.yaml (plan §2.3: the real
    file and the synthetic stand-in are "chosen by a config switch").

    source: synthetic -> cdm.data.worldB.synthetic.generate()
    source: real      -> load_world_b on the files and options of the
                         `real` block (UNTESTED ON REAL DATA); the OQ39
                         policy is the one of config/defaults.yaml.
    """
    import yaml
    from cdm.config import CODE_ROOT, get_switch, load_config
    path = Path(path) if path else CODE_ROOT / "config" / "worldB.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    source = cfg.get("source")
    if source not in SOURCES:
        raise ValueError(f"{path.name}: source must be one of {SOURCES}, got {source!r}")
    if source == "synthetic":
        from cdm.data.worldB import synthetic
        return synthetic.generate()
    real = cfg.get("real") or {}
    for key in ("responses_file", "tasks_file", "confidence_range", "ps_scale"):
        if real.get(key) is None:
            raise ValueError(
                f"{path.name}: real.{key} is not set. The real MTurk file has "
                "not been supplied; its location, confidence scale and "
                "prediction scale must be stated here (plan §2.2, §10 item 2)")
    policy = get_switch("OQ39", defaults if defaults is not None else load_config())
    return load_world_b(real["responses_file"], real["tasks_file"],
                        confidence_range=tuple(real["confidence_range"]),
                        ps_scale=real["ps_scale"], ps_sum_policy=policy,
                        incomplete=real.get("incomplete", "raise"))
