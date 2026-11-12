"""Shared pieces of the World-B drivers of steps B6-B11 (run_b6_rp.py ...
run_b11_report.py). Import this module FIRST: it caps the numeric
libraries at one thread per process before numpy is loaded (the drivers
run the partition seeds in at most 3 processes).

Conventions (as scripts/run_b5_rules.py, decision B3-13, R5):
  * data = config/worldB.yaml (`source: synthetic` shipped; `source: real`
    for the MTurk file: UNTESTED ON REAL DATA);
  * synthetic runs go to <runs-root>/worldB_synthetic/<step>/, a real file
    to <runs-root>/worldB_real/<step>/ — never mixed;
  * every file of a synthetic run carries "SYNTHETIC — not a reproduction":
    first key of a JSON, first line (a "#" comment) of a CSV, first and
    last line of a Markdown file, first line of a log, title of a figure;
  * a run on fewer partitions than the config asks for (--seeds N) is a
    pipeline check: it goes to "<step>_seeds<N>" and is never copied to
    the results folder;
  * nothing is copied to results/ unless --copy-results is given.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                 # noqa: E402
import csv                                                      # noqa: E402
import hashlib                                                  # noqa: E402
import json                                                     # noqa: E402
import sys                                                      # noqa: E402
import time                                                     # noqa: E402
from pathlib import Path                                        # noqa: E402

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np                                              # noqa: E402

from _common import LOG, copy_results, setup_logging, utc_now, write_json, write_text  # noqa: E402,F401
from cdm.config import load_config                              # noqa: E402
from cdm.data.worldB import loader as L                         # noqa: E402
from cdm.eval import worldB_ml as ML                            # noqa: E402

RUNS_ROOT = CODE_ROOT / "runs"
RESULTS_ROOT = str(CODE_ROOT / "results") + "/"
MAX_JOBS = 3


def parser(doc):
    ap = argparse.ArgumentParser(description=doc.split("\n\n")[0])
    ap.add_argument("--config", default=None,
                    help="World-B source file (default config/worldB.yaml)")
    ap.add_argument("--runs-root", default=str(RUNS_ROOT))
    ap.add_argument("--copy-results", action="store_true",
                    help="also copy the small result files of a full run to results/ "
                         "(off by default)")
    ap.add_argument("--no-copy", action="store_true",
                    help="never copy to results/ (the default; overrides --copy-results)")
    ap.add_argument("--n-jobs", type=int, default=MAX_JOBS,
                    help="processes (at most 3)")
    ap.add_argument("--seeds", type=int, default=None, metavar="N",
                    help="pipeline check on the first N partition seeds only; written "
                         "to <step>_seeds<N>, never copied")
    return ap


class Run:
    """Data, options, output folder and marker of one driver call."""

    def __init__(self, args, step, title):
        self.args = args
        self.config = load_config()
        self.data = L.load_from_config(args.config)
        self.synthetic = self.data.synthetic
        self.name = "worldB_synthetic" if self.synthetic else "worldB_real"
        self.mark = self.data.marker if self.synthetic else L.REAL_DATA_STATUS
        self.opts = ML.options_from_config(self.config)
        seeds, self.assignment = ML.seeds_from_config(self.data, self.config)
        self.full = True
        if args.seeds is not None and self.assignment == "random":
            if not 1 <= args.seeds <= len(seeds):
                raise SystemExit(f"--seeds must be between 1 and {len(seeds)}")
            self.full = args.seeds == len(seeds)
            seeds = seeds[:args.seeds]
        self.seeds = tuple(seeds)
        self.step = step if self.full else f"{step}_seeds{len(self.seeds)}"
        self.root = Path(args.runs_root) / self.name
        self.out = self.root / self.step
        self.n_jobs = max(1, min(int(args.n_jobs), MAX_JOBS))
        self.started = time.time()
        setup_logging(self.out / f"{step}.log")
        LOG.info("%s -- %s", self.mark, title)
        LOG.info("data: %d tasks, %d responses, fingerprint %s; partition %s, %d seed(s)",
                 len(self.data.tasks), self.data.n_responses, self.fingerprint[:16],
                 self.assignment, len(self.seeds))

    # -- identity of the data, so that a later step never mixes two data sets
    @property
    def fingerprint(self):
        if not hasattr(self, "_fp"):
            h = hashlib.sha256()
            for case in self.data.tasks:
                h.update(repr((case.case_id, case.answers, case.correct)).encode())
                for r in case.responses:
                    h.update(repr((r.v, r.c, r.ps)).encode())
            self._fp = h.hexdigest()
        return self._fp

    def sibling(self, step):
        """Folder of another step of the same run (same seeds suffix)."""
        return self.root / (step if self.full else f"{step}_seeds{len(self.seeds)}")

    def head(self, step, plan):
        """First keys of every JSON document of the run."""
        doc = {"_marker": self.mark} if self.synthetic else {"_status": self.mark}
        doc.update({
            "written": utc_now(), "step": step, "plan": plan,
            "synthetic": self.synthetic,
            "real_data_path": L.REAL_DATA_STATUS,
            "data": {"n_tasks": len(self.data.tasks),
                     "n_responses": self.data.n_responses,
                     "fingerprint": self.fingerprint,
                     "generator_seed": self.data.log.get("seed")},
            "partition": {"assignment": self.assignment,
                          "seeds": [s for s in self.seeds],
                          "full_set_of_seeds": self.full},
            "options": {k: v for k, v in self.opts.items()},
            "versions": versions(),
        })
        return doc

    def csv(self, path, fieldnames, rows):
        """CSV whose first line is a comment with the marker / status."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(f"# {self.mark}\n")
            w = csv.DictWriter(fh, fieldnames=list(fieldnames), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        return path

    def copy(self, files):
        """Copy result files to results/ (full runs, and only on --copy-results)."""
        if not getattr(self.args, "copy_results", False) or self.args.no_copy \
                or not self.full:
            return
        # Safety: only runs under the default runs root are copied, so tests and
        # trial runs can never write to the results/ folder.
        if Path(self.args.runs_root).resolve() != RUNS_ROOT.resolve():
            return
        copy_results(files, RESULTS_ROOT + self.name + "/" + self.step + "/")

    def minutes(self):
        return (time.time() - self.started) / 60.0


def versions():
    import shap
    import sklearn
    import statsmodels
    import xgboost
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
            "shap": shap.__version__, "statsmodels": statsmodels.__version__}


def run_seeds(run, approach, classifier, keep_predictions=True):
    """ML.run_partition for every seed of the run, in up to 3 processes."""
    from joblib import Parallel, delayed
    t0 = time.time()
    jobs = (delayed(ML.run_partition)(run.data, seed, run.assignment, approach,
                                      classifier, run.opts, keep_predictions)
            for seed in run.seeds)
    runs = Parallel(n_jobs=min(run.n_jobs, len(run.seeds)))(jobs)
    LOG.info("%s / %s: %d partition(s) in %.1f min", approach, classifier,
             len(runs), (time.time() - t0) / 60.0)
    return runs


def seed_key(seed):
    return "original" if seed is None else str(int(seed))


def bits(arr):
    return "".join("1" if x else "0" for x in arr)


def unbits(text):
    return np.asarray([c == "1" for c in text], dtype=bool)


def outcomes_document(run, runs, approach, classifier):
    """Per partition and sub-set: was the answer correct (per strategy for
    RP), and the four rules. Read by the later steps (B8, B9, B11)."""
    doc = run.head("outcomes", "§5.7")
    doc.update({"approach": approach, "classifier": classifier,
                "n_subsets": runs[0]["n_subsets"], "per_seed": {}})
    for r in runs:
        cell = {"case_ids": r["case_ids"],
                "rules": {k: bits(v) for k, v in r["rules"].items()}}
        if approach == "RP":
            cell["correct"] = {k: bits(v) for k, v in r["correct"].items()}
            cell["summary"] = r["summary"]
        else:
            cell["correct"] = bits(r["correct"])
            cell["summary"] = r["summary"]
        cell["instances"] = r["instances"]
        cell["predictions_log"] = r["predictions_log"]
        doc["per_seed"][seed_key(r["seed"])] = cell
    return doc


def load_outcomes(run, path, approach, classifier):
    """Read an outcomes file of this run's data and seeds; None if it does
    not exist. A file of other data, seeds or options is refused."""
    path = Path(path)
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    want = [seed_key(s) for s in run.seeds]
    problems = []
    if doc["data"]["fingerprint"] != run.fingerprint:
        problems.append("other data")
    if list(doc["per_seed"]) != want:
        problems.append("other partition seeds")
    if doc["approach"] != approach or doc["classifier"] != classifier:
        problems.append("other approach / classifier")
    if json.loads(json.dumps(doc["options"])) != json.loads(json.dumps(run.opts)):
        problems.append("other options")
    if problems:
        raise SystemExit(f"{path}: written for {', '.join(problems)}; re-run its step")
    return doc


def runs_from_outcomes(doc):
    """The parts of ML.run_partition's output that the summaries need."""
    runs = []
    for key, cell in doc["per_seed"].items():
        r = {"seed": None if key == "original" else int(key),
             "n_subsets": doc["n_subsets"], "case_ids": cell["case_ids"],
             "rules": {k: unbits(v) for k, v in cell["rules"].items()},
             "summary": cell["summary"], "instances": cell["instances"],
             "predictions_log": cell["predictions_log"]}
        if doc["approach"] == "RP":
            r["correct"] = {k: unbits(v) for k, v in cell["correct"].items()}
        else:
            r["correct"] = unbits(cell["correct"])
        runs.append(r)
    return runs


def prediction_rows(runs):
    """One row per instance and partition (the out-of-group predictions)."""
    for r in runs:
        p = r["predictions"]
        for i in range(len(p["y"])):
            g = int(p["group"][i])
            yield {"partition_seed": seed_key(r["seed"]), "subset": r["case_ids"][g],
                   "member": int(p["member"][i]), "answer_index": int(p["answer_index"][i]),
                   "target": int(p["y"][i]), "label": int(p["label"][i]),
                   "p_true": repr(float(p["p_true"][i])),
                   "p_false": repr(float(p["p_false"][i]))}


PREDICTION_FIELDS = ("partition_seed", "subset", "member", "answer_index", "target",
                     "label", "p_true", "p_false")
