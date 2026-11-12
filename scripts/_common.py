"""Shared parts of the driver scripts (scripts/run_a13_loo.py,
scripts/run_a14_nested.py, scripts/run_a15_tables.py ... run_a18_bagging.py).

  * `load_real_build`  : the ONLY way a script reads a real-data build. It
                         first calls cdm.eval.gate_guard.require_gate, which
                         refuses every build that is not listed as passed or
                         waived in config/gate_status.yaml.
  * `smoke_instances`  : synthetic cases for an end-to-end test of a script
                         (no real data is read; the guard is not involved).
  * result summaries, CSV / JSON writers, the copy to the results folder.

Seeds. The instance table (27 CDMCR features, rule answers) is built with
the config seed (plan OQ15: 0); it fixes the per-case sub-set partitions.
The `--seed` of a script is the classifier seed only (`random_state` of
RF / LR; AM18: RF seed 0 as primary; LOO repeated for seeds 1-4).
"""
import csv
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np                                             # noqa: E402

from cdm.config import load_config                             # noqa: E402
from cdm.eval import gate_guard                                # noqa: E402
from cdm.models import multilabel as M                         # noqa: E402
from cdm.report import thesis_values as T                            # noqa: E402

#: Repo-relative defaults, resolved from this file's location. Frozen builds:
#: <repo>/build/worldA/<scenario>/ (env CDM_BUILD_ROOT replaces <repo>/build,
#: as in cdm.data.worldA.build_worldA.build_root); runs and checkpoints:
#: <repo>/runs/. Both folders are git-ignored.
DEFAULT_BUILD_ROOT = (Path(os.environ["CDM_BUILD_ROOT"]).expanduser().resolve()
                      if os.environ.get("CDM_BUILD_ROOT") else CODE_ROOT / "build") / "worldA"
DEFAULT_RUNS_ROOT = CODE_ROOT / "runs"
#: The published result files live in <repo>/results/. Nothing is copied
#: there unless a driver is called with `--results-root results` (or another
#: folder); the default "" means "do not copy".
DEFAULT_RESULTS_ROOT = ""
PUBLISHED_RESULTS_ROOT = CODE_ROOT / "results"
METHODS_REPORT_ORDER = ("SP", "MR", "WC", "HAC", "DA")
LOG = logging.getLogger("cdm.scripts")


# ---- logging -----------------------------------------------------------------------

def setup_logging(log_path):
    """INFO log of the script and of cdm.eval to `log_path` (appended) and
    to stderr."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fmt.converter = time.gmtime
    handlers = [logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
    for name in ("cdm.eval", "cdm.scripts"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = False
        for h in list(lg.handlers):
            lg.removeHandler(h)
        for h in handlers:
            h.setFormatter(fmt)
            lg.addHandler(h)
    return log_path


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---- data ----------------------------------------------------------------------------

def load_real_cases(build, build_root=DEFAULT_BUILD_ROOT, limit=None):
    """The cases of a frozen real-data build, after the gate guard.
    Returns (cases, info) with info = the gate record and the manifest's
    status; `limit` as in `load_real_build`. Used by the drivers that need
    the raw responses (bagging, step A18)."""
    from cdm.data.worldA.build_worldA import load_frozen
    build_dir = Path(build_root) / build
    gate = gate_guard.require_gate(build_dir)           # raises GateError
    LOG.info("gate guard: build %s is %s (%s, %s, %s)", gate["build"], gate["status"],
             gate["date"], gate["decided_by"], gate["reference"])
    cases, manifest = load_frozen(path=build_dir, verify=True)
    if limit:
        idx = sorted(set(np.linspace(0, len(cases) - 1, int(limit)).round().astype(int)))
        cases = [cases[i] for i in idx]
    info = {"kind": "real", "build": build, "build_dir": str(build_dir), "gate": gate,
            "manifest_status": manifest.get("status"),
            "manifest_created": manifest.get("created"),
            "limit": int(limit) if limit else None}
    return cases, info


def load_real_build(build, build_root=DEFAULT_BUILD_ROOT, limit=None, return_cases=False):
    """Instance table of a frozen real-data build, after the gate guard.

    Returns (inst, info): inst = cdm.approaches.instances.Instances in the
    frozen case order (= the fold order, plan OQ16); info = the gate
    record, the manifest's status / description and the rule baselines of
    the build next to the thesis's. With return_cases: (inst, info, cases).

    limit : None, or a number of cases: an evenly spaced sub-sample (for a
            quick check of the pipeline only; never a result).
    """
    from cdm.approaches import instances as I
    cases, info = load_real_cases(build, build_root, limit)
    t0 = time.time()
    inst = I.build_instances(cases)
    LOG.info("instance table: %d cases, %.0f s; log: %s", len(inst), time.time() - t0,
             inst.log())
    info.update({"n_cases": len(inst), "instance_seed": inst.seed,
                 "instance_log": inst.log(),
                 "rule_baselines": rule_baselines(inst, real=not limit),
                 "rule_baselines_by_domain": rule_baselines_by_domain(inst, real=not limit)})
    return (inst, info, cases) if return_cases else (inst, info)


def smoke_cases(n_cases, seed=0, n_min=12, n_max=70):
    """`n_cases` SYNTHETIC cases (random responses, between n_min and
    n_max - 1 responses each), in four blocks named like the domains. For
    testing a script end to end. No real data is read."""
    from cdm.schema import Case, Response
    rng = np.random.default_rng(seed)
    answers = ("a1", "a2")
    cases = []
    for k in range(n_cases):
        n = int(rng.integers(n_min, n_max))
        truth = answers[int(rng.integers(2))]
        skill = rng.uniform(0.3, 0.7)                  # share voting for the truth
        rs = []
        for _ in range(n):
            right = rng.random() < skill
            v = truth if right else answers[1 - answers.index(truth)]
            c = float(np.clip(rng.normal(0.62 if right else 0.55, 0.25), 0, 1))
            p1 = float(np.clip(rng.normal(0.55 if v == "a1" else 0.45, 0.25), 0, 1))
            rs.append(Response(v=v, c=round(c, 2), ps=(round(p1, 2), round(1 - round(p1, 2), 2))))
        dom = ("General", "NBA", "NFL", "Politics")[min(3, 4 * k // n_cases)]
        cases.append(Case(case_id=f"smoke_{k}", answers=answers, correct=truth,
                          responses=tuple(rs), domain=dom))
    return cases


def smoke_instances(n_cases, seed=0, n_min=12, n_max=70, return_cases=False):
    """Instance table of `n_cases` SYNTHETIC cases (`smoke_cases`). For
    testing a script end to end. No real data is read. Returns (inst,
    info), or (inst, info, cases) with return_cases."""
    from cdm.approaches import instances as I
    cases = smoke_cases(n_cases, seed, n_min, n_max)
    inst = I.build_instances(cases)
    info = {"kind": "SYNTHETIC SMOKE TEST - not a result", "build": "_smoke",
            "gate": None, "n_cases": len(inst), "instance_seed": inst.seed,
            "limit": None, "rule_baselines": rule_baselines(inst, real=False),
            "rule_baselines_by_domain": rule_baselines_by_domain(inst, real=False)}
    return (inst, info, cases) if return_cases else (inst, info)


def rule_baselines(inst, real=True):
    """{method: {recomputed, thesis, difference}}: correct cases of the
    five methods on this instance table, next to the thesis-reported counts
    (both are reported: docs/RULE_BASELINE_CHECK.md)."""
    out = {}
    for m in METHODS_REPORT_ORDER:
        k = int(inst.outcomes(m).sum())
        row = {"recomputed": k, "n": len(inst), "rate": k / len(inst)}
        if real:
            row.update(thesis=T.RULES_THESIS[m], difference=k - T.RULES_THESIS[m])
        out[m] = row
    return out


def rule_baselines_by_domain(inst, real=True):
    """{block: {method: {recomputed, n, thesis, thesis_n}}} for "all" and
    every domain of the instance table (cdm.eval.effectiveness.
    rule_baselines; thesis counts T3 / T3b when `real`)."""
    from cdm.eval import effectiveness as E
    return E.rule_baselines({m: inst.outcomes(m) for m in METHODS_REPORT_ORDER},
                            inst.domains, with_thesis=real)


def hyper_for_seed(seed, config=None, smoke=False):
    """The config's OQ15 hyperparameters with the classifier seed replaced.
    smoke: 10 trees instead of 100 (speed of the synthetic test only)."""
    hyper = M.hyperparameters_from_config(config)
    hyper["seed"] = int(seed)
    if smoke:
        hyper["rf_n_estimators"] = 10
    return hyper


# ---- summaries ----------------------------------------------------------------------

def _f(x):
    x = float(x)
    return None if x != x else x


def result_summary(res, thesis_successes=None, thesis_domains=None,
                   thesis_selected=None, thesis_selected_success=None,
                   thesis_discordant=None):
    """Plain-dict summary of a cdm.eval.loo.EvalResult with the
    thesis-reported values beside every number (None where none is given)."""
    n = res.n
    out = {"approach": res.approach, "protocol": res.protocol, "n": n,
           "successes": res.n_success, "rate": res.success_rate,
           "thesis_successes": thesis_successes,
           "difference": None if thesis_successes is None
           else res.n_success - thesis_successes,
           "nan_in_outputs": int(np.isnan(np.asarray(res.proba, dtype=float)).sum()),
           "techniques": res.summary()["techniques"],
           "ties": res.tie_log(),
           "lr_not_converged": res.meta.get("lr_not_converged")}
    if res.domains is not None:
        dom = {}
        for d, row in res.domain_table().items():
            row = dict(row)
            if thesis_domains and d in thesis_domains:
                row["thesis_successes"] = thesis_domains[d]
                row["difference"] = row["successes"] - thesis_domains[d]
            dom[d] = row
        out["domains"] = dom
    if res.approach == "AMP":
        eff = res.effectiveness()
        sel = {}
        for m in METHODS_REPORT_ORDER:
            if m not in eff:
                continue
            r = eff[m]
            row = {"n_selected": r["n_selected"], "p_selected": r["p_selected"],
                   "n_selected_success": r["n_selected_success"],
                   "p_success_given_selected": _f(r["p_success_given_selected"]),
                   "method_successes": r["n_success"], "p_success": r["p_success"],
                   "pre": _f(r["pre"]), "z": _f(r["z"]), "p": _f(r["p"])}
            if thesis_selected:
                row["thesis_n_selected"] = thesis_selected[m]
            if thesis_selected_success and thesis_selected:
                row["thesis_n_selected_success"] = thesis_selected_success[m]
                row["thesis_p_success_given_selected"] = (
                    thesis_selected_success[m] / thesis_selected[m])
            sel[m] = row
        out["selection"] = sel
        out["selection_total"] = int(sum(r["n_selected"] for r in sel.values()))
    if res.outcomes is not None:
        cmp = {}
        for m in METHODS_REPORT_ORDER:
            if m not in res.labels:
                continue
            b, c = res.discordant(m)
            try:
                chi2, p = res.mcnemar(m)
            except ValueError:                     # b + c = 0: undefined
                chi2 = p = float("nan")
            try:
                z, pz = res.proportion_test(m)
            except (ValueError, ZeroDivisionError):
                z = pz = float("nan")
            row = {"method_successes": int(res.outcomes[:, res.labels.index(m)].sum()),
                   "approach_only_correct": b, "method_only_correct": c,
                   "mcnemar_chi2": _f(chi2), "mcnemar_p": _f(p),
                   "proportion_z": _f(z), "proportion_p": _f(pz)}
            if thesis_discordant and m in thesis_discordant:
                row["thesis_discordant"] = list(thesis_discordant[m])
            cmp[m] = row
        out["versus_rules"] = cmp
    return out


def format_summary(s, title):
    """Short text table of a `result_summary` dict."""
    lines = [title]
    t = s.get("thesis_successes")
    lines.append(f"  successes {s['successes']}/{s['n']} = {s['rate']:.3f}"
                 + ("" if t is None else f"   thesis {t}/{s['n']} = {t / s['n']:.3f}"
                    f"   difference {s['difference']:+d}"))
    for d, r in (s.get("domains") or {}).items():
        if d == "all":
            continue
        tt = r.get("thesis_successes")
        lines.append(f"    {d:<9} {r['successes']:>4}/{r['n']:<4} = {r['rate']:.3f}"
                     + ("" if tt is None else f"   thesis {tt:>4} = {tt / r['n']:.3f}"
                        f"   diff {r['difference']:+d}"))
    if "selection" in s:
        lines.append("  method   selected (thesis)   correct when selected (thesis)   "
                     "P(succ|sel) (thesis)   p")
        for m, r in s["selection"].items():
            ps = r["p_success_given_selected"]
            lines.append(
                f"    {m:<4} {r['n_selected']:>5} ({r.get('thesis_n_selected', '-'):>4})"
                f"   {r['n_selected_success']:>5} ({r.get('thesis_n_selected_success', '-'):>4})"
                f"   {'nan' if ps is None else format(ps, '.3f')}"
                f" ({format(r['thesis_p_success_given_selected'], '.3f') if 'thesis_p_success_given_selected' in r else '-'})"
                f"   {'nan' if r['p'] is None else format(r['p'], '.2g')}")
        lines.append(f"    selection counts sum to {s['selection_total']}")
    if "versus_rules" in s:
        lines.append("  versus rule: (approach only, rule only) correct, thesis pair, McNemar p")
        for m, r in s["versus_rules"].items():
            lines.append(f"    {m:<4} ({r['approach_only_correct']}, {r['method_only_correct']})"
                         f"   thesis {tuple(r['thesis_discordant']) if 'thesis_discordant' in r else '-'}"
                         f"   p = {'undefined' if r['mcnemar_p'] is None else format(r['mcnemar_p'], '.3g')}")
    lines.append(f"  argmax / 0.5 ties: {s['ties']}; fits not converged: "
                 f"{s['lr_not_converged']}; NaN in outputs: {s['nan_in_outputs']}")
    return "\n".join(lines)


# ---- files ---------------------------------------------------------------------------

def _plain(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path, obj):
    """Atomic JSON write (NaN is written as null)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (list, tuple, set)):
            return [clean(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (float, np.floating)):
            return None if o != o or o in (float("inf"), float("-inf")) else float(o)
        return o
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(clean(obj), indent=1, default=_plain, allow_nan=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_cases_csv(path, res, rule_outcomes=None):
    """One row per case of an EvalResult.
    AMP: case_id, domain, [fold, technique,] success, selected, answer,
         P_<method> x q, O_<method> x q.
    DAP: case_id, domain, [fold, technique,] success, predicted, target,
         p_code1, answer, O_<method> of the rules (if stored)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nested = res.folds is not None
    head = ["case_id", "domain"] + (["fold", "technique"] if nested else []) + ["success"]
    if res.approach == "AMP":
        head += ["selected", "answer"] + [f"P_{m}" for m in res.labels] \
            + [f"O_{m}" for m in res.labels]
    else:
        head += ["predicted", "target", "p_code1", "answer"] + [f"O_{m}" for m in res.labels]
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for i in range(res.n):
            row = [res.case_ids[i], res.domains[i] if res.domains else ""]
            if nested:
                row += [int(res.folds[i]), res.techniques[i]]
            row.append(int(res.success[i]))
            ans = res.answers[i] if res.answers is not None else ""
            if res.approach == "AMP":
                row += [res.labels[int(res.selected[i])], ans]
                row += [repr(float(x)) for x in res.proba[i]]
                row += [int(x) for x in res.outcomes[i]]
            else:
                row += [int(res.predicted[i]), int(res.target[i]),
                        repr(float(res.proba[i])), ans]
                if res.outcomes is not None:
                    row += [int(x) for x in res.outcomes[i]]
            w.writerow(row)
    os.replace(tmp, path)


def copy_results(files, dest_dir):
    """Copy small result files to `dest_dir` (a results folder; None or ""
    = do nothing). A failure is logged and never stops a run."""
    if not dest_dir:
        return
    dest_dir = str(dest_dir).rstrip("/") + "/"
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError as exc:
        LOG.warning("results folder %s cannot be created: %s", dest_dir, exc)
        return
    for f in files:
        f = Path(f)
        if not f.is_file():
            continue
        try:
            shutil.copyfile(f, dest_dir + f.name)
        except OSError as exc:
            LOG.warning("copy of %s to %s failed: %s", f, dest_dir, exc)


def run_header(args, data_info, hyper, config=None):
    """What every summary file starts with: when, what data, which gate
    record, which settings."""
    config = config if config is not None else load_config()
    from cdm.eval import loo as L
    return {"written": utc_now(), "argv": sys.argv,
            "data": data_info, "hyper": hyper,
            "settings": {
                "lr_max_iter": hyper.get("lr_max_iter"),
                "technique_tie_rule": config.get("technique_tie_rule"),
                "classifier_seed": hyper.get("seed"),
                "note": "AM18 settings (LR max_iter 10000 + convergence count; "
                        "technique tie = first in table order; RF seed 0 primary): "
                        "defaults of this implementation "
                        "(docs/RECONSTRUCTION_PLAN.md, appendix)"},
            "versions": L._versions(), "code_fingerprint": L.code_fingerprint(),
            "n_jobs": getattr(args, "n_jobs", None)}
