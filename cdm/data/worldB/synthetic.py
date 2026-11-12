"""SYNTHETIC World-B stand-in (plan §2.3, §9.3 row B3, OQ43).

    *********************************************************************
    *  SYNTHETIC — not a reproduction.  The responses made here are      *
    *  INVENTED.  They exist only so that the RP/AP code can be built    *
    *  and tested before the private MTurk data arrives (plan §2.3).     *
    *  No number computed from them is a result about the thesis.        *
    *********************************************************************

What is taken from the papers and what is invented (tags as in plan §0):

  [EVIDENCE]  the 22 tasks with |A_P|, |R_P| and S(a*) of TH Table 4.1
              (cdm/data/worldB/table41.py); the number of solvers of a task
              is round(S(a*) * |R_P|) (plan §2.3) and is met exactly. For
              four tasks the printed S(a*) allows a second count
              (table41.AMBIGUOUS_SOLVER_COUNTS).
  [EVIDENCE]  the tasks are "misleading" problems (TH p.52), which motivates
              a lure answer; nothing more.
  [INFERRED]  predicted supports are shares that sum to 1 (plan §2.3: TH
              Table A.4 PChiSq = .682 needs it).
  [INVENTED]  everything else: which option is correct and which is the
              lure, the lure's share, how the other wrong answers are
              spread, the confidence model, the prediction model, the order
              of the responses. See PARAMETERS; every entry carries its tag.
  [INVENTED-CALIBRATED]  the parameters marked so are chosen by the seeded
              grid search `calibrate` on the four rule counts MR 34, WC 34,
              SP 33, HAC 22 of 77 (T24), averaged over the partition seeds
              0..29 -- the only calibration plan §2.3 allows. RP/AP
              outcomes (T23, T26-T29) are never looked at.

Model of one task t with n responses, m options, k solvers:
  layout   the correct option and the lure are two different options drawn
           uniformly; lure share lambda_t ~ Beta(mean = lure_share_mean,
           concentration = lure_share_concentration).
  votes    exactly k responses choose the correct option. The n - k others
           are multinomial over the m - 1 wrong options with probabilities
           lambda_t for the lure and (1 - lambda_t) * Dirichlet(
           distractor_alpha, ...) for the m - 2 others. The responses are
           then put in a random order.
  q        the task's realised vote shares (counts / n): the "true shares".
  conf     c ~ Beta(mean mu, concentration conf_concentration),
           mu = conf_base + conf_popularity_slope * q[v]
                + conf_solver_bonus * [v is correct], clipped to [.05, .95].
  ps       mean vector  (1 - w - u_r) * q + w * e_v + u_r * e_lure,
           w = ps_own_shift (false consensus), u_r = ps_solver_lure_shift
           for a solver and 0 otherwise (solvers anticipate the lure);
           ps ~ Dirichlet(ps_concentration * mean + ps_floor), so it is
           non-negative and sums to 1. With probability ps_onehot_rate the
           response instead predicts 100% for its own answer (TH Table A.4
           shows such a response: PSv = 1).

Random streams: one numpy Generator per task and component (layout, votes,
conf, ps), seeded by the first 8 bytes of
SHA-256("<seed>|synthetic_worldB|<task name>|<component>"). So the votes do
not change when only confidence or prediction parameters change, which is
what lets `calibrate` search the three blocks one after the other.

The calibrated values are stored in config/synthetic_worldB.yaml (written
by scripts/run_b3_synthetic.py --calibrate) and read by `load_parameters`.
"""
import hashlib
import math
import textwrap
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yaml

from cdm.config import CODE_ROOT
from cdm.data.worldB import table41 as T41
from cdm.data.worldB.loader import SYNTHETIC_MARKER, WorldBData
from cdm.schema import Case, Response

PARAMETERS_PATH = CODE_ROOT / "config" / "synthetic_worldB.yaml"
DEFAULT_SEED = 0                                   # plan §8 OQ15

TAGS = ("EVIDENCE", "INFERRED", "INVENTED", "INVENTED-CALIBRATED")

#: Every parameter of the generator: tag, the grid searched by `calibrate`
#: (INVENTED-CALIBRATED) or the fixed value (INVENTED), and what it is.
#: The grids are part of the invention (plan §2.3: "a seeded grid search").
#: They were written down before the first search was run (this implementation's
#: statement; it cannot be checked from the files), with ONE later
#: change (decision B3-9): the first search ended at the corner of the
#: prediction grid (ps_own_shift 0.4, ps_solver_lure_shift 0.3,
#: ps_concentration 40; SP mean 31.0 against the target 33), so 0.4 and 0.5
#: were added to the ps_solver_lure_shift grid and the search was run again.
#: The first grid is kept in FIRST_GRID_PS_SOLVER_LURE_SHIFT.
PARAMETERS = OrderedDict([
    ("lure_share_mean", {
        "tag": "INVENTED-CALIBRATED", "block": "votes",
        "grid": (0.3, 0.4, 0.5, 0.6, 0.7),
        "what": "mean share of the wrong votes that go to the lure answer "
                "(lambda_t is drawn per task around it)"}),
    ("lure_share_concentration", {
        "tag": "INVENTED", "block": "votes", "fixed": 10.0,
        "what": "concentration (a + b) of the Beta that lambda_t is drawn from"}),
    ("distractor_alpha", {
        "tag": "INVENTED-CALIBRATED", "block": "votes",
        "grid": (0.5, 1.0, 2.0, 4.0),
        "what": "Dirichlet alpha of the split of the remaining wrong votes "
                "over the m - 2 other wrong options"}),
    ("conf_base", {
        "tag": "INVENTED", "block": "conf", "fixed": 0.6,
        "what": "mean confidence of a response whose answer nobody else chose"}),
    ("conf_popularity_slope", {
        "tag": "INVENTED-CALIBRATED", "block": "conf",
        "grid": (0.0, 0.1, 0.2, 0.3, 0.4),
        "what": "rise of the mean confidence with the vote share of the "
                "response's answer (consensuality, Koriat 2008)"}),
    ("conf_solver_bonus", {
        "tag": "INVENTED-CALIBRATED", "block": "conf",
        "grid": (0.0, 0.02, 0.05, 0.08),
        "what": "extra mean confidence of a solver (plan: 'a small solver bonus')"}),
    ("conf_concentration", {
        "tag": "INVENTED-CALIBRATED", "block": "conf",
        "grid": (4.0, 8.0, 16.0),
        "what": "concentration (a + b) of the Beta the confidence is drawn from"}),
    ("ps_own_shift", {
        "tag": "INVENTED-CALIBRATED", "block": "ps",
        "grid": (0.1, 0.2, 0.3, 0.4),
        "what": "weight moved from the true shares to the response's own "
                "answer (false consensus)"}),
    ("ps_solver_lure_shift", {
        "tag": "INVENTED-CALIBRATED", "block": "ps",
        "grid": (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
        "what": "weight a solver moves from the true shares to the lure "
                "(solvers anticipate the lure; Prelec 2017)"}),
    ("ps_concentration", {
        "tag": "INVENTED-CALIBRATED", "block": "ps",
        "grid": (5.0, 10.0, 20.0, 40.0),
        "what": "Dirichlet concentration of a prediction around its mean "
                "vector (larger = less noise)"}),
    ("ps_onehot_rate", {
        "tag": "INVENTED", "block": "ps", "fixed": 0.05,
        "what": "share of responses that predict 100% for their own answer "
                "(such a response exists: TH Table A.4, PSv = 1; the rate is "
                "invented)"}),
    ("ps_floor", {
        "tag": "INVENTED", "block": "ps", "fixed": 1e-6,
        "what": "added to every Dirichlet alpha so that none is 0 (numerical)"}),
])

#: The ps_solver_lure_shift grid of the first search (see above).
FIRST_GRID_PS_SOLVER_LURE_SHIFT = (0.0, 0.05, 0.1, 0.2, 0.3)

#: Plan §2.3, row "Calibration targets": the rates are [EVIDENCE] (TH Fig
#: 5.3); the counts are the only integers of 77 that round to them.
CALIBRATION_TARGETS = OrderedDict([("MR", 34), ("WC", 34), ("SP", 33), ("HAC", 22)])
CALIBRATION_SEEDS = tuple(range(30))
#: The three blocks are searched in this order; each block minimises the
#: squared error of the rules that it alone decides.
CALIBRATION_STAGES = (("votes", ("MR",)), ("conf", ("WC", "HAC")), ("ps", ("SP",)))

#: Clip of the mean confidence mu. [INVENTED] constant, not calibrated and
#: not an entry of PARAMETERS; named in the header of the YAML file.
CONF_CLIP = (0.05, 0.95)

#: Generator-seed caveat (decision B3-10). The numbers come from a separate
#: check, not from this module; they hold for the calibrated parameter set in
#: config/synthetic_worldB.yaml and generator seed 0.
GENERATOR_SEED_NOTE = (
    "Note on the generator seed: the calibration was done on "
    "the single data set drawn with generator seed 0. With the same parameters "
    "and other generator seeds the counts move well outside ±2 ("
    "generator seeds 1–8, 10 partition seeds each: MR 25–35, WC 26–34, SP "
    "31–37, HAC 16–21; 2 of the 8 seeds are within ±2 on all four rules). The "
    "agreement holds for generator seed 0 only: it is a property of this one "
    "data set, not of the model.")

TASK_ID_FORMAT = "SYNTHETIC-T{:02d}"


# ---- parameters ----------------------------------------------------------------------

def check_parameters(params):
    """Validate a full {name: value} parameter set; returns floats."""
    missing = [k for k in PARAMETERS if k not in params]
    extra = [k for k in params if k not in PARAMETERS]
    if missing or extra:
        raise ValueError(f"parameter set: missing {missing}, unknown {extra}")
    p = {k: float(params[k]) for k in PARAMETERS}
    if not all(math.isfinite(v) for v in p.values()):
        raise ValueError("parameters must be finite numbers")
    if not 0 < p["lure_share_mean"] < 1:
        raise ValueError("lure_share_mean must be in (0, 1)")
    for k in ("lure_share_concentration", "distractor_alpha",
              "conf_concentration", "ps_concentration", "ps_floor"):
        if not p[k] > 0:
            raise ValueError(f"{k} must be > 0")
    if not (p["ps_own_shift"] >= 0 and p["ps_solver_lure_shift"] >= 0
            and p["ps_own_shift"] + p["ps_solver_lure_shift"] <= 1):
        raise ValueError("ps_own_shift and ps_solver_lure_shift must be >= 0 "
                         "and sum to at most 1")
    if not 0 <= p["ps_onehot_rate"] <= 1:
        raise ValueError("ps_onehot_rate must be in [0, 1]")
    return p


def start_parameters():
    """The fixed values plus the FIRST grid value of every calibrated
    parameter: the starting point of `calibrate` (not a calibrated set)."""
    return {k: (spec["fixed"] if "fixed" in spec else spec["grid"][0])
            for k, spec in PARAMETERS.items()}


def load_parameters(path=None):
    """Read config/synthetic_worldB.yaml: (parameters, the whole document).

    Raises if the file's marker or tags do not match this module, or if a
    calibrated value is not on its grid.
    """
    path = Path(path or PARAMETERS_PATH)
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if doc.get("marker") != SYNTHETIC_MARKER:
        raise ValueError(f"{path.name}: the marker {SYNTHETIC_MARKER!r} is missing")
    params = {}
    for name, spec in PARAMETERS.items():
        if name not in doc["parameters"]:
            raise ValueError(f"{path.name}: parameter {name!r} is missing")
        entry = doc["parameters"][name]
        if entry.get("tag") != spec["tag"]:
            raise ValueError(f"{path.name}: {name} must be tagged {spec['tag']}")
        value = float(entry["value"])
        if "grid" in spec and not any(value == g for g in spec["grid"]):
            raise ValueError(f"{path.name}: {name} = {value} is not on its grid "
                             f"{spec['grid']}")
        if "fixed" in spec and value != spec["fixed"]:
            raise ValueError(f"{path.name}: {name} = {value}, the fixed value "
                             f"is {spec['fixed']}")
        params[name] = value
    unknown = [k for k in doc["parameters"] if k not in PARAMETERS]
    if unknown:
        raise ValueError(f"{path.name}: unknown parameters {unknown}")
    return check_parameters(params), doc


# ---- generator -----------------------------------------------------------------------

def _rng(seed, task_name, component):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"the generator seed must be an integer, got {seed!r}")
    digest = hashlib.sha256(
        f"{int(seed)}|synthetic_worldB|{task_name}|{component}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def option_labels(m):
    """Option labels "o1".."om": the canonical order of a task's answers."""
    return tuple(f"o{i}" for i in range(1, m + 1))


def generate_task(task, params, seed=DEFAULT_SEED):
    """One task of TH Table 4.1 -> arrays (see the module docstring).

    Returns a dict: votes (option index per response), c, ps (n x m),
    correct, lure (option indices), lure_share, counts, q.
    """
    p = params
    n, m, k = task.n_responses, task.n_answers, T41.n_solvers(task)

    # layout: drawn before anything that depends on a parameter, so it is
    # the same for every parameter set
    r = _rng(seed, task.name, "layout")
    correct = int(r.integers(m))
    wrong = [i for i in range(m) if i != correct]
    lure = int(wrong[int(r.integers(m - 1))])
    others = [i for i in wrong if i != lure]
    a = p["lure_share_mean"] * p["lure_share_concentration"]
    b = (1 - p["lure_share_mean"]) * p["lure_share_concentration"]
    lure_share = float(r.beta(a, b))

    # votes: exactly k solvers
    r = _rng(seed, task.name, "votes")
    probs = np.zeros(m)
    probs[lure] = lure_share
    probs[others] = (1 - lure_share) * r.dirichlet([p["distractor_alpha"]] * len(others))
    wrong_probs = probs[wrong] / probs[wrong].sum()
    wrong_counts = r.multinomial(n - k, wrong_probs)
    counts = np.zeros(m, dtype=int)
    counts[correct] = k
    counts[wrong] = wrong_counts
    votes = np.repeat(np.arange(m), counts)
    votes = votes[r.permutation(n)]
    q = counts / n

    # confidence
    r = _rng(seed, task.name, "conf")
    mu = (p["conf_base"] + p["conf_popularity_slope"] * q[votes]
          + p["conf_solver_bonus"] * (votes == correct))
    mu = np.clip(mu, *CONF_CLIP)
    kc = p["conf_concentration"]
    c = np.clip(r.beta(mu * kc, (1 - mu) * kc), 0.0, 1.0)

    # predictions
    r = _rng(seed, task.name, "ps")
    onehot = r.random(n) < p["ps_onehot_rate"]
    w = p["ps_own_shift"]
    ps = np.zeros((n, m))
    for i in range(n):
        v = votes[i]
        u = p["ps_solver_lure_shift"] if v == correct else 0.0
        mean = (1 - w - u) * q
        mean[v] += w
        mean[lure] += u
        draw = r.dirichlet(p["ps_concentration"] * mean + p["ps_floor"])
        if onehot[i]:
            draw = np.zeros(m)
            draw[v] = 1.0
        ps[i] = draw
    ps = np.clip(ps, 0.0, 1.0)
    ps = ps / ps.sum(axis=1, keepdims=True)
    return {"votes": votes, "c": c, "ps": ps, "correct": correct, "lure": lure,
            "lure_share": lure_share, "counts": counts, "q": q}


def generate(params=None, seed=DEFAULT_SEED):
    """The synthetic World-B data set: a WorldBData with `synthetic = True`
    and the marker "SYNTHETIC — not a reproduction".

    params : a full parameter set; default = config/synthetic_worldB.yaml.
    seed   : generator seed (default 0).

    The task ids are "SYNTHETIC-T01".. in the order of TH Table 4.1; the
    names and types are those of the table (so that the Table 4.1 check of
    the loader can match them); every Case carries the marker as `domain`.
    `log["layout"]` records the invented correct / lure options and lure
    shares per task.
    """
    if params is None:
        params, _ = load_parameters()
    params = check_parameters(params)
    cases, ids, layout = [], [], []
    for t, task in enumerate(T41.TABLE_4_1, start=1):
        g = generate_task(task, params, seed)
        labels = option_labels(task.n_answers)
        tid = TASK_ID_FORMAT.format(t)
        responses = tuple(
            Response(v=labels[int(v)], c=float(ci), ps=tuple(float(x) for x in row))
            for v, ci, row in zip(g["votes"], g["c"], g["ps"]))
        cases.append(Case(case_id=tid, answers=labels, correct=labels[g["correct"]],
                          responses=responses, domain=SYNTHETIC_MARKER))
        ids.append(tuple(f"{tid}-r{i:04d}" for i in range(1, task.n_responses + 1)))
        layout.append({"task_id": tid, "name": task.name,
                       "correct": labels[g["correct"]], "lure": labels[g["lure"]],
                       "lure_share": g["lure_share"],
                       "votes": {labels[i]: int(x) for i, x in enumerate(g["counts"])}})
    log = {"status": SYNTHETIC_MARKER, "generator": "cdm.data.worldB.synthetic",
           "seed": int(seed), "parameters": dict(params), "layout": layout,
           "tags": {k: PARAMETERS[k]["tag"] for k in PARAMETERS}}
    return WorldBData(
        tasks=tuple(cases),
        names=tuple(t.name for t in T41.TABLE_4_1),
        types=tuple(t.type for t in T41.TABLE_4_1),
        response_ids=tuple(ids), original_subsets=None,
        synthetic=True, marker=SYNTHETIC_MARKER, log=log)


# ---- feature oracle of plan §2.3 --------------------------------------------------------

def a4_like_instances(subsets):
    """Sub-set answers that look like the example of TH Tables A.4 / A.6
    (plan §2.3, row "Feature oracle"): an answer with support 8/33 that is
    not the majority answer and has a supporter who predicts 100% for it.

    Returns a list of (case_id, answer, number of such supporters).
    """
    out = []
    for case in subsets.cases:
        if case.n != 33:
            continue
        counts = {a: 0 for a in case.answers}
        for r in case.responses:
            counts[r.v] += 1
        top = max(counts.values())
        for i, a in enumerate(case.answers):
            if counts[a] == 8 and counts[a] < top:
                k = sum(1 for r in case.responses if r.v == a and r.ps[i] == 1.0)
                if k:
                    out.append((case.case_id, a, k))
    return out


# ---- calibration (plan §2.3 "Rules") ----------------------------------------------------

def rule_means(params, seed=DEFAULT_SEED, partition_seeds=CALIBRATION_SEEDS,
               rules=None):
    """Mean success count of the rules over the partition seeds for one
    parameter set: {rule: mean count of 77}."""
    from cdm.eval import worldB_rules as WR
    from cdm.data.worldB import partition as WP
    rules = tuple(rules or WR.RULES_B)
    data = generate(params, seed)
    totals = {r: 0 for r in rules}
    for s in partition_seeds:
        res = WR.evaluate_subsets(WP.subset_cases(data, seed=s), rules=rules)
        for r in rules:
            totals[r] += res[r]["correct"]
    return {r: totals[r] / len(partition_seeds) for r in rules}


def _loss(means, rules):
    return math.fsum((means[r] - CALIBRATION_TARGETS[r]) ** 2 for r in rules)


def calibrate(seed=DEFAULT_SEED, partition_seeds=CALIBRATION_SEEDS, grid=None,
              progress=None):
    """Seeded grid search of the INVENTED-CALIBRATED parameters on the four
    rule targets (plan §2.3).

    The search is done block by block (CALIBRATION_STAGES): the vote
    parameters on MR, then the confidence parameters on WC and HAC, then
    the prediction parameters on SP. Within a block every combination of
    the block's grids is tried; the loss is the sum over the block's rules
    of (mean count over the partition seeds - target)^2; the first
    combination in grid order with the smallest loss is kept. Because each
    rule depends only on the votes and on its own block, and the blocks
    have separate random streams, the order of the last two blocks does not
    matter; it is not the same as one joint search, which could trade MR
    error against the other rules.

    grid : {name: values} to replace grids of PARAMETERS (tests).

    Returns a dict: parameters (the calibrated set), stages (per block: the
    table of every combination with its means and loss, and the chosen
    one), means (all four rules at the calibrated set), loss.
    """
    import itertools
    grid = dict(grid or {})
    params = start_parameters()
    stages = []
    for block, rules in CALIBRATION_STAGES:
        names = [k for k, s in PARAMETERS.items()
                 if s["block"] == block and "grid" in s]
        values = [tuple(grid.get(k, PARAMETERS[k]["grid"])) for k in names]
        table, best = [], None
        for combo in itertools.product(*values):
            trial = dict(params)
            trial.update(zip(names, combo))
            means = rule_means(check_parameters(trial), seed, partition_seeds, rules)
            loss = _loss(means, rules)
            row = {"values": dict(zip(names, combo)), "means": means, "loss": loss}
            table.append(row)
            if best is None or loss < best["loss"]:
                best = row
            if progress:
                progress(block, row)
        params.update(best["values"])
        stages.append({"block": block, "rules": list(rules), "names": names,
                       "table": table, "chosen": best})
    params = check_parameters(params)
    means = rule_means(params, seed, partition_seeds)
    return {"parameters": params, "stages": stages, "means": means,
            "loss": _loss(means, tuple(CALIBRATION_TARGETS)),
            "seed": int(seed), "partition_seeds": list(partition_seeds)}


def _yaml_float(x):
    """repr of a float that PyYAML reads back as a float ("1e-06" without a
    decimal point would be read as text)."""
    text = repr(float(x))
    if "e" in text and "." not in text.split("e")[0]:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}.0e{exponent}"
    return text


def parameters_document(result):
    """The YAML text of config/synthetic_worldB.yaml for a `calibrate`
    result: every parameter with its tag, value, grid and meaning; the
    targets; the rule means reached."""
    lines = [
        "# " + SYNTHETIC_MARKER,
        "# Parameters of the synthetic World-B stand-in (plan §2.3, OQ43).",
        "# Written by scripts/run_b3_synthetic.py --calibrate; do not edit by hand.",
        "# Every value below is INVENTED: no paper supports it. The values tagged",
        "# INVENTED-CALIBRATED were picked by a seeded grid search on the four rule",
        "# counts of TH Fig 5.3 (T24) and on nothing else. RP/AP results are not",
        "# calibration targets. To be replaced by the real MTurk data.",
        "#",
        *["# " + ln for ln in textwrap.wrap(GENERATOR_SEED_NOTE, 76)],
        "#",
        "# Also INVENTED, fixed in cdm/data/worldB/synthetic.py and not listed as",
        "# parameters below: the clip of the mean confidence to "
        f"[{CONF_CLIP[0]}, {CONF_CLIP[1]}]",
        "# (CONF_CLIP) [INVENTED]; the uniform draw of the correct option and of the",
        "# lure option of each task (generate_task, 'layout') [INVENTED].",
        f"marker: \"{SYNTHETIC_MARKER}\"",
        "plan: \"§2.3; §8 OQ43\"",
        f"seed: {result['seed']}",
        "parameters:",
    ]
    for name, spec in PARAMETERS.items():
        lines.append(f"  {name}:")
        lines.append(f"    value: {_yaml_float(result['parameters'][name])}")
        lines.append(f"    tag: {spec['tag']}")
        if "grid" in spec:
            lines.append(f"    grid: [{', '.join(_yaml_float(g) for g in spec['grid'])}]")
        lines.append(f"    block: {spec['block']}")
        lines.append(f"    what: \"{spec['what']}\"")
    lines += [
        "evidence:",
        "  tasks: \"TH Table 4.1 p.54 = MCF Table 3: 22 tasks, |A_P|, |R_P|, S(a*) [EVIDENCE]\"",
        "  solvers_per_task: \"round(S(a*) * |R_P|), met exactly [EVIDENCE to the printed "
        "2 d.p., plan §2.3; 4 tasks ambiguous, decision B3-2]\"",
        "  partition: \"disjoint [EVIDENCE, TH p.53, p.55]; k = floor(|R_P| / 30) -> 77 [INFERRED, strong]\"",
        "  ps_sum_to_one: \"[INFERRED] from TH Table A.4 (PChiSq = .682)\"",
        "  confidence_scale: \"continuous in [0, 1] [INVENTED]; the scale is not reported (TH p.53)\"",
        "calibration:",
        "  method: \"seeded grid search, block by block: votes on MR, confidence on WC and HAC, predictions on SP\"",
        "  targets_tag: \"rates EVIDENCE (TH Fig 5.3); counts = the only integers of 77 that round to them\"",
        "  targets: {" + ", ".join(f"{k}: {v}" for k, v in CALIBRATION_TARGETS.items()) + "}",
        f"  partition_seeds: [{result['partition_seeds'][0]}, {result['partition_seeds'][-1]}]",
        f"  n_partition_seeds: {len(result['partition_seeds'])}",
        "  mean_counts_reached: {" + ", ".join(
            f"{k}: {result['means'][k]!r}" for k in CALIBRATION_TARGETS) + "}",
        f"  loss: {result['loss']!r}",
        "  combinations_tried: {" + ", ".join(
            f"{s['block']}: {len(s['table'])}" for s in result["stages"]) + "}",
        "  not_calibration_targets: \"RP/AP outcomes T23, T26-T29 (plan §2.3)\"",
    ]
    return "\n".join(lines) + "\n"
