"""World-B rule baselines: MR, WC, SP and HAC on the 77 sub-sets (plan §5.7,
§9.3 row B5; TH §5.2 p.75-76, Fig 5.3; MCF §5).

"Baselines MR, WC, HAC and SP on the same 77 sub-sets (no DA)" (plan §5.7).
The aggregators are those of cdm/aggregators.py with the options of
`aggregators.options_from_config` (OQ8 tie policy, AM3 tie tolerance).

Thesis-reported values (T24, TH Fig 5.3 p.76): success rates 0.442 / 0.442 / 0.429 / 0.286
for MR / WC / SP / HAC. The thesis prints rates only; 34 / 34 / 33 / 22 of
77 are the only integers that round to them (plan §2.3; checked in
tests/test_b5_rules.py).

The partition into sub-sets is random and the original one is unknown
(plan §8 OQ25): the baselines are computed for 30 partition seeds (0..29)
and reported as mean and SD of the success COUNT over the seeds, with the
per-seed counts. Ties are logged per rule (OQ8): the number of tied
sub-sets and the range of the count over every resolution of the ties.

On the SYNTHETIC stand-in (plan §2.3) the four rule rates are the
calibration targets of the generator, so their agreement with the thesis
is BY CONSTRUCTION and is no evidence of anything. Every table this module
renders for synthetic data says so and keeps the thesis numbers in a
separate column.

The thesis-reported values are kept here, beside the code that uses them
(decision R1).
"""
import math
from dataclasses import dataclass, field

from cdm import aggregators as AG
from cdm.data.worldB import partition as WP
from cdm.data.worldB.loader import REAL_DATA_STATUS, SYNTHETIC_MARKER
from cdm.data.worldB.synthetic import GENERATOR_SEED_NOTE

#: Rule order of TH Fig 5.3 (after RP and AP).
RULES_B = ("MR", "WC", "SP", "HAC")

#: T24. TH Fig 5.3 p.76 (printed rates) and the plan's counts of 77.
N_SUBSETS = 77
THESIS_RATE = {"MR": 0.442, "WC": 0.442, "SP": 0.429, "HAC": 0.286}
THESIS_COUNT = {"MR": 34, "WC": 34, "SP": 33, "HAC": 22}

#: Plan §8 OQ25: "repeated over 30 seeds".
DEFAULT_SEEDS = tuple(range(30))


def subset_outcome(case, policy="canonical_first", tol=AG.P.DEFAULT_TIE_ABS_TOL):
    """The four rules on one sub-set case.

    Returns {rule: {"answer", "correct" (bool), "tied" (list of the tied
    best answers, canonical order), "correct_possible" (bool: the correct
    answer is among the tied best ones)}}.
    """
    out = {}
    for rule in RULES_B:
        tied = AG.tied_answers(rule, case.responses, case.answers, tol)
        answer = AG.choose(rule, case.responses, case.answers, policy, tol)
        out[rule] = {"answer": answer, "correct": answer == case.correct,
                     "tied": tied, "correct_possible": case.correct in tied}
    return out


def evaluate_subsets(subsets, policy="canonical_first",
                     tol=AG.P.DEFAULT_TIE_ABS_TOL, rules=RULES_B):
    """Success counts of the rules over the sub-sets of one partition.

    Returns {rule: {"correct", "ties" (tied sub-sets), "low", "high" (range
    of the count over every resolution of the ties)}} plus "n".
    """
    res = {r: {"correct": 0, "ties": 0, "low": 0, "high": 0} for r in rules}
    for case in subsets.cases:
        for rule in rules:
            tied = AG.tied_answers(rule, case.responses, case.answers, tol)
            answer = AG.choose(rule, case.responses, case.answers, policy, tol)
            cell = res[rule]
            cell["correct"] += answer == case.correct
            if len(tied) > 1:
                cell["ties"] += 1
                cell["high"] += case.correct in tied
            else:
                cell["low"] += answer == case.correct
                cell["high"] += answer == case.correct
    res["n"] = len(subsets)
    return res


def _mean(xs):
    return math.fsum(xs) / len(xs)


def _sd(xs, ddof):
    if len(xs) - ddof <= 0:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - ddof))


@dataclass
class RuleRun:
    """Rule baselines over several partition seeds.

    per_seed : {seed: evaluate_subsets(...)}
    """
    seeds: tuple
    per_seed: dict
    n_subsets: int
    synthetic: bool
    marker: str
    assignment: str
    policy: str
    tol: float
    meta: dict = field(default_factory=dict)

    def counts(self, rule):
        return [self.per_seed[s][rule]["correct"] for s in self.seeds]

    def summary(self):
        """Per rule: mean, SD (sample, ddof 1, and population), min, max of
        the success count over the seeds; the same as rates; ties."""
        out = {}
        for rule in RULES_B:
            xs = self.counts(rule)
            out[rule] = {
                "mean_count": _mean(xs),
                "sd_count": _sd(xs, 1), "sd_count_population": _sd(xs, 0),
                "min_count": min(xs), "max_count": max(xs),
                "mean_rate": _mean(xs) / self.n_subsets,
                "sd_rate": _sd(xs, 1) / self.n_subsets,
                "mean_ties": _mean([self.per_seed[s][rule]["ties"] for s in self.seeds]),
                "mean_low": _mean([self.per_seed[s][rule]["low"] for s in self.seeds]),
                "mean_high": _mean([self.per_seed[s][rule]["high"] for s in self.seeds]),
                "min_low": min(self.per_seed[s][rule]["low"] for s in self.seeds),
                "max_high": max(self.per_seed[s][rule]["high"] for s in self.seeds),
                "thesis_count": THESIS_COUNT[rule],
                "thesis_rate": THESIS_RATE[rule],
                "seeds_within_1_of_thesis": sum(
                    1 for x in xs if abs(x - THESIS_COUNT[rule]) <= 1),
                "seeds_within_2_of_thesis": sum(
                    1 for x in xs if abs(x - THESIS_COUNT[rule]) <= 2),
            }
        return out


def evaluate_seeds(data, seeds=DEFAULT_SEEDS, policy="canonical_first",
                   tol=AG.P.DEFAULT_TIE_ABS_TOL, assignment="random"):
    """Rule baselines of a WorldBData over the partition seeds (plan OQ25).

    assignment="original" uses the file's sub-set column once (the seeds
    are ignored; the result has the single "seed" None).
    """
    if assignment == "original":
        seeds = (None,)
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one partition seed is needed")
    per_seed = {}
    n = None
    for seed in seeds:
        subsets = WP.subset_cases(data, seed=0 if seed is None else seed,
                                  assignment=assignment)
        per_seed[seed] = evaluate_subsets(subsets, policy, tol)
        if n is None:
            n = len(subsets)
        elif n != len(subsets):
            raise AssertionError("the number of sub-sets depends on the seed")
    return RuleRun(seeds=seeds, per_seed=per_seed, n_subsets=n,
                   synthetic=data.synthetic, marker=data.marker,
                   assignment=assignment, policy=policy, tol=float(tol))


# ---- rendering -------------------------------------------------------------------------

def _run_label(run):
    return "run (synthetic)" if run.synthetic else "run (real file)"


def summary_rows(run):
    """Rows of the T24 table: rule, the run's columns, the thesis columns.
    The run and the thesis are kept in separate columns; no difference
    column is given for synthetic data (SYNTHETIC — not a reproduction)."""
    s = run.summary()
    rows = []
    for rule in RULES_B:
        c = s[rule]
        rows.append({
            "rule": rule,
            "run_mean_count": c["mean_count"], "run_sd_count": c["sd_count"],
            "run_min_count": c["min_count"], "run_max_count": c["max_count"],
            "run_mean_rate": c["mean_rate"], "run_sd_rate": c["sd_rate"],
            "run_mean_ties": c["mean_ties"],
            "run_mean_low": c["mean_low"], "run_mean_high": c["mean_high"],
            "thesis_count": c["thesis_count"], "thesis_rate": c["thesis_rate"],
        })
    return rows


def format_summary(run, title="World-B rule baselines on the sub-sets (T24)"):
    """Markdown table with separate "run" and "thesis" columns."""
    label = _run_label(run)
    lines = []
    if run.synthetic:
        lines += [f"**{SYNTHETIC_MARKER}**", ""]
    lines += [f"### {title}" + (f" — {SYNTHETIC_MARKER}" if run.synthetic else ""), ""]
    n_seeds = len(run.seeds)
    lines += [
        f"Sub-sets per partition: {run.n_subsets}. Partition: {run.assignment}"
        + (f", {n_seeds} seeds ({run.seeds[0]}..{run.seeds[-1]})"
           if run.assignment == "random" else "")
        + f". Tie policy: {run.policy} (tolerance {run.tol:g}).",
        "",
        f"| Rule | {label}: mean count ± SD | {label}: min–max | "
        f"{label}: mean rate ± SD | {label}: tied sub-sets (mean) | "
        f"{label}: mean count if every tie went wrong – right | "
        f"thesis: count of 77 | thesis: rate (TH Fig 5.3) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary_rows(run):
        lines.append(
            f"| {r['rule']} | {r['run_mean_count']:.2f} ± {r['run_sd_count']:.2f} "
            f"| {r['run_min_count']}–{r['run_max_count']} "
            f"| {r['run_mean_rate']:.3f} ± {r['run_sd_rate']:.3f} "
            f"| {r['run_mean_ties']:.2f} "
            f"| {r['run_mean_low']:.2f} – {r['run_mean_high']:.2f} "
            f"| {r['thesis_count']} | {r['thesis_rate']:.3f} |")
    lines.append("")
    lines.append("SD = sample standard deviation over the partition seeds "
                 "(ddof 1). Thesis counts are the only integers of 77 that "
                 "round to the printed rates (plan §2.3).")
    if run.synthetic:
        lines += ["",
                  "The synthetic generator was calibrated on exactly these four "
                  "rule counts over exactly these partition seeds (plan §2.3). "
                  "Agreement between the two column groups is therefore BY "
                  "CONSTRUCTION; it is not a reproduction of T24 and says "
                  "nothing about the thesis data.",
                  "", GENERATOR_SEED_NOTE]
    else:
        lines += ["", f"Loader status: {REAL_DATA_STATUS}."]
    return "\n".join(lines) + "\n"


def per_seed_rows(run):
    """One row per seed: the four counts, ties and tie ranges."""
    rows = []
    for seed in run.seeds:
        row = {"partition_seed": "original" if seed is None else seed,
               "n_subsets": run.per_seed[seed]["n"]}
        for rule in RULES_B:
            cell = run.per_seed[seed][rule]
            row[f"{rule}_correct"] = cell["correct"]
            row[f"{rule}_ties"] = cell["ties"]
            row[f"{rule}_low"] = cell["low"]
            row[f"{rule}_high"] = cell["high"]
        rows.append(row)
    return rows
