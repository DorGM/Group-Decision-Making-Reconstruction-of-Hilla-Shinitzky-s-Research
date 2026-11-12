"""World-B sub-sets: every task's responses split into disjoint sub-sets,
each treated as a separate case (TH §4.2.2 p.53-55; plan §2.2, §9.3 row B4).

    UNTESTED ON REAL DATA (see cdm/data/worldB/loader.py).

Evidence:
  [EVIDENCE]  sub-sets are sampled "without overlap" (TH p.53), "there are
              no overlaps in the responses between the sub-sets" (p.55);
              "sub-sets of 30-45 responses (according to |R_P|)"; 77
              sub-sets in total; "randomly sampled" (TH p.75).
  [INFERRED, strong]  k = floor(|R_P| / 30) near-equal sub-sets that use
              every response: sum k = 77 and every "Avg. Subset Size" of TH
              Table 4.1 equals |R_P| / k (plan §2.2).

The split itself is cdm.partition.partition (the same function as the
CDMCR sub-sets, plan §3.3.1: "very likely one shared function").

Choices the plan leaves open (IMPLEMENTATION_DECISIONS, World B):
  * seed of a task's split = first 8 bytes of
    SHA-256("<seed>|worldB_partition|<task_id>") (the M1 / T13 pattern);
    plan §8 OQ25 asks for 30 seeds: seeds 0..29;
  * a task with fewer than 60 responses does not occur in Table 4.1
    (minimum 66). 30 <= n < 60 gives k = 1 (the whole task is one sub-set),
    n < 30 raises: the two-halves rule of TH p.27 belongs to the CDMCR
    features, not to this sampling;
  * `assignment="original"` uses the file's `subset_id` column (plan §8
    OQ25: use the original assignment if it is available).

A sub-set keeps the task's full answer list A_P (answers without a vote in
the sub-set included) and the task's correct answer. Its case_id is
"<task_id>#sub<j>" (j from 1); `task_ids` gives the group of every sub-set
for the leave-one-group-out evaluation of plan §5.7.
"""
import hashlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from cdm.data.worldB import table41 as T41
from cdm.data.worldB.loader import SYNTHETIC_MARKER
from cdm.partition import partition
from cdm.schema import Case

ASSIGNMENTS = ("random", "original")


@dataclass(frozen=True)
class WorldBSubsets:
    """The sub-set cases of one partition.

    cases            : one Case per sub-set, tasks in data order.
    task_ids         : the task of every sub-set.
    response_indices : per sub-set, the positions of its responses in the
                       task's Case.responses.
    seed             : the global partition seed (None for "original").
    assignment       : "random" or "original".
    synthetic, marker: copied from the data.
    """
    cases: Tuple[Case, ...]
    task_ids: Tuple[str, ...]
    response_indices: Tuple[Tuple[int, ...], ...]
    seed: object
    assignment: str
    synthetic: bool
    marker: str

    def __post_init__(self):
        if not (len(self.cases) == len(self.task_ids) == len(self.response_indices)):
            raise ValueError("one task id and one index list per sub-set")
        if self.synthetic != (self.marker == SYNTHETIC_MARKER):
            raise ValueError("synthetic sub-sets must carry SYNTHETIC_MARKER")

    def __len__(self):
        return len(self.cases)

    def sizes(self):
        return [c.n for c in self.cases]


def task_partition_seed(seed, task_id):
    """Seed of one task's split, derived from the global seed (an int)."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"the partition seed must be an integer, got {seed!r}")
    digest = hashlib.sha256(f"{int(seed)}|worldB_partition|{task_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def n_subsets(n_responses):
    """k = floor(|R_P| / 30) [INFERRED, plan §2.2]. Raises for n < 30
    (no evidence for such a task)."""
    k = T41.n_subsets(n_responses)
    if k < 1:
        raise ValueError(
            f"a task with {n_responses} responses gives no sub-set under "
            "k = floor(|R_P| / 30); TH Table 4.1 has no such task and the "
            "plan gives no rule")
    return k


def task_index_sets(n_responses, seed, task_id):
    """The random split of one task: a list of k lists of positions."""
    k = n_subsets(n_responses)
    parts = partition(n_responses, task_partition_seed(seed, task_id),
                      subset_size=T41.SUBSET_SIZE,
                      scheme="a_disjoint_floor_n_over_30",
                      halves_if_k_below_2=False)
    if len(parts) != k:
        raise AssertionError(f"expected {k} sub-sets, got {len(parts)}")
    # positions in ascending order: a sub-set is a set; the order of its
    # responses follows the task's order
    return [sorted(p) for p in parts]


def _original_index_sets(labels):
    order = []
    groups = {}
    for i, lab in enumerate(labels):
        if lab not in groups:
            groups[lab] = []
            order.append(lab)
        groups[lab].append(i)
    return [groups[lab] for lab in order]


def subset_cases(data, seed=0, assignment="random"):
    """Split every task of a WorldBData into its sub-set cases.

    assignment="random"   : seeded random split (plan §8 OQ25), k =
                            floor(|R_P| / 30) near-equal disjoint sub-sets.
    assignment="original" : the `subset_id` column of the file; `seed` is
                            ignored and stored as None. The sub-sets of a
                            task are taken in the order in which their
                            labels first appear.
    """
    if assignment not in ASSIGNMENTS:
        raise ValueError(f"assignment must be one of {ASSIGNMENTS}")
    if assignment == "original" and data.original_subsets is None:
        raise ValueError("the data has no original sub-set assignment "
                         "(no subset_id column)")
    cases, task_ids, indices = [], [], []
    for t, task in enumerate(data.tasks):
        if assignment == "original":
            parts = _original_index_sets(data.original_subsets[t])
        else:
            parts = task_index_sets(task.n, seed, task.case_id)
        flat = sorted(i for p in parts for i in p)
        if flat != list(range(task.n)):
            raise AssertionError(f"task {task.case_id!r}: the sub-sets are not "
                                 "a disjoint cover of the responses")
        for j, part in enumerate(parts, start=1):
            cases.append(Case(case_id=f"{task.case_id}#sub{j}",
                              answers=task.answers, correct=task.correct,
                              responses=tuple(task.responses[i] for i in part),
                              domain=task.domain))
            task_ids.append(task.case_id)
            indices.append(tuple(part))
    return WorldBSubsets(cases=tuple(cases), task_ids=tuple(task_ids),
                         response_indices=tuple(indices),
                         seed=None if assignment == "original" else int(seed),
                         assignment=assignment, synthetic=data.synthetic,
                         marker=data.marker)


def partition_report(subsets):
    """Counts for the log: number of sub-sets, sizes, and the comparison
    with TH p.53-55 (77 sub-sets of 30-45 responses)."""
    sizes = subsets.sizes()
    return {"n_subsets": len(subsets), "min_size": min(sizes),
            "max_size": max(sizes),
            "is_77": len(subsets) == T41.N_SUBSETS,
            "sizes_within_30_45": all(30 <= s <= 45 for s in sizes),
            "assignment": subsets.assignment, "seed": subsets.seed}
