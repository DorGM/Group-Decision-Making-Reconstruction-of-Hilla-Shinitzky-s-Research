"""Small World-B data sets for the tests of steps B6-B11.

`small_synthetic()` : the four smallest tasks of the synthetic stand-in
    (plan §2.3), still marked "SYNTHETIC — not a reproduction".
`small_unmarked()`  : the same responses under made-up task ids, without
    the marker: a FABRICATED stand-in for "a real file" in the assumed
    layout. It exercises the `source: real` path of the drivers; it says
    nothing about the real MTurk file (UNTESTED ON REAL DATA).
`write_config(...)` : writes the data in the loader's layout and a
    config/worldB.yaml-style file that points at it.
"""
import dataclasses

import yaml

from cdm.data.worldB import loader as L
from cdm.data.worldB import synthetic as S
from cdm.schema import Case

N_TASKS = 4


def small_synthetic(n_tasks=N_TASKS):
    data = S.generate()
    order = sorted(range(len(data.tasks)), key=lambda t: (data.tasks[t].n, t))[:n_tasks]
    order.sort()
    pick = lambda xs: tuple(xs[t] for t in order)               # noqa: E731
    return dataclasses.replace(data, tasks=pick(data.tasks), names=pick(data.names),
                               types=pick(data.types),
                               response_ids=pick(data.response_ids))


def small_unmarked(n_tasks=N_TASKS):
    data = small_synthetic(n_tasks)
    tasks = tuple(Case(case_id=f"fab{t + 1}", answers=c.answers, correct=c.correct,
                       responses=c.responses, domain=None)
                  for t, c in enumerate(data.tasks))
    ids = tuple(tuple(f"r{i}" for i in range(1, c.n + 1)) for c in tasks)
    return L.WorldBData(tasks=tasks, names=tuple(f"Fabricated {t + 1}" for t in range(n_tasks)),
                        types=(None,) * n_tasks, response_ids=ids, original_subsets=None,
                        synthetic=False, marker="",
                        log={"status": L.REAL_DATA_STATUS, "fabricated": True})


def write_config(tmp_path, data):
    """Write `data` and a source file pointing at it; returns its path."""
    paths = L.write_world_b(data, tmp_path / "data")
    cfg = {"source": "real",
           "real": {"status": "FABRICATED TEST FILE - " + L.REAL_DATA_STATUS,
                    "responses_file": str(paths["responses"]),
                    "tasks_file": str(paths["tasks"]),
                    "confidence_range": [0, 1], "ps_scale": 1, "incomplete": "raise"}}
    f = tmp_path / "worldB_test.yaml"
    f.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return f
