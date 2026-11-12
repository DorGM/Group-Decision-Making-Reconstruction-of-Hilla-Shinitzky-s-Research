"""B9 (plan §9.3): significance tests of RP / AP against the rules, shape
of TH Table 5.11.

The first tests need no data: they recompute the printed Table 5.11 from
the published counts with cdm.eval.stats (plan §5.8 items 2 and 4). The
driver test runs on the synthetic stand-in (UNTESTED ON REAL DATA).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import b_small
from cdm.data.worldB import loader as L
from cdm.eval import stats as ST
from cdm.eval import worldB_ml as ML
from cdm.eval import worldB_stats as WS
from cdm.eval import worldB_thesis_values as T
from test_b6_rp import _marked

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MARK = L.SYNTHETIC_MARKER


def _as_printed(p, printed):
    """Does p, rounded the way the cell is printed, equal the cell?"""
    if "E" in printed:
        mant, exp = f"{p:.1e}".split("e")
        return f"{mant}E{int(exp):+03d}" == printed
    return f"{p:.3f}" == printed


def test_table_5_11_is_reproduced_from_the_published_counts():
    # plan §5.8 item 2 (proportion, pooled, one-tailed, n = 77) and item 4
    # (McNemar chi-square with continuity correction)
    rec = WS.thesis_recomputed()
    for rule in T.TABLE_5_11_ORDER:
        for approach in ("RP", "AP"):
            mc_printed, prop_printed = T.TABLE_5_11[rule][approach]
            cell = rec[rule][approach]
            assert _as_printed(cell["proportion"], prop_printed), (rule, approach)
            assert all(_as_printed(p, mc_printed) for p in cell["mcnemar"]), (rule, approach)
    assert WS.fmt_p(rec["MR"]["RP"]["mcnemar"][0]) == "3.0E-04"
    assert WS.fmt_p(rec["HAC"]["RP"]["proportion"]) == "6.4E-06"


def test_the_pairs_are_consistent_with_the_success_counts():
    for (approach, rule), pairs in WS.PLAN_PAIRS.items():
        for b, c in pairs:
            assert b - c == WS.PLAN_COUNTS[approach] - WS.PLAN_COUNTS[rule]
            assert b <= WS.PLAN_COUNTS[approach] and c <= WS.PLAN_COUNTS[rule]
    assert T.ALPHA == 0.05


def test_compare_on_a_hand_example():
    a = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    r = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
    out = WS.compare(a, r)
    assert (out["n"], out["x_approach"], out["x_rule"], out["b"], out["c"]) == (10, 6, 2, 5, 1)
    assert out["mcnemar_chi2"] == pytest.approx((abs(5 - 1) - 1) ** 2 / 6)
    assert out["mcnemar_p"] == pytest.approx(ST.mcnemar_statistic(5, 1)[1])
    z, p = ST.two_proportion_ztest(6, 10, 2, 10, alternative="larger")
    assert out["z"] == pytest.approx(z) and out["proportion_p"] == pytest.approx(p)
    with pytest.raises(ValueError):
        WS.compare(a, r[:5])


def test_undefined_tests_are_reported_as_undefined_not_as_a_p_value():
    same = np.array([1, 0, 1, 0], dtype=bool)
    out = WS.compare(same, same)
    assert out["b"] == out["c"] == 0 and out["mcnemar_p"] is None     # b + c = 0
    assert out["proportion_p"] == pytest.approx(0.5)
    out = WS.compare(np.ones(4, dtype=bool), np.ones(4, dtype=bool))
    assert out["mcnemar_p"] is None and out["proportion_p"] is None   # pooled SE = 0
    assert WS.fmt_p(None) == "undefined"


def test_table_over_partitions_summarises_without_pooling():
    rng = np.random.default_rng(0)
    rules = [{r: rng.random(77) < 0.4 for r in WS.RULES} for _ in range(5)]
    rp_out = [rng.random(77) < 0.7 for _ in range(5)]
    tab = WS.table({"RP": rp_out}, rules)
    assert set(tab) == set(WS.RULES) and set(tab["SP"]) == {"RP"}
    cell = tab["MR"]["RP"]
    ps = [c["proportion_p"] for c in cell["per_partition"]]
    assert cell["proportion"]["median"] == sorted(ps)[2]
    assert cell["proportion"]["min"] == min(ps) and cell["proportion"]["n_defined"] == 5
    assert cell["proportion"]["n_below_alpha"] == sum(p < 0.05 for p in ps)
    assert cell["n_approach_ahead"] == 5
    text = WS.format_table(tab, True, MARK, "partition text.", ML.wrap)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert MARK in lines[0] and MARK in lines[-1]
    assert "| not run | not run " in text                        # the AP columns
    assert "run (synthetic): vs RP, McNemar" in text and "thesis: vs AP, proportion" in text
    with pytest.raises(ValueError):
        WS.table({"RP": rp_out[:2]}, rules)


def test_b9_driver(tmp_path):
    import run_b6_rp as B6
    import run_b7_ap as B7
    import run_b9_stats as B9
    cfg = b_small.write_config(tmp_path, b_small.small_synthetic())
    common = ["--config", str(cfg), "--runs-root", str(tmp_path / "runs"), "--seeds", "2",
              "--n-jobs", "2"]
    with pytest.raises(SystemExit, match="run scripts/run_b6_rp.py"):
        B9.main(common)
    assert B6.main(common) == 0 and B7.main(common) == 0 and B9.main(common) == 0
    out = tmp_path / "runs" / "worldB_synthetic" / "b9_stats_seeds2"
    names = sorted(p.name for p in out.iterdir())
    assert names == ["b9_stats.log", "stats_per_seed.csv", "stats_summary.json",
                     "stats_table_5_11.md"]
    for p in out.iterdir():
        assert _marked(p), p.name
    doc = json.loads((out / "stats_summary.json").read_text(encoding="utf-8"))
    assert set(doc["run"]) == {"SP", "MR", "WC", "HAC"}
    assert set(doc["run"]["SP"]) == {"RP", "AP"}
    assert len(doc["run"]["SP"]["RP"]["per_partition"]) == 2
    assert doc["thesis"]["table_5_11"]["HAC"]["RP"] == ["4.9E-05", "6.4E-06"]
    rows = (out / "stats_per_seed.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 1 + 2 * 2 * 4
    md = (out / "stats_table_5_11.md").read_text(encoding="utf-8")
    assert "NOT a reproduction of Table 5.11" in md
    assert [ln.split("|")[1].strip() for ln in md.splitlines()
            if ln.startswith(("| SP", "| MR", "| WC", "| HAC"))][:4] == ["SP", "MR", "WC", "HAC"]
