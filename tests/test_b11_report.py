"""B11 (plan §9.3): the World-B report — every thesis-reported item T23-T30 with the
thesis value, the run value and whether the comparison is meaningful;
synthetic runs labelled "SYNTHETIC — not a reproduction".

    UNTESTED ON REAL DATA.
"""
import json
import sys
from pathlib import Path

import pytest

import b_small
from cdm.data.worldB import loader as L

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MARK = L.SYNTHETIC_MARKER


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    import run_b6_rp as B6
    import run_b7_ap as B7
    import run_b8_alt as B8
    import run_b9_stats as B9
    import run_b10_shap as B10
    import run_b11_report as B11
    tmp = tmp_path_factory.mktemp("b11")
    cfg = b_small.write_config(tmp, b_small.small_synthetic())
    common = ["--config", str(cfg), "--runs-root", str(tmp / "runs"), "--n-jobs", "2"]
    # before any step: the report still builds and says "not run"
    assert B11.main(common + ["--seeds", "2"]) == 0
    out = tmp / "runs" / "worldB_synthetic" / "b11_report_seeds2"
    empty = (out / "results_worldB_synthetic.md").read_text(encoding="utf-8")
    for mod in (B6, B7, B8, B9):
        assert mod.main(common + ["--seeds", "2"]) == 0
    assert B10.main(common + ["--kernel-rows", "5", "--kernel-background", "8",
                              "--kernel-nsamples", "50"]) == 0
    assert B11.main(common + ["--seeds", "2"]) == 0
    text = (out / "results_worldB_synthetic.md").read_text(encoding="utf-8")
    doc = json.loads((out / "report_summary.json").read_text(encoding="utf-8"))
    return empty, text, doc


def test_report_says_at_the_top_what_it_is_and_is_not(report):
    _, text, _ = report
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert MARK in lines[0] and MARK in lines[-1]
    head = "\n".join(lines[:12])
    assert "runs end to end" in head
    assert "carry NO information about the thesis's RP / AP results" in head
    assert "UNTESTED ON REAL DATA" in head
    assert "No difference between a run value and a thesis value is computed" in head
    # every section heading carries the marker
    for ln in lines:
        if ln.startswith("## "):
            assert MARK in ln, ln


def test_every_world_b_target_is_in_the_side_by_side_table(report):
    _, text, doc = report
    targets = [r["item"] for r in doc["items"]]
    assert set(targets) == {"T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30"}
    assert targets.count("T23") == 2 and targets.count("T24") == 4
    assert targets.count("T26") == 9 and targets.count("T28") == 3
    assert targets.count("T29") == 8 + 1 and targets.count("T30") == 2
    head = [ln for ln in text.splitlines() if ln.startswith("| Item |")][0]
    assert "thesis" in head and "run (synthetic)" in head and "meaningful" in head
    for r in doc["items"]:
        assert r["run"] != "not run", r["quantity"]
        if "no data involved" in r["meaningful"]:
            assert r["meaningful"].startswith("YES")
            assert r["run"].startswith("all 16 printed values are matched after rounding")
        else:
            # on synthetic data no comparison is meaningful
            assert r["meaningful"].startswith("NO"), r
    by_construction = [r for r in doc["items"] if r["item"] == "T24"]
    assert all("BY CONSTRUCTION" in r["meaningful"] for r in by_construction)
    assert [r["thesis"] for r in doc["items"] if r["item"] == "T23"] == [
        "0.636 (49/77)", "0.545 (42/77)"]


def test_report_says_near_the_top_that_synthetic_rates_are_not_comparable(report):
    # decision D3
    import run_b11_report as B11
    _, text, _ = report
    top = text.split("## 1.")[0]
    assert B11.SYNTHETIC_RATES_NOTE in top
    for phrase in ("solvers are easy to identify", "AUC of about 0.83",
                   "NOT comparable with the thesis's 0.636 / 0.545",
                   "generator seed 0 only"):
        assert phrase in top, phrase


def test_report_covers_the_special_cases_and_tie_rules(report):
    _, text, _ = report
    for phrase in ("Empty out-group", "decision B1-2", "B6-4", "only voted answer",
                   "Correct answer without a vote", "still tied after the tie breaker",
                   "seeded random choice", "three-case rule vs argmax P(True)",
                   "Leave-one-group-out: a group is the set of instances of one sub-set",
                   "not comparable with the PRE reported in the thesis",
                   "source: real"):
        assert phrase in text, phrase
    assert "Spearman" not in text                               # no SHAP agreement claim
    for word in ("reproduces the thesis", "close to the thesis", "matches the thesis"):
        assert word not in text


def test_report_before_the_steps_says_not_run(report):
    empty, _, _ = report
    assert empty.count("not run") > 20 and MARK in empty.splitlines()[0]
