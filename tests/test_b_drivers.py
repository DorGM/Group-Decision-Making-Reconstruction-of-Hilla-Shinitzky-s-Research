"""Driver scripts of steps B3 and B5 (scripts/run_b3_synthetic.py,
scripts/run_b5_rules.py), run into a temporary folder.

The point of these tests: EVERY file the synthetic runs write carries the
marker "SYNTHETIC — not a reproduction" (plan §2.3, §9.3 row B11), and the
run and the thesis sit in separate columns.
"""
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_b3_synthetic as B3                                   # noqa: E402
import run_b5_rules as B5                                       # noqa: E402
from cdm.data.worldB import loader as L                         # noqa: E402
from cdm.data.worldB import synthetic as S                      # noqa: E402

MARK = L.SYNTHETIC_MARKER


def _marked(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        doc = json.loads(text)
        return list(doc)[0] == "_marker" and doc["_marker"] == MARK
    if path.suffix == ".csv":
        first = text.splitlines()[0]
        return first.startswith("#") and MARK in first
    if path.suffix == ".md":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return MARK in lines[0] and MARK in lines[-1]
    return MARK in text.splitlines()[0]                         # .log


def test_b3_generate_writes_marked_files(tmp_path):
    assert B3.main(["--no-copy", "--runs-root", str(tmp_path)]) == 0
    out = tmp_path / "b3_data"
    names = sorted(p.name for p in out.iterdir())
    assert names == ["b3_checks.json", "b3_report.md", "b3_table_4_1_check.csv",
                     "generate.log", "worldB_manifest.json",
                     "worldB_responses.csv", "worldB_tasks.csv"]
    for p in out.iterdir():
        assert _marked(p), p.name
    checks = json.loads((out / "b3_checks.json").read_text(encoding="utf-8"))
    assert checks["round_trip_exact"] is True
    assert checks["table_4_1_all_match"] is True
    assert checks["n_responses"] == 2490 and checks["n_subsets_over_30_seeds"] == [77]
    assert checks["subset_sizes_match_printed_averages"] is True
    assert checks["ps_sum_max_abs_deviation_from_1"] < 1e-12
    assert checks["feature_oracle_a4_like"]["n_instances_over_30_seeds"] >= 1
    # the written files are the generator's data
    assert L.load_written(out) == S.generate()
    manifest = json.loads((out / "worldB_manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameter_tags"] == {k: v["tag"] for k, v in S.PARAMETERS.items()}
    assert all(t.startswith("INVENTED") for t in manifest["parameter_tags"].values())


def test_b5_rules_writes_marked_files_with_separate_columns(tmp_path):
    assert B5.main(["--no-copy", "--runs-root", str(tmp_path)]) == 0
    out = tmp_path / "worldB_synthetic" / "b5_rules"        # never "worldB_real"
    names = sorted(p.name for p in out.iterdir())
    assert names == ["b5_rules.log", "rules_per_seed.csv", "rules_summary.json",
                     "rules_summary.md"]
    for p in out.iterdir():
        assert _marked(p), p.name
    doc = json.loads((out / "rules_summary.json").read_text(encoding="utf-8"))
    assert doc["synthetic"] is True and "BY CONSTRUCTION" in doc["note"]
    assert set(doc["run"]) == set(doc["thesis"]["count"]) == {"MR", "WC", "SP", "HAC"}
    assert doc["thesis"]["count"] == {"MR": 34, "WC": 34, "SP": 33, "HAC": 22}
    assert not any("thesis" in k for r in doc["run"].values() for k in r)
    assert "generator seed 0 only" in doc["generator_seed_note"]
    assert doc["partition"]["seeds"] == list(range(30)) and doc["partition"]["n_subsets"] == 77
    assert all(len(v) == 30 for v in doc["counts_per_seed"].values())
    md = (out / "rules_summary.md").read_text(encoding="utf-8")
    assert "run (synthetic):" in md and "thesis:" in md
    assert "generator seed 0 only" in md
    # the thesis counts are not a row of the run's per-seed table
    assert not any(ln.startswith("| thesis") for ln in md.splitlines())
    assert "thesis (count of 77; TH Fig 5.3, not part of the run): MR 34 / WC 34 / "\
        "SP 33 / HAC 22" in md
    rows = (out / "rules_per_seed.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 1 + 30


def test_first_grid_flag_needs_calibrate():
    with pytest.raises(SystemExit):
        B3.main(["--first-grid"])
