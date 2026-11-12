"""Driver scripts of A13 / A14 and the rule-gate guard.

SYNTHETIC DATA ONLY in this file: the scripts are run with `--smoke`
(synthetic cases; no build is read), and the guard is tested on small
fake build folders in a temporary directory.

Tested:
  * cdm.eval.gate_guard.require_gate: only a build listed as passed or
    waived in the gate record, with the frozen cases file, is accepted;
  * config/gate_status.yaml lists exactly the two waived builds, and
    docs/RULE_BASELINE_CHECK.md is in the code tree;
  * a driver script refuses an unlisted build before anything is fitted;
  * both scripts run end to end on synthetic cases, write their files,
    resume from the checkpoints, and report ties / convergence counts;
  * the thesis tables used by the scripts equal the tables checked
    against the rendered thesis page (tests/test_a14_nested_cv.py).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from cdm.config import CODE_ROOT
from cdm.eval import gate_guard as G
from cdm.report import thesis_values as T

SCRIPTS = CODE_ROOT / "scripts"


def fake_build(root, name, text='{"case": 1}\n', scenario=None):
    d = root / name
    d.mkdir(parents=True)
    (d / G.CASES_FILE).write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    (d / G.MANIFEST_FILE).write_text(json.dumps(
        {"scenario": scenario or name, "files": {G.CASES_FILE: sha}}))
    return d, sha


def gate_file(path, **builds):
    path.write_text(yaml.safe_dump({"builds": builds}))
    return path


def record(sha, status="waived"):
    return {"status": status, "date": "2026-11-11", "decided_by": "project owner",
            "reference": "docs/RULE_BASELINE_CHECK.md", "cases_sha256": sha}


# ---- the guard -------------------------------------------------------------------

def test_guard_accepts_only_listed_frozen_builds(tmp_path):
    d, sha = fake_build(tmp_path, "b_ok")
    gf = gate_file(tmp_path / "gate.yaml", b_ok=record(sha))
    rec = G.require_gate(d, gf)
    assert rec["build"] == "b_ok" and rec["status"] == "waived"
    assert rec["cases_sha256"] == sha and rec["reference"] == "docs/RULE_BASELINE_CHECK.md"
    assert G.require_gate(d, gate_file(tmp_path / "g2.yaml",
                                       b_ok=record(sha, "passed")))["status"] == "passed"
    # not listed
    other, sha2 = fake_build(tmp_path, "b_other")
    with pytest.raises(G.GateError, match="not listed"):
        G.require_gate(other, gf)
    # listed with another status
    for status in ("failed", "pending", None, "hypothesis"):
        with pytest.raises(G.GateError, match="status"):
            G.require_gate(d, gate_file(tmp_path / "g3.yaml", b_ok=record(sha, status)))
    # an empty record file
    (tmp_path / "empty.yaml").write_text("")
    with pytest.raises(G.GateError, match="not listed"):
        G.require_gate(d, tmp_path / "empty.yaml")


def test_guard_refuses_changed_or_misnamed_builds(tmp_path):
    d, sha = fake_build(tmp_path, "b_ok")
    gf = gate_file(tmp_path / "gate.yaml", b_ok=record(sha))
    # the folder name must be the manifest's scenario (a copy under a waived name fails)
    wrong, _ = fake_build(tmp_path, "b_copy", scenario="b_ok")
    with pytest.raises(G.GateError, match="does not match the folder"):
        G.require_gate(wrong, gf)
    # the record covers the frozen bytes: another cases file under the same name fails
    with pytest.raises(G.GateError, match="not the frozen file"):
        G.require_gate(d, gate_file(tmp_path / "g2.yaml", b_ok=record("0" * 64)))
    # cases file edited after the freeze (manifest hash no longer true)
    (d / G.CASES_FILE).write_text('{"case": 2}\n')
    with pytest.raises(G.GateError, match="hash in its manifest"):
        G.require_gate(d, gf)
    # incomplete record, missing reference document, missing manifest
    rec = record(sha)
    del rec["date"]
    d2, sha2 = fake_build(tmp_path, "b2")
    with pytest.raises(G.GateError, match="date"):
        G.require_gate(d2, gate_file(tmp_path / "g3.yaml", b2=dict(rec, cases_sha256=sha2)))
    with pytest.raises(G.GateError, match="reference"):
        G.require_gate(d2, gate_file(tmp_path / "g4.yaml",
                                     b2=dict(record(sha2), reference="docs/none.md")))
    (tmp_path / "nothing").mkdir()
    with pytest.raises(G.GateError, match="manifest"):
        G.require_gate(tmp_path / "nothing", gf)


def test_gate_status_file_lists_exactly_the_waived_builds():
    builds = G.load_gate_status()
    assert sorted(builds) == ["waiver_main", "waiver_politics_alt"]
    assert builds["waiver_main"]["role"] == "primary"
    assert builds["waiver_politics_alt"]["role"] == "sensitivity"
    for rec in builds.values():
        assert rec["status"] == "waived" and str(rec["date"]) == "2026-11-11"
        assert rec["decided_by"] == "project owner" and rec["amendment"] == "AM17"
        assert (CODE_ROOT / rec["reference"]).is_file()
        assert len(rec["cases_sha256"]) == 64
    text = (CODE_ROOT / "docs" / "RULE_BASELINE_CHECK.md").read_text(encoding="utf-8")
    assert "waiver_main" in text and "waiver_politics_alt" in text and "project owner" in text


def test_guard_refuses_a_real_named_build_that_is_not_waived(tmp_path):
    """With the real record file: a build called like an unwaived scenario
    is refused, and a fake folder named `waiver_main` is refused too (its
    bytes are not the frozen ones)."""
    d, _ = fake_build(tmp_path, "plan_default")
    with pytest.raises(G.GateError, match="not listed"):
        G.require_gate(d)
    w, _ = fake_build(tmp_path, "waiver_main")
    with pytest.raises(G.GateError, match="not the frozen file"):
        G.require_gate(w)


# ---- the scripts -----------------------------------------------------------------

def run_script(name, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, args)],
                          capture_output=True, text=True, cwd=str(SCRIPTS))


@pytest.mark.parametrize("script", ["run_a13_loo.py", "run_a14_nested.py"])
def test_scripts_refuse_an_unlisted_build(tmp_path, script):
    fake_build(tmp_path / "builds", "plan_default")
    r = run_script(script, "--build", "plan_default", "--build-root", tmp_path / "builds",
                   "--runs-root", tmp_path / "runs", "--results-root", "")
    assert r.returncode != 0
    assert "GateError" in r.stderr and "not listed" in r.stderr
    assert not list((tmp_path / "runs").rglob("batch_*"))        # nothing was fitted
    assert not list((tmp_path / "runs").rglob("*.csv"))


def test_a13_script_smoke_and_resume(tmp_path):
    args = ("--smoke", 48, "--n-jobs", 2, "--seed", 3, "--runs-root", tmp_path / "runs",
            "--results-root", tmp_path / "results")
    r = run_script("run_a13_loo.py", *args)
    assert r.returncode == 0, r.stderr[-3000:]
    out = tmp_path / "runs" / "_smoke" / "a13_seed3"
    s = json.loads((out / "summary.json").read_text())
    assert s["complete"] and s["header"]["data"]["kind"].startswith("SYNTHETIC")
    assert s["header"]["hyper"]["seed"] == 3 and s["header"]["hyper"]["lr_max_iter"] == 10000
    for a in ("AMP", "DAP"):
        assert s[a]["n"] == 48 and 0 <= s[a]["successes"] <= 48
        assert s[a]["nan_in_outputs"] == 0 and s[a]["lr_not_converged"] == 0
        assert s[a]["thesis_successes"] is None           # no thesis value beside a smoke run
        assert set(s[a]["domains"]) == {"General", "NBA", "NFL", "Politics", "all"}
        assert set(s[a]["versus_rules"]) == {"SP", "MR", "WC", "HAC", "DA"}
    sel = s["AMP"]["selection"]
    assert list(sel) == ["SP", "MR", "WC", "HAC", "DA"]
    assert sum(r_["n_selected"] for r_ in sel.values()) == 48 == s["AMP"]["selection_total"]
    assert sum(r_["n_selected_success"] for r_ in sel.values()) == s["AMP"]["successes"]
    assert s["AMP"]["technique"] == "BR+RF" and s["DAP"]["technique"] == "RF"
    assert s["AMP"]["n_features"] == 27 and s["DAP"]["n_features"] == 32
    rows = (out / "amp_loo_cases.csv").read_text().splitlines()
    assert len(rows) == 49 and rows[0].startswith("case_id,domain,success,selected,answer,P_MR")
    assert len((out / "dap_loo_cases.csv").read_text().splitlines()) == 49
    assert "NOT A RESULT" in (out / "summary.txt").read_text()
    assert not (tmp_path / "results").exists()            # a smoke run is never copied
    # resume: everything comes from the checkpoints, same numbers
    r2 = run_script("run_a13_loo.py", *args)
    assert r2.returncode == 0, r2.stderr[-3000:]
    s2 = json.loads((out / "summary.json").read_text())
    for a in ("AMP", "DAP"):
        assert s2[a]["n_from_checkpoint"] == 48 and s[a]["n_from_checkpoint"] == 0
        assert s2[a]["successes"] == s[a]["successes"]
        assert s2[a].get("selection") == s[a].get("selection")


def test_a14_script_smoke_and_resume(tmp_path):
    args = ("--smoke", 40, "--n-jobs", 2, "--n-splits", 4, "--approaches", "DAP",
            "--runs-root", tmp_path / "runs", "--results-root", tmp_path / "results")
    r = run_script("run_a14_nested.py", *args)
    assert r.returncode == 0, r.stderr[-3000:]
    res = tmp_path / "runs" / "_smoke" / "a14" / "results"
    names = {p.name for p in res.iterdir()}
    assert {"progress.log", "status.json", "dap_nested_partial.json",
            "dap_nested_table.csv", "dap_nested_table.txt", "dap_nested_summary.json",
            "dap_nested_summary.txt", "dap_nested_cases.csv"} <= names
    st = json.loads((res / "status.json").read_text())
    assert st["state"] == "finished" and st["done"] == ["DAP"] and st["not_a_result"]
    s = json.loads((res / "dap_nested_summary.json").read_text())
    assert s["complete"] and s["techniques"] == ["KNN", "LR", "RF"] and s["n_splits"] == 4
    assert s["technique_tie_rule"] == "first_in_table_order"
    assert len(s["folds"]) == 4 and len(s["selected_by_fold"]) == 4
    assert sum(f["test_successes"] for f in s["folds"]) == s["result"]["successes"]
    assert sum(f["n_test"] for f in s["folds"]) == 40 == s["result"]["n"]
    for f in s["folds"]:
        assert f["inner_successes"][f["selected"]] == max(f["inner_successes"].values())
        assert set(f["test_successes_by_tied_technique"]) == set(f["tied"])
        assert set(f["inner_not_converged"]) == {"KNN", "LR", "RF"}
        assert sum(f["inner_fresh_fits"].values()) == 3 * f["n_train"]
    lo, hi = s["successes_range_over_tie_resolutions"]
    assert lo <= s["result"]["successes"] <= hi
    assert set(s["not_converged"]) == {"KNN", "LR", "RF", "test_refits"}
    p = json.loads((res / "dap_nested_partial.json").read_text())
    assert p["folds_done"] == 4 and p["test_cases_so_far"] == 40
    assert len((res / "dap_nested_cases.csv").read_text().splitlines()) == 41
    assert not (tmp_path / "results").exists()
    # resume: no fresh inner fit, same result
    r2 = run_script("run_a14_nested.py", *args)
    assert r2.returncode == 0, r2.stderr[-3000:]
    s2 = json.loads((res / "dap_nested_summary.json").read_text())
    assert all(sum(f["inner_fresh_fits"].values()) == 0 for f in s2["folds"])
    assert s2["selected_by_fold"] == s["selected_by_fold"]
    assert s2["result"]["successes"] == s["result"]["successes"]


def test_copy_results_copies_small_files_and_never_raises(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    try:
        import _common as C
    finally:
        sys.path.remove(str(SCRIPTS))
    a, b = tmp_path / "a.json", tmp_path / "b.txt"
    a.write_text("{}")
    b.write_text("x")
    dest = tmp_path / "mount" / "results" / "build" / "a14"
    C.copy_results([a, b, tmp_path / "missing.txt"], dest)
    assert (dest / "a.json").read_text() == "{}" and (dest / "b.txt").read_text() == "x"
    C.copy_results([a], None)                              # disabled: nothing happens
    blocked = tmp_path / "file_not_dir"
    blocked.write_text("")
    C.copy_results([a], blocked / "sub")                   # cannot be created: logged only
    # write_json turns NaN into null and numpy types into plain ones
    import numpy as np
    C.write_json(tmp_path / "x.json", {"a": float("nan"), "b": np.int64(3),
                                       "c": np.array([1.5, np.nan]), 4: (1, 2)})
    assert json.loads((tmp_path / "x.json").read_text()) == {
        "a": None, "b": 3, "c": [1.5, None], "4": [1, 2]}


# ---- the thesis values shown beside the results -----------------------------------------

def test_thesis_values_equal_the_checked_tables_and_close():
    import test_a14_nested_cv as A14
    assert [tuple(r) for r in T.TH_TABLE_5_1] == [tuple(r) for r in A14.TH_TABLE_5_1]
    assert [tuple(r) for r in T.TH_TABLE_5_2] == [tuple(r) for r in A14.TH_TABLE_5_2]
    assert T.AMP_TECHNIQUES == A14.AMP_NAMES
    # plan T19-T21, T12: the counts close
    assert sum(T.LOO_AMP_SELECTED.values()) == 1209
    assert sum(T.LOO_AMP_SELECTED_SUCCESS.values()) == T.LOO_SUCCESSES["AMP"] == 975
    assert sum(T.LOO_DOMAIN_SUCCESSES["AMP"].values()) == 975
    assert sum(T.LOO_DOMAIN_SUCCESSES["DAP"].values()) == T.LOO_SUCCESSES["DAP"] == 982
    for a in ("AMP", "DAP"):                               # T22: b - c = approach - rule
        for m, (b, c) in T.LOO_DISCORDANT[a].items():
            assert b - c == T.LOO_SUCCESSES[a] - T.RULES_THESIS[m]
    # plan T1, T8, T10
    assert sum(T.NESTED_AMP_SELECTED.values()) == 1209
    assert sum(T.NESTED_AMP_SELECTED_SUCCESS.values()) == T.NESTED_SUCCESSES["AMP"] == 935
    assert sum(T.NESTED_AMP_DOMAIN_SUCCESSES.values()) == 935
    assert sum(T.DOMAIN_N.values()) == T.N_CASES == 1209
