"""Step A16: ablation experiments (TH Tables 5.6, 5.7).

SYNTHETIC DATA ONLY: counts, a small synthetic instance table, and the
driver run with `--smoke`. No build is read.

Tested:
  * the table: rows, thesis rates, two-tailed test against row 1 (equal to
    statsmodels), significance = p <= 0.05;
  * the thesis's significance pattern follows from its printed rates
    (AMP rows 8 and 11 significantly below row 1, no DAP row; plan §9.3
    row A16, amendment AM2: the pattern, not the p-values, is the target);
  * `run_row` on synthetic cases: the methods and columns of a row, DA
    recomputed, row 1 = the plain LOO of step A13;
  * the driver: smoke run, resume from the checkpoints, refusal of an
    unlisted build.
"""
import json
import subprocess
import sys

import numpy as np
import pytest
from statsmodels.stats.proportion import proportions_ztest

from cdm.approaches import instances as I
from cdm.config import CODE_ROOT
from cdm.eval import ablation as A
from cdm.eval import loo as L

SCRIPTS = CODE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
N = 1209


def thesis_counts(approach):
    """Success counts implied by the printed rates (row 1: the exact T19
    counts 975 / 982)."""
    counts = {row: round(A.thesis_rate(approach, row) * N) for row in A.ROWS}
    counts[1] = {"AMP": 975, "DAP": 982}[approach]
    return counts


# ---- the table -----------------------------------------------------------------------

def test_rows_and_thesis_rates():
    assert A.ROWS == tuple(range(1, 12))
    assert [A.thesis_rate("AMP", r) for r in A.ROWS] == [
        .806, .800, .816, .795, .808, .797, .791, .754, .806, .799, .763]      # T14
    assert [A.thesis_rate("DAP", r) for r in A.ROWS] == [
        .812, .812, .825, .803, .804, .805, .806, .813, .811, .815, .798]      # T15
    assert A.TECHNIQUE == {"AMP": ("BR", "RF"), "DAP": "RF"}


def test_two_tailed_test_equals_statsmodels():
    z, p = A.versus_full(912, N, 975)
    z_sm, p_sm = proportions_ztest([912, 975], [N, N], alternative="two-sided")
    assert z == pytest.approx(z_sm) and p == pytest.approx(p_sm)
    assert z < 0
    assert A.versus_full(975, N, 975) == (0.0, 1.0)
    z2, p2 = A.versus_full(975, N, 912)
    assert z2 == pytest.approx(-z) and p2 == pytest.approx(p)
    assert all(np.isnan(x) for x in A.versus_full(10, 10, 10))        # pooled SE = 0


@pytest.mark.parametrize("approach,significant", [("AMP", [8, 11]), ("DAP", [])])
def test_thesis_pattern_follows_from_printed_rates(approach, significant):
    table = A.ablation_table(approach, thesis_counts(approach), N)
    assert [r["row"] for r in table] == list(A.ROWS)
    assert table[0]["p"] is None and table[0]["significant"] is None
    assert [r["row"] for r in table if r["significant"]] == significant
    assert all(r["rounded"] == r["thesis_rate"] for r in table)
    assert all(r["difference"] == 0 for r in table)
    pattern = A.significance_pattern(table)
    assert pattern["same_pattern"] is True
    assert pattern["significantly_below_full_run"] == significant
    assert pattern["significant_rows_thesis"] == significant
    if approach == "AMP":           # every non-significant row is far from 0.05
        assert min(r["p"] for r in table if r["row"] not in (1, 8, 11)) > 0.3
        assert max(r["p"] for r in table if r["row"] in (8, 11)) < 0.02


def test_table_with_missing_rows():
    table = A.ablation_table("AMP", {8: 900, 11: 920}, N)          # no row 1 yet
    assert all(r["p"] is None for r in table)
    assert A.significance_pattern(table)["same_pattern"] is None
    table = A.ablation_table("AMP", {1: 975, 8: 912}, N)
    assert table[1]["significant"] and table[1]["below_full"]
    assert A.significance_pattern(table)["same_pattern"] is None    # rows missing
    # a row significantly ABOVE the full version is significant but not "below"
    table = A.ablation_table("AMP", {1: 900, 3: 975}, N)
    assert table[1]["significant"] and not table[1]["below_full"]


def test_significance_threshold_is_p_at_most_005():
    table = A.ablation_table("DAP", {1: 600, 2: 551}, N)
    assert 0.04 < table[1]["p"] < 0.05 and table[1]["significant"]
    table = A.ablation_table("DAP", {1: 600, 2: 552}, N)
    assert 0.05 < table[1]["p"] < 0.06 and table[1]["significant"] is False
    assert A.ablation_table("DAP", {1: 600, 2: 552}, N, alpha=0.06)[1]["significant"]
    assert table[1]["thesis_significant"] is False


# ---- runs on synthetic cases -----------------------------------------------------------

@pytest.fixture(scope="module")
def smoke():
    import _common as C
    inst, _ = C.smoke_instances(40)
    return inst, C.hyper_for_seed(0, smoke=True)


def test_run_row_1_is_the_plain_loo(smoke):
    inst, hyper = smoke
    res, desc = A.run_row(inst, "AMP", 1, hyper=hyper)
    inp = L.amp_inputs(inst)
    plain = L.run_amp_loo(inp["X"], inp["Y"], labels=inp["labels"], hyper=hyper)
    assert np.array_equal(res.success, plain.success)
    assert np.array_equal(res.proba, plain.proba)
    assert desc["methods"] == ["MR", "HAC", "WC", "SP", "DA"]
    assert desc["n_cdmcr_features"] == 27 and desc["n_columns"] == 27
    assert res.protocol == "loo" and set(res.techniques) == {"BR+RF"}


def test_run_row_components(smoke):
    inst, hyper = smoke
    res8, d8 = A.run_row(inst, "AMP", 8, hyper=hyper)              # no DA label
    assert res8.labels == ("MR", "HAC", "WC", "SP") and d8["n_columns"] == 27
    res11, d11 = A.run_row(inst, "AMP", 11, hyper=hyper)           # votes + MR + DA only
    assert res11.labels == ("MR", "DA") and d11["n_columns"] == 9
    assert d11["cdmcr_feature_numbers"] == [1, 2, 3, 4, 5, 18, 19, 20, 21]
    assert d11["rules_heard_by_DA"] == ["MR"]
    # with MR alone, DA = "not MR" (TH p.48): exactly one of the two is right
    assert np.array_equal(res11.outcomes.sum(axis=1), np.ones(len(inst), dtype=int))
    dap11, dd = A.run_row(inst, "DAP", 11, hyper=hyper)
    assert dd["n_columns"] == 9 + 2 and dap11.labels == ("MR", "DA")
    assert set(dap11.techniques) == {"RF"}
    dap9, d9 = A.run_row(inst, "DAP", 9, hyper=hyper)              # no Conf, no WC/HAC
    assert d9["methods"] == ["MR", "SP", "DA"] and d9["rules_heard_by_DA"] == ["MR", "SP"]
    assert not set(d9["cdmcr_feature_numbers"]) & {6, 7, 8, 9, 10, 11, 12, 16, 17}
    with pytest.raises(ValueError):
        A.run_row(inst, "RP", 1, hyper=hyper)
    with pytest.raises(ValueError):
        A.run_row(inst, "AMP", 12, hyper=hyper)


def test_oq41_switch_reaches_the_row(smoke):
    inst, hyper = smoke
    _, default = A.run_row(inst, "DAP", 2, hyper=hyper)
    _, alt = A.run_row(inst, "DAP", 2, hyper=hyper, oq41=I.OQ41_OPTIONS[1])
    assert 16 not in default["cdmcr_feature_numbers"]
    assert {16, 17} <= set(alt["cdmcr_feature_numbers"])
    assert alt["n_columns"] == default["n_columns"] + 2


# ---- the driver -----------------------------------------------------------------------------

def run_script(*args):
    return subprocess.run([sys.executable, str(SCRIPTS / "run_a16_ablation.py"), *args],
                          capture_output=True, text=True, cwd=str(SCRIPTS))


def test_driver_smoke_and_resume(tmp_path):
    args = ["--smoke", "30", "--rows", "1,11", "--n-jobs", "1", "--runs-root", str(tmp_path),
            "--results-root", str(tmp_path / "results")]
    p = run_script(*args)
    assert p.returncode == 0, p.stderr
    out = tmp_path / "_smoke" / "a16"
    doc = json.loads((out / "a16_summary.json").read_text())
    assert doc["complete"] and doc["header"]["data"]["kind"].startswith("SYNTHETIC")
    for a in ("AMP", "DAP"):
        table = doc[a]["table"]
        assert [r["row"] for r in table] == [1, 11]
        assert table[0]["p"] is None and table[1]["p"] is not None
        assert doc[a]["pattern"]["same_pattern"] is None             # rows missing
        assert doc[a]["rows"]["1"]["n_from_checkpoint"] == 0
        assert (out / f"a16_{a.lower()}_row11_cases.csv").is_file()
        assert set(doc[a]["domains"]["1"]) == {"General", "NBA", "NFL", "Politics", "all"}
    text = (out / "a16_ablation.md").read_text()
    assert "NOT A RESULT" in text and "## AMP (TH Table 5.6)" in text
    assert not (tmp_path / "results").exists()
    first = {a: [r["successes"] for r in doc[a]["table"]] for a in ("AMP", "DAP")}
    p = run_script(*args)                                           # resume
    assert p.returncode == 0, p.stderr
    doc = json.loads((out / "a16_summary.json").read_text())
    for a in ("AMP", "DAP"):
        assert [r["successes"] for r in doc[a]["table"]] == first[a]
        assert doc[a]["rows"]["11"]["n_from_checkpoint"] == 30


def test_driver_refuses_unlisted_build(tmp_path):
    p = run_script("--build", "h1209_nba_games", "--runs-root", str(tmp_path),
                   "--results-root", "")
    assert p.returncode != 0 and "GateError" in p.stderr
    assert not list(tmp_path.rglob("ckpt"))
