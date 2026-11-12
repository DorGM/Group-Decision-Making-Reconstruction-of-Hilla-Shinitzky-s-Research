"""Step A15: AMP effectiveness tables (TH Tables 5.4, 5.5, A.9, A.10).

SYNTHETIC DATA ONLY: hand-built results, results built to carry the
thesis's back-solved counts, and the driver run with `--smoke`. No build is
read and no model is fitted.

Tested:
  * the thesis-reported tables: every printed cell of Tables 5.4, 5.5 and
    A.10 is reproduced from the plan's counts (T3, T3b, T8, T11, T13) with
    round-half-up; PRE from exact counts; the printed p-values (Table 5.4,
    MLA Fig 2); the count identities between the tables;
  * `effectiveness_row`: rounding half up, undefined cells, bad counts;
  * `effectiveness_table` on a result that carries the thesis counts
    gives the printed tables exactly, and the qualitative statements of
    plan §9.3 row A15 (HAC the only non-significant row, NBA the strongest
    PRE, negative PREs in NFL);
  * the per-case CSV round trip used by the driver;
  * the driver: smoke run, missing inputs, refusal of an unlisted build.
"""
import json
import subprocess
import sys

import numpy as np
import pytest

from cdm.config import CODE_ROOT
from cdm.eval import effectiveness as E
from cdm.eval.loo import EvalResult
from cdm.report import tables as R
from cdm.report import thesis_values as T

SCRIPTS = CODE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
LABELS = ("MR", "HAC", "WC", "SP", "DA")


# ---- thesis-reported values ------------------------------------------------------------------

@pytest.mark.parametrize("protocol,printed", [("nested_cv", T.TH_TABLE_5_5),
                                              ("loo", T.TH_TABLE_A_10)])
def test_domain_tables_reproduced_from_counts(protocol, printed):
    table, _ = E.thesis_effectiveness(protocol)
    for d in T.DOMAINS:
        for j, m in enumerate(T.METHODS):
            for k in E.RATE_KEYS:
                assert table[d][m]["rounded"][k] == printed[d][k][j], (d, m, k)
                assert table[d][m]["printed"][k] == printed[d][k][j]


def test_table_5_4_reproduced_from_counts():
    table, _ = E.thesis_effectiveness("nested_cv")
    for j, m in enumerate(T.METHODS):
        for k in E.RATE_KEYS:
            assert table[E.ALL][m]["rounded"][k] == T.TH_TABLE_5_4[k][j], (m, k)
    # printed p-values to two significant digits; DA is printed as 0 (AM1)
    for m, printed in zip(T.METHODS, T.TH_TABLE_5_4["p"]):
        row = table[E.ALL][m]
        if m == "DA":
            assert row["p"] < 1e-20 and row["p_one_minus_cdf"] == 0.0
        else:
            assert float(f"{row['p']:.2g}") == pytest.approx(printed, rel=0.03)


def test_mla_loo_global_block():
    table, _ = E.thesis_effectiveness("loo")
    for j, m in enumerate(T.METHODS):
        row = table[E.ALL][m]
        assert row["rounded"]["p_success_given_selected"] == \
            T.MLA_LOO_PRINTED["p_success_given_selected"][j]
        assert round(100 * row["p_selected"]) == T.MLA_LOO_PRINTED["p_selected_percent"][j]
        if m != "DA":
            assert float(f"{row['p']:.2g}") == pytest.approx(T.MLA_LOO_PRINTED["p"][j], rel=0.03)


def test_count_identities():
    for sel, ok, g_sel, g_ok, total in (
            (T.NESTED_AMP_DOMAIN_SELECTED, T.NESTED_AMP_DOMAIN_SELECTED_SUCCESS,
             T.NESTED_AMP_SELECTED, T.NESTED_AMP_SELECTED_SUCCESS, T.NESTED_AMP_DOMAIN_SUCCESSES),
            (T.LOO_AMP_DOMAIN_SELECTED, T.LOO_AMP_DOMAIN_SELECTED_SUCCESS,
             T.LOO_AMP_SELECTED, T.LOO_AMP_SELECTED_SUCCESS, T.LOO_DOMAIN_SUCCESSES["AMP"])):
        for m in T.METHODS:
            assert sum(sel[d][m] for d in T.DOMAINS) == g_sel[m]
            assert sum(ok[d][m] for d in T.DOMAINS) == g_ok[m]
        for d in T.DOMAINS:
            assert sum(sel[d].values()) == T.DOMAIN_N[d]
            assert sum(ok[d].values()) == total[d]
    for m in T.METHODS:
        assert sum(T.RULES_THESIS_DOMAIN[d][m] for d in T.DOMAINS) == T.RULES_THESIS[m]
    from cdm.eval.rule_gate import THESIS_T3B
    assert {d: {m: THESIS_T3B[m][d] for m in T.METHODS} for d in T.DOMAINS} == \
        T.RULES_THESIS_DOMAIN


def test_success_by_domain_targets():
    for a in ("AMP", "DAP"):
        th = E.thesis_success_by_domain(a, "loo")
        for d in T.DOMAINS:
            assert th[d]["rounded"] == T.TH_TABLE_A_9[a][d]
        assert th[E.ALL]["successes"] == sum(th[d]["successes"] for d in T.DOMAINS)
    th = E.thesis_success_by_domain("AMP", "nested_cv")
    assert [th[d]["rounded"] for d in T.DOMAINS] == [0.824, 0.919, 0.472, 0.596]
    assert list(E.thesis_success_by_domain("DAP", "nested_cv")) == [E.ALL]


# ---- one row ----------------------------------------------------------------------------

def test_row_rounds_half_up_and_uses_exact_counts():
    row = E.effectiveness_row(104, 16, 84, 13)             # TH Table 5.5, Politics SP
    assert row["rounded"]["p_success_given_selected"] == 0.813      # 0.8125 -> half up
    assert row["rounded"]["pre"] == 0.025
    assert row["pre"] == pytest.approx(1 - (3 / 16) / (20 / 104))
    row = E.effectiveness_row(1209, 349, 864, 287)         # TH Table 5.4, SP
    assert row["rounded"] == {"p_selected": 0.289, "p_success": 0.715,
                              "p_success_given_selected": 0.822, "pre": 0.377}
    assert float(f"{row['p']:.2g}") == 4.2e-6


def test_row_undefined_cells():
    never = E.effectiveness_row(50, 0, 20, 0)
    assert never["p_success_given_selected"] is None and never["pre"] is None
    assert never["p"] is None and never["rounded"]["pre"] is None
    always_right = E.effectiveness_row(50, 10, 50, 10)     # E1 = 0
    assert always_right["pre"] is None and always_right["p"] is None
    assert always_right["rounded"]["p_success_given_selected"] == 1.0
    never_right = E.effectiveness_row(50, 10, 0, 0)        # p0 = 0: no test, PRE defined
    assert never_right["p"] is None and never_right["pre"] == 0.0
    perfect = E.effectiveness_row(104, 8, 83, 8)           # Politics WC, 10-fold
    assert perfect["rounded"]["pre"] == 1.0


@pytest.mark.parametrize("counts", [(10, 11, 5, 5), (10, 5, 11, 5), (10, 5, 5, 6),
                                    (10, 5, 3, 4), (0, 0, 0, 0)])
def test_row_rejects_inconsistent_counts(counts):
    with pytest.raises(ValueError):
        E.effectiveness_row(*counts)


# ---- a result that carries the thesis counts -------------------------------------------------

def result_with_counts(protocol):
    """An AMP EvalResult whose selection and outcome counts are exactly the
    thesis's (per domain); the order of the cases inside a domain is
    arbitrary."""
    if protocol == "nested_cv":
        sel, ok = T.NESTED_AMP_DOMAIN_SELECTED, T.NESTED_AMP_DOMAIN_SELECTED_SUCCESS
    else:
        sel, ok = T.LOO_AMP_DOMAIN_SELECTED, T.LOO_AMP_DOMAIN_SELECTED_SUCCESS
    selected, outcomes, domains = [], [], []
    for d in T.DOMAINS:
        n = T.DOMAIN_N[d]
        s = np.concatenate([np.full(sel[d][m], LABELS.index(m)) for m in LABELS])
        O = np.zeros((n, len(LABELS)), dtype=int)
        for j, m in enumerate(LABELS):
            mine = np.flatnonzero(s == j)
            rest = np.flatnonzero(s != j)
            O[mine[:ok[d][m]], j] = 1
            O[rest[:T.RULES_THESIS_DOMAIN[d][m] - ok[d][m]], j] = 1
        selected.append(s)
        outcomes.append(O)
        domains += [d] * n
    selected, outcomes = np.concatenate(selected), np.vstack(outcomes)
    n = len(selected)
    return EvalResult(approach="AMP", protocol=protocol, techniques=["BR+RF"] * n,
                      case_ids=list(range(n)), domains=domains,
                      success=outcomes[np.arange(n), selected], labels=LABELS,
                      outcomes=outcomes, selected=selected,
                      proba=np.eye(len(LABELS))[selected])


@pytest.mark.parametrize("protocol", ["nested_cv", "loo"])
def test_table_from_result_equals_thesis(protocol):
    res = result_with_counts(protocol)
    table = E.effectiveness_table(res)
    thesis, sources = E.thesis_effectiveness(protocol)
    assert list(table) == [E.ALL] + list(T.DOMAINS)
    assert list(table[E.ALL]) == list(T.METHODS)            # thesis column order
    for block in table:
        for m in T.METHODS:
            for k in ("n", "n_selected", "n_success", "n_selected_success", "rounded", "p"):
                assert table[block][m][k] == thesis[block][m][k], (block, m, k)
    cells = E.compare(table, thesis)
    assert cells and all(c["difference"] in (0, None) for c in cells)
    assert all(c["within_tolerance"] for c in cells if c.get("within_tolerance") is not None)
    by_dom = E.success_by_domain(res)
    th = E.thesis_success_by_domain("AMP", protocol)
    assert {d: by_dom[d]["successes"] for d in T.DOMAINS} == \
        {d: th[d]["successes"] for d in T.DOMAINS}
    assert set(sources) == set(table)


def test_qualitative_checks_on_thesis_counts():
    q = E.qualitative_checks(E.effectiveness_table(result_with_counts("nested_cv")))
    assert q["non_significant_methods"] == ["HAC"]          # TH p.67; plan A15
    assert q["strongest_pre_domain"] == "NBA"               # TH p.70
    assert q["negative_pre"]["NFL"] == ["SP", "MR", "HAC", "DA"]
    assert q["negative_pre"]["NBA"] == [] and q["negative_pre"]["General"] == []
    q = E.qualitative_checks(E.thesis_effectiveness("loo")[0])
    assert q["strongest_pre_domain"] == "NBA"
    assert q["non_significant_methods"] == []


def test_table_of_small_result_by_hand():
    #            selected  O_MR O_HAC
    rows = [("A", 0, 1, 0), ("A", 0, 0, 1), ("A", 1, 1, 1), ("B", 1, 0, 0), ("B", 1, 1, 1)]
    sel = np.array([r[1] for r in rows])
    O = np.array([r[2:] for r in rows])
    res = EvalResult(approach="AMP", protocol="loo", techniques=["BR+RF"] * 5,
                     case_ids=list("abcde"), domains=[r[0] for r in rows],
                     success=O[np.arange(5), sel], labels=("MR", "HAC"), outcomes=O,
                     selected=sel, proba=np.eye(2)[sel])
    t = E.effectiveness_table(res)
    assert list(t) == [E.ALL, "A", "B"] and list(t["A"]) == ["MR", "HAC"]
    a = t["A"]["MR"]
    assert (a["n"], a["n_selected"], a["n_success"], a["n_selected_success"]) == (3, 2, 2, 1)
    assert a["p_success_given_selected"] == 0.5 and a["pre"] == pytest.approx(-0.5)
    b = t["B"]["MR"]
    assert b["n_selected"] == 0 and b["pre"] is None
    g = t[E.ALL]["HAC"]
    assert (g["n_selected"], g["n_success"], g["n_selected_success"]) == (3, 3, 2)
    base = E.rule_baselines_of_result(res, with_thesis=False)
    assert base["A"]["MR"] == {"recomputed": 2, "n": 3, "thesis": None, "thesis_n": None}
    with pytest.raises(ValueError):
        E.effectiveness_table(EvalResult(approach="DAP", protocol="loo", techniques=[],
                                         case_ids=[], domains=None, success=np.array([])))


def test_rule_baselines_carry_the_thesis_counts():
    res = result_with_counts("loo")
    base = E.rule_baselines_of_result(res)
    assert base[E.ALL]["SP"] == {"recomputed": 864, "n": 1209, "thesis": 864,
                                 "thesis_n": 1209}
    assert base["NFL"]["HAC"]["thesis"] == 131 and base["NFL"]["HAC"]["thesis_n"] == 212
    text = R.rule_baselines_md(base)
    assert "| NBA | 393 | 258 (0.656) / 258 (0.656) / +0 |" in text


# ---- rendering ---------------------------------------------------------------------------------

def test_markdown_shows_run_and_thesis_side_by_side():
    res = result_with_counts("nested_cv")
    thesis, sources = E.thesis_effectiveness("nested_cv")
    text = R.effectiveness_md("T", E.effectiveness_table(res), thesis, sources)
    assert "### NBA (n = 393; thesis n = 393)" in text
    assert "| P(success by AG given AMP selected AG) | 0.848 / 0.848 | 0.974 / 0.974 |" in text
    assert "| PRE | -0.029 / -0.029 | -0.349 / -0.349 |" in text         # NFL
    assert "4.2e-06 / 4.2e-06" in text                                    # Table 5.4 SP
    rows = R.effectiveness_csv_rows("x", E.effectiveness_table(res), thesis)
    assert len(rows) == 1 + 5 * 5 and len(set(len(r) for r in rows)) == 1
    assert R.pvalue(0.0) == "0" and R.pvalue(None) == R.DASH and R.pvalue(0.0624) == "0.062"
    assert R.rate(None) == R.DASH and R.rate(0.813) == "0.813"


# ---- CSV round trip and the driver ------------------------------------------------------------

def test_cases_csv_round_trip(tmp_path):
    import _common as C
    res = result_with_counts("loo")
    C.write_cases_csv(tmp_path / "amp.csv", res)
    back = E.eval_result_from_cases_csv(tmp_path / "amp.csv", "AMP", "loo")
    assert back.labels == LABELS and back.domains == res.domains
    assert np.array_equal(back.selected, res.selected)
    assert np.array_equal(back.outcomes, res.outcomes)
    assert E.effectiveness_table(back) == E.effectiveness_table(res)
    # a file whose success column contradicts the selection is refused
    text = (tmp_path / "amp.csv").read_text().splitlines()
    cells = text[1].split(",")
    cells[2] = str(1 - int(cells[2]))
    (tmp_path / "bad.csv").write_text("\n".join([text[0], ",".join(cells)] + text[2:]))
    with pytest.raises(ValueError, match="success"):
        E.eval_result_from_cases_csv(tmp_path / "bad.csv", "AMP", "loo")
    with pytest.raises(ValueError, match="not a per-case file"):
        E.eval_result_from_cases_csv(tmp_path / "amp.csv", "DAP", "loo")


def run_script(name, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          capture_output=True, text=True, cwd=str(SCRIPTS))


def test_driver_smoke_and_missing_inputs(tmp_path):
    p = run_script("run_a15_tables.py", "--smoke", "60", "--runs-root", str(tmp_path),
                   "--results-root", str(tmp_path / "results"))
    assert p.returncode == 0, p.stderr
    out = tmp_path / "_smoke" / "a15"
    doc = json.loads((out / "a15_summary.json").read_text())
    assert doc["not_a_result"] and doc["complete"]
    assert set(doc["sections"]) == {"nested_cv", "loo_seed0", "loo_seed1"}
    eff = doc["sections"]["nested_cv"]["effectiveness"]
    assert sum(r["n_selected"] for r in eff["all"].values()) == 60
    assert "thesis_effectiveness" not in doc["sections"]["nested_cv"]     # synthetic data
    text = (out / "a15_tables.md").read_text()
    assert "NOT A RESULT" in text and "### General (n = 15)" in text
    assert not (tmp_path / "results").exists()                # smoke is never copied
    # an input that does not exist is reported as missing, not invented
    import run_a15_tables as D
    missing = []
    assert D.read_result(tmp_path / "nope.csv", "AMP", "loo", missing) is None
    assert missing == [str(tmp_path / "nope.csv")]
    text, rows, part = D.section("loo", None, None, False, "x")
    assert rows == [] and "effectiveness" not in part


def test_driver_refuses_unlisted_build(tmp_path):
    p = run_script("run_a15_tables.py", "--build", "h1209_nba_games", "--runs-root",
                   str(tmp_path), "--results-root", "")
    assert p.returncode != 0 and "GateError" in p.stderr
    assert not (tmp_path / "h1209_nba_games" / "a15" / "a15_tables.md").exists()
