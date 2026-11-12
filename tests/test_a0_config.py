"""A0 acceptance: the config lists every plan §8 open question by its ID."""
import numpy as np

from cdm import config as cfg
from cdm.seeding import set_seed

# The 41 open questions of plan §8, plus OQ1b (sub-question) and OQ31 (resolved).
EXPECTED_IDS = (
    {f"OQ{i}" for i in range(1, 20)}
    | {f"OQ{i}" for i in (21, 22, 23, 25, 26, 27, 28)}
    | {f"OQ{i}" for i in range(30, 45)}
    | {"OQ1b", "OQ8b"}
)


def test_every_open_question_is_listed():
    oq = cfg.load_config()["open_questions"]
    assert set(oq) == EXPECTED_IDS
    assert len(EXPECTED_IDS - {"OQ1b", "OQ31"}) == 41  # plan §8 count
    for key, entry in oq.items():
        assert "default" in entry and "plan" in entry, key


def test_selected_defaults_match_plan():
    assert cfg.get_switch("OQ1") == "rutchick_s1"
    assert cfg.get_switch("OQ2")["likert"] == "a_c_minus_1_over_4"
    assert cfg.get_switch("OQ4")["min_n"] is None
    assert cfg.get_switch("OQ6")["subset_size"] == 30
    assert cfg.get_switch("OQ7")["ties"] == "canonical_first"
    assert cfg.get_switch("OQ8") == "canonical_first"
    assert cfg.get_switch("OQ9") == "population_sd"
    assert cfg.get_switch("OQ10") == 2
    assert cfg.get_switch("OQ16")["domain_order"] == ["General", "NBA", "NFL", "Politics"]
    assert cfg.load_config()["open_questions"]["OQ17"]["label_order"] == ["MR", "HAC", "WC", "SP", "DA"]
    assert cfg.get_switch("OQ22")["z"] == 1.645
    assert cfg.load_config()["seed"] == 0


def test_data_root_default_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv("CDM_DATA_ROOT", raising=False)
    assert cfg.data_root() == (cfg.CODE_ROOT / "data").resolve()
    monkeypatch.setenv("CDM_DATA_ROOT", str(tmp_path))
    assert cfg.data_root() == tmp_path.resolve()


def test_seed_is_reproducible():
    a = set_seed(0).random(3)
    b = set_seed(0).random(3)
    assert np.array_equal(a, b)
    set_seed(5)
    x = np.random.rand()
    set_seed(5)
    assert x == np.random.rand()


# ---- tie tolerance switch --------------------

def test_f2_lr_max_iter_is_a_number_not_a_text_placeholder():
    # plan §8 OQ15 gives no number; amendment AM18 (a default of this
    # implementation) sets 10000. Never a string.
    v = cfg.get_switch("OQ15")["lr_max_iter"]
    assert type(v) is int and v == 10000


def test_technique_tie_rule_am18():
    # AM18: the nested-CV technique tie rule is a config entry
    assert cfg.load_config()["technique_tie_rule"] == "first_in_table_order"


def test_tie_abs_tol_switch_am3():
    """Amendment AM3/AM9: the tolerance is a config entry
    and equals the code default; 0 means exact comparison."""
    import pytest
    from cdm import primitives as P
    assert cfg.load_config()["tie_abs_tol"] == 1e-9
    assert cfg.tie_abs_tol() == 1e-9 == P.DEFAULT_TIE_ABS_TOL
    assert cfg.tie_abs_tol({"tie_abs_tol": 0}) == 0.0
    for bad in ("1e-9", -1e-9, None, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cfg.tie_abs_tol({"tie_abs_tol": bad})
    with pytest.raises(ValueError):
        cfg.tie_abs_tol({})


def test_f1_requirements_pin_matches_installed_statsmodels():
    import statsmodels
    pins = dict(line.strip().split("==") for line in
                (cfg.CODE_ROOT / "requirements.txt").read_text().split() if "==" in line)
    assert pins["statsmodels"] == statsmodels.__version__
