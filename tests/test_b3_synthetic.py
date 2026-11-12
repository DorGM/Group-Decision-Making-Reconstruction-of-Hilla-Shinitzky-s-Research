"""B3 acceptance (plan §9.3, §2.3): the SYNTHETIC World-B stand-in.

"Table 4.1 reproduced exactly (|A|, |R|, solver counts); 77 sub-sets with
the listed average sizes; sum ps = 1; rule rates over 30 seeds within +-2
of MR 34, WC 34, SP 33, HAC 22. Every parameter is tagged
EVIDENCE/INFERRED/INVENTED."

SYNTHETIC — not a reproduction: nothing asserted here is a result about
the thesis data. The rule rates are the generator's calibration targets.
"""
import json
import math
from fractions import Fraction

import pytest
import yaml

from cdm.data.worldB import loader as L
from cdm.data.worldB import partition as WP
from cdm.data.worldB import synthetic as S
from cdm.data.worldB import table41 as T41
from cdm.eval import worldB_rules as WR

PARAMS, DOC = S.load_parameters()
DATA = S.generate()
SEEDS = S.CALIBRATION_SEEDS


# ---- TH Table 4.1 [EVIDENCE] ----------------------------------------------------------

def test_table_4_1_constants():
    assert len(T41.TABLE_4_1) == T41.N_TASKS == 22                  # TH p.52
    assert sum(t.n_responses for t in T41.TABLE_4_1) == 2490        # plan §2.2, D9
    assert sum(T41.n_subsets(t.n_responses) for t in T41.TABLE_4_1) == 77   # TH p.55
    # The Type column as printed: 16 + 2 + 4. The text says "14 logical
    # riddles ... and eight gamified tasks" (TH p.52-53; MCF §4.1): an
    # difference between the table and the text (decision B3-14), not used by any code.
    types = [t.type for t in T41.TABLE_4_1]
    assert (types.count("Logic riddle"), types.count("NP complete"),
            types.count("P-space complete")) == (16, 2, 4)
    # the same |R_P| column as the partition test of step A5
    from test_a5_partition import TABLE_4_1 as A5
    assert [(t.name, t.n_responses, t.avg_subset_size) for t in T41.TABLE_4_1] == A5
    # plan §2.2 / D10: unweighted mean of the printed S(a*) is 30%
    assert round(sum(t.s_correct for t in T41.TABLE_4_1) / 22, 3) == 0.303


@pytest.mark.parametrize("task", T41.TABLE_4_1, ids=lambda t: t.name)
def test_solver_count_gives_the_printed_share(task):
    # plan §2.3: solver count = round(S(a*) * |R_P|)
    k = T41.n_solvers(task)
    assert k == round(task.s_correct * task.n_responses)
    fits = tuple(j for j in range(task.n_responses + 1)
                 if T41.printed_share(j, task.n_responses) == task.s_correct)
    assert k in fits
    # 18 rows: the only integer; 4 rows: the printed share allows others
    if task.name in T41.AMBIGUOUS_SOLVER_COUNTS:
        assert fits == T41.AMBIGUOUS_SOLVER_COUNTS[task.name] and len(fits) > 1
    else:
        assert fits == (k,)
    # S(a*) * |R_P| is not on a rounding half
    assert (Fraction(str(task.s_correct)) * task.n_responses) % 1 != Fraction(1, 2)


def test_solver_totals():
    assert len(T41.AMBIGUOUS_SOLVER_COUNTS) == 4
    total = sum(T41.n_solvers(t) for t in T41.TABLE_4_1)
    assert total == 697
    # response-weighted solver share 28.0% (= MCF-arXiv's 28%, plan §2.2)
    assert round(total / 2490, 3) == 0.280


@pytest.mark.parametrize("task", T41.TABLE_4_1, ids=lambda t: t.name)
def test_printed_average_subset_size_is_n_over_k(task):
    k = T41.n_subsets(task.n_responses)
    avg = Fraction(task.n_responses, k)
    assert float(Fraction(int(avg * 10 + Fraction(1, 2)), 10)) == task.avg_subset_size


# ---- the generated data ----------------------------------------------------------------

def test_marker_is_carried_by_the_data_object():
    assert L.SYNTHETIC_MARKER == "SYNTHETIC — not a reproduction"
    assert DATA.synthetic is True and DATA.marker == L.SYNTHETIC_MARKER
    assert DATA.log["status"] == L.SYNTHETIC_MARKER
    for case in DATA.tasks:
        assert case.domain == L.SYNTHETIC_MARKER
        assert case.case_id.startswith("SYNTHETIC-T")
    subs = WP.subset_cases(DATA, seed=0)
    assert subs.synthetic is True and subs.marker == L.SYNTHETIC_MARKER
    assert all(c.domain == L.SYNTHETIC_MARKER and c.case_id.startswith("SYNTHETIC-T")
               for c in subs.cases)
    # a data object cannot be built with the flag and the marker disagreeing
    with pytest.raises(ValueError):
        L.WorldBData(DATA.tasks, DATA.names, DATA.types, DATA.response_ids, None,
                     synthetic=False, marker=L.SYNTHETIC_MARKER)
    with pytest.raises(ValueError):
        L.WorldBData(DATA.tasks, DATA.names, DATA.types, DATA.response_ids, None,
                     synthetic=True, marker="")


def test_table_4_1_is_reproduced_exactly():
    rep = L.table_4_1_report(DATA)
    assert rep["all_match"] is True
    assert rep["n_tasks"] == 22 and rep["n_responses"] == 2490 and rep["n_subsets"] == 77
    assert rep["unmatched_tasks"] == [] and rep["missing_table_rows"] == []
    assert rep["oq34"].startswith("2490")
    for case, task, row in zip(DATA.tasks, T41.TABLE_4_1, rep["rows"]):
        assert len(case.answers) == task.n_answers
        assert case.n == task.n_responses
        solvers = sum(1 for r in case.responses if r.v == case.correct)
        assert solvers == T41.n_solvers(task) == row["n_solvers"]
        assert T41.printed_share(solvers, case.n) == task.s_correct


def test_77_subsets_with_the_listed_sizes_for_every_seed():
    for seed in SEEDS:
        subs = WP.subset_cases(DATA, seed=seed)
        assert len(subs) == 77
        rep = WP.partition_report(subs)
        assert rep["is_77"] and rep["sizes_within_30_45"]          # TH p.53
        by_task = {}
        for tid, case, idx in zip(subs.task_ids, subs.cases, subs.response_indices):
            by_task.setdefault(tid, []).append((case, idx))
        for task, printed in zip(DATA.tasks, T41.TABLE_4_1):
            parts = by_task[task.case_id]
            sizes = [c.n for c, _ in parts]
            assert len(sizes) == T41.n_subsets(printed.n_responses)
            assert max(sizes) - min(sizes) <= 1                    # near-equal
            # disjoint and complete (TH p.53 "without overlap")
            flat = sorted(i for _, idx in parts for i in idx)
            assert flat == list(range(task.n))
            for c, idx in parts:
                assert c.responses == tuple(task.responses[i] for i in idx)
                assert c.answers == task.answers and c.correct == task.correct
            avg = Fraction(sum(sizes), len(sizes))
            assert float(Fraction(int(avg * 10 + Fraction(1, 2)), 10)) \
                == printed.avg_subset_size


def test_predicted_supports_sum_to_one_and_values_are_in_range():
    for case in DATA.tasks:
        for r in case.responses:
            assert sum(r.ps) == pytest.approx(1.0, abs=1e-12)      # [INFERRED]
            assert all(0.0 <= p <= 1.0 for p in r.ps)
            assert 0.0 <= r.c <= 1.0                               # [INVENTED] scale
            assert len(r.ps) == len(case.answers)


def test_generator_is_deterministic_and_seeded():
    assert S.generate() == DATA
    assert S.generate(PARAMS, seed=0) == DATA
    other = S.generate(PARAMS, seed=1)
    assert other != DATA
    assert L.table_4_1_report(other)["all_match"] is True          # any seed


def test_random_streams_are_separate_per_block():
    # changing a prediction parameter leaves votes and confidences unchanged
    p = dict(PARAMS)
    p["ps_concentration"] = 5.0
    alt = S.generate(p)
    for a, b in zip(alt.tasks, DATA.tasks):
        assert [r.v for r in a.responses] == [r.v for r in b.responses]
        assert [r.c for r in a.responses] == [r.c for r in b.responses]
        assert a.correct == b.correct
    assert any(x.ps != y.ps for a, b in zip(alt.tasks, DATA.tasks)
               for x, y in zip(a.responses, b.responses))
    # changing a confidence parameter leaves votes and predictions unchanged
    p = dict(PARAMS)
    p["conf_concentration"] = 4.0
    alt = S.generate(p)
    for a, b in zip(alt.tasks, DATA.tasks):
        assert [(r.v, r.ps) for r in a.responses] == [(r.v, r.ps) for r in b.responses]


def test_layout_is_logged_and_the_lure_is_a_wrong_option():
    for lay, case in zip(DATA.log["layout"], DATA.tasks):
        assert lay["correct"] == case.correct
        assert lay["lure"] in case.answers and lay["lure"] != case.correct
        assert 0 < lay["lure_share"] < 1
        assert sum(lay["votes"].values()) == case.n


# ---- written files ----------------------------------------------------------------------

def test_written_files_carry_the_marker_and_round_trip_exactly(tmp_path):
    paths = L.write_world_b(DATA, tmp_path)
    for key in ("tasks", "responses"):
        first = paths[key].read_text(encoding="utf-8").splitlines()[0]
        assert first.startswith("#") and L.SYNTHETIC_MARKER in first
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert list(manifest)[0] == "_marker" and manifest["_marker"] == L.SYNTHETIC_MARKER
    assert manifest["synthetic"] is True
    back = L.load_written(tmp_path)
    assert back == DATA                                 # the loader schema, exactly
    assert back.synthetic is True and back.marker == L.SYNTHETIC_MARKER
    assert back.log["ps_sum_violations"] == 0
    assert back.log["n_responses"] == 2490
    # a marker removed from one file is detected
    text = paths["responses"].read_text(encoding="utf-8").split("\n", 1)[1]
    paths["responses"].write_text(text, encoding="utf-8")
    with pytest.raises(L.WorldBFormatError):
        L.load_written(tmp_path)                        # hash differs
    with pytest.raises(L.WorldBFormatError):
        L.load_written(tmp_path, verify=False)          # marker in one file only
    # ... and from both: the task ids still say SYNTHETIC
    text = paths["tasks"].read_text(encoding="utf-8").split("\n", 1)[1]
    paths["tasks"].write_text(text, encoding="utf-8")
    with pytest.raises(L.WorldBFormatError):
        L.load_world_b(paths["responses"], paths["tasks"],
                       confidence_range=(0, 1), ps_scale=1)


# ---- parameters and tags ----------------------------------------------------------------

def test_every_parameter_is_tagged():
    assert set(S.TAGS) == {"EVIDENCE", "INFERRED", "INVENTED", "INVENTED-CALIBRATED"}
    for name, spec in S.PARAMETERS.items():
        assert spec["tag"] in ("INVENTED", "INVENTED-CALIBRATED"), name
        assert ("grid" in spec) == (spec["tag"] == "INVENTED-CALIBRATED"), name
        assert ("fixed" in spec) == (spec["tag"] == "INVENTED"), name
        assert spec["block"] in ("votes", "conf", "ps") and spec["what"]
    # the plan's list of calibrated parameters (§2.3 "Rules"): lambda_t,
    # Dirichlet alpha, Beta parameters, noise and shift sizes
    calibrated = [k for k, s in S.PARAMETERS.items() if "grid" in s]
    assert calibrated == ["lure_share_mean", "distractor_alpha",
                          "conf_popularity_slope", "conf_solver_bonus",
                          "conf_concentration", "ps_own_shift",
                          "ps_solver_lure_shift", "ps_concentration"]


def test_config_file_lists_every_parameter_with_its_tag():
    text = S.PARAMETERS_PATH.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "# " + L.SYNTHETIC_MARKER
    doc = yaml.safe_load(text)
    assert doc["marker"] == L.SYNTHETIC_MARKER
    assert list(doc["parameters"]) == list(S.PARAMETERS)
    for name, entry in doc["parameters"].items():
        assert entry["tag"] == S.PARAMETERS[name]["tag"]
        assert entry["tag"].startswith("INVENTED")
        assert PARAMS[name] == entry["value"]
        if "grid" in S.PARAMETERS[name]:
            assert tuple(entry["grid"]) == S.PARAMETERS[name]["grid"]
            assert entry["value"] in entry["grid"]
    for key in ("tasks", "solvers_per_task", "partition", "ps_sum_to_one",
                "confidence_scale"):
        assert any(tag in doc["evidence"][key] for tag in S.TAGS)
    cal = doc["calibration"]
    assert cal["targets"] == {"MR": 34, "WC": 34, "SP": 33, "HAC": 22}
    assert cal["n_partition_seeds"] == 30 and cal["partition_seeds"] == [0, 29]
    assert "T23" in cal["not_calibration_targets"]
    assert DATA.log["parameters"] == PARAMS and DATA.log["seed"] == doc["seed"] == 0


def test_config_file_is_what_the_calibration_driver_writes():
    """The shipped YAML equals `parameters_document` for its own values, so
    the notes of the header (the generator-seed note, the untagged
    constants) are printed again by a re-run."""
    text = S.PARAMETERS_PATH.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    cal = doc["calibration"]
    res = {"seed": doc["seed"],
           "parameters": {k: v["value"] for k, v in doc["parameters"].items()},
           "means": cal["mean_counts_reached"], "loss": cal["loss"],
           "partition_seeds": list(range(cal["partition_seeds"][0],
                                         cal["partition_seeds"][1] + 1)),
           "stages": [{"block": b, "table": [None] * n}
                      for b, n in cal["combinations_tried"].items()]}
    assert S.parameters_document(res) == text
    comments = " ".join(ln[1:].strip() for ln in text.splitlines() if ln.startswith("#"))
    assert S.GENERATOR_SEED_NOTE in comments
    for phrase in ("generator seed 0 only", "MR 25–35, WC 26–34, SP 31–37, HAC 16–21",
                   "2 of the 8 seeds"):
        assert phrase in S.GENERATOR_SEED_NOTE
    # the invented constants that are not parameters are named and tagged
    assert f"[{S.CONF_CLIP[0]}, {S.CONF_CLIP[1]}]" in comments
    assert comments.count("[INVENTED]") >= 2 and "uniform draw" in comments
    assert "4 tasks ambiguous" in doc["evidence"]["solvers_per_task"]
    assert len(T41.AMBIGUOUS_SOLVER_COUNTS) == 4
    # the calibrated values themselves
    assert {k: v["value"] for k, v in doc["parameters"].items()} == {
        "lure_share_mean": 0.4, "lure_share_concentration": 10.0,
        "distractor_alpha": 2.0, "conf_base": 0.6, "conf_popularity_slope": 0.2,
        "conf_solver_bonus": 0.0, "conf_concentration": 16.0, "ps_own_shift": 0.3,
        "ps_solver_lure_shift": 0.5, "ps_concentration": 20.0,
        "ps_onehot_rate": 0.05, "ps_floor": 1e-06}


def test_bad_parameter_sets_are_refused(tmp_path):
    p = dict(PARAMS)
    p["ps_own_shift"], p["ps_solver_lure_shift"] = 0.6, 0.5
    with pytest.raises(ValueError):
        S.generate(p)
    p = dict(PARAMS)
    del p["conf_base"]
    with pytest.raises(ValueError):
        S.check_parameters(p)
    # a config file with a value off its grid, or without the marker
    doc = yaml.safe_load(S.PARAMETERS_PATH.read_text(encoding="utf-8"))
    doc["parameters"]["lure_share_mean"]["value"] = 0.123
    f = tmp_path / "x.yaml"
    f.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        S.load_parameters(f)
    doc = yaml.safe_load(S.PARAMETERS_PATH.read_text(encoding="utf-8"))
    doc["marker"] = "real"
    f.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        S.load_parameters(f)


# ---- calibration (plan §2.3 "Rules") ---------------------------------------------------

def test_calibration_targets_are_t24_and_nothing_else():
    assert dict(S.CALIBRATION_TARGETS) == WR.THESIS_COUNT == \
        {"MR": 34, "WC": 34, "SP": 33, "HAC": 22}
    assert S.CALIBRATION_SEEDS == tuple(range(30)) == WR.DEFAULT_SEEDS
    stages = dict(S.CALIBRATION_STAGES)
    assert sorted(r for rules in stages.values() for r in rules) == \
        sorted(S.CALIBRATION_TARGETS)
    # the module does not import or mention the RP/AP approaches' targets
    src = open(S.__file__, encoding="utf-8").read()
    for word in ("0.636", "0.545", "49/77", "42/77", "approaches"):
        assert word not in src


def test_rule_counts_over_30_seeds_are_within_2_of_the_targets():
    """Plan §9.3 row B3. BY CONSTRUCTION (the calibration targets); also
    checks that the config file records what the generator gives."""
    means = S.rule_means(PARAMS)
    recorded = DOC["calibration"]["mean_counts_reached"]
    for rule, target in S.CALIBRATION_TARGETS.items():
        assert means[rule] == pytest.approx(recorded[rule], abs=1e-9)
        assert abs(means[rule] - target) <= 2, (rule, means[rule])
    assert DOC["calibration"]["loss"] == pytest.approx(
        sum((means[r] - t) ** 2 for r, t in S.CALIBRATION_TARGETS.items()))


def test_calibrate_on_a_tiny_grid_picks_the_smallest_loss_first_in_order():
    grid = {k: (PARAMS[k],) for k, s in S.PARAMETERS.items() if "grid" in s}
    grid["lure_share_mean"] = (0.7, 0.4)
    grid["ps_solver_lure_shift"] = (0.0, 0.5)
    seen = []
    res = S.calibrate(partition_seeds=(0, 1), grid=grid,
                      progress=lambda block, row: seen.append(block))
    assert seen == ["votes", "votes", "conf", "ps", "ps"]
    for st in res["stages"]:
        best = min(r["loss"] for r in st["table"])
        assert st["chosen"]["loss"] == best
        first = [r for r in st["table"] if r["loss"] == best][0]
        assert st["chosen"] is first
    assert set(res["parameters"]) == set(S.PARAMETERS)
    assert res["partition_seeds"] == [0, 1]
    text = S.parameters_document(res)
    assert text.startswith("# " + L.SYNTHETIC_MARKER)
    assert yaml.safe_load(text)["marker"] == L.SYNTHETIC_MARKER


def test_first_grid_is_recorded():
    # decision B3-9: one extension of one grid
    grid = S.PARAMETERS["ps_solver_lure_shift"]["grid"]
    assert grid[:5] == S.FIRST_GRID_PS_SOLVER_LURE_SHIFT
    assert grid[5:] == (0.4, 0.5)


# ---- feature oracle (plan §2.3, last row) ------------------------------------------------

def test_a_generated_subset_can_look_like_tables_a4_a6():
    """"at least one generated sub-set must be able to produce A.4/A.6-like
    values (S=8/33, ps[v]=1 for a minority voter)"."""
    from cdm.features import acr, rcr
    found = []
    for seed in SEEDS:
        subs = WP.subset_cases(DATA, seed=seed)
        found += [(subs, x) for x in S.a4_like_instances(subs)]
    assert found
    subs, (case_id, answer, _) = found[0]
    case = [c for c in subs.cases if c.case_id == case_id][0]
    assert case.n == 33                       # Knapsack 2 or Pills (99, 66)
    i = case.answers.index(answer)
    rows = rcr.rcr_features(case.responses, case.answers)
    hits = [f for r, f in zip(case.responses, rows) if r.v == answer and r.ps[i] == 1.0]
    f = hits[0]
    assert f["ChosenMajAns"] == 0 and f["PSv"] == 1.0
    assert round(f["dPSv"], 3) == -0.758                           # TH Table A.4
    assert round(f["ChiSq"], 3) == 0.61                            # TH Table A.4
    a = dict(acr.acr_features(case.responses, case.answers))[answer]
    assert a["IsMajority"] == 0 and round(a["Support"], 3) == 0.242    # TH Table A.6
    assert all(math.isfinite(v) for v in f.values())
