"""B4 (plan §9.3): World-B loader for the real file + partition.

    UNTESTED ON REAL DATA. The MTurk file has not been supplied. These
    tests use a tiny FABRICATED fixture written in the layout the loader
    assumes (cdm/data/worldB/loader.py; decision World B, L1-L12). They
    show that the loader does what its documentation says on that layout;
    they say nothing about the real file.

Plan row B4: "Same schema as B3; Table 4.1 per-task stats; 2490 or 2514
rows logged (OQ34)".
"""
import pytest
import yaml

from cdm.data.worldB import loader as L
from cdm.data.worldB import partition as WP
from cdm.data.worldB import table41 as T41
from cdm.schema import Case, Response

TASKS = """task_id,name,type,answers,correct
t1,Apple,Logic riddle,A|B|C,B
t2,Made-up task,,yes|no,yes
"""

# confidence on a fabricated raw scale 1..5; predictions in percent
RESPONSES = """task_id,response_id,vote,confidence,ps_1,ps_2,ps_3
t1,r1,A,5,60,30,10
t1,r2,B,1,20,70,10
t1,r3,B,3,45,55,0
t2,r1,yes,2,80,20,
t2,r2,no,4,50,50,
"""


def write(tmp_path, tasks=TASKS, responses=RESPONSES):
    t, r = tmp_path / "tasks.csv", tmp_path / "responses.csv"
    t.write_text(tasks, encoding="utf-8")
    r.write_text(responses, encoding="utf-8")
    return r, t


def load(tmp_path, tasks=TASKS, responses=RESPONSES, **kw):
    r, t = write(tmp_path, tasks, responses)
    opts = dict(confidence_range=(1, 5), ps_scale=100)
    opts.update(kw)
    return L.load_world_b(r, t, **opts)


def test_status_is_stated():
    assert L.REAL_DATA_STATUS == "UNTESTED ON REAL DATA"
    assert "UNTESTED ON REAL DATA" in L.__doc__
    assert "UNTESTED ON REAL DATA" in WP.__doc__


def test_fixture_loads_into_the_schema_of_plan_3_1(tmp_path):
    data = load(tmp_path)
    assert data.synthetic is False and data.marker == ""
    assert data.log["status"] == "UNTESTED ON REAL DATA"
    assert [c.case_id for c in data.tasks] == ["t1", "t2"]
    assert data.names == ("Apple", "Made-up task") and data.types == ("Logic riddle", None)
    t1, t2 = data.tasks
    assert isinstance(t1, Case) and t1.domain is None
    assert t1.answers == ("A", "B", "C") and t1.correct == "B"
    # c = (raw - low) / (high - low); ps = percent / 100, in the order of `answers`
    assert t1.responses == (Response("A", 1.0, (0.6, 0.3, 0.1)),
                            Response("B", 0.0, (0.2, 0.7, 0.1)),
                            Response("B", 0.5, (0.45, 0.55, 0.0)))
    assert t2.answers == ("yes", "no")
    assert t2.responses == (Response("yes", 0.25, (0.8, 0.2)),
                            Response("no", 0.75, (0.5, 0.5)))
    assert data.response_ids == (("r1", "r2", "r3"), ("r1", "r2"))
    assert data.original_subsets is None
    assert data.n_responses == 5
    log = data.log
    assert log["n_rows_in_file"] == 5 and log["n_responses"] == 5 and log["n_tasks"] == 2
    assert log["ps_sum_violations"] == 0 and log["incomplete_dropped"] == 0
    assert log["confidence_range"] == [1.0, 5.0] and log["ps_scale"] == 100
    assert len(log["sha256"]["responses"]) == 64


def test_scales_must_be_stated(tmp_path):
    # the confidence scale and the prediction scale are not reported in the
    # papers: no silent default
    r, t = write(tmp_path)
    with pytest.raises(TypeError):
        L.load_world_b(r, t)
    with pytest.raises(ValueError, match="confidence_range"):
        L.load_world_b(r, t, confidence_range=None, ps_scale=100)
    with pytest.raises(ValueError, match="ps_scale"):
        L.load_world_b(r, t, confidence_range=(1, 5), ps_scale=None)
    with pytest.raises(ValueError):
        L.load_world_b(r, t, confidence_range=(5, 1), ps_scale=100)
    with pytest.raises(ValueError):
        L.load_world_b(r, t, confidence_range=(1, 5), ps_scale=10)
    with pytest.raises(ValueError):
        load(tmp_path, ps_sum_policy="clip")
    with pytest.raises(ValueError):
        load(tmp_path, incomplete="ignore")


def test_oq39_predictions_that_do_not_sum_to_100(tmp_path):
    bad = RESPONSES.replace("t2,r2,no,4,50,50,", "t2,r2,no,4,60,30,")
    # default: renormalize if violated (plan §8 OQ39)
    data = load(tmp_path, responses=bad)
    assert data.tasks[1].responses[1].ps == pytest.approx((2 / 3, 1 / 3))
    assert data.log["ps_sum_violations"] == 1
    assert data.log["ps_sum_max_deviation"] == pytest.approx(0.1)
    # untouched responses stay exactly as given
    assert data.tasks[0].responses[0].ps == (0.6, 0.3, 0.1)
    # sensitivity: raw
    raw = load(tmp_path, responses=bad, ps_sum_policy="raw")
    assert raw.tasks[1].responses[1].ps == (0.6, 0.3)
    assert raw.log["ps_sum_violations"] == 1
    # all zeros cannot be renormalized
    zero = RESPONSES.replace("t2,r2,no,4,50,50,", "t2,r2,no,4,0,0,")
    with pytest.raises(L.WorldBFormatError, match="all predictions are 0"):
        load(tmp_path, responses=zero)
    # a total above 100% with a single value above 100 is refused by the schema
    over = RESPONSES.replace("t2,r2,no,4,50,50,", "t2,r2,no,4,150,50,")
    with pytest.raises(L.WorldBFormatError):
        load(tmp_path, responses=over, ps_sum_policy="raw")


def test_wrong_ps_scale_is_refused_not_renormalised(tmp_path):
    # decision L7: percentages read with ps_scale=1
    # used to load, every row "repaired" by the OQ39 renormalisation
    for policy in L.PS_SUM_POLICIES:
        with pytest.raises(L.WorldBFormatError, match="ps_scale=1"):
            load(tmp_path, ps_scale=1, ps_sum_policy=policy)
    # the correct scale still loads, with no violation
    data = load(tmp_path, ps_scale=100)
    assert data.log["ps_sum_violations"] == 0
    assert data.tasks[0].responses[0].ps == (0.6, 0.3, 0.1)
    # one value above 100% is refused under the default policy too (it was
    # renormalised before), whatever the row's total
    over = RESPONSES.replace("t2,r2,no,4,50,50,", "t2,r2,no,4,150,50,")
    with pytest.raises(L.WorldBFormatError, match="above 1"):
        load(tmp_path, responses=over)
    # a row that only fails to sum to 100% is still renormalised (OQ39)
    low = RESPONSES.replace("t2,r2,no,4,50,50,", "t2,r2,no,4,60,30,")
    assert load(tmp_path, responses=low).log["ps_sum_violations"] == 1


def test_synthetic_file_as_percentages_needs_the_right_scale(tmp_path):
    """The synthetic stand-in written as percentages.
    SYNTHETIC -- not a reproduction."""
    import csv
    from cdm.data.worldB import synthetic as S
    data = S.generate()
    paths = L.write_world_b(data, tmp_path / "shares")
    # round trip of the written (share) file: unchanged by the new check,
    # under the default OQ39 policy as well as under "raw"
    for policy in L.PS_SUM_POLICIES:
        back = L.load_world_b(paths["responses"], paths["tasks"],
                              confidence_range=(0, 1), ps_scale=1,
                              ps_sum_policy=policy)
        assert back == data and back.log["ps_sum_violations"] == 0
    assert L.load_written(tmp_path / "shares") == data
    # the same file with the predictions as percentages
    lines = paths["responses"].read_text(encoding="utf-8").splitlines()
    rows = list(csv.reader(lines[1:]))
    cols = [i for i, h in enumerate(rows[0]) if h.startswith("ps_")]
    for row in rows[1:]:
        for i in cols:
            if row[i] != "":
                row[i] = repr(float(row[i]) * 100)
    pct = tmp_path / "pct.csv"
    with open(pct, "w", encoding="utf-8", newline="") as fh:
        fh.write(lines[0] + "\n")
        csv.writer(fh, lineterminator="\n").writerows(rows)
    with pytest.raises(L.WorldBFormatError, match="ps_scale=1"):
        L.load_world_b(pct, paths["tasks"], confidence_range=(0, 1), ps_scale=1)
    ok = L.load_world_b(pct, paths["tasks"], confidence_range=(0, 1), ps_scale=100)
    assert ok.synthetic is True and ok.n_responses == data.n_responses
    assert ok.log["ps_sum_violations"] == 0
    for a, b in zip(ok.tasks, data.tasks):
        for ra, rb in zip(a.responses, b.responses):
            assert ra.v == rb.v and ra.c == rb.c
            assert ra.ps == pytest.approx(rb.ps, abs=1e-12)


def test_byte_order_mark_is_tolerated(tmp_path):
    # decision L1: Excel writes "UTF-8 with BOM"
    plain = load(tmp_path)
    t, r = tmp_path / "tasks_bom.csv", tmp_path / "responses_bom.csv"
    t.write_bytes(b"\xef\xbb\xbf" + TASKS.encode("utf-8"))
    r.write_bytes(b"\xef\xbb\xbf" + RESPONSES.encode("utf-8"))
    data = L.load_world_b(r, t, confidence_range=(1, 5), ps_scale=100)
    assert data == plain and data.synthetic is False
    # a marked (synthetic) file with a BOM keeps its marker
    from cdm.data.worldB import synthetic as S
    paths = L.write_world_b(S.generate(), tmp_path / "syn")
    for p in (paths["tasks"], paths["responses"]):
        p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())
    back = L.load_world_b(paths["responses"], paths["tasks"],
                          confidence_range=(0, 1), ps_scale=1)
    assert back.synthetic is True and back == S.generate()


def test_incomplete_responses(tmp_path):
    bad = RESPONSES.replace("t1,r2,B,1,20,70,10", "t1,r2,B,,20,70,10") \
                   .replace("t2,r1,yes,2,80,20,", "t2,r1,yes,2,80,,")
    with pytest.raises(L.WorldBFormatError, match="incomplete"):
        load(tmp_path, responses=bad)
    data = load(tmp_path, responses=bad, incomplete="drop")
    assert data.log["incomplete_dropped"] == 2 and data.log["n_rows_in_file"] == 5
    assert data.response_ids == (("r1", "r3"), ("r2",))
    # a task left without any response is an error
    gone = RESPONSES.replace("t2,r1,yes,2,80,20,", "t2,r1,,2,80,20,") \
                    .replace("t2,r2,no,4,50,50,", "t2,r2,,4,50,50,")
    with pytest.raises(L.WorldBFormatError, match="no response"):
        load(tmp_path, responses=gone, incomplete="drop")


@pytest.mark.parametrize("old,new", [
    ("t1,r1,A,5,60,30,10", "t9,r1,A,5,60,30,10"),          # unknown task
    ("t1,r1,A,5,60,30,10", "t1,r1,D,5,60,30,10"),          # vote not an option
    ("t1,r1,A,5,60,30,10", "t1,r1,A,6,60,30,10"),          # confidence off the scale
    ("t1,r1,A,5,60,30,10", "t1,r1,A,x,60,30,10"),          # not a number
    ("t1,r1,A,5,60,30,10", "t1,r1,A,5,-60,30,10"),         # negative prediction
    ("t1,r1,A,5,60,30,10", "t1,r2,A,5,60,30,10"),          # duplicate response id
    ("t1,r1,A,5,60,30,10", "t1,,A,5,60,30,10"),            # empty response id
    ("t2,r1,yes,2,80,20,", "t2,r1,yes,2,80,20,5"),         # prediction beyond the options
    ("t1,r1,A,5,60,30,10", "t1,r1,A,5,60,30"),             # wrong number of cells
    ("task_id,response_id,vote,confidence,ps_1,ps_2,ps_3",
     "task_id,response_id,vote,conf,ps_1,ps_2,ps_3"),      # missing column
    ("task_id,response_id,vote,confidence,ps_1,ps_2,ps_3",
     "task_id,response_id,vote,confidence,ps_1,ps_2"),     # too few prediction columns
])
def test_malformed_responses_file_is_refused(tmp_path, old, new):
    assert old in RESPONSES
    text = RESPONSES.replace(old, new)
    if new.endswith("ps_2"):                 # drop the third prediction cell too
        text = "\n".join(",".join(line.split(",")[:6]) for line in text.splitlines()) + "\n"
    with pytest.raises((L.WorldBFormatError, ValueError)):
        load(tmp_path, responses=text)


@pytest.mark.parametrize("old,new", [
    ("t1,Apple,Logic riddle,A|B|C,B", "t1,Apple,Logic riddle,A|B|C,D"),   # correct not an option
    ("t1,Apple,Logic riddle,A|B|C,B", "t1,Apple,Logic riddle,A,A"),       # one option
    ("t1,Apple,Logic riddle,A|B|C,B", "t1,Apple,Logic riddle,A|A|C,A"),   # repeated option
    ("t2,Made-up task,,yes|no,yes", "t1,Made-up task,,yes|no,yes"),       # duplicate task id
    ("task_id,name,type,answers,correct", "task_id,name,type,options,correct"),
])
def test_malformed_tasks_file_is_refused(tmp_path, old, new):
    assert old in TASKS
    with pytest.raises(L.WorldBFormatError):
        load(tmp_path, tasks=TASKS.replace(old, new))


def test_extra_prediction_columns_are_refused(tmp_path):
    text = "\n".join(line + "," for line in RESPONSES.splitlines()) + "\n"
    text = text.replace("ps_3,", "ps_3,ps_4", 1)
    with pytest.raises(L.WorldBFormatError, match="beyond"):
        load(tmp_path, responses=text)


def test_real_layout_file_must_not_say_synthetic(tmp_path):
    with pytest.raises(L.WorldBFormatError):
        load(tmp_path, tasks=TASKS.replace("t1,", "synthetic-t1,"),
             responses=RESPONSES.replace("t1,", "synthetic-t1,"))


# ---- Table 4.1 check and OQ34 ------------------------------------------------------------

def test_table_4_1_report_reports_and_does_not_raise(tmp_path):
    rep = L.table_4_1_report(load(tmp_path))
    assert rep["all_match"] is False
    assert rep["n_tasks"] == 2 and rep["n_responses"] == 5 and rep["n_subsets"] == 0
    assert rep["unmatched_tasks"] == ["Made-up task"]
    assert len(rep["missing_table_rows"]) == 21 and "Apple" not in rep["missing_table_rows"]
    apple = rep["rows"][0]
    assert apple["printed"]["n_responses"] == 96
    assert apple["n_answers"] == 3 and apple["n_solvers"] == 2
    assert (apple["match_n_answers"], apple["match_n_responses"],
            apple["match_s_correct"]) == (False, False, False)
    assert rep["rows"][1]["printed"] is None
    assert rep["oq34"].startswith("5 responses: neither 2490")


def test_oq34_note():
    assert L.oq34_note(2490).startswith("2490 responses = the sum of TH Table 4.1")
    assert L.oq34_note(2514).startswith("2514 responses = the total stated")
    assert "neither" in L.oq34_note(2500)
    assert T41.N_RESPONSES_TABLE == 2490 and T41.N_RESPONSES_TEXT == 2514


# ---- partition ---------------------------------------------------------------------------

def big_fixture(tmp_path, n1=66, n2=95, subset_col=False):
    """Two fabricated tasks with n1 and n2 responses."""
    tasks = "task_id,name,type,answers,correct\nt1,Pills,,A|B|C,B\nt2,Other,,yes|no,yes\n"
    head = "task_id,response_id,vote,confidence,ps_1,ps_2,ps_3"
    lines = [head + (",subset_id" if subset_col else "")]
    for i in range(n1):
        row = f"t1,r{i},{'ABC'[i % 3]},{1 + i % 5},50,30,20"
        lines.append(row + (f",s{i % 2}" if subset_col else ""))
    for i in range(n2):
        row = f"t2,r{i},{('yes', 'no')[i % 2]},{1 + i % 5},40,60,"
        lines.append(row + (f",g{i // 32}" if subset_col else ""))
    return load(tmp_path, tasks=tasks, responses="\n".join(lines) + "\n")


def test_random_partition_floor_n_over_30(tmp_path):
    data = big_fixture(tmp_path)
    subs = WP.subset_cases(data, seed=3)
    # 66 -> 2 x 33 ; 95 -> 3 sub-sets of 32, 32, 31   (k = floor(n / 30))
    assert [c.case_id for c in subs.cases] == \
        ["t1#sub1", "t1#sub2", "t2#sub1", "t2#sub2", "t2#sub3"]
    assert subs.task_ids == ("t1", "t1", "t2", "t2", "t2")
    assert sorted(subs.sizes()[:2]) == [33, 33] and sorted(subs.sizes()[2:]) == [31, 32, 32]
    assert subs.seed == 3 and subs.assignment == "random"
    assert subs.synthetic is False and subs.marker == ""
    for t, task in enumerate(data.tasks):
        mine = [i for i, tid in enumerate(subs.task_ids) if tid == task.case_id]
        flat = sorted(j for i in mine for j in subs.response_indices[i])
        assert flat == list(range(task.n))                       # disjoint cover
        for i in mine:
            c = subs.cases[i]
            assert c.answers == task.answers and c.correct == task.correct
            assert c.responses == tuple(task.responses[j] for j in subs.response_indices[i])
    # seeded: same seed -> same split; another seed -> another split
    assert WP.subset_cases(data, seed=3) == subs
    assert WP.subset_cases(data, seed=4).response_indices != subs.response_indices
    rep = WP.partition_report(subs)
    assert rep["n_subsets"] == 5 and rep["is_77"] is False and rep["sizes_within_30_45"]


def test_task_seed_depends_on_seed_and_task():
    a = WP.task_partition_seed(0, "t1")
    assert a == WP.task_partition_seed(0, "t1")
    assert a != WP.task_partition_seed(1, "t1") and a != WP.task_partition_seed(0, "t2")
    with pytest.raises(ValueError):
        WP.task_partition_seed("0", "t1")


def test_small_tasks(tmp_path):
    # not covered by TH Table 4.1 (minimum |R_P| = 66): 30 <= n < 60 -> one
    # sub-set; n < 30 -> refused
    assert min(t.n_responses for t in T41.TABLE_4_1) == 66
    data = big_fixture(tmp_path, n1=45, n2=60)
    subs = WP.subset_cases(data, seed=0)
    assert subs.sizes() == [45, 30, 30]
    with pytest.raises(ValueError, match="no sub-set"):
        WP.subset_cases(big_fixture(tmp_path, n1=29, n2=60), seed=0)
    assert WP.n_subsets(30) == 1 and WP.n_subsets(59) == 1 and WP.n_subsets(60) == 2


def test_original_assignment_from_the_file(tmp_path):
    data = big_fixture(tmp_path, subset_col=True)
    assert data.log["has_original_subsets"] is True
    assert data.original_subsets[0][:3] == ("s0", "s1", "s0")
    subs = WP.subset_cases(data, assignment="original")
    assert subs.seed is None and subs.assignment == "original"
    assert subs.sizes() == [33, 33, 32, 32, 31]
    assert subs.response_indices[0] == tuple(range(0, 66, 2))
    assert subs.response_indices[2] == tuple(range(32))
    # without the column the original assignment cannot be used
    with pytest.raises(ValueError, match="no original"):
        WP.subset_cases(big_fixture(tmp_path), assignment="original")
    with pytest.raises(ValueError):
        WP.subset_cases(data, assignment="file")


def test_empty_subset_id_is_refused(tmp_path):
    tasks = "task_id,answers,correct\nt1,A|B,A\n"
    resp = ("task_id,response_id,vote,confidence,ps_1,ps_2,subset_id\n"
            "t1,r1,A,1,50,50,s1\nt1,r2,B,1,50,50,\n")
    with pytest.raises(L.WorldBFormatError, match="subset_id"):
        load(tmp_path, tasks=tasks, responses=resp)


def test_name_and_type_columns_are_optional(tmp_path):
    tasks = "task_id,answers,correct\nt1,A|B,A\n"
    resp = "task_id,response_id,vote,confidence,ps_1,ps_2\nt1,r1,A,1,50,50\n"
    data = load(tmp_path, tasks=tasks, responses=resp)
    assert data.names == (None,) and data.types == (None,)


# ---- write / read of the layout, and the config switch -----------------------------------

def test_write_and_read_back_a_non_synthetic_object(tmp_path):
    data = load(tmp_path)
    out = tmp_path / "out"
    paths = L.write_world_b(data, out)
    for key in ("tasks", "responses"):
        assert "SYNTHETIC" not in paths[key].read_text(encoding="utf-8")
    back = L.load_written(out)
    assert back == data and back.synthetic is False


def test_config_switch(tmp_path):
    from cdm.config import CODE_ROOT
    cfg = yaml.safe_load((CODE_ROOT / "config" / "worldB.yaml").read_text(encoding="utf-8"))
    # the shipped switch points at the synthetic stand-in; the real block is empty
    assert cfg["source"] == "synthetic"
    assert all(cfg["real"][k] is None for k in
               ("responses_file", "tasks_file", "confidence_range", "ps_scale"))
    assert "UNTESTED ON REAL DATA" in cfg["real"]["status"]
    data = L.load_from_config()
    assert data.synthetic is True and data.marker == L.SYNTHETIC_MARKER
    # real without the file and the scales: refused, nothing is guessed
    f = tmp_path / "w.yaml"
    cfg["source"] = "real"
    f.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="not set"):
        L.load_from_config(f)
    # real with a (fabricated) file
    r, t = write(tmp_path)
    cfg["real"].update(responses_file=str(r), tasks_file=str(t),
                       confidence_range=[1, 5], ps_scale=100)
    f.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    data = L.load_from_config(f)
    assert data.synthetic is False and data.n_responses == 5
    assert data.log["ps_sum_policy"] == "renormalize_if_violated"      # OQ39 default
    cfg["source"] = "both"
    f.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        L.load_from_config(f)
