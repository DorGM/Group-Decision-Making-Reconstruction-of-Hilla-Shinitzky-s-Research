"""A11 acceptance (plan §9.3 row A11): AMP/DAP instance builder + ablation
masks.

  "Shapes 1209x27 (AMP), 1209x32 (DAP); label means = T3; the §5.4 matrix
   encoded; groups per TH p.47"  (OQ5 / OQ41)

Part 1: hand-made cases and the ablation matrix (no data).
Part 2: the frozen builds `h1209_nba_games` (formerly
`best_evidence_nba_games`), `waiver_main` and `waiver_politics_alt` (only
when CDM_DATA_ROOT is set and the builds exist). All three are hypotheses.
On each, some rule-baseline counts differ from the thesis-reported counts
(docs/RULE_BASELINE_CHECK.md), so "label means = T3" is NOT asserted
against the thesis counts: the label sums are asserted to equal the
rule-baseline success counts of the SAME build (consistency only).

The waiver builds treat Politics as binary, so for them the A11 shape
acceptance is met in full: 1209 x 27 (AMP), 1209 x 32 (DAP), no non-binary
case and no NaN feature.

No model is trained or evaluated anywhere in this file.
"""
import os
import re
from pathlib import Path

import numpy as np
import pytest

from cdm import aggregators as AG
from cdm.approaches import amp, dap
from cdm.approaches import instances as I
from cdm.config import load_config
from cdm.features import cdmcr
from cdm.partition import partition
from cdm.schema import Case, Response


# ---- helpers -------------------------------------------------------------------

def resp(v, c, ps_first):
    return Response(v=v, c=c, ps=(ps_first, 1.0 - ps_first))


def make_case(responses, correct="a1", case_id="c", answers=("a1", "a2"), domain="General"):
    return Case(case_id=case_id, answers=answers, correct=correct,
                responses=tuple(responses), domain=domain)


def case_lone_mr(case_id="lone", correct="a1"):
    """MR = a1 (6 of 10 votes); WC = HAC = SP = a2.
    WC: 6*0.1 = 0.6 < 4*0.9 = 3.6. HAC: 0.1 < 0.9.
    SP: everybody predicts 90% for a1 -> a1: .6-.9 = -.3, a2: .4-.1 = +.3."""
    return make_case([resp("a1", 0.1, 0.9)] * 6 + [resp("a2", 0.9, 0.9)] * 4,
                     correct=correct, case_id=case_id)


def case_all_agree(case_id="agree", correct="a1"):
    """All four rules choose a1."""
    return make_case([resp("a1", 0.9, 0.5)] * 7 + [resp("a2", 0.2, 0.5)] * 3,
                     correct=correct, case_id=case_id)


def random_cases(n_cases, seed=0, m=2):
    rng = np.random.default_rng(seed)
    answers = tuple(f"a{i + 1}" for i in range(m))
    out = []
    for k in range(n_cases):
        n = int(rng.integers(8, 75))
        rs = []
        for _ in range(n):
            ps = rng.dirichlet(np.ones(m))
            rs.append(Response(v=answers[int(rng.integers(m))],
                               c=float(rng.integers(0, 5)) / 4, ps=tuple(float(x) for x in ps)))
        out.append(Case(case_id=f"r{seed}_{k}", answers=answers,
                        correct=answers[int(rng.integers(m))], responses=tuple(rs),
                        domain="General"))
    return out


# ---- label order ---------------------------------------------------------------

def test_label_order_is_the_thesis_order():
    # TH p.37: {O_MR, O_HAC, O_WC, O_SP, O_DA}; TH p.38: {f_MR, f_HAC, f_WC, f_SP, f_DA}
    assert I.LABEL_ORDER == ("MR", "HAC", "WC", "SP", "DA")
    assert tuple(load_config()["open_questions"]["OQ17"]["label_order"]) == I.LABEL_ORDER


# ---- the §5.4 matrix -----------------------------------------------------------

#: Plan §5.4 table, typed again here: row, Conf, PS, WC+HAC, SP, DA, AMP, DAP.
PLAN_5_4 = """
1 1 1 1 1 1 .806 .812
2 0 1 1 1 1 .800 .812
3 1 0 1 1 1 .816 .825
4 0 0 1 1 1 .795 .803
5 1 1 0 1 1 .808 .804
6 1 1 1 0 1 .797 .805
7 1 1 0 0 1 .791 .806
8 1 1 1 1 0 .754 .813
9 0 1 0 1 1 .806 .811
10 1 0 1 0 1 .799 .815
11 0 0 0 0 1 .763 .798
"""


def _rows(text):
    out = {}
    for line in text.strip().splitlines():
        p = line.split()
        out[int(p[0])] = (tuple(int(x) for x in p[1:6]), (float(p[6]), float(p[7])))
    return out


def test_ablation_matrix_equals_plan_5_4():
    rows = _rows(PLAN_5_4)
    assert sorted(I.ABLATION_MATRIX) == list(range(1, 12))
    for r, (bits, thesis) in rows.items():
        assert I.ABLATION_MATRIX[r] == bits, r
        assert I.ABLATION_THESIS[r] == thesis, r
        assert I.ablation_config(r).as_tuple() == bits
    assert I.ABLATION_COLUMNS == ("conf", "ps", "wc_hac", "sp", "da")
    assert len(I.all_ablation_configs()) == 11


def test_ablation_matrix_equals_the_plan_file():
    """Parse the table out of docs/RECONSTRUCTION_PLAN.md itself (skipped if the plan
    file is not reachable)."""
    path = Path(os.environ.get("CDM_PLAN_FILE",
                               str(Path(__file__).resolve().parent.parent / "docs"
                                   / "RECONSTRUCTION_PLAN.md")))
    if not path.is_file():
        pytest.skip("plan file not reachable")
    text = path.read_text(encoding="utf-8")
    start = text.index("### 5.4 Ablations")
    block = text[start:text.index("### 5.5", start)]
    found = {}
    for line in block.splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*([01])\s*\|\s*([01])\s*\|\s*([01])\s*\|\s*([01])"
                     r"\s*\|\s*([01])\s*\|\s*(\.\d+)\s*\|\s*(\.\d+)\s*\|", line)
        if m:
            g = m.groups()
            found[int(g[0])] = (tuple(int(x) for x in g[1:6]), (float(g[6]), float(g[7])))
    assert len(found) == 11
    for r in found:
        assert I.ABLATION_MATRIX[r] == found[r][0], r
        assert I.ABLATION_THESIS[r] == found[r][1], r


def test_matrix_structure_stated_in_plan():
    # Plan §5.4: rows 9, 10, 11 are the literal "no confidence info", "no
    # predicted-support info" and "neither"; row 8 is the only one without DA.
    c = {r: I.ablation_config(r) for r in range(1, 12)}
    assert (c[9].conf, c[9].wc_hac, c[9].ps, c[9].sp) == (False, False, True, True)
    assert (c[10].ps, c[10].sp, c[10].conf, c[10].wc_hac) == (False, False, True, True)
    assert c[11].as_tuple() == (0, 0, 0, 0, 1)
    assert [r for r in c if not c[r].da] == [8]
    with pytest.raises(ValueError):
        I.ablation_config(12)
    with pytest.raises(ValueError):
        I.ablation_config(0)


# ---- feature groups (TH p.47) and masks ----------------------------------------

VOTE = {1, 2, 3, 4, 5, 18, 19, 20, 21}
CONF = {6, 7, 8, 9, 10, 11, 12, 16, 17, 22, 23, 24, 25, 26, 27}
PS = {13, 14, 15, 16, 17}


def test_groups_per_th_p47():
    assert set(cdmcr.GROUPS["Vote"]) == VOTE
    assert set(cdmcr.GROUPS["Conf"]) == CONF
    assert set(cdmcr.GROUPS["PS"]) == PS
    assert VOTE | CONF | PS == set(range(1, 28))
    assert CONF & PS == {16, 17} and not VOTE & (CONF | PS)


def expected_features(bits, oq41):
    conf, ps, wc_hac, sp, _ = bits
    keep = set(VOTE)
    if conf:
        keep |= CONF - {16, 17}
    if ps:
        keep |= PS - {16, 17}
    if conf and ps:
        keep |= {16, 17}
    return keep


@pytest.mark.parametrize("row", range(1, 12))
def test_feature_mask_default_oq41(row):
    """OQ41 default: 16/17 dropped whenever either group is dropped."""
    bits = I.ABLATION_MATRIX[row]
    kept = set(I.feature_numbers_kept(row))
    assert kept == expected_features(bits, "default")
    assert VOTE <= kept                                   # always kept (TH p.48)
    mask = I.feature_mask(row)
    assert mask.dtype == bool and mask.shape == (27,)
    assert {i + 1 for i in range(27) if mask[i]} == kept


def test_feature_counts_per_row():
    n = {r: len(I.feature_numbers_kept(r)) for r in range(1, 12)}
    #        full  -Conf  -PS  -both  (labels only: rows 5-8)   9    10   11
    assert n == {1: 27, 2: 12, 3: 22, 4: 9, 5: 27, 6: 27, 7: 27, 8: 27,
                 9: 12, 10: 22, 11: 9}
    alt = {r: len(I.feature_numbers_kept(r, "kept_unless_both_groups_dropped"))
           for r in range(1, 12)}
    # Plan OQ41 alternative "(rows 2-4 only)": 16/17 kept in rows 2 and 3,
    # dropped in row 4 (both groups dropped); rows 9-11 cannot compute them.
    assert alt == {1: 27, 2: 14, 3: 24, 4: 9, 5: 27, 6: 27, 7: 27, 8: 27,
                   9: 12, 10: 22, 11: 9}
    for r in (2, 3):
        assert {16, 17} <= set(I.feature_numbers_kept(r, "kept_unless_both_groups_dropped"))
        assert not {16, 17} & set(I.feature_numbers_kept(r))
    with pytest.raises(ValueError):
        I.feature_numbers_kept(1, "something_else")


def test_oq41_from_config():
    assert I.oq41_from_config() == "dropped_whenever_either_group_dropped"
    cfg = load_config()
    cfg["open_questions"]["OQ41"]["default"] = "nonsense"
    with pytest.raises(ValueError):
        I.oq41_from_config(cfg)


EXPECTED_METHODS = {
    1: ("MR", "HAC", "WC", "SP", "DA"), 2: ("MR", "HAC", "WC", "SP", "DA"),
    3: ("MR", "HAC", "WC", "SP", "DA"), 4: ("MR", "HAC", "WC", "SP", "DA"),
    5: ("MR", "SP", "DA"), 6: ("MR", "HAC", "WC", "DA"), 7: ("MR", "DA"),
    8: ("MR", "HAC", "WC", "SP"), 9: ("MR", "SP", "DA"),
    10: ("MR", "HAC", "WC", "DA"), 11: ("MR", "DA"),
}
EXPECTED_RULES = {
    1: ("MR", "WC", "HAC", "SP"), 2: ("MR", "WC", "HAC", "SP"),
    3: ("MR", "WC", "HAC", "SP"), 4: ("MR", "WC", "HAC", "SP"),
    5: ("MR", "SP"), 6: ("MR", "WC", "HAC"), 7: ("MR",),
    8: ("MR", "WC", "HAC", "SP"), 9: ("MR", "SP"), 10: ("MR", "WC", "HAC"),
    11: ("MR",),
}


@pytest.mark.parametrize("row", range(1, 12))
def test_methods_and_rules_kept(row):
    assert I.methods_kept(row) == EXPECTED_METHODS[row]
    assert I.rules_kept(row) == EXPECTED_RULES[row]
    assert "MR" in I.methods_kept(row)                    # always kept (TH p.48)


# ---- per-case seed -------------------------------------------------------------

def test_case_seed_is_deterministic_and_documented():
    import hashlib
    s = I.case_seed(0, "General:1")
    assert s == int.from_bytes(
        hashlib.sha256(b"0|cdmcr_partition|General:1").digest()[:8], "big")
    assert I.case_seed(0, "General:1") == s
    assert I.case_seed(1, "General:1") != s
    assert I.case_seed(0, "General:2") != s
    assert 0 <= s < 2 ** 64
    for bad in (None, "0", 0.0, True):
        with pytest.raises(ValueError):
            I.case_seed(bad, "x")


def test_partition_of_a_case_does_not_depend_on_the_other_cases():
    cases = random_cases(6, seed=3)
    a = I.build_instances(cases)
    b = I.build_instances(list(reversed(cases)))
    assert a.case_ids == list(reversed(b.case_ids))
    np.testing.assert_array_equal(a.F, b.F[::-1])
    assert a.partitions == list(reversed(b.partitions))
    sub = I.build_instances(cases[2:4])
    np.testing.assert_array_equal(sub.F, a.F[2:4])
    # stored seed and partition are the ones of cdm.partition with case_seed
    opts = cdmcr.options_from_config()["partition_kwargs"]
    for i, c in enumerate(cases):
        assert a.seeds[i] == I.case_seed(0, c.case_id)
        assert a.partitions[i] == partition(len(c.responses), a.seeds[i], **opts)
    assert a.seed == 0 and a.seed_rule == I.SEED_RULE
    # another global seed changes the SG features of some case, not 1-17
    c = I.build_instances(cases, seed=5)
    np.testing.assert_array_equal(a.F[:, :17], c.F[:, :17])
    assert not np.array_equal(a.F[:, 17:], c.F[:, 17:])


def test_duplicate_case_ids_are_rejected():
    with pytest.raises(ValueError):
        I.build_instances([case_all_agree("x"), case_lone_mr("x")])


# ---- AMP instances -------------------------------------------------------------

def test_amp_shapes_and_labels_hand_made():
    cases = [case_lone_mr("c1", correct="a1"), case_lone_mr("c2", correct="a2"),
             case_all_agree("c3", correct="a1"), case_all_agree("c4", correct="a2")]
    inst = I.build_instances(cases)
    xy = amp.amp_xy(inst)
    assert xy.X.shape == (4, 27) and xy.Y.shape == (4, 5)
    assert xy.feature_names == cdmcr.FEATURE_NAMES
    assert xy.target_names == ("O_MR", "O_HAC", "O_WC", "O_SP", "O_DA")
    # c1: MR=a1, others a2 -> DA = a1 (MR is the lone dissenter), a* = a1
    # c2: same answers, a* = a2
    # c3: all a1 -> DA = a2, a* = a1;  c4: a* = a2
    np.testing.assert_array_equal(xy.Y, [[1, 0, 0, 0, 1],
                                         [0, 1, 1, 1, 0],
                                         [1, 1, 1, 1, 0],
                                         [0, 0, 0, 0, 1]])
    # features are those of cdmcr_features with the per-case seed
    opts = cdmcr.options_from_config()
    for i, c in enumerate(cases):
        f = cdmcr.cdmcr_features(c.responses, c.answers,
                                 seed=I.case_seed(0, c.case_id), **opts)
        np.testing.assert_array_equal(xy.X[i], [f[n] for n in cdmcr.FEATURE_NAMES])
    assert xy.X[0, 0] == 10                               # feature 1 = N


def test_da_label_changes_under_method_subsets():
    """TH p.48: when WC/HAC/SP labels are excluded, DA is recomputed over
    the remaining methods."""
    inst = I.build_instances([case_lone_mr("c1", correct="a1")])
    assert {m: inst.choices[m][0] for m in AG.RULES} == {
        "MR": "a1", "WC": "a2", "HAC": "a2", "SP": "a2"}
    # all four: a1 x1, a2 x3 -> DA = a1
    assert inst.da() == ["a1"]
    # MR + WC + HAC (row 6, 10): a1 x1, a2 x2 -> a1
    assert inst.da(("MR", "WC", "HAC")) == ["a1"]
    # MR + SP (rows 5, 9): 1-1 tie -> the answer not chosen by MR -> a2
    assert inst.da(("MR", "SP")) == ["a2"]
    # MR only (rows 7, 11): DA = "not MR" (plan §5.4)
    assert inst.da(("MR",)) == ["a2"]
    o_da = {r: int(amp.amp_xy(inst, r).Y[0, list(I.methods_kept(r)).index("DA")])
            for r in range(1, 12) if r != 8}
    assert o_da == {1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 1, 7: 0, 9: 0, 10: 1, 11: 0}
    assert "O_DA" not in amp.amp_xy(inst, 8).target_names


@pytest.mark.parametrize("row", range(1, 12))
def test_amp_ablation_rows_on_random_cases(row):
    cases = random_cases(25, seed=11)
    inst = I.build_instances(cases)
    xy = amp.amp_xy(inst, row)
    methods, rules = EXPECTED_METHODS[row], EXPECTED_RULES[row]
    assert xy.target_names == tuple("O_" + m for m in methods)
    assert xy.X.shape == (25, len(I.feature_numbers_kept(row)))
    assert xy.Y.shape == (25, len(methods))
    assert set(np.unique(xy.Y)) <= {0, 1}
    full = amp.amp_xy(inst, 1)
    kept = [n - 1 for n in I.feature_numbers_kept(row)]
    np.testing.assert_array_equal(xy.X, full.X[:, kept])
    assert xy.feature_names == tuple(cdmcr.FEATURE_NAMES[k] for k in kept)
    # labels equal a direct call of the aggregators on the kept rules
    opts = AG.options_from_config()
    for i, c in enumerate(cases):
        ch = AG.aggregate_all(c.responses, c.answers, rules, **opts)
        assert [int(ch[m] == c.correct) for m in methods] == list(xy.Y[i])
    if row in (7, 11):                                    # DA = not MR (binary)
        np.testing.assert_array_equal(xy.Y[:, 0] + xy.Y[:, 1], 1)


def test_amp_works_for_more_than_two_answers():
    cases = random_cases(8, seed=2, m=3)
    xy = amp.amp_xy(I.build_instances(cases))
    assert xy.X.shape == (8, 27) and xy.Y.shape == (8, 5)


def test_sp_half_credit_policy_has_no_label():
    """A tied SP case has no single answer under the scoring policy
    'sp_tie_half_credit', hence no 0/1 label: the builder raises."""
    tied = make_case([resp("a1", 0.5, 0.5), resp("a2", 0.5, 0.5)], case_id="t")
    with pytest.raises(NotImplementedError):
        I.build_instances([tied], aggregator_options={"policy": "sp_tie_half_credit",
                                                      "tol": 1e-9})


# ---- DAP instances -------------------------------------------------------------

def test_dap_shapes_and_encoding_hand_made():
    cases = [case_lone_mr("c1", correct="a1"), case_lone_mr("c2", correct="a2"),
             case_all_agree("c3", correct="a1"), case_all_agree("c4", correct="a2")]
    inst = I.build_instances(cases)
    xy = dap.dap_xy(inst)
    assert xy.X.shape == (4, 32) and xy.Y.shape == (4,)
    assert xy.feature_names == cdmcr.FEATURE_NAMES + (
        "f_MR", "f_HAC", "f_WC", "f_SP", "f_DA")
    assert xy.target_names == ("a*",)
    np.testing.assert_array_equal(xy.X[:, :27], amp.amp_xy(inst).X)
    # a1 (the first answer of the case) is coded 1 (plan §4.2 / OQ5)
    np.testing.assert_array_equal(xy.X[:, 27:], [[1, 0, 0, 0, 1],
                                                 [1, 0, 0, 0, 1],
                                                 [1, 1, 1, 1, 0],
                                                 [1, 1, 1, 1, 0]])
    np.testing.assert_array_equal(xy.Y, [1, 0, 1, 0])
    # O_AG = 1 iff the f_AG code equals the target code
    np.testing.assert_array_equal((xy.X[:, 27:] == xy.Y[:, None]).astype(int),
                                  amp.amp_xy(inst).Y)


def test_encode_decode_answer():
    assert dap.encode_answer("True", ("True", "False")) == 1
    assert dap.encode_answer("False", ("True", "False")) == 0
    assert dap.decode_answer(1, ("home", "away")) == "home"
    assert dap.decode_answer(0, ("home", "away")) == "away"
    with pytest.raises(ValueError):
        dap.encode_answer("x", ("a", "b"))
    with pytest.raises(NotImplementedError):
        dap.encode_answer("a", ("a", "b", "c"))
    with pytest.raises(NotImplementedError):
        dap.decode_answer(1, ("a", "b", "c"))


def test_dap_encoding_follows_the_answer_order_of_the_case():
    """OQ5 is decided by the order of Case.answers (set when the data is
    built): swapping the order flips every code of that case."""
    rs = [resp("a1", 0.9, 0.5)] * 7 + [resp("a2", 0.2, 0.5)] * 3
    c1 = make_case(rs, correct="a1", case_id="k")
    swapped = [Response(v=r.v, c=r.c, ps=(r.ps[1], r.ps[0])) for r in rs]
    c2 = make_case(swapped, correct="a1", case_id="k", answers=("a2", "a1"))
    x1 = dap.dap_xy(I.build_instances([c1]))
    x2 = dap.dap_xy(I.build_instances([c2]))
    np.testing.assert_array_equal(x1.X[:, 27:], 1 - x2.X[:, 27:])
    assert (x1.Y[0], x2.Y[0]) == (1, 0)


@pytest.mark.parametrize("row", range(1, 12))
def test_dap_ablation_rows_on_random_cases(row):
    cases = random_cases(25, seed=12)
    inst = I.build_instances(cases)
    xy = dap.dap_xy(inst, row)
    methods, rules = EXPECTED_METHODS[row], EXPECTED_RULES[row]
    n_feat = len(I.feature_numbers_kept(row))
    assert xy.X.shape == (25, n_feat + len(methods))
    assert xy.feature_names[n_feat:] == tuple("f_" + m for m in methods)
    np.testing.assert_array_equal(xy.X[:, :n_feat], amp.amp_xy(inst, row).X)
    opts = AG.options_from_config()
    for i, c in enumerate(cases):
        ch = AG.aggregate_all(c.responses, c.answers, rules, **opts)
        assert [int(ch[m] == c.answers[0]) for m in methods] == list(xy.X[i, n_feat:])
        assert xy.Y[i] == int(c.correct == c.answers[0])
    if row in (7, 11):                                    # f_DA = not f_MR
        np.testing.assert_array_equal(xy.X[:, n_feat] + xy.X[:, n_feat + 1], 1)


def test_dap_stops_on_non_binary_cases():
    cases = random_cases(4, seed=1) + random_cases(2, seed=9, m=3)
    inst = I.build_instances(cases)
    assert dap.non_binary_positions(inst) == [4, 5]
    with pytest.raises(NotImplementedError):
        dap.dap_xy(inst)
    assert dap.dap_xy(inst.select(range(4))).X.shape == (4, 32)


def test_select_and_log():
    cases = random_cases(6, seed=4)
    inst = I.build_instances(cases)
    sub = inst.select([4, 1])
    assert sub.case_ids == [cases[4].case_id, cases[1].case_id] and len(sub) == 2
    np.testing.assert_array_equal(sub.F, inst.F[[4, 1]])
    assert sub.da() == [inst.da()[4], inst.da()[1]]
    log = inst.log()
    assert log["non_binary_cases"] == 0 and "nan_features" in log
    assert set(inst.options) == {"features", "aggregators", "da_tie_rule"}


# ---- Part 2: the frozen build (consistency only) -------------------------------

BEST = "h1209_nba_games"
WAIVER = ("waiver_main", "waiver_politics_alt")


def _frozen_available(name=BEST):
    if not os.environ.get("CDM_DATA_ROOT"):
        return False
    from cdm.data.worldA.build_worldA import build_root
    return (build_root() / "worldA" / name / "manifest.json").is_file()


needs_build = pytest.mark.skipif(
    not _frozen_available(),
    reason="CDM_DATA_ROOT not set or no frozen build of " + BEST)


@pytest.fixture(scope="module")
def frozen():
    from cdm.data.worldA.build_worldA import load_frozen
    from cdm.eval import rule_gate as G
    cases, manifest = load_frozen(scenario=BEST)
    inst = I.build_instances(cases)
    gate = G.run_gate(cases)
    return cases, inst, gate


@needs_build
def test_real_amp_shape(frozen):
    cases, inst, _ = frozen
    xy = amp.amp_xy(inst)
    assert len(cases) == 1209
    assert xy.X.shape == (1209, 27)                       # plan row A11
    assert xy.Y.shape == (1209, 5)
    assert xy.case_ids == [c.case_id for c in cases]      # build order kept (OQ16)
    assert set(np.unique(xy.Y)) == {0, 1}


@needs_build
def test_real_label_sums_equal_the_rule_gate_of_the_same_build(frozen):
    """Consistency with step A10 on the SAME build. Not the thesis T3: some
    rule-baseline counts of this build differ from the thesis-reported counts."""
    from cdm.eval import rule_gate as G
    cases, inst, gate = frozen
    assert gate.policy == "canonical_first"
    xy = amp.amp_xy(inst)
    cols = {m: xy.Y[:, j] for j, m in enumerate(I.LABEL_ORDER)}
    domains = np.array(xy.domains)
    for m in I.LABEL_ORDER:
        assert int(cols[m].sum()) == gate.count(m), m
        for d in G.DOMAINS:
            assert int(cols[m][domains == d].sum()) == gate.count(m, d), (m, d)
    # case by case, including the answers
    for m in AG.RULES:
        assert inst.choices[m] == [r["choice"][m] for r in gate.records], m
    for m in I.LABEL_ORDER:
        assert list(cols[m]) == [int(r["credit"][m]) for r in gate.records], m
    assert inst.da() == [r["choice"]["DA"] for r in gate.records]
    print("\nlabel sums (same build, NOT the thesis-reported counts):",
          {m: int(cols[m].sum()) for m in I.LABEL_ORDER})


@needs_build
def test_real_dap_shape_and_consistency(frozen):
    cases, inst, gate = frozen
    non_binary = dap.non_binary_positions(inst)
    binary = [i for i in range(len(inst)) if i not in set(non_binary)]
    if non_binary:
        # the plan defines the DAP coding for binary cases only -> stop
        with pytest.raises(NotImplementedError):
            dap.dap_xy(inst)
    sub = inst.select(binary)
    xy = dap.dap_xy(sub)
    assert xy.X.shape == (len(binary), 32)                # 27 + 5
    assert xy.Y.shape == (len(binary),)
    assert set(np.unique(xy.X[:, 27:])) <= {0.0, 1.0}
    a = amp.amp_xy(sub)
    np.testing.assert_array_equal(xy.X[:, :27], a.X)
    np.testing.assert_array_equal((xy.X[:, 27:] == xy.Y[:, None]).astype(int), a.Y)
    for k, i in enumerate(binary):
        assert xy.Y[k] == int(cases[i].correct == cases[i].answers[0])
    print(f"\nDAP: {len(binary)} binary cases x 32; {len(non_binary)} non-binary "
          f"cases not encoded (plan silent); share of a* coded 1: {xy.Y.mean():.3f}")


@needs_build
@pytest.mark.parametrize("row", range(1, 12))
def test_real_ablation_rows(frozen, row):
    cases, inst, gate = frozen
    xy = amp.amp_xy(inst, row)
    methods, rules = EXPECTED_METHODS[row], EXPECTED_RULES[row]
    assert xy.X.shape == (1209, len(I.feature_numbers_kept(row)))
    assert xy.Y.shape == (1209, len(methods))
    full = amp.amp_xy(inst, 1)
    for m in methods:                                     # rule labels never change
        if m != "DA":
            np.testing.assert_array_equal(
                xy.Y[:, methods.index(m)], full.Y[:, I.LABEL_ORDER.index(m)])
    if "DA" in methods:
        da = xy.Y[:, methods.index("DA")]
        opts = AG.options_from_config()
        for i in range(0, 1209, 7):                       # direct recomputation
            c = cases[i]
            assert da[i] == int(AG.da(c.responses, c.answers, rules, **opts) == c.correct)
        if rules == AG.RULES:
            assert int(da.sum()) == gate.count("DA")
        if rules == ("MR",):
            binary = np.array([len(a) == 2 for a in inst.answers])
            np.testing.assert_array_equal(da[binary], 1 - xy.Y[binary, 0])


@needs_build
def test_real_da_label_changes_under_subsets(frozen):
    cases, inst, gate = frozen
    full = inst.outcomes("DA")
    changed = {rules: int((inst.outcomes("DA", rules) != full).sum())
               for rules in (("MR", "SP"), ("MR", "WC", "HAC"), ("MR",))}
    print("\ncases whose O_DA changes vs the full DA:", changed)
    assert all(v > 0 for v in changed.values())


@needs_build
def test_real_table_log(frozen):
    cases, inst, gate = frozen
    log = inst.log()
    print("\ninstance-table log:", log)
    assert len(inst.seeds) == len(set(inst.seeds)) == 1209
    assert all(len(p) >= 2 for p in inst.partitions)
    for i in (0, 600, 1208):
        assert sorted(j for part in inst.partitions[i] for j in part) == list(
            range(inst.n_responses[i]))
    # every feature other than those logged as NaN is finite
    assert np.isfinite(inst.F).sum() == inst.F.size - sum(log["nan_features"].values())


# ---- Part 3: the waiver builds (binary Politics; hypotheses) --------------------

needs_waiver_builds = pytest.mark.skipif(
    not all(_frozen_available(n) for n in WAIVER),
    reason="CDM_DATA_ROOT not set or no frozen waiver builds")


@pytest.fixture(scope="module", params=WAIVER)
def waiver(request):
    from cdm.data.worldA.build_worldA import load_frozen
    from cdm.eval import rule_gate as G
    cases, manifest = load_frozen(scenario=request.param)
    return request.param, cases, manifest, I.build_instances(cases), G.run_gate(cases)


@needs_waiver_builds
def test_waiver_build_is_binary_with_no_nan_and_full_dap(waiver):
    """With Politics binary, DAP builds for all 1209
    cases, no NaN CAmin remains at the full-set level, and the OQ8b rule
    is never used. Features and labels only; nothing is trained."""
    name, cases, manifest, inst, gate = waiver
    assert "rule-baseline differences documented" in manifest["status"]
    log = inst.log()
    print(f"\n{name}: instance-table log:", log)
    assert len(cases) == 1209
    assert log["non_binary_cases"] == 0
    assert log["nan_features"] == {}
    assert np.isfinite(inst.F).all()
    assert dap.non_binary_positions(inst) == []
    a = amp.amp_xy(inst)
    d = dap.dap_xy(inst)
    assert a.X.shape == (1209, 27) and a.Y.shape == (1209, 5)     # plan row A11
    assert d.X.shape == (1209, 32) and d.Y.shape == (1209,)       # plan row A11
    assert np.isfinite(d.X).all()
    assert set(np.unique(d.X[:, 27:])) == {0.0, 1.0} and set(np.unique(d.Y)) == {0, 1}
    np.testing.assert_array_equal(d.X[:, :27], a.X)
    np.testing.assert_array_equal((d.X[:, 27:] == d.Y[:, None]).astype(int), a.Y)
    # binary: with two answers at least one method is right, never none
    assert int((a.Y.sum(axis=1) == 0).sum()) == 0
    assert gate.da_oq8b()["n_cases"] == 0
    assert a.domains == ["General"] * 500 + ["NBA"] * 393 + ["NFL"] * 212 + ["Politics"] * 104
    print(f"{name}: DAP 1209 x 32; share of a* coded 1: {d.Y.mean():.3f}; "
          f"label means (MR, HAC, WC, SP, DA): {np.round(a.Y.mean(axis=0), 3).tolist()}")


@needs_waiver_builds
def test_waiver_label_sums_equal_the_rule_gate_of_the_same_build(waiver):
    """Consistency with step A10 on the SAME build, not the thesis T3 (some
    rule-baseline counts differ from the thesis-reported counts)."""
    from cdm.eval import rule_gate as G
    name, cases, manifest, inst, gate = waiver
    xy = amp.amp_xy(inst)
    domains = np.array(xy.domains)
    for j, m in enumerate(I.LABEL_ORDER):
        assert int(xy.Y[:, j].sum()) == gate.count(m) == \
            manifest["rule_gate"]["global"][m]["recomputed"], m
        for dname in G.DOMAINS:
            assert int(xy.Y[domains == dname, j].sum()) == gate.count(m, dname), (m, dname)
    assert inst.da() == [r["choice"]["DA"] for r in gate.records]
    assert inst.options["da_tie_rule"] == "not_mr" == gate.da_tie_rule


@needs_waiver_builds
@pytest.mark.parametrize("row", range(1, 12))
def test_waiver_ablation_rows_build_for_amp_and_dap(waiver, row):
    name, cases, manifest, inst, gate = waiver
    a, d = amp.amp_xy(inst, row), dap.dap_xy(inst, row)
    n_feat, methods = len(I.feature_numbers_kept(row)), EXPECTED_METHODS[row]
    assert a.X.shape == (1209, n_feat) and a.Y.shape == (1209, len(methods))
    assert d.X.shape == (1209, n_feat + len(methods))
    assert np.isfinite(a.X).all() and np.isfinite(d.X).all()
    if row in (7, 11):                                    # DA = not MR in every case
        np.testing.assert_array_equal(a.Y[:, 0] + a.Y[:, 1], 1)
