"""B5 (plan §9.3, §5.7): World-B rule baselines MR, WC, SP, HAC on the
sub-sets, over 30 partition seeds, beside T24.

Plan row B5: "Real data: MR .442, WC .442, SP .429, HAC .286 (+-1 case per
seed)". The real file is not available; on the SYNTHETIC stand-in the four
counts are the generator's calibration targets, so nothing here is a
reproduction. The tests check the mechanics and the labelling.
"""
import math

import pytest

from cdm import aggregators as AG
from cdm.data.worldB import loader as L
from cdm.data.worldB import partition as WP
from cdm.data.worldB import synthetic as S
from cdm.eval import worldB_rules as WR
from cdm.schema import Case, Response

DATA = S.generate()


def test_t24_targets():
    # TH Fig 5.3 p.76: rates as printed; plan T24: counts of 77
    assert WR.RULES_B == ("MR", "WC", "SP", "HAC")            # no DA (plan §5.7)
    assert WR.THESIS_RATE == {"MR": 0.442, "WC": 0.442, "SP": 0.429, "HAC": 0.286}
    assert WR.THESIS_COUNT == {"MR": 34, "WC": 34, "SP": 33, "HAC": 22}
    assert WR.N_SUBSETS == 77
    # "the counts are the only integers that round to them" (plan §2.3)
    for rule, rate in WR.THESIS_RATE.items():
        fits = [k for k in range(78) if round(k / 77, 3) == rate]
        assert fits == [WR.THESIS_COUNT[rule]]
    assert WR.DEFAULT_SEEDS == tuple(range(30))               # plan §8 OQ25


def _case(case_id, correct, rows, answers=("a", "b", "c")):
    rs = []
    for v, c, ps, k in rows:
        rs += [Response(v, c, ps)] * k
    return Case(case_id, answers, correct, tuple(rs))


def _subsets(cases):
    return WP.WorldBSubsets(cases=tuple(cases), task_ids=tuple("t" for _ in cases),
                            response_indices=tuple(() for _ in cases), seed=0,
                            assignment="random", synthetic=False, marker="")


def test_hand_made_subsets():
    # case 1: a has 3 votes, b 2, c 1.
    #   MR -> a.  WC: a 0.9, b 1.6, c 0.5 -> b.  HAC: a .3, b .8, c .5 -> b.
    #   SP: S - AvgPS: a .5 - .6 = -.1 ; b 1/3 - .3 ; c 1/6 - .1 -> c (.0667)
    c1 = _case("c1", "b", [("a", .3, (.6, .3, .1), 3), ("b", .8, (.6, .3, .1), 2),
                           ("c", .5, (.6, .3, .1), 1)])
    out = WR.subset_outcome(c1)
    assert [out[r]["answer"] for r in WR.RULES_B] == ["a", "b", "c", "b"]
    assert [out[r]["correct"] for r in WR.RULES_B] == [False, True, False, True]
    # case 2: a full MR tie (2 / 2); canonical first -> a; the correct answer b
    # is among the tied ones
    # (SP: a .5 - .5 = 0, b .5 - .4 = .1, c -.1 -> b, no tie)
    c2 = _case("c2", "b", [("a", .5, (.5, .4, .1), 2), ("b", .5, (.5, .4, .1), 2)])
    out = WR.subset_outcome(c2)
    assert out["MR"]["answer"] == "a" and out["MR"]["tied"] == ["a", "b"]
    assert out["MR"]["correct"] is False and out["MR"]["correct_possible"] is True
    res = WR.evaluate_subsets(_subsets([c1, c2]))
    assert res["n"] == 2
    assert res["MR"] == {"correct": 0, "ties": 1, "low": 0, "high": 1}
    assert res["WC"] == {"correct": 1, "ties": 1, "low": 1, "high": 2}
    assert res["HAC"]["correct"] == 1 and res["HAC"]["ties"] == 1
    assert res["SP"] == {"correct": 1, "ties": 0, "low": 1, "high": 1}
    # the other tie policy of the Step D5 ladder resolves the two-way ties to b
    res = WR.evaluate_subsets(_subsets([c1, c2]), policy="canonical_second")
    assert res["MR"]["correct"] == 1 and res["WC"]["correct"] == 2
    # ... but it is undefined for a tie among more than two answers, which
    # multiple-choice sub-sets can have (decision A5)
    c3 = _case("c3", "b", [("a", .5, (.4, .3, .3), 1), ("b", .5, (.4, .3, .3), 1),
                           ("c", .5, (.4, .3, .3), 1)])
    with pytest.raises(NotImplementedError):
        WR.evaluate_subsets(_subsets([c3]), policy="canonical_second")
    assert WR.evaluate_subsets(_subsets([c3]))["MR"] == \
        {"correct": 0, "ties": 1, "low": 0, "high": 1}


def test_counts_equal_the_aggregators_on_every_subset():
    subs = WP.subset_cases(DATA, seed=7)
    res = WR.evaluate_subsets(subs)
    for rule, fn in (("MR", AG.mr), ("WC", AG.wc), ("SP", AG.sp), ("HAC", AG.hac)):
        want = sum(fn(c.responses, c.answers) == c.correct for c in subs.cases)
        assert res[rule]["correct"] == want
        assert res[rule]["low"] <= res[rule]["correct"] <= res[rule]["high"]
    assert res["n"] == 77


RUN = WR.evaluate_seeds(DATA)


def test_run_over_30_seeds():
    assert RUN.seeds == tuple(range(30)) and RUN.n_subsets == 77
    assert RUN.synthetic is True and RUN.marker == L.SYNTHETIC_MARKER
    s = RUN.summary()
    for rule in WR.RULES_B:
        xs = RUN.counts(rule)
        assert len(xs) == 30 and all(0 <= x <= 77 for x in xs)
        assert s[rule]["mean_count"] == pytest.approx(sum(xs) / 30)
        m = sum(xs) / 30
        assert s[rule]["sd_count"] == pytest.approx(
            math.sqrt(sum((x - m) ** 2 for x in xs) / 29))
        assert s[rule]["mean_rate"] == pytest.approx(m / 77)
        assert s[rule]["min_count"] == min(xs) and s[rule]["max_count"] == max(xs)
        assert s[rule]["thesis_count"] == WR.THESIS_COUNT[rule]
    # seeds change the partition, hence the counts
    assert len({tuple(RUN.per_seed[k][r]["correct"] for r in WR.RULES_B)
                for k in RUN.seeds}) > 1
    # the same means as the calibration recorded (same data, same seeds)
    recorded = S.load_parameters()[1]["calibration"]["mean_counts_reached"]
    for rule in WR.RULES_B:
        assert s[rule]["mean_count"] == pytest.approx(recorded[rule])


def test_summary_table_keeps_run_and_thesis_apart_and_is_marked():
    text = WR.format_summary(RUN)
    lines = text.splitlines()
    assert lines[0] == f"**{L.SYNTHETIC_MARKER}**"
    header = [ln for ln in lines if ln.startswith("| Rule")][0]
    cols = [c.strip() for c in header.strip("|").split("|")]
    assert cols[0] == "Rule"
    assert all(c.startswith("run (synthetic):") for c in cols[1:6])
    assert all(c.startswith("thesis:") for c in cols[6:])
    assert "BY CONSTRUCTION" in text and "not a reproduction" in text
    assert S.GENERATOR_SEED_NOTE in text and "generator seed 0 only" in text
    # no difference / delta column for synthetic data
    assert "Δ" not in text and "diff" not in text.lower()
    rows = WR.summary_rows(RUN)
    assert [r["rule"] for r in rows] == list(WR.RULES_B)
    assert all(set(r) >= {"run_mean_count", "run_sd_count", "thesis_count",
                          "thesis_rate"} for r in rows)
    per_seed = WR.per_seed_rows(RUN)
    assert len(per_seed) == 30 and per_seed[0]["partition_seed"] == 0
    assert per_seed[0]["n_subsets"] == 77


def test_non_synthetic_run_is_labelled_as_such(tmp_path):
    # a fabricated two-task file in the real layout (UNTESTED ON REAL DATA)
    tasks = "task_id,name,type,answers,correct\nt1,x,,A|B|C,B\nt2,y,,yes|no,yes\n"
    lines = ["task_id,response_id,vote,confidence,ps_1,ps_2,ps_3,subset_id"]
    for i in range(66):
        lines.append(f"t1,r{i},{'ABB'[i % 3]},{(i % 5) / 4},50,30,20,s{i % 2}")
    for i in range(60):
        lines.append(f"t2,r{i},{('yes', 'no')[i % 2]},{(i % 5) / 4},40,60,,g{i % 2}")
    (tmp_path / "t.csv").write_text(tasks, encoding="utf-8")
    (tmp_path / "r.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    data = L.load_world_b(tmp_path / "r.csv", tmp_path / "t.csv",
                          confidence_range=(0, 1), ps_scale=100)
    run = WR.evaluate_seeds(data, seeds=(0, 1, 2))
    assert run.synthetic is False and run.n_subsets == 4
    text = WR.format_summary(run)
    assert "SYNTHETIC" not in text and "run (real file):" in text
    assert "UNTESTED ON REAL DATA" in text
    # MR is right in both sub-sets of t1 (B has 2/3) for every seed
    assert all(run.per_seed[s]["MR"]["correct"] >= 2 for s in run.seeds)
    # the file's own assignment: one evaluation, seed None
    orig = WR.evaluate_seeds(data, assignment="original")
    assert orig.seeds == (None,) and orig.assignment == "original"
    assert WR.per_seed_rows(orig)[0]["partition_seed"] == "original"
    assert math.isnan(orig.summary()["MR"]["sd_count"])        # one partition only
    with pytest.raises(ValueError):
        WR.evaluate_seeds(data, seeds=())
