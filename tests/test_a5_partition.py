"""A5 acceptance (plan §9.3): partition.

  90 -> 3 x 30; 121 -> {30, 30, 30, 31}; N < 30 -> two halves (TH p.27);
  30 <= N < 60 -> two halves (plan §8 OQ6 default); the 22 |R_P| values of
  TH Table 4.1 (p.54) give 77 sub-sets (TH p.55) with the printed
  "Avg. Subset Size" column.
"""
import pytest

from cdm.config import get_switch
from cdm.partition import n_subsets, partition, subset_sizes

# TH Table 4.1 p.54: (problem name, |R_P|, printed Avg. Subset Size)
TABLE_4_1 = [
    ("Apple", 96, 32.0),
    ("Bakery", 91, 30.3),
    ("Blue Yellow Crowns", 93, 31.0),
    ("Bobs Coins", 88, 44.0),
    ("Boxes of Diamonds", 90, 30.0),
    ("Coin", 96, 32.0),
    ("Colliding Robots", 248, 31.0),
    ("Frog", 96, 32.0),
    ("Knapsack 1", 96, 32.0),
    ("Knapsack 2", 99, 33.0),
    ("Letter Dice", 93, 31.0),
    ("Magic Potion", 310, 31.0),
    ("Pills", 66, 33.0),
    ("Race Cars", 96, 32.0),
    ("Rush Hour 1", 121, 30.3),
    ("Rush Hour 2", 114, 38.0),
    ("Rush Hour 3", 111, 37.0),
    ("Rush Hour 4", 121, 30.3),
    ("The Hats Game", 88, 44.0),
    ("Three Robots", 93, 31.0),
    ("Traveler", 93, 31.0),
    ("Wine Barrels", 91, 30.3),
]


def _check_disjoint_cover(parts, n):
    flat = [i for p in parts for i in p]
    assert sorted(flat) == list(range(n))          # disjoint, uses everything


def test_90_gives_three_subsets_of_30():
    parts = partition(90, seed=0)
    assert [len(p) for p in parts] == [30, 30, 30]
    _check_disjoint_cover(parts, 90)


def test_121_gives_30_30_30_31():
    parts = partition(121, seed=0)
    assert sorted(len(p) for p in parts) == [30, 30, 30, 31]
    _check_disjoint_cover(parts, 121)


@pytest.mark.parametrize("n,sizes", [(10, [5, 5]), (28, [14, 14]),
                                     (29, [15, 14]), (11, [6, 5]), (2, [1, 1])])
def test_below_30_gives_two_halves(n, sizes):
    # TH p.27: "dividing the set of responses into two sub-sets of equal size"
    parts = partition(n, seed=3)
    assert sorted((len(p) for p in parts), reverse=True) == sizes
    _check_disjoint_cover(parts, n)


@pytest.mark.parametrize("n,sizes", [(30, [15, 15]), (45, [23, 22]),
                                     (59, [30, 29])])
def test_30_to_59_gives_two_halves(n, sizes):
    # plan §8 OQ6 default: "if k<2, use 2 halves"
    assert subset_sizes(n) == sizes
    _check_disjoint_cover(partition(n, seed=1), n)


def test_60_gives_two_subsets_of_30():
    assert subset_sizes(60) == [30, 30]


def test_table_4_1_total_is_77_subsets():
    # TH p.55: "The sampling process generated a total of 77 sub-sets"
    assert len(TABLE_4_1) == 22
    assert sum(n_subsets(n) for _, n, _ in TABLE_4_1) == 77
    assert sum(len(partition(n, seed=0)) for _, n, _ in TABLE_4_1) == 77
    # plan §2.2: the |R_P| column sums to 2490 (not the 2514 of the text)
    assert sum(n for _, n, _ in TABLE_4_1) == 2490


@pytest.mark.parametrize("name,n,printed", TABLE_4_1)
def test_table_4_1_average_subset_size(name, n, printed):
    parts = partition(n, seed=0)
    _check_disjoint_cover(parts, n)
    avg = sum(len(p) for p in parts) / len(parts)
    # printed to 1 d.p. (121/4 = 30.25 is printed as 30.3)
    assert abs(avg - printed) <= 0.05 + 1e-9, (name, avg, printed)
    # TH p.53: "sub-sets of 30-45 responses"
    assert all(30 <= len(p) <= 45 for p in parts)
    # near-equal sizes
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 1


def test_seed_is_reproducible_and_matters():
    assert partition(90, seed=5) == partition(90, seed=5)
    assert partition(90, seed=5) != partition(90, seed=6)


def test_no_shuffle_gives_consecutive_blocks():
    parts = partition(90, seed=None, shuffle=False)
    assert parts == [list(range(0, 30)), list(range(30, 60)), list(range(60, 90))]


def test_alternative_schemes_of_oq6():
    # (c) k = ceil(N/30)
    assert subset_sizes(121, scheme="c_ceil_n_over_30") == [25, 24, 24, 24, 24]
    # (d) k = floor(N/30) sub-sets of exactly 30, leftover unused
    parts = partition(121, seed=0, scheme="d_floor_n_over_30_exactly_30_leftover_unused")
    assert [len(p) for p in parts] == [30, 30, 30, 30]
    assert len({i for p in parts for i in p}) == 120
    # (b) is not specified enough to implement
    with pytest.raises(NotImplementedError):
        partition(121, seed=0, scheme="b_overlapping_samples_of_30")
    with pytest.raises(ValueError):
        partition(121, seed=0, scheme="zzz")


def test_too_small():
    with pytest.raises(ValueError):
        partition(1, seed=0)


def test_defaults_match_config():
    d = get_switch("OQ6")
    assert d["scheme"] == "a_disjoint_floor_n_over_30"
    assert d["subset_size"] == 30
    assert d["halves_if_k_below_2"] is True
    assert partition(45, 0, subset_size=d["subset_size"], scheme=d["scheme"],
                     halves_if_k_below_2=d["halves_if_k_below_2"]) == partition(45, 0)


def test_r4_halves_switch_off_does_not_affect_n_below_30():
    # TH p.27: N < 30 -> two halves, whatever the OQ6 switch says.
    for n in (2, 10, 11, 29):
        assert subset_sizes(n, halves_if_k_below_2=False) == subset_sizes(n)
        parts = partition(n, seed=0, halves_if_k_below_2=False)
        assert len(parts) == 2 and sorted(i for p in parts for i in p) == list(range(n))
    # 30 <= N < 60 with the switch off: one sub-set holding every response
    for n in (30, 45, 59):
        parts = partition(n, seed=0, halves_if_k_below_2=False)
        assert len(parts) == 1 and sorted(parts[0]) == list(range(n))
    # N >= 60 is not affected
    assert subset_sizes(121, halves_if_k_below_2=False) == subset_sizes(121)
    # scheme (c): N < 30 gives two halves too
    assert subset_sizes(20, scheme="c_ceil_n_over_30", halves_if_k_below_2=False) == [10, 10]
