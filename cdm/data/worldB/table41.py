"""TH Table 4.1 (p.54) = MCF Table 3: the 22 problem-solving tasks of
World B, as printed. [EVIDENCE]

Used by the synthetic stand-in (plan §2.3: "the 22 rows of TH Table 4.1
exactly") and by the checks of the real-file loader (plan §9.3 row B4:
"Table 4.1 per-task stats").

Columns: name, type, |A_P|, |R_P|, S(a*, R_P) (2 d.p. as printed),
"Avg. Subset Size" (1 d.p. as printed; in TH only, plan §7 D11).

Derived, not printed:
  * number of sub-sets k = floor(|R_P| / 30)  [INFERRED, strong; plan §2.2:
    sum k = 77 and every printed average size equals |R_P| / k];
  * number of solvers = round(S(a*) * |R_P|)  [plan §2.3]. Its share
    rounds to the printed S(a*) in every row. For 18 rows it is the only
    such integer; for 4 rows the printed two decimals allow more than one
    (AMBIGUOUS_SOLVER_COUNTS; total 694-699, the rule gives 697). The real
    counts of these four tasks are therefore not known from the table
    (decision B3-2; checked in tests/test_b3_synthetic.py).

Source note (decision B3-14): the Type column lists
16 logic riddles, 2 NP-complete and 4 P-space-complete tasks (the same in
MCF Table 3); the text says "14 logical riddles and misleading math
problems ... and eight gamified tasks" (TH p.52-53, MCF §4.1).

Source note (plan §2.2, D9, OQ34): the |R_P| column sums to 2490; TH p.53
and MCF state 2514 responses.
"""
from collections import namedtuple
from fractions import Fraction

Task41 = namedtuple("Task41", "name type n_answers n_responses s_correct avg_subset_size")

TABLE_4_1 = (
    Task41("Apple", "Logic riddle", 4, 96, 0.26, 32.0),
    Task41("Bakery", "Logic riddle", 7, 91, 0.26, 30.3),
    Task41("Blue Yellow Crowns", "Logic riddle", 8, 93, 0.23, 31.0),
    Task41("Bobs Coins", "Logic riddle", 4, 88, 0.15, 44.0),
    Task41("Boxes of Diamonds", "Logic riddle", 5, 90, 0.14, 30.0),
    Task41("Coin", "Logic riddle", 7, 96, 0.29, 32.0),
    Task41("Colliding Robots", "Logic riddle", 8, 248, 0.18, 31.0),
    Task41("Frog", "Logic riddle", 5, 96, 0.47, 32.0),
    Task41("Knapsack 1", "NP complete", 6, 96, 0.39, 32.0),
    Task41("Knapsack 2", "NP complete", 6, 99, 0.54, 33.0),
    Task41("Letter Dice", "Logic riddle", 6, 93, 0.52, 31.0),
    Task41("Magic Potion", "Logic riddle", 8, 310, 0.10, 31.0),
    Task41("Pills", "Logic riddle", 6, 66, 0.48, 33.0),
    Task41("Race Cars", "Logic riddle", 8, 96, 0.10, 32.0),
    Task41("Rush Hour 1", "P-space complete", 6, 121, 0.45, 30.3),
    Task41("Rush Hour 2", "P-space complete", 6, 114, 0.29, 38.0),
    Task41("Rush Hour 3", "P-space complete", 6, 111, 0.32, 37.0),
    Task41("Rush Hour 4", "P-space complete", 6, 121, 0.37, 30.3),
    Task41("The Hats Game", "Logic riddle", 7, 88, 0.10, 44.0),
    Task41("Three Robots", "Logic riddle", 12, 93, 0.28, 31.0),
    Task41("Traveler", "Logic riddle", 4, 93, 0.45, 31.0),
    Task41("Wine Barrels", "Logic riddle", 6, 91, 0.30, 30.3),
)

N_TASKS = 22                      # TH p.52
N_SUBSETS = 77                    # TH p.55
N_RESPONSES_TABLE = 2490          # sum of the |R_P| column
N_RESPONSES_TEXT = 2514           # TH p.53, MCF §4.1 (plan D9, OQ34)
SUBSET_SIZE = 30                  # k = floor(|R_P| / 30), plan §2.2

BY_NAME = {t.name: t for t in TABLE_4_1}

#: Rows whose printed S(a*) is met by more than one solver count:
#: name -> the admissible counts (round(S * |R_P|) is one of them).
AMBIGUOUS_SOLVER_COUNTS = {
    "Colliding Robots": (44, 45),
    "Magic Potion": (30, 31, 32),
    "Rush Hour 1": (54, 55),
    "Rush Hour 3": (35, 36),
}


def n_subsets(n_responses):
    """k = floor(|R_P| / 30). Plan §2.2 [INFERRED from Table 4.1]."""
    return int(n_responses) // SUBSET_SIZE


def n_solvers(task):
    """round(S(a*) * |R_P|), plan §2.3. Exact decimal arithmetic (the
    printed share has two decimals), half rounded up; no row is on a half."""
    exact = Fraction(str(task.s_correct)) * task.n_responses
    return int(exact + Fraction(1, 2))


def printed_share(k, n):
    """k / n rounded half up to the two decimals of the S(a*) column."""
    return float(Fraction(int(Fraction(k, n) * 100 + Fraction(1, 2)), 100))
