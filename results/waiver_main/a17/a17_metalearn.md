# A17 classic meta-learning (decision tree on the rule answers) - build `waiver_main`

Written 2026-11-12T10:03:05Z. `DecisionTreeClassifier` with default settings, random_state 0; leave-one-out over 1209 cases (plan §8 OQ21: the thesis does not state the protocol). Features: the 0/1 codes of the methods' answers (1 = the first answer of the case); f_DA is recomputed over the methods of the row. Thesis side: TH Table 5.8 p.73.

Note: the rule baselines of this build differ from the thesis-reported counts by a few cases (below). Where the rows of this run differ from the thesis-reported rows, the present reconstruction does not establish the reason. Nothing is tuned.

## Rule baselines: recomputed on the build vs printed in the thesis

Cells: recomputed count (rate) / thesis count (rate) / difference in cases.

| block | n | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|---|
| All cases | 1209 | 869 (0.719) / 864 (0.715) / +5 | 838 (0.693) / 831 (0.687) / +7 | 831 (0.687) / 830 (0.687) / +1 | 767 (0.634) / 774 (0.640) / -7 | 368 (0.304) / 372 (0.308) / -4 |
| General | 500 | 413 (0.826) / 413 (0.826) / +0 | 385 (0.770) / 385 (0.770) / +0 | 380 (0.760) / 380 (0.760) / +0 | 338 (0.676) / 338 (0.676) / +0 | 111 (0.222) / 108 (0.216) / +3 |
| NBA | 393 | 260 (0.662) / 258 (0.656) / +2 | 256 (0.651) / 252 (0.641) / +4 | 252 (0.641) / 253 (0.644) / -1 | 230 (0.585) / 237 (0.603) / -7 | 137 (0.349) / 142 (0.361) / -5 |
| NFL | 212 | 112 (0.528) / 109 (0.514) / +3 | 114 (0.538) / 111 (0.524) / +3 | 117 (0.552) / 114 (0.538) / +3 | 131 (0.618) / 131 (0.618) / +0 | 99 (0.467) / 101 (0.476) / -2 |
| Politics | 104 | 84 (0.808) / 84 (0.808) / +0 | 83 (0.798) / 83 (0.798) / +0 | 82 (0.788) / 83 (0.798) / -1 | 68 (0.654) / 68 (0.654) / +0 | 21 (0.202) / 21 (0.202) / +0 |

## Table 5.8

| row | f_MR | f_WC, f_HAC | f_SP | f_DA | successes | rate: run | thesis | diff | 50/50 leaves used |
|---|---|---|---|---|---|---|---|---|---|
| 1 | V | V | V | V | 869/1209 | 0.719 | 0.713 | +0.006 | 0 |
| 2 | V |  | V | V | 869/1209 | 0.719 | 0.708 | +0.011 | 0 |
| 3 | V | V |  | V | 838/1209 | 0.693 | 0.695 | -0.002 | 0 |
| 4 | V |  |  | V | 838/1209 | 0.693 | 0.687 | +0.006 | 0 |
| 5 | V | V | V |  | 869/1209 | 0.719 | 0.713 | +0.006 | 0 |

## Expected relations (plan §9.3 row A17)

- Row 4 (f_MR, f_DA) against MR applied to every case: 838 vs 838 successes (equal); expected about equal (plan §5.5; thesis .687 = MR).
- Rows 1 and 5: 869 and 869 successes; same prediction for every case: True; expected equal (f_DA is a function of the other four; thesis .713 = .713).
- Best row (1): 869 successes; SP applied to every case: 869; best below SP: False (thesis: .713 < .715).

## Successes per domain and row (no thesis counterpart)

| row | General (n = 500) | NBA (n = 393) | NFL (n = 212) | Politics (n = 104) | all (n = 1209) |
|---|---|---|---|---|---|
| 1 | 413 (0.826) | 260 (0.662) | 112 (0.528) | 84 (0.808) | 869 (0.719) |
| 2 | 413 (0.826) | 260 (0.662) | 112 (0.528) | 84 (0.808) | 869 (0.719) |
| 3 | 385 (0.770) | 256 (0.651) | 114 (0.538) | 83 (0.798) | 838 (0.693) |
| 4 | 385 (0.770) | 256 (0.651) | 114 (0.538) | 83 (0.798) | 838 (0.693) |
| 5 | 413 (0.826) | 260 (0.662) | 112 (0.528) | 84 (0.808) | 869 (0.719) |

The trees' rules are in `a17_tree_rules.txt`.
