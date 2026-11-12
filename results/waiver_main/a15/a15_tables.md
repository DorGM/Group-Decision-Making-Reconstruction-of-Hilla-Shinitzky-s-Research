# A15 tables: AMP effectiveness and success by domain - build `waiver_main`

Written 2026-11-12T09:59:43Z by scripts/run_a15_tables.py from the per-case files of the A13 / A14 runs; no model is fitted here.
Rates are rounded half up to three decimals. "run" = this reconstruction, "thesis" = the printed value / the plan's back-solved counts.

Rule-baseline check: status `waived` for `waiver_main` (docs/RULE_BASELINE_CHECK.md, AM17); frozen cases file c23769515355....

Note: under the documented reconstruction configuration, the values in the `run` columns differ from the thesis-reported values in several blocks; the differences are shown cell by cell. The present reconstruction does not establish the reason. Nothing is tuned.

## Rule baselines: recomputed on the build vs printed in the thesis

Cells: recomputed count (rate) / thesis count (rate) / difference in cases.

| block | n | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|---|
| All cases | 1209 | 869 (0.719) / 864 (0.715) / +5 | 838 (0.693) / 831 (0.687) / +7 | 831 (0.687) / 830 (0.687) / +1 | 767 (0.634) / 774 (0.640) / -7 | 368 (0.304) / 372 (0.308) / -4 |
| General | 500 | 413 (0.826) / 413 (0.826) / +0 | 385 (0.770) / 385 (0.770) / +0 | 380 (0.760) / 380 (0.760) / +0 | 338 (0.676) / 338 (0.676) / +0 | 111 (0.222) / 108 (0.216) / +3 |
| NBA | 393 | 260 (0.662) / 258 (0.656) / +2 | 256 (0.651) / 252 (0.641) / +4 | 252 (0.641) / 253 (0.644) / -1 | 230 (0.585) / 237 (0.603) / -7 | 137 (0.349) / 142 (0.361) / -5 |
| NFL | 212 | 112 (0.528) / 109 (0.514) / +3 | 114 (0.538) / 111 (0.524) / +3 | 117 (0.552) / 114 (0.538) / +3 | 131 (0.618) / 131 (0.618) / +0 | 99 (0.467) / 101 (0.476) / -2 |
| Politics | 104 | 84 (0.808) / 84 (0.808) / +0 | 83 (0.798) / 83 (0.798) / +0 | 82 (0.788) / 83 (0.798) / -1 | 68 (0.654) / 68 (0.654) / +0 | 21 (0.202) / 21 (0.202) / +0 |

## Success by domain, 10-fold (leave-one-out within 10-fold CV) [10-fold, seed 0]

Cells: rate (successes). Thesis side: AMP: TH §5.1.0.1 p.68 (counts: plan T10); the thesis prints no DAP success per domain for the 10-fold protocol.

| domain | n (run) | AMP: run | AMP: thesis | AMP: diff (cases) | DAP: run | DAP: thesis | DAP: diff (cases) |
|---|---|---|---|---|---|---|---|
| General | 500 | 0.828 (414) | 0.824 (412/500) | +2 | 0.840 (420) | – | – |
| NBA | 393 | 0.631 (248) | 0.919 (361/393) | -113 | 0.639 (251) | – | – |
| NFL | 212 | 0.566 (120) | 0.472 (100/212) | +20 | 0.557 (118) | – | – |
| Politics | 104 | 0.663 (69) | 0.596 (62/104) | +7 | 0.615 (64) | – | – |
| All cases | 1209 | 0.704 (851) | 0.773 (935/1209) | -84 | 0.706 (853) | 0.775 (937/1209) | -84 |

## Effectiveness of the AMP selection, 10-fold (leave-one-out within 10-fold CV) [10-fold, seed 0] - TH Tables 5.4 and 5.5

P(success by AG) is the rule applied to every case of the block: the recomputed rule baseline beside the thesis's. PRE from exact counts; p = one-sample one-tailed z-test (plan §5.8 item 3), exact upper tail; the thesis prints p for the global block only.

### All cases (n = 1209; thesis n = 1209)
Thesis side: TH Table 5.4 p.66 (counts: plan T8).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 428/1209 | 349/1209 | 0.354 | 0.289 | 0.719 (869) | 0.715 (864) | 295/428 | 287/349 | 0.689 | 0.822 | -0.133 | -0.105 | 0.377 | 0.913 | 4.2e-06 |
| MR | 185/1209 | 244/1209 | 0.153 | 0.202 | 0.693 (838) | 0.687 (831) | 143/185 | 200/244 | 0.773 | 0.820 | -0.047 | 0.260 | 0.423 | 0.009 | 4.1e-06 |
| WC | 322/1209 | 241/1209 | 0.266 | 0.199 | 0.687 (831) | 0.687 (830) | 263/322 | 201/241 | 0.817 | 0.834 | -0.017 | 0.414 | 0.471 | 2.7e-07 | 4.0e-07 |
| HAC | 171/1209 | 132/1209 | 0.141 | 0.109 | 0.634 (767) | 0.640 (774) | 109/171 | 93/132 | 0.637 | 0.705 | -0.067 | 0.008 | 0.179 | 0.467 | 0.062 |
| DA | 103/1209 | 243/1209 | 0.085 | 0.201 | 0.304 (368) | 0.308 (372) | 41/103 | 154/243 | 0.398 | 0.634 | -0.236 | 0.135 | 0.471 | 0.019 | 0 |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.354 / 0.289 | 0.153 / 0.202 | 0.266 / 0.199 | 0.141 / 0.109 | 0.085 / 0.201 |
| P(success by AG) | 0.719 / 0.715 | 0.693 / 0.687 | 0.687 / 0.687 | 0.634 / 0.640 | 0.304 / 0.308 |
| P(success by AG given AMP selected AG) | 0.689 / 0.822 | 0.773 / 0.820 | 0.817 / 0.834 | 0.637 / 0.705 | 0.398 / 0.634 |
| PRE | -0.105 / 0.377 | 0.260 / 0.423 | 0.414 / 0.471 | 0.008 / 0.179 | 0.135 / 0.471 |
| p (one-tailed) | 0.913 / 4.2e-06 | 0.009 / 4.1e-06 | 2.7e-07 / 4.0e-07 | 0.467 / 0.062 | 0.019 / 0 |

### General (n = 500; thesis n = 500)
Thesis side: TH Table 5.5 p.69 (counts: plan T11).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 244/500 | 230/500 | 0.488 | 0.460 | 0.826 (413) | 0.826 (413) | 186/244 | 195/230 | 0.762 | 0.848 | -0.086 | -0.366 | 0.125 | 0.996 | – |
| MR | 80/500 | 119/500 | 0.160 | 0.238 | 0.770 (385) | 0.770 (385) | 68/80 | 97/119 | 0.850 | 0.815 | +0.035 | 0.348 | 0.196 | 0.045 | – |
| WC | 150/500 | 82/500 | 0.300 | 0.164 | 0.760 (380) | 0.760 (380) | 139/150 | 71/82 | 0.927 | 0.866 | +0.061 | 0.694 | 0.441 | 8.8e-07 | – |
| HAC | 23/500 | 50/500 | 0.046 | 0.100 | 0.676 (338) | 0.676 (338) | 19/23 | 42/50 | 0.826 | 0.840 | -0.014 | 0.463 | 0.506 | 0.062 | – |
| DA | 3/500 | 19/500 | 0.006 | 0.038 | 0.222 (111) | 0.216 (108) | 2/3 | 7/19 | 0.667 | 0.368 | +0.298 | 0.572 | 0.194 | 0.032 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.488 / 0.460 | 0.160 / 0.238 | 0.300 / 0.164 | 0.046 / 0.100 | 0.006 / 0.038 |
| P(success by AG) | 0.826 / 0.826 | 0.770 / 0.770 | 0.760 / 0.760 | 0.676 / 0.676 | 0.222 / 0.216 |
| P(success by AG given AMP selected AG) | 0.762 / 0.848 | 0.850 / 0.815 | 0.927 / 0.866 | 0.826 / 0.840 | 0.667 / 0.368 |
| PRE | -0.366 / 0.125 | 0.348 / 0.196 | 0.694 / 0.441 | 0.463 / 0.506 | 0.572 / 0.194 |
| p (one-tailed) | 0.996 / – | 0.045 / – | 8.8e-07 / – | 0.062 / – | 0.032 / – |

### NBA (n = 393; thesis n = 393)
Thesis side: TH Table 5.5 p.69 (counts: plan T11).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 154/393 | 79/393 | 0.392 | 0.201 | 0.662 (260) | 0.656 (258) | 92/154 | 67/79 | 0.597 | 0.848 | -0.251 | -0.190 | 0.558 | 0.954 | – |
| MR | 47/393 | 78/393 | 0.120 | 0.198 | 0.651 (256) | 0.641 (252) | 31/47 | 76/78 | 0.660 | 0.974 | -0.315 | 0.023 | 0.929 | 0.453 | – |
| WC | 114/393 | 102/393 | 0.290 | 0.260 | 0.641 (252) | 0.644 (253) | 84/114 | 94/102 | 0.737 | 0.922 | -0.185 | 0.267 | 0.780 | 0.017 | – |
| HAC | 41/393 | 19/393 | 0.104 | 0.048 | 0.585 (230) | 0.603 (237) | 29/41 | 13/19 | 0.707 | 0.684 | +0.023 | 0.294 | 0.204 | 0.056 | – |
| DA | 37/393 | 115/393 | 0.094 | 0.293 | 0.349 (137) | 0.361 (142) | 12/37 | 111/115 | 0.324 | 0.965 | -0.641 | -0.037 | 0.946 | 0.622 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.392 / 0.201 | 0.120 / 0.198 | 0.290 / 0.260 | 0.104 / 0.048 | 0.094 / 0.293 |
| P(success by AG) | 0.662 / 0.656 | 0.651 / 0.641 | 0.641 / 0.644 | 0.585 / 0.603 | 0.349 / 0.361 |
| P(success by AG given AMP selected AG) | 0.597 / 0.848 | 0.660 / 0.974 | 0.737 / 0.922 | 0.707 / 0.684 | 0.324 / 0.965 |
| PRE | -0.190 / 0.558 | 0.023 / 0.929 | 0.267 / 0.780 | 0.294 / 0.204 | -0.037 / 0.946 |
| p (one-tailed) | 0.954 / – | 0.453 / – | 0.017 / – | 0.056 / – | 0.622 / – |

### NFL (n = 212; thesis n = 212)
Thesis side: TH Table 5.5 p.69 (counts: plan T11).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 19/212 | 24/212 | 0.090 | 0.113 | 0.528 (112) | 0.514 (109) | 9/19 | 12/24 | 0.474 | 0.500 | -0.026 | -0.116 | -0.029 | 0.683 | – |
| MR | 40/212 | 28/212 | 0.189 | 0.132 | 0.538 (114) | 0.524 (111) | 27/40 | 10/28 | 0.675 | 0.357 | +0.318 | 0.297 | -0.349 | 0.041 | – |
| WC | 42/212 | 49/212 | 0.198 | 0.231 | 0.552 (117) | 0.538 (114) | 24/42 | 28/49 | 0.571 | 0.571 | +0.000 | 0.044 | 0.073 | 0.399 | – |
| HAC | 69/212 | 37/212 | 0.325 | 0.175 | 0.618 (131) | 0.618 (131) | 40/69 | 22/37 | 0.580 | 0.595 | -0.015 | -0.100 | -0.061 | 0.743 | – |
| DA | 42/212 | 74/212 | 0.198 | 0.349 | 0.467 (99) | 0.476 (101) | 20/42 | 28/74 | 0.476 | 0.378 | +0.098 | 0.017 | -0.187 | 0.452 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.090 / 0.113 | 0.189 / 0.132 | 0.198 / 0.231 | 0.325 / 0.175 | 0.198 / 0.349 |
| P(success by AG) | 0.528 / 0.514 | 0.538 / 0.524 | 0.552 / 0.538 | 0.618 / 0.618 | 0.467 / 0.476 |
| P(success by AG given AMP selected AG) | 0.474 / 0.500 | 0.675 / 0.357 | 0.571 / 0.571 | 0.580 / 0.595 | 0.476 / 0.378 |
| PRE | -0.116 / -0.029 | 0.297 / -0.349 | 0.044 / 0.073 | -0.100 / -0.061 | 0.017 / -0.187 |
| p (one-tailed) | 0.683 / – | 0.041 / – | 0.399 / – | 0.743 / – | 0.452 / – |

### Politics (n = 104; thesis n = 104)
Thesis side: TH Table 5.5 p.69 (counts: plan T11).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 11/104 | 16/104 | 0.106 | 0.154 | 0.808 (84) | 0.808 (84) | 8/11 | 13/16 | 0.727 | 0.813 | -0.085 | -0.418 | 0.025 | 0.751 | – |
| MR | 18/104 | 19/104 | 0.173 | 0.183 | 0.798 (83) | 0.798 (83) | 17/18 | 17/19 | 0.944 | 0.895 | +0.050 | 0.725 | 0.479 | 0.061 | – |
| WC | 16/104 | 8/104 | 0.154 | 0.077 | 0.788 (82) | 0.798 (83) | 16/16 | 8/8 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | 0.019 | – |
| HAC | 38/104 | 26/104 | 0.365 | 0.250 | 0.654 (68) | 0.654 (68) | 21/38 | 16/26 | 0.553 | 0.615 | -0.063 | -0.292 | -0.111 | 0.905 | – |
| DA | 21/104 | 35/104 | 0.202 | 0.337 | 0.202 (21) | 0.202 (21) | 7/21 | 8/35 | 0.333 | 0.229 | +0.105 | 0.165 | 0.033 | 0.067 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.106 / 0.154 | 0.173 / 0.183 | 0.154 / 0.077 | 0.365 / 0.250 | 0.202 / 0.337 |
| P(success by AG) | 0.808 / 0.808 | 0.798 / 0.798 | 0.788 / 0.798 | 0.654 / 0.654 | 0.202 / 0.202 |
| P(success by AG given AMP selected AG) | 0.727 / 0.813 | 0.944 / 0.895 | 1.000 / 1.000 | 0.553 / 0.615 | 0.333 / 0.229 |
| PRE | -0.418 / 0.025 | 0.725 / 0.479 | 1.000 / 1.000 | -0.292 / -0.111 | 0.165 / 0.033 |
| p (one-tailed) | 0.751 / – | 0.061 / – | 0.019 / – | 0.905 / – | 0.067 / – |

### Qualitative checks (plan §9.3 row A15)

| check | run | thesis |
|---|---|---|
| methods whose global p is not below 0.05 | SP, HAC | HAC |
| domain with the highest mean PRE | General | NBA |
| mean PRE per domain | General 0.342, NBA 0.071, NFL 0.028, Politics 0.236 | General 0.293, NBA 0.683, NFL -0.111, Politics 0.285 |
| methods with a negative PRE, per domain | General: SP; NBA: SP, DA; NFL: SP, HAC; Politics: SP, HAC | General: none; NBA: none; NFL: SP, MR, HAC, DA; Politics: HAC |

Rates within ±0.03 of the thesis: 35 of 75 cells (P(selected), P(success), P(success given selected); all blocks).

## Success by domain, single leave-one-out [LOO, classifier seed 0]

Cells: rate (successes). Thesis side: TH Table A.9 p.115 (counts: plan T12).

| domain | n (run) | AMP: run | AMP: thesis | AMP: diff (cases) | DAP: run | DAP: thesis | DAP: diff (cases) |
|---|---|---|---|---|---|---|---|
| General | 500 | 0.828 (414) | 0.832 (416/500) | -2 | 0.850 (425) | 0.852 (426/500) | -1 |
| NBA | 393 | 0.623 (245) | 0.921 (362/393) | -117 | 0.606 (238) | 0.908 (357/393) | -119 |
| NFL | 212 | 0.552 (117) | 0.552 (117/212) | +0 | 0.575 (122) | 0.580 (123/212) | -1 |
| Politics | 104 | 0.760 (79) | 0.769 (80/104) | -1 | 0.712 (74) | 0.731 (76/104) | -2 |
| All cases | 1209 | 0.707 (855) | 0.806 (975/1209) | -120 | 0.711 (859) | 0.812 (982/1209) | -123 |

## Effectiveness of the AMP selection, single leave-one-out [LOO, classifier seed 0] - TH Table A.10 (and MLA Figs 2-3 for the global block)

P(success by AG) is the rule applied to every case of the block: the recomputed rule baseline beside the thesis's. PRE from exact counts; p = one-sample one-tailed z-test (plan §5.8 item 3), exact upper tail; the thesis prints p for the global block only.

### All cases (n = 1209; thesis n = 1209)
Thesis side: MLA Figs 2-3 (counts: plan T20/T21; no global LOO table in the thesis).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 390/1209 | 422/1209 | 0.323 | 0.349 | 0.719 (869) | 0.715 (864) | 266/390 | 351/422 | 0.682 | 0.832 | -0.150 | -0.131 | 0.410 | 0.947 | 5.0e-08 |
| MR | 307/1209 | 268/1209 | 0.254 | 0.222 | 0.693 (838) | 0.687 (831) | 239/307 | 222/268 | 0.779 | 0.828 | -0.050 | 0.278 | 0.451 | 5.9e-04 | 3.2e-07 |
| WC | 221/1209 | 185/1209 | 0.183 | 0.153 | 0.687 (831) | 0.687 (830) | 165/221 | 149/185 | 0.747 | 0.805 | -0.059 | 0.190 | 0.379 | 0.029 | 2.5e-04 |
| HAC | 221/1209 | 180/1209 | 0.183 | 0.149 | 0.634 (767) | 0.640 (774) | 151/221 | 127/180 | 0.683 | 0.706 | -0.022 | 0.134 | 0.182 | 0.066 | 0.034 |
| DA | 70/1209 | 154/1209 | 0.058 | 0.127 | 0.304 (368) | 0.308 (372) | 34/70 | 126/154 | 0.486 | 0.818 | -0.332 | 0.261 | 0.737 | 4.9e-04 | 0 |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.323 / 0.349 | 0.254 / 0.222 | 0.183 / 0.153 | 0.183 / 0.149 | 0.058 / 0.127 |
| P(success by AG) | 0.719 / 0.715 | 0.693 / 0.687 | 0.687 / 0.687 | 0.634 / 0.640 | 0.304 / 0.308 |
| P(success by AG given AMP selected AG) | 0.682 / 0.832 | 0.779 / 0.828 | 0.747 / 0.805 | 0.683 / 0.706 | 0.486 / 0.818 |
| PRE | -0.131 / 0.410 | 0.278 / 0.451 | 0.190 / 0.379 | 0.134 / 0.182 | 0.261 / 0.737 |
| p (one-tailed) | 0.947 / 5.0e-08 | 5.9e-04 / 3.2e-07 | 0.029 / 2.5e-04 | 0.066 / 0.034 | 4.9e-04 / 0 |

### General (n = 500; thesis n = 500)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 195/500 | 261/500 | 0.390 | 0.522 | 0.826 (413) | 0.826 (413) | 153/195 | 221/261 | 0.785 | 0.847 | -0.062 | -0.238 | 0.119 | 0.936 | – |
| MR | 126/500 | 83/500 | 0.252 | 0.166 | 0.770 (385) | 0.770 (385) | 116/126 | 72/83 | 0.921 | 0.867 | +0.053 | 0.655 | 0.424 | 2.9e-05 | – |
| WC | 82/500 | 83/500 | 0.164 | 0.166 | 0.760 (380) | 0.760 (380) | 70/82 | 70/83 | 0.854 | 0.843 | +0.010 | 0.390 | 0.347 | 0.024 | – |
| HAC | 77/500 | 65/500 | 0.154 | 0.130 | 0.676 (338) | 0.676 (338) | 65/77 | 50/65 | 0.844 | 0.769 | +0.075 | 0.519 | 0.288 | 8.1e-04 | – |
| DA | 20/500 | 8/500 | 0.040 | 0.016 | 0.222 (111) | 0.216 (108) | 10/20 | 3/8 | 0.500 | 0.375 | +0.125 | 0.357 | 0.203 | 0.001 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.390 / 0.522 | 0.252 / 0.166 | 0.164 / 0.166 | 0.154 / 0.130 | 0.040 / 0.016 |
| P(success by AG) | 0.826 / 0.826 | 0.770 / 0.770 | 0.760 / 0.760 | 0.676 / 0.676 | 0.222 / 0.216 |
| P(success by AG given AMP selected AG) | 0.785 / 0.847 | 0.921 / 0.867 | 0.854 / 0.843 | 0.844 / 0.769 | 0.500 / 0.375 |
| PRE | -0.238 / 0.119 | 0.655 / 0.424 | 0.390 / 0.347 | 0.519 / 0.288 | 0.357 / 0.203 |
| p (one-tailed) | 0.936 / – | 2.9e-05 / – | 0.024 / – | 8.1e-04 / – | 0.001 / – |

### NBA (n = 393; thesis n = 393)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 107/393 | 91/393 | 0.272 | 0.232 | 0.662 (260) | 0.656 (258) | 66/107 | 82/91 | 0.617 | 0.901 | -0.284 | -0.132 | 0.712 | 0.836 | – |
| MR | 117/393 | 107/393 | 0.298 | 0.272 | 0.651 (256) | 0.641 (252) | 77/117 | 100/107 | 0.658 | 0.935 | -0.276 | 0.019 | 0.818 | 0.439 | – |
| WC | 76/393 | 54/393 | 0.193 | 0.137 | 0.641 (252) | 0.644 (253) | 49/76 | 49/54 | 0.645 | 0.907 | -0.263 | 0.010 | 0.740 | 0.475 | – |
| HAC | 67/393 | 21/393 | 0.170 | 0.053 | 0.585 (230) | 0.603 (237) | 44/67 | 16/21 | 0.657 | 0.762 | -0.105 | 0.172 | 0.400 | 0.118 | – |
| DA | 26/393 | 120/393 | 0.066 | 0.305 | 0.349 (137) | 0.361 (142) | 9/26 | 115/120 | 0.346 | 0.958 | -0.612 | -0.004 | 0.935 | 0.510 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.272 / 0.232 | 0.298 / 0.272 | 0.193 / 0.137 | 0.170 / 0.053 | 0.066 / 0.305 |
| P(success by AG) | 0.662 / 0.656 | 0.651 / 0.641 | 0.641 / 0.644 | 0.585 / 0.603 | 0.349 / 0.361 |
| P(success by AG given AMP selected AG) | 0.617 / 0.901 | 0.658 / 0.935 | 0.645 / 0.907 | 0.657 / 0.762 | 0.346 / 0.958 |
| PRE | -0.132 / 0.712 | 0.019 / 0.818 | 0.010 / 0.740 | 0.172 / 0.400 | -0.004 / 0.935 |
| p (one-tailed) | 0.836 / – | 0.439 / – | 0.475 / – | 0.118 / – | 0.510 / – |

### NFL (n = 212; thesis n = 212)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 52/212 | 41/212 | 0.245 | 0.193 | 0.528 (112) | 0.514 (109) | 23/52 | 25/41 | 0.442 | 0.610 | -0.167 | -0.182 | 0.197 | 0.893 | – |
| MR | 37/212 | 45/212 | 0.175 | 0.212 | 0.538 (114) | 0.524 (111) | 21/37 | 22/45 | 0.568 | 0.489 | +0.079 | 0.065 | -0.073 | 0.358 | – |
| WC | 43/212 | 30/212 | 0.203 | 0.142 | 0.552 (117) | 0.538 (114) | 28/43 | 16/30 | 0.651 | 0.533 | +0.118 | 0.222 | -0.010 | 0.095 | – |
| HAC | 58/212 | 73/212 | 0.274 | 0.344 | 0.618 (131) | 0.618 (131) | 30/58 | 46/73 | 0.517 | 0.630 | -0.113 | -0.264 | 0.032 | 0.943 | – |
| DA | 22/212 | 23/212 | 0.104 | 0.108 | 0.467 (99) | 0.476 (101) | 15/22 | 8/23 | 0.682 | 0.348 | +0.334 | 0.403 | -0.246 | 0.022 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.245 / 0.193 | 0.175 / 0.212 | 0.203 / 0.142 | 0.274 / 0.344 | 0.104 / 0.108 |
| P(success by AG) | 0.528 / 0.514 | 0.538 / 0.524 | 0.552 / 0.538 | 0.618 / 0.618 | 0.467 / 0.476 |
| P(success by AG given AMP selected AG) | 0.442 / 0.610 | 0.568 / 0.489 | 0.651 / 0.533 | 0.517 / 0.630 | 0.682 / 0.348 |
| PRE | -0.182 / 0.197 | 0.065 / -0.073 | 0.222 / -0.010 | -0.264 / 0.032 | 0.403 / -0.246 |
| p (one-tailed) | 0.893 / – | 0.358 / – | 0.095 / – | 0.943 / – | 0.022 / – |

### Politics (n = 104; thesis n = 104)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 36/104 | 29/104 | 0.346 | 0.279 | 0.808 (84) | 0.808 (84) | 24/36 | 23/29 | 0.667 | 0.793 | -0.126 | -0.733 | -0.076 | 0.984 | – |
| MR | 27/104 | 33/104 | 0.260 | 0.317 | 0.798 (83) | 0.798 (83) | 25/27 | 28/33 | 0.926 | 0.848 | +0.077 | 0.633 | 0.250 | 0.049 | – |
| WC | 20/104 | 18/104 | 0.192 | 0.173 | 0.788 (82) | 0.798 (83) | 18/20 | 14/18 | 0.900 | 0.778 | +0.122 | 0.527 | -0.101 | 0.111 | – |
| HAC | 19/104 | 21/104 | 0.183 | 0.202 | 0.654 (68) | 0.654 (68) | 12/19 | 15/21 | 0.632 | 0.714 | -0.083 | -0.064 | 0.175 | 0.581 | – |
| DA | 2/104 | 3/104 | 0.019 | 0.029 | 0.202 (21) | 0.202 (21) | 0/2 | 0/3 | 0.000 | 0.000 | +0.000 | -0.253 | -0.253 | 0.762 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.346 / 0.279 | 0.260 / 0.317 | 0.192 / 0.173 | 0.183 / 0.202 | 0.019 / 0.029 |
| P(success by AG) | 0.808 / 0.808 | 0.798 / 0.798 | 0.788 / 0.798 | 0.654 / 0.654 | 0.202 / 0.202 |
| P(success by AG given AMP selected AG) | 0.667 / 0.793 | 0.926 / 0.848 | 0.900 / 0.778 | 0.632 / 0.714 | 0.000 / 0.000 |
| PRE | -0.733 / -0.076 | 0.633 / 0.250 | 0.527 / -0.101 | -0.064 / 0.175 | -0.253 / -0.253 |
| p (one-tailed) | 0.984 / – | 0.049 / – | 0.111 / – | 0.581 / – | 0.762 / – |

### Qualitative checks (plan §9.3 row A15)

| check | run | thesis |
|---|---|---|
| methods whose global p is not below 0.05 | SP, HAC | none |
| domain with the highest mean PRE | General | NBA |
| mean PRE per domain | General 0.337, NBA 0.013, NFL 0.049, Politics 0.022 | General 0.276, NBA 0.721, NFL -0.020, Politics -0.001 |
| methods with a negative PRE, per domain | General: SP; NBA: SP, DA; NFL: SP, HAC; Politics: SP, HAC, DA | General: none; NBA: none; NFL: MR, WC, DA; Politics: SP, WC, DA |

Rates within ±0.03 of the thesis: 38 of 75 cells (P(selected), P(success), P(success given selected); all blocks).

<details><summary>Single LOO, classifier seed 1 (seed noise)</summary>

## Success by domain, single leave-one-out [LOO, classifier seed 1]

Cells: rate (successes). Thesis side: TH Table A.9 p.115 (counts: plan T12).

| domain | n (run) | AMP: run | AMP: thesis | AMP: diff (cases) | DAP: run | DAP: thesis | DAP: diff (cases) |
|---|---|---|---|---|---|---|---|
| General | 500 | 0.814 (407) | 0.832 (416/500) | -9 | 0.840 (420) | 0.852 (426/500) | -6 |
| NBA | 393 | 0.646 (254) | 0.921 (362/393) | -108 | 0.611 (240) | 0.908 (357/393) | -117 |
| NFL | 212 | 0.571 (121) | 0.552 (117/212) | +4 | 0.575 (122) | 0.580 (123/212) | -1 |
| Politics | 104 | 0.750 (78) | 0.769 (80/104) | -2 | 0.721 (75) | 0.731 (76/104) | -1 |
| All cases | 1209 | 0.711 (860) | 0.806 (975/1209) | -115 | 0.709 (857) | 0.812 (982/1209) | -125 |

## Effectiveness of the AMP selection, single leave-one-out [LOO, classifier seed 1] - TH Table A.10 (and MLA Figs 2-3 for the global block)

P(success by AG) is the rule applied to every case of the block: the recomputed rule baseline beside the thesis's. PRE from exact counts; p = one-sample one-tailed z-test (plan §5.8 item 3), exact upper tail; the thesis prints p for the global block only.

### All cases (n = 1209; thesis n = 1209)
Thesis side: MLA Figs 2-3 (counts: plan T20/T21; no global LOO table in the thesis).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 375/1209 | 422/1209 | 0.310 | 0.349 | 0.719 (869) | 0.715 (864) | 268/375 | 351/422 | 0.715 | 0.832 | -0.117 | -0.015 | 0.410 | 0.570 | 5.0e-08 |
| MR | 318/1209 | 268/1209 | 0.263 | 0.222 | 0.693 (838) | 0.687 (831) | 231/318 | 222/268 | 0.726 | 0.828 | -0.102 | 0.108 | 0.451 | 0.099 | 3.2e-07 |
| WC | 222/1209 | 185/1209 | 0.184 | 0.153 | 0.687 (831) | 0.687 (830) | 163/222 | 149/185 | 0.734 | 0.805 | -0.071 | 0.150 | 0.379 | 0.066 | 2.5e-04 |
| HAC | 234/1209 | 180/1209 | 0.194 | 0.149 | 0.634 (767) | 0.640 (774) | 165/234 | 127/180 | 0.705 | 0.706 | -0.000 | 0.193 | 0.182 | 0.012 | 0.034 |
| DA | 60/1209 | 154/1209 | 0.050 | 0.127 | 0.304 (368) | 0.308 (372) | 33/60 | 126/154 | 0.550 | 0.818 | -0.268 | 0.353 | 0.737 | 1.8e-05 | 0 |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.310 / 0.349 | 0.263 / 0.222 | 0.184 / 0.153 | 0.194 / 0.149 | 0.050 / 0.127 |
| P(success by AG) | 0.719 / 0.715 | 0.693 / 0.687 | 0.687 / 0.687 | 0.634 / 0.640 | 0.304 / 0.308 |
| P(success by AG given AMP selected AG) | 0.715 / 0.832 | 0.726 / 0.828 | 0.734 / 0.805 | 0.705 / 0.706 | 0.550 / 0.818 |
| PRE | -0.015 / 0.410 | 0.108 / 0.451 | 0.150 / 0.379 | 0.193 / 0.182 | 0.353 / 0.737 |
| p (one-tailed) | 0.570 / 5.0e-08 | 0.099 / 3.2e-07 | 0.066 / 2.5e-04 | 0.012 / 0.034 | 1.8e-05 / 0 |

### General (n = 500; thesis n = 500)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 204/500 | 261/500 | 0.408 | 0.522 | 0.826 (413) | 0.826 (413) | 160/204 | 221/261 | 0.784 | 0.847 | -0.062 | -0.240 | 0.119 | 0.942 | – |
| MR | 115/500 | 83/500 | 0.230 | 0.166 | 0.770 (385) | 0.770 (385) | 97/115 | 72/83 | 0.843 | 0.867 | -0.024 | 0.319 | 0.424 | 0.031 | – |
| WC | 87/500 | 83/500 | 0.174 | 0.166 | 0.760 (380) | 0.760 (380) | 74/87 | 70/83 | 0.851 | 0.843 | +0.007 | 0.377 | 0.347 | 0.024 | – |
| HAC | 77/500 | 65/500 | 0.154 | 0.130 | 0.676 (338) | 0.676 (338) | 66/77 | 50/65 | 0.857 | 0.769 | +0.088 | 0.559 | 0.288 | 3.4e-04 | – |
| DA | 17/500 | 8/500 | 0.034 | 0.016 | 0.222 (111) | 0.216 (108) | 10/17 | 3/8 | 0.588 | 0.375 | +0.213 | 0.471 | 0.203 | 1.4e-04 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.408 / 0.522 | 0.230 / 0.166 | 0.174 / 0.166 | 0.154 / 0.130 | 0.034 / 0.016 |
| P(success by AG) | 0.826 / 0.826 | 0.770 / 0.770 | 0.760 / 0.760 | 0.676 / 0.676 | 0.222 / 0.216 |
| P(success by AG given AMP selected AG) | 0.784 / 0.847 | 0.843 / 0.867 | 0.851 / 0.843 | 0.857 / 0.769 | 0.588 / 0.375 |
| PRE | -0.240 / 0.119 | 0.319 / 0.424 | 0.377 / 0.347 | 0.559 / 0.288 | 0.471 / 0.203 |
| p (one-tailed) | 0.942 / – | 0.031 / – | 0.024 / – | 3.4e-04 / – | 1.4e-04 / – |

### NBA (n = 393; thesis n = 393)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 108/393 | 91/393 | 0.275 | 0.232 | 0.662 (260) | 0.656 (258) | 71/108 | 82/91 | 0.657 | 0.901 | -0.244 | -0.012 | 0.712 | 0.536 | – |
| MR | 117/393 | 107/393 | 0.298 | 0.272 | 0.651 (256) | 0.641 (252) | 74/117 | 100/107 | 0.632 | 0.935 | -0.302 | -0.054 | 0.818 | 0.666 | – |
| WC | 79/393 | 54/393 | 0.201 | 0.137 | 0.641 (252) | 0.644 (253) | 48/79 | 49/54 | 0.608 | 0.907 | -0.300 | -0.094 | 0.740 | 0.733 | – |
| HAC | 71/393 | 21/393 | 0.181 | 0.053 | 0.585 (230) | 0.603 (237) | 52/71 | 16/21 | 0.732 | 0.762 | -0.030 | 0.355 | 0.400 | 0.006 | – |
| DA | 18/393 | 120/393 | 0.046 | 0.305 | 0.349 (137) | 0.361 (142) | 9/18 | 115/120 | 0.500 | 0.958 | -0.458 | 0.232 | 0.935 | 0.089 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.275 / 0.232 | 0.298 / 0.272 | 0.201 / 0.137 | 0.181 / 0.053 | 0.046 / 0.305 |
| P(success by AG) | 0.662 / 0.656 | 0.651 / 0.641 | 0.641 / 0.644 | 0.585 / 0.603 | 0.349 / 0.361 |
| P(success by AG given AMP selected AG) | 0.657 / 0.901 | 0.632 / 0.935 | 0.608 / 0.907 | 0.732 / 0.762 | 0.500 / 0.958 |
| PRE | -0.012 / 0.712 | -0.054 / 0.818 | -0.094 / 0.740 | 0.355 / 0.400 | 0.232 / 0.935 |
| p (one-tailed) | 0.536 / – | 0.666 / – | 0.733 / – | 0.006 / – | 0.089 / – |

### NFL (n = 212; thesis n = 212)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 30/212 | 41/212 | 0.142 | 0.193 | 0.528 (112) | 0.514 (109) | 15/30 | 25/41 | 0.500 | 0.610 | -0.110 | -0.060 | 0.197 | 0.622 | – |
| MR | 61/212 | 45/212 | 0.288 | 0.212 | 0.538 (114) | 0.524 (111) | 36/61 | 22/45 | 0.590 | 0.489 | +0.101 | 0.113 | -0.073 | 0.206 | – |
| WC | 33/212 | 30/212 | 0.156 | 0.142 | 0.552 (117) | 0.538 (114) | 21/33 | 16/30 | 0.636 | 0.533 | +0.103 | 0.189 | -0.010 | 0.165 | – |
| HAC | 65/212 | 73/212 | 0.307 | 0.344 | 0.618 (131) | 0.618 (131) | 35/65 | 46/73 | 0.538 | 0.630 | -0.092 | -0.208 | 0.032 | 0.906 | – |
| DA | 23/212 | 23/212 | 0.108 | 0.108 | 0.467 (99) | 0.476 (101) | 14/23 | 8/23 | 0.609 | 0.348 | +0.261 | 0.266 | -0.246 | 0.087 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.142 / 0.193 | 0.288 / 0.212 | 0.156 / 0.142 | 0.307 / 0.344 | 0.108 / 0.108 |
| P(success by AG) | 0.528 / 0.514 | 0.538 / 0.524 | 0.552 / 0.538 | 0.618 / 0.618 | 0.467 / 0.476 |
| P(success by AG given AMP selected AG) | 0.500 / 0.610 | 0.590 / 0.489 | 0.636 / 0.533 | 0.538 / 0.630 | 0.609 / 0.348 |
| PRE | -0.060 / 0.197 | 0.113 / -0.073 | 0.189 / -0.010 | -0.208 / 0.032 | 0.266 / -0.246 |
| p (one-tailed) | 0.622 / – | 0.206 / – | 0.165 / – | 0.906 / – | 0.087 / – |

### Politics (n = 104; thesis n = 104)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 33/104 | 29/104 | 0.317 | 0.279 | 0.808 (84) | 0.808 (84) | 22/33 | 23/29 | 0.667 | 0.793 | -0.126 | -0.733 | -0.076 | 0.980 | – |
| MR | 25/104 | 33/104 | 0.240 | 0.317 | 0.798 (83) | 0.798 (83) | 24/25 | 28/33 | 0.960 | 0.848 | +0.112 | 0.802 | 0.250 | 0.022 | – |
| WC | 23/104 | 18/104 | 0.221 | 0.173 | 0.788 (82) | 0.798 (83) | 20/23 | 14/18 | 0.870 | 0.778 | +0.092 | 0.383 | -0.101 | 0.170 | – |
| HAC | 21/104 | 21/104 | 0.202 | 0.202 | 0.654 (68) | 0.654 (68) | 12/21 | 15/21 | 0.571 | 0.714 | -0.143 | -0.238 | 0.175 | 0.786 | – |
| DA | 2/104 | 3/104 | 0.019 | 0.029 | 0.202 (21) | 0.202 (21) | 0/2 | 0/3 | 0.000 | 0.000 | +0.000 | -0.253 | -0.253 | 0.762 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.317 / 0.279 | 0.240 / 0.317 | 0.221 / 0.173 | 0.202 / 0.202 | 0.019 / 0.029 |
| P(success by AG) | 0.808 / 0.808 | 0.798 / 0.798 | 0.788 / 0.798 | 0.654 / 0.654 | 0.202 / 0.202 |
| P(success by AG given AMP selected AG) | 0.667 / 0.793 | 0.960 / 0.848 | 0.870 / 0.778 | 0.571 / 0.714 | 0.000 / 0.000 |
| PRE | -0.733 / -0.076 | 0.802 / 0.250 | 0.383 / -0.101 | -0.238 / 0.175 | -0.253 / -0.253 |
| p (one-tailed) | 0.980 / – | 0.022 / – | 0.170 / – | 0.786 / – | 0.762 / – |

### Qualitative checks (plan §9.3 row A15)

| check | run | thesis |
|---|---|---|
| methods whose global p is not below 0.05 | SP, MR, WC | none |
| domain with the highest mean PRE | General | NBA |
| mean PRE per domain | General 0.297, NBA 0.085, NFL 0.060, Politics -0.008 | General 0.276, NBA 0.721, NFL -0.020, Politics -0.001 |
| methods with a negative PRE, per domain | General: SP; NBA: SP, MR, WC; NFL: SP, HAC; Politics: SP, HAC, DA | General: none; NBA: none; NFL: MR, WC, DA; Politics: SP, WC, DA |

Rates within ±0.03 of the thesis: 38 of 75 cells (P(selected), P(success), P(success given selected); all blocks).

</details>

<details><summary>Single LOO, classifier seed 2 (seed noise)</summary>

## Success by domain, single leave-one-out [LOO, classifier seed 2]

Cells: rate (successes). Thesis side: TH Table A.9 p.115 (counts: plan T12).

| domain | n (run) | AMP: run | AMP: thesis | AMP: diff (cases) | DAP: run | DAP: thesis | DAP: diff (cases) |
|---|---|---|---|---|---|---|---|
| General | 500 | 0.824 (412) | 0.832 (416/500) | -4 | 0.850 (425) | 0.852 (426/500) | -1 |
| NBA | 393 | 0.636 (250) | 0.921 (362/393) | -112 | 0.616 (242) | 0.908 (357/393) | -115 |
| NFL | 212 | 0.594 (126) | 0.552 (117/212) | +9 | 0.580 (123) | 0.580 (123/212) | +0 |
| Politics | 104 | 0.721 (75) | 0.769 (80/104) | -5 | 0.750 (78) | 0.731 (76/104) | +2 |
| All cases | 1209 | 0.714 (863) | 0.806 (975/1209) | -112 | 0.718 (868) | 0.812 (982/1209) | -114 |

## Effectiveness of the AMP selection, single leave-one-out [LOO, classifier seed 2] - TH Table A.10 (and MLA Figs 2-3 for the global block)

P(success by AG) is the rule applied to every case of the block: the recomputed rule baseline beside the thesis's. PRE from exact counts; p = one-sample one-tailed z-test (plan §5.8 item 3), exact upper tail; the thesis prints p for the global block only.

### All cases (n = 1209; thesis n = 1209)
Thesis side: MLA Figs 2-3 (counts: plan T20/T21; no global LOO table in the thesis).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 363/1209 | 422/1209 | 0.300 | 0.349 | 0.719 (869) | 0.715 (864) | 260/363 | 351/422 | 0.716 | 0.832 | -0.116 | -0.009 | 0.410 | 0.543 | 5.0e-08 |
| MR | 298/1209 | 268/1209 | 0.246 | 0.222 | 0.693 (838) | 0.687 (831) | 230/298 | 222/268 | 0.772 | 0.828 | -0.057 | 0.256 | 0.451 | 0.002 | 3.2e-07 |
| WC | 231/1209 | 185/1209 | 0.191 | 0.153 | 0.687 (831) | 0.687 (830) | 155/231 | 149/185 | 0.671 | 0.805 | -0.134 | -0.052 | 0.379 | 0.704 | 2.5e-04 |
| HAC | 257/1209 | 180/1209 | 0.213 | 0.149 | 0.634 (767) | 0.640 (774) | 187/257 | 127/180 | 0.728 | 0.706 | +0.022 | 0.255 | 0.182 | 9.6e-04 | 0.034 |
| DA | 60/1209 | 154/1209 | 0.050 | 0.127 | 0.304 (368) | 0.308 (372) | 31/60 | 126/154 | 0.517 | 0.818 | -0.302 | 0.305 | 0.737 | 1.8e-04 | 0 |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.300 / 0.349 | 0.246 / 0.222 | 0.191 / 0.153 | 0.213 / 0.149 | 0.050 / 0.127 |
| P(success by AG) | 0.719 / 0.715 | 0.693 / 0.687 | 0.687 / 0.687 | 0.634 / 0.640 | 0.304 / 0.308 |
| P(success by AG given AMP selected AG) | 0.716 / 0.832 | 0.772 / 0.828 | 0.671 / 0.805 | 0.728 / 0.706 | 0.517 / 0.818 |
| PRE | -0.009 / 0.410 | 0.256 / 0.451 | -0.052 / 0.379 | 0.255 / 0.182 | 0.305 / 0.737 |
| p (one-tailed) | 0.543 / 5.0e-08 | 0.002 / 3.2e-07 | 0.704 / 2.5e-04 | 9.6e-04 / 0.034 | 1.8e-04 / 0 |

### General (n = 500; thesis n = 500)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 193/500 | 261/500 | 0.386 | 0.522 | 0.826 (413) | 0.826 (413) | 149/193 | 221/261 | 0.772 | 0.847 | -0.075 | -0.310 | 0.119 | 0.976 | – |
| MR | 126/500 | 83/500 | 0.252 | 0.166 | 0.770 (385) | 0.770 (385) | 117/126 | 72/83 | 0.929 | 0.867 | +0.061 | 0.689 | 0.424 | 1.2e-05 | – |
| WC | 76/500 | 83/500 | 0.152 | 0.166 | 0.760 (380) | 0.760 (380) | 63/76 | 70/83 | 0.829 | 0.843 | -0.014 | 0.287 | 0.347 | 0.080 | – |
| HAC | 89/500 | 65/500 | 0.178 | 0.130 | 0.676 (338) | 0.676 (338) | 72/89 | 50/65 | 0.809 | 0.769 | +0.040 | 0.410 | 0.288 | 0.004 | – |
| DA | 16/500 | 8/500 | 0.032 | 0.016 | 0.222 (111) | 0.216 (108) | 11/16 | 3/8 | 0.688 | 0.375 | +0.312 | 0.598 | 0.203 | 3.7e-06 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.386 / 0.522 | 0.252 / 0.166 | 0.152 / 0.166 | 0.178 / 0.130 | 0.032 / 0.016 |
| P(success by AG) | 0.826 / 0.826 | 0.770 / 0.770 | 0.760 / 0.760 | 0.676 / 0.676 | 0.222 / 0.216 |
| P(success by AG given AMP selected AG) | 0.772 / 0.847 | 0.929 / 0.867 | 0.829 / 0.843 | 0.809 / 0.769 | 0.688 / 0.375 |
| PRE | -0.310 / 0.119 | 0.689 / 0.424 | 0.287 / 0.347 | 0.410 / 0.288 | 0.598 / 0.203 |
| p (one-tailed) | 0.976 / – | 1.2e-05 / – | 0.080 / – | 0.004 / – | 3.7e-06 / – |

### NBA (n = 393; thesis n = 393)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 110/393 | 91/393 | 0.280 | 0.232 | 0.662 (260) | 0.656 (258) | 74/110 | 82/91 | 0.673 | 0.901 | -0.228 | 0.033 | 0.712 | 0.402 | – |
| MR | 94/393 | 107/393 | 0.239 | 0.272 | 0.651 (256) | 0.641 (252) | 56/94 | 100/107 | 0.596 | 0.935 | -0.339 | -0.160 | 0.818 | 0.871 | – |
| WC | 91/393 | 54/393 | 0.232 | 0.137 | 0.641 (252) | 0.644 (253) | 53/91 | 49/54 | 0.582 | 0.907 | -0.325 | -0.164 | 0.740 | 0.879 | – |
| HAC | 71/393 | 21/393 | 0.181 | 0.053 | 0.585 (230) | 0.603 (237) | 56/71 | 16/21 | 0.789 | 0.762 | +0.027 | 0.491 | 0.400 | 2.5e-04 | – |
| DA | 27/393 | 120/393 | 0.069 | 0.305 | 0.349 (137) | 0.361 (142) | 11/27 | 115/120 | 0.407 | 0.958 | -0.551 | 0.090 | 0.935 | 0.261 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.280 / 0.232 | 0.239 / 0.272 | 0.232 / 0.137 | 0.181 / 0.053 | 0.069 / 0.305 |
| P(success by AG) | 0.662 / 0.656 | 0.651 / 0.641 | 0.641 / 0.644 | 0.585 / 0.603 | 0.349 / 0.361 |
| P(success by AG given AMP selected AG) | 0.673 / 0.901 | 0.596 / 0.935 | 0.582 / 0.907 | 0.789 / 0.762 | 0.407 / 0.958 |
| PRE | 0.033 / 0.712 | -0.160 / 0.818 | -0.164 / 0.740 | 0.491 / 0.400 | 0.090 / 0.935 |
| p (one-tailed) | 0.402 / – | 0.871 / – | 0.879 / – | 2.5e-04 / – | 0.261 / – |

### NFL (n = 212; thesis n = 212)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 31/212 | 41/212 | 0.146 | 0.193 | 0.528 (112) | 0.514 (109) | 17/31 | 25/41 | 0.548 | 0.610 | -0.061 | 0.043 | 0.197 | 0.411 | – |
| MR | 47/212 | 45/212 | 0.222 | 0.212 | 0.538 (114) | 0.524 (111) | 31/47 | 22/45 | 0.660 | 0.489 | +0.171 | 0.264 | -0.073 | 0.047 | – |
| WC | 43/212 | 30/212 | 0.203 | 0.142 | 0.552 (117) | 0.538 (114) | 24/43 | 16/30 | 0.558 | 0.533 | +0.025 | 0.014 | -0.010 | 0.467 | – |
| HAC | 77/212 | 73/212 | 0.363 | 0.344 | 0.618 (131) | 0.618 (131) | 45/77 | 46/73 | 0.584 | 0.630 | -0.046 | -0.088 | 0.032 | 0.727 | – |
| DA | 14/212 | 23/212 | 0.066 | 0.108 | 0.467 (99) | 0.476 (101) | 9/14 | 8/23 | 0.643 | 0.348 | +0.295 | 0.330 | -0.246 | 0.094 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.146 / 0.193 | 0.222 / 0.212 | 0.203 / 0.142 | 0.363 / 0.344 | 0.066 / 0.108 |
| P(success by AG) | 0.528 / 0.514 | 0.538 / 0.524 | 0.552 / 0.538 | 0.618 / 0.618 | 0.467 / 0.476 |
| P(success by AG given AMP selected AG) | 0.548 / 0.610 | 0.660 / 0.489 | 0.558 / 0.533 | 0.584 / 0.630 | 0.643 / 0.348 |
| PRE | 0.043 / 0.197 | 0.264 / -0.073 | 0.014 / -0.010 | -0.088 / 0.032 | 0.330 / -0.246 |
| p (one-tailed) | 0.411 / – | 0.047 / – | 0.467 / – | 0.727 / – | 0.094 / – |

### Politics (n = 104; thesis n = 104)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 29/104 | 29/104 | 0.279 | 0.279 | 0.808 (84) | 0.808 (84) | 20/29 | 23/29 | 0.690 | 0.793 | -0.103 | -0.614 | -0.076 | 0.947 | – |
| MR | 31/104 | 33/104 | 0.298 | 0.317 | 0.798 (83) | 0.798 (83) | 26/31 | 28/33 | 0.839 | 0.848 | -0.010 | 0.201 | 0.250 | 0.287 | – |
| WC | 21/104 | 18/104 | 0.202 | 0.173 | 0.788 (82) | 0.798 (83) | 15/21 | 14/18 | 0.714 | 0.778 | -0.063 | -0.351 | -0.101 | 0.797 | – |
| HAC | 20/104 | 21/104 | 0.192 | 0.202 | 0.654 (68) | 0.654 (68) | 14/20 | 15/21 | 0.700 | 0.714 | -0.014 | 0.133 | 0.175 | 0.332 | – |
| DA | 3/104 | 3/104 | 0.029 | 0.029 | 0.202 (21) | 0.202 (21) | 0/3 | 0/3 | 0.000 | 0.000 | +0.000 | -0.253 | -0.253 | 0.808 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.279 / 0.279 | 0.298 / 0.317 | 0.202 / 0.173 | 0.192 / 0.202 | 0.029 / 0.029 |
| P(success by AG) | 0.808 / 0.808 | 0.798 / 0.798 | 0.788 / 0.798 | 0.654 / 0.654 | 0.202 / 0.202 |
| P(success by AG given AMP selected AG) | 0.690 / 0.793 | 0.839 / 0.848 | 0.714 / 0.778 | 0.700 / 0.714 | 0.000 / 0.000 |
| PRE | -0.614 / -0.076 | 0.201 / 0.250 | -0.351 / -0.101 | 0.133 / 0.175 | -0.253 / -0.253 |
| p (one-tailed) | 0.947 / – | 0.287 / – | 0.797 / – | 0.332 / – | 0.808 / – |

### Qualitative checks (plan §9.3 row A15)

| check | run | thesis |
|---|---|---|
| methods whose global p is not below 0.05 | SP, WC | none |
| domain with the highest mean PRE | General | NBA |
| mean PRE per domain | General 0.335, NBA 0.058, NFL 0.112, Politics -0.177 | General 0.276, NBA 0.721, NFL -0.020, Politics -0.001 |
| methods with a negative PRE, per domain | General: SP; NBA: MR, WC; NFL: HAC; Politics: SP, WC, DA | General: none; NBA: none; NFL: MR, WC, DA; Politics: SP, WC, DA |

Rates within ±0.03 of the thesis: 42 of 75 cells (P(selected), P(success), P(success given selected); all blocks).

</details>

<details><summary>Single LOO, classifier seed 3 (seed noise)</summary>

## Success by domain, single leave-one-out [LOO, classifier seed 3]

Cells: rate (successes). Thesis side: TH Table A.9 p.115 (counts: plan T12).

| domain | n (run) | AMP: run | AMP: thesis | AMP: diff (cases) | DAP: run | DAP: thesis | DAP: diff (cases) |
|---|---|---|---|---|---|---|---|
| General | 500 | 0.818 (409) | 0.832 (416/500) | -7 | 0.852 (426) | 0.852 (426/500) | +0 |
| NBA | 393 | 0.639 (251) | 0.921 (362/393) | -111 | 0.621 (244) | 0.908 (357/393) | -113 |
| NFL | 212 | 0.557 (118) | 0.552 (117/212) | +1 | 0.561 (119) | 0.580 (123/212) | -4 |
| Politics | 104 | 0.769 (80) | 0.769 (80/104) | +0 | 0.731 (76) | 0.731 (76/104) | +0 |
| All cases | 1209 | 0.710 (858) | 0.806 (975/1209) | -117 | 0.715 (865) | 0.812 (982/1209) | -117 |

## Effectiveness of the AMP selection, single leave-one-out [LOO, classifier seed 3] - TH Table A.10 (and MLA Figs 2-3 for the global block)

P(success by AG) is the rule applied to every case of the block: the recomputed rule baseline beside the thesis's. PRE from exact counts; p = one-sample one-tailed z-test (plan §5.8 item 3), exact upper tail; the thesis prints p for the global block only.

### All cases (n = 1209; thesis n = 1209)
Thesis side: MLA Figs 2-3 (counts: plan T20/T21; no global LOO table in the thesis).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 360/1209 | 422/1209 | 0.298 | 0.349 | 0.719 (869) | 0.715 (864) | 242/360 | 351/422 | 0.672 | 0.832 | -0.160 | -0.166 | 0.410 | 0.975 | 5.0e-08 |
| MR | 297/1209 | 268/1209 | 0.246 | 0.222 | 0.693 (838) | 0.687 (831) | 231/297 | 222/268 | 0.778 | 0.828 | -0.051 | 0.276 | 0.451 | 7.8e-04 | 3.2e-07 |
| WC | 240/1209 | 185/1209 | 0.199 | 0.153 | 0.687 (831) | 0.687 (830) | 170/240 | 149/185 | 0.708 | 0.805 | -0.097 | 0.067 | 0.379 | 0.242 | 2.5e-04 |
| HAC | 253/1209 | 180/1209 | 0.209 | 0.149 | 0.634 (767) | 0.640 (774) | 187/253 | 127/180 | 0.739 | 0.706 | +0.034 | 0.286 | 0.182 | 2.7e-04 | 0.034 |
| DA | 59/1209 | 154/1209 | 0.049 | 0.127 | 0.304 (368) | 0.308 (372) | 28/59 | 126/154 | 0.475 | 0.818 | -0.344 | 0.245 | 0.737 | 0.002 | 0 |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.298 / 0.349 | 0.246 / 0.222 | 0.199 / 0.153 | 0.209 / 0.149 | 0.049 / 0.127 |
| P(success by AG) | 0.719 / 0.715 | 0.693 / 0.687 | 0.687 / 0.687 | 0.634 / 0.640 | 0.304 / 0.308 |
| P(success by AG given AMP selected AG) | 0.672 / 0.832 | 0.778 / 0.828 | 0.708 / 0.805 | 0.739 / 0.706 | 0.475 / 0.818 |
| PRE | -0.166 / 0.410 | 0.276 / 0.451 | 0.067 / 0.379 | 0.286 / 0.182 | 0.245 / 0.737 |
| p (one-tailed) | 0.975 / 5.0e-08 | 7.8e-04 / 3.2e-07 | 0.242 / 2.5e-04 | 2.7e-04 / 0.034 | 0.002 / 0 |

### General (n = 500; thesis n = 500)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 186/500 | 261/500 | 0.372 | 0.522 | 0.826 (413) | 0.826 (413) | 143/186 | 221/261 | 0.769 | 0.847 | -0.078 | -0.329 | 0.119 | 0.980 | – |
| MR | 125/500 | 83/500 | 0.250 | 0.166 | 0.770 (385) | 0.770 (385) | 109/125 | 72/83 | 0.872 | 0.867 | +0.005 | 0.443 | 0.424 | 0.003 | – |
| WC | 86/500 | 83/500 | 0.172 | 0.166 | 0.760 (380) | 0.760 (380) | 70/86 | 70/83 | 0.814 | 0.843 | -0.029 | 0.225 | 0.347 | 0.121 | – |
| HAC | 90/500 | 65/500 | 0.180 | 0.130 | 0.676 (338) | 0.676 (338) | 78/90 | 50/65 | 0.867 | 0.769 | +0.097 | 0.588 | 0.288 | 5.6e-05 | – |
| DA | 13/500 | 8/500 | 0.026 | 0.016 | 0.222 (111) | 0.216 (108) | 9/13 | 3/8 | 0.692 | 0.375 | +0.317 | 0.605 | 0.203 | 2.2e-05 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.372 / 0.522 | 0.250 / 0.166 | 0.172 / 0.166 | 0.180 / 0.130 | 0.026 / 0.016 |
| P(success by AG) | 0.826 / 0.826 | 0.770 / 0.770 | 0.760 / 0.760 | 0.676 / 0.676 | 0.222 / 0.216 |
| P(success by AG given AMP selected AG) | 0.769 / 0.847 | 0.872 / 0.867 | 0.814 / 0.843 | 0.867 / 0.769 | 0.692 / 0.375 |
| PRE | -0.329 / 0.119 | 0.443 / 0.424 | 0.225 / 0.347 | 0.588 / 0.288 | 0.605 / 0.203 |
| p (one-tailed) | 0.980 / – | 0.003 / – | 0.121 / – | 5.6e-05 / – | 2.2e-05 / – |

### NBA (n = 393; thesis n = 393)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 94/393 | 91/393 | 0.239 | 0.232 | 0.662 (260) | 0.656 (258) | 53/94 | 82/91 | 0.564 | 0.901 | -0.337 | -0.289 | 0.712 | 0.977 | – |
| MR | 108/393 | 107/393 | 0.275 | 0.272 | 0.651 (256) | 0.641 (252) | 79/108 | 100/107 | 0.731 | 0.935 | -0.203 | 0.230 | 0.818 | 0.040 | – |
| WC | 87/393 | 54/393 | 0.221 | 0.137 | 0.641 (252) | 0.644 (253) | 54/87 | 49/54 | 0.621 | 0.907 | -0.287 | -0.057 | 0.740 | 0.655 | – |
| HAC | 82/393 | 21/393 | 0.209 | 0.053 | 0.585 (230) | 0.603 (237) | 57/82 | 16/21 | 0.695 | 0.762 | -0.067 | 0.265 | 0.400 | 0.022 | – |
| DA | 22/393 | 120/393 | 0.056 | 0.305 | 0.349 (137) | 0.361 (142) | 8/22 | 115/120 | 0.364 | 0.958 | -0.595 | 0.023 | 0.935 | 0.441 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.239 / 0.232 | 0.275 / 0.272 | 0.221 / 0.137 | 0.209 / 0.053 | 0.056 / 0.305 |
| P(success by AG) | 0.662 / 0.656 | 0.651 / 0.641 | 0.641 / 0.644 | 0.585 / 0.603 | 0.349 / 0.361 |
| P(success by AG given AMP selected AG) | 0.564 / 0.901 | 0.731 / 0.935 | 0.621 / 0.907 | 0.695 / 0.762 | 0.364 / 0.958 |
| PRE | -0.289 / 0.712 | 0.230 / 0.818 | -0.057 / 0.740 | 0.265 / 0.400 | 0.023 / 0.935 |
| p (one-tailed) | 0.977 / – | 0.040 / – | 0.655 / – | 0.022 / – | 0.441 / – |

### NFL (n = 212; thesis n = 212)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 35/212 | 41/212 | 0.165 | 0.193 | 0.528 (112) | 0.514 (109) | 16/35 | 25/41 | 0.457 | 0.610 | -0.153 | -0.151 | 0.197 | 0.800 | – |
| MR | 42/212 | 45/212 | 0.198 | 0.212 | 0.538 (114) | 0.524 (111) | 24/42 | 22/45 | 0.571 | 0.489 | +0.083 | 0.073 | -0.073 | 0.331 | – |
| WC | 46/212 | 30/212 | 0.217 | 0.142 | 0.552 (117) | 0.538 (114) | 25/46 | 16/30 | 0.543 | 0.533 | +0.010 | -0.019 | -0.010 | 0.546 | – |
| HAC | 69/212 | 73/212 | 0.325 | 0.344 | 0.618 (131) | 0.618 (131) | 44/69 | 46/73 | 0.638 | 0.630 | +0.008 | 0.052 | 0.032 | 0.368 | – |
| DA | 20/212 | 23/212 | 0.094 | 0.108 | 0.467 (99) | 0.476 (101) | 9/20 | 8/23 | 0.450 | 0.348 | +0.102 | -0.032 | -0.246 | 0.560 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.165 / 0.193 | 0.198 / 0.212 | 0.217 / 0.142 | 0.325 / 0.344 | 0.094 / 0.108 |
| P(success by AG) | 0.528 / 0.514 | 0.538 / 0.524 | 0.552 / 0.538 | 0.618 / 0.618 | 0.467 / 0.476 |
| P(success by AG given AMP selected AG) | 0.457 / 0.610 | 0.571 / 0.489 | 0.543 / 0.533 | 0.638 / 0.630 | 0.450 / 0.348 |
| PRE | -0.151 / 0.197 | 0.073 / -0.073 | -0.019 / -0.010 | 0.052 / 0.032 | -0.032 / -0.246 |
| p (one-tailed) | 0.800 / – | 0.331 / – | 0.546 / – | 0.368 / – | 0.560 / – |

### Politics (n = 104; thesis n = 104)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 45/104 | 29/104 | 0.433 | 0.279 | 0.808 (84) | 0.808 (84) | 30/45 | 23/29 | 0.667 | 0.793 | -0.126 | -0.733 | -0.076 | 0.992 | – |
| MR | 22/104 | 33/104 | 0.212 | 0.317 | 0.798 (83) | 0.798 (83) | 19/22 | 28/33 | 0.864 | 0.848 | +0.015 | 0.325 | 0.250 | 0.222 | – |
| WC | 21/104 | 18/104 | 0.202 | 0.173 | 0.788 (82) | 0.798 (83) | 21/21 | 14/18 | 1.000 | 0.778 | +0.222 | 1.000 | -0.101 | 0.009 | – |
| HAC | 12/104 | 21/104 | 0.115 | 0.202 | 0.654 (68) | 0.654 (68) | 8/12 | 15/21 | 0.667 | 0.714 | -0.048 | 0.037 | 0.175 | 0.463 | – |
| DA | 4/104 | 3/104 | 0.038 | 0.029 | 0.202 (21) | 0.202 (21) | 2/4 | 0/3 | 0.500 | 0.000 | +0.500 | 0.373 | -0.253 | 0.069 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.433 / 0.279 | 0.212 / 0.317 | 0.202 / 0.173 | 0.115 / 0.202 | 0.038 / 0.029 |
| P(success by AG) | 0.808 / 0.808 | 0.798 / 0.798 | 0.788 / 0.798 | 0.654 / 0.654 | 0.202 / 0.202 |
| P(success by AG given AMP selected AG) | 0.667 / 0.793 | 0.864 / 0.848 | 1.000 / 0.778 | 0.667 / 0.714 | 0.500 / 0.000 |
| PRE | -0.733 / -0.076 | 0.325 / 0.250 | 1.000 / -0.101 | 0.037 / 0.175 | 0.373 / -0.253 |
| p (one-tailed) | 0.992 / – | 0.222 / – | 0.009 / – | 0.463 / – | 0.069 / – |

### Qualitative checks (plan §9.3 row A15)

| check | run | thesis |
|---|---|---|
| methods whose global p is not below 0.05 | SP, WC | none |
| domain with the highest mean PRE | General | NBA |
| mean PRE per domain | General 0.307, NBA 0.034, NFL -0.015, Politics 0.200 | General 0.276, NBA 0.721, NFL -0.020, Politics -0.001 |
| methods with a negative PRE, per domain | General: SP; NBA: SP, WC; NFL: SP, WC, DA; Politics: SP | General: none; NBA: none; NFL: MR, WC, DA; Politics: SP, WC, DA |

Rates within ±0.03 of the thesis: 41 of 75 cells (P(selected), P(success), P(success given selected); all blocks).

</details>

<details><summary>Single LOO, classifier seed 4 (seed noise)</summary>

## Success by domain, single leave-one-out [LOO, classifier seed 4]

Cells: rate (successes). Thesis side: TH Table A.9 p.115 (counts: plan T12).

| domain | n (run) | AMP: run | AMP: thesis | AMP: diff (cases) | DAP: run | DAP: thesis | DAP: diff (cases) |
|---|---|---|---|---|---|---|---|
| General | 500 | 0.818 (409) | 0.832 (416/500) | -7 | 0.842 (421) | 0.852 (426/500) | -5 |
| NBA | 393 | 0.634 (249) | 0.921 (362/393) | -113 | 0.613 (241) | 0.908 (357/393) | -116 |
| NFL | 212 | 0.552 (117) | 0.552 (117/212) | +0 | 0.575 (122) | 0.580 (123/212) | -1 |
| Politics | 104 | 0.721 (75) | 0.769 (80/104) | -5 | 0.731 (76) | 0.731 (76/104) | +0 |
| All cases | 1209 | 0.703 (850) | 0.806 (975/1209) | -125 | 0.711 (860) | 0.812 (982/1209) | -122 |

## Effectiveness of the AMP selection, single leave-one-out [LOO, classifier seed 4] - TH Table A.10 (and MLA Figs 2-3 for the global block)

P(success by AG) is the rule applied to every case of the block: the recomputed rule baseline beside the thesis's. PRE from exact counts; p = one-sample one-tailed z-test (plan §5.8 item 3), exact upper tail; the thesis prints p for the global block only.

### All cases (n = 1209; thesis n = 1209)
Thesis side: MLA Figs 2-3 (counts: plan T20/T21; no global LOO table in the thesis).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 372/1209 | 422/1209 | 0.308 | 0.349 | 0.719 (869) | 0.715 (864) | 265/372 | 351/422 | 0.712 | 0.832 | -0.119 | -0.023 | 0.410 | 0.608 | 5.0e-08 |
| MR | 302/1209 | 268/1209 | 0.250 | 0.222 | 0.693 (838) | 0.687 (831) | 223/302 | 222/268 | 0.738 | 0.828 | -0.090 | 0.148 | 0.451 | 0.044 | 3.2e-07 |
| WC | 224/1209 | 185/1209 | 0.185 | 0.153 | 0.687 (831) | 0.687 (830) | 153/224 | 149/185 | 0.683 | 0.805 | -0.122 | -0.014 | 0.379 | 0.555 | 2.5e-04 |
| HAC | 250/1209 | 180/1209 | 0.207 | 0.149 | 0.634 (767) | 0.640 (774) | 180/250 | 127/180 | 0.720 | 0.706 | +0.014 | 0.234 | 0.182 | 0.002 | 0.034 |
| DA | 61/1209 | 154/1209 | 0.050 | 0.127 | 0.304 (368) | 0.308 (372) | 29/61 | 126/154 | 0.475 | 0.818 | -0.343 | 0.246 | 0.737 | 0.002 | 0 |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.308 / 0.349 | 0.250 / 0.222 | 0.185 / 0.153 | 0.207 / 0.149 | 0.050 / 0.127 |
| P(success by AG) | 0.719 / 0.715 | 0.693 / 0.687 | 0.687 / 0.687 | 0.634 / 0.640 | 0.304 / 0.308 |
| P(success by AG given AMP selected AG) | 0.712 / 0.832 | 0.738 / 0.828 | 0.683 / 0.805 | 0.720 / 0.706 | 0.475 / 0.818 |
| PRE | -0.023 / 0.410 | 0.148 / 0.451 | -0.014 / 0.379 | 0.234 / 0.182 | 0.246 / 0.737 |
| p (one-tailed) | 0.608 / 5.0e-08 | 0.044 / 3.2e-07 | 0.555 / 2.5e-04 | 0.002 / 0.034 | 0.002 / 0 |

### General (n = 500; thesis n = 500)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 208/500 | 261/500 | 0.416 | 0.522 | 0.826 (413) | 0.826 (413) | 161/208 | 221/261 | 0.774 | 0.847 | -0.073 | -0.299 | 0.119 | 0.976 | – |
| MR | 125/500 | 83/500 | 0.250 | 0.166 | 0.770 (385) | 0.770 (385) | 116/125 | 72/83 | 0.928 | 0.867 | +0.061 | 0.687 | 0.424 | 1.3e-05 | – |
| WC | 83/500 | 83/500 | 0.166 | 0.166 | 0.760 (380) | 0.760 (380) | 68/83 | 70/83 | 0.819 | 0.843 | -0.024 | 0.247 | 0.347 | 0.103 | – |
| HAC | 69/500 | 65/500 | 0.138 | 0.130 | 0.676 (338) | 0.676 (338) | 55/69 | 50/65 | 0.797 | 0.769 | +0.028 | 0.374 | 0.288 | 0.016 | – |
| DA | 15/500 | 8/500 | 0.030 | 0.016 | 0.222 (111) | 0.216 (108) | 9/15 | 3/8 | 0.600 | 0.375 | +0.225 | 0.486 | 0.203 | 2.1e-04 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.416 / 0.522 | 0.250 / 0.166 | 0.166 / 0.166 | 0.138 / 0.130 | 0.030 / 0.016 |
| P(success by AG) | 0.826 / 0.826 | 0.770 / 0.770 | 0.760 / 0.760 | 0.676 / 0.676 | 0.222 / 0.216 |
| P(success by AG given AMP selected AG) | 0.774 / 0.847 | 0.928 / 0.867 | 0.819 / 0.843 | 0.797 / 0.769 | 0.600 / 0.375 |
| PRE | -0.299 / 0.119 | 0.687 / 0.424 | 0.247 / 0.347 | 0.374 / 0.288 | 0.486 / 0.203 |
| p (one-tailed) | 0.976 / – | 1.3e-05 / – | 0.103 / – | 0.016 / – | 2.1e-04 / – |

### NBA (n = 393; thesis n = 393)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 96/393 | 91/393 | 0.244 | 0.232 | 0.662 (260) | 0.656 (258) | 59/96 | 82/91 | 0.615 | 0.901 | -0.287 | -0.139 | 0.712 | 0.835 | – |
| MR | 95/393 | 107/393 | 0.242 | 0.272 | 0.651 (256) | 0.641 (252) | 58/95 | 100/107 | 0.611 | 0.935 | -0.324 | -0.117 | 0.818 | 0.798 | – |
| WC | 89/393 | 54/393 | 0.226 | 0.137 | 0.641 (252) | 0.644 (253) | 53/89 | 49/54 | 0.596 | 0.907 | -0.312 | -0.127 | 0.740 | 0.816 | – |
| HAC | 91/393 | 21/393 | 0.232 | 0.053 | 0.585 (230) | 0.603 (237) | 69/91 | 16/21 | 0.758 | 0.762 | -0.004 | 0.417 | 0.400 | 4.0e-04 | – |
| DA | 22/393 | 120/393 | 0.056 | 0.305 | 0.349 (137) | 0.361 (142) | 10/22 | 115/120 | 0.455 | 0.958 | -0.504 | 0.163 | 0.935 | 0.149 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.244 / 0.232 | 0.242 / 0.272 | 0.226 / 0.137 | 0.232 / 0.053 | 0.056 / 0.305 |
| P(success by AG) | 0.662 / 0.656 | 0.651 / 0.641 | 0.641 / 0.644 | 0.585 / 0.603 | 0.349 / 0.361 |
| P(success by AG given AMP selected AG) | 0.615 / 0.901 | 0.611 / 0.935 | 0.596 / 0.907 | 0.758 / 0.762 | 0.455 / 0.958 |
| PRE | -0.139 / 0.712 | -0.117 / 0.818 | -0.127 / 0.740 | 0.417 / 0.400 | 0.163 / 0.935 |
| p (one-tailed) | 0.835 / – | 0.798 / – | 0.816 / – | 4.0e-04 / – | 0.149 / – |

### NFL (n = 212; thesis n = 212)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 36/212 | 41/212 | 0.170 | 0.193 | 0.528 (112) | 0.514 (109) | 24/36 | 25/41 | 0.667 | 0.610 | +0.057 | 0.293 | 0.197 | 0.048 | – |
| MR | 54/212 | 45/212 | 0.255 | 0.212 | 0.538 (114) | 0.524 (111) | 24/54 | 22/45 | 0.444 | 0.489 | -0.044 | -0.202 | -0.073 | 0.915 | – |
| WC | 35/212 | 30/212 | 0.165 | 0.142 | 0.552 (117) | 0.538 (114) | 19/35 | 16/30 | 0.543 | 0.533 | +0.010 | -0.020 | -0.010 | 0.543 | – |
| HAC | 65/212 | 73/212 | 0.307 | 0.344 | 0.618 (131) | 0.618 (131) | 40/65 | 46/73 | 0.615 | 0.630 | -0.015 | -0.007 | 0.032 | 0.517 | – |
| DA | 22/212 | 23/212 | 0.104 | 0.108 | 0.467 (99) | 0.476 (101) | 10/22 | 8/23 | 0.455 | 0.348 | +0.107 | -0.023 | -0.246 | 0.547 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.170 / 0.193 | 0.255 / 0.212 | 0.165 / 0.142 | 0.307 / 0.344 | 0.104 / 0.108 |
| P(success by AG) | 0.528 / 0.514 | 0.538 / 0.524 | 0.552 / 0.538 | 0.618 / 0.618 | 0.467 / 0.476 |
| P(success by AG given AMP selected AG) | 0.667 / 0.610 | 0.444 / 0.489 | 0.543 / 0.533 | 0.615 / 0.630 | 0.455 / 0.348 |
| PRE | 0.293 / 0.197 | -0.202 / -0.073 | -0.020 / -0.010 | -0.007 / 0.032 | -0.023 / -0.246 |
| p (one-tailed) | 0.048 / – | 0.915 / – | 0.543 / – | 0.517 / – | 0.547 / – |

### Politics (n = 104; thesis n = 104)
Thesis side: TH Table A.10 p.115 (counts: plan T13).

| method | selected: run | thesis | P(sel): run | thesis | P(success by AG) = rule baseline: run | thesis | correct when selected: run | thesis | P(success given selected): run | thesis | diff | PRE: run | thesis | p: run | thesis (printed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | 32/104 | 29/104 | 0.308 | 0.279 | 0.808 (84) | 0.808 (84) | 21/32 | 23/29 | 0.656 | 0.793 | -0.137 | -0.788 | -0.076 | 0.985 | – |
| MR | 28/104 | 33/104 | 0.269 | 0.317 | 0.798 (83) | 0.798 (83) | 25/28 | 28/33 | 0.893 | 0.848 | +0.044 | 0.469 | 0.250 | 0.106 | – |
| WC | 17/104 | 18/104 | 0.163 | 0.173 | 0.788 (82) | 0.798 (83) | 13/17 | 14/18 | 0.765 | 0.778 | -0.013 | -0.112 | -0.101 | 0.595 | – |
| HAC | 25/104 | 21/104 | 0.240 | 0.202 | 0.654 (68) | 0.654 (68) | 16/25 | 15/21 | 0.640 | 0.714 | -0.074 | -0.040 | 0.175 | 0.558 | – |
| DA | 2/104 | 3/104 | 0.019 | 0.029 | 0.202 (21) | 0.202 (21) | 0/2 | 0/3 | 0.000 | 0.000 | +0.000 | -0.253 | -0.253 | 0.762 | – |

| run / thesis | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|
| P(AMP selected AG) | 0.308 / 0.279 | 0.269 / 0.317 | 0.163 / 0.173 | 0.240 / 0.202 | 0.019 / 0.029 |
| P(success by AG) | 0.808 / 0.808 | 0.798 / 0.798 | 0.788 / 0.798 | 0.654 / 0.654 | 0.202 / 0.202 |
| P(success by AG given AMP selected AG) | 0.656 / 0.793 | 0.893 / 0.848 | 0.765 / 0.778 | 0.640 / 0.714 | 0.000 / 0.000 |
| PRE | -0.788 / -0.076 | 0.469 / 0.250 | -0.112 / -0.101 | -0.040 / 0.175 | -0.253 / -0.253 |
| p (one-tailed) | 0.985 / – | 0.106 / – | 0.595 / – | 0.558 / – | 0.762 / – |

### Qualitative checks (plan §9.3 row A15)

| check | run | thesis |
|---|---|---|
| methods whose global p is not below 0.05 | SP, WC | none |
| domain with the highest mean PRE | General | NBA |
| mean PRE per domain | General 0.299, NBA 0.039, NFL 0.008, Politics -0.145 | General 0.276, NBA 0.721, NFL -0.020, Politics -0.001 |
| methods with a negative PRE, per domain | General: SP; NBA: SP, MR, WC; NFL: MR, WC, HAC, DA; Politics: SP, WC, HAC, DA | General: none; NBA: none; NFL: MR, WC, DA; Politics: SP, WC, DA |

Rates within ±0.03 of the thesis: 44 of 75 cells (P(selected), P(success), P(success given selected); all blocks).

</details>

## Single LOO: successes by classifier seed (AM18: seed noise)

| seed | AMP all | AMP General | AMP NBA | AMP NFL | AMP Politics | DAP all | DAP General | DAP NBA | DAP NFL | DAP Politics |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 855 | 414 | 245 | 117 | 79 | 859 | 425 | 238 | 122 | 74 |
| 1 | 860 | 407 | 254 | 121 | 78 | 857 | 420 | 240 | 122 | 75 |
| 2 | 863 | 412 | 250 | 126 | 75 | 868 | 425 | 242 | 123 | 78 |
| 3 | 858 | 409 | 251 | 118 | 80 | 865 | 426 | 244 | 119 | 76 |
| 4 | 850 | 409 | 249 | 117 | 75 | 860 | 421 | 241 | 122 | 76 |
