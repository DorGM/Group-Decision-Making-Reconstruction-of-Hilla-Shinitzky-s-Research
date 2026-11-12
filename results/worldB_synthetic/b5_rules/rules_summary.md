**SYNTHETIC — not a reproduction**

### World-B rule baselines on the sub-sets (T24) — SYNTHETIC — not a reproduction

Sub-sets per partition: 77. Partition: random, 30 seeds (0..29). Tie policy: canonical_first (tolerance 1e-09).

| Rule | run (synthetic): mean count ± SD | run (synthetic): min–max | run (synthetic): mean rate ± SD | run (synthetic): tied sub-sets (mean) | run (synthetic): mean count if every tie went wrong – right | thesis: count of 77 | thesis: rate (TH Fig 5.3) |
|---|---|---|---|---|---|---|---|
| MR | 33.53 ± 1.93 | 30–38 | 0.435 ± 0.025 | 5.87 | 31.37 – 35.17 | 34 | 0.442 |
| WC | 34.30 ± 2.09 | 30–37 | 0.445 ± 0.027 | 0.00 | 34.30 – 34.30 | 34 | 0.442 |
| SP | 33.00 ± 2.53 | 27–37 | 0.429 ± 0.033 | 0.00 | 33.00 – 33.00 | 33 | 0.429 |
| HAC | 22.73 ± 2.86 | 18–30 | 0.295 ± 0.037 | 0.00 | 22.73 – 22.73 | 22 | 0.286 |

SD = sample standard deviation over the partition seeds (ddof 1). Thesis counts are the only integers of 77 that round to the printed rates (plan §2.3).

The synthetic generator was calibrated on exactly these four rule counts over exactly these partition seeds (plan §2.3). Agreement between the two column groups is therefore BY CONSTRUCTION; it is not a reproduction of T24 and says nothing about the thesis data.

Note on the generator seed: the calibration was done on the single data set drawn with generator seed 0. With the same parameters and other generator seeds the counts move well outside ±2 (generator seeds 1–8, 10 partition seeds each: MR 25–35, WC 26–34, SP 31–37, HAC 16–21; 2 of the 8 seeds are within ±2 on all four rules). The agreement holds for generator seed 0 only: it is a property of this one data set, not of the model.

#### Success counts per partition seed — SYNTHETIC — not a reproduction

| seed | MR | WC | SP | HAC |
|---|---|---|---|---|
| 0 | 33 | 37 | 34 | 18 |
| 1 | 33 | 34 | 36 | 19 |
| 2 | 32 | 33 | 31 | 22 |
| 3 | 36 | 36 | 37 | 25 |
| 4 | 30 | 33 | 31 | 24 |
| 5 | 32 | 35 | 35 | 23 |
| 6 | 33 | 34 | 35 | 25 |
| 7 | 33 | 36 | 35 | 22 |
| 8 | 34 | 34 | 34 | 21 |
| 9 | 32 | 32 | 34 | 28 |
| 10 | 36 | 37 | 29 | 30 |
| 11 | 38 | 37 | 34 | 20 |
| 12 | 33 | 32 | 36 | 22 |
| 13 | 35 | 35 | 32 | 22 |
| 14 | 35 | 36 | 35 | 22 |
| 15 | 31 | 33 | 27 | 21 |
| 16 | 33 | 30 | 35 | 22 |
| 17 | 33 | 34 | 32 | 23 |
| 18 | 34 | 36 | 36 | 22 |
| 19 | 32 | 30 | 32 | 21 |
| 20 | 36 | 33 | 35 | 19 |
| 21 | 33 | 34 | 32 | 27 |
| 22 | 36 | 37 | 30 | 21 |
| 23 | 35 | 36 | 35 | 21 |
| 24 | 34 | 35 | 33 | 23 |
| 25 | 30 | 30 | 29 | 22 |
| 26 | 34 | 35 | 34 | 20 |
| 27 | 36 | 37 | 31 | 29 |
| 28 | 33 | 34 | 32 | 24 |
| 29 | 31 | 34 | 29 | 24 |

thesis (count of 77; TH Fig 5.3, not part of the run): MR 34 / WC 34 / SP 33 / HAC 22

**SYNTHETIC — not a reproduction**
