**SYNTHETIC — not a reproduction**

### RP and AP against the rule baselines: p-values (shape of TH Table 5.11, T29) — SYNTHETIC — not a reproduction

McNemar = two-sided chi-square with continuity correction; proportion = pooled two-sample z-test, one-tailed, H1: approach > rule (plan §5.8). Sub-sets per partition: 77; partition: random, 30 seeds (0..29) (plan OQ25); SD = sample SD over the seeds (ddof 1).

| Rule | run (synthetic): vs RP, McNemar | run (synthetic): vs RP, proportion | run (synthetic): vs AP, McNemar | run (synthetic): vs AP, proportion | thesis: vs RP, McNemar | thesis: vs RP, proportion | thesis: vs AP, McNemar | thesis: vs AP, proportion |
|---|---|---|---|---|---|---|---|---|
| SP | median 1.1E-09 (1.2E-11 – 8.6E-08); p < 0.05 in 30 of 30 | median 1.8E-12 (3.4E-17 – 2.4E-09); p < 0.05 in 30 of 30 | median 1.9E-09 (5.4E-11 – 4.0E-08); p < 0.05 in 30 of 30 | median 5.4E-12 (1.2E-15 – 2.4E-09); p < 0.05 in 30 of 30 | 0.001 | 0.005 | 0.027 | 0.073 |
| MR | median 8.8E-10 (3.2E-11 – 6.3E-08); p < 0.05 in 30 of 30 | median 3.6E-12 (2.8E-16 – 5.7E-09); p < 0.05 in 30 of 30 | median 1.2E-09 (2.5E-10 – 2.5E-08); p < 0.05 in 30 of 30 | median 7.5E-12 (2.5E-13 – 3.4E-09); p < 0.05 in 30 of 30 | 3.0E-04 | 0.008 | 0.027 | 0.099 |
| WC | median 1.9E-09 (3.2E-11 – 1.4E-08); p < 0.05 in 30 of 30 | median 1.1E-11 (6.1E-16 – 4.5E-10); p < 0.05 in 30 of 30 | median 1.9E-09 (1.5E-10 – 2.5E-08); p < 0.05 in 30 of 30 | median 1.5E-11 (9.0E-14 – 8.0E-09); p < 0.05 in 30 of 30 | 3.0E-04 | 0.008 | 0.027 | 0.099 |
| HAC | median 2.5E-12 (3.3E-13 – 2.3E-10); p < 0.05 in 30 of 30 | median 1.4E-17 (2.2E-20 – 3.3E-14); p < 0.05 in 30 of 30 | median 6.8E-12 (9.1E-13 – 5.4E-09); p < 0.05 in 30 of 30 | median 1.1E-16 (3.6E-19 – 2.7E-12); p < 0.05 in 30 of 30 | 4.9E-05 | 6.4E-06 | 0.003 | 0.001 |

Success counts behind the tests (mean over the partitions; partitions in which the approach has more successes than the rule):

| Rule | run (synthetic): RP mean count / rule mean count / approach ahead | run (synthetic): AP mean count / rule mean count / approach ahead |
|---|---|---|
| SP | 73.10 / 33.00 / 30 of 30 | 72.00 / 33.00 / 30 of 30 |
| MR | 73.10 / 33.53 / 30 of 30 | 72.00 / 33.53 / 30 of 30 |
| WC | 73.10 / 34.30 / 30 of 30 | 72.00 / 34.30 / 30 of 30 |
| HAC | 73.10 / 22.73 / 30 of 30 | 72.00 / 22.73 / 30 of 30 |

#### Check of the test conventions on the thesis's own counts (no data involved)

Recomputed with cdm.eval.stats from the published counts (RP 49, AP 42, SP 33, MR 34, WC 34, HAC 22 of 77) and the discordant pairs of plan §5.8 [INFERRED]; beside the printed values of TH Table 5.11.

| Rule | vs RP, McNemar: recomputed (pair) / printed | vs RP, proportion: recomputed / printed | vs AP, McNemar: recomputed (pair) / printed | vs AP, proportion: recomputed / printed |
|---|---|---|---|---|
| SP | 8.0E-04 (18, 2); 0.001 (19, 3) / 0.001 | 0.005 / 0.005 | 0.027 (11, 2) / 0.027 | 0.073 / 0.073 |
| MR | 3.0E-04 (15, 0) / 3.0E-04 | 0.008 / 0.008 | 0.027 (9, 1) / 0.027 | 0.099 / 0.099 |
| WC | 3.0E-04 (15, 0) / 3.0E-04 | 0.008 / 0.008 | 0.027 (9, 1) / 0.027 | 0.099 / 0.099 |
| HAC | 4.9E-05 (34, 7) / 4.9E-05 | 6.4E-06 / 6.4E-06 | 0.003 (30, 10); 0.003 (31, 11) / 0.003 | 5.4E-04 / 0.001 |

A p-value is "undefined" when the two methods never disagree (McNemar, b + c = 0) or the pooled standard error is 0 (decision B9-1). The p-values of different partitions are not independent and are not pooled (decision B9-2).

The run columns are NOT a reproduction of Table 5.11: on the synthetic stand-in they only show that the tests run on the pipeline's output. The convention check above is independent of any data.

**SYNTHETIC — not a reproduction**
