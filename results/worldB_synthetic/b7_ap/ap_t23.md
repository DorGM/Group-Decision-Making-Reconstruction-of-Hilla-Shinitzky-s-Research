**SYNTHETIC — not a reproduction**

### RP, AP and the rule baselines on the sub-sets (shape of TH Fig 5.3; T23, T24, T25) — SYNTHETIC — not a reproduction

RP = soft vote, prop + avgp (TH p.57); AP = soft vote, three-case rule (TH p.41). Sub-sets per partition: 77; partition: random, 30 seeds (0..29) (plan OQ25); SD = sample SD over the seeds (ddof 1).

| Method | run (synthetic): mean count ± SD | run (synthetic): min–max | run (synthetic): mean rate ± SD | thesis: count of 77 | thesis: rate |
|---|---|---|---|---|---|
| RP | 73.10 ± 1.94 | 69–77 | 0.949 ± 0.025 | 49 | 0.636 |
| AP | 72.00 ± 1.58 | 68–75 | 0.935 ± 0.020 | 42 | 0.545 |
| MR | 33.53 ± 1.93 | 30–38 | 0.435 ± 0.025 | 34 | 0.442 |
| WC | 34.30 ± 2.09 | 30–37 | 0.445 ± 0.027 | 34 | 0.442 |
| SP | 33.00 ± 2.53 | 27–37 | 0.429 ± 0.033 | 33 | 0.429 |
| HAC | 22.73 ± 2.86 | 18–30 | 0.295 ± 0.037 | 22 | 0.286 |

| PRE against MR (T25) | run (synthetic): from the mean rates | run (synthetic): mean ± SD over the partitions |
|---|---|---|
| RP | 0.910 | 0.910 ± 0.045 |
| AP | 0.885 | 0.885 ± 0.036 |

Thesis counts are the only integers of 77 that round to the printed rates (plan T23, T24). The synthetic PRE is a within-synthetic quantity (a property of the invented response model) and is not comparable with the PRE reported in the thesis; no thesis PRE is printed beside it.

The four rule rows agree with the thesis BY CONSTRUCTION (the generator was calibrated on them, plan §2.3). The RP and AP rows are NOT a reproduction of T23: they show that the pipeline runs end to end, and nothing about the thesis's result.

#### AP run diagnostics — SYNTHETIC — not a reproduction

- The three-case rule (TH p.41) and argmax P(True) return the same answer in every sub-set of every partition (plan §4.4; asserted): 2310 sub-set evaluations, 0 differences.
- Which case applied (mean sub-sets per partition): exactly one answer "True" 61.03; several "True" 0.43; none "True" 15.53; single voted answer 0.00.
- Ties of the deciding probability (canonical order decides; decision B7-2): 0.00 sub-sets per partition.
- Instances per partition: 441–459 (share of correct answers 0.170); model fits: 2310; logistic regressions stopped at max_iter: 0.
- Sub-sets with a single voted answer (empty out-group, decision B1-2 / B6-4): 0 over all partitions; sub-sets whose correct answer has no vote (no correct instance, B2-3): 0.40 per partition.
- Library warnings during the fits: none.

**SYNTHETIC — not a reproduction**
