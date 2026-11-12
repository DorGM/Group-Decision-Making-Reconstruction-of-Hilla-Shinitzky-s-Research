**SYNTHETIC — not a reproduction**

### Response-Prediction (RP): classification-aggregation strategies (shape of TH Table 4.2, T26) — SYNTHETIC — not a reproduction

Classifier: Vote soft (soft voting over LDA, RF, XGBoost, LR, KNN; plan OQ26). Leave-one-group-out, group = sub-set (TH p.57). Sub-sets per partition: 77; partition: random, 30 seeds (0..29) (plan OQ25); SD = sample SD over the seeds (ddof 1).

| Strategy | Tie breaker | run (synthetic): success rate, mean ± SD | run (synthetic): count of 77, min–max | run (synthetic): tie breakers, mean % (min–max) | run (synthetic): ties left after the tie breaker (mean sub-sets) | run (synthetic): mean count if every remaining tie went wrong – right | thesis: success rate | thesis: tie breakers |
|---|---|---|---|---|---|---|---|---|
| maxp | – | 94.5 ± 1.5% | 71–75 | 0.0% (0–0) | 0.00 | 72.77 – 72.77 | 58% | 0% |
| avgp | – | 95.2 ± 2.1% | 70–77 | 0.0% (0–0) | 0.00 | 73.30 – 73.30 | 61% | 0% |
| wm | – | 91.5 ± 2.0% | 66–73 | 0.0% (0–0) | 0.00 | 70.43 – 70.43 | 47% | 0% |
| maj | avgp | 95.9 ± 2.0% | 70–77 | 7.8% (4–13) | 0.00 | 73.83 – 73.83 | 62% | 16% |
| maj | maxp | 94.8 ± 1.6% | 71–75 | 7.8% (4–13) | 0.00 | 72.97 – 72.97 | 60% | 17% |
| maj | wm | 92.9 ± 2.0% | 68–74 | 7.8% (4–13) | 0.00 | 71.53 – 71.53 | 57% | 16% |
| prop | avgp | 94.9 ± 2.5% | 69–77 | 7.8% (3–13) | 0.00 | 73.10 – 73.10 | 64% | 18% |
| prop | maxp | 93.9 ± 2.2% | 68–76 | 7.8% (3–13) | 0.00 | 72.33 – 72.33 | 60% | 16% |
| prop | wm | 92.3 ± 2.6% | 67–75 | 7.8% (3–13) | 0.00 | 71.07 – 71.07 | 57% | 17% |

- "Tie breakers" = share of the sub-sets in which the primary criterion is tied between two or more answers (plan OQ27), including the case that no supporter of any answer is classified "True" (maj and prop are then 0 for all).
- A tie left after the tie breaker goes to the first answer in the order of A_P (plan OQ27). maxp / avgp / wm alone have no tie breaker in Table 4.2 ("–"); Table A.7 prints "rnd" for them: a tie is resolved by a seeded random choice.
- With this definition the tie percentage cannot depend on the tie breaker, so the three maj rows (and the three prop rows) are equal in the run by construction. The thesis prints 16 / 17 / 16 and 18 / 16 / 17; the reason is not stated (decision B6-7).

As in the thesis (TH p.58), prop + avgp is the RP strategy used for the headline result; all nine strategies are reported.

The thesis columns are printed for the table's shape only. The synthetic stand-in was calibrated on the four rule baselines and on nothing else (plan §2.3); its RP numbers carry no information about the thesis's Table 4.2. No difference is computed.

#### Run diagnostics — SYNTHETIC — not a reproduction

- Instances per partition: 2490–2490 (share of correct responses 0.280); model fits: 2310; logistic regressions stopped at max_iter: 0.
- Out-of-group accuracy of the response classifier: 0.959; share classified "True": 0.248.
- Sub-sets in which every response gives the same answer (empty out-group, decision B1-2 / B6-4): 0 over all partitions (0 instances without out-group features).
- Sub-sets whose correct answer has no vote (RP cannot succeed there): 0.40 per partition; sub-sets with a tied majority (OQ28): 5.87 per partition.
- Library warnings during the fits: none.

**SYNTHETIC — not a reproduction**
