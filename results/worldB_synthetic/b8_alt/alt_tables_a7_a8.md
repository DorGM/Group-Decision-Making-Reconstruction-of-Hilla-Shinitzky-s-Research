**SYNTHETIC — not a reproduction**

### RP: classifiers and classification-aggregation strategies (shape of TH Table A.7, T27) — SYNTHETIC — not a reproduction

Bag KNN = BaggingClassifier(KNeighborsClassifier()), SVM = SVC(probability=True), library defaults (plan OQ35). Sub-sets per partition: 77; partition: random, 30 seeds (0..29) (plan OQ25); SD = sample SD over the seeds (ddof 1).

| Classifier | Strategy | Tie breaker | run (synthetic): success rate, mean ± SD | run (synthetic): count of 77, min–max | run (synthetic): tie breakers, mean % (min–max) | run (synthetic): ties left after the tie breaker (mean sub-sets) | run (synthetic): mean count if every remaining tie went wrong – right | thesis: success rate | thesis: tie breakers |
|---|---|---|---|---|---|---|---|---|---|
| Vote soft | maxp | – | 94.5 ± 1.5% | 71–75 | 0.0% (0–0) | 0.00 | 72.77 – 72.77 | 58% | 0% |
| Vote soft | avgp | – | 95.2 ± 2.1% | 70–77 | 0.0% (0–0) | 0.00 | 73.30 – 73.30 | 61% | 0% |
| Vote soft | wm | – | 91.5 ± 2.0% | 66–73 | 0.0% (0–0) | 0.00 | 70.43 – 70.43 | 47% | 0% |
| Vote soft | maj | avgp | 95.9 ± 2.0% | 70–77 | 7.8% (4–13) | 0.00 | 73.83 – 73.83 | 62% | 16% |
| Vote soft | maj | maxp | 94.8 ± 1.6% | 71–75 | 7.8% (4–13) | 0.00 | 72.97 – 72.97 | 60% | 17% |
| Vote soft | maj | wm | 92.9 ± 2.0% | 68–74 | 7.8% (4–13) | 0.00 | 71.53 – 71.53 | 57% | 16% |
| Vote soft | prop | avgp | 94.9 ± 2.5% | 69–77 | 7.8% (3–13) | 0.00 | 73.10 – 73.10 | 64% | 18% |
| Vote soft | prop | maxp | 93.9 ± 2.2% | 68–76 | 7.8% (3–13) | 0.00 | 72.33 – 72.33 | 60% | 16% |
| Vote soft | prop | wm | 92.3 ± 2.6% | 67–75 | 7.8% (3–13) | 0.00 | 71.07 – 71.07 | 57% | 17% |
| Bag KNN | maxp | – | 92.1 ± 2.6% | 65–74 | 1.8% (0–4) | 1.37 | 70.43 – 71.63 | 51% | 0% |
| Bag KNN | avgp | – | 94.2 ± 2.4% | 69–76 | 0.1% (0–1) | 0.10 | 72.57 – 72.63 | 56% | 0% |
| Bag KNN | wm | – | 91.4 ± 2.4% | 67–74 | 0.3% (0–1) | 0.27 | 70.27 – 70.47 | 48% | 0% |
| Bag KNN | maj | avgp | 93.7 ± 2.3% | 67–75 | 7.1% (4–12) | 0.07 | 72.10 – 72.13 | 49% | 8% |
| Bag KNN | maj | maxp | 92.9 ± 2.6% | 65–75 | 7.1% (4–12) | 0.40 | 71.27 – 71.53 | 49% | 16% |
| Bag KNN | maj | wm | 92.2 ± 2.4% | 67–74 | 7.1% (4–12) | 0.20 | 70.83 – 70.97 | 48% | 12% |
| Bag KNN | prop | avgp | 92.9 ± 2.8% | 66–75 | 5.0% (1–12) | 0.07 | 71.50 – 71.53 | 53% | 10% |
| Bag KNN | prop | maxp | 92.5 ± 3.0% | 64–75 | 5.0% (1–12) | 0.27 | 71.10 – 71.27 | 53% | 6% |
| Bag KNN | prop | wm | 92.4 ± 2.9% | 66–75 | 5.0% (1–12) | 0.17 | 71.07 – 71.17 | 52% | 5% |
| SVM | maxp | – | 93.8 ± 2.1% | 68–75 | 0.0% (0–0) | 0.00 | 72.23 – 72.23 | 43% | 0% |
| SVM | avgp | – | 93.9 ± 2.3% | 67–75 | 0.0% (0–0) | 0.00 | 72.33 – 72.33 | 43% | 0% |
| SVM | wm | – | 93.1 ± 2.4% | 68–74 | 0.0% (0–0) | 0.00 | 71.67 – 71.67 | 45% | 0% |
| SVM | maj | avgp | 95.1 ± 1.9% | 71–76 | 8.3% (3–14) | 0.00 | 73.23 – 73.23 | 43% | 91% |
| SVM | maj | maxp | 94.5 ± 2.2% | 68–76 | 8.3% (3–14) | 0.00 | 72.77 – 72.77 | 43% | 91% |
| SVM | maj | wm | 93.9 ± 2.4% | 69–75 | 8.3% (3–14) | 0.00 | 72.30 – 72.30 | 45% | 91% |
| SVM | prop | avgp | 93.8 ± 2.3% | 68–75 | 10.0% (4–16) | 0.00 | 72.20 – 72.20 | 43% | 91% |
| SVM | prop | maxp | 93.1 ± 2.5% | 68–74 | 10.0% (4–16) | 0.00 | 71.70 – 71.70 | 43% | 91% |
| SVM | prop | wm | 92.6 ± 2.8% | 67–74 | 10.0% (4–16) | 0.00 | 71.33 – 71.33 | 45% | 91% |

- "Tie breakers" = share of the sub-sets in which the primary criterion is tied between two or more answers (plan OQ27), including the case that no supporter of any answer is classified "True" (maj and prop are then 0 for all).
- A tie left after the tie breaker goes to the first answer in the order of A_P (plan OQ27). maxp / avgp / wm alone have no tie breaker in Table 4.2 ("–"); Table A.7 prints "rnd" for them: a tie is resolved by a seeded random choice.
- With this definition the tie percentage cannot depend on the tie breaker, so the three maj rows (and the three prop rows) are equal in the run by construction. The thesis prints 16 / 17 / 16 and 18 / 16 / 17; the reason is not stated (decision B6-7).

Printed for the table's shape only; no difference is computed and the synthetic values carry no information about the thesis's Table A.7.

**SYNTHETIC — not a reproduction**

**SYNTHETIC — not a reproduction**

### AP: classifiers (shape of TH Table A.8, T28) — SYNTHETIC — not a reproduction

Sub-sets per partition: 77; partition: random, 30 seeds (0..29) (plan OQ25); SD = sample SD over the seeds (ddof 1).

| Classifier | run (synthetic): success rate, mean ± SD | run (synthetic): count of 77, min–max | run (synthetic): tied sub-sets (mean) | run (synthetic): sub-sets where the three-case rule and argmax P(True) differ (total over the partitions) | thesis: success rate |
|---|---|---|---|---|---|
| Bag KNN | 92.2 ± 2.8% | 67–75 | 1.97 | 0 | 44% |
| SVM | 94.4 ± 2.3% | 68–76 | 0.00 | 0 | 43% |
| Vote soft | 93.5 ± 2.0% | 68–75 | 0.00 | 0 | 55% |

AP answer = the thesis's three-case rule (TH p.41). For Vote soft and Bag KNN it must equal argmax P(True) (plan §4.4; asserted in the run). For SVC the classification and the probability can disagree, so the two are compared and counted, not asserted (decision B8-1).

Printed for the table's shape only; no difference is computed and the synthetic values carry no information about the thesis's Table A.8.

#### Classifier diagnostics — SYNTHETIC — not a reproduction

- RP/soft_vote: share classified "True" 0.248 (share of correct instances 0.280); instances classified "False" although P(True) > 0.5: 0, classified "True" although P(True) <= 0.5: 0 (over all partitions); library warnings: none.
- RP/bag_knn: share classified "True" 0.251 (share of correct instances 0.280); instances classified "False" although P(True) > 0.5: 3, classified "True" although P(True) <= 0.5: 4 (over all partitions); library warnings: none.
- RP/svm: share classified "True" 0.249 (share of correct instances 0.280); instances classified "False" although P(True) > 0.5: 1399, classified "True" although P(True) <= 0.5: 0 (over all partitions); library warnings: {'FutureWarning: The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC()': 2310}.
- AP/soft_vote: share classified "True" 0.137 (share of correct instances 0.170); instances classified "False" although P(True) > 0.5: 0, classified "True" although P(True) <= 0.5: 0 (over all partitions); library warnings: none.
- AP/bag_knn: share classified "True" 0.136 (share of correct instances 0.170); instances classified "False" although P(True) > 0.5: 0, classified "True" although P(True) <= 0.5: 2 (over all partitions); library warnings: none.
- AP/svm: share classified "True" 0.127 (share of correct instances 0.170); instances classified "False" although P(True) > 0.5: 370, classified "True" although P(True) <= 0.5: 0 (over all partitions); library warnings: {'FutureWarning: The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC()': 2310}.

**SYNTHETIC — not a reproduction**
