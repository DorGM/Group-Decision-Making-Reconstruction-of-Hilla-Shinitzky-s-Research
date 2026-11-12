**SYNTHETIC — not a reproduction**

# World B (RP / AP) — results on the SYNTHETIC stand-in — SYNTHETIC — not a reproduction

**What this report is.** The MTurk data of the thesis (22 multiple-choice tasks, 77 sub-sets) is not public (available on request according to the 2024 article) and has not been supplied. Everything below was computed on a synthetic stand-in: invented responses that copy only the published task sizes and solver shares (TH Table 4.1) and were calibrated on the four rule baselines of TH Fig 5.3.

**What it shows.** The RP / AP code (steps B6–B10) runs end to end: instances, classifiers, leave-one-group-out, aggregation strategies, significance tests, SHAP, tables in the shape of the thesis's tables.

**What it does not show.** The numbers in the "run (synthetic)" columns say carry NO information about the thesis's RP / AP results. Whether a run value lies near a thesis value or far from it carries no information: no synthetic parameter was tuned towards an RP / AP result (plan §2.3), and none should be. No difference between a run value and a thesis value is computed anywhere in this report.

**Why the synthetic RP / AP rates must not be read against the thesis.** On the shipped stand-in (generator seed 0, `config/synthetic_worldB.yaml`) solvers are easy to identify: the single feature dPSv separates correct from wrong responses with an AUC of about 0.83 on partition seed 0 (computed once, not by these drivers; ChiSq alone 0.75, ChosenMajAns alone 0.69). This is a side effect of the invented prediction mechanism (a solver moves half of the predicted mass to the lure, the value at which the SP baseline reached its target of 33), not of any tuning towards RP / AP. The synthetic RP / AP success rates therefore reflect that invented mechanism and are NOT comparable with the thesis's 0.636 / 0.545. In addition, the calibrated rule baselines hold for generator seed 0 only (decision B3-10).

**Status of the real-data path.** Every driver takes the real file through `config/worldB.yaml` (`source: real`) with the same commands (section 5). That path is UNTESTED ON REAL DATA: the file's format is an assumption (decision L1–L12).

## 1. Thesis-reported items T23–T30 side by side — SYNTHETIC — not a reproduction

Data: 22 tasks, 2490 responses (generator seed 0). Sub-sets per partition: 77; partition: random, 30 seeds (0..29) (plan OQ25); SD = sample SD over the seeds (ddof 1). Run values are mean ± SD over the partitions with the range of the count of 77 in brackets.

| Item | Quantity | thesis | run (synthetic) | Is the comparison meaningful? | Why |
|---|---|---|---|---|---|
| T23 | RP success rate (soft vote, prop + avgp), leave-one-group-out over the 77 sub-sets | 0.636 (49/77) | 0.949 ± 0.025 (count 73.10, range 69–77) | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T23 | AP success rate (soft vote, three-case rule) | 0.545 (42/77) | 0.935 ± 0.020 (count 72.00, range 68–75) | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T24 | MR success rate on the sub-sets | 0.442 (34/77) | 0.435 ± 0.025 (count 33.53, range 30–38) | NO — agreement is BY CONSTRUCTION | the generator was calibrated on exactly these four counts over these seeds (plan §2.3), for generator seed 0 only |
| T24 | WC success rate on the sub-sets | 0.442 (34/77) | 0.445 ± 0.027 (count 34.30, range 30–37) | NO — agreement is BY CONSTRUCTION | the generator was calibrated on exactly these four counts over these seeds (plan §2.3), for generator seed 0 only |
| T24 | SP success rate on the sub-sets | 0.429 (33/77) | 0.429 ± 0.033 (count 33.00, range 27–37) | NO — agreement is BY CONSTRUCTION | the generator was calibrated on exactly these four counts over these seeds (plan §2.3), for generator seed 0 only |
| T24 | HAC success rate on the sub-sets | 0.286 (22/77) | 0.295 ± 0.037 (count 22.73, range 18–30) | NO — agreement is BY CONSTRUCTION | the generator was calibrated on exactly these four counts over these seeds (plan §2.3), for generator seed 0 only |
| T25 | PRE of RP against MR | 0.348 | 0.910 (from the mean rates) | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T25 | PRE of AP against MR | 0.185 | 0.885 (from the mean rates) | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T26 | RP strategy maxp (Table 4.2) | 58% success; 0% tie breakers | 94.5 ± 1.5% success; 0.0% tie breakers | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T26 | RP strategy avgp (Table 4.2) | 61% success; 0% tie breakers | 95.2 ± 2.1% success; 0.0% tie breakers | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T26 | RP strategy wm (Table 4.2) | 47% success; 0% tie breakers | 91.5 ± 2.0% success; 0.0% tie breakers | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T26 | RP strategy maj+avgp (Table 4.2) | 62% success; 16% tie breakers | 95.9 ± 2.0% success; 7.8% tie breakers | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T26 | RP strategy maj+maxp (Table 4.2) | 60% success; 17% tie breakers | 94.8 ± 1.6% success; 7.8% tie breakers | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T26 | RP strategy maj+wm (Table 4.2) | 57% success; 16% tie breakers | 92.9 ± 2.0% success; 7.8% tie breakers | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T26 | RP strategy prop+avgp (Table 4.2) | 64% success; 18% tie breakers | 94.9 ± 2.5% success; 7.8% tie breakers | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T26 | RP strategy prop+maxp (Table 4.2) | 60% success; 16% tie breakers | 93.9 ± 2.2% success; 7.8% tie breakers | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T26 | RP strategy prop+wm (Table 4.2) | 57% success; 17% tie breakers | 92.3 ± 2.6% success; 7.8% tie breakers | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T27 | RP with Bag KNN, 9 strategies (Table A.7) | 48–56% success; 5–16% tie breakers (maj / prop rows) | 91.4–94.2% success over the 9 strategies; 5.0–7.1% tie breakers (maj / prop rows) | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T27 | RP with SVM, 9 strategies (Table A.7) | 43–45% success; 91–91% tie breakers (maj / prop rows) | 92.6–95.1% success over the 9 strategies; 8.3–10.0% tie breakers (maj / prop rows) | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T28 | AP with Bag KNN (Table A.8) | 44% | 92.2 ± 2.8% | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T28 | AP with SVM (Table A.8) | 43% | 94.4 ± 2.3% | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T28 | AP with Vote soft (Table A.8) | 55% | 93.5 ± 2.0% | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T29 | RP vs SP: McNemar / one-tailed proportion p-value (Table 5.11) | 0.001 / 0.005 | McNemar median 1.1E-09, p < 0.05 in 30 of 30; proportion median 1.8E-12, p < 0.05 in 30 of 30 | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T29 | RP vs MR: McNemar / one-tailed proportion p-value (Table 5.11) | 3.0E-04 / 0.008 | McNemar median 8.8E-10, p < 0.05 in 30 of 30; proportion median 3.6E-12, p < 0.05 in 30 of 30 | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T29 | RP vs WC: McNemar / one-tailed proportion p-value (Table 5.11) | 3.0E-04 / 0.008 | McNemar median 1.9E-09, p < 0.05 in 30 of 30; proportion median 1.1E-11, p < 0.05 in 30 of 30 | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T29 | RP vs HAC: McNemar / one-tailed proportion p-value (Table 5.11) | 4.9E-05 / 6.4E-06 | McNemar median 2.5E-12, p < 0.05 in 30 of 30; proportion median 1.4E-17, p < 0.05 in 30 of 30 | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T29 | AP vs SP: McNemar / one-tailed proportion p-value (Table 5.11) | 0.027 / 0.073 | McNemar median 1.9E-09, p < 0.05 in 30 of 30; proportion median 5.4E-12, p < 0.05 in 30 of 30 | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T29 | AP vs MR: McNemar / one-tailed proportion p-value (Table 5.11) | 0.027 / 0.099 | McNemar median 1.2E-09, p < 0.05 in 30 of 30; proportion median 7.5E-12, p < 0.05 in 30 of 30 | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T29 | AP vs WC: McNemar / one-tailed proportion p-value (Table 5.11) | 0.027 / 0.099 | McNemar median 1.9E-09, p < 0.05 in 30 of 30; proportion median 1.5E-11, p < 0.05 in 30 of 30 | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T29 | AP vs HAC: McNemar / one-tailed proportion p-value (Table 5.11) | 0.003 / 0.001 | McNemar median 6.8E-12, p < 0.05 in 30 of 30; proportion median 1.1E-16, p < 0.05 in 30 of 30 | NO | RP / AP outcomes are not calibration targets; the response model is invented (plan §2.3) |
| T29 | Test conventions: Table 5.11 recomputed from the published counts (49, 42 vs 33, 34, 34, 22 of 77; pairs of plan §5.8) | 16 printed p-values | all 16 printed values are matched after rounding to three decimals: the 8 proportion-test values follow from the published counts alone; the 8 McNemar values are matched by discordant pairs that the plan inferred because they fit (they are not published). AP vs HAC, proportion: 0.00054 against a printed 0.001 | YES — no data involved | identifies the test variants from the thesis's own numbers; it is not a recomputation on data |
| T30 | SHAP rank order of the RCR features, top 5 (TH Fig 5.8 p.82 (MCF Fig 2)) | dPSv, PSvD_OG, PSvD, ConfD_IG, PSv | dPSv, PSvD_IG, PSvD_OG, PChiSq_OG, ChiSq (XGBoost, TreeExplainer) | NO — smoke run only | plan row B10: acceptance on real data only; which invented features carry information is a property of the generator. That dPSv is first in both columns is by design of the generator (its prediction model is the SP mechanism) and is not evidence |
| T30 | SHAP rank order of the ACR features, top 5 (TH Fig 5.9 p.83 (MCF Fig 3)) | dPS_OG, AvgConf, AvgPS_OG, AvgPS_IG, dPS | AvgChiSq, dPS_IG, AvgPS_OG, AvgPS_IG, AvgConf (XGBoost, TreeExplainer) | NO — smoke run only | plan row B10: acceptance on real data only; which invented features carry information is a property of the generator. |

Reading aid, not a result: on this stand-in the response classifier's out-of-group accuracy is 0.959, and RP / AP succeed on 94.9% / 93.5% of the sub-sets. These levels are a property of the generator (plan §2.3: its confidence and prediction models are invented and were calibrated on the rule baselines only), not a finding. The generator was left as it is: adjusting it until the RP / AP columns looked like the thesis's would be exactly the tuning the plan forbids, and would make the columns meaningless in a less visible way. One practical consequence: this run exercises the tie rules and the empty-out-group rule rarely or never (section 3, last column); they are covered by unit tests.

## 2. Tables in the shape of the thesis's tables — SYNTHETIC — not a reproduction

### 2.1 RP strategies (TH Table 4.2) — SYNTHETIC — not a reproduction

#### Response-Prediction (RP): classification-aggregation strategies (shape of TH Table 4.2, T26) — SYNTHETIC — not a reproduction

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

##### Run diagnostics — SYNTHETIC — not a reproduction

- Instances per partition: 2490–2490 (share of correct responses 0.280); model fits: 2310; logistic regressions stopped at max_iter: 0.
- Out-of-group accuracy of the response classifier: 0.959; share classified "True": 0.248.
- Sub-sets in which every response gives the same answer (empty out-group, decision B1-2 / B6-4): 0 over all partitions (0 instances without out-group features).
- Sub-sets whose correct answer has no vote (RP cannot succeed there): 0.40 per partition; sub-sets with a tied majority (OQ28): 5.87 per partition.
- Library warnings during the fits: none.

### 2.2 Headline (TH Fig 5.3) — SYNTHETIC — not a reproduction

#### RP, AP and the rule baselines on the sub-sets (shape of TH Fig 5.3; T23, T24, T25) — SYNTHETIC — not a reproduction

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

##### AP run diagnostics — SYNTHETIC — not a reproduction

- The three-case rule (TH p.41) and argmax P(True) return the same answer in every sub-set of every partition (plan §4.4; asserted): 2310 sub-set evaluations, 0 differences.
- Which case applied (mean sub-sets per partition): exactly one answer "True" 61.03; several "True" 0.43; none "True" 15.53; single voted answer 0.00.
- Ties of the deciding probability (canonical order decides; decision B7-2): 0.00 sub-sets per partition.
- Instances per partition: 441–459 (share of correct answers 0.170); model fits: 2310; logistic regressions stopped at max_iter: 0.
- Sub-sets with a single voted answer (empty out-group, decision B1-2 / B6-4): 0 over all partitions; sub-sets whose correct answer has no vote (no correct instance, B2-3): 0.40 per partition.
- Library warnings during the fits: none.

### 2.3 Other classifiers (TH Tables A.7, A.8) — SYNTHETIC — not a reproduction

#### RP: classifiers and classification-aggregation strategies (shape of TH Table A.7, T27) — SYNTHETIC — not a reproduction

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

#### AP: classifiers (shape of TH Table A.8, T28) — SYNTHETIC — not a reproduction

Sub-sets per partition: 77; partition: random, 30 seeds (0..29) (plan OQ25); SD = sample SD over the seeds (ddof 1).

| Classifier | run (synthetic): success rate, mean ± SD | run (synthetic): count of 77, min–max | run (synthetic): tied sub-sets (mean) | run (synthetic): sub-sets where the three-case rule and argmax P(True) differ (total over the partitions) | thesis: success rate |
|---|---|---|---|---|---|
| Bag KNN | 92.2 ± 2.8% | 67–75 | 1.97 | 0 | 44% |
| SVM | 94.4 ± 2.3% | 68–76 | 0.00 | 0 | 43% |
| Vote soft | 93.5 ± 2.0% | 68–75 | 0.00 | 0 | 55% |

AP answer = the thesis's three-case rule (TH p.41). For Vote soft and Bag KNN it must equal argmax P(True) (plan §4.4; asserted in the run). For SVC the classification and the probability can disagree, so the two are compared and counted, not asserted (decision B8-1).

Printed for the table's shape only; no difference is computed and the synthetic values carry no information about the thesis's Table A.8.

##### Classifier diagnostics — SYNTHETIC — not a reproduction

- RP/soft_vote: share classified "True" 0.248 (share of correct instances 0.280); instances classified "False" although P(True) > 0.5: 0, classified "True" although P(True) <= 0.5: 0 (over all partitions); library warnings: none.
- RP/bag_knn: share classified "True" 0.251 (share of correct instances 0.280); instances classified "False" although P(True) > 0.5: 3, classified "True" although P(True) <= 0.5: 4 (over all partitions); library warnings: none.
- RP/svm: share classified "True" 0.249 (share of correct instances 0.280); instances classified "False" although P(True) > 0.5: 1399, classified "True" although P(True) <= 0.5: 0 (over all partitions); library warnings: {'FutureWarning: The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC()': 2310}.
- AP/soft_vote: share classified "True" 0.137 (share of correct instances 0.170); instances classified "False" although P(True) > 0.5: 0, classified "True" although P(True) <= 0.5: 0 (over all partitions); library warnings: none.
- AP/bag_knn: share classified "True" 0.136 (share of correct instances 0.170); instances classified "False" although P(True) > 0.5: 0, classified "True" although P(True) <= 0.5: 2 (over all partitions); library warnings: none.
- AP/svm: share classified "True" 0.127 (share of correct instances 0.170); instances classified "False" although P(True) > 0.5: 370, classified "True" although P(True) <= 0.5: 0 (over all partitions); library warnings: {'FutureWarning: The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC()': 2310}.

### 2.4 Significance tests (TH Table 5.11) — SYNTHETIC — not a reproduction

#### RP and AP against the rule baselines: p-values (shape of TH Table 5.11, T29) — SYNTHETIC — not a reproduction

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

##### Check of the test conventions on the thesis's own counts (no data involved)

Recomputed with cdm.eval.stats from the published counts (RP 49, AP 42, SP 33, MR 34, WC 34, HAC 22 of 77) and the discordant pairs of plan §5.8 [INFERRED]; beside the printed values of TH Table 5.11.

| Rule | vs RP, McNemar: recomputed (pair) / printed | vs RP, proportion: recomputed / printed | vs AP, McNemar: recomputed (pair) / printed | vs AP, proportion: recomputed / printed |
|---|---|---|---|---|
| SP | 8.0E-04 (18, 2); 0.001 (19, 3) / 0.001 | 0.005 / 0.005 | 0.027 (11, 2) / 0.027 | 0.073 / 0.073 |
| MR | 3.0E-04 (15, 0) / 3.0E-04 | 0.008 / 0.008 | 0.027 (9, 1) / 0.027 | 0.099 / 0.099 |
| WC | 3.0E-04 (15, 0) / 3.0E-04 | 0.008 / 0.008 | 0.027 (9, 1) / 0.027 | 0.099 / 0.099 |
| HAC | 4.9E-05 (34, 7) / 4.9E-05 | 6.4E-06 / 6.4E-06 | 0.003 (30, 10); 0.003 (31, 11) / 0.003 | 5.4E-04 / 0.001 |

A p-value is "undefined" when the two methods never disagree (McNemar, b + c = 0) or the pooled standard error is 0 (decision B9-1). The p-values of different partitions are not independent and are not pooled (decision B9-2).

The run columns are NOT a reproduction of Table 5.11: on the synthetic stand-in they only show that the tests run on the pipeline's output. The convention check above is independent of any data.

### 2.5 SHAP (TH Figs 5.8, 5.9) — SYNTHETIC — not a reproduction

#### SHAP rank orders of the RCR and ACR features (TH Figs 5.8, 5.9; T30) — SYNTHETIC — not a reproduction

**End-to-end smoke run on the synthetic stand-in.** Plan row B10 puts the acceptance of this step on real data only. The two orders below are printed side by side so that the output format can be checked; no agreement measure is computed and none should be read off: the synthetic responses were invented, so which of their features carry information is a property of the generator, not of the thesis data. Where a row of the two orders happens to coincide, that is not evidence either: the generator's prediction model was built on the SP mechanism (plan §2.3), which makes the predicted-support features informative by design.

Models trained on all instances of one partition (partition seed 0); importance = mean |SHAP|. KernelSHAP explains a seeded sample of 150 instances (background 50).

##### RCR features (RP); thesis order: TH Fig 5.8 p.82 (MCF Fig 2) — SYNTHETIC — not a reproduction

Instances: 2490 (left out for an undefined feature: 0).

| Rank | run (synthetic): xgboost_tree | run (synthetic): soft_vote_kernel | thesis: figure order |
|---|---|---|---|
| 1 | dPSv | dPSv | dPSv |
| 2 | PSvD_IG | PSvD_OG | PSvD_OG |
| 3 | PSvD_OG | PChiSq_OG | PSvD |
| 4 | PChiSq_OG | PSvD_IG | ConfD_IG |
| 5 | ChiSq | ChiSq | PSv |
| 6 | PChiSq_IG | PSv | ChosenMajAns |
| 7 | PSv | PChiSq_IG | ConfD_OG |
| 8 | ConfD_IG | PSvD | ConfD |
| 9 | Conf | Conf | PSvD_IG |
| 10 | PChiSq | ConfD_IG | ChiSq |
| 11 | ConfD | ChosenMajAns | PChiSq_IG |
| 12 | PSvD | ConfD_OG | Conf |
| 13 | ConfD_OG | PChiSq | PChiSq_OG |
| 14 | ChosenMajAns | ConfD | PChiSq |

- xgboost_tree: XGBoost, TreeExplainer, log-odds (plan OQ30 (i)); 2490 instances explained; largest |SHAP| 7.58.
- soft_vote_kernel: soft-voting ensemble of RP / AP, KernelExplainer, P(True) (plan OQ30 (ii)); 150 instances explained; largest |SHAP| 0.50.

##### ACR features (AP); thesis order: TH Fig 5.9 p.83 (MCF Fig 3) — SYNTHETIC — not a reproduction

Instances: 457 (left out for an undefined feature: 0).

| Rank | run (synthetic): xgboost_tree | run (synthetic): soft_vote_kernel | thesis: figure order |
|---|---|---|---|
| 1 | AvgChiSq | dPS_IG | dPS_OG |
| 2 | dPS_IG | AvgChiSq | AvgConf |
| 3 | AvgPS_OG | AvgPS_OG | AvgPS_OG |
| 4 | AvgPS_IG | AvgPS_IG | AvgPS_IG |
| 5 | AvgConf | Support | dPS |
| 6 | AvgPS | AvgPS | AvgChiSq |
| 7 | dPS | dPS | dPS_IG |
| 8 | Support | AvgConf | AvgPS |
| 9 | dPS_OG | dPS_OG | Support |
| 10 | IsMajority | IsMajority | IsMajority |

- xgboost_tree: XGBoost, TreeExplainer, log-odds (plan OQ30 (i)); 457 instances explained; largest |SHAP| 5.33.
- soft_vote_kernel: soft-voting ensemble of RP / AP, KernelExplainer, P(True) (plan OQ30 (ii)); 150 instances explained; largest |SHAP| 0.57.

### 3. Special cases and tie rules

| Case | Rule in use | Basis | How often in this run |
|---|---|---|---|
| Empty out-group: every response of a sub-set gives the same answer, so RCR features 5, 10, 14 and ACR features 5, 8 are undefined (decision B1-2, B2-1) | The instances are left out of every training set and get no prediction; the sub-set's answer is its only voted answer (the only candidate for RP and AP). Switch: `raise` (decision B6-4) | [INVENTED] labelled default; the thesis and the plan give no rule. Open question (decision B6-4) | RP: 0 sub-sets over all partitions; AP: 0 |
| Correct answer without a vote in the sub-set | RP and AP can only return a voted answer, so the sub-set counts as a failure | [EVIDENCE] TH p.41 (AP: only answers with at least one vote); implicit for RP | 0.40 sub-sets per partition |
| Tied majority answer (ChosenMajAns / IsMajority) | 1 for every answer tied for the most votes | plan OQ28 default [OPEN] | 5.87 sub-sets per partition |
| RP, maj / prop: tie on the primary criterion (including: no supporter of any answer classified "True") | the strategy's tie breaker (avgp, maxp or wm) among the tied answers; counted as a "tie breaker" case | [EVIDENCE] TH p.40; counting rule plan OQ27 [OPEN] | see the tie-breaker column of section 2.1 |
| RP, maj / prop: still tied after the tie breaker | first answer in the order of A_P | plan OQ27 [OPEN] | 0.00 sub-sets per partition (prop + avgp) |
| RP, maxp / avgp / wm alone: tie | seeded random choice among the tied answers; switch `canonical_first` | [EVIDENCE] TH Table A.7 prints the tie breaker "rnd"; reading it as random is plan §5.7 [INFERRED] | maxp 0.00, avgp 0.00, wm 0.00 sub-sets per partition |
| AP: tie of the deciding probability | first answer in the order of A_P | [INVENTED] by analogy with plan OQ27 (decision B7-2) | 0.00 sub-sets per partition |
| AP: three-case rule vs argmax P(True) | the literal three-case rule gives the answer; equality with argmax P(True) is asserted for the soft vote and Bag KNN, counted for SVC (its classification and probability can disagree) | [EVIDENCE] TH p.41; [INFERRED] plan §4.4; decision B7-1, B8-1 | soft vote: 0 differences; SVM: 0 (all partitions) |
| Float comparisons (probabilities, proportions) | two values within 1e-9 are tied | AM3 tolerance, extended to RP / AP [INVENTED] (decision B6-6) | — |
| Rule baselines MR / WC / SP / HAC: ties | first answer in the order of A_P | plan OQ8, AM3 (step B5, decision R3) | see the B5 output |

Leave-one-group-out: a group is the set of instances of one sub-set (TH p.57; 77 groups), as in the thesis. This implementation also contains a leave-one-task-out option (`group_by="task"`, cdm/eval/worldB_logo.py); it is a controlled variant of this implementation, is off in every stored run, and is planned to be reported beside the thesis procedure on the real data.

Strategy selection: as in the thesis (TH p.58), prop + avgp is the RP strategy used for the headline result; all nine strategies are reported.

## 4. Open points raised by these steps — SYNTHETIC — not a reproduction

Full entries with tags: `docs/IMPLEMENTATION_DECISIONS.md`, section "Phase B — World B (steps B6–B11)". The open questions:

- B6-2: weights, hyperparameters, scaling and seeds of the soft-voting ensemble are not stated (plan OQ26 defaults in use).
- B6-4: rule for a sub-set in which every response gives the same answer.
- B6-6: tie rules beyond the stated tie breaker.
- B6-7: the thesis's tie percentages differ between tie breakers of the same strategy (16 / 17 / 16; Bag KNN 8 / 16 / 12), which the plan's definition cannot produce from one set of predictions.
- B8-1: SVC's classification and probability can disagree; sklearn 1.9 deprecates `probability=True`.
- B9-2: how to report significance over 30 partitions.
- B10-1..3: model, partition and sample of the SHAP analysis.

## 5. Commands (the same for the real file) — SYNTHETIC — not a reproduction

```
python scripts/run_b5_rules.py     # rule baselines (T24)
python scripts/run_b6_rp.py        # RP, Table 4.2 (T26)
python scripts/run_b7_ap.py        # AP, Fig 5.3 (T23, T25)
python scripts/run_b8_alt.py       # Bag-KNN, SVM: Tables A.7, A.8 (T27, T28)
python scripts/run_b9_stats.py     # Table 5.11 (T29)
python scripts/run_b10_shap.py     # Figs 5.8, 5.9 (T30)
python scripts/run_b11_report.py   # this report
```

For the real file set `source: real` and the `real:` block of `config/worldB.yaml` (file paths, confidence scale, prediction scale). Outputs then go to `runs/worldB_real/` and the tables are labelled "run (real file)". UNTESTED ON REAL DATA.

Partitions: 30. Wall-clock minutes of the steps (3 processes): b6 13.5, b7 5.1, b8 8.7, b9 0.0, b10 4.8.
Library versions: {'python': '3.13.18', 'numpy': '2.5.3', 'sklearn': '1.9.1', 'xgboost': '3.4.1', 'shap': '0.54.1', 'statsmodels': '0.15.0'}.

**SYNTHETIC — not a reproduction**
