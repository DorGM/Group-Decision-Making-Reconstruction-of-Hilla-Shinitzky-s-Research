# RP / AP pipeline run on synthetic stand-in data

> **SYNTHETIC — not a reproduction.** Every number produced by this repository for RP and AP was computed on generated data. The original participant data has not yet been supplied.

## 1. What this run is, and what it is not

**It is** an end-to-end execution of the RP / AP code: instance construction with the RCR and ACR features, the classifiers, leave-one-group-out evaluation over 77 sub-sets, the nine classification-aggregation strategies, the alternative classifiers, the statistical tests and SHAP, producing tables in the layout of the corresponding thesis tables.

**It is not** a reproduction of the original study, a validation of its empirical findings, or evidence about the performance RP and AP achieve on real participants. Specifically:

- The stand-in responses come from an invented response model ([DATA.md](DATA.md), section 3). The success rates below are properties of that model.
- The rule-aggregator baselines were calibration targets, so their agreement with the published counts is by construction, and it holds for the one shipped dataset (generator seed 0).
- RP / AP outcomes were not calibration targets and were never tuned.
- No difference, improvement or reproduction gap is computed between these synthetic outcomes and the values reported for the original data, and none should be read into the numbers.

For reference only, the values reported for the original data are RP 0.636 and AP 0.545 of 77 sub-sets, against 0.442 for the majority rule (thesis Figure 5.3; Shinitzky et al., 2024). They are not comparable with the synthetic run.

## 2. Configuration of the run

| Item | Value |
|---|---|
| Data | synthetic stand-in, 22 tasks, 2,490 responses (generator seed 0) |
| Sub-sets | 77 per partition; random partition of each task's responses; 30 partition seeds (0–29) |
| Reported statistic | mean ± sample standard deviation over the 30 partitions |
| RP classifier | soft-voting ensemble (LDA, RF, XGBoost, LR, KNN), library defaults |
| Evaluation | leave-one-group-out; a group is the set of instances of one sub-set, as in the thesis |

## 3. Outputs of the synthetic run — SYNTHETIC — not a reproduction

### 3.1 Success on the 77 sub-sets

| Method | Successes of 77 (mean ± SD) | Range | Rate |
|---|---|---|---|
| RP (soft vote, proportion with average-probability tie breaker) | 73.10 ± 1.94 | 69–77 | 0.949 |
| AP (soft vote) | 72.00 ± 1.58 | 68–75 | 0.935 |
| MR (calibration target) | 33.53 ± 1.93 | 30–38 | 0.435 |
| WC (calibration target) | 34.30 ± 2.09 | 30–37 | 0.445 |
| SP (calibration target) | 33.00 ± 2.53 | 27–37 | 0.429 |
| HAC (calibration target) | 22.73 ± 2.86 | 18–30 | 0.295 |

In the stand-in, responses with the correct answer are comparatively easy to separate by their features (a single feature, `dPSv`, reached an AUC of about 0.83 on one partition; computed once, not a stored output). The high RP / AP rates reflect this property of the generator.

### 3.2 RP classification-aggregation strategies (layout of thesis Table 4.2)

| Strategy | Tie breaker | Success (%) | Sub-sets decided by the tie breaker (%) |
|---|---|---|---|
| maxp | – | 94.5 ± 1.5 | 0.0 |
| avgp | – | 95.2 ± 2.1 | 0.0 |
| wm | – | 91.5 ± 2.0 | 0.0 |
| maj | avgp | 95.9 ± 2.0 | 7.8 |
| maj | maxp | 94.8 ± 1.6 | 7.8 |
| maj | wm | 92.9 ± 2.0 | 7.8 |
| prop | avgp | 94.9 ± 2.5 | 7.8 |
| prop | maxp | 93.9 ± 2.2 | 7.8 |
| prop | wm | 92.3 ± 2.6 | 7.8 |

### 3.3 Alternative classifiers (layout of thesis Tables A.7 and A.8)

| Classifier | RP success over the nine strategies (%) | AP success (%) |
|---|---|---|
| Bagged KNN | 91.4–94.2 | 92.2 ± 2.8 |
| SVM | 92.6–95.1 | 94.4 ± 2.3 |
| Soft vote | 91.5–95.9 | 93.5 ± 2.0 |

### 3.4 Statistical tests and SHAP

- The McNemar and proportion tests of RP and AP against each rule run on every partition; on the synthetic data all are below 0.05 in all 30 partitions. This carries no information about the original study.
- SHAP was run as a smoke test only (one partition; XGBoost with TreeExplainer, and KernelExplainer on a sample for the soft vote). Which generated features carry information is a property of the generator, so rank orders are not interpreted.

Full outputs: `results/worldB_synthetic/` ([RESULT_FILES.md](RESULT_FILES.md)).

## 4. A check that does not depend on the synthetic data

The statistical conventions were checked against the publication itself. From the published success counts (RP 49, AP 42; SP 33, MR 34, WC 34, HAC 22 of 77), the pooled one-tailed two-proportion test gives the eight proportion-test values of thesis Table 5.11 after rounding to three decimals. The eight McNemar values are matched, with a continuity-corrected chi-square, by integer discordant-pair counts that were inferred because they fit; these counts are not published. This identifies the test variants implemented in `cdm/eval/stats.py`; it is not a recomputation on data.

## 5. Points that remain open until the original data is available

- File format, confidence scale and prediction scale of the original data (the loader's assumptions L1–L12).
- The original assignment of responses to the 77 sub-sets; the thesis text gives 2,514 responses, and the per-task response counts in Table 4.1 sum to 2,490.
- Settings of the soft-voting ensemble (weights, hyperparameters, scaling, seeds).
- Tie rules beyond the stated tie breaker, the definition behind the printed "percentage of tie breakers", and the handling of a sub-set in which all responses choose the same answer.
- Leave-one-group-out follows the thesis (a group is one sub-set). A leave-one-task-out option exists in this implementation as a controlled variant and is planned to be reported beside the thesis procedure on the real data.

All are recorded in [IMPLEMENTATION_DECISIONS.md](IMPLEMENTATION_DECISIONS.md).

## 6. Real-data path

The same commands run on a real data file through `config/worldB.yaml` (`source: real`) and write to a separate `worldB_real` folder. **This path is untested on real data.** It will be exercised, and this document replaced by a real-data report, once the original data is received.
