# AMP / DAP results on publicly available data

**Data:** 1,209 binary decision cases rebuilt from public datasets (General 500, NBA 393, NFL 212, Politics 104) under the assumptions in [DATA.md](DATA.md); build scenario `waiver_main`.
**Reported values:** from the thesis ("TH") and, where stated, the 2022 preprint of the AMP / DAP article (arXiv:2204.01721v1). The published 2025 article was not consulted.
**Reading the tables:** "reconstruction" is what this code produced; "reported" is what the source prints. Where the two differ, the reconstruction does not establish the reason. Stored artifacts for every table are indexed in [RESULT_FILES.md](RESULT_FILES.md).

Counts are successes out of the block size. Reported counts are derived from the printed rates; the derivation is in [RECONSTRUCTION_PLAN.md](RECONSTRUCTION_PLAN.md).

## 1. Summary

- In the General, NFL and Politics domains, the single leave-one-out results of the reconstruction are within 0 to 2 cases of the reported values for classifier seed 0, and within 9 cases across five seeds.
- In the NBA domain the values differ: about 0.63 in the reconstruction (245 to 254 of 393 across five seeds for AMP), 0.92 reported (362 of 393).
- Overall, under the thesis's protocol, AMP achieved 0.704 and DAP 0.706; the thesis reports 0.773 and 0.775. In the reconstruction, the AMP and DAP success counts are below the count of SP applied to every case (869), in every protocol and seed that was run; the thesis reports AMP and DAP above SP.
- The qualitative ablation pattern for DAP (no row significantly different from the full model) is the same as reported. For AMP the thesis reports two rows as significantly lower; in the reconstruction no row is.
- Classic meta-learning rates are within about one percentage point of the reported rates. Bagging shows no clear gain, as reported.

## 2. Rule aggregators on the rebuilt data

Successes of each rule applied to every case. Cells: reconstruction / reported / difference in cases.

| Block | n | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|---|
| All | 1209 | 869 / 864 / +5 | 838 / 831 / +7 | 831 / 830 / +1 | 767 / 774 / −7 | 368 / 372 / −4 |
| General | 500 | 413 / 413 / 0 | 385 / 385 / 0 | 380 / 380 / 0 | 338 / 338 / 0 | 111 / 108 / +3 |
| NBA | 393 | 260 / 258 / +2 | 256 / 252 / +4 | 252 / 253 / −1 | 230 / 237 / −7 | 137 / 142 / −5 |
| NFL | 212 | 112 / 109 / +3 | 114 / 111 / +3 | 117 / 114 / +3 | 131 / 131 / 0 | 99 / 101 / −2 |
| Politics | 104 | 84 / 84 / 0 | 83 / 83 / 0 | 82 / 83 / −1 | 68 / 68 / 0 | 21 / 21 / 0 |

The four classic rules agree exactly in the General domain; DA in General and the rules in the other domains differ by 1 to 7 cases. The present reconstruction does not establish the reason for these differences. Details: [RULE_BASELINE_CHECK.md](RULE_BASELINE_CHECK.md).

Coverage: at least one of MR, WC, HAC, SP is correct in 1,006 cases (83.2%; reported 83.3%). DA is correct in the remaining 203 cases, so coverage with DA is 100%, as reported.

## 3. Headline results

### 3.1 Thesis protocol: leave-one-out inside 10 folds, with model selection

| | Reconstruction | Reported (TH) | Difference (cases) |
|---|---|---|---|
| AMP, all cases | 851 / 1209 = 0.704 | 935 = 0.773 | −84 |
| DAP, all cases | 853 / 1209 = 0.706 | 937 = 0.775 | −84 |
| AMP, General | 414 / 500 = 0.828 | 412 = 0.824 | +2 |
| AMP, NBA | 248 / 393 = 0.631 | 361 = 0.919 | −113 |
| AMP, NFL | 120 / 212 = 0.566 | 100 = 0.472 | +20 |
| AMP, Politics | 69 / 104 = 0.663 | 62 = 0.596 | +7 |
| DAP by domain (General / NBA / NFL / Politics) | 420 / 251 / 118 / 64 | not printed | – |

- Proportional reduction in error against SP: the thesis reports 0.204 (AMP) and 0.211 (DAP). In the reconstruction AMP and DAP are below SP, so the quantity is negative.
- One-tailed proportion tests against each rule. Reported: all significant. Reconstruction, p for AMP / DAP: SP 0.79 / 0.76; MR 0.28 / 0.25; WC 0.19 / 0.17; HAC 1.4e-4 / 1.0e-4; DA < 1e-10 for both.
- Technique selected per fold. Reported: random-forest variants (AMP BR+RF in 8 folds and CC+RF in 2; DAP RF in all 10). Reconstruction: AMP LP+LR in 4 folds, BR+RF in 5, CC+RF in 1; DAP LR in 9 folds, RF in 1. In the reconstruction the inner comparisons are close (a margin of 1 or 2 cases in 5 of 10 AMP folds, on one seed), so the list of selected techniques is sensitive to small changes; the totals are less so.

Per-fold tables for all nine AMP techniques and three DAP techniques, beside thesis Tables 5.1 and 5.2: `results/waiver_main/a14/`.

### 3.2 Single leave-one-out (AMP = BR+RF, DAP = RF)

Reported values: thesis Tables 5.6, 5.7 (row 1), A.9, and the preprint.

| | Seed 0 | Seeds 0–4: min–max (mean) | Reported |
|---|---|---|---|
| AMP, all | 855 (0.707) | 850–863 (857.2) | 975 (0.806) |
| AMP, General | 414 | 407–414 (410.2) | 416 |
| AMP, NBA | 245 | 245–254 (249.8) | 362 |
| AMP, NFL | 117 | 117–126 (119.8) | 117 |
| AMP, Politics | 79 | 75–80 (77.4) | 80 |
| DAP, all | 859 (0.711) | 857–868 (861.8) | 982 (0.812) |
| DAP, General | 425 | 420–426 (423.4) | 426 |
| DAP, NBA | 238 | 238–244 (241.0) | 357 |
| DAP, NFL | 122 | 119–123 (121.6) | 123 |
| DAP, Politics | 74 | 74–78 (75.8) | 76 |

The sources do not state random seeds. Seed 0 is this repository's default; seeds 1 to 4 show the spread.

## 4. Effectiveness of AMP's selection (thesis protocol, all cases)

Cells: reconstruction / reported (TH Table 5.4).

| Method | Times selected | P(success of the method) | P(success given selected) | PRE | p |
|---|---|---|---|---|---|
| SP | 428 / 349 | .719 / .715 | .689 / .822 | −.105 / .377 | .91 / 4.2e-6 |
| MR | 185 / 244 | .693 / .687 | .773 / .820 | .260 / .423 | .009 / 4.1e-6 |
| WC | 322 / 241 | .687 / .687 | .817 / .834 | .414 / .471 | 2.7e-7 / 4.0e-7 |
| HAC | 171 / 132 | .634 / .640 | .637 / .705 | .008 / .179 | .47 / .062 |
| DA | 103 / 243 | .304 / .308 | .398 / .634 | .135 / .471 | .019 / 0 |

Selection counts depend on how ties between labels with equal predicted probability are broken (81 tied cases in this run); the sources do not state a rule. Over all resolutions of the 81 tied cases, successes range from 846 to 854. Per-domain and leave-one-out versions (TH Tables 5.5 and A.10): `results/waiver_main/a15/`.

## 5. Ablations (single leave-one-out, seed 0)

Reported values: TH Tables 5.6 (AMP) and 5.7 (DAP).

| Row | Removed | AMP reconstruction | AMP reported | DAP reconstruction | DAP reported |
|---|---|---|---|---|---|
| 1 | nothing | .707 | .806 | .711 | .812 |
| 2 | confidence features | .688 | .800 | .684 | .812 |
| 3 | prediction features | .711 | .816 | .712 | .825 |
| 4 | confidence and prediction features | .685 | .795 | .686 | .803 |
| 5 | WC, HAC | .718 | .808 | .718 | .804 |
| 6 | SP | .701 | .797 | .709 | .805 |
| 7 | WC, HAC, SP | .701 | .791 | .715 | .806 |
| 8 | DA | .711 | .754 | .709 | .813 |
| 9 | confidence features, WC, HAC | .696 | .806 | .694 | .811 |
| 10 | prediction features, SP | .691 | .799 | .714 | .815 |
| 11 | confidence and prediction features, WC, HAC, SP | .682 | .763 | .695 | .798 |

- DAP: no row differs significantly from row 1 in either the reconstruction or the thesis.
- AMP: the thesis reports rows 8 and 11 as significantly lower than row 1. In the reconstruction no row differs significantly, and row 8 (.711) is not lower than row 1 (.707).
- The thesis names a "two-tailed proportion test"; the pooled two-proportion z-test used here is this implementation's reading. Most row differences in the reconstruction are of the size of the seed-to-seed spread in section 3.2.

## 6. Classic meta-learning (decision tree on the aggregators' answers)

Leave-one-out; the thesis does not state the protocol for this analysis. Reported: TH Table 5.8.

| Row | Reconstruction | Reported |
|---|---|---|
| 1, all aggregators | .719 | .713 |
| 2, without WC, HAC | .719 | .708 |
| 3, without SP | .693 | .695 |
| 4, without WC, HAC, SP | .693 | .687 |
| 5, without DA | .719 | .713 |

In the reconstruction the fitted trees return SP's answer (rows 1, 2, 5) or MR's answer (rows 3, 4) for every case.

## 7. Bagging

Overall successes of 1,209 over 10 folds, RF (DAP) and BR+RF (AMP) in every fold. Per-fold values beside TH Tables 5.9 and 5.10: `results/waiver_main/a18_*/`.

| F (new instances per case) | DAP | AMP |
|---|---|---|
| 10 | 857 | 842 |
| 5 | 852 | 845 |
| 3 | 850 | 837 |
| 2 | 866 | 830 |
| no bagging, technique selected per fold (section 3.1) | 853 | 851 |

- Only cases with at least 70 responses can be bagged (768 of 1,209 in the rebuilt data).
- Three details are not specified in the sources and are implemented as documented defaults with sensitivity runs: the sub-group size formula (as printed; Cochran's form as sensitivity), sampling without replacement, and the treatment of a sampled sub-group with no minority answer.
- The thesis reports no significant and consistent evidence that bagging increased or decreased the success rate; the reconstruction also shows no consistent gain or loss.

## 8. Feature importance (SHAP)

Models fitted on all 1,209 cases; two variants (XGBoost with TreeExplainer; the random forest of BR+RF with KernelExplainer), because the sources do not state which model the figures explain. Rank tables beside the orders read from TH Figures 5.4–5.7 and A.1: `results/waiver_main/a19/`.

- Reported: the sub-group feature `SG_B_Amax` is the highest-ranked feature for the MR, DA, SP and WC labels.
- Reconstruction: `SG_B_Amax` ranks between 24 and 27 of 27; the highest-ranked features are `D_Smax_Smin` and `AvgC` (MR, DA, SP) and `B_Amax_MaxCa` (HAC).
- The thesis's observation that the MR and DA explanations mirror each other is also seen in the reconstruction (the ten most important features have opposite directions).
- Of the 12 feature directions stated in the thesis text, 11 have the same sign under the XGBoost variant.

## 9. What was checked

Every number in this document was recomputed from the stored per-case files by scripts independent of the table-generating code, a sample of predictions was refitted from scratch, and train/test separation was checked for each protocol. These were automated, AI-assisted checks; see the [README](../README.md#attribution). They establish that the numbers are what this implementation produces under the documented configuration. They do not establish how closely the implementation or the rebuilt data matches the original study.
