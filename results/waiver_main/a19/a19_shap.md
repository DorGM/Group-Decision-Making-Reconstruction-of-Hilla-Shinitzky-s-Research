# A19 SHAP values of the CDMCR features - build `waiver_main`, seed 0

Written 2026-11-12T10:38:45Z. 1209 cases, 27 features; each AMP label (O_MR, O_DA, O_SP, O_HAC, O_WC) is the target of its own binary model (TH p.60). Models are trained on all cases and SHAP values are computed for all cases (plan OQ30). Importance = mean |SHAP| over the cases; direction = Spearman correlation between a feature's value and its SHAP value (positive: high values push towards "the method is correct").

The thesis states neither the model nor the explainer behind Figs 5.4-5.7 and A.1 (plan §5.9). Two analyses are run:

- `xgboost_tree`: XGBoost per label, TreeExplainer, log-odds (plan OQ30 (i)) (10 s).
- `rf_kernel`: Random Forest of the label (as in BR+RF), KernelExplainer, probability (plan OQ30 (ii)); background 50 cases, nsamples auto (1392 s).

The `thesis` columns were read from the rendered figures (19 features listed per figure; the other 8 are merged there into "Sum of 8 other features"). `run` and `thesis` are always separate columns.

Note: on this build SG_B_Amax is close to the bottom of every ranking below; the thesis figures list it as the top feature for MR, DA and SP, with SHAP values down to about -5. These rank orders differ, and the present reconstruction does not establish the reason. Nothing is tuned.

## Statements of TH p.78-79: thesis-reported and run

| statement (plan §9.3 row A19; TH p.78-79) | model | run | thesis | same |
|---|---|---|---|---|
| top feature for MR | xgboost_tree | D_Smax_Smin (SG_B_Amax is rank 24) | SG_B_Amax | no |
| top feature for DA | xgboost_tree | AvgC (SG_B_Amax is rank 24) | SG_B_Amax | no |
| top feature for SP | xgboost_tree | AvgC (SG_B_Amax is rank 24) | SG_B_Amax | no |
| top feature for HAC | xgboost_tree | B_Amax_MaxCa (CAmin is rank 7) | CAmin | no |
| MR vs DA mirror image | xgboost_tree | opposite direction in 10 of MR's top 10 features; Spearman of the importances 0.94; correlation of the SHAP values -0.88 | "an almost mirror image" (no number) | yes |
| top feature for MR | rf_kernel | D_Smax_Smin (SG_B_Amax is rank 26) | SG_B_Amax | no |
| top feature for DA | rf_kernel | D_Smax_Smin (SG_B_Amax is rank 26) | SG_B_Amax | no |
| top feature for SP | rf_kernel | AvgC (SG_B_Amax is rank 27) | SG_B_Amax | no |
| top feature for HAC | rf_kernel | B_Amax_MaxCa (CAmin is rank 8) | CAmin | no |
| MR vs DA mirror image | rf_kernel | opposite direction in 10 of MR's top 10 features; Spearman of the importances 0.94; correlation of the SHAP values -0.87 | "an almost mirror image" (no number) | yes |

## Rank order against the thesis figures (T30)

| label | thesis figure | model | top feature: run | thesis | run rank of the thesis top feature | Spearman, 27 features | Spearman, 19 listed | top-5 overlap | top-10 overlap | top-19 overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| MR | TH Fig. 5.4 (p.80) | xgboost_tree | D_Smax_Smin | SG_B_Amax | 24 | 0.41 | 0.02 | 3/5 | 5/10 | 16/19 |
| MR | TH Fig. 5.4 (p.80) | rf_kernel | D_Smax_Smin | SG_B_Amax | 26 | -0.06 | -0.16 | 1/5 | 3/10 | 14/19 |
| DA | TH Fig. 5.5 (p.80) | xgboost_tree | AvgC | SG_B_Amax | 24 | 0.36 | 0.19 | 2/5 | 4/10 | 16/19 |
| DA | TH Fig. 5.5 (p.80) | rf_kernel | D_Smax_Smin | SG_B_Amax | 26 | -0.04 | 0.13 | 1/5 | 4/10 | 14/19 |
| SP | TH Fig. 5.6 (p.81) | xgboost_tree | AvgC | SG_B_Amax | 24 | 0.36 | -0.28 | 1/5 | 5/10 | 17/19 |
| SP | TH Fig. 5.6 (p.81) | rf_kernel | AvgC | SG_B_Amax | 27 | -0.12 | -0.08 | 0/5 | 5/10 | 13/19 |
| HAC | TH Fig. 5.7 (p.81) | xgboost_tree | B_Amax_MaxCa | CAmin | 7 | 0.66 | 0.51 | 2/5 | 8/10 | 16/19 |
| HAC | TH Fig. 5.7 (p.81) | rf_kernel | B_Amax_MaxCa | CAmin | 8 | 0.07 | 0.47 | 2/5 | 5/10 | 12/19 |
| WC | TH Fig. A.1 (p.113) | xgboost_tree | D_Smax_Smin | SG_B_Amax | 24 | 0.47 | 0.04 | 2/5 | 5/10 | 15/19 |
| WC | TH Fig. A.1 (p.113) | rf_kernel | AvgC | SG_B_Amax | 26 | -0.04 | 0.18 | 1/5 | 3/10 | 14/19 |

Spearman, 27 features: the thesis's 8 unlisted features share the average of ranks 20-27. Spearman, 19 listed: over the features the figure lists.

## Directions stated in the thesis text

| label | feature | thesis text (TH p.78-79): high values go with | run (xgboost_tree): Spearman(feature, SHAP) | agrees | run (rf_kernel): Spearman(feature, SHAP) | agrees |
|---|---|---|---|---|---|---|
| MR | SG_B_Amax | label = 0 (incorrect) | -0.29 | yes | 0.11 | no |
| MR | AvgPSV | label = 1 (correct) | 0.65 | yes | 0.69 | yes |
| MR | D_Smax_Smin | label = 1 (correct) | 0.92 | yes | 0.77 | yes |
| MR | CAmin | label = 0 (incorrect) | -0.67 | yes | -0.07 | yes |
| DA | SG_B_Amax | label = 1 (correct) | 0.19 | yes | -0.16 | no |
| DA | D_Smax_Smin | label = 0 (incorrect) | -0.89 | yes | -0.78 | yes |
| DA | CAmin | label = 1 (correct) | 0.39 | yes | 0.00 | yes |
| SP | SG_B_Amax | label = 0 (incorrect) | 0.03 | no | 0.15 | no |
| SP | CAmin | label = 0 (incorrect) | -0.35 | yes | -0.13 | yes |
| SP | N | label = 0 (incorrect) | -0.47 | yes | -0.26 | yes |
| HAC | CAmin | label = 0 (incorrect) | -0.64 | yes | -0.61 | yes |
| HAC | B_Amax_MaxCa | label = 1 (correct) | 0.73 | yes | 0.69 | yes |

## Size of the SHAP values

| label | model | smallest SHAP value | largest | base value | thesis x-axis (read from the figure, approximate) |
|---|---|---|---|---|---|
| MR | xgboost_tree | -2.24 | 2.25 | 0.949 | -5.3 .. +3.9 |
| MR | rf_kernel | -0.14 | 0.10 | 0.717 | -5.3 .. +3.9 |
| DA | xgboost_tree | -2.35 | 2.46 | -0.953 | -4.1 .. +5.5 |
| DA | rf_kernel | -0.10 | 0.14 | 0.280 | -4.1 .. +5.5 |
| SP | xgboost_tree | -1.96 | 2.54 | 1.089 | -5.2 .. +2.9 |
| SP | rf_kernel | -0.15 | 0.08 | 0.718 | -5.2 .. +2.9 |
| HAC | xgboost_tree | -2.77 | 2.29 | 0.638 | -3.9 .. +2.5 |
| HAC | rf_kernel | -0.28 | 0.09 | 0.639 | -3.9 .. +2.5 |
| WC | xgboost_tree | -2.60 | 2.50 | 0.914 | -5.6 .. +3.2 |
| WC | rf_kernel | -0.15 | 0.10 | 0.744 | -5.6 .. +3.2 |

`xgboost_tree` values are in log-odds units, `rf_kernel` values in probability units (at most 1 in absolute value).

## MR label (TH Fig. 5.4 (p.80))

| rank | run (xgboost_tree): feature | mean abs SHAP | direction | run (rf_kernel): feature | mean abs SHAP | direction | thesis (TH Fig. 5.4 (p.80)): feature | run rank (xgboost_tree) of the thesis feature | run rank (rf_kernel) of the thesis feature |
|---|---|---|---|---|---|---|---|---|---|
| 1 | D_Smax_Smin | 0.6702 | 0.92 | D_Smax_Smin | 0.0254 | 0.77 | SG_B_Amax | 24 | 26 |
| 2 | AvgC | 0.5996 | 0.65 | D_S_Uniform | 0.0245 | 0.75 | SG_D_MaxES_MinES | 18 | 21 |
| 3 | MaxPSa | 0.5151 | -0.77 | AvgC | 0.0238 | 0.67 | MaxPSa | 3 | 8 |
| 4 | AvgPSV | 0.4854 | 0.65 | VarS | 0.0235 | 0.73 | AvgPSV | 4 | 6 |
| 5 | CAmax | 0.3520 | 0.67 | ES | 0.0228 | -0.76 | D_Smax_Smin | 1 | 1 |
| 6 | SG_D_MaxVarC_MinVarC | 0.3469 | -0.27 | AvgPSV | 0.0183 | 0.69 | CAmin | 10 | 11 |
| 7 | VarC | 0.3258 | 0.39 | SG_AvgCAmax | 0.0177 | 0.71 | SG_VarSAmax | 8 | 16 |
| 8 | SG_VarSAmax | 0.3247 | -0.73 | MaxPSa | 0.0163 | -0.57 | N | 12 | 15 |
| 9 | P_lowPSv_highC | 0.3159 | -0.56 | CAmax | 0.0149 | 0.61 | SG_VarCAmin | 16 | 25 |
| 10 | CAmin | 0.2803 | -0.67 | MinPSa | 0.0138 | 0.56 | SG_VarCAmax | 17 | 17 |
| 11 | SG_D_MaxAvgC_MinAvgC | 0.2638 | 0.04 | CAmin | 0.0107 | -0.07 | D_MaxC_AvgC | 20 | 18 |
| 12 | N | 0.2431 | -0.00 | SG_AvgCAmin | 0.0086 | 0.11 | VarC | 7 | 13 |
| 13 | SG_AvgCAmin | 0.2394 | 0.07 | VarC | 0.0083 | 0.41 | P_lowPSv_highC | 9 | 14 |
| 14 | SG_AvgCAmax | 0.2296 | 0.46 | P_lowPSv_highC | 0.0078 | -0.60 | SG_D_MaxAvgC_MinAvgC | 11 | 23 |
| 15 | P_lowC_highPSv | 0.2294 | -0.14 | N | 0.0072 | -0.08 | CAmax | 5 | 9 |
| 16 | SG_VarCAmin | 0.2124 | -0.30 | SG_VarSAmax | 0.0070 | -0.30 | SG_D_MaxVarS_MinVarS | 23 | 20 |
| 17 | SG_VarCAmax | 0.2116 | 0.51 | SG_VarCAmax | 0.0068 | 0.11 | AvgC | 2 | 3 |
| 18 | SG_D_MaxES_MinES | 0.2042 | 0.43 | D_MaxC_AvgC | 0.0064 | -0.20 | SG_AvgCAmax | 14 | 7 |
| 19 | ES | 0.1858 | -0.82 | SG_D_MaxVarC_MinVarC | 0.0058 | -0.17 | SG_AvgCAmin | 13 | 12 |
| 20 | D_MaxC_AvgC | 0.1355 | 0.09 | SG_D_MaxVarS_MinVarS | 0.0053 | -0.34 | (one of the 8 not listed) | – | – |
| 21 | MinPSa | 0.1242 | 0.60 | SG_D_MaxES_MinES | 0.0052 | 0.02 | (one of the 8 not listed) | – | – |
| 22 | MinC | 0.1066 | -0.58 | P_lowC_highPSv | 0.0051 | -0.15 | (one of the 8 not listed) | – | – |
| 23 | SG_D_MaxVarS_MinVarS | 0.0745 | 0.04 | SG_D_MaxAvgC_MinAvgC | 0.0048 | -0.03 | (one of the 8 not listed) | – | – |
| 24 | SG_B_Amax | 0.0260 | -0.29 | MinC | 0.0034 | -0.40 | (one of the 8 not listed) | – | – |
| 25 | B_Amax_MaxCa | 0.0083 | -0.37 | SG_VarCAmin | 0.0032 | -0.34 | (one of the 8 not listed) | – | – |
| 26 | VarS | 0.0000 | – | SG_B_Amax | 0.0008 | 0.11 | (one of the 8 not listed) | – | – |
| 27 | D_S_Uniform | 0.0000 | – | B_Amax_MaxCa | 0.0002 | -0.29 | (one of the 8 not listed) | – | – |

Figures: `a19_xgboost_tree_MR.png`, `a19_rf_kernel_MR.png`

## DA label (TH Fig. 5.5 (p.80))

| rank | run (xgboost_tree): feature | mean abs SHAP | direction | run (rf_kernel): feature | mean abs SHAP | direction | thesis (TH Fig. 5.5 (p.80)): feature | run rank (xgboost_tree) of the thesis feature | run rank (rf_kernel) of the thesis feature |
|---|---|---|---|---|---|---|---|---|---|
| 1 | AvgC | 0.7471 | -0.67 | D_Smax_Smin | 0.0258 | -0.78 | SG_B_Amax | 24 | 26 |
| 2 | D_Smax_Smin | 0.6539 | -0.89 | D_S_Uniform | 0.0234 | -0.76 | SG_D_MaxES_MinES | 16 | 16 |
| 3 | MaxPSa | 0.5626 | 0.84 | VarS | 0.0211 | -0.71 | MaxPSa | 3 | 8 |
| 4 | AvgPSV | 0.5055 | -0.64 | AvgC | 0.0209 | -0.70 | D_Smax_Smin | 2 | 1 |
| 5 | SG_D_MaxVarC_MinVarC | 0.3570 | 0.31 | AvgPSV | 0.0202 | -0.74 | CAmin | 6 | 10 |
| 6 | CAmin | 0.3128 | 0.39 | ES | 0.0201 | 0.74 | AvgPSV | 4 | 5 |
| 7 | VarC | 0.3073 | -0.33 | CAmax | 0.0187 | -0.75 | SG_VarSAmax | 12 | 20 |
| 8 | CAmax | 0.3073 | -0.41 | MaxPSa | 0.0161 | 0.56 | D_MaxC_AvgC | 17 | 12 |
| 9 | P_lowPSv_highC | 0.2999 | 0.33 | SG_AvgCAmax | 0.0148 | -0.57 | N | 15 | 15 |
| 10 | SG_D_MaxAvgC_MinAvgC | 0.2736 | -0.19 | CAmin | 0.0142 | 0.00 | SG_VarCAmax | 13 | 18 |
| 11 | P_lowC_highPSv | 0.2569 | 0.20 | MinPSa | 0.0135 | -0.57 | SG_D_MaxAvgC_MinAvgC | 10 | 17 |
| 12 | SG_VarSAmax | 0.2330 | 0.70 | D_MaxC_AvgC | 0.0101 | 0.37 | VarC | 7 | 14 |
| 13 | SG_VarCAmax | 0.2278 | -0.33 | SG_AvgCAmin | 0.0082 | -0.01 | SG_VarCAmin | 19 | 24 |
| 14 | ES | 0.2113 | 0.86 | VarC | 0.0077 | -0.35 | P_lowPSv_highC | 9 | 22 |
| 15 | N | 0.2100 | 0.06 | N | 0.0070 | 0.17 | SG_D_MaxVarS_MinVarS | 22 | 23 |
| 16 | SG_D_MaxES_MinES | 0.2024 | -0.55 | SG_D_MaxES_MinES | 0.0068 | -0.28 | CAmax | 8 | 7 |
| 17 | D_MaxC_AvgC | 0.1986 | -0.52 | SG_D_MaxAvgC_MinAvgC | 0.0065 | 0.04 | SG_AvgCAmax | 18 | 9 |
| 18 | SG_AvgCAmax | 0.1949 | -0.53 | SG_VarCAmax | 0.0065 | -0.21 | SG_AvgCAmin | 20 | 13 |
| 19 | SG_VarCAmin | 0.1824 | 0.20 | SG_D_MaxVarC_MinVarC | 0.0059 | 0.18 | SG_D_MaxVarC_MinVarC | 5 | 19 |
| 20 | SG_AvgCAmin | 0.1798 | 0.07 | SG_VarSAmax | 0.0058 | 0.37 | (one of the 8 not listed) | – | – |
| 21 | MinPSa | 0.1537 | -0.58 | P_lowC_highPSv | 0.0052 | 0.17 | (one of the 8 not listed) | – | – |
| 22 | SG_D_MaxVarS_MinVarS | 0.1147 | 0.56 | P_lowPSv_highC | 0.0046 | 0.45 | (one of the 8 not listed) | – | – |
| 23 | MinC | 0.0840 | 0.25 | SG_D_MaxVarS_MinVarS | 0.0046 | 0.33 | (one of the 8 not listed) | – | – |
| 24 | SG_B_Amax | 0.0331 | 0.19 | SG_VarCAmin | 0.0035 | 0.40 | (one of the 8 not listed) | – | – |
| 25 | B_Amax_MaxCa | 0.0208 | 0.71 | MinC | 0.0026 | 0.34 | (one of the 8 not listed) | – | – |
| 26 | VarS | 0.0000 | – | SG_B_Amax | 0.0003 | -0.16 | (one of the 8 not listed) | – | – |
| 27 | D_S_Uniform | 0.0000 | – | B_Amax_MaxCa | 0.0003 | 0.31 | (one of the 8 not listed) | – | – |

Figures: `a19_xgboost_tree_DA.png`, `a19_rf_kernel_DA.png`

## SP label (TH Fig. 5.6 (p.81))

| rank | run (xgboost_tree): feature | mean abs SHAP | direction | run (rf_kernel): feature | mean abs SHAP | direction | thesis (TH Fig. 5.6 (p.81)): feature | run rank (xgboost_tree) of the thesis feature | run rank (rf_kernel) of the thesis feature |
|---|---|---|---|---|---|---|---|---|---|
| 1 | AvgC | 0.8570 | 0.76 | AvgC | 0.0318 | 0.70 | SG_B_Amax | 24 | 27 |
| 2 | D_Smax_Smin | 0.5353 | 0.74 | VarS | 0.0218 | 0.69 | D_Smax_Smin | 2 | 8 |
| 3 | VarC | 0.4200 | 0.65 | CAmax | 0.0203 | 0.68 | SG_D_MaxES_MinES | 15 | 23 |
| 4 | SG_D_MaxVarC_MinVarC | 0.4143 | -0.39 | AvgPSV | 0.0182 | 0.65 | MaxPSa | 6 | 15 |
| 5 | CAmax | 0.3511 | 0.76 | D_MaxC_AvgC | 0.0176 | -0.49 | SG_VarSAmax | 17 | 19 |
| 6 | MaxPSa | 0.3493 | -0.84 | D_S_Uniform | 0.0154 | 0.69 | N | 14 | 16 |
| 7 | AvgPSV | 0.3193 | 0.05 | SG_AvgCAmax | 0.0136 | 0.49 | D_MaxC_AvgC | 22 | 5 |
| 8 | P_lowC_highPSv | 0.2888 | 0.05 | D_Smax_Smin | 0.0133 | 0.66 | CAmax | 5 | 3 |
| 9 | SG_D_MaxAvgC_MinAvgC | 0.2803 | -0.09 | SG_D_MaxVarC_MinVarC | 0.0131 | -0.33 | AvgPSV | 7 | 4 |
| 10 | CAmin | 0.2520 | -0.35 | CAmin | 0.0114 | -0.13 | CAmin | 10 | 10 |
| 11 | SG_AvgCAmax | 0.2519 | 0.37 | MinPSa | 0.0107 | 0.60 | VarC | 3 | 13 |
| 12 | P_lowPSv_highC | 0.2470 | -0.42 | ES | 0.0103 | -0.62 | SG_VarCAmin | 16 | 24 |
| 13 | SG_VarCAmax | 0.2418 | -0.33 | VarC | 0.0100 | 0.43 | SG_VarCAmax | 13 | 22 |
| 14 | N | 0.2377 | -0.47 | SG_AvgCAmin | 0.0095 | 0.57 | SG_D_MaxAvgC_MinAvgC | 9 | 18 |
| 15 | SG_D_MaxES_MinES | 0.2178 | 0.26 | MaxPSa | 0.0089 | -0.56 | SG_D_MaxVarS_MinVarS | 19 | 21 |
| 16 | SG_VarCAmin | 0.2125 | -0.03 | N | 0.0088 | -0.26 | P_lowPSv_highC | 12 | 17 |
| 17 | SG_VarSAmax | 0.2118 | -0.62 | P_lowPSv_highC | 0.0074 | -0.54 | AvgC | 1 | 1 |
| 18 | MinPSa | 0.1926 | 0.79 | SG_D_MaxAvgC_MinAvgC | 0.0069 | -0.23 | SG_D_MaxVarC_MinVarC | 4 | 9 |
| 19 | SG_D_MaxVarS_MinVarS | 0.1672 | -0.58 | SG_VarSAmax | 0.0061 | -0.37 | P_lowC_highPSv | 8 | 20 |
| 20 | SG_AvgCAmin | 0.1516 | 0.28 | P_lowC_highPSv | 0.0056 | 0.02 | (one of the 8 not listed) | – | – |
| 21 | ES | 0.1470 | -0.64 | SG_D_MaxVarS_MinVarS | 0.0047 | -0.28 | (one of the 8 not listed) | – | – |
| 22 | D_MaxC_AvgC | 0.1374 | -0.09 | SG_VarCAmax | 0.0043 | -0.28 | (one of the 8 not listed) | – | – |
| 23 | MinC | 0.0890 | 0.17 | SG_D_MaxES_MinES | 0.0040 | -0.32 | (one of the 8 not listed) | – | – |
| 24 | SG_B_Amax | 0.0205 | 0.03 | SG_VarCAmin | 0.0032 | -0.10 | (one of the 8 not listed) | – | – |
| 25 | B_Amax_MaxCa | 0.0135 | 0.54 | MinC | 0.0007 | -0.11 | (one of the 8 not listed) | – | – |
| 26 | VarS | 0.0000 | – | B_Amax_MaxCa | 0.0001 | -0.19 | (one of the 8 not listed) | – | – |
| 27 | D_S_Uniform | 0.0000 | – | SG_B_Amax | 0.0000 | 0.15 | (one of the 8 not listed) | – | – |

Figures: `a19_xgboost_tree_SP.png`, `a19_rf_kernel_SP.png`

## HAC label (TH Fig. 5.7 (p.81))

| rank | run (xgboost_tree): feature | mean abs SHAP | direction | run (rf_kernel): feature | mean abs SHAP | direction | thesis (TH Fig. 5.7 (p.81)): feature | run rank (xgboost_tree) of the thesis feature | run rank (rf_kernel) of the thesis feature |
|---|---|---|---|---|---|---|---|---|---|
| 1 | B_Amax_MaxCa | 0.6583 | 0.73 | B_Amax_MaxCa | 0.0505 | 0.69 | CAmin | 7 | 8 |
| 2 | D_Smax_Smin | 0.4882 | 0.77 | SG_AvgCAmax | 0.0222 | 0.65 | B_Amax_MaxCa | 1 | 1 |
| 3 | AvgPSV | 0.4596 | 0.42 | AvgPSV | 0.0207 | 0.33 | AvgPSV | 3 | 3 |
| 4 | N | 0.4415 | 0.70 | D_S_Uniform | 0.0199 | 0.62 | SG_B_Amax | 25 | 27 |
| 5 | SG_AvgCAmax | 0.4264 | 0.57 | AvgC | 0.0179 | 0.68 | SG_D_MaxES_MinES | 18 | 21 |
| 6 | MaxPSa | 0.3515 | -0.37 | VarS | 0.0178 | 0.59 | D_Smax_Smin | 2 | 7 |
| 7 | CAmin | 0.3515 | -0.64 | D_Smax_Smin | 0.0175 | 0.63 | SG_AvgCAmax | 5 | 2 |
| 8 | SG_VarCAmin | 0.3070 | 0.54 | CAmin | 0.0158 | -0.61 | MaxPSa | 6 | 14 |
| 9 | SG_D_MaxAvgC_MinAvgC | 0.3055 | -0.57 | CAmax | 0.0157 | 0.59 | N | 4 | 12 |
| 10 | VarC | 0.2970 | -0.03 | SG_AvgCAmin | 0.0131 | -0.58 | SG_VarCAmin | 8 | 22 |
| 11 | SG_VarCAmax | 0.2778 | 0.19 | ES | 0.0128 | -0.41 | VarC | 10 | 18 |
| 12 | CAmax | 0.2756 | 0.66 | N | 0.0127 | 0.51 | SG_VarSAmax | 20 | 20 |
| 13 | SG_AvgCAmin | 0.2715 | -0.53 | MinPSa | 0.0114 | 0.25 | SG_AvgCAmin | 13 | 10 |
| 14 | P_lowC_highPSv | 0.2636 | -0.44 | MaxPSa | 0.0110 | -0.28 | SG_D_MaxAvgC_MinAvgC | 9 | 24 |
| 15 | SG_D_MaxVarC_MinVarC | 0.2634 | 0.29 | D_MaxC_AvgC | 0.0100 | -0.07 | SG_VarCAmax | 11 | 17 |
| 16 | P_lowPSv_highC | 0.2584 | -0.06 | SG_D_MaxVarC_MinVarC | 0.0071 | 0.22 | MinC | 22 | 26 |
| 17 | D_MaxC_AvgC | 0.1855 | 0.59 | SG_VarCAmax | 0.0068 | -0.09 | P_lowPSv_highC | 16 | 23 |
| 18 | SG_D_MaxES_MinES | 0.1847 | 0.11 | VarC | 0.0060 | 0.15 | SG_D_MaxVarC_MinVarC | 15 | 16 |
| 19 | SG_D_MaxVarS_MinVarS | 0.1523 | -0.58 | P_lowC_highPSv | 0.0060 | -0.21 | P_lowC_highPSv | 14 | 19 |
| 20 | SG_VarSAmax | 0.1483 | -0.10 | SG_VarSAmax | 0.0060 | -0.22 | (one of the 8 not listed) | – | – |
| 21 | AvgC | 0.1475 | 0.25 | SG_D_MaxES_MinES | 0.0059 | 0.27 | (one of the 8 not listed) | – | – |
| 22 | MinC | 0.1306 | 0.47 | SG_VarCAmin | 0.0049 | 0.35 | (one of the 8 not listed) | – | – |
| 23 | MinPSa | 0.0975 | 0.54 | P_lowPSv_highC | 0.0047 | -0.02 | (one of the 8 not listed) | – | – |
| 24 | ES | 0.0606 | -0.70 | SG_D_MaxAvgC_MinAvgC | 0.0047 | -0.03 | (one of the 8 not listed) | – | – |
| 25 | SG_B_Amax | 0.0076 | -0.43 | SG_D_MaxVarS_MinVarS | 0.0046 | -0.12 | (one of the 8 not listed) | – | – |
| 26 | VarS | 0.0000 | – | MinC | 0.0011 | 0.21 | (one of the 8 not listed) | – | – |
| 27 | D_S_Uniform | 0.0000 | – | SG_B_Amax | 0.0000 | -0.07 | (one of the 8 not listed) | – | – |

Figures: `a19_xgboost_tree_HAC.png`, `a19_rf_kernel_HAC.png`

## WC label (TH Fig. A.1 (p.113))

| rank | run (xgboost_tree): feature | mean abs SHAP | direction | run (rf_kernel): feature | mean abs SHAP | direction | thesis (TH Fig. A.1 (p.113)): feature | run rank (xgboost_tree) of the thesis feature | run rank (rf_kernel) of the thesis feature |
|---|---|---|---|---|---|---|---|---|---|
| 1 | D_Smax_Smin | 0.8153 | 0.90 | AvgC | 0.0338 | 0.70 | SG_B_Amax | 24 | 26 |
| 2 | AvgC | 0.7112 | 0.71 | D_S_Uniform | 0.0263 | 0.77 | SG_D_MaxES_MinES | 12 | 13 |
| 3 | MaxPSa | 0.5166 | -0.83 | D_Smax_Smin | 0.0251 | 0.76 | CAmin | 6 | 11 |
| 4 | AvgPSV | 0.4737 | 0.63 | ES | 0.0240 | -0.80 | MaxPSa | 3 | 8 |
| 5 | VarC | 0.3484 | 0.23 | VarS | 0.0233 | 0.74 | D_Smax_Smin | 1 | 3 |
| 6 | CAmin | 0.3399 | -0.66 | MinPSa | 0.0198 | 0.56 | AvgPSV | 4 | 9 |
| 7 | P_lowPSv_highC | 0.3020 | -0.24 | SG_AvgCAmax | 0.0157 | 0.71 | SG_VarSAmax | 21 | 20 |
| 8 | SG_D_MaxVarC_MinVarC | 0.2888 | -0.26 | MaxPSa | 0.0157 | -0.59 | N | 9 | 16 |
| 9 | N | 0.2811 | 0.32 | AvgPSV | 0.0144 | 0.52 | D_MaxC_AvgC | 14 | 14 |
| 10 | SG_VarCAmax | 0.2802 | 0.29 | CAmax | 0.0143 | 0.71 | SG_VarCAmin | 20 | 25 |
| 11 | SG_AvgCAmax | 0.2632 | 0.73 | CAmin | 0.0112 | -0.15 | SG_AvgCAmax | 11 | 7 |
| 12 | SG_D_MaxES_MinES | 0.2579 | 0.41 | SG_VarCAmax | 0.0083 | 0.18 | SG_VarCAmax | 10 | 12 |
| 13 | SG_D_MaxAvgC_MinAvgC | 0.2560 | -0.17 | SG_D_MaxES_MinES | 0.0070 | 0.27 | SG_AvgCAmin | 19 | 19 |
| 14 | D_MaxC_AvgC | 0.2482 | 0.72 | D_MaxC_AvgC | 0.0067 | 0.11 | SG_D_MaxVarC_MinVarC | 8 | 22 |
| 15 | P_lowC_highPSv | 0.2428 | -0.21 | VarC | 0.0066 | 0.23 | VarC | 5 | 15 |
| 16 | MinPSa | 0.2074 | 0.78 | N | 0.0063 | 0.00 | AvgC | 2 | 1 |
| 17 | CAmax | 0.2059 | 0.73 | SG_D_MaxAvgC_MinAvgC | 0.0058 | -0.14 | SG_D_MaxAvgC_MinAvgC | 13 | 17 |
| 18 | ES | 0.1902 | -0.73 | SG_D_MaxVarS_MinVarS | 0.0053 | -0.24 | P_lowPSv_highC | 7 | 21 |
| 19 | SG_AvgCAmin | 0.1854 | 0.28 | SG_AvgCAmin | 0.0050 | -0.20 | SG_D_MaxVarS_MinVarS | 23 | 18 |
| 20 | SG_VarCAmin | 0.1599 | -0.02 | SG_VarSAmax | 0.0049 | -0.15 | (one of the 8 not listed) | – | – |
| 21 | SG_VarSAmax | 0.1480 | -0.46 | P_lowPSv_highC | 0.0048 | -0.40 | (one of the 8 not listed) | – | – |
| 22 | MinC | 0.1193 | -0.37 | SG_D_MaxVarC_MinVarC | 0.0046 | -0.21 | (one of the 8 not listed) | – | – |
| 23 | SG_D_MaxVarS_MinVarS | 0.1127 | -0.30 | P_lowC_highPSv | 0.0046 | -0.24 | (one of the 8 not listed) | – | – |
| 24 | SG_B_Amax | 0.0147 | 0.45 | MinC | 0.0036 | -0.40 | (one of the 8 not listed) | – | – |
| 25 | B_Amax_MaxCa | 0.0147 | -0.50 | SG_VarCAmin | 0.0027 | -0.11 | (one of the 8 not listed) | – | – |
| 26 | VarS | 0.0000 | – | SG_B_Amax | 0.0001 | 0.29 | (one of the 8 not listed) | – | – |
| 27 | D_S_Uniform | 0.0000 | – | B_Amax_MaxCa | 0.0001 | -0.07 | (one of the 8 not listed) | – | – |

Figures: `a19_xgboost_tree_WC.png`, `a19_rf_kernel_WC.png`

## Rule baselines: recomputed on the build vs printed in the thesis

Cells: recomputed count (rate) / thesis count (rate) / difference in cases.

| block | n | SP | MR | WC | HAC | DA |
|---|---|---|---|---|---|---|
| All cases | 1209 | 869 (0.719) / 864 (0.715) / +5 | 838 (0.693) / 831 (0.687) / +7 | 831 (0.687) / 830 (0.687) / +1 | 767 (0.634) / 774 (0.640) / -7 | 368 (0.304) / 372 (0.308) / -4 |
| General | 500 | 413 (0.826) / 413 (0.826) / +0 | 385 (0.770) / 385 (0.770) / +0 | 380 (0.760) / 380 (0.760) / +0 | 338 (0.676) / 338 (0.676) / +0 | 111 (0.222) / 108 (0.216) / +3 |
| NBA | 393 | 260 (0.662) / 258 (0.656) / +2 | 256 (0.651) / 252 (0.641) / +4 | 252 (0.641) / 253 (0.644) / -1 | 230 (0.585) / 237 (0.603) / -7 | 137 (0.349) / 142 (0.361) / -5 |
| NFL | 212 | 112 (0.528) / 109 (0.514) / +3 | 114 (0.538) / 111 (0.524) / +3 | 117 (0.552) / 114 (0.538) / +3 | 131 (0.618) / 131 (0.618) / +0 | 99 (0.467) / 101 (0.476) / -2 |
| Politics | 104 | 84 (0.808) / 84 (0.808) / +0 | 83 (0.798) / 83 (0.798) / +0 | 82 (0.788) / 83 (0.798) / -1 | 68 (0.654) / 68 (0.654) / +0 | 21 (0.202) / 21 (0.202) / +0 |


## Reading notes

- RF + Kernel uses the shap default `l1_reg="num_features(10)"`: at most 10 non-zero values per case (decision SH4).
- In the direction tables, "agrees" for correlations with |rho| < 0.1 should be read as "weak".
- The thesis x-axis ranges quoted here were read by eye and may be off by up to 0.4.