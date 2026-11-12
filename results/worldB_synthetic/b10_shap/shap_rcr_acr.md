**SYNTHETIC — not a reproduction**

### SHAP rank orders of the RCR and ACR features (TH Figs 5.8, 5.9; T30) — SYNTHETIC — not a reproduction

**End-to-end smoke run on the synthetic stand-in.** Plan row B10 puts the acceptance of this step on real data only. The two orders below are printed side by side so that the output format can be checked; no agreement measure is computed and none should be read off: the synthetic responses were invented, so which of their features carry information is a property of the generator, not of the thesis data. Where a row of the two orders happens to coincide, that is not evidence either: the generator's prediction model was built on the SP mechanism (plan §2.3), which makes the predicted-support features informative by design.

Models trained on all instances of one partition (partition seed 0); importance = mean |SHAP|. KernelSHAP explains a seeded sample of 150 instances (background 50).

#### RCR features (RP); thesis order: TH Fig 5.8 p.82 (MCF Fig 2) — SYNTHETIC — not a reproduction

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

#### ACR features (AP); thesis order: TH Fig 5.9 p.83 (MCF Fig 3) — SYNTHETIC — not a reproduction

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

**SYNTHETIC — not a reproduction**
