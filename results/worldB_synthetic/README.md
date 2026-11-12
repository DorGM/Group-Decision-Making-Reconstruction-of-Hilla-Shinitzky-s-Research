**SYNTHETIC — not a reproduction**

# results/worldB_synthetic — SYNTHETIC — not a reproduction

Everything in this folder was computed on INVENTED responses (the synthetic World-B stand-in of plan §2.3). The MTurk data of the thesis is not public (available on request according to the 2024 article) and is not part of this repository. No number here is a result about the thesis; thesis values appear only in separate "thesis" columns and no difference between a synthetic value and a thesis value is computed.

| Folder | Step | Content |
|---|---|---|
| `b3_calibration_first_grid/` | B3 | first grid search of the invented parameters on the four rule counts of T24 (SP ended at the grid corner, mean 31.0) |
| `b3_calibration/` | B3 | the search after one grid extension (decision B3-9); wrote `config/synthetic_worldB.yaml` |
| `b3_data/` | B3 | the synthetic data in the loader layout, the check against the task design of TH Table 4.1, the checks of plan §9.3 row B3 |
| `b5_rules/` | B5 | MR / WC / SP / HAC on the 77 sub-sets over 30 partition seeds, beside T24 |
| `b6_rp/` | B6 | RP: RCR instances, soft vote, leave-one-group-out over the 77 sub-sets, the nine classification-aggregation strategies; table in the shape of TH Table 4.2; 30 partition seeds |
| `b7_ap/` | B7 | AP: ACR instances, soft vote, three-case rule (checked equal to argmax P(True)); headline table in the shape of TH Fig 5.3 (RP, AP, MR, WC, SP, HAC) and PRE against MR |
| `b8_alt/` | B8 | RP and AP with Bag-KNN and SVM; tables in the shape of TH Tables A.7 and A.8 |
| `b9_stats/` | B9 | McNemar and one-tailed proportion tests per partition; table in the shape of TH Table 5.11; plus a data-free recomputation of the printed table from the published counts |
| `b10_shap/` | B10 | SHAP of the RCR / ACR features: end-to-end run only, no comparison with the thesis |
| `b11_report/` | B11 | `results_worldB_synthetic.md`: the thesis-reported items T23–T30 and the synthetic run values in separate columns |

**Rule baselines.** The four rule counts (MR, WC, SP, HAC) are the generator's calibration targets: their agreement with the thesis-reported counts is by construction. The calibration was done on the single data set drawn with generator seed 0. With the same parameters and other generator seeds the counts move well outside ±2 (generator seeds 1–8, 10 partition seeds each: MR 25–35, WC 26–34, SP 31–37, HAC 16–21; 2 of the 8 seeds are within ±2 on all four rules). The agreement holds for generator seed 0 only: it is a property of this one data set, not of the model (decision B3-10).

**RP / AP rates.** On the shipped stand-in solvers are easy to identify: the single feature dPSv separates correct from wrong responses with an AUC of about 0.83 on partition seed 0 (ChiSq alone 0.75, ChosenMajAns alone 0.69). This follows from the invented prediction mechanism (a solver moves half of the predicted mass to the lure, the value at which the SP baseline reached its calibration value of 33), not from any tuning towards RP / AP. The synthetic RP / AP success rates reflect that invented mechanism and are NOT comparable with the thesis's 0.636 / 0.545.

**Real-data path.** The same commands take a real file through `config/worldB.yaml` (`source: real`); outputs would then go to `runs/worldB_real/`. That path is UNTESTED ON REAL DATA.

Commands, in this order: `python scripts/run_b3_synthetic.py`, `run_b5_rules.py`, `run_b6_rp.py`, `run_b7_ap.py`, `run_b8_alt.py`, `run_b9_stats.py`, `run_b10_shap.py`, `run_b11_report.py` (see `docs/VERIFIED_COMMANDS.md`). Which files of this folder were regenerated with the published code is listed in `docs/RESULT_FILES.md`.

Code: `cdm/data/worldB/`, `cdm/approaches/{worldB_instances,rp,ap}.py`, `cdm/models/worldB_classifiers.py`, `cdm/eval/worldB_*.py`, `scripts/run_b3_synthetic.py` … `run_b11_report.py`. Open questions and every invented value: `docs/IMPLEMENTATION_DECISIONS.md`, Phase B.

**SYNTHETIC — not a reproduction**
