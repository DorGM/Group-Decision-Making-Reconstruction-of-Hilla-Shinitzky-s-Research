# Result files

`results/` holds the stored outputs of the runs described in
`docs/RESULTS_AMP_DAP.md` (Phase A, build `waiver_main`) and
`docs/RESULTS_RP_AP_SYNTHETIC.md` (Phase B, SYNTHETIC stand-in — not a
reproduction). This page lists every result folder, the command that produces
it, and how the stored files relate to the code of this repository.

## How the stored files were brought in line with the published code

The runs were first made with an earlier internal state of the code. For
publication only wording was changed in the code (comments, messages, labels,
key names, default paths); no algorithm, number or calibrated value was changed.
Each result folder was then treated in one of three ways:

- **regenerated** — the command was run again in full with the code of this
  repository (Linux, Python 3.13, 2026-11-12) and the stored text / JSON / CSV
  files were replaced by the new outputs. The numbers of the new outputs were
  compared with the original stored files (last column). Figures (`.png`) were
  not replaced: the stored figures are the original ones.
- **re-worded** — the run was not repeated (long fits). The same wording
  changes that were made in the generating code were applied to the stored text
  and to the metadata of the stored JSON files; per-case CSV files were not
  touched. The last column states the numeric comparison with the original files.
- **unchanged** — every file is byte-identical to the original stored file.

In all folders, machine-specific absolute paths in the run metadata (`argv`,
`build_dir`) were replaced by repository-relative ones (`runs`, `build/...`),
and the process id was removed from the JSON headers.

Numeric comparison: for JSON files every numeric and boolean value outside the
timing / timestamp fields is compared in document order; for Markdown and text
files every number in table and data lines is compared; CSV files are compared
byte for byte. A "regenerated" row with identical numbers therefore means that
the published code reproduced the stored numbers of that folder exactly.

| Folder | Command (from the repository root) | Status | Files | Comparison with the original stored files |
|---|---|---|---|---|
| `results/waiver_main/a13_seed0/` | `scripts/run_a13_loo.py --build waiver_main --seed 0 --n-jobs 4` | re-worded | 4 | 2 × byte-identical to the original file; 1 × JSON: every numeric value identical; 1 × text re-worded, every number in table / data lines identical |
| `results/waiver_main/a13_seed1/` | `scripts/run_a13_loo.py --build waiver_main --seed 1 --n-jobs 4` | re-worded | 4 | 2 × byte-identical to the original file; 1 × JSON: every numeric value identical; 1 × text re-worded, every number in table / data lines identical |
| `results/waiver_main/a13_seed2/` | `scripts/run_a13_loo.py --build waiver_main --seed 2 --n-jobs 4` | re-worded | 4 | 2 × byte-identical to the original file; 1 × JSON: every numeric value identical; 1 × text re-worded, every number in table / data lines identical |
| `results/waiver_main/a13_seed3/` | `scripts/run_a13_loo.py --build waiver_main --seed 3 --n-jobs 4` | re-worded | 4 | 2 × byte-identical to the original file; 1 × JSON: every numeric value identical; 1 × text re-worded, every number in table / data lines identical |
| `results/waiver_main/a13_seed4/` | `scripts/run_a13_loo.py --build waiver_main --seed 4 --n-jobs 4` | re-worded | 4 | 2 × byte-identical to the original file; 1 × JSON: every numeric value identical; 1 × text re-worded, every number in table / data lines identical |
| `results/waiver_main/a14/` | `scripts/run_a14_nested.py --build waiver_main --n-jobs 4 --approaches DAP,AMP` | re-worded | 10 | 6 × byte-identical to the original file; 2 × JSON: every numeric value identical; 2 × text re-worded, every number in table / data lines identical |
| `results/waiver_main/a15/` | `scripts/run_a15_tables.py --build waiver_main` (no model; reads the A13 / A14 per-case files) | regenerated | 3 | 1 × byte-identical to the original file; 1 × JSON: every numeric value identical; 1 × text re-worded, every number in table / data lines identical |
| `results/waiver_main/a16/` | `scripts/run_a16_ablation.py --build waiver_main --n-jobs 4` | re-worded | 26 | 1 × text re-worded, every number in table / data lines identical; 24 × byte-identical to the original file; 1 × JSON: every numeric value identical |
| `results/waiver_main/a17/` | `scripts/run_a17_metalearn.py --build waiver_main` | regenerated | 9 | 7 × byte-identical to the original file; 1 × text re-worded, every number in table / data lines identical; 1 × JSON: every numeric value identical |
| `results/waiver_main/a18_impute_subgroup_avgc/` | `scripts/run_a18_bagging.py --build waiver_main --n-jobs 4 --undefined-camin impute_subgroup_avgc` | regenerated | 12 | 10 × byte-identical to the original file; 1 × text re-worded, every number in table / data lines identical; 1 × JSON: every numeric value identical |
| `results/waiver_main/a18_drop_instance/` | `scripts/run_a18_bagging.py --build waiver_main --n-jobs 4 --undefined-camin drop_instance --out-dir runs/waiver_main/a18_drop_instance` | regenerated | 12 | 10 × byte-identical to the original file; 1 × text re-worded, every number in table / data lines identical; 1 × JSON: every numeric value identical |
| `results/waiver_main/a18_cochran_impute/` | `scripts/run_a18_bagging.py --build waiver_main --n-jobs 4 --undefined-camin impute_subgroup_avgc --size-formula b_cochran_Z2_S_1_minus_S_over_0.01 --out-dir runs/waiver_main/a18_cochran_impute` | regenerated | 12 | 10 × byte-identical to the original file; 1 × text re-worded, every number in table / data lines identical; 1 × JSON: every numeric value identical |
| `results/waiver_main/a19/` | `scripts/run_a19_shap.py --build waiver_main --n-jobs 3` | regenerated | 13 | 11 × byte-identical to the original file; 1 × text re-worded, every number in table / data lines identical; 1 × JSON: every numeric value identical |
| `results/worldB_synthetic/b3_calibration_first_grid/` | `scripts/run_b3_synthetic.py --calibrate --first-grid` | re-worded | 4 | 3 × byte-identical to the original file; 1 × JSON: every numeric value identical |
| `results/worldB_synthetic/b3_calibration/` | `scripts/run_b3_synthetic.py --calibrate` | re-worded | 4 | 3 × byte-identical to the original file; 1 × JSON: every numeric value identical |
| `results/worldB_synthetic/b3_data/` | `scripts/run_b3_synthetic.py` | regenerated | 6 | 2 × JSON: every numeric value identical; 1 × text re-worded, every number in table / data lines identical; 3 × byte-identical to the original file |
| `results/worldB_synthetic/b5_rules/` | `scripts/run_b5_rules.py` | regenerated | 3 | 1 × byte-identical to the original file; 1 × JSON: every numeric value identical; 1 × text re-worded, every number in table / data lines identical |
| `results/worldB_synthetic/b6_rp/` | `scripts/run_b6_rp.py` | regenerated | 4 | 2 × JSON: every numeric value identical; 1 × byte-identical to the original file; 1 × text re-worded, every number in table / data lines identical |
| `results/worldB_synthetic/b7_ap/` | `scripts/run_b7_ap.py` | regenerated | 5 | 2 × JSON: every numeric value identical; 2 × byte-identical to the original file; 1 × text re-worded, every number in table / data lines identical |
| `results/worldB_synthetic/b8_alt/` | `scripts/run_b8_alt.py` | regenerated | 9 | 3 × byte-identical to the original file; 5 × JSON: every numeric value identical; 1 × text re-worded, every number in table / data lines identical |
| `results/worldB_synthetic/b9_stats/` | `scripts/run_b9_stats.py` | regenerated | 3 | 1 × byte-identical to the original file; 1 × JSON: every numeric value identical; 1 × text re-worded, every number in table / data lines identical |
| `results/worldB_synthetic/b10_shap/` | `scripts/run_b10_shap.py` | regenerated | 8 | 6 × byte-identical to the original file; 1 × text re-worded, every number in table / data lines identical; 1 × JSON: every numeric value identical |
| `results/worldB_synthetic/b11_report/` | `scripts/run_b11_report.py` (no model; reads the outputs of B5-B10) | regenerated | 2 | 1 × JSON: every numeric value identical; 1 × text re-worded, every number in table / data lines identical |

`results/worldB_synthetic/README.md` was rewritten for this release (it merges
two earlier notes).

## Notes

- **`code_fingerprint` in the JSON headers.** The Phase-A summaries carry a hash
  of the model code (`cdm.eval.loo.code_fingerprint`). The hash covers the
  source text including comments, so the wording changes alter it. In the
  *re-worded* folders the stored value is the fingerprint of the code state
  that made the run and does not equal the fingerprint of the published code;
  in the *regenerated* folders it is the fingerprint of the published code.
- **Timestamps.** Regenerated files carry the time of the new run
  (2026-11-12); re-worded files keep the time of the original run (2026-11-11).
- **Scenario status string.** The `manifest_status` field of the Phase-A
  headers quotes the scenario status of `config/worldA_scenarios.yaml`; in the
  re-worded folders it was replaced by the current wording of that status.
- **Per-case CSV files** (`*_cases.csv`) and the per-seed / per-row tables are
  byte-identical to the original files in every folder.

## Files that were removed

Not part of the release (37 files):

- run logs (`*.log`) and `status.json` progress files (machine paths, process ids);
- `a14/*_partial.json` (intermediate state of the finished A14 run);
- `b6_rp/rp_predictions.csv` (5.2 MB), `b8_alt/rp_predictions_svm.csv` (5.3 MB),
  `b8_alt/rp_predictions_bag_knn.csv` (3.2 MB): per-instance out-of-group
  predictions. No later script reads them (B9 and B11 read the `*_outcomes.json`
  and `*_summary.json` files). They are written again by `run_b6_rp.py` and
  `run_b8_alt.py`; the full re-run of 2026-11-12 reproduced the three original
  files byte for byte;
- `a19/a19_shap_values.npz` (1.3 MB): the raw SHAP arrays. No later script reads
  the file; `run_a19_shap.py` writes it again. The per-feature summary is in
  `a19_mean_abs_shap.csv`.

The smaller per-instance files `b7_ap/ap_predictions.csv`,
`b8_alt/ap_predictions_*.csv` and `b10_shap/shap_values.npz` are kept. No file
in the repository is larger than 1 MB.

## Reading the folders

- `results/waiver_main/` — Phase A. Every table shows the value of this
  implementation and the thesis-reported value in separate columns. Under the
  documented reconstruction configuration several values differ from the
  thesis-reported values; the present reconstruction does not establish the
  reason (`docs/RESULTS_AMP_DAP.md`, `docs/RULE_BASELINE_CHECK.md`).
- `results/worldB_synthetic/` — Phase B. SYNTHETIC — not a reproduction. Every
  file carries that marker. The agreement of the four rule baselines with the
  thesis-reported counts is by construction (they are the calibration values
  of the generator); no difference between a synthetic value and a
  thesis-reported value is computed; the path for a real data file is untested.
