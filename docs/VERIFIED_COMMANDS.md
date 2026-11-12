# Verified commands

The commands below were run from a copy of this repository on Linux with
Python 3.13 and the package versions of `requirements.txt`, on 2026-11-12.
They were **not** run on Windows or macOS. `scripts/run_phaseA_queue.sh` is a
bash script and needs a POSIX shell.

All commands are run from the repository root.

## Set-up

```
python -m pip install -r requirements.txt
```

Data: the raw files of the three public sources are not part of the
repository. Put them under `data/` in the layout of `docs/DATA.md`, or point the
environment variable `CDM_DATA_ROOT` at the folder that holds them:

```
export CDM_DATA_ROOT=/path/to/Data      # optional; default is ./data
```

Where things are written (all three folders are in `.gitignore`):

| Folder | Content | Override |
|---|---|---|
| `build/` | frozen World-A builds | env `CDM_BUILD_ROOT`, or `--out` / `--build-root` |
| `runs/` | outputs of the driver scripts, with checkpoints and logs | `--runs-root` |
| `results/` | the published result files; a driver writes here only on request (`--results-root results` for the Phase-A drivers, `--copy-results` for the Phase-B drivers) | — |

## Tests

```
python -m pytest -q
```

| Situation | Result |
|---|---|
| raw data available (`CDM_DATA_ROOT` set) and the three builds `waiver_main`, `waiver_politics_alt`, `h1209_nba_games` frozen under the build root | 1081 passed, 54 xfailed |
| no raw data (`CDM_DATA_ROOT` not set, `data/` empty) | 866 passed, 269 skipped |

The 54 strict expected failures are the rule-baseline counts that differ from
the thesis-reported counts (`docs/RULE_BASELINE_CHECK.md`); each is encoded with
its measured value, so a change of any count shows up as a test failure. The
tests that need a frozen build are skipped when the build is absent. To freeze
the three builds used by the tests:

```
python -m cdm.data.worldA.build_worldA --scenario waiver_main --scenario waiver_politics_alt --scenario h1209_nba_games
```

## Phase A (AMP / DAP on the public data)

Freeze the build (about 10 s; writes `build/worldA/waiver_main/`):

```
python -m cdm.data.worldA.build_worldA --scenario waiver_main
```

Verified: 500 / 393 / 212 / 104 cases; SHA-256 of `worldA_cases.jsonl` =
`c237695153557e02862adf8f823098f849831a98ed6e737595b9d6950794635c`, the value
recorded in `config/gate_status.yaml`.

Pipeline checks (neither is a result; both mark their outputs "NOT A RESULT"):

```
python scripts/run_a13_loo.py --smoke 60                              # synthetic cases, no data needed; about 10 s
python scripts/run_a13_loo.py --build waiver_main --limit 40 --n-jobs 2   # 40 cases of the real build; about 30 s
```

Outputs: `runs/_smoke/a13_seed0/` and `runs/waiver_main/a13_seed0_limit40/`.

Full runs. The commands marked "re-run" were executed again in full with the
code of this repository and gave the stored numbers (`docs/RESULT_FILES.md`);
the others are the commands of the stored runs and were not executed again in
full for this release (their code path was exercised by the tests and by the
pipeline checks above).

```
python scripts/run_a13_loo.py --build waiver_main --seed 0 --n-jobs 4      # also seeds 1-4; about 12 min each; not re-run
python scripts/run_a14_nested.py --build waiver_main --n-jobs 4 --approaches DAP,AMP   # about 3 h; not re-run
python scripts/run_a15_tables.py --build waiver_main                       # re-run; reads the A13 / A14 per-case files; seconds
python scripts/run_a16_ablation.py --build waiver_main --n-jobs 4          # about 1.5 h; not re-run
python scripts/run_a17_metalearn.py --build waiver_main                    # re-run; about 10 s
python scripts/run_a18_bagging.py --build waiver_main --n-jobs 4 --undefined-camin impute_subgroup_avgc     # re-run; about 2-4 min
python scripts/run_a18_bagging.py --build waiver_main --n-jobs 4 --undefined-camin drop_instance --out-dir runs/waiver_main/a18_drop_instance     # re-run
python scripts/run_a18_bagging.py --build waiver_main --n-jobs 4 --undefined-camin impute_subgroup_avgc --size-formula b_cochran_Z2_S_1_minus_S_over_0.01 --out-dir runs/waiver_main/a18_cochran_impute   # re-run
python scripts/run_a19_shap.py --build waiver_main --n-jobs 3              # re-run; about 20 min
```

`run_a15_tables.py` and `run_a18_bagging.py` read the outputs of A13 / A14 from
`runs/waiver_main/` (`a13_seed<k>/`, `a14/results/`). To use the stored result
files instead of new runs, copy `results/waiver_main/a13_seed*` to
`runs/waiver_main/` and `results/waiver_main/a14/` to
`runs/waiver_main/a14/results/`.

## Phase B (RP / AP) — SYNTHETIC stand-in, not a reproduction

No data file is needed: the data are generated from
`config/synthetic_worldB.yaml`. The path for a real file
(`config/worldB.yaml`, `source: real`) is untested on real data.

Pipeline check on two partition seeds (outputs go to `<step>_seeds2/` folders):

```
python scripts/run_b3_synthetic.py
python scripts/run_b5_rules.py
python scripts/run_b6_rp.py --seeds 2
python scripts/run_b7_ap.py --seeds 2
python scripts/run_b8_alt.py --seeds 2
python scripts/run_b9_stats.py --seeds 2
python scripts/run_b10_shap.py --seeds 2
python scripts/run_b11_report.py --seeds 2
```

Verified: every step ended without error; `run_b3_synthetic.py` wrote
`worldB_tasks.csv`, `worldB_responses.csv` and `b3_table_4_1_check.csv`
byte-identical to the files in `results/worldB_synthetic/b3_data/` (SHA-256 of
the tasks file `ec0b572bd4dd17d5…`, of the responses file `293a4725c5bd358c…`).

Full run (30 partition seeds; the same commands without `--seeds 2`). Executed
in full with the code of this repository; see `docs/RESULT_FILES.md` for the
comparison with the stored files:

```
python scripts/run_b3_synthetic.py
python scripts/run_b5_rules.py
python scripts/run_b6_rp.py
python scripts/run_b7_ap.py
python scripts/run_b8_alt.py
python scripts/run_b9_stats.py
python scripts/run_b10_shap.py
python scripts/run_b11_report.py
```

Not executed again: `python scripts/run_b3_synthetic.py --calibrate` (the grid
search that wrote `config/synthetic_worldB.yaml`; it overwrites that file).
