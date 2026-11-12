# Data provenance, assumptions and synthetic generation

This repository contains **no participant data and no third-party files**. The public datasets must be downloaded from their sources. The original RP / AP data is not public and has not yet been supplied to this project.

| Part | Data | Status in this repository |
|---|---|---|
| AMP / DAP | Three public datasets, rebuilt into 1,209 binary decision cases | Used. Selection and preprocessing assumptions are listed in section 2. |
| RP / AP | Responses collected by the original authors (22 multiple-choice tasks) | **Not available.** A synthetic stand-in is used (section 3). The loader for the real format is untested. |

## 1. Public datasets for AMP / DAP: where to get them

The thesis describes a collection of 1,209 binary decision cases in four domains, taken from previously published studies.

| Domain | Source | Where to obtain | Folder expected under `data/` |
|---|---|---|---|
| General knowledge (500 true/false statements) | Martinie, Wilkening & Howe (2020), *PLOS ONE* 15(4): e0232058, supporting information S3 (analysis code and raw data) | The article's supporting-information page (file `pone.0232058.s003.zip`) | `Martinie2020_General/extracted/` |
| NFL, Politics (2018 US elections), NBA | Rutchick, Ross, Calvillo & Mesick (2020), *Cognitive Research: Principles and Implications* 5(1), Studies 1, 2 and 3 | https://osf.io/u9k72 | `Rutchick2020_NBA_NFL_Politics/extracted/` |
| NFL (alternative source named in the thesis) | Lee, Danileiko & Vi (2018), *Judgment and Decision Making* 13(4) | https://osf.io/3kjmu | `Lee2018_NFL/extracted/` |

Download each archive, unzip it into the `extracted/` folder shown, and keep the internal folder structure of the archive. The location of `data/` can be changed with the environment variable `CDM_DATA_ROOT`. Each dataset remains under the license chosen by its authors; check the source pages before redistributing.

The main build of this repository takes the NFL cases from Rutchick et al., Study 1. The thesis and the preprint list Lee et al. (2018), an NFL study, among their data sources without assigning it to a domain. The reported NFL case count (212) and the reported maximum of 129 responses per case are obtainable from the Rutchick Study 1 file (Lee et al. recruited 100 workers per week), and the Lee archive available to this project contains no game outcomes, so cases cannot be built from it alone. This is recorded as an open question.

## 2. How the 1,209 cases are built (`cdm/data/worldA/`)

The build used for all published AMP / DAP results is the scenario named `waiver_main` in `config/worldA_scenarios.yaml`. The name refers to the rule-baseline check described in [RULE_BASELINE_CHECK.md](RULE_BASELINE_CHECK.md). The sources do not fully specify how the cases were selected and preprocessed. Each rule below is therefore an assumption of this reconstruction; alternatives are available as switches and scenarios in the same file.

| Domain | Cases in source | Cases kept | Selection and preprocessing assumptions |
|---|---|---|---|
| General | 500 | 500 | All raw responses are used. Confidence of a response is p/100 when the vote is "true" and (100 − p)/100 when it is "false", where p is the stated probability. With this reading, features 1–21 of the worked example in thesis Table A.2 are obtained from item 1 of the dataset (features 18–21 under a 3 × 30 partition with the matching vote counts); features 22–27 depend on a sub-set assignment that is not known and are not obtained (implementation decision W10). |
| NBA | 465 | 393 | Games in which 0% or 100% of the responses are correct are removed, as the thesis states (17 and 52 games in the source file). Three further games are removed to reach the reported 393; they are the games whose share of correct responses is below the reported minimum. Incomplete responses are dropped. |
| NFL | 216 | 212 | Rutchick Study 1, original file. Four games whose three column headers do not name the same game are removed to reach the reported 212. Confidence is recoded to the 1–5 scale used in the revised file. Responses lacking a vote, a confidence or a prediction are dropped. |
| Politics | 110 | 104 | Races are treated as binary: votes for a third or fourth option are dropped, and two races won by such an option fall out. Four races whose prediction and agreement column headers differ are removed to reach the reported 104. This choice of four races is not determined by the sources. |

Resulting collection: 1,209 cases, in the order General, NBA, NFL, Politics and source-file order within each domain. The thesis reports 1,209 cases, with 500 General, 393 NBA and 212 NFL cases (Politics = 104 by subtraction). Descriptive statistics of the rebuilt collection are written to `build/worldA/waiver_main/manifest.json` when the build is frozen (not stored in `results/`). Every 1,209-case build gives a response-count SD of 35.6 and a correct-rate SD of 23.7; the thesis reports 35.5 and 23.8 (implementation decision W1).

Rows of the source spreadsheets that hold game metadata rather than participant responses (for example the row recording the game result, which supplies the correct answer) are not loaded as responses.

Because several selection rules are assumptions, the rebuilt collection may not be identical to the one used in the original research. The rule-aggregator baselines computed on it agree exactly with the reported counts for MR, WC, HAC and SP in the General domain; DA in General and the rules in the other domains differ by 1 to 7 cases.

## 3. RP / AP: synthetic stand-in data

> **SYNTHETIC — not a reproduction.** Everything in `results/worldB_synthetic/` was computed on generated data.

### 3.1 Why

The RP / AP study used 22 multiple-choice tasks answered by participants recruited by the original authors. That data is not public. Dor Meir is awaiting it from Hilla Shinitzky. Until it is received, a generated stand-in lets the RP / AP code be written, tested and run end to end.

### 3.2 What is generated (`cdm/data/worldB/synthetic.py`, `scripts/run_b3_synthetic.py`)

- **Taken from the published description (thesis Table 4.1):** for each of the 22 tasks, the number of answer options, the number of responses and the share of correct responses. This gives 2,490 responses in total and, with the partition rule of the thesis (sub-sets of about 30 responses per task), 77 sub-sets. For four tasks the printed share is compatible with two or three different integer counts of correct responses; one is chosen and the alternatives are listed in the implementation decisions.
- **Generated:** for every response a vote, a confidence in [0, 1] and a predicted distribution of the others' votes that sums to 1.

### 3.3 The response model and its assumptions

All of the following is introduced by this reconstruction. None of it is taken from the original data.

| Element | Assumption |
|---|---|
| Wrong answers | One wrong option per task is designated a "lure" and receives a task-specific share of the wrong votes; the remaining wrong votes are spread over the other options. Which option is correct and which is the lure is drawn at random. |
| Confidence | Drawn from a Beta distribution whose mean rises with the popularity of the chosen answer. |
| Predictions of others | Centred on the actual vote shares, shifted towards the respondent's own answer; respondents who answered correctly additionally shift weight towards the lure. A small share of respondents predict 100% for their own answer. |
| Randomness | One generator seed (0). The shipped dataset is the single dataset drawn with that seed. |

The directions of the response model were motivated by the thesis's description of the tasks as misleading (TH p.52), by Koriat (2008) for confidence, by Prelec et al. (2017) for predictions, and by TH Table A.4 (a response with PSv = 1 exists; predictions sum to 1). These motivate the form only; all values are invented.

Parameter values, their search grids and their tags (`INVENTED` or `INVENTED-CALIBRATED`) are in `config/synthetic_worldB.yaml`.

### 3.4 Calibration

Parameters tagged `INVENTED-CALIBRATED` were chosen by a seeded grid search whose only targets were the four published rule-aggregator success counts on the 77 sub-sets (MR 34, WC 34, SP 33, HAC 22 of 77; thesis Figure 5.3). The search proceeded in blocks (vote parameters against MR, confidence parameters against WC and HAC, prediction parameters against SP). One grid was extended once after the first search ended at its boundary; both searches are stored in `results/worldB_synthetic/`.

Consequences that matter when reading any synthetic result:

- Agreement of the synthetic rule baselines with the published counts is **by construction**.
- The agreement holds for the shipped dataset (generator seed 0). With the same parameters and other generator seeds the rule counts move outside the calibration tolerance.
- RP / AP outcomes were **not** calibration targets, and no parameter was adjusted in response to them.
- In the stand-in, responses with the correct answer are comparatively easy to tell apart from their features (a single feature, `dPSv`, reached an AUC of about 0.83 on one partition; computed once, not a stored output). Synthetic RP / AP success rates reflect the invented response model. They are not comparable with the values in the thesis or the 2024 article, and no difference, improvement or gap between the two is computed anywhere in this repository.

### 3.5 Markers

Every file generated from the stand-in carries the marker `SYNTHETIC — not a reproduction` (first line of text files, first key of JSON files, and in the data object). The loader refuses data whose markers are inconsistent.

## 4. The original RP / AP data: planned integration

`cdm/data/worldB/loader.py` reads a real data file through `config/worldB.yaml` (`source: real`). This path is **untested on real data**: the file format, the confidence scale and the prediction scale are assumptions (entries L1–L12 in [IMPLEMENTATION_DECISIONS.md](IMPLEMENTATION_DECISIONS.md)), and the loader refuses to run until the scales are stated. Once the original data is received, the plan is to confirm the format, adapt the loader, rerun the RP / AP evaluation with the same commands, and write the results to a separate `worldB_real` folder so that synthetic and real outputs can never be mixed.
