# Reconstruction plan

Reconstruction of the pipeline of H. Shinitzky's doctoral thesis, "Optimization of
Collective Decision-Making using Data-Science Methods and Cognitive Features"
(Ben-Gurion University of the Negev; supervisor: Prof. Y. Shahar).

**What this document is.** The plan that was written from the published sources
before any code was written and before any data file was opened. It fixes the
identifiers that the code, the tests and the other documents refer to:

- `plan §x` — a section of this document;
- `T1`–`T30` — thesis-reported values (§1.1);
- `D0`–`D6` — data steps (§2.1.1); `D1`–`D21` in §7 — differences between the sources;
- `OQ1`–`OQ44` — open questions with their defaults (§8; switches in `config/defaults.yaml`);
- `A0`–`A20`, `B1`–`B11` — implementation steps (§9.3);
- `AM1`–`AM19` — amendments made during implementation (Appendix A at the end).

**How to read it.** The text states what was planned. Where implementation or the
data led to a change, the change is an amendment in Appendix A, and the amendment
prevails over the body text. Decisions taken below the level of the plan are in
`docs/IMPLEMENTATION_DECISIONS.md`. Statements about the sources are source notes:
they record what is printed and which reading this implementation uses.
"Reproduce" in this plan names the aim of a step; what was obtained is reported in
the result documents, not here.

## 0. Source abbreviations and citation convention

| Abbrev. | Source |
|---|---|
| **TH** | PhD thesis (167 pp.). Cited by *printed* page number, e.g. TH §3.2.1 p.27. PDF page ≈ printed page + 21 |
| **MLA** | Shinitzky, Shahar, Parpara, Ezrets, Klein. "Meta-Learning Approaches for a One-Shot Collective-Decision Aggregation". arXiv:2204.01721v1 (3 Apr 2022, only version). AMP/DAP |
| **MCF** | Shinitzky, Avraham, Vaisman, Tsizer, Leedon, Shahar. "Exploiting Meta-cognitive Features for a ML-Based One-Shot Group-Decision Aggregation". *Group Decision and Negotiation* 33:87–111 (2024). RP/AP |
| **MCF-arXiv** | arXiv:2201.08247, the preprint of MCF (thesis ref [106]) |

Tags used below:
- **[EVIDENCE]**: stated in the papers.
- **[INFERRED]**: derived in this plan from the published numbers. The check is shown.
- **[OPEN-Qn]**: unspecified. Section 8 gives options and a recommended default.
- **[INVENTED]** / **[INVENTED-CALIBRATED]**: used only for the synthetic World-B stand-in (§2.3). No paper support; never used for World A.

---

## 1. Scope and thesis-reported values

Scope: four ML aggregation approaches (AMP, DAP, RP, AP) and three feature representations (CDMCR, RCR, ACR). Also in scope: the rule-based aggregators MR, WC, HAC, SP, and the novel DA, plus every experiment in TH Ch.4–5 and Appendix A.

Two data worlds, which must stay separate:
- **World A (AMP/DAP):** public binary datasets, 1209 cases (TH §4.1.1).
- **World B (RP/AP):** Shinitzky's own multiple-choice MTurk data, 22 tasks, 77 sub-sets (TH §4.2.1). This data is **not public**.

### 1.1 Headline numbers (TH Ch.5 unless noted)

| ID | Quantity | Value | Source |
|---|---|---|---|
| T1 | AMP success rate (10-fold nested CV, accumulated test) | 0.773 (=935/1209 [INFERRED]; independent closure via Table 5.4: 287+200+201+93+154 = 935) | TH Fig 5.1 p.64; abstract p.vi; §6.1 p.86 |
| T2 | DAP success rate (same protocol) | 0.775 (=937/1209) | TH Fig 5.1 |
| T3 | SP / MR / WC / HAC / DA uniform success rates | 0.715 / 0.687 / 0.687 / 0.640 / 0.308 (=864/831/**830**/774/372 of 1209; WC=830 back-solved from Table 5.3) | TH Fig 5.1; Table 5.4 row 2; Table 5.3 |
| T3b | Per-domain rule counts, General/NBA/NFL/Politics (n=500/393/212/104) | SP 413/258/109/84; MR 385/252/111/83; WC 380/253/114/83; HAC 338/237/131/68; DA 108/142/101/21. Rates: SP .826/.656/.514/.808; MR .770/.641/.524/.798; WC .760/.644/.538/.798; HAC .676/.603/.618/.654; DA .216/.361/.476/.202. Column sums equal T3 exactly | TH Table 5.5 p.69 (row 2 of each block) = Table A.10 p.115 |
| T4 | PRE of AMP / DAP vs best rule (SP) | 0.204 / 0.211 | TH abstract; §6.1. Check: (0.773−0.715)/0.285=0.2035 ✓; (0.775−0.715)/0.285=0.2105 ✓. **Computed from rounded rates.** From counts, 935/864 gives .2058. This holds only for the headline PRE; the PRE rows of Tables 5.4/5.5/A.10 use exact counts |
| T5 | Per-fold LOO train scores of 9 AMP techniques + test score | Table 5.1 (10×10 numbers) | TH Table 5.1 p.63 = MLA Table 5 (MLA adds BNB rows) |
| T6 | Per-fold LOO train scores of 3 DAP techniques + test | Table 5.2 | TH Table 5.2 p.63 = MLA Table 6 |
| T7 | One-tailed proportion test p-values (AMP/DAP vs each rule) | SP 4.69e-4/3.30e-4; MR 9.40e-7/5.80e-7; WC 7.57e-7/4.65e-7 (counts 830); HAC 3.19e-13/1.58e-13; DA 0/0 | TH Table 5.3 p.65 |
| T8 | AMP effectiveness: P(selected), P(success), P(success given selected), PRE, p | SP .289/.715/.822/.377/4.2e-6; MR .202/.687/.820/.423/4.1e-6; WC .199/.687/.834/.471/4.0e-7; HAC .109/.640/.705/.179/.062; DA .201/.308/.634/.471/0. Counts [INFERRED, unique] selected/correct: SP 349/287, MR 244/200, WC 241/201, HAC 132/93, DA 243/154 (Σ 1209 / 935). PRE from exact counts: .3775/.4232/.4705/.1788/.4710 ✓ (rounded rates would give .375/.425/.470/.181) | TH Table 5.4 p.66 |
| T9 | Coverage: share of cases solved by ≥1 of {MR,WC,HAC,SP} → with DA | 83.3% → 100%; 202/1209 solved only by DA. Derived: union of 4 rules = 1007; DA ∩ (≥1 rule correct) = 372−202 = 170. Fig 5.2 shows no region counts | TH Fig 5.2 p.67, §6.1 p.87, §6.2.1.3 |
| T10 | AMP success by domain (10-fold) | General .824, NBA .919, NFL .472, Politics .596 (= 412/361/100/62, Σ 935) | TH §5.1.0.1 p.68 |
| T11 | AMP effectiveness by domain (10-fold) | Table 5.5 (4 domains × 5 methods × 4 rows). Counts [INFERRED, unique] selected/correct in the order SP, MR, WC, HAC, DA: General 230/195, 119/97, 82/71, 50/42, 19/7; NBA 79/67, 78/76, 102/94, 19/13, 115/111; NFL 24/12, 28/10, 49/28, 37/22, 74/28; Politics 16/13, 19/17, 8/8, 26/16, 35/8. They sum to the T8 counts, and all 20 printed PRE values are reproduced from them | TH Table 5.5 p.69 |
| T12 | AMP/DAP success by domain (LOO) | General .832/.852; NBA .921/.908; NFL .552/.580; Politics .769/.731. Counts: AMP 416/362/117/80 = **975**; DAP 426/357/123/76 = **982**. So Tables A.9/A.10 come from the same LOO run as MLA Figs 1–3 (T19–T21) | TH Table A.9 p.115 |
| T13 | AMP effectiveness by domain (LOO) | Table A.10. Counts [INFERRED, unique] selected/correct in the order SP, MR, WC, HAC, DA: General 261/221, 83/72, 83/70, 65/50, 8/3; NBA 91/82, 107/100, 54/49, 21/16, 120/115; NFL 41/25, 45/22, 30/16, 73/46, 23/8; Politics 29/23, 33/28, 18/14, 21/15, 3/0 | TH Table A.10 p.115 |
| T14 | AMP ablations (LOO, BR+RF) | 11 rows: .806 (full), .800, .816, .795, .808, .797, .791, .754, .806, .799, .763 | TH Table 5.6 p.72 = MLA Table 3 |
| T15 | DAP ablations (LOO, RF) | .812, .812, .825, .803, .804, .805, .806, .813, .811, .815, .798 | TH Table 5.7 = MLA Table 4 |
| T16 | Classic meta-learning (decision tree on f_AG only) | .713, .708, .695, .687, .713 | TH Table 5.8 p.73 |
| T17 | Bagging AMP (F=10,5,3,2 × 10 folds) | Table 5.9 | TH Table 5.9 p.74 |
| T18 | Bagging DAP | Table 5.10 | TH Table 5.10 p.74 |
| T19 | MLA headline (single LOO over all 1209 with BR+RF / RF) | AMP 0.806 (=**975**/1209), DAP 0.812 (=**982**/1209); counts back-solved exactly from MLA Table 7 proportion column | MLA Fig 1 p.10 (= TH Table 5.6/5.7 row 1); MLA Table 7 p.17 |
| T20 | MLA AMP selection distribution | SP 35%, MR 22%, WC 15%, HAC 15%, DA 13%. Counts from Table A.10: **422/268/185/180/154** (34.9/22.2/15.3/14.9/12.7%) | MLA Fig 3 p.12; TH Table A.10 |
| T21 | MLA P(success given selected) / p | SP .832/5.0e-8; MR .828/3.2e-7; WC .805/2.5e-4; HAC .706/3.4e-2; DA .818/0. Correct-when-selected counts from Table A.10: **351/222/149/127/126** (Σ 975); these reproduce all five rates and the four non-zero p-values exactly (one-sample z, §5.8 item 3) | MLA Fig 2 p.11; TH Table A.10 |
| T22 | MLA McNemar + proportion p-values | MLA Table 7 p.17. Discordant pairs (approach-only correct, rule-only correct) [INFERRED, unique at 3 s.f. under χ² with continuity correction]: AMP vs SP (159,48), MR (201,57), WC (195,50), HAC (279,78), DA (782,179); DAP vs SP (189,71), MR (225,74), WC (226,74), HAC (299,91), DA (763,153) | MLA |
| T23 | RP / AP success (LOGO over 77 sub-sets) | 0.636 (=49/77) / 0.545 (=42/77) | TH Fig 5.3 p.76; MCF §5 p.104–105 |
| T24 | Rules on the 77 sub-sets: MR / WC / SP / HAC | 0.442 (34/77) / 0.442 / 0.429 (33/77) / 0.286 (22/77) | TH Fig 5.3 |
| T25 | PRE RP / AP vs MR | 0.348 / 0.185. Check: (0.636−0.442)/0.558=0.3477 ✓; 0.1846 ✓ (from rounded rates; counts give .3488) | TH §6.1; MCF p.105 |
| T26 | RP 9 classification-aggregation strategies | maxp .58, avgp .61, wm .47, maj+avgp .62, maj+maxp .60, maj+wm .57, prop+avgp .64, prop+maxp .60, prop+wm .57; tie % 0/0/0/16/17/16/18/16/17 | TH Table 4.2 p.57 = MCF Table 4 |
| T27 | RP with Bag-KNN and SVM (9 strategies each) | TH Table A.7 p.114 | TH only |
| T28 | AP with Bag-KNN / SVM / Soft-vote | .44 / .43 / .55 | TH Table A.8 |
| T29 | RP/AP statistical tests (McNemar, one-tailed proportion) | TH Table 5.11 = MCF Table 5 | TH p.76 |
| T30 | SHAP rank orders: CDMCR per label (MR, DA, SP, HAC, WC), RCR, ACR | Top RCR: dPSv, PSvD_OG, PSvD, ConfD_IG, PSv…; top ACR: dPS_OG, AvgConf, AvgPS_OG, AvgPS_IG, dPS…; top CDMCR-MR: SG_B_Amax, SG_D_MaxES_MinES, MaxPSa, AvgPSv, D_Smax_Smin, CAmin… | TH Figs 5.4–5.9, A.1; MCF Figs 2–3 |
| T31 | Example feature values (unit-test oracles) | TH Tables A.2, A.4, A.6 | TH pp.110–113 |

Reproduction tolerance (proposal): exact integer counts are not achievable, because the original random partitions, seeds, data-cleaning order, and library versions are unknown.
- Rule-based rates T3/T3b/T9/T24 are deterministic given the data. **Aim: exact** (±1 case per domain for tie conventions). World-B rule rates (T24) also depend on the random 77-subset partition (OQ25).
- ML results: within ±0.02 absolute success rate, and the same qualitative conclusions (ordering, significance).

---

## 2. Data

### 2.1 World A — public binary datasets (AMP/DAP)

**What the papers say** [EVIDENCE]
- Data comes from refs [25]–[27]: Martinie et al. 2020; Rutchick et al. 2020; Lee, Danileiko & Vi 2018 (TH §4.1.1 p.43; MLA §5.1).
- Content: binary prediction problems (NBA, NFL, US elections) and general-knowledge science true/false items (TH p.43).
- Cases with 0% or 100% correct-solution rate were excluded.
- Final set: **1209 cases** (709 prediction + 500 general), mean 70 responses per case (SD 35.5, min 10, max 129), mean 62% correct (SD 23.8%, min 2.7%, max 98.9%) (TH p.43).
- Domain counts: General 500, NBA 393, NFL 212; Politics = 1209−500−393−212 = **104** [INFERRED] (TH §5.1.0.1 p.68).
- Responder sources: "psychology course students, participants … via online crowdsourcing platforms and online forums, and students" (TH p.43). This matches Rutchick S1 (psychology course), Rutchick S3 (r/NBA, r/sportsbook, sport-psychology course), and Rutchick S2 / Martinie / Lee (MTurk).
- Confidence was "normalized … between 0 to 1" (TH p.43–44). No formula is given.

**Datasets found on the web**

| Ref | Content | Elicitation | Download URL | Availability |
|---|---|---|---|---|
| [25] Martinie, Wilkening & Howe 2020, PLOS ONE 15(4):e0232058 | 500 true/false grade-school science statements (5 difficulty levels × 100, half true). 459 MTurk workers (500 recruited, 41 excluded for self-reported cheating). Each worker answered 100 items, ≈92 responses per item. → **"General" (500)** | Vote T/F; probability the statement is true; meta-prediction of % of others answering "true"; meta-prediction of others' average probability. Responses whose vote conflicts with the probability were dropped for that item | S1 File (questions, responses, code, ZIP): `https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0232058.s003`. Article: `https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0232058` | Public ("All relevant data are within the paper and its Supporting Information") |
| [26] Rutchick, Ross, Calvillo & Mesick 2020, Cogn. Res. 5:50 | **S1 NFL** 2018 weeks 1–15: 216 games analysed, 121–161 participants per week. **S2 Midterms 2018**: 110 races (41 House, 33 Senate, 36 governor), 309 MTurk workers, 63–76 predictions per race. **S3 NBA** 2018–19: 465 games, 111 predictors | Pick winner; confidence 1–5 ("guess"…"very high"); % of other participants who agree with you | Project (prereg + materials + data): `https://osf.io/u9k72/`. S1 data `https://osf.io/y7z6s/`; S2 `https://osf.io/x8s76/`; S3 `https://osf.io/2rzcs/`. Supplement: `https://pmc.ncbi.nlm.nih.gov/articles/instance/7658271/bin/41235_2020_256_MOESM1_ESM.docx` | Public (OSF) |
| [27] Lee, Danileiko & Vi 2018, JDM 13(4):322–333 | 256 NFL games, 2017 season; ≈100 MTurk workers per week (fresh each week). The paper's analysis uses only self-rated "extremely knowledgeable" workers (12–33 per week) | Pick winner; confidence on a 5-point verbal scale; % of others picking the same team (0–100 slider) | OSF: `https://osf.io/3kjmu/` (raw data, code, SP results, interface screenshot) | Public (OSF) |

**Mapping to thesis domains** [INFERRED; see OQ1]
- **General** = Martinie (500 items; per-item n≈92 fits mean 70 and max 129).
- **NBA** = Rutchick S3 (465 → 393 after exclusions).
- **Politics** = Rutchick S2 (110 → 104).
- **NFL = Rutchick S1 (216 → 212)** [INFERRED, moderate–strong]. Evidence:
  - The thesis NFL SP rate .514 (109/212, TH Table 5.5) equals Rutchick S1's published SP 111/216 = 51.4%. The thesis NFL MR .524 vs Rutchick democratic 116.5/216 = 53.9% (Rutchick scores ties 0.5).
  - Lee 2018 reports that SP on **all** workers "performed much worse" than its expert-only 174/256 (68%).
  - Lee recruited exactly 100 workers per week, so no Lee game can reach N>100. The thesis max N=129 then comes from Rutchick S1 (121–161 per week) or Martinie.
  - Psychology-course students appear in the thesis sample description (TH p.43); they exist only in Rutchick S1.
  - **Conflict that stays open:** TH p.43 says "We used data collected from previous studies with open-access material [25–27]", and MLA §5.1 names Lee et al. 2018 among the sources. So the papers state that Lee data **was used**. The papers do not say that Lee was cited only as background. Recorded as D21. The NFL column of T3b decides: SP 109, MR 111, WC 114, HAC 131, DA 101 of 212. HAC is .618 against MR .524, so the HAC count separates the candidate sources well.
  - Caution on the SP match: 109/212 and 111/216 are different counts that round to the same rate, and Rutchick's 111 includes half-credit ties. Rutchick's democratic 116.5 cannot become the thesis MR 111 by removing 4 games alone (at most −4), the counts are not related by game removal alone.
  - Caution on N: Rutchick S1 had 121–161 participants per week, but the thesis maximum is 129. Under S1 the per-case maximum is below the weekly participant counts. A budget check after loading: 393·mean N(NBA) + 212·mean N(NFL) ≈ 1209·70 − 500·mean N(General) − 104·mean N(Politics).
- The per-case maximum of 129 is lower than Rutchick S1's 121–161 weekly participants. This fits if not every participant predicted every game, or if incomplete responses were dropped (TH p.43 requires all three components). Check after download.

**Case ordering** [INFERRED]
The reported per-fold test scores fit an **unshuffled** `KFold(n_splits=10)` over cases ordered General (0–499) → NBA (500–892) → NFL (893–1104) → Politics (1105–1208). Evidence:
- 1209 rows give fold sizes 121×9 + 120. Fold 1 test 0.851 = 103/121; fold 10 test 0.592 = 71/120 (sklearn gives the extra row to the first folds). All 20 AMP/DAP test scores are k/121 or k/120 within 5e-4. The AMP counts sum to 935 and the DAP counts to 937 (checked).
- Folds 1–4 (rows 0–483, all General) have test scores .851/.793/.826/.835, mean .826. AMP-General is .824.
- Fold 5 is **mixed**: rows 484–604 = 16 General + 105 NBA. Folds 6–7 are pure NBA. Scores .934/.917/.860; AMP-NBA is .919.
- Integer back-solve with the per-domain AMP counts (412/361/100/62) is feasible: fold 5 has 12/16 General correct; fold 8 has 45/46 NBA and 33/75 NFL; fold 10 has 9/16 NFL. This is consistency, not proof.
- Textual evidence: on the 10-fold Politics result the thesis says "the trained model has learned primarily **or even solely** from the sports outcome prediction problems and general knowledge problems" (TH p.92). This reading is consistent with Politics lying in the last fold [INFERRED].
- Fold 8 (rows 847–967, ≈46 NBA + 75 NFL) predicts (46·.92+75·.47)/121 ≈ .64. Observed .645.
- Fold 9 (all NFL) is .479. AMP-NFL is .472.
- Fold 10 (16 NFL + 104 Politics) is .592. AMP-Politics is .596.

Bagging Tables 5.9/5.10 show the same fold pattern. This implementation therefore uses unshuffled folds (OQ16, [INFERRED]).

**Preprocessing pipeline** (each step tagged)
1. Load raw responses per item/game/race [EVIDENCE: data sources].
2. Map each response to (v, c∈[0,1], ps vector over the 2 answers):
   - Rutchick/Lee: ps[v]=x/100 and ps[other]=1−x/100, where x is "% who agree with me" [INFERRED from the elicitation; OQ3].
   - Martinie: ps[True]=y/100 and ps[False]=1−y/100, where y is "% of others answering true" [INFERRED; OQ3].
   - Confidence normalization: OQ2.
3. Drop responses missing any of the three components (TH p.43 requires all of them) [EVIDENCE].
4. Exclude cases with 0% or 100% correct [EVIDENCE].
5. Minimum number of responses: Min N = 10 is a reported statistic, not a stated filter. Not applied by default; N ≥ 10 is a gate-ladder switch (Step D5) [OPEN: OQ4].
6. Order cases by domain as above (OQ16).

Acceptance check: 1209 cases with domain counts 500/393/212/104, the N and correct-rate statistics above, **and the per-domain rule counts T3b** (see §2.1.1).

### 2.1.1 Acquisition and cleaning procedure (World A, public data only)

The data steps are numbered D0–D6. Step IDs A0–A20 (§9.3) are code steps; step A8 implements D1–D2, step A9 implements D3, D4 and D6, and step A10 implements D5.

AMP/DAP use **only the original public datasets** (a requirement of the project). There is no synthetic substitute for World A.

**Step D0 — Acquire** the files of Martinie, Rutchick and Lee (see D0b; layout in docs/DATA.md). Store the raw files read-only under `data/raw/`, with SHA-256 recorded:

| Domain | Source file | Expected raw unit count | Evidence |
|---|---|---|---|
| General | Martinie 2020, S1 File ZIP (`journal.pone.0232058.s003`) | 500 statements; 459 workers × 100 trials | PLOS ONE article; TH p.43 example statement is a grade-school science item |
| NBA | Rutchick 2020 S3 data, `osf.io/2rzcs` | 465 games, 111 predictors | PMC7658271 |
| NFL | Rutchick 2020 S1 data, `osf.io/y7z6s` (default); Lee 2018 `osf.io/3kjmu` (alternative, OQ1) | 216 games (Lee: 256) | PMC7658271; Lee 2018 |
| Politics | Rutchick 2020 S2 data, `osf.io/x8s76` | 110 races, 309 workers after quality checks | PMC7658271 |

**Step D0b — Local file layout (written before the files were opened; the layout actually used is in docs/DATA.md).** Root: the data folder (`data/`, or env `CDM_DATA_ROOT`).

| Folder | Contents (as listed before opening) | Planned use |
|---|---|---|
| `Martinie2020_General/` | PLOS "S1 File" zip (file id `s003`) extracted: `AnalysisCode/Raws/r2a..r2e_rawdata.xlsx`, `r2_outcomes.xlsx`, `QuestionsList.xlsx`; `Files/grdata.mat`, `GenPreds.mat`; `R/1.csv`, `2.csv`; `readme.pdf`; Matlab `preprocessData.m` | General domain (500). Read `readme.pdf` and `preprocessData.m` first: they show how the authors excluded cheaters and inconsistent votes, which are reused (Step D1). **To confirm on opening:** what r2a..r2e are. The article says each difficulty level's 100 statements is one data set, that each worker saw 20 statements from each level, and that statements came "in one of five randomized orders"; so a..e may be the five levels or five survey versions. Also to confirm: that `r2_outcomes.xlsx` holds the true answer per item; whether the 41 excluded workers are identifiable; and whether probabilities are whole percentages (this decides OQ2, see §3.3.2) |
| `Rutchick2020_NBA_NFL_Politics/` | OSF u9k72 archive: `Data Revisions.docx`; Study 1 `NFL Data Raw (Study 1)[ REVISED].xlsx`, `NFL Data Analysis (Study 1)[ REVISED].xlsx`, `NFL Pickwatch Data`; Study 2 `Election Data Raw (Study 2)[ REVISED].xlsx` / `Election Data Analysis (Study 2)[ REVISED].xlsx`; Study 3 `NBA Data Raw (Study 3)[ REVISED].xlsx` / `NBA Data Analysis (Study 3)[ REVISED].xlsx`; `.qsf` survey files; pre-registration PDFs | NFL ← Study 1 Raw; Politics ← Study 2 Raw; NBA ← Study 3 Raw. **Expected, to confirm on opening:** the "Analysis" files hold the authors' own per-game democratic/CW/SP results (if so, they are the primary loader sanity check for step A8, ahead of the published totals, because the files are REVISED); the `.qsf` files hold the exact question wording and raw scales (confidence 1–5; agreement %). The raw scale settles the input side of OQ2/OQ3 only; the thesis's normalization formula stays inferred. `NFL Pickwatch Data` is media picks and is **not** used |
| `Lee2018_NFL/` | OSF `3kjmu` archive (added later; see AM6, AM8) | The papers say Lee data was used (D21). Obtained so that the NFL check can compare both sources (OQ1b). Not loaded into the default build |
| `Shinitzky_MTurk_private_RP_AP/` | not part of this repository | World B real data. The synthetic stand-in (§2.3) is used until this is filled |

- **OQ44 Original vs REVISED Rutchick files.** Only "[ REVISED]" versions are present, and `Data Revisions.docx` describes the changes. The thesis does not say which version Shinitzky used, and the revision date is not known from the sources. Date bound: the thesis PDF was produced on 11 Apr 2023, but the World-A data set was fixed by 3 Apr 2022, because MLA (arXiv v1 of that date) reports the same 1209 cases and the same rule counts (864/831/830). A revision dated after 3 Apr 2022 is post-dated relative to the preprint; the sources do not state which file version was used; check the date inside `Data Revisions.docx` or the OSF file history. → **Default: use REVISED.** Read `Data Revisions.docx` first and log every revision that touches games, participants or responses. If the A10 per-domain check differs for NFL/NBA/Politics by more than ±1 and the revision log lists edits to the affected games, try reverting those specific edits. The original files, if still on OSF (version history), can be fetched. (Both versions were later found in the archive: AM8.)
- **OQ1b Is Lee 2018 needed?** Options: (a) do not obtain it unless the S1 gate fails; (b) obtain it now and run the NFL gate on both sources; (c) build with Lee as the NFL source. → **(b).** The default build still uses Rutchick S1 (SP .514 = 51.4%; Lee N=100 < max 129), but the papers say Lee data was used (D21), the download is small (`osf.io/3kjmu`), and running the NFL column of T3b on both sources settles OQ1 with data instead of argument.

**Step D1 — Per-source loaders → common Response schema** (v, c∈[0,1], ps∈[0,1]², domain, case_id, a*):
- Martinie:
  - v = T/F vote;
  - p = the probability that the statement is true (0–100);
  - c = |p−50|/50 (OQ2 option a, default; options (b) and (c) stay switchable, see OQ2 and §3.3.2);
  - ps[True] = y/100, where y = the predicted % of others answering "true"; ps[False] = 1−ps[True] (OQ3).
  - Exclusions [OPEN: OQ4]: Martinie's own analysis drops the 41 self-reported cheaters and, per item, any vote inconsistent with its probability (Martinie Methods). The thesis does not say whether Shinitzky did the same. Default: apply both. The loader keeps a flag for each, because the inconsistent-vote rule interacts with OQ2 option (c).
- Rutchick S1/S2/S3 (and Lee):
  - v = picked team/candidate;
  - c = (Likert−1)/4 (OQ2);
  - ps[v] = x/100, where x = % of others agreeing; ps[other] = 1−x/100. Rutchick's own SP footnote uses the same complement construction.
  - Apply the authors' preregistered exclusions (Rutchick S2: 92 workers failed quality checks; S1: 8 games with a wrong home team are already excluded in the 216; S3: games below 25% of peak participation are already excluded in the 465).
- Answer encoding (OQ5): a1 = True / home team / first-listed candidate.

**Step D2 — Completeness filter:** drop any response missing v, c or ps (TH p.42–43: every response "must … include all of the following information").

**Step D3 — Case filters:** drop cases with 0% or 100% correct (TH p.43). The minimum N=10 is a reported statistic and not a stated filter (OQ4). N≥10 is off by default and is switch 4 of the Step D5 ladder.

**Step D4 — Count gate:** 500/393/212/104; 709 prediction cases + 500 general (TH p.43). N mean 70, SD 35.5, min 10, max 129; correct-rate mean 62%, SD 23.8%, min 2.7%, max 98.9% (TH p.43). If a domain count is off, log which filter would fix it. Do not tune silently.

**Step D5 — Rule gate (before any ML):** compute MR/WC/HAC/SP/DA per case and compare per domain to **T3b** (20 counts), then to T3/T9.
- Tolerance: ±1 case per cell for ties.
- What each rule can and cannot tell us:
  - MR depends only on votes. SP depends on votes and ps. A difference in MR or SP points to the loader, the filters or the tie rule (OQ1/OQ3/OQ4/OQ8/OQ44). Confidence is not involved.
  - HAC compares mean confidences, so it is **unchanged by any positive linear rescaling** of confidence: (c−1)/4 and c/5 give the same HAC answers, and so do Martinie options (a) and (b). A HAC difference therefore cannot be removed by a normalization switch. It points to the responses or to ties, or (General only) to option (c) versus (a), which differ on vote-inconsistent responses.
  - WC = Σ c over supporters / N. It **does** change with the offset: (c−1)/4 versus c/5, and (a) versus (b). So WC is the only rule that tests the normalization. DA follows from the other four.
- If the gate fails, try these switches **in this order, one at a time**, and log every run (counts before and after):
  1. Tie rule for MR/WC/HAC/SP (OQ8): canonical first → canonical second → Rutchick-style half credit, reported as a range.
  2. Confidence normalization (OQ2), judged on WC only: Likert (c−1)/4 ↔ c/5; Martinie (a) ↔ (b) ↔ (c).
  3. Martinie exclusions on/off (OQ4): cheaters; vote-inconsistent responses.
  4. Minimum-N filter (OQ4): none ↔ N ≥ 10.
  5. Rutchick revisions (OQ44): revert the specific edits listed in `Data Revisions.docx`.
  6. NFL source (OQ1/OQ1b): Rutchick S1 ↔ Lee 2018.
- A switch is adopted only if it brings **all five rules** of the affected domain within tolerance and does not break another domain. A combination that fits one cell by breaking another is not adopted.
- If no listed switch passes, **stop and report to the project owner** with the gate log. No other tuning is allowed, and no ML step (A11–A19) starts without the project owner's decision. (Outcome: AM17, docs/RULE_BASELINE_CHECK.md.)

**Step D6 — Order** cases General → NBA → NFL → Politics, in source-file order within each domain (OQ16). Freeze the dataset (parquet + hash). Every later step reads only this frozen file.

### 2.2 World B — Shinitzky's MTurk multiple-choice data (RP/AP)

**What the papers say** [EVIDENCE]
- 22 tasks (14 logic riddles / misleading math, 4 of them from Ackerman 2014; 8 "gamified" P-space/NP-complete tasks: Rush Hour, Knapsack). Collected on MTurk (TH §4.2.1 p.52–53; MCF §4.1).
- Each response has: vote; confidence ("How confident are you in your answer?"; scale not reported); predicted % of responders selecting *each* option.
- Table 4.1 (TH p.54) gives per-task |A_P|, |R_P| and S(a*). Appendix B gives 6 task texts with correct answers.

**Partition rule** [INFERRED from Table 4.1; strong]
k = ⌊|R_P|/30⌋ disjoint sub-sets, near-equal in size, covering all responses.
- Σk = 77 ✓ (matches "77 sub-sets").
- Every "Avg. Subset Size" matches |R_P|/k. Examples: 88→2×44; 114→3×38; 248→8×31; 310→10×31; 121→4×30.25.
- This also explains "30–45 responses" (TH p.53).

**Availability:** MCF says "Data is available on request" (MCF p.108). The real run needs the raw per-response table:
- task id, response id;
- vote, confidence (raw scale), predicted % per option;
- the correct answer per task;
- ideally the original 77-subset assignment.

The demographics are not needed.

**Source notes**
- Σ|R_P| in Table 4.1 = **2490**, not the 2514 stated in TH/MCF.
- MCF-arXiv (2201.08247) states **2490** responses and **28%** average solvers. TH/MCF state 2514 and 30%.
- Response-weighted mean S(a*) from Table 4.1 is 0.280 (= arXiv's 28%). The **unweighted** mean over the 22 tasks is **0.303** (= TH/MCF's 30%). So the 30% vs 28% difference is an averaging choice, not a data difference. For 2514 vs 2490 (24 responses) the sources give no reason.
- Which count the real file has is OQ34.

### 2.3 Synthetic World-B stand-in (RP/AP only, until the real MTurk file is available)

Purpose: let the RP/AP code (steps B1–B11) be built and tested end to end before the private data arrives. **Synthetic results are never reported as reproductions.** The generator writes exactly the loader schema of §3.1 plus `task_id`, so the real file replaces it without code changes (`data/worldB/load_mturk.py` vs `synth_worldB.py`, chosen by a config switch).

| Generator element | Value | Source / status |
|---|---|---|
| Tasks, \|A_P\|, \|R_P\|, S(a*) per task | the 22 rows of TH Table 4.1 exactly (Σ\|R\|=2490); the solver count per task = round(S(a*)·\|R\|) | **[EVIDENCE]** TH Table 4.1 p.54 = MCF Table 3 |
| Sub-set partition | ⌊\|R\|/30⌋ disjoint near-equal sub-sets → 77 | Disjointness is **[EVIDENCE]** (TH p.53: sampled "without overlap"; p.55: "no overlaps"). The count rule ⌊\|R\|/30⌋ is **[INFERRED, strong]** §2.2 |
| ps vector per response | non-negative, Σ=1 (shares) | **[INFERRED]** from Table A.4: PChiSq=.682 needs Σ_a AvgPS(a)=1 |
| Confidence | continuous in [0,1] | **[INVENTED]** The scale is not reported (TH p.53 gives only the question text) |
| Distractor shares (non-a* mass across \|A\|−1 answers) | Dirichlet with one "lure" answer receiving a share λ_t of the non-solver mass | **[INVENTED-CALIBRATED]** "misleading" tasks (TH p.52) motivate a lure |
| Confidence model | c ~ Beta, with the mean rising with the popularity of the voter's answer (consensuality, Koriat 2008) and a small solver bonus | **[INVENTED-CALIBRATED]** The direction is backed by TH p.84 (higher in-group-relative confidence → lower correctness) |
| Prediction model | each responder predicts shares = a noisy version of the true shares, shifted toward their own answer (false-consensus); solvers additionally anticipate the lure | **[INVENTED-CALIBRATED]** The SP mechanism follows Prelec 2017. The direction is backed by the SHAP findings (dPSv top, PSvD_OG strong; TH pp.83–84) |
| Calibration targets | MR 34/77, WC 34/77, SP 33/77, HAC 22/77 (T24), averaged over 30 partition seeds; per-task S(a*) exact | Rates are **[EVIDENCE]** (TH Fig 5.3); the counts are the only integers that round to them |
| Feature oracle | at least one generated sub-set must be able to produce A.4/A.6-like values (S=8/33, ps[v]=1 for a minority voter) | TH A.4/A.6 |

Rules:
- Calibrate only the INVENTED parameters (λ_t, Dirichlet α, Beta parameters, noise and shift sizes) by a seeded grid search on the 4 rule targets.
- Report the calibrated values in `config/synthetic_worldB.yaml`, each tagged INVENTED.
- RP/AP outcomes (T23, T26–T29) are **not** calibration targets. Matching them on synthetic data would be meaningless.
- OQ43 lists these parameters.

---

## 3. Formal definitions

### 3.1 Response and case (TH §3.1 p.21–24, Table 3.1)
- A case is (P, A_P, a*, R_P).
- A response is r = (v ∈ A_P, c ∈ [0,1], ps ∈ [0,1]^{|A_P|}).
- Supporters R_P(a) = {r : r.v = a}. Non-supporters R̄_P(a) = R_P \ R_P(a).
- In-group of r = R_P(r.v); out-group of r = R̄_P(r.v) (MCF §3).
- S(a,R) = |R(a)|/|R|. S(R) = the vector over **all** a ∈ A_P, including zero-vote answers (OQ7).
- a_max = argmax S; a_min = argmin S (ties: OQ7).
- C(R) = {r.c}. PS(a,R) = {r.ps[a]}. PSV(R) = {r.ps[r.v]}.
- Functions: Average, Var, Max, Min, Entropy (TH p.24).
  - **"Var" = population standard deviation** [INFERRED, see 3.4].
  - Entropy uses log base 2 [INFERRED].

### 3.2 Rule-based aggregators

| Method | Thesis definition (TH §3.1 p.24–25) | Origin in related work | Differences to note |
|---|---|---|---|
| MR | argmax_a S(a) | Hastie & Kameda 2005; Sorkin et al. 1998; plurality ("largest minority", TH p.14) | Ties unspecified (OQ8) |
| WC | argmax_a S(a)·Average(C(R(a))) (= Σ_{r∈R(a)} r.c / \|R\|, i.e. a confidence-weighted vote) | Koriat 2008/2012; Hertwig 2012 ("weighting"); Prelec 2017 "confidence-weighted vote"; Rutchick/Lee "confidence-weighted tally" | Equivalent to summing raw confidences per answer. Prelec treats c as a probability averaged linearly or non-linearly; here it is linear, after rescaling c to [0,1]. With a 1–5 scale, the rescaling changes WC results (OQ2) |
| HAC | argmax_a Average(C(R(a))) | Prelec 2017 "Max. confidence: selects the answer endorsed with highest average confidence"; Koriat 2012 maximum-confidence slating (dyads: follow the more confident member) | Koriat's MCS acts on individuals; HAC acts on answer-level means. MLA §3 words the rule as "received the highest support"; TH and MCF give the average-confidence formula, which is used here (D6). An answer with one very confident supporter can win |
| SP | argmax_a [S(a) − Average(PS(a,R))] | Prelec, Seung & McCoy 2017, Nature 541:532 | For binary questions Prelec's rule ("more popular than people predict", predictions arithmetically averaged) equals the thesis rule. For m>2 Prelec defines a "prediction-normalized vote" (Theorem 3, ratio form, 0/0≡0) and notes that more than one answer may be under-predicted. **The thesis uses the difference form for all m.** Prelec elicits predicted vote shares for each answer; Rutchick/Lee elicit "% agreeing with me", which in the binary case fully defines both shares. Rutchick S1 scored SP ties as 0.5 correct; the thesis has no tie rule (OQ8) |
| DA | Count, for every a ∈ A_P (including zero-vote answers), how many of {f_MR, f_WC, f_HAC, f_SP} chose it. Return the answer with the lowest count. **Tie-break: the answer not chosen by MR** | Novel (TH §3.3.1.1 p.34–36; MLA §3.1) | For \|A\|>2, ties among several non-MR answers are unresolved ("implementation-design choice", TH p.35) — OQ8b. In ablations, DA uses only the methods kept in that experiment (TH §4.1.5 p.48) |

Deterministic unit tests from TH p.35:
- (MR,WC,HAC,SP) = (a1,a1,a2,a1) → DA = a2.
- (a1,a1,a2,a2) → DA = a2 (tie-break).

Binary sanity checks [INFERRED]:
- When all 4 rules agree, DA returns the other answer. So DA is correct on every case all 4 rules fail: 202 = 1209×(1−0.833). This matches T9.
- When MR is the lone dissenter (1 vs 3), DA = MR's answer.

### 3.3 CDMCR — 27 features (TH Table 3.2 p.28, Table A.1 p.109; MLA Table 2)

Notation: R = R_P. {R'} = sub-sets from the partition (3.3.1).

| # | Name | Formula | Group (TH §4.1.5) | Notes |
|---|---|---|---|---|
| 1 | N | \|R\| | Vote | |
| 2 | D_Smax_Smin | S(a_max)−S(a_min) | Vote | |
| 3 | ES | Entropy(S(R)), base 2 | Vote | A.2: 0.820 ✓ |
| 4 | VarS | Var(S(R)) = **pop. SD** over the \|A\| shares | Vote | A.2: 0.244 ✓ |
| 5 | D_S_Uniform | Euclidean ‖S(R) − (1/\|A\|)·1‖₂ | Vote | A.2: 0.346 ✓ |
| 6 | CAmin | Average(C(R(a_min))) | Conf | |
| 7 | CAmax | Average(C(R(a_max))) | Conf | |
| 8 | MinC | Min(C(R)) | Conf | |
| 9 | AvgC | Average(C(R)) | Conf | |
| 10 | VarC | pop. SD of C(R) | Conf | A.2 0.222 is read as a population SD: with mean 0.749 on [0.05, 1] a variance cannot exceed 0.188 |
| 11 | D_MaxC_AvgC | Max(C)−Average(C) | Conf | |
| 12 | B_Amax_MaxCa | 1 if f_HAC(R)=a_max (TH). MLA: 1 if Avg C(a_max) = max over answers of Avg C(a_i) | Conf | Differ only under HAC ties (OQ14) |
| 13 | MaxPSa | max_a Average(PS(a,R)) | PS | |
| 14 | MinPSa | min_a Average(PS(a,R)) | PS | |
| 15 | AvgPSV | Average(PSV(R)) = mean r.ps[r.v] | PS | |
| 16 | P_lowC_highPSv | share of r with r.c < AvgC **and** r.ps[r.v] > AvgPSV | Conf+PS | Strict inequalities as written |
| 17 | P_lowPSv_highC | share of r with r.c > AvgC **and** r.ps[r.v] < AvgPSV | Conf+PS | |
| 18 | SG_B_Amax | 1 if a_max differs across sub-sets | Vote | OQ12 |
| 19 | SG_VarSAmax | pop. SD over sub-sets of S(a_max, R') | Vote | Per-subset vs global a_max: OQ12 |
| 20 | SG_D_MaxVarS_MinVarS | max−min over sub-sets of VarS(R') | Vote | |
| 21 | SG_D_MaxES_MinES | max−min over sub-sets of ES(R') | Vote | |
| 22 | SG_D_MaxAvgC_MinAvgC | max−min over sub-sets of **Average(C(R'))** | Conf | TH Table 3.2 prints "Average(S(R'))"; TH A.1 and MLA give confidence; this implementation uses confidence (D7, OQ36) |
| 23 | SG_D_MaxVarC_MinVarC | max−min over sub-sets of Var(C(R')) | Conf | |
| 24 | SG_VarCAmin | pop. SD over sub-sets of Average(C(R'(a_min))) | Conf | Empty supporters in a sub-set: OQ7 |
| 25 | SG_VarCAmax | pop. SD over sub-sets of Average(C(R'(a_max))) | Conf | |
| 26 | SG_AvgCAmin | mean over sub-sets of Average(C(R'(a_min))) | Conf | |
| 27 | SG_AvgCAmax | mean over sub-sets of Average(C(R'(a_max))) | Conf | |

Feature groups (TH §4.1.5 p.47; MLA §7):
- Vote = {1,2,3,4,5,18,19,20,21}
- Conf = {6–12,16,17,22–27}
- PS = {13,14,15,16,17}

16 and 17 belong to both Conf and PS.

#### 3.3.1 Sub-set sampling for SG features
- [EVIDENCE] "sampled sub-sets of 30 responses. If … smaller than 30, … two sub-sets of equal size" (TH p.27).
- [INFERRED, moderate for World A] **Disjoint partition** into k=⌊N/30⌋ sub-sets. Table A.2 (N=90) is reproduced exactly by a partition into 3 disjoint sub-sets of 30 whose a_max counts are (21,23,23), summing to all 67 a_max votes. (21,23,23) is the unique 3×30 split giving these values. Caveat: 3 *overlapping* random samples of 30 also sum to 67 with probability ≈0.12, so A.2 is consistent with, not proof of, disjointness. The strong evidence for the ⌊N/30⌋ rule is World B's Table 4.1.
  - SG_VarSAmax = popSD(.700,.767,.767) = 0.0314 → **0.031 ✓**
  - SG_D_MaxVarS_MinVarS = 2/30 = 0.0667 → **0.067 ✓** (binary: SD of S = \|S−0.5\|)
  - SG_D_MaxES_MinES = H(.700)−H(.767) = 0.0975 → **0.098 ✓**
- The same ⌊N/30⌋ rule produces exactly the 77 RP/AP sub-sets (2.2). One shared function is assumed [INFERRED].
- **Gap:** for 30 ≤ N < 60, ⌊N/30⌋ = 1, so the SG features are degenerate. Recommended default: if k<2, use two halves (OQ6). This matters: SG_B_Amax is the top SHAP feature for MR, DA and SP (TH p.78), NBA games may have small N, and every bagged instance (size 35–68, §5.6) falls in this gap. After the data is loaded, report per domain how many cases have N<30, 30≤N<60 and N≥60.

#### 3.3.2 Table A.2 consistency checks [INFERRED]
Inputs: N=90, S(a_max)=67/90=0.7444.
- ES = 0.8200 ✓; VarS = 0.2444 ✓; D_S_Uniform = 0.3457 ✓.
- AvgC = (67·.788 + 23·.634)/90 = 0.7487 ✓ (0.749).
- P_lowC_highPSv = 11/90 = 0.122 ✓; P_lowPSv_highC = 8/90 = 0.089 ✓.
- MaxPSa + MinPSa = 1 ✓ (binary predicted shares sum to 1).
- MinC = 0.05 and Max = AvgC + D = 1.0. This constrains the confidence normalization (OQ2):
  - Confirmed: 0.05 cannot be obtained under either Likert normalization (values {0,.25,…} or {.2,…}) and under max(p,1−p) when vote and probability agree (≥.5). So A.2 is not a Rutchick/Lee case; it is taken to be a Martinie item [INFERRED] (N=90 fits ≈92 per item, but N alone would not show this).
  - **Not** established: that the formula is c=|p−50|/50. That gives 0.05 only for p = 52.5 or 47.5. If Martinie probabilities are whole percentages, option (a) yields only multiples of 0.02, and 0.05 cannot occur. The article does not state the response format.
  - A second candidate fits with whole percentages: option (c), c = p/100 for a True vote and (100−p)/100 for a False vote, **with vote-inconsistent responses kept** (a True voter with p = 5, or a False voter with p = 95, gives 0.05).
  - Decision rule once the file is open: if the probability column contains values ending in .5 (or finer), (a) stays the default. If it holds whole numbers only, (a) does not fit A.2; test (c) with inconsistent responses kept, and any other form that yields 0.05, against the General WC and HAC counts (380 and 338 of 500).
- Feature 22 (0.098) is not used as an oracle (see AM13).

### 3.4 RCR — 14 features (TH Table 3.3 p.30, A.3; MCF Table 1)

Per response r in case R. The thesis PDF formulas were checked by rendering the page (the text extraction was garbled).

| # | Name | Formula |
|---|---|---|
| 1 | ChosenMajAns | 1 if r.v = a_max (OQ28 for ties) |
| 2 | Conf | r.c |
| 3 | ConfD | r.c − Average(C(R)) |
| 4 | ConfD_IG | r.c − Average(C(R(r.v))) (in-group **includes r**; [INFERRED, strong]: A.4 ConfD_IG=.175 ⇒ IG mean .825 = A.6 AvgConf .825, and MCF §3 defines the in-group as R_P(r.v)) |
| 5 | ConfD_OG | r.c − Average(C(R̄(r.v))) |
| 6 | PSv | r.ps[r.v] |
| 7 | dPSv | S(r.v,R) − r.ps[r.v] |
| 8 | PSvD | r.ps[r.v] − Average(PS(r.v,R)) |
| 9 | PSvD_IG | r.ps[r.v] − Average(PS(r.v,R(r.v))) |
| 10 | PSvD_OG | r.ps[r.v] − Average(PS(r.v,R̄(r.v))) |
| 11 | ChiSq | ½ Σ_{a∈A} (r.ps[a]−S(a,R))² / (r.ps[a]+S(a,R)) |
| 12 | PChiSq | ½ Σ_a (r.ps[a]−Avg PS(a,R))² / (r.ps[a]+Avg PS(a,R)) |
| 13 | PChiSq_IG | same, with the reference Avg PS(a, R(r.v)) |
| 14 | PChiSq_OG | same, with the reference Avg PS(a, R̄(r.v)) |

- 0/0 terms are taken as 0 [INFERRED from A.4: answers with ps=0 and S=0 must contribute 0].
- This is the standard symmetric χ² histogram distance (½Σ(x−y)²/(x+y)). The thesis does not cite a source for it.

**Table A.4 checks** [INFERRED]
Implied quantities:
- avgC = .797, in-group avgC = .825, out-group avgC = .788. Consistency: .242·.825 + .758·.788 = .797 ✓.
- S(v) = .242, Avg PS(v) = .189, in-group Avg PS(v) = .594.

Checks:
- ChiSq = ½[(1−.242)²/1.242 + .758] = **0.610 ✓**
- PChiSq = **0.682 ✓**
- PChiSq_IG = **0.255 ✓**
- PChiSq_OG = **0.887 ✓**, but only with out-group Avg PS(v) = **0.060**, which matches ACR A.6 AvgPS_OG = 0.06. The listed PSvD_OG = 0.904 implies 0.096; Table A.4 prints .904; the value implied by the other printed entries of the same table and by A.6 is .940; this implementation uses .940 (D12, OQ37).
- These checks show A.4 and A.6 describe the same answer in the same sub-set (S = 8/33, so the sub-set size is taken to be 33).
- The PChiSq check needs Σ_a AvgPS(a) = 1. So World-B predicted-share vectors sum to 1, at least on average [bears on OQ39].

### 3.5 ACR — 10 features (TH Table 3.4 p.31, A.5; MCF Table 2)
One instance per answer a with ≥1 vote (TH p.30, p.41).

| # | Name | Formula |
|---|---|---|
| 1 | IsMajority | 1 if a = a_max (OQ28) |
| 2 | Support | S(a,R) |
| 3 | AvgPS | Average(PS(a,R)) |
| 4 | AvgPS_IG | Average(PS(a,R(a))) |
| 5 | AvgPS_OG | Average(PS(a,R̄(a))) |
| 6 | dPS | S(a) − AvgPS |
| 7 | dPS_IG | S(a) − AvgPS_IG |
| 8 | dPS_OG | S(a) − AvgPS_OG |
| 9 | AvgChiSq | mean over r∈R(a) of ChiSq(r) (RCR #11) |
| 10 | AvgConf | Average(C(R(a))) |

Table A.6 checks [INFERRED]:
- AvgPS = S·AvgPS_IG + (1−S)·AvgPS_OG → .242·.594 + .758·.06 = .189 ✓
- dPS = .053 ✓; dPS_IG = −.352 ✓ (listed −.351); dPS_OG = .182 ✓.

---

## 4. Approaches

### 4.1 AMP (TH §3.3.1.2 p.36–37; MLA §4.2)
- Instance: one case → the 27 CDMCR features.
- Labels: multi-label vector (O_MR, O_HAC, O_WC, O_SP, O_DA) ∈ {0,1}⁵, where O_AG = 1 iff f_AG(R) = a*.
- Model: a multi-label classifier (BR, CC or LP wrapping KNN, LR or RF) with per-label probabilities.
- Inference: choose AG = argmax_label P(O_AG = 1 | x) and return f_AG(R).
  - The 0.5 threshold described in the text does not affect the argmax.
  - Argmax ties: OQ17.

### 4.2 DAP (TH §3.3.1.3 p.38; MLA §4.3)
- Instance: 27 CDMCR features + (f_MR, f_HAC, f_WC, f_SP, f_DA).
- Target: a*. Binary classification for |A|=2 (all World-A data).
- Inference: return the predicted class.
- Answer encoding for binary data (OQ5): a fixed per-domain labelling (True/False; team A/B; candidate A/B). The f_AG features use the same encoding (0/1).

### 4.3 RP (TH §3.3.2.1 p.39–40, §4.2.4 p.56–58; MCF §3.2.1, §4.4)
- Instance: one response → 14 RCR features (computed within its sub-set).
- Target: 1 iff r.v = a*.
- Model: soft-voting ensemble of LDA, RF, XGBoost, LR and KNN ("Weighted Average Probability voting") → p_r = P(correct).
- Aggregation strategies (per answer a, over its supporters in the sub-set):
  - maj: count of predicted-True supporters.
  - prop: (# predicted True)/(# supporters).
  - wm: Σ p_r.
  - avgp: mean p_r.
  - maxp: max p_r.
  - maj and prop are tie-broken by wm, avgp or maxp → 9 strategies.
- **Final choice: prop, tie-broken by avgp** (TH p.57; MCF p.101–102).
- Predicted True means p_r > 0.5 (sklearn `predict`).
- Only answers with ≥1 supporter can win (implicit).

### 4.4 AP (TH §3.3.2.2 p.40–41; MCF §3.2.2)
- Instance: one answer with ≥1 vote in the sub-set → 10 ACR features.
- Target: 1 iff a = a*.
- Rule:
  - exactly one predicted True → choose it;
  - several True → the highest P(True);
  - none True → the lowest P(False), which is the same as the highest P(True).
- In every case this reduces to argmax P(True) [INFERRED]. Implement it literally anyway and assert the two agree.

---

## 5. Evaluation protocols

### 5.1 Metrics
- **Success rate** = share of cases where the returned answer = a* (TH §4.1.3, §4.2.3).
- **PRE** = (E1−E2)/E1 with E = 1−success (TH p.65). PRE has two uses:
  - approach vs best rule (T4, T25);
  - per method, P(success) vs P(success given selected by AMP) (T8).
- **Origin of PRE:** Proportional Reduction in Error is a standard statistic (Goodman & Kruskal style). The thesis gives the formula explicitly and cites no source. No difference between the sources.

### 5.2 AMP/DAP model selection — nested CV (TH §4.1.4 p.45–46; MLA §5.4)
- Outer: 10-fold, **unshuffled, dataset order General→NBA→NFL→Politics** [INFERRED; OQ16].
- Inner, per outer fold: for each technique, LOO over the 90% train set → success rate under the AMP/DAP protocol (TH p.45: "highest rate of successful aggregation outcomes").
- Pick the best technique, retrain on the 90%, test on the 10%.
- **Thesis headline (T1/T2)** = accumulated outer test successes (TH p.64).
- Techniques:
  - AMP: {BR, CC, LP} × {KNN, LR, RF} = 9 (TH).
  - MLA additionally tests Bernoulli NB → 12 (MLA Table 5).
  - DAP: KNN, LR, RF.
- Library: scikit-learn (TH p.46).
- **Bold cells checked on the rendered page** (TH p.63): AMP best = BR+RF in folds 1–5, 7, 8 and 10; **CC+RF in folds 6 (0.790) and 9 (0.836)**. DAP best = RF in all folds. The nested-CV code must therefore really select per fold, and step A14 must reproduce CC+RF winning when it scores highest.
- **Fingerprint:** BR+KNN and LP+KNN give identical scores in every fold (TH Table 5.1). This happens when LP returns per-label marginals over the same k neighbours (scikit-multilearn `LabelPowerset.predict_proba`). It supports using scikit-multilearn for BR/CC/LP. Use it as an acceptance test.
- TH §5.1.1 says BR+RF "performed best" overall, and BR+RF / RF are used for the ablation, bagging and per-domain LOO runs.
- Cost: an inner LOO of ~1088 fits × 9 techniques × 10 folds ≈ 98k fits. RF is the main cost. Parallelize with joblib.

### 5.3 MLA "final evaluation" (MLA §5.5)
- A single LOO over all 1209 cases with BR+RF (AMP) and RF (DAP) → T19–T21.
- The ablation row 1 values (TH Tables 5.6/5.7) equal these: **0.806 / 0.812** [EVIDENCE TH §4.1.5 p.48 "leave-one-out" + MLA Fig 1].
- Both protocols (5.2 and 5.3) are reproduced and reported side by side.

### 5.4 Ablations (TH §4.1.5 p.46–49; MLA §7)
- 11 configurations. Row matrix from the rendered TH p.72 (identical in MLA Tables 3/4). 1 = included; Voting features and O_MR/f_MR are always included:

| Row | Conf feats | PS feats | WC,HAC | SP | DA | AMP | DAP |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1 | 1 | .806 | .812 |
| 2 | 0 | 1 | 1 | 1 | 1 | .800 | .812 |
| 3 | 1 | 0 | 1 | 1 | 1 | .816 | .825 |
| 4 | 0 | 0 | 1 | 1 | 1 | .795 | .803 |
| 5 | 1 | 1 | 0 | 1 | 1 | .808 | .804 |
| 6 | 1 | 1 | 1 | 0 | 1 | .797 | .805 |
| 7 | 1 | 1 | 0 | 0 | 1 | .791 | .806 |
| 8 | 1 | 1 | 1 | 1 | 0 | .754 | .813 |
| 9 | 0 | 1 | 0 | 1 | 1 | .806 | .811 |
| 10 | 1 | 0 | 1 | 0 | 1 | .799 | .815 |
| 11 | 0 | 0 | 0 | 0 | 1 | .763 | .798 |

- Ablation is **component-wise**. Rows 2 and 4 drop the confidence features but keep the confidence-based labels WC/HAC; rows 3 and 4 drop the PS features but keep SP. The thesis wording "all engineered features and labels that depend on that information" (TH p.47) is therefore applied literally only in rows 9 (no confidence information: Conf features and WC/HAC), 10 (no predicted-support information: PS features and SP) and 11 (neither). Implement exactly the matrix.
- Features 16/17 belong to both groups: OQ41. In rows 9, 10 and 11 they cannot be computed at all, because one of their two inputs is missing, so dropping them there is forced.
- When WC, HAC or SP is removed, DA is recomputed from the remaining methods (TH p.48).
  - E.g. with only MR + DA, DA = "not MR".
- Protocol: LOO over all 1209 cases with BR+RF / RF.
- Significance: **two-tailed** two-proportion z-test against the full version (TH p.70).
  - The thesis prints no ablation p-values (TH pp.70–71); only the significance pattern is compared: AMP rows 8 and 11 differ significantly from the full version, no other AMP row and no DAP row does (amendment AM2, Appendix A).

### 5.5 Classic meta-learning (TH §4.1.6 p.49–50; §5.1.2)
- DAP with only the f_AG features (no CDMCR), using a basic decision tree (`DecisionTreeClassifier`).
- 5 configurations (Table 5.8). DA is recomputed when methods are dropped.
- CV protocol not stated → OQ21. Default: LOO, as in the ablations.
- Row patterns (rendered TH p.73): (1) MR, WC/HAC, SP, DA .713; (2) MR, SP, DA .708; (3) MR, WC/HAC, DA .695; (4) MR, DA .687; (5) MR, WC/HAC, SP .713.
- Note: with only MR + DA (where DA = ¬MR), the tree can only learn a function of f_MR: MR, ¬MR or a constant answer. So **≈0.687 = MR** is expected (row 4), but it is not guaranteed under answer-identity encoding, because class priors could favour a constant. Rows 1 and 5 should coincide, since DA is a function of the other four.

### 5.6 Bagging (TH §4.1.7 p.50–52; §5.1.3)
- For each case with |R| ≥ 70: draw F random sub-groups.
  - Size = max(35, Z_{0.95}·S(a*)·(1−S(a*))/0.1²), formula as printed (rendered PDF p.50). Note there is **no square on Z**; Cochran's formula is Z²p(1−p)/e² (OQ22).
  - Each sub-group becomes a new instance in BI_j.
- F ∈ {2,3,5,10}.
- Test_K = the original outer fold K (the same folds as 5.2).
- BagTrain_K = the other 90% of originals + all bagged instances **except** those from problems in Test_K.
- Train BR+RF / RF in **every** fold (TH p.73); report per-fold success (Tables 5.9/5.10). The "–" row repeats the Table 5.1/5.2 test column, which in folds 6 and 9 came from CC+RF.
- Every bagged instance is a full new case: its CDMCR features (including SG features), labels O_AG and f_AG, and DA are all recomputed on the sub-group. a* is unchanged [OQ22].
- Size check. The maximum of S(1−S) is 0.25, so the largest size is:
  - as printed (no square), Z=1.645: 41.1; Z=1.96: 49.0. Sizes 35–42 (or 35–49).
  - Cochran (Z²), Z=1.645: 67.7; Z=1.96: 96.0.
  - Only Z=1.96 with the square is ruled out by the threshold (96 > 70). **Z=1.645 with the square gives at most 68, just under the |R| ≥ 70 threshold**, and the floor 35 is half of 70. So both the printed formula and the squared (Cochran) form are compatible with the stated threshold. The papers do not decide between them (OQ22); the printed form is the default and the squared form a labelled switch.
- With either formula the sub-group has fewer than 60 responses, so its SG features come from the k<2 rule of OQ6.

### 5.7 RP/AP evaluation (TH §4.2.4 p.56–59; MCF §4.4)
- Leave-one-group-out over the 77 sub-sets: train on 76 groups' instances, predict the held-out group, aggregate.
- Baselines MR, WC, HAC and SP on the same 77 sub-sets (no DA).
- Fidelity note: as in the thesis (TH p.58), prop + avgp is the RP strategy used for the headline result (T23); all nine strategies of Table 4.2 are reported.
- Out of scope: an "answer-response combination voting approach" was tested but not reported (TH pp.89–90: better than AP, worse than RP, no numbers).
- Other classifiers in Tables A.7/A.8: Bagging-KNN and SVM. In A.7, the "rnd" tie-breaker for maxp/avgp/wm means random tie-breaking.

### 5.8 Statistical tests — conventions resolved by recomputation [INFERRED, strong]
1. **TH Table 5.3** = pooled two-sample z-test for proportions, one-tailed (H1: approach > rule), n1=n2=1209. Recomputing from counts 935/937 vs 864/831/**830**/774 gives 4.69057e-4, 9.40195e-7, 7.57021e-7, 3.1885e-13 (AMP) and 3.30493e-4, 5.8033e-7, 4.65415e-7, 1.5783e-13 (DAP). These match to ≥4 significant figures (HAC differs in the 5th).
   - DA: the true p ≈ 1e-117; the thesis prints 0.
   - Equivalent to `statsmodels.stats.proportion.proportions_ztest([x1,x2],[n1,n2],alternative='larger')`.
2. **TH Table 5.11 / MCF Table 5 proportion column**: the same pooled test with n=77. 49 vs 33 → 0.0049 (0.005 ✓); 49 vs 34 → 0.0077 (0.008 ✓); 49 vs 22 → 6.4e-6 ✓; AP 42 vs 33 → 0.073 ✓, 42 vs 34 → 0.099 ✓.
3. **TH Table 5.4 per-method p-values** = **one-sample** z-test of P(success given selected), n=#selected, against the fixed p0 = P(success), with null SE √(p0(1−p0)/n).
   - Reproduces 4.2e-6 (SP: n=349, k=287), 4.1e-6 (MR: 244/200), 4.0e-7 (WC: 241/201 with p0=830/1209 → 3.98e-7), 0.062 (HAC: 132/93). MLA Fig 2 uses the same convention.
   - The pooled two-sample variant does **not** match.
   - The PRE rows of Tables 5.4, 5.5 and A.10 are computed from exact counts, not from the printed rates: all 5 + 20 + 20 printed values are reproduced from the back-solved counts in T8, T11 and T13.
4. **McNemar** (TH Table 5.11; MLA Table 7) — **resolved [INFERRED, strong]**: two-sided χ² **with continuity correction**, (|b−c|−1)²/(b+c). This is `statsmodels.stats.contingency_tables.mcnemar(table, exact=False, correction=True)`. Brute force over discordant pairs with b−c fixed by the success counts gives integer solutions for all 6 cells only for this variant: RP–MR (b,c)=(15,0) → 3.0e-4; RP–HAC (34,7) → 4.9e-5; AP–SP (11,2) → .027; AP–MR (9,1) → .027; RP–SP c∈{2,3} → .001; AP–HAC c∈{10,11} → .003. Two-sided exact, uncorrected χ² and mid-p each fail 4 cells; one-sided exact fails 3. **Independent confirmation on World A:** all 10 McNemar cells of MLA Table 7 have exactly one integer solution each at 3 s.f. under this variant, given 975/982 and 864/831/830/774/372 (pairs listed in T22). No other variant fits more than 2 of the 10.
5. **MLA Table 7 proportion column** = test 1 with the LOO counts AMP 975 and DAP 982 (AMP vs SP 6.13e-8; DAP vs SP 8.18e-9; MR 831 and WC 830 — all exact ✓).

### 5.9 SHAP (TH §4.3 p.59–61, §5.3; MCF §4.5)
- CDMCR: each AMP label is treated as its own binary target. Use the models trained on the World-A data; report beeswarm plots for MR, DA, SP, HAC (WC in A.1).
- RCR/ACR: the RP/AP models trained on World-B data.
- [INFERRED] The plots are `shap.plots.beeswarm` (new API: "Sum of 8 other features" = 27−19 shown).
- **The x-axes span about ±4 (log-odds units)**. SHAP values of a probability output lie within ±1, so the printed axes correspond to a margin (log-odds) output; the thesis does not name the explained model (OQ30).
- TH p.77 describes importance as the "sum of SHAP values"; `beeswarm` orders features by mean |SHAP|, which is the ordering used here.
- Lundberg & Lee 2017 (thesis ref [105]) is cited only in general terms.

---

## 6. Related-work study (what each source contributes; definitional differences)

1. **Prelec, Seung & McCoy 2017 (Nature)** — source of SP.
   - Data: 490 items across Studies 1a–4b (state capitals, general knowledge, dermatology, art prices).
   - Each respondent gives a vote, predicted vote shares and (in some studies) confidence. Predicted frequencies are arithmetic means of individual predictions.
   - For binary questions the thesis SP is identical.
   - For m>2, Prelec's Theorem 3 uses a prediction-normalized vote (ratio-based, 0/0≡0) and warns that "more popular than predicted" may not pick a unique answer. The thesis uses the difference rule for all m.
   - Prelec also defines a confidence-weighted vote and a "max confidence" rule: the closest originals of WC and HAC.
   - The multi-bin art Studies 4a,b were scored by SP only on a binary split ($30k). Theorem 3 averages predictions *within same-vote groups*, a conceptual relative of the IG/OG features. The m>2 general case is only sketched (SI §1.4).
   - Relevance: World-B tasks are multi-answer. Thesis SP ≠ Prelec SP there, so the reconstruction uses the thesis rule. Prelec's version can be offered as an optional extra baseline.
2. **Koriat 2008 (JEP:LMC, consensuality principle)** — confidence tracks consensus rather than accuracy. This motivates confidence features and explains why HAC and WC fail on misleading items. **Koriat 2012 (Science)** — Maximum-Confidence Slating in dyads (follow the more confident member) beats the better individual when members are similarly competent.
   - HAC generalizes MCS from individuals to answer-level mean confidence.
   - The descriptions come from abstracts and secondary statements in TH §2.2 and MLA §2.
3. **Hertwig 2012 (Science)** — commentary on Koriat 2012. The source of the "weighting vs maximization" distinction behind WC vs HAC (TH p.17).
4. **Martinie, Wilkening & Howe 2020** — World-A "General" data.
   - Proposes Meta-Probability Weighting (MPW): weight each forecaster by the normalized |own probability − meta-prediction of others' average probability|, then extremize with a=2.5.
   - MPW is **not** used as a thesis baseline.
   - Its confidence is a probability that the statement is *true*, not confidence in the chosen answer. This needs conversion (OQ2).
5. **Rutchick et al. 2020** — NBA and Politics data, and (default of OQ1) NFL data.
   - Reported baselines: democratic (= MR), confidence-weighted (≈ WC with a 1–5 scale), SP.
   - SP ties were scored 0.5 in S1.
   - Their published accuracies serve as **sanity checks on loader correctness** (ties = 0.5):
     - S1 NFL: democratic 116.5/216 (53.9%), CW 120 (55.6%), SP 111 (51.4%);
     - S2 elections: democratic 87/110 (79.1%), CW 87.5, SP 88 (80.0%);
     - S3 NBA: democratic 306.5/465 (65.9%), CW 304.5 (65.5%), SP 313 (67.3%).
     - The thesis NFL SP .514 equals S1's.
   - S1 weekly participants: 121–161; S2: 63–76 predictions per race; S3: 111 predictors (PMC7658271).
   - After the thesis's exclusions and normalization the numbers will differ somewhat. A large difference is a reason to check the loader.
6. **Lee, Danileiko & Vi 2018** — NFL 2017, 256 games, ~100 workers per week.
   - Their SP analysis uses only self-rated experts (SP 174/256 correct).
   - Unknown whether the thesis used all workers or only the experts (OQ1).
7. **Multi-label transformations:**
   - Zhang & Zhou 2014 (TKDE review) — general taxonomy.
   - Boutell et al. 2004 — BR: an independent binary classifier per label.
   - Read et al. 2009/2011 — Classifier Chains: label j's classifier gets x plus labels 1..j−1. At test time the chain uses its own predictions. Results depend on chain order (random or fixed) — OQ19.
   - Read et al. 2008 (pruned sets) — cited for LP: a multi-class problem over observed label combinations.
   - **Probability for AMP's argmax:** BR and CC give per-label P(1). For LP the per-label marginal is the sum of P(combination) over combinations containing the label (scikit-multilearn's behaviour). This matches the BR=LP fingerprint for KNN in Table 5.1.
   - Note: with 5 labels at most 32 combinations, fewer observed (the DA structure rules out many combinations).
8. **SHAP — Lundberg & Lee 2017 (NeurIPS)**: additive Shapley attributions; KernelSHAP is model-agnostic. TreeSHAP (Lundberg et al. 2020) is the tree-model version. The thesis cites only the 2017 paper. Explainer choice: OQ30.
9. **Soft voting** (refs [100]–[103]: Breiman 1996 bagging, Wolpert 1992 stacking, Freund 1995 boosting, Dietterich 2000).
   - Thesis "soft voting" = averaging predicted class probabilities (sklearn `VotingClassifier(voting='soft')`). Equal weights unless stated (OQ26).
   - The cited refs are general ensemble references, not specific to soft voting.
10. **XGBoost — Chen & Guestrin 2016**; **scikit-learn — Pedregosa 2011**: implementation libraries.
11. **Ackerman 2014** — source of 4 of the 14 riddles (World B); **Flake & Baum 2002** — Rush Hour; **Mathews 1896** — Knapsack. Task content only.

---

## 7. Differences between the thesis and the two papers

| # | Topic | TH | MLA | MCF | Resolution |
|---|---|---|---|---|---|
| D1 | AMP/DAP headline | 0.773/0.775 (10-fold nested test) | 0.806/0.812 (LOO, all data, Fig 1) | — | Different protocols. Reproduce both (5.2, 5.3) |
| D2 | AMP selection distribution | SP .289, MR .202, WC .199, HAC .109, DA .201 | SP 35%, MR 22%, WC 15%, HAC 15%, DA 13% | — | Values from different protocols (10-fold vs LOO); both reported |
| D3 | DA success given selected | 0.634 | 0.818 | — | Values from different protocols (10-fold vs LOO); both reported |
| D4 | AMP model selection candidates | 9 (no BNB) | 12 (incl. Bernoulli NB) | — | Default: TH's 9; BNB optional (it never wins: ≤0.69) |
| D5 | Statistical tests (AMP/DAP) | Proportion test only | McNemar + proportion (LOO counts) | — | Implement both |
| D6 | HAC definition | avg confidence | worded "highest support" | avg confidence | Use TH/MCF |
| D7 | Feature 22 | prints "Average(S(R'))" in Table 3.2 | Average(C) | — | Use confidence |
| D8 | Feature 12 | f_HAC = a_max | Avg C(a_max) = max Avg C | — | Equivalent except ties |
| D9 | Responses in World B | 2514 | — | 2514 (2490 in MCF-arXiv) | Table 4.1 sums to 2490. Open (OQ34) |
| D10 | Mean solver rate | 30% | — | 30% (28% arXiv) | Not a data difference: unweighted task mean = 30.3%, response-weighted = 28.0% |
| D11 | Sub-set size column | in TH Table 4.1 | — | absent in MCF Table 3 | Use TH |
| D12 | PSvD_OG example | 0.904 (A.4) | — | — | Differs from the value implied by A.6 and PChiSq_OG (0.940); 0.940 is used |
| D13 | Example tasks | Appendix B (6 tasks) | — | "supplementary material" | Use TH App. B |
| D15 | MCF-arXiv naming | RP/AP | — | arXiv used "RCR-Agg / ACR-Agg", 64%/55% | Naming only |
| D16 | Data availability | — | — | "Data is available on request" | World B needs the real file; a synthetic stand-in is used (§2.3) |
| D17 | Bagging, classic meta-learning, per-domain tables, SHAP of CDMCR | present | absent | — | TH is the only source |
| D18 | RP/AP classifier alternatives | A.7/A.8 (Bag-KNN, SVM) | — | absent | TH only |
| D19 | LOO counts | AMP .806 / DAP .812 (ablation row 1) | 975 / 982 (back-solved from Table 7) | — | Consistent |
| D20 | Selection-effectiveness p-values | Table 5.4 (10-fold) | Fig 2 (LOO) | — | Both use the one-sample z-test |
| D21 | Lee 2018 as a data source | "We used data collected from previous studies … [25–27]" (p.43); domains listed are General, NBA, NFL, Politics only | Lee et al. 2018 named as a source (§5.1) | — | **Open.** The counts and the NFL SP rate fit Rutchick S1 alone. Default NFL = S1; Lee compared at the rule-baseline check (OQ1b) |
| D22 | PRE arithmetic | Headline PRE (.204/.211) from rounded rates; Table 5.4/5.5/A.10 PRE from exact counts | — | — | Reproduce each the way it was computed |

---

## 8. Open questions / unspecified details

Format: options → **recommended default** (reason).

**Data (World A)**
- **OQ1 Source → domain mapping; NFL source.** (a) NFL = Rutchick S1 only; (b) Lee 2018 only; (c) both, subsampled.
  - → **(a) Rutchick S1** [thesis NFL SP .514 = Rutchick S1 SP 51.4%; Lee N=100 per game < max 129; Lee all-worker SP "much worse"; psychology-course participants on TH p.43]. Strength: moderate–strong, not proof. Against it: the papers say Lee data was used (D21), and S1's 121–161 weekly participants exceed the thesis maximum of 129. Validate by the NFL column of T3b; run the same gate on (b) (OQ1b).
  - The thesis's merged data file, if available, would settle most World-A questions.
- **OQ2 Confidence normalization.**
  - Martinie (probability p that the statement is true, 0–100): (a) c=|p−50|/50; (b) c=max(p,100−p)/100; (c) p/100 when voting True and (100−p)/100 otherwise (differs from (b) only on vote-inconsistent responses, so it needs them kept).
  - Likert 1–5: (a) (c−1)/4; (b) c/5.
  - → **Martinie (a); Likert (a)**, both provisional.
  - Evidence: Table A.2 has MinC=0.05 and max=1.0. This excludes both Likert forms and (b), so A.2 is a Martinie item. It supports (a) only if p can be 52.5/47.5; with whole-percentage p it supports (c) with inconsistent responses kept instead. Decision rule in §3.3.2.
  - What the choice changes: MR, SP and **HAC** not at all for (a)↔(b) or for the Likert pair (HAC is invariant to positive linear rescaling). **WC** changes, and so do the confidence features (6–11, 22–27 by scale; 16/17 not at all). So the WC column of T3b is the test (Step D5).
  - The thesis says only "normalized … between 0 to 1" (TH p.43–44). A per-source min-max rule applied to Likert 1–5 and to max(p,100−p) gives exactly (a) in both cases; applied to raw p it would not.
- **OQ3 Predicted-support construction.** Binary: ps[v]=x (agreement %), ps[other]=1−x. Martinie: ps[True]=y.
  - → **as stated** (the only coherent mapping). Clip to [0,1].
- **OQ4 Filters.** Minimum responses (10?); dropping incomplete or inconsistent responses (Martinie's own exclusions); Lee expert-only filter.
  - → Use the original authors' exclusions + complete responses + 0/100% exclusion. **No N≥10 filter by default** (it is switch 4 of the Step D5 ladder). Whether Shinitzky applied the original authors' exclusions is not stated; each is a loader flag. Report the counts after every filter.
- **OQ5 Binary answer encoding for DAP targets and f_AG features.** The papers say only that the target is a* and the features are "the answer selected" (TH p.38, p.44). Which of the two answers is coded 1 in each case is not stated, and it defines the DAP learning problem.
  - Options: (a) natural source coding: True=1; home team=1; for elections the first option as stored in the raw file; (b) the option order of the raw source file, whatever it is; (c) random per case (seeded), which removes any prior on a*.
  - → **(a)**, fixed after the raw files are opened (the Rutchick coding may be by team or party name).
  - What is and is not invariant: swapping 0 and 1 for **all** cases changes nothing for RF. Changing **which answer is 1 case by case** (e.g. home/away vs alphabetical vs party) changes the base rate of a*=1 and what the 27 case features can predict, for every classifier. So DAP numbers (T2, T6, T15, T16, T18, T19) depend on this choice. AMP does not use it.
  - Required: run DAP LOO under (a) and (c) and report both. The two runs show how far the DAP rate depends on the answer coding.
- **OQ16 Fold assignment.** → **Unshuffled KFold(10) with order General, NBA, NFL, Politics** ([INFERRED], strong). Within-domain order: the source file order (unknown → OQ, small effect). Alternative: shuffled stratified, as a robustness check.

**Feature definitions**
- **OQ6 Sub-set partition for SG features.** → disjoint random partition into k=⌊N/30⌋ near-equal sub-sets that use every response; if k<2, use 2 halves (extending the stated N<30 rule; for odd N the halves differ by one). Store the seed. Alternatives: (b) overlapping repeated samples of size 30 (the number of samples is then a further unknown); (c) k=⌈N/30⌉; (d) k=⌊N/30⌋ sub-sets of exactly 30, leftover responses unused. The thesis says only "sampled sub-sets of 30 responses" (TH p.27) and "random sub-sets" (Table A.1).
- **OQ7 a_max/a_min ties, zero-vote answers, empty supporter sets.**
  - Does S(R) include zero-vote answers? → yes, over all A_P (DA does this explicitly; ES and D_S_Uniform need the full vector).
  - a_min with zero supporters → CAmin undefined. In World A this cannot happen at the full-set level (0/100% exclusion). In sub-sets it can → **NaN, then impute the global AvgC of that sub-set** (flag; alternative 0).
  - Ties → first answer in a fixed canonical order (deterministic). Log the frequency.
- **OQ9 "Variance" = population SD.** → yes ([INFERRED] from A.2: VarS=0.244=\|S−.5\|; VarC=0.222 exceeds the feasible variance). Keep a switch.
- **OQ10 Entropy base 2** → yes (A.2 ES=0.820 ✓).
- **OQ11 D_S_Uniform Euclidean** → yes (A.2 0.346 ✓).
- **OQ12 Per-subset a_max** (for SG_VarSAmax, SG_B_Amax, SG_*CAmax/min) vs the global a_max. → **per-subset a_max/a_min**, following the wording "a_max differs across sub-sets". Global as an alternative. A.2 cannot discriminate.
- **OQ13 Strict vs non-strict inequalities in features 16/17** → strict, as printed.
- **OQ14 B_Amax_MaxCa under HAC ties** → use f_HAC with the same tie rule as OQ8.
- **OQ28 ChosenMajAns / IsMajority with tied majorities** → 1 for all tied top answers (flag); alternative: canonical first.
- **OQ36/37** source notes: use confidence for feature 22; PSvD_OG(A.4) = 0.940.
- **OQ39 Predicted-support vectors in World B may not sum to 100%** → use as given / 100. Table A.4 implies Σ AvgPS = 1, so a 100% total is assumed. If the real data violates this, renormalize (default) and report the raw variant as sensitivity.
- **OQ41 Features 16/17 in ablations** (they belong to both Conf and PS) → **dropped whenever either group is dropped**. Support: the columns of Tables 5.6/5.7 are the named groups, and both group lists contain 16 and 17 (TH p.47); in rows 9–11 the features cannot be computed anyway. Alternative (rows 2–4 only): keep them unless both groups are dropped.
- **OQ42 Bagging sampling** → each sub-group is drawn without replacement within the case; sub-groups are independent of each other.
- **OQ43 Synthetic World-B generator parameters** (§2.3) → invented, calibrated to T24 only, and to be replaced by the real data.

**Aggregators**
- **OQ8 Ties in MR/WC/HAC/SP** → deterministic canonical order (first answer), logged. Alternatives: random; SP tie = 0.5 (Rutchick).
  - Binary MR ties occur when N is even and the split is 50/50. Expected rare but non-zero; it affects T3 by a few cases.
- **OQ8b DA ties among several non-MR answers (m>2)** → not needed for World A. If DA is used on multi-class data: choose among tied non-MR answers by lowest support (most "devil's advocate") and flag it.
- **OQ38 SP for m>2** → thesis difference form. Prelec's normalized form as an optional extra.

**ML set-up**
- **OQ15 Hyperparameters, versions, scaling.** → sklearn defaults (RF n_estimators=100, KNN k=5, LR lbfgs max_iter raised to avoid non-convergence). **No scaling**: the thesis mentions none and the BR=LP KNN identity holds either way. StandardScaler as a sensitivity run. Seeds fixed and recorded (0).
  - Flag: KNN/LR on unscaled N (10–129) behave differently from scaled data.
- **OQ17 AMP argmax ties** (KNN probabilities in steps of 0.2) → first label in the order (MR, HAC, WC, SP, DA) (thesis order, TH p.37). Alternative: prefer MR.
- **OQ18 Inner-LOO score** = AMP-protocol success rate (TH p.45) → yes.
- **OQ19 CC chain order; LP marginals** → scikit-multilearn defaults (order = label order; LP marginalization).
- **OQ21 Classic meta-learning protocol** → LOO, DecisionTreeClassifier defaults, random_state=0.
- **OQ22 Bagging details.**
  - Size formula: (a) as printed, Z·S(1−S)/0.1²; (b) Cochran, Z²·S(1−S)/0.1². → **(a)**, because it is what the thesis prints (rendered p.50). (b) with Z=1.645 is equally self-consistent (max 68 < 70) and is run as a sensitivity.
  - Z value → 1.645 ("Z_0.95", one-sided). Alternative 1.96 (possible only with formula (a)).
  - Size rounding → ceil (the formula gives non-integers, e.g. 41.1). Labels and f_AG are recomputed per sub-group.
  - Sampling within a sub-group: without replacement.
  - Sub-groups are drawn independently of each other, so overlap between them is allowed.
  - SG features of bagged instances are recomputed inside each sub-group.
- **OQ23** The bagging folds equal the model-selection folds (TH p.51 says so) ✓ evidence.
- **OQ26 Soft-voting ensemble** → `VotingClassifier(voting='soft')` with equal weights. Defaults for LDA, RF, XGBClassifier, LR and KNN; no scaling (flag; scaling matters for KNN/LR).
- **OQ27 RP strategy details.**
  - prop = #True/#supporters.
  - Ties → the chosen tie-breaker; a further tie → canonical order.
  - "Percentage of tie breakers" = share of the 77 sub-sets where the primary criterion tied.
  - When no supporter is predicted True, prop is 0 for all answers → tie → avgp decides.
- **OQ35 Bag-KNN / SVM for A.7/A.8** → `BaggingClassifier(KNeighborsClassifier())` and `SVC(probability=True)` defaults.
- **OQ25 World-B partition randomness** → use the original assignment if it is available. Otherwise random with a seed, repeated over 30 seeds; report mean ± SD (a single seed can swing ±1–2 of 77 cases, i.e. ±2.6% each).
- **OQ33 Class imbalance** (RP ~30% positive) → none (not mentioned).

**Evaluation and reporting**
- **OQ30 SHAP model/explainer.** → (i) closest reading: XGBoost per label (CDMCR) and for RCR/ACR, TreeExplainer, log-odds, beeswarm max_display=20. (ii) Also KernelExplainer on the actual ensemble/RF for comparison. Compare rank orders to T30 (Spearman). SHAP is computed on models trained on all data.
- **OQ31 McNemar variant** → **resolved**: χ² with continuity correction, two-sided (§5.8 item 4).
- **OQ32 Which AMP/DAP headline is compared** → both: TH 10-fold (T1/T2) as primary, MLA LOO (T19) as secondary.
- **OQ34 2514 vs 2490 responses** → whichever the real file has. Document it.
- **OQ40 Per-domain tables** → slice the global predictions by domain (TH p.69); the LOO version comes from the 5.3 run.

**Count: 41 open questions**, including OQ41, OQ42, OQ43 and OQ44 (§2.1.1); OQ31 is resolved. OQ1b is a sub-question of OQ1. These are OQ1–OQ40 plus OQ8b, minus OQ20, OQ24 and OQ29: those numbers are unused because their content went into OQ5, D16 and OQ7.

For 8 of them, back-solving the published numbers gives a strong evidence-based default: OQ6, OQ9, OQ10, OQ11, OQ16, OQ23, OQ36, OQ37. For OQ2 the evidence fixes only the Likert-vs-Martinie identity of Table A.2, not the formula. The statistical-test conventions (§5.8) and the World-B partition rule (§2.2) were settled the same way, so they are not listed as open.

---

## 9. Code architecture and implementation steps

### 9.1 Stack
- Python 3.11, numpy, pandas.
- scikit-learn (pin a version, e.g. 1.3.x, so the RF default n_estimators=100).
- scikit-multilearn (BR/CC/LP). Fallback: sklearn `MultiOutputClassifier` + `ClassifierChain` + a custom LP wrapper.
- xgboost, shap, statsmodels, joblib, pytest.

### 9.2 Layout
```
<repo>/   # planned layout; the layout of this repository is in README.md
  cdm/
    schema.py          # Response, Case dataclasses; validation
    primitives.py      # S, C, PS, PSV, avg, popsd ("Var"), entropy2, tie policy
    aggregators.py     # MR, WC, HAC, SP, DA(+ablation subsets)
    partition.py       # floor(N/30) disjoint partition, halves fallback, seeded
    features/cdmcr.py  # 27 features
    features/rcr.py    # 14 features
    features/acr.py    # 10 features
    data/worldA/{martinie,rutchick,lee}.py, build_worldA.py
    data/worldB/load_mturk.py, synth_worldB.py (stand-in, same schema), build_worldB.py
    models/multilabel.py  # BR/CC/LP x KNN/LR/RF(+BNB) factories
    models/ensemble.py    # soft vote LDA/RF/XGB/LR/KNN, BagKNN, SVM
    approaches/{amp,dap,rp,ap}.py
    eval/{nested_cv,loo,logo,ablation,metalearn,bagging,stats,shap_analysis}.py
    report/tables.py      # render each thesis table next to the reproduced table
  config/defaults.yaml    # every OQ default as a switch
  tests/
```

### 9.3 Ordered steps (reordered: AMP/DAP first)
Each step is small and independently verifiable. "Acceptance" cites the paper evidence. Steps marked ⚠ depend on an open question named in the row.

**Phase A — AMP/DAP (World A, public data). Highest priority.**

| # | Goal | In → Out | Acceptance check (citation) |
|---|---|---|---|
| A0 | Skeleton, config with every OQ switch, seeding, pinned env | — | Config lists the §8 defaults verbatim |
| A1 | Data schema + validation | raw dict → Case | Rejects c∉[0,1], v∉A, ps length ≠ \|A\| (TH Table 3.1) |
| A2 | Primitives (S, C, PS, PSV, popSD, entropy base 2, tie policy) | Case → numbers | S=(67,23)/90 → ES 0.820, VarS 0.244, D_S_Uniform 0.346, D 0.489 (TH A.2) |
| A3 | Stats module (no data) | counts → p | Table 5.3 to 4 s.f. from 935/937 vs 864/831/**830**/774; Table 5.4/5.5/A.10 PRE from the exact counts in T8/T11/T13 (45 values); MLA Fig 2 p-values from 422/351, 268/222, 185/149, 180/127 → 5.0e-8, 3.2e-7, 2.5e-4, .034; MLA Table 7 McNemar from the 10 pairs in T22; Table 5.4 one-sample (349/287, 244/200, 241/201, 132/93 → 4.2e-6, 4.1e-6, 4.0e-7, .062); MLA Table 7 proportion column from 975/982; ablation two-tailed p (.0023, .0102); PRE from rounded rates (.204/.211/.348/.185); McNemar χ²+cc unit cases (15,0)→3.0e-4, (34,7)→4.9e-5, (11,2)→.027, (9,1)→.027 |
| A4 | Aggregators MR/WC/HAC/SP + DA with method subsets | Case → answer | The two DA examples (TH p.35); binary identity: DA = MR iff MR is the lone dissenter, else ¬MR; DA recomputes under ablation subsets (TH p.48) ⚠OQ8 ties |
| A5 | Partition | N, seed → index sets | 90→3×30; 121→{30,30,30,31}; N<30 → two halves (TH p.27); 30≤N<60 → two halves (OQ6 default); the 22 \|R_P\| of TH Table 4.1 give 77 sub-sets with the printed average sizes ⚠OQ6 |
| A6 | CDMCR features 1–17 | Case → 17 | A synthetic N=90 case matching A.2's free parameters reproduces A.2 features 1–17 to 3 d.p. |
| A7 | CDMCR SG features 18–27 | Case + partition → 10 | The (21,23,23) case gives .031/.067/.098 and SG_B_Amax=0 (A.2). Feature 22 is not an oracle ⚠OQ12 |
| A8 | World-A loaders (one per source) = Steps D1–D2 | raw files → Responses | **Needs the D0 files under the data root.** First deliverable: a file-inspection note (sheets, columns, value ranges, the `Data Revisions.docx` log, whether Martinie probabilities are whole numbers) before any loader code. Raw counts: Martinie 500 items / 459 workers; Rutchick 216/110/465; (Lee 256). Sanity: the authors' own per-game results in the "Analysis" files if present, else the published accuracies: S1 SP 111, dem. 116.5, CW 120 of 216; S2 87/87.5/88 of 110; S3 306.5/304.5/313 of 465 (ties 0.5). REVISED files may differ slightly from the published totals; log any gap ⚠OQ2/OQ3/OQ44 |
| A9 | World-A build (Steps D3, D4, D6): filters, normalization, order, freeze | → 1209-case frozen file | 500/393/212/104; N 70±35.5 [10,129]; correct 62%±23.8 [2.7, 98.9] (TH p.43) ⚠OQ1/OQ4/OQ16 |
| A10 | **Rule gate** (Step D5) | frozen data → rates | **T3b per domain** (20 counts, ±1); T3 864/831/830/774/372; T9 union 1007, DA-only 202, DA∩rules 170. On failure, only the ordered switch ladder of Step D5, each run logged; if the ladder is exhausted, stop and report to the project owner. No ML before a pass or the project owner's written authorisation (AM17) |
| A11 | AMP/DAP instance builder + ablation masks | → X, Y | Shapes 1209×27 (AMP), 1209×32 (DAP); label means = T3; the §5.4 matrix encoded; groups per TH p.47 ⚠OQ5/OQ41 |
| A12 | Multi-label factories + AMP protocol | X, Y → model | BR+KNN ≡ LP+KNN choices on random data (Table 5.1 fingerprint); argmax tie order ⚠OQ17 |
| A13 | MLA LOO evaluation (BR+RF / RF) | → T19–T22 | AMP ≈975/1209, DAP ≈982 (±25); selection counts ≈ 422/268/185/180/154 (±3 pp each); P(succ\|sel) ≈ .832/.828/.805/.706/.818 (MLA Figs 1–3); McNemar pairs compared with T22 ⚠OQ5 (DAP under encodings (a) and (c)) |
| A14 | Nested CV (10-fold unshuffled × inner LOO, 9 techniques) | → Tables 5.1/5.2, T1/T2 | Selection logic unit test: fed the 10×9 numbers of TH Table 5.1, it returns BR+RF in folds 1–5, 7, 8, 10 and CC+RF in folds 6 and 9; fed Table 5.2, RF in all folds. Own run: inner scores ±0.02; an RF technique wins every fold (which of BR/CC/LP is not a pass criterion: fold 6 was decided by 1 case in 1088); BR+KNN ≡ LP+KNN in every fold; fold-test pattern (high 1–7, drop 8–10); AMP ≈935, DAP ≈937 (±25) ⚠OQ15/OQ16 |
| A15 | AMP effectiveness + per-domain (10-fold and LOO) | → Tables 5.4, 5.5, A.9, A.10 | ±0.03 on rates; reproduced tables shown next to the back-solved counts of T8/T11/T12/T13; PRE from exact counts; HAC the only non-significant row of 5.4; NBA the strongest PRE, NFL negative PREs |
| A16 | Ablations (LOO) | → Tables 5.6, 5.7 | AMP rows 8 and 11 significantly below row 1, others not; DAP none significant ⚠OQ41 |
| A17 | Classic meta-learning (DecisionTree) | → Table 5.8 | Row 4 ≈ .687 (= MR, expected); rows 1 and 5 equal; best < SP .715 ⚠OQ21 |
| A18 | Bagging | → Tables 5.9, 5.10 | Size formula unit test: printed form with Z=1.645 gives 35 ≤ size ≤ 42 (ceil); the Cochran option gives 35 ≤ size ≤ 68; both ≤ 70 (TH p.50). Train/test separation across problems (assert); labels recomputed; BR+RF / RF in all folds; "–" row = Table 5.1/5.2 test column ⚠OQ22/OQ42 |
| A19 | SHAP for CDMCR | → Figs 5.4–5.7, A.1 | Top feature for MR/DA/SP = SG_B_Amax; HAC top = CAmin; MR vs DA mirror image (TH pp.78–79) ⚠OQ30 |
| A20 | Report Phase A | → results_worldA.md | Every Phase-A T-ID shown as thesis-reported vs this implementation |

**Phase B — RP/AP (World B).** Starts on synthetic data and switches to real data when the real file is available.

| # | Goal | In → Out | Acceptance check (citation) |
|---|---|---|---|
| B1 | RCR features | Case → 14 per response | A synthetic sub-set matching A.4 (S=8/33, ps[v]=1, IG/OG means) gives ChiSq .610, PChiSq .682, PChiSq_IG .255, PChiSq_OG .887, ConfD .203/.175/.212, dPSv −.758, PSvD .811, PSvD_IG .406; PSvD_OG = .940 (D12) |
| B2 | ACR features | Case → 10 per answer | The same sub-set gives A.6 (.242, .189, .594, .06, .053, −.352, .182, .825) |
| B3 | Synthetic World-B generator (§2.3) | config → Responses (loader schema) | Table 4.1 reproduced exactly (\|A\|, \|R\|, solver counts); 77 sub-sets with the listed average sizes; Σps=1; rule rates over 30 seeds within ±2 of MR 34, WC 34, SP 33, HAC 22. Every parameter is tagged EVIDENCE/INFERRED/INVENTED ⚠OQ43 |
| B4 | World-B loader for the real file + partition | real file → 77 Cases | Same schema as B3; Table 4.1 per-task stats; 2490 or 2514 rows logged (OQ34) |
| B5 | World-B rule baselines | → T24 | Real data: MR .442, WC .442, SP .429, HAC .286 (±1 case per seed) ⚠OQ25 |
| B6 | RP pipeline (instances, soft vote, LOGO, 9 strategies) | → Table 4.2, A.7 | Real data: prop+avgp ≈ .64 best; tie rate 16–18% for maj/prop. Synthetic: runs end to end only ⚠OQ26/OQ27 |
| B7 | AP pipeline | → T23, A.8 | AP ≈ .545 (real); assert three-case rule ≡ argmax P(True) |
| B8 | Alternatives Bag-KNN, SVM | → A.7/A.8 | SVM ≈ .43–.45 with ~91% ties (real) ⚠OQ35 |
| B9 | World-B significance | → Table 5.11 | McNemar χ²+cc and one-tailed pooled proportion; same significance pattern (AP vs MR/WC/SP n.s. on the proportion test) |
| B10 | SHAP for RCR/ACR | → Figs 5.8, 5.9 | Top-3 RCR features PS-based (dPSv first); top ACR dPS_OG (real data only) |
| B11 | Report | → results.md | Every T-ID side by side; synthetic runs labelled "SYNTHETIC — not a reproduction" |

Dependency gates:
- A0–A7 need no data.
- A8 onwards need the World-A files. A8 starts with an inspection of the files.
- A10 must pass, or the project owner must authorise proceeding (AM17), before A11–A19 are **run on the real data**. Their code and unit tests may be written earlier on toy cases.
- B1–B3 need no external data.
- B4–B10 "real" acceptance needs the real MTurk file.

---

## 10. Inputs that would settle open questions
1. The OQ defaults in §8 are provisional, especially OQ1/OQ1b (NFL = Rutchick S1, Lee compared at the rule-baseline check), OQ2, OQ5 (DAP answer coding), OQ6, OQ16, OQ22, OQ32, OQ41 and OQ43 (synthetic World-B parameters are invented placeholders).
2. The World-B MTurk raw data (2514 or 2490 responses); if possible, also the original 77-subset assignment and the confidence scale used on the form.
3. The thesis's original code, merged World-A file, or analysis workbooks. Any of these would settle OQ1–OQ5, OQ8 and OQ15.
4. The World-A files (§2.1.1 D0b; docs/DATA.md), including the Lee 2018 OSF data (`osf.io/3kjmu`), so that the NFL check can compare both sources (OQ1b, D21).
5. Two facts that the sources do not state: whether Lee 2018 data is in the final 1209 cases, and how confidence was normalized for the Martinie items.

## 11. Main sources of uncertainty
1. **World-B data is private** (MCF: "available on request"). RP/AP (T23–T29) cannot be reproduced without the real file. The response count is 2514 in TH/MCF and 2490 in Table 4.1 (D9).
2. **World-A cleaning is under-specified** (sources→domains, NFL source, confidence normalization, filters). Every AMP/DAP number depends on it. Step A10's rule gate (exact per-domain T3b, then T3) is the check; see docs/RULE_BASELINE_CHECK.md for its outcome.
3. **Random partitions and seeds** for the SG features and the 77 sub-sets are unknown. With only 77 test cases, one case = 1.3 pp. Expect ±2–3 pp noise and report means over seeds.
4. **Hyperparameters, library versions and scaling** are unstated (sklearn defaults assumed). KNN/LR results and model-selection picks may shift. The thesis does not name the model explained by SHAP (OQ30), so rank orders may differ.
5. **Protocol ambiguity:** two different AMP/DAP headline protocols (10-fold nested 0.773/0.775 vs LOO 0.806/0.812). The fold order is inferred, not stated. Results depend on the unshuffled domain-ordered folds: a shuffled CV would give much flatter per-fold scores.

---

## Appendix A. Amendments AM1–AM19

Changes to the plan that were found necessary during implementation. Each is in
use in the code unless marked otherwise. "Open" means that the sources do not
settle the point and the stated rule is a default of this implementation.

| # | Concerns | Amendment | Basis | Status |
|---|---|---|---|---|
| AM1 | §5.8 item 1 (Table 5.3) | The printed p-values of Table 5.3 equal 1 − Φ(z) evaluated in double precision, not the exact upper tail. The exact tail is reported as the p-value and 1 − Φ(z) is shown beside it when a printed cell is compared. | Matches all 8 cells of Table 5.3 to 6 s.f. and all nine printed zeros | in use |
| AM2 | §5.4 (ablations) | The ablation p-values .0023 / .0102 of the first plan version are withdrawn (they were derived from 974 correct cases; T19 gives 975). The thesis prints no ablation p-values (TH pp.70–71); only the significance pattern (rows 8 and 11) is compared. | TH pp.70–71; recomputation | in use |
| AM3 | OQ8 (ties) | An absolute tolerance (1e-9) is used when ties between WC / HAC / SP scores are detected, so that true ties are not decided by rounding noise. | 6962 of 8385 exact SP ties (N ≤ 130) are not detected by exact float equality | in use |
| AM4 | OQ7 | On a full vote tie a_max == a_min (literal OQ7 default), so CAmin and CAmax use the same supporters. | OQ7 | in use; open |
| AM5 | OQ2, OQ4 (Martinie) | Martinie confidence: option (c) with all raw responses kept (no vote-inconsistency exclusion), in place of default (a) with Martinie's exclusions. | Matches TH Table A.2 on Martinie item 1 (AvgC .7487, CAmax .7879, CAmin .6343, MinC .05, 11/90, 8/90, thirds 21/23/23); probabilities are whole numbers, so (a) / (b) cannot give MinC .05; with the exclusions 499 items remain, not 500 | in use (scenario switches) |
| AM6 | OQ1 (NFL source) | NFL source = Rutchick Study 1 with complete (vote, confidence, others%) triples. | Thesis max N = 129 equals the S1 maximum under complete triples; Lee is fixed at 100 per game and its folder holds no game outcomes | in use |
| AM7 | Step D3 (case counts) | The stated 0% / 100% filter gives 500 / 396 / 216 / 110 cases; the thesis reports 500 / 393 / 212 / 104. Candidate selection rules that give the thesis-reported counts (hypotheses, not established): 4 NFL games and 6 election races with differing column headers dropped; 3 NBA games with 1/61 correct dropped. | Counts; thesis minimum correct rate 2.7% = 1/37 | in use as labelled scenario switches |
| AM8 | §2.1.1 D0b, OQ44 | File inventory: the original and the REVISED Rutchick files and the Lee 2018 archive are all present. REVISED differs in the NFL confidence coding of 25 games (affects WC / HAC only), one NBA vote and one Elections header. | File comparison | recorded; open |
| AM9 | features 16 / 17 | The AM3 tolerance is also used in the comparisons with AvgC and AvgPSV in features 16 / 17. | Without it the feature value depends on the order of the responses (3/20 against 10/20 for the same 20 responses) | in use |
| AM10 | OQ7 (sub-sets) | Sub-set vote ties are resolved canonical-first, so SG_B_Amax depends on the listing order of the two answers in about 5% of simulated cases. | Simulation with the statistics of TH p.43 | in use; open |
| AM11 | feature 12 | TH Table A.1 words feature 12 as "a_max has the highest average confidence" (as MLA); Table 3.2 uses f_HAC == a_max. The two differ only under HAC ties. The code follows Table 3.2. | TH Table A.1 p.109; Table 3.2 p.28 | in use |
| AM12 | OQ7 (sub-sets) | Empty supporter set in a sub-set: the code uses the sub-set's own AvgC. A rule for the CAmin of a unanimous bagged sub-group is given by AM19. | OQ7 | in use; open |
| AM13 | §3.3.1 (sub-sets) | Sub-set features: Table A.2 features 22–27 are not matched by consecutive file-order thirds or by any deterministic variant tried; random disjoint partitions match 23–27 (not feature 22 jointly). The seeds are not known, so features 18–27 cannot be matched value for value. | 20,000 seeds × 6 shuffle idioms; 11.3% of random 3 × 30 partitions give (21, 23, 23) | recorded |
| AM14 | Step D3 (case counts) | No 1209-case build is established as the thesis data set: each gives SDs 35.6 / 23.7 where the thesis prints 35.5 / 23.8, while the stated-filters build of 1222 cases matches the printed SDs. The sources state no rule that gives the NBA count 393; for Politics 104 there are two candidate rules (differing headers; races with ≥ 6 third-option votes). Scenarios are labelled "hypothesis". | Two independent parsers agree on all 1209 cases; random search | recorded |
| AM15 | §2.1 (Politics) | Politics is not binary in the raw data (46 of 110 races offer 3 or 4 options). Rutchick's own analysis uses the arg-max over options and, in SP, 100 − x for each other option. `waiver_main` treats Politics as binary (votes for options 3–4 dropped). | Rutchick analysis workbooks | in use as labelled scenario switches; open |
| AM16 | OQ5 (NFL) | NFL: "home = second-named team" and the 1–7 reading of the original confidence coding are placeholders without evidence in the sources. | — | open |
| AM17 | Step D5 / A10 | On every build some rule-baseline counts differ from the thesis-reported counts. The project owner authorised proceeding with AMP / DAP on the build `waiver_main` (sensitivity build `waiver_politics_alt`) with the differences documented. | docs/RULE_BASELINE_CHECK.md; config/gate_status.yaml | decided 2026-11-11 |
| AM18 | OQ15, §5.2 | LR `max_iter` 10000 with a count of non-converged fits; technique tie in the nested CV → first in table order (the outcome under each tied technique is also reported); RF seed 0, with seeds 1–4 for seed-to-seed variation. | Plan gives no number / no rule | in use (defaults of this implementation) |
| AM19 | §5.6 (bagging) | A bagged sub-group in which all responses give the same answer has no CAmin. Rule in use: CAmin := the sub-group's AvgC (the rule of OQ7 for sub-sets); sensitivity: drop such instances. Affects 62 of 7,680 new instances at F = 10. | docs/IMPLEMENTATION_DECISIONS.md, T12 | in use (labelled switch); open |
