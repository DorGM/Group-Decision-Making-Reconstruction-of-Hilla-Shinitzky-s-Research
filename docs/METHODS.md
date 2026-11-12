# Methods and feature representations

This page summarises what the code implements. The methods are the work of Hilla Shinitzky and her coauthors; the authoritative descriptions are the thesis and the two publications listed in the [README](../README.md#references). Section and table numbers below refer to the thesis ("TH"). Where the sources leave a detail open, the choice made here is recorded in [IMPLEMENTATION_DECISIONS.md](IMPLEMENTATION_DECISIONS.md).

## 1. Setting

A *decision case* is one question answered once by a group ("one-shot": no history of the participants is used). Each participant gives a *response* with three parts:

- a **vote** for one of the answer options,
- a **confidence** in that vote, scaled to [0, 1],
- a **prediction of the others' votes**: the share of the group the participant expects to choose each option.

The goal is to choose the group's answer from these responses alone.

## 2. Rule aggregators (`cdm/aggregators.py`)

| Name | Rule |
|---|---|
| MR, majority rule | the answer with the most votes |
| WC, weighted confidence | the answer with the largest sum of confidences |
| HAC, highest average confidence | the answer whose supporters have the highest mean confidence |
| SP, surprisingly popular | the answer whose actual vote share most exceeds its predicted share |
| DA, Devil's Advocate | introduced in the thesis: the answer chosen least often by MR, WC, HAC and SP; on a tie, the answer not chosen by MR |

Tie handling for each rule is documented in the implementation decisions.

## 3. Meta-aggregation: AMP and DAP (binary cases)

**CDMCR, 27 features per decision case** (`cdm/features/cdmcr.py`, TH Table 3.2):

| Group | Features |
|---|---|
| Votes | `N`, `D_Smax_Smin`, `ES` (entropy of the vote shares), `VarS`, `D_S_Uniform` |
| Confidence | `CAmin`, `CAmax`, `MinC`, `AvgC`, `VarC`, `D_MaxC_AvgC`, `B_Amax_MaxCa` |
| Predictions of others | `MaxPSa`, `MinPSa`, `AvgPSV`, `P_lowC_highPSv`, `P_lowPSv_highC` |
| Sub-groups (10 features) | the responses are split into sub-sets and the features describe agreement and spread across them: `SG_B_Amax`, `SG_VarSAmax`, `SG_D_MaxVarS_MinVarS`, `SG_D_MaxES_MinES`, `SG_D_MaxAvgC_MinAvgC`, `SG_D_MaxVarC_MinVarC`, `SG_VarCAmin`, `SG_VarCAmax`, `SG_AvgCAmin`, `SG_AvgCAmax` |

**AMP, Aggregation-Method-Prediction** (`cdm/approaches/amp.py`). A multi-label classifier learns, from the 27 features, which of the five aggregators returns the correct answer for a case. For a new case the aggregator with the highest predicted probability of success is applied. Multi-label schemes: binary relevance (BR), classifier chains (CC), label powerset (LP), each with KNN, logistic regression (LR) and random forest (RF).

**DAP, Direct-Answer-Prediction** (`cdm/approaches/dap.py`). A classifier predicts the correct answer directly from the 27 features plus the five aggregators' answers.

**Evaluation** (`cdm/eval/`):

| Analysis | Protocol in the sources | Code |
|---|---|---|
| Headline, thesis | 10 folds; inside each training part a leave-one-out comparison selects the technique, which is then applied to the held-out fold | `nested_cv.py`, `scripts/run_a14_nested.py` |
| Headline, preprint; thesis Tables 5.6, 5.7, A.9, A.10 | single leave-one-out, AMP with BR+RF, DAP with RF | `loo.py`, `scripts/run_a13_loo.py` |
| Effectiveness of AMP's selection | P(selected), P(success given selected), PRE, one-sample tests | `effectiveness.py`, `scripts/run_a15_tables.py` |
| Ablations | 11 feature / label subsets | `ablation.py`, `scripts/run_a16_ablation.py` |
| Classic meta-learning | decision tree on the aggregators' answers only | `metalearn.py`, `scripts/run_a17_metalearn.py` |
| Bagging | additional training instances from sampled sub-groups | `bagging.py`, `scripts/run_a18_bagging.py` |
| Feature importance | SHAP | `shap_analysis.py`, `scripts/run_a19_shap.py` |
| Statistics | two-proportion z-tests, McNemar test with continuity correction, PRE | `stats.py` |

## 4. Response and answer prediction: RP and AP (multiple-choice cases)

Each task's responses are split into sub-sets of about 30 responses; each sub-set is treated as one decision problem.

**RCR, 14 features per response** (`cdm/features/rcr.py`, TH Table 3.3): `ChosenMajAns`, `Conf`, `ConfD`, `ConfD_IG`, `ConfD_OG`, `PSv`, `dPSv`, `PSvD`, `PSvD_IG`, `PSvD_OG`, `ChiSq`, `PChiSq`, `PChiSq_IG`, `PChiSq_OG`. "IG" and "OG" compare the response with those who chose the same answer (in-group) and with the rest (out-group).

**ACR, 10 features per answer** (`cdm/features/acr.py`, TH Table 3.4): `IsMajority`, `Support`, `AvgPS`, `AvgPS_IG`, `AvgPS_OG`, `dPS`, `dPS_IG`, `dPS_OG`, `AvgChiSq`, `AvgConf`.

**RP, Response-Prediction** (`cdm/approaches/rp.py`). A classifier predicts for every response whether its vote is correct. The answer of the sub-set is chosen from these per-response predictions by one of nine classification-aggregation strategies (maximum probability, average probability, weighted mean, and majority or proportion of positive classifications with each of the three as tie breaker).

**AP, Answer-Prediction** (`cdm/approaches/ap.py`). A classifier predicts for every voted answer whether it is the correct one; the answer is chosen by the three-case rule described in the thesis.

**Classifiers.** A soft-voting ensemble (LDA, RF, XGBoost, LR, KNN); bagged KNN and SVM as alternatives (`cdm/models/worldB_classifiers.py`).

**Evaluation.** Leave-one-group-out, where a group is the set of instances generated from one sub-set (`cdm/eval/worldB_logo.py`); McNemar and one-tailed proportion tests against the rule aggregators (`cdm/eval/worldB_stats.py`); SHAP (`cdm/eval/worldB_shap.py`).

In this repository RP and AP have been run on **synthetic stand-in data only**. See [DATA.md](DATA.md) and [RESULTS_RP_AP_SYNTHETIC.md](RESULTS_RP_AP_SYNTHETIC.md).
