# Reconstruction workflow and source-to-implementation mapping

This page explains how the code in this repository was derived from the written sources, and what was learned about reconstructing a pipeline from its description. It describes the process honestly, including its limits.

## 1. Sources

| Short name | Source | Used for |
|---|---|---|
| TH | Hilla Shinitzky's doctoral thesis | everything; primary reference |
| Preprint | Shinitzky et al., arXiv:2204.01721v1 (2022), preprint of the 2025 *Collective Intelligence* article | AMP / DAP |
| GDN | Shinitzky et al. (2024), *Group Decision and Negotiation* 33, 87–111 | RP / AP |
| Dataset papers | Martinie et al. (2020), Rutchick et al. (2020), Lee et al. (2018) and their data archives | AMP / DAP data |

The published 2025 *Collective Intelligence* article was not consulted for this release. No code, data files or unpublished material from the original authors were available.

## 2. Principles

1. **Evidence only.** Every implemented formula, protocol and constant is either tied to a passage, table or figure of the sources or recorded as a labelled default (principle 2). Code comments cite the sources (for example "TH Table 3.2").
2. **Nothing unspecified is decided silently.** Where the sources do not state a detail, the choice is recorded in [IMPLEMENTATION_DECISIONS.md](IMPLEMENTATION_DECISIONS.md) with a tag: `EVIDENCE` (stated in a source), `INFERRED` (derived from printed numbers or examples), `INVENTED` (no support in the sources; a default was needed), or `OPEN`. Where practical, the alternative is implemented as a switch and run as a sensitivity analysis.
3. **No tuning toward reported numbers.** Algorithms, hyperparameters and data rules were not adjusted to make results agree with published values. The only calibration in the repository is that of the synthetic stand-in on the rule baselines, which is disclosed as such.
4. **Reported values stay separate.** Values taken from the sources live in dedicated modules (`cdm/report/thesis_values.py`, `cdm/eval/worldB_thesis_values.py`; a few thesis columns are kept beside their code in `cdm/eval/shap_analysis.py` and `cdm/eval/worldB_rules.py`) and appear only in columns labelled as reported.
5. **Real and synthetic outputs cannot mix.** Synthetic data and everything computed from it carry a marker, and real-data runs write to a separate folder.

## 3. Workflow

1. **Plan.** A reconstruction plan was written from the sources: definitions, feature formulas, protocols, a numbered list of reported values to compare against, and a numbered list of open questions. A second pass re-read the plan against the sources before any code was written. The plan is published as [RECONSTRUCTION_PLAN.md](RECONSTRUCTION_PLAN.md); identifiers such as "plan §5.3", "OQ16" or "T12" in code comments refer to it.
2. **Small steps.** The code was written in small units (primitives, aggregators, features, data loaders, approaches, evaluation, reporting), each with tests, in the order of their dependencies.
3. **Checks on each step.** Each unit was checked against the plan and the sources separately from the writing of it, and results were recomputed from stored per-case outputs by independent scripts. This work was carried out with the AI model Claude under the repository owner's direction; it is not independent human review.
4. **Worked examples as tests.** The thesis prints worked examples of the feature representations (Tables A.2, A.4, A.6). They serve as unit-test oracles.
5. **A baseline check before any learning.** Before any model was trained on the rebuilt data, the five rule aggregators were compared with their reported success counts per domain ([RULE_BASELINE_CHECK.md](RULE_BASELINE_CHECK.md)). MR, WC, HAC and SP agree exactly in the General domain; DA there and the rules in the other domains differ by 1 to 7 cases. The repository owner decided to proceed with the differences documented.
6. **Evaluation and reporting.** All protocols were run once with fixed seeds, and additional seeds were run to show the spread. Results are reported beside the reported values without adjustment.

## 4. Source-to-implementation mapping

| Topic | Source location | Implementation |
|---|---|---|
| Response, vote, confidence, predicted support | TH ch. 3 | `cdm/schema.py`, `cdm/primitives.py` |
| MR, WC, HAC, SP | TH ch. 2–3 | `cdm/aggregators.py` |
| Devil's Advocate aggregator | TH ch. 3 | `cdm/aggregators.py` |
| CDMCR, 27 features | TH Table 3.2; example Table A.2 | `cdm/features/cdmcr.py` |
| Sub-set partition | TH ch. 3–4 | `cdm/partition.py` |
| Public datasets and case selection | TH ch. 4 (data description) | `cdm/data/worldA/` |
| AMP | TH §3.3, §4.1; Preprint | `cdm/approaches/amp.py`, `cdm/models/multilabel.py` |
| DAP | TH §3.3, §4.1; Preprint | `cdm/approaches/dap.py` |
| 10-fold with inner leave-one-out | TH Tables 5.1, 5.2, Fig. 5.1 | `cdm/eval/nested_cv.py` |
| Single leave-one-out | Preprint; TH Tables 5.6, 5.7, A.9, A.10 | `cdm/eval/loo.py` |
| Effectiveness and PRE | TH Tables 5.4, 5.5 | `cdm/eval/effectiveness.py` |
| Ablations | TH Tables 5.6, 5.7 | `cdm/eval/ablation.py` |
| Classic meta-learning | TH Table 5.8 | `cdm/eval/metalearn.py` |
| Bagging | TH §4.1.7, Tables 5.9, 5.10 | `cdm/eval/bagging.py` |
| SHAP for CDMCR | TH Figs 5.4–5.7, A.1 | `cdm/eval/shap_analysis.py` |
| RCR, 14 features | TH Table 3.3; example Table A.4; GDN Table 1 | `cdm/features/rcr.py` |
| ACR, 10 features | TH Table 3.4; example Table A.6; GDN Table 2 | `cdm/features/acr.py` |
| Task list for RP / AP | TH Table 4.1; GDN Table 3 | `cdm/data/worldB/table41.py` |
| RP and its nine strategies | TH §3.3.2, §4.2.4, Table 4.2; GDN §3–4 | `cdm/approaches/rp.py` |
| AP and the three-case rule | TH §3.3.2, §4.2.4; GDN | `cdm/approaches/ap.py` |
| Leave-one-group-out | TH p. 57; GDN §4.4 | `cdm/eval/worldB_logo.py` |
| Alternative classifiers | TH Tables A.7, A.8 | `cdm/models/worldB_classifiers.py` |
| Statistical tests | TH Tables 5.3, 5.11; GDN Table 5 | `cdm/eval/stats.py`, `cdm/eval/worldB_stats.py` |
| SHAP for RCR / ACR | TH Figs 5.8, 5.9 | `cdm/eval/worldB_shap.py` |
| Synthetic stand-in | not in the sources (this repository) | `cdm/data/worldB/synthetic.py` |

## 5. Practical insights from the reconstruction

These are observations about reconstructing a pipeline from written descriptions. They are not statements about how the original research was conducted.

- **Data construction dominates.** Most of the open questions concern how published datasets become decision cases: which responses count as complete, how confidence scales are mapped to [0, 1], which cases are excluded. Small differences here move the rule baselines by a few cases, and learned models inherit them.
- **A baseline check is a useful gate.** Comparing deterministic rule aggregators with reported counts before any learning separates "is the data the same?" from "is the model the same?".
- **Worked examples are valuable.** A single printed example of a feature vector resolved several ambiguities at once (the dispersion measure, the entropy base, the confidence mapping for one dataset).
- **Ties matter more than expected.** With binary answers and a few dozen responses, ties in votes, in confidence sums and in predicted probabilities are common. Tie rules are rarely stated and can change selection counts noticeably while changing overall success only slightly.
- **Seeds and near-ties in model selection.** When candidate techniques differ by one or two cases in an inner comparison, the selected technique can change with the random seed. Totals are more stable than the list of selected techniques.
- **Protocols must be named with every number.** The same approach evaluated by a single leave-one-out and by leave-one-out inside 10 folds gives different headline values, in the sources and here. Every table in this repository states its protocol and the source of the value it is placed beside.
- **A synthetic stand-in is good for plumbing and nothing else.** It allowed the RP / AP code path to be built and tested before data is available. Its outcome levels depend on invented assumptions and say nothing about the original findings.
- **What was not verified.** Agreement with the original implementation cannot be assessed without the original code or intermediate data. Where results differ from reported values, this repository reports the difference and does not establish its reason.
