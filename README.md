# Group Decision-Making: Reconstruction of Hilla Shinitzky’s Research

An independent reconstruction, in Python, of the machine-learning pipelines for one-shot group-decision aggregation developed by **Hilla Shinitzky** in her doctoral thesis *“Optimization of Collective Decision-Making using Data-Science Methods and Cognitive Features”*, supervised by **Prof. Yuval Shahar** at **Ben-Gurion University of the Negev**.

> **Status at a glance**
>
> | Part | Data used here | What the results in this repository are |
> |---|---|---|
> | AMP / DAP (meta-aggregation) | Publicly available datasets, rebuilt under documented selection and preprocessing assumptions | An available-data reconstruction. Some values agree with the thesis within a few cases and others differ. See [results](docs/RESULTS_AMP_DAP.md). |
> | RP / AP (response and answer prediction) | **SYNTHETIC stand-in data.** The original participant data has not yet been supplied. | An end-to-end pipeline check only. **Not a reproduction** of the original empirical findings. See [results](docs/RESULTS_RP_AP_SYNTHETIC.md). |
>
> This is **not** the original authors' code base. Hilla Shinitzky has not reviewed or endorsed this implementation.

## Attribution

**Original research.** The methods implemented here are the scientific contributions of Hilla Shinitzky and her coauthors: the meta-aggregation approaches AMP (Aggregation-Method-Prediction) and DAP (Direct-Answer-Prediction), the Devil's Advocate aggregator, the collective-decision feature representation (CDMCR), the Response-Prediction (RP) and Answer-Prediction (AP) approaches with their meta-cognitive feature representations (RCR, ACR), and the evaluation designs. They are described in the thesis and in these publications:

- Hilla Shinitzky, Ortal Parpara, Michal Ezrets, Raz Klein, and Yuval Shahar (2025). “The meta-aggregation approaches: Correctly choosing how to choose correctly in groups.” *Collective Intelligence*. https://doi.org/10.1177/26339137251315410 (relevant to AMP and DAP).
- Hilla Shinitzky, Dan Avraham, Yizhak Vaisman, Yakir Tsizer, Yaniv Leedon, and Yuval Shahar (2024). “Exploiting Meta-cognitive Features for a Machine-Learning-Based One-Shot Group-Decision Aggregation.” *Group Decision and Negotiation*, 33, 87–111. https://doi.org/10.1007/s10726-023-09855-9 (first published online in 2023; relevant to RP and AP).

**This repository.** The contribution of **Dor Meir** is the reconstruction: implementing the pipeline from the written descriptions, documenting every implementation decision the sources leave open, evaluating the reconstruction on the data that is available, and developing a synthetic stand-in for the data that is not. Hilla Shinitzky's research provides the methodological foundation for Dor Meir's own research, and this repository is meant as a reproducible base for it.

**AI assistance.** The implementation, the test suite and the internal consistency checks were produced with the assistance of Claude, an AI model made by Anthropic, working under Dor Meir's direction. Checks described in this repository as "recomputed" or "verified" are automated or AI-assisted checks (independent recomputation from stored per-case files, refits, unit tests). They are **not** independent human review. The code has not undergone independent human review.

## What is implemented

| Component | Description | Where |
|---|---|---|
| Rule aggregators | Majority rule (MR), weighted confidence (WC), highest average confidence (HAC), surprisingly popular (SP), Devil's Advocate (DA) | `cdm/aggregators.py` |
| Feature representations | CDMCR (27 features per decision case), RCR (14 per response), ACR (10 per answer) | `cdm/features/` |
| AMP | Multi-label prediction of which aggregator will succeed (BR / CC / LP with KNN, LR, RF) | `cdm/approaches/amp.py`, `cdm/models/multilabel.py` |
| DAP | Direct prediction of the correct answer from features and aggregator outputs | `cdm/approaches/dap.py` |
| RP / AP | Response- and answer-level classification with aggregation strategies, leave-one-group-out | `cdm/approaches/rp.py`, `cdm/approaches/ap.py` |
| Evaluation | Leave-one-out, leave-one-out inside 10 folds with model selection, ablations, classic meta-learning, bagging, statistical tests, SHAP | `cdm/eval/` |
| Data | Loaders and case construction for the public datasets; synthetic generator and a loader for the original RP / AP data format (untested on real data) | `cdm/data/` |

An overview of the methods and features is in [docs/METHODS.md](docs/METHODS.md).

## Results in brief

### AMP / DAP on publicly available data

1,209 binary decision cases (General knowledge 500, NBA 393, NFL 212, Politics 104), rebuilt from the public sources named in the thesis under the assumptions listed in [docs/DATA.md](docs/DATA.md).

| Quantity | This reconstruction | Reported | Source of reported value |
|---|---|---|---|
| AMP, leave-one-out inside 10 folds | 851 / 1209 = 0.704 | 0.773 | Thesis |
| DAP, leave-one-out inside 10 folds | 853 / 1209 = 0.706 | 0.775 | Thesis |
| AMP, single leave-one-out (5 seeds) | 850–863 / 1209 = 0.703–0.714 | 0.806 | Thesis Table 5.6; preprint |
| DAP, single leave-one-out (5 seeds) | 857–868 / 1209 = 0.709–0.718 | 0.812 | Thesis Table 5.7; preprint |
| Best rule aggregator (SP) on all cases | 869 / 1209 = 0.719 | 0.715 | Thesis |

Under the documented reconstruction configuration, AMP achieved 0.704 and DAP 0.706 in the thesis's evaluation protocol; the thesis reports 0.773 and 0.775. These values differ, and the present reconstruction does not establish the reason. By domain, the single leave-one-out results (classifier seed 0) are within 0 to 2 cases of the thesis-reported values in General knowledge, NFL and Politics (up to 9 cases across five seeds) and differ in NBA (about 0.63 here; 0.92 reported). Under the thesis's 10-fold protocol, AMP differs by +2 (General), −113 (NBA), +20 (NFL) and +7 (Politics) cases. Full tables, including every protocol, are in [docs/RESULTS_AMP_DAP.md](docs/RESULTS_AMP_DAP.md).

The thesis and the 2022 preprint use different evaluation protocols for the headline numbers (thesis: leave-one-out inside 10-fold cross-validation with model selection; the 2022 preprint of the AMP / DAP article: a single leave-one-out). Reported values are therefore always labelled with their source and protocol. The AMP / DAP part of this reconstruction was written from the thesis and from the preprint version of the article (arXiv:2204.01721v1). The published 2025 article was not consulted for this release, so its values are not compared here.

### RP / AP on a synthetic stand-in

**SYNTHETIC — not a reproduction.** The RP / AP study used responses collected by the original authors, which are not public. Until that data is supplied, the RP / AP pipeline is exercised on generated data:

- **What is generated:** 22 multiple-choice tasks with the published number of answer options, number of responses and share of correct responses per task (thesis Table 4.1), giving 2,490 responses and 77 response sub-sets. Each response has a vote, a confidence and a predicted distribution of others' votes.
- **What informed calibration:** the four rule-aggregator success counts published for the 77 sub-sets (MR, WC, SP, HAC). Agreement of the synthetic rule baselines with those counts is **by construction**, and holds for the one generated dataset that is shipped.
- **What is invented:** the response model itself (how confidence and predictions depend on the vote) and all its parameters. They are listed with their tags in `config/synthetic_worldB.yaml` and [docs/DATA.md](docs/DATA.md).

The synthetic data supports implementation and end-to-end checks. It does not reproduce the original participants, does not validate the original empirical findings, and does not establish comparable RP / AP performance. No difference or improvement is computed between synthetic outcomes and the original study. Dor Meir is awaiting the original data from Hilla Shinitzky and plans to adapt the loader and rerun the evaluation once it is received. The real-data path in this repository is **untested** until then.

## Getting started

Requires Python 3.13. Developed and run on Linux only.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # tests that need the public datasets are skipped when data/ is empty
```

The public datasets are not redistributed here. [docs/DATA.md](docs/DATA.md) explains where to obtain them and where to place them. The exact commands that were run for this release, and their scope, are in [docs/VERIFIED_COMMANDS.md](docs/VERIFIED_COMMANDS.md).

## Documentation

| Document | Content |
|---|---|
| [docs/METHODS.md](docs/METHODS.md) | Methods and feature representations, with pointers to the sources |
| [docs/RECONSTRUCTION.md](docs/RECONSTRUCTION.md) | How the reconstruction was carried out, source-to-implementation mapping, practical insights |
| [docs/DATA.md](docs/DATA.md) | Data provenance, acquisition, selection and preprocessing assumptions, synthetic generation |
| [docs/RESULTS_AMP_DAP.md](docs/RESULTS_AMP_DAP.md) | AMP / DAP results on public data, side by side with reported values |
| [docs/RESULTS_RP_AP_SYNTHETIC.md](docs/RESULTS_RP_AP_SYNTHETIC.md) | RP / AP pipeline run on synthetic data |
| [docs/IMPLEMENTATION_DECISIONS.md](docs/IMPLEMENTATION_DECISIONS.md) | Every decision, default and open question, tagged by its support in the sources |
| [docs/RECONSTRUCTION_PLAN.md](docs/RECONSTRUCTION_PLAN.md) | The reconstruction plan that code comments refer to |
| [docs/RULE_BASELINE_CHECK.md](docs/RULE_BASELINE_CHECK.md) | Rule-aggregator baselines on the rebuilt data compared with reported counts |
| [docs/RESULT_FILES.md](docs/RESULT_FILES.md) | Index of the stored result artifacts in `results/` |
| [docs/VERIFIED_COMMANDS.md](docs/VERIFIED_COMMANDS.md) | Setup and execution commands that were actually run |

## Current limitations

- The AMP / DAP reconstruction depends on dataset selection and preprocessing choices that the sources do not fully specify. The four classic rule baselines (MR, WC, HAC, SP) of the rebuilt data match the reported counts exactly in the General domain; DA there, and the rules in the other domains, differ by 1 to 7 cases ([docs/RULE_BASELINE_CHECK.md](docs/RULE_BASELINE_CHECK.md)).
- Several AMP / DAP results differ from the reported values, most visibly in the NBA domain. The reconstruction does not establish the reason.
- Hyperparameters, tie rules, random seeds and several procedural details are not stated in the sources. Defaults were chosen and are documented; other choices may give different numbers.
- RP / AP results exist only for synthetic data, and the loader for the original data format is untested.
- The published 2025 *Collective Intelligence* article was not consulted for this release; the preprint and the thesis were.
- Run and tested on Linux with the pinned package versions only.

## Planned work

Integration of the original RP / AP data once received: confirm the file format and scales, adapt the loader, rerun the RP / AP evaluation, and report it beside the values in the thesis and the 2024 article. No date is promised.

## References

Primary sources of the reconstructed methods:

1. Shinitzky, H. *Optimization of Collective Decision-Making using Data-Science Methods and Cognitive Features.* Doctoral thesis, Ben-Gurion University of the Negev. Supervisor: Prof. Yuval Shahar.
2. Shinitzky, H., Parpara, O., Ezrets, M., Klein, R., & Shahar, Y. (2025). The meta-aggregation approaches: Correctly choosing how to choose correctly in groups. *Collective Intelligence*. https://doi.org/10.1177/26339137251315410. Preprint: arXiv:2204.01721.
3. Shinitzky, H., Avraham, D., Vaisman, Y., Tsizer, Y., Leedon, Y., & Shahar, Y. (2024). Exploiting Meta-cognitive Features for a Machine-Learning-Based One-Shot Group-Decision Aggregation. *Group Decision and Negotiation*, 33, 87–111. https://doi.org/10.1007/s10726-023-09855-9

Public datasets used for AMP / DAP (see [docs/DATA.md](docs/DATA.md)):

4. Martinie, M., Wilkening, T., & Howe, P. D. L. (2020). Using meta-predictions to identify experts in the crowd when past performance is unknown. *PLOS ONE*, 15(4), e0232058.
5. Rutchick, A. M., Ross, B. J., Calvillo, D. P., & Mesick, C. C. (2020). Does the “surprisingly popular” method yield accurate crowdsourced predictions? *Cognitive Research: Principles and Implications*, 5(1). Data: https://osf.io/u9k72
6. Lee, M. D., Danileiko, I., & Vi, J. (2018). Testing the ability of the surprisingly popular method to predict NFL games. *Judgment and Decision Making*, 13(4). Data: https://osf.io/3kjmu

## License and third-party material

No third-party data, papers or thesis files are included in this repository. They remain under their own licenses and must be obtained from their sources. No license has yet been chosen for the code in this repository.
