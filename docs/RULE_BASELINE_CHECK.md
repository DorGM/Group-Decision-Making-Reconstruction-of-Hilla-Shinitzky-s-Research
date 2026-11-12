# Rule-baseline check

This document describes the check that is run on a frozen World-A data build
before any model is trained on it (plan §2.1.1 Step D5, step A10), what the
check gives on the builds used in this repository, and the decision under which
the model runs were made. In the code the check is called the "rule gate"
(`cdm/eval/rule_gate.py`, `cdm/eval/gate_guard.py`, `config/gate_status.yaml`).

## What the check compares

The five rule-based aggregators MR, WC, HAC, SP and DA involve no model and no
random number. Given the data set, their success counts are fixed. The thesis
prints the corresponding success rates (Fig. 5.1, Table 5.4; per domain
Table 5.5 = Table A.10); the counts are derived from them (plan T3, T3b).

The check recomputes the same counts on a frozen build and compares them with
the thesis-reported counts:

| Block | Compared values | Tolerance |
|---|---|---|
| N | number of cases per domain (500 / 393 / 212 / 104) | exact |
| T3b | success count of each rule in each domain (5 rules × 4 domains = 20 cells) | ±1 case per cell |
| T3 | success count of each rule over all 1209 cases | exact |
| T9 | cases solved by at least one of MR / WC / HAC / SP; cases solved by DA only; cases solved by DA and at least one rule | exact |

The check is met ("passed") when every cell is within its tolerance.

## Result on the builds used here

Build `waiver_main` (1209 cases; tie rule `canonical_first`, tie tolerance 1e-9,
DA tie rule = thesis text). Cells: this implementation / thesis-reported /
difference.

| Rule | General (500) | NBA (393) | NFL (212) | Politics (104) | All (1209) |
|---|---|---|---|---|---|
| SP | 413 / 413 / 0 | 260 / 258 / +2 | 112 / 109 / +3 | 84 / 84 / 0 | 869 / 864 / +5 |
| MR | 385 / 385 / 0 | 256 / 252 / +4 | 114 / 111 / +3 | 83 / 83 / 0 | 838 / 831 / +7 |
| WC | 380 / 380 / 0 | 252 / 253 / −1 | 117 / 114 / +3 | 82 / 83 / −1 | 831 / 830 / +1 |
| HAC | 338 / 338 / 0 | 230 / 237 / −7 | 131 / 131 / 0 | 68 / 68 / 0 | 767 / 774 / −7 |
| DA | 111 / 108 / +3 | 137 / 142 / −5 | 99 / 101 / −2 | 21 / 21 / 0 | 368 / 372 / −4 |

T9 on `waiver_main`: union of the four rules 1006 / 1007 / −1; DA only
203 / 202 / +1; DA and at least one rule 165 / 170 / −5.

Of the 20 per-domain cells, 9 are equal, 2 differ by one case, and 9 differ by
2 to 7 cases. The case counts
per domain equal the thesis-reported counts. The check is therefore not met on
this build: the status recorded in `config/gate_status.yaml` is `waived`, not
`passed`.

Build `waiver_politics_alt` (sensitivity build; identical to `waiver_main`
except for the selection of Politics races): General, NBA and NFL as above;
Politics SP 85 / 84 / +1, MR 84 / 83 / +1, WC 84 / 83 / +1, HAC 69 / 68 / +1,
DA 20 / 21 / −1; all cases SP 870, MR 839, WC 833, HAC 768, DA 367; T9
1007 / 202 / 165 against 1007 / 202 / 170.

The same comparison was made on every scenario of
`config/worldA_scenarios.yaml`; on each of them some cells differ
(`docs/IMPLEMENTATION_DECISIONS.md`, G1). The differences on `waiver_main` are
stated here as differences. The present reconstruction does not establish the
reason for them.

## Decision

The plan allows model runs on the real data only after the check is met or the
project owner decides otherwise. On 2026-11-11 the project owner authorised
proceeding with AMP / DAP on the frozen build `waiver_main` (primary) and on
`waiver_politics_alt` (sensitivity), with the differences documented
(plan amendment AM17). The scenario name `waiver_main` and the status value
`waived` are kept from that decision.

Conditions of the decision, as implemented:

- the build is frozen before any model is trained; a driver script refuses a
  build whose cases file does not have the recorded SHA-256
  (`config/gate_status.yaml`; `cdm.eval.gate_guard.require_gate`);
- the rule baselines are reported both as recomputed on the build and as
  printed in the thesis;
- no per-domain number is described as an exact reproduction;
- the thesis-reported value is shown beside every result of this
  implementation.

## Settings fixed together with the decision (AM18)

The plan leaves three settings without a value. The defaults of this
implementation are:

- logistic regression `max_iter` = 10000, with a count of the fits that did
  not converge (sensitivity value: 100);
- a tie between techniques in the nested-CV selection goes to the technique
  that comes first in the column order of TH Table 5.1 / 5.2; the result under
  each tied technique is also reported;
- random-forest seed 0 is the primary run; the leave-one-out run is repeated
  with seeds 1–4 to state the seed-to-seed variation.

## How to run the check

```
python -m cdm.data.worldA.build_worldA --scenario waiver_main
```

freezes the build under `build/worldA/waiver_main/` and writes the result of
the check into its `manifest.json` (`rule_gate`). The exact commands that were
run are in `docs/VERIFIED_COMMANDS.md`.
