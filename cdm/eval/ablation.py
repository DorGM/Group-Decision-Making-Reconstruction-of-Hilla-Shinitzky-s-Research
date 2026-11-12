"""Ablation experiments of AMP and DAP.

Step A16 of plan §9.3 ("Ablations (LOO)" -> TH Tables 5.6, 5.7; plan items
T14, T15). Plan §5.4 (TH §4.1.5 p.46-49, §5.1.1 p.70-72; MLA §7).

  * 11 configurations: the matrix of plan §5.4 (rendered TH p.72), held in
    cdm.approaches.instances.ABLATION_MATRIX. A row drops exactly the
    components marked 0: Confidence features, Predicted-Support features,
    {WC, HAC}, SP, DA. Voting features and O_MR / f_MR are always kept.
    Features 16 / 17: plan §8 OQ41. When WC/HAC or SP is removed, DA is
    recomputed over the methods kept (TH p.48). All of this is done by
    cdm.approaches.amp.amp_xy / dap.dap_xy (step A11).
  * Protocol: leave-one-out over all cases with BR+RF (AMP) / RF (DAP)
    (TH p.48, p.70), i.e. cdm.eval.loo.run_amp_loo / run_dap_loo. Row 1 is
    the run of step A13 (plan §5.3).
  * Significance: TWO-tailed pooled two-proportion z-test of every row
    against the full version, row 1 (TH p.70 "p > 0.05 using a two-tailed
    proportion test"; plan §5.4), n1 = n2 = number of cases. The thesis
    prints no ablation p-value (amendment AM2): what is compared is the pattern
    only: AMP rows 8 and 11 significantly below row 1, no DAP row
    significant (plan §9.3 row A16). A row counts as significant when
    p <= 0.05 (the complement of the thesis's "p > 0.05").

Nothing here reads a data build; the caller passes the instance table.
"""
from cdm.approaches import instances as I
from cdm.eval import loo as L
from cdm.eval import stats as S

ALPHA = 0.05
#: Rows the thesis reports as significantly different from row 1
#: (TH p.70-71; plan §9.3 row A16, amendment AM2).
THESIS_SIGNIFICANT_ROWS = {"AMP": (8, 11), "DAP": ()}
TECHNIQUE = {"AMP": ("BR", "RF"), "DAP": "RF"}
ROWS = tuple(sorted(I.ABLATION_MATRIX))


def thesis_rate(approach, row):
    """Printed success rate of a row (TH Table 5.6 AMP / 5.7 DAP)."""
    return I.ABLATION_THESIS[row][0 if approach == "AMP" else 1]


def run_row(inst, approach, row, hyper=None, oq41=None, config=None, n_jobs=1,
            batch_size=L.DEFAULT_BATCH_SIZE, checkpoint_dir=None):
    """LOO of one ablation row. Returns (EvalResult, description) with
    description = {row, components, methods, n_features, feature_names}.

    inst     : cdm.approaches.instances.Instances.
    approach : "AMP" (BR+RF) or "DAP" (RF).
    row      : 1..11 of plan §5.4.
    oq41     : plan §8 OQ41; default: the config value.
    hyper, n_jobs, batch_size, checkpoint_dir : see cdm.eval.loo.
    """
    if approach not in TECHNIQUE:
        raise ValueError(f"unknown approach {approach!r}")
    cfg = I.ablation_config(row)
    oq41 = I.oq41_from_config(config) if oq41 is None else oq41
    if approach == "AMP":
        inp = L.amp_inputs(inst, row, oq41, config)
        res = L.run_amp_loo(inp["X"], inp["Y"], technique=TECHNIQUE["AMP"],
                            labels=inp["labels"], case_ids=inp["case_ids"],
                            domains=inp["domains"], method_answers=inp["method_answers"],
                            hyper=hyper, config=config, n_jobs=n_jobs,
                            batch_size=batch_size, checkpoint_dir=checkpoint_dir)
    else:
        inp = L.dap_inputs(inst, row, oq41, config)
        res = L.run_dap_loo(inp["X"], inp["y"], technique=TECHNIQUE["DAP"],
                            case_ids=inp["case_ids"], domains=inp["domains"],
                            answer_sets=inp["answer_sets"],
                            rule_outcomes=inp["rule_outcomes"], hyper=hyper, config=config,
                            n_jobs=n_jobs, batch_size=batch_size,
                            checkpoint_dir=checkpoint_dir)
    kept = I.feature_numbers_kept(cfg, oq41)
    desc = {"row": cfg.row, "components": dict(zip(I.ABLATION_COLUMNS, cfg.as_tuple())),
            "methods": list(I.methods_kept(cfg)), "rules_heard_by_DA": list(I.rules_kept(cfg)),
            "n_cdmcr_features": len(kept), "cdmcr_feature_numbers": list(kept),
            "n_columns": int(inp["X"].shape[1]), "oq41": oq41}
    return res, desc


def versus_full(successes, n, full_successes, full_n=None):
    """Two-tailed pooled two-proportion z-test of a row against the full
    version (plan §5.4; TH p.70). Returns (z, p); z < 0 when the row is
    below the full version. (nan, nan) when the pooled standard error is 0
    (both runs solve every case or none)."""
    full_n = n if full_n is None else full_n
    try:
        return S.two_proportion_ztest(successes, n, full_successes, full_n,
                                      alternative="two-sided")
    except ZeroDivisionError:
        return float("nan"), float("nan")


def ablation_table(approach, counts, n, alpha=ALPHA):
    """The rows of TH Table 5.6 / 5.7 from success counts.

    counts : {row number: successes}; row 1 (the full version) must be in
             it for the tests; rows not yet run are simply absent.
    Returns a list of dicts in row order: row, components, successes, n,
    rate, rounded (half up, 3 decimals), thesis_rate, difference (rounded
    rate - thesis rate), z, p (two-tailed vs row 1; None for row 1 or
    when row 1 is missing), significant (p <= alpha), below_full,
    thesis_significant.
    """
    full = counts.get(1)
    out = []
    for row in sorted(counts):
        k = int(counts[row])
        rounded = S.round_half_up(k, n)
        r = {"row": row,
             "components": dict(zip(I.ABLATION_COLUMNS, I.ABLATION_MATRIX[row])),
             "successes": k, "n": int(n), "rate": k / n, "rounded": rounded,
             "thesis_rate": thesis_rate(approach, row),
             "difference": rounded - thesis_rate(approach, row),
             "z": None, "p": None, "significant": None, "below_full": None,
             "thesis_significant": row in THESIS_SIGNIFICANT_ROWS[approach]}
        if full is not None and row != 1:
            z, p = versus_full(k, n, full, n)
            if p == p:
                r.update(z=z, p=p, significant=bool(p <= alpha), below_full=bool(k < full))
        out.append(r)
    return out


def significance_pattern(table):
    """Summary of the acceptance check of plan §9.3 row A16 for one
    approach: the rows significantly different from row 1 in the run, the
    thesis's rows, and whether the two sets are equal (None while rows are
    missing)."""
    rows = [r for r in table if r["row"] != 1]
    run = sorted(r["row"] for r in rows if r["significant"])
    thesis = sorted(r["row"] for r in table if r["thesis_significant"])
    complete = len(table) == len(ROWS) and all(r["significant"] is not None for r in rows)
    return {"significant_rows_run": run,
            "significantly_below_full_run": sorted(
                r["row"] for r in rows if r["significant"] and r["below_full"]),
            "significant_rows_thesis": thesis,
            "same_pattern": (run == thesis) if complete else None,
            "rows_done": [r["row"] for r in table]}
