#!/usr/bin/env python
"""A19 driver: SHAP values of the 27 CDMCR features for each AMP label
(MR, DA, SP, HAC, WC as its own binary target; TH §4.3.1, §5.3.1, Figs
5.4-5.7 and A.1; plan §5.9, OQ30, plan item T30).

  python scripts/run_a19_shap.py --build waiver_main --n-jobs 3

Two analyses, both on models trained on ALL cases, SHAP for all cases:
  xgboost_tree : XGBoost (defaults) per label, shap.TreeExplainer, log-odds
                 (plan OQ30 (i), the "faithful attempt");
  rf_kernel    : the Random Forest of the label as fitted inside BR+RF,
                 shap.KernelExplainer on P(label = 1) (plan OQ30 (ii)).

The build must be listed as passed or waived in config/gate_status.yaml
(cdm.eval.gate_guard); any other build is refused before it is loaded.

Outputs, under <runs-root>/<build>/a19/  (seed 0; otherwise a19_seed<k>):
  a19_shap.md               rank orders, run and thesis in separate columns;
                            the acceptance statements of plan §9.3 row A19
  a19_mean_abs_shap.csv     one line per model, label and feature: mean
                            |SHAP|, rank and direction of the run; the
                            thesis rank in its own column
  a19_<model>_<label>.png   beeswarm plots (max_display = 20)
  a19_shap_values.npz       the SHAP values, base values and X
  a19_summary.json, status.json, run.log
With --results-root all files are copied to <results-root>/<build>/<tag>/.

--models xgboost_tree   : skip the KernelExplainer part (about 10 minutes).
--smoke N : end-to-end test on N synthetic cases (no real data); written
            to <runs-root>/_smoke/, never to results.
--limit N : pipeline check on a sub-sample of a real build; never written
            to the results folder.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

import _common as C
from cdm.approaches import amp
from cdm.eval import shap_analysis as SH
from cdm.report import tables as R


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build", default="waiver_main")
    p.add_argument("--build-root", default=str(C.DEFAULT_BUILD_ROOT))
    p.add_argument("--seed", type=int, default=0, help="seed of the models and of KernelSHAP")
    p.add_argument("--n-jobs", type=int, default=3)
    p.add_argument("--models", default=",".join(SH.MODELS))
    p.add_argument("--kernel-background", type=int, default=50)
    p.add_argument("--kernel-nsamples", default="auto")
    p.add_argument("--runs-root", default=str(C.DEFAULT_RUNS_ROOT))
    p.add_argument("--results-root", default=str(C.DEFAULT_RESULTS_ROOT), help="folder to copy the small result files to (e.g. results); "
                        "default '' = do not copy")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--smoke", type=int, default=0, metavar="N")
    p.add_argument("--limit", type=int, default=0, metavar="N")
    return p.parse_args(argv)


DIFFERENCE_NOTE = (
    "Note: on this build SG_B_Amax is close to the bottom of every ranking below; the "
    "thesis figures list it as the top feature for MR, DA and SP, with SHAP values down to "
    "about -5. These rank orders differ, and the present reconstruction does not establish "
    "the reason. Nothing is tuned.")

#: Reading notes, written at the end of the report.
READING_NOTES = [
    "- RF + Kernel uses the shap default `l1_reg=\"num_features(10)\"`: at most 10 non-zero "
    "values per case (decision SH4).",
    "- In the direction tables, \"agrees\" for correlations with |rho| < 0.1 should be read "
    "as \"weak\".",
    "- The thesis x-axis ranges quoted here were read by eye and may be off by up to 0.4.",
]


def _fmt(x, d=3):
    return R.DASH if x is None or x != x else format(x, f".{d}f")


def ranking_md(rows, label, full):
    """Run order of a label under both models beside the thesis order."""
    by_model = {}
    for r in rows:
        if r["label"] == label:
            by_model.setdefault(r["model"], []).append(r)
    models = [m for m in SH.MODELS if m in by_model]
    orders = {m: sorted(by_model[m], key=lambda r: r["rank_run"]) for m in models}
    th = SH.THESIS_RANKING.get(label, ()) if full else ()
    head = ["rank"]
    for m in models:
        head += [f"run ({m}): feature", "mean abs SHAP", "direction"]
    head += [f"thesis ({SH.THESIS_FIGURE.get(label, R.DASH)}): feature"]
    for m in models:
        head += [f"run rank ({m}) of the thesis feature"]
    out = []
    n = len(orders[models[0]])
    for k in range(n):
        row = [k + 1]
        for m in models:
            r = orders[m][k]
            row += [r["feature"], _fmt(r["mean_abs_shap"], 4), _fmt(r["direction_run"], 2)]
        tf = th[k] if k < len(th) else None
        row.append(tf if tf else (R.DASH if not full else
                                  ("(one of the 8 not listed)" if th else R.DASH)))
        for m in models:
            row.append(next(r["rank_run"] for r in by_model[m] if r["feature"] == tf)
                       if tf else R.DASH)
        out.append(row)
    return R.md_table(head, out)


def comparison_md(res, models):
    head = ["label", "thesis figure", "model", "top feature: run", "thesis",
            "run rank of the thesis top feature", "Spearman, 27 features",
            "Spearman, 19 listed", "top-5 overlap", "top-10 overlap", "top-19 overlap"]
    rows = []
    for lab in SH.LABELS_REPORT_ORDER:
        for m in models:
            c = res["comparison"][m].get(lab)
            if c is None:
                continue
            rows.append([lab, c["figure"], m, c["top1_run"], c["top1_thesis"],
                         c["rank_of_thesis_top"], _fmt(c["spearman_all"], 2),
                         _fmt(c["spearman_listed"], 2), f"{c['top5_overlap']}/5",
                         f"{c['top10_overlap']}/10", f"{c['top19_overlap']}/19"])
    return R.md_table(head, rows)


def acceptance_md(res, models):
    head = ["statement (plan §9.3 row A19; TH p.78-79)", "model", "run", "thesis", "same"]
    rows = []
    for m in models:
        acc = res["acceptance"][m]
        for lab, t in acc["top_feature"].items():
            rows.append([f"top feature for {lab}", m,
                         f"{t['run']} ({t['expected']} is rank {t['rank_of_expected']})",
                         t["expected"], "yes" if t["met"] else "no"])
        mir = acc["mirror_MR_DA"]
        if mir:
            rows.append(["MR vs DA mirror image", m,
                         f"opposite direction in {mir['opposite']} of MR's top "
                         f"{mir['top_defined']} features; Spearman of the importances "
                         f"{_fmt(mir['spearman_importance'], 2)}; correlation of the SHAP "
                         f"values {_fmt(mir['shap_correlation'], 2)}",
                         '"an almost mirror image" (no number)', "yes" if mir["met"] else "no"])
    return R.md_table(head, rows)


def directions_md(res, models):
    head = ["label", "feature", "thesis text (TH p.78-79): high values go with"]
    for m in models:
        head += [f"run ({m}): Spearman(feature, SHAP)", "agrees"]
    rows = []
    for lab in SH.LABELS_REPORT_ORDER:
        per = {m: {d["feature"]: d for d in res["directions"][m].get(lab, [])} for m in models}
        for feat in per[models[0]]:
            th = per[models[0]][feat]["thesis"]
            row = [lab, feat, "label = 1 (correct)" if th > 0 else "label = 0 (incorrect)"]
            for m in models:
                d = per[m][feat]
                row += [_fmt(d["run"], 2), R.DASH if d["agrees"] is None
                        else ("yes" if d["agrees"] else "no")]
            rows.append(row)
    return R.md_table(head, rows)


def range_md(res, models, labels):
    head = ["label", "model", "smallest SHAP value", "largest", "base value",
            "thesis x-axis (read from the figure, approximate)"]
    axis = {"MR": "-5.3 .. +3.9", "DA": "-4.1 .. +5.5", "SP": "-5.2 .. +2.9",
            "HAC": "-3.9 .. +2.5", "WC": "-5.6 .. +3.2"}
    rows = []
    for lab in labels:
        for m in models:
            v = res["values"][m][lab]
            rows.append([lab, m, _fmt(float(v.min()), 2), _fmt(float(v.max()), 2),
                         _fmt(res["base"][m][lab], 3), axis.get(lab, R.DASH)])
    return R.md_table(head, rows)


def main(argv=None):
    args = parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models or any(m not in SH.MODELS for m in models):
        raise SystemExit(f"--models: a comma-separated sub-set of {SH.MODELS}")
    nsamples = args.kernel_nsamples if args.kernel_nsamples == "auto" \
        else int(args.kernel_nsamples)
    real = not args.smoke
    build = args.build if real else "_smoke"
    tag = "a19" + (f"_seed{args.seed}" if args.seed else "") \
        + (f"_limit{args.limit}" if args.limit and real else "")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.runs_root) / build / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(out_dir / "run.log")
    C.LOG.info("A19 SHAP start (pid %d): %s", os.getpid(), vars(args))
    oq30 = SH.oq30_from_config()

    if real:
        inst, data = C.load_real_build(args.build, args.build_root, args.limit or None)
    else:
        inst, data = C.smoke_instances(args.smoke)
    full = real and not args.limit
    hyper = C.hyper_for_seed(args.seed, smoke=not real)
    header = C.run_header(args, data, hyper)
    dest = str(Path(args.results_root) / build / tag) + "/" if full and args.results_root else None
    xy = amp.amp_xy(inst)                                  # row 1: 27 features, 5 labels
    labels = tuple(t[2:] for t in xy.target_names)
    report_labels = [lab for lab in SH.LABELS_REPORT_ORDER if lab in labels]
    t_start = time.time()
    status = {"state": "running", "pid": os.getpid(), "started": C.utc_now(),
              "out_dir": str(out_dir), "not_a_result": not full}
    C.write_json(out_dir / "status.json", status)

    res = {"values": {}, "base": {}, "rows": [], "comparison": {}, "directions": {},
           "mirror_MR_DA": {}, "acceptance": {}}
    seconds = {}
    for m in models:                                       # one model at a time: progress
        t0 = time.time()
        part = SH.analyse(xy.X, xy.Y, labels, xy.feature_names, models=(m,), seed=args.seed,
                          hyper=hyper, n_jobs=args.n_jobs,
                          background_n=args.kernel_background, nsamples=nsamples,
                          with_thesis=True)
        for k in ("values", "base", "comparison", "directions", "mirror_MR_DA", "acceptance"):
            res[k].update(part[k])
        res["rows"] += part["rows"]
        seconds[m] = time.time() - t0
        C.LOG.info("A19 %s: %.0f s; top features %s", m, seconds[m],
                   {lab: part["comparison"][m][lab]["top1_run"] for lab in report_labels})

    files = []
    for m in models:
        for lab in report_labels:
            path = out_dir / f"a19_{m}_{lab}.png"
            SH.beeswarm_png(res["values"][m][lab], res["base"][m][lab], xy.X,
                            xy.feature_names, path,
                            title=f"{lab} label, {m}, build {build}"
                            + ("" if full else "  (NOT A RESULT)"))
            files.append(path)
    path = out_dir / "a19_mean_abs_shap.csv"
    cols = ["model", "label", "feature", "feature_number", "mean_abs_shap", "rank_run",
            "direction_run", "shap_min", "shap_max", "rank_thesis", "direction_thesis",
            "thesis_figure"]
    tmp = out_dir / f".{path.name}.{os.getpid()}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in res["rows"]:
            w.writerow(["" if r[c] is None or (isinstance(r[c], float) and r[c] != r[c])
                        else (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in cols])
    os.replace(tmp, path)
    files.append(path)
    arrays = {"X": xy.X, "Y": xy.Y, "labels": np.array(labels),
              "feature_names": np.array(xy.feature_names),
              "case_ids": np.array([str(c) for c in xy.case_ids])}
    for m in models:
        for lab in labels:
            arrays[f"{m}__{lab}"] = res["values"][m][lab]
            arrays[f"{m}__{lab}__base"] = np.array(res["base"][m][lab])
    np.savez_compressed(out_dir / "a19_shap_values.npz", **arrays)
    files.append(out_dir / "a19_shap_values.npz")

    n = len(inst)
    texts = [f"# A19 SHAP values of the CDMCR features - build `{build}`, seed {args.seed}", "",
             f"Written {C.utc_now()}. {n} cases, 27 features; each AMP label (O_MR, O_DA, "
             "O_SP, O_HAC, O_WC) is the target of its own binary model (TH p.60). Models are "
             "trained on all cases and SHAP values are computed for all cases (plan OQ30). "
             "Importance = mean |SHAP| over the cases; direction = Spearman correlation "
             "between a feature's value and its SHAP value (positive: high values push "
             "towards \"the method is correct\")."
             + ("" if full else "  **NOT A RESULT: smoke / limited run.**"), "",
             "The thesis states neither the model nor the explainer behind Figs 5.4-5.7 and "
             "A.1 (plan §5.9). Two analyses are run:", ""]
    texts += [f"- `{m}`: {SH.MODEL_TITLE[m]}" + (
        f"; background {args.kernel_background} cases, nsamples {args.kernel_nsamples}"
        if m == "rf_kernel" else "") + f" ({seconds[m]:.0f} s)." for m in models]
    texts += ["", "The `thesis` columns were read from the rendered figures (19 features "
              "listed per figure; the other 8 are merged there into \"Sum of 8 other "
              "features\"). `run` and `thesis` are always separate columns.", ""]
    if full:
        texts += [DIFFERENCE_NOTE, ""]
    if not full:
        texts += ["Thesis columns are shown for layout only: the data are not the thesis's.", ""]
    texts += ["## Statements of TH p.78-79: thesis-reported and run", "",
              acceptance_md(res, models), "",
              "## Rank order against the thesis figures (T30)", "",
              comparison_md(res, models), "",
              "Spearman, 27 features: the thesis's 8 unlisted features share the average of "
              "ranks 20-27. Spearman, 19 listed: over the features the figure lists.", "",
              "## Directions stated in the thesis text", "", directions_md(res, models), "",
              "## Size of the SHAP values", "", range_md(res, models, report_labels), "",
              "`xgboost_tree` values are in log-odds units, `rf_kernel` values in "
              "probability units (at most 1 in absolute value).", ""]
    for lab in report_labels:
        texts += [f"## {lab} label ({SH.THESIS_FIGURE[lab]})", "",
                  ranking_md(res["rows"], lab, True), "",
                  "Figures: " + ", ".join(f"`a19_{m}_{lab}.png`" for m in models), ""]
    texts += [R.rule_baselines_md(data["rule_baselines_by_domain"])]
    texts += ["", "## Reading notes", ""] + READING_NOTES
    C.write_text(out_dir / "a19_shap.md", "\n".join(texts))
    summary = {"step": "A19 SHAP for CDMCR (TH Figs 5.4-5.7, A.1; T30)", "header": header,
               "oq30": oq30, "models": models, "labels": labels,
               "kernel": {"background_n": args.kernel_background,
                          "nsamples": args.kernel_nsamples},
               "seconds": seconds, "comparison": res["comparison"],
               "directions": res["directions"], "mirror_MR_DA": res["mirror_MR_DA"],
               "acceptance": res["acceptance"],
               "thesis_ranking": {k: list(v) for k, v in SH.THESIS_RANKING.items()},
               "thesis_columns_are_comparable": full, "complete": True}
    C.write_json(out_dir / "a19_summary.json", summary)
    status.update(state="finished", updated=C.utc_now(), elapsed_seconds=time.time() - t_start)
    C.write_json(out_dir / "status.json", status)
    files += [out_dir / f for f in ("a19_shap.md", "a19_summary.json", "status.json",
                                    "run.log")]
    C.copy_results(sorted(files), dest)
    C.LOG.info("A19 SHAP finished in %.0f s; outputs in %s%s", time.time() - t_start, out_dir,
               f"; copied to {dest}" if dest else "")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException as exc:                   # leave a trace in the log
        if not isinstance(exc, SystemExit) or exc.code not in (0, None):
            C.LOG.exception("A19 SHAP stopped: %r", exc)
        raise
