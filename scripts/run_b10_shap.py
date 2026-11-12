"""Step B10 (plan §5.9, §8 OQ30, §9.3): SHAP values of the RCR and ACR
features (TH Figs 5.8, 5.9; target T30, World-B part).

  python scripts/run_b10_shap.py         data = config/worldB.yaml

Two analyses per representation, on a model trained on ALL instances of one
partition (the first seed, or the file's original assignment):
  xgboost_tree     : XGBoost + shap.TreeExplainer, log-odds (plan OQ30 (i));
  soft_vote_kernel : shap.KernelExplainer on the soft-voting ensemble of
                     RP / AP (plan OQ30 (ii)), for a seeded sub-sample of
                     --kernel-rows instances (0 = all).

Plan row B10 puts the acceptance on REAL DATA ONLY. On the synthetic
stand-in this is an end-to-end SMOKE RUN: every output is marked
"SYNTHETIC — not a reproduction", the run's and the thesis's rank orders
are printed in separate columns, and NO agreement measure or acceptance
statement is computed. Output: runs/worldB_synthetic/b10_shap/.

A real file is written to runs/worldB_real/b10_shap/ with the Spearman
correlation to the thesis ranking and the acceptance statements; that path
is UNTESTED ON REAL DATA.

Outputs: shap_rcr_acr.md, shap_mean_abs.csv, shap_summary.json,
shap_values.npz, b10_<model>_<approach>.png, b10_shap.log.
"""
import sys

import numpy as np

import _worldb as W
from _worldb import LOG, write_json, write_text
from cdm.approaches import worldB_instances as WI
from cdm.data.worldB import partition as WP
from cdm.eval import shap_analysis as SH
from cdm.eval import worldB_ml as ML
from cdm.eval import worldB_shap as WS
from cdm.eval import worldB_thesis_values as T

FEATURES_OF = {"RP": "RCR", "AP": "ACR"}
FIELDS = ("approach", "features", "model", "feature", "mean_abs_shap", "run_rank",
          "direction_spearman", "thesis_rank")


def main(argv=None):
    ap = W.parser(__doc__)
    ap.add_argument("--models", default=",".join(WS.MODELS))
    ap.add_argument("--kernel-rows", type=int, default=WS.DEFAULT_KERNEL_ROWS,
                    help="instances explained by KernelSHAP (seeded sample; 0 = all)")
    ap.add_argument("--kernel-background", type=int, default=50)
    ap.add_argument("--kernel-nsamples", default="auto")
    args = ap.parse_args(argv)
    models = tuple(m for m in args.models.split(",") if m)
    nsamples = args.kernel_nsamples if args.kernel_nsamples == "auto" \
        else int(args.kernel_nsamples)
    run = W.Run(args, "b10_shap", "SHAP of the RCR / ACR features (step B10)")
    if run.synthetic:
        LOG.info("%s -- end-to-end SMOKE RUN; no comparison with the thesis is made "
                 "(plan row B10: real data only)", run.mark)
    SH.oq30_from_config(run.config)
    seed = run.seeds[0]
    subsets = WP.subset_cases(run.data, seed=0 if seed is None else seed,
                              assignment=run.assignment)
    model_seed = int(run.opts["hyper"]["seed"])

    rows, res, arrays, pngs = [], {}, {}, []
    for approach in ("RP", "AP"):
        inst = WI.build_instances(subsets, approach, run.opts["majority_rule"])
        out, X, y = WS.analyse(inst, run.opts["hyper"], models, model_seed,
                               args.kernel_rows, args.kernel_background, nsamples,
                               n_jobs=run.n_jobs)
        names = list(inst.feature_names)
        th_rank = WS.thesis_ranks(approach, names)
        res[approach] = {"features": FEATURES_OF[approach], "n_instances": int(len(X)),
                         "instances_left_out_undefined": int(inst.undefined.sum()),
                         "thesis_figure": T.SHAP_FIGURE[approach],
                         "thesis_ranking": list(T.SHAP_RANKING[approach]), "models": {}}
        for model, r in out.items():
            ranks = SH.ranks(r["importance"])
            cell = {"title": WS.MODEL_TITLE[model], "n_explained": int(len(r["rows"])),
                    "base_value": r["base"], "run_ranking": r["ranking"],
                    "mean_abs_shap": dict(zip(names, map(float, r["importance"]))),
                    "max_abs_shap": float(np.abs(r["values"]).max())}
            if not run.synthetic:                    # plan row B10: real data only
                cell["comparison_with_thesis"] = WS.compare_with_thesis(
                    approach, r["importance"], names, synthetic=False)
            res[approach]["models"][model] = cell
            for j, name in enumerate(names):
                rows.append({"approach": approach, "features": FEATURES_OF[approach],
                             "model": model, "feature": name,
                             "mean_abs_shap": repr(float(r["importance"][j])),
                             "run_rank": int(ranks[j]),
                             "direction_spearman": repr(float(r["direction"][j])),
                             "thesis_rank": int(th_rank[j])})
            arrays[f"{approach}_{model}_values"] = r["values"]
            arrays[f"{approach}_{model}_rows"] = r["rows"]
            arrays[f"{approach}_{model}_base"] = np.asarray(r["base"])
            png = run.out / f"b10_{model}_{approach}.png"
            title = f"{FEATURES_OF[approach]} — {WS.MODEL_TITLE[model]}"
            if run.synthetic:
                title = f"{run.mark}\n{title}"
            SH.beeswarm_png(r["values"], r["base"], X[r["rows"]], names, png, title=title)
            pngs.append(png)
        arrays[f"{approach}_X"] = X
        arrays[f"{approach}_y"] = y
        arrays[f"{approach}_feature_names"] = np.asarray(names)
        LOG.info("%s: %d instances, models %s", approach, len(X), ", ".join(out))

    arrays["marker"] = np.asarray(run.mark)
    np.savez_compressed(run.out / "shap_values.npz", **arrays)
    doc = run.head("B10", "§5.9, §8 OQ30, §9.3 row B10, T30")
    doc.update({
        "partition_used": {"assignment": run.assignment, "seed": seed},
        "models_trained_on": "all instances of that partition (plan OQ30)",
        "kernel": {"rows": args.kernel_rows, "background": args.kernel_background,
                   "nsamples": nsamples, "seed": model_seed},
        "importance": "mean |SHAP| over the explained instances",
        "run": res, "minutes": run.minutes(),
    })
    if run.synthetic:
        doc["note"] = ("SYNTHETIC stand-in: END-TO-END SMOKE RUN. Plan row B10 puts the "
                       "acceptance on real data only; no agreement with the thesis's rank "
                       "order is computed or claimed.")
    write_json(run.out / "shap_summary.json", doc)
    run.csv(run.out / "shap_mean_abs.csv", FIELDS, rows)

    label = ML.run_label(run.synthetic)
    lines = ["### SHAP rank orders of the RCR and ACR features (TH Figs 5.8, 5.9; T30)"
             + (f" — {run.mark}" if run.synthetic else ""), ""]
    if run.synthetic:
        lines += ["**End-to-end smoke run on the synthetic stand-in.** Plan row B10 puts "
                  "the acceptance of this step on real data only. The two orders below "
                  "are printed side by side so that the output format can be checked; no "
                  "agreement measure is computed and none should be read off: the "
                  "synthetic responses were invented, so which of their features carry "
                  "information is a property of the generator, not of the thesis data. "
                  "Where a row of the two orders happens to coincide, that is not evidence "
                  "either: the generator's prediction model was built on the SP mechanism "
                  "(plan §2.3), which makes the predicted-support features informative by "
                  "design.", ""]
    part = ("the file's original assignment" if run.assignment == "original"
            else f"partition seed {seed}")
    lines += [f"Models trained on all instances of one partition ({part}); importance = "
              "mean |SHAP|. KernelSHAP explains a seeded sample of "
              f"{args.kernel_rows or 'all'} instances (background {args.kernel_background}).",
              ""]
    for approach in ("RP", "AP"):
        r = res[approach]
        lines += [f"#### {r['features']} features ({approach}); thesis order: "
                  f"{r['thesis_figure']}" + (f" — {run.mark}" if run.synthetic else ""), "",
                  f"Instances: {r['n_instances']} (left out for an undefined feature: "
                  f"{r['instances_left_out_undefined']}).", ""]
        head = "| Rank "
        for model in r["models"]:
            head += f"| {label}: {model} "
        lines += [head + "| thesis: figure order |",
                  "|---" * (2 + len(r["models"])) + "|"]
        for k, th in enumerate(r["thesis_ranking"]):
            row = f"| {k + 1} "
            for model in r["models"]:
                row += f"| {r['models'][model]['run_ranking'][k]} "
            lines.append(row + f"| {th} |")
        lines.append("")
        for model, cell in r["models"].items():
            lines.append(f"- {model}: {cell['title']}; {cell['n_explained']} instances "
                         f"explained; largest |SHAP| {cell['max_abs_shap']:.2f}.")
            if "comparison_with_thesis" in cell:
                c = cell["comparison_with_thesis"]
                lines.append(f"  Spearman with the thesis order {c['spearman_with_thesis']:.3f}; "
                             f"acceptance (plan row B10): {c['acceptance']}.")
        lines.append("")
    write_text(run.out / "shap_rcr_acr.md", ML.wrap(lines[:-1], run.synthetic))
    LOG.info("\n%s", (run.out / "shap_rcr_acr.md").read_text(encoding="utf-8"))
    run.copy([run.out / "shap_rcr_acr.md", run.out / "shap_summary.json",
              run.out / "shap_mean_abs.csv", run.out / "shap_values.npz"] + pngs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
