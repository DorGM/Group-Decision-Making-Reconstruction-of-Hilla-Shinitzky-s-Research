"""Text rendering of result tables next to the thesis values.

Step A15 of plan §9.3 (TH Tables 5.4, 5.5, A.9, A.10) and the small
Markdown helpers shared by the drivers of A16-A18. A condition under
which the runs were authorised (docs/RULE_BASELINE_CHECK.md) is that every
result is shown beside the thesis-reported number, per domain, together with the rule baselines recomputed on
the build beside the printed ones; the "P(success by AG)" column of every
effectiveness table is exactly that pair.

Rates are shown rounded half up to three decimals (cdm.eval.stats).
Nothing is computed from data here.
"""
from cdm.eval import effectiveness as E
from cdm.eval import stats as S

DASH = "–"


def rate(x, decimals=3):
    """A rate / PRE cell: fixed decimals, DASH when undefined."""
    return DASH if x is None or x != x else format(x, f".{decimals}f")


def signed(x, decimals=3):
    return DASH if x is None or x != x else format(x, f"+.{decimals}f")


def pvalue(p):
    """A p-value cell: two significant digits; 0 stays 0."""
    if p is None or p != p:
        return DASH
    if p == 0:
        return "0"
    return format(p, ".3f") if p >= 0.001 else format(p, ".1e")


def md_table(head, rows):
    """A Markdown table from a header list and rows of cells."""
    out = ["| " + " | ".join(str(h) for h in head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _count(k, n):
    return f"{k}/{n}"


def effectiveness_block_md(rep, th):
    """Markdown table of one block: one row per method, reproduced and
    thesis side by side. `th` may be None (no thesis counterpart)."""
    head = ["method",
            "selected: run", "thesis",
            "P(sel): run", "thesis",
            "P(success by AG) = rule baseline: run", "thesis",
            "correct when selected: run", "thesis",
            "P(success given selected): run", "thesis", "diff",
            "PRE: run", "thesis",
            "p: run", "thesis (printed)"]
    rows = []
    for m, r in rep.items():
        t = (th or {}).get(m)
        rr = r["rounded"]
        tr = t["rounded"] if t else {}
        printed_p = (t or {}).get("printed", {}).get("p") if t else None
        rows.append([
            m,
            _count(r["n_selected"], r["n"]), _count(t["n_selected"], t["n"]) if t else DASH,
            rate(rr["p_selected"]), rate(tr.get("p_selected")),
            f"{rate(rr['p_success'])} ({r['n_success']})",
            f"{rate(tr.get('p_success'))} ({t['n_success']})" if t else DASH,
            _count(r["n_selected_success"], r["n_selected"]),
            _count(t["n_selected_success"], t["n_selected"]) if t else DASH,
            rate(rr["p_success_given_selected"]), rate(tr.get("p_success_given_selected")),
            signed(None if not t or r["p_success_given_selected"] is None
                   or t["p_success_given_selected"] is None
                   else r["p_success_given_selected"] - t["p_success_given_selected"]),
            rate(rr["pre"]), rate(tr.get("pre")),
            pvalue(r["p"]), pvalue(printed_p) if t else DASH])
    return md_table(head, rows)


def thesis_layout_md(rep, th):
    """The same block in the layout of the thesis tables (rows = the four
    quantities, columns = methods), each cell "run / thesis"."""
    labels = {"p_selected": "P(AMP selected AG)", "p_success": "P(success by AG)",
              "p_success_given_selected": "P(success by AG given AMP selected AG)",
              "pre": "PRE"}
    methods = list(rep)
    rows = []
    for k in E.RATE_KEYS:
        rows.append([labels[k]] + [
            f"{rate(rep[m]['rounded'][k])} / "
            f"{rate(th[m]['rounded'][k]) if th and m in th else DASH}" for m in methods])
    rows.append(["p (one-tailed)"] + [
        f"{pvalue(rep[m]['p'])} / "
        f"{pvalue(th[m].get('printed', {}).get('p')) if th and m in th else DASH}"
        for m in methods])
    return md_table(["run / thesis"] + methods, rows)


def effectiveness_md(title, reproduced, thesis=None, sources=None, note=None):
    """Markdown section of an effectiveness table: the global block and
    every domain block, each as a side-by-side table and in the thesis
    layout."""
    out = [f"## {title}", ""]
    if note:
        out += [note, ""]
    for block, rep in reproduced.items():
        th = (thesis or {}).get(block)
        n = next(iter(rep.values()))["n"]
        name = "All cases" if block == E.ALL else block
        out.append(f"### {name} (n = {n}"
                   + (f"; thesis n = {next(iter(th.values()))['n']}" if th else "") + ")")
        if sources and block in sources:
            out.append(f"Thesis side: {sources[block]}.")
        out += ["", effectiveness_block_md(rep, th), "", thesis_layout_md(rep, th), ""]
    return "\n".join(out)


def effectiveness_csv_rows(tag, reproduced, thesis=None):
    """Long-format rows (list of lists, first = header) of an
    effectiveness table: one row per block x method, reproduced and thesis
    counts and rounded rates."""
    head = ["table", "block", "method", "n", "n_selected", "n_success",
            "n_selected_success", "p_selected", "p_success", "p_success_given_selected",
            "pre", "z", "p", "p_one_minus_cdf",
            "thesis_n", "thesis_n_selected", "thesis_n_success",
            "thesis_n_selected_success", "thesis_p_selected", "thesis_p_success",
            "thesis_p_success_given_selected", "thesis_pre", "thesis_p_printed"]
    rows = [head]

    def cell(x):
        return "" if x is None else x

    for block, rep in reproduced.items():
        for m, r in rep.items():
            t = (thesis or {}).get(block, {}).get(m)
            row = [tag, block, m, r["n"], r["n_selected"], r["n_success"],
                   r["n_selected_success"]] + [cell(r["rounded"][k]) for k in E.RATE_KEYS] \
                + [cell(r["z"]), cell(r["p"]), cell(r["p_one_minus_cdf"])]
            if t:
                row += [t["n"], t["n_selected"], t["n_success"], t["n_selected_success"]] \
                    + [cell(t["rounded"][k]) for k in E.RATE_KEYS] \
                    + [cell(t.get("printed", {}).get("p"))]
            else:
                row += [""] * 9
            rows.append(row)
    return rows


def success_by_domain_md(title, reproduced, thesis=None, note=None):
    """Markdown table of success per domain (TH Table A.9 / T10).

    reproduced, thesis : {approach: {domain: {n, successes, rounded, ...}}}
    (cdm.eval.effectiveness.success_by_domain / thesis_success_by_domain);
    a thesis entry may be None or lack a domain.
    """
    out = [f"## {title}", ""]
    if note:
        out += [note, ""]
    head = ["domain", "n (run)"]
    for a in reproduced:
        head += [f"{a}: run", f"{a}: thesis", f"{a}: diff (cases)"]
    blocks = list(next(iter(reproduced.values())))
    rows = []
    for d in blocks:
        row = ["All cases" if d == E.ALL else d,
               next(iter(reproduced.values()))[d]["n"]]
        for a, rep in reproduced.items():
            r = rep[d]
            t = ((thesis or {}).get(a) or {}).get(d)
            row.append(f"{rate(r['rounded'])} ({r['successes']})")
            if t:
                row.append(f"{rate(t['rounded'])} ({t['successes']}/{t['n']})")
                row.append(format(r["successes"] - t["successes"], "+d")
                           if t["n"] == r["n"] else "n differs")
            else:
                row += [DASH, DASH]
        rows.append(row)
    out.append(md_table(head, rows))
    return "\n".join(out) + "\n"


def rule_baselines_md(inst_baselines, title="Rule baselines: recomputed on the build "
                                            "vs printed in the thesis"):
    """Markdown table of {block: {method: {recomputed, n, thesis?}}}:
    recomputed and thesis-reported counts, globally and per domain
    (docs/RULE_BASELINE_CHECK.md)."""
    methods = list(next(iter(inst_baselines.values())))
    rows = []
    for block, per in inst_baselines.items():
        row = ["All cases" if block == E.ALL else block,
               next(iter(per.values()))["n"]]
        for m in methods:
            r = per[m]
            cell = f"{r['recomputed']} ({rate(S.round_half_up(r['recomputed'], r['n']))})"
            if r.get("thesis") is not None:
                cell += f" / {r['thesis']} ({rate(S.round_half_up(r['thesis'], r['thesis_n']))})"
                cell += f" / {r['recomputed'] - r['thesis']:+d}"
            row.append(cell)
        rows.append(row)
    return (f"## {title}\n\nCells: recomputed count (rate) / thesis count (rate) / "
            "difference in cases.\n\n" + md_table(["block", "n"] + methods, rows) + "\n")
