"""The report of a table with its own catalogue of checks (a run with `checks:`, e.g. player_signal): every column
against its workbook rule, in Spanish and English (output/runs/<run>/informe/auditoria.pdf and audit.pdf). The LaTeX
is in templates/checks_*.tex. From the cache only: run_audit.py fills it first.

    AUDIT_RUN=<run> RUN_S3=0 .venv/bin/python -m signal_audit.report.check_report
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from signal_audit import settings
from signal_audit.checks import formula_check as fc
from signal_audit.checks import recompute_signal as rs
from signal_audit.checks import row_evidence as evidence
from signal_audit.data import clickhouse_mirror
from signal_audit.report import figures, plots, report, tables

GOOD = 99   # % of rows (or players) for a check to pass
LIMIT = re.compile(r"^(\w+) (?:BETWEEN (\S+) AND (\S+)|>= (\S+))$")
WORDS = {
    "es": {"file": "auditoria",
           "formula": "\\textbf{{Fórmula del Excel ({}) con las columnas de la misma fila:}} se cumple en el {:.1f}\\% de {:,} filas; en las demás, la tabla queda por encima en el {:.1f}\\% y por debajo en el {:.1f}\\%.",
           "bound": "\\textbf{{Fuera de su rango:}} el {:.1f}\\% de {:,} filas no cumple {}.",
           "recomputed": "\\textbf{{Con los días con apuesta de \\code{{player_daily}}:}} coincide en $D$ en el {:.1f}\\% de {:,} jugadores con los días activos oficiales y en el {:.1f}\\% con los de cualquier monedero; mediana de tabla / recalculada: {}; mediana de la diferencia: {}.",
           "best": " Donde mejor coincide es en $D+{}$ ({:.1f}\\%).", "no_future": " No coincide mejor después de $D$: no usa datos del futuro.",
           "why": "Por qué:", "annex": "anexo~\\ref{sec:checks}", "all_brands": "todas las de la tabla ({})",
           "clear": ("Fórmulas y comprobaciones lógicas que fallan", "La regla del Excel no se cumple con las columnas de la propia tabla, o la tabla tiene valores imposibles para su regla."),
           "confirm": ("Columnas de días activos que fallan", "No coinciden con los días con apuesta de \\code{player_daily} de la misma base, con ninguna de las dos definiciones de día activo, o coinciden mejor con días posteriores."),
           "activity_head": "Columna & Oficial & Cualquier monedero & Mejor día & Usa & Futuro\\\\",
           "official": "oficial", "any": "cualquier monedero", "neither": "ninguna", "yes": "sí", "no": "no",
           "nothing": "Todas las columnas comprobadas pasan.",
           "captions": {
               "coverage": "\\textbf{Cobertura.} Filas por día y marca, y \\% con historia suficiente. \\textbf{Cálculo:} arriba, $N_d$ = filas del día $d$; abajo, $100\\,\\#\\{\\text{filas con \\texttt{has\\_sufficient\\_data} = 1}\\} / N_d$.",
               "formulas": "\\textbf{La tabla frente a su fórmula.} Si la columna siguiera la fórmula del Excel, todas las filas estarían sobre la línea roja. \\textbf{Cálculo:} $x$ = la fórmula con las otras columnas de la fila; $y$ = el valor de la columna; cada celda es $100\\,\\#\\{\\text{filas con } (x, y) \\text{ en ella}\\} / N$; sobre la línea: $|y - x| \\le 0.01 + 0.001\\,|x|$.",
               "limits": "\\textbf{Valores fuera de su regla.} En rojo, lo que la regla del Excel hace imposible. \\textbf{Cálculo:} $y_b = 100\\,\\#\\{\\text{filas con } x \\in b\\} / N$, con $x$ el valor de la columna y $b$ una barra.",
               "active_days": "\\textbf{Día activo y datos del futuro.} \\textbf{Cálculo:} $100\\,\\#\\{p : x_p = f_p(D+s)\\} / N$, con $x_p$ el valor de la tabla y $f_p(D+s)$ la regla del Excel con los días con apuesta de \\code{player_daily} hasta $D+s$; azul, con los días activos oficiales (M053); rojo, con los de cualquier monedero.",
               "score_limits": "\\textbf{Scores y tasas.} La distribución de cada columna con límites, entre esos límites. \\textbf{Cálculo:} $y_b = 100\\,\\#\\{\\text{filas con } x \\in b\\} / N$."}},
    "en": {"file": "audit",
           "formula": "\\textbf{{Workbook formula ({}) with the columns of the same row:}} it holds in {:.1f}\\% of {:,} rows; in the others, the table is above in {:.1f}\\% and below in {:.1f}\\%.",
           "bound": "\\textbf{{Outside its range:}} {:.1f}\\% of {:,} rows do not meet {}.",
           "recomputed": "\\textbf{{With the days with a bet of \\code{{player_daily}}:}} it matches on $D$ for {:.1f}\\% of {:,} players with the official active days and {:.1f}\\% with those of any wallet; median of table / recomputed: {}; median of the difference: {}.",
           "best": " It matches best on $D+{}$ ({:.1f}\\%).", "no_future": " It does not match better after $D$: it does not use later data.",
           "why": "Why:", "annex": "annex~\\ref{sec:checks}", "all_brands": "all of the table ({})",
           "clear": ("Formulas and Logical Checks that Fail", "The workbook rule does not hold with the columns of the table itself, or the table has values that its rule makes impossible."),
           "confirm": ("Active Day Columns that Fail", "They do not match the days with a bet of \\code{player_daily} of the same database, with either definition of an active day, or they match later days better."),
           "activity_head": "Column & Official & Any wallet & Best day & Uses & Later data\\\\",
           "official": "official", "any": "any wallet", "neither": "neither", "yes": "yes", "no": "no",
           "nothing": "Every checked column passes.",
           "captions": {
               "coverage": "\\textbf{Coverage.} Rows per day and brand, and \\% with enough history. \\textbf{Calculation:} top, $N_d$ = rows of day $d$; bottom, $100\\,\\#\\{\\text{rows with \\texttt{has\\_sufficient\\_data} = 1}\\} / N_d$.",
               "formulas": "\\textbf{The table against its formula.} If the column followed the workbook formula, every row would be on the red line. \\textbf{Calculation:} $x$ = the formula with the other columns of the row; $y$ = the value of the column; each cell is $100\\,\\#\\{\\text{rows with } (x, y) \\text{ in it}\\} / N$; on the line: $|y - x| \\le 0.01 + 0.001\\,|x|$.",
               "limits": "\\textbf{Values outside their rule.} In red, what the workbook rule makes impossible. \\textbf{Calculation:} $y_b = 100\\,\\#\\{\\text{rows with } x \\in b\\} / N$, with $x$ the value of the column and $b$ a bar.",
               "active_days": "\\textbf{Active day and later data.} \\textbf{Calculation:} $100\\,\\#\\{p : x_p = f_p(D+s)\\} / N$, with $x_p$ the value in the table and $f_p(D+s)$ the workbook rule with the days with a bet of \\code{player_daily} up to $D+s$; blue, with the official active days (M053); red, with those of any wallet.",
               "score_limits": "\\textbf{Scores and rates.} The distribution of each column with limits, between those limits. \\textbf{Calculation:} $y_b = 100\\,\\#\\{\\text{rows with } x \\in b\\} / N$."}},
}


def _limit(expr: str) -> tuple | None:
    """(column, lower limit, upper limit) of a check `col BETWEEN a AND b` or `col >= a`, else None."""
    m = LIMIT.match(expr)
    return None if not m else (m[1], float(m[2]), float(m[3])) if m[2] else (m[1], float(m[4]), None)


def coverage() -> pd.DataFrame:
    """Rows per day and brand of the local copy, and the share with has_sufficient_data = 1."""
    files = sorted(str(p) for p in settings.MIRROR.glob(f"{settings.TABLE}/*/*/*/part-0.parquet"))
    return duckdb.sql(f"""SELECT snapshot_date AS day, brand_id, count(*) AS rows, avg(has_sufficient_data) AS sufficient
                          FROM read_parquet({files}, union_by_name = true) GROUP BY ALL ORDER BY ALL""").df()


def statuses(ks: pd.DataFrame, res: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    """Per column: wrong (a check below 99%), correct (every check passes), unconfirmed (the workbook has a rule
    we cannot check, and no other check) or unchecked (keys, metadata, columns not in the workbook)."""
    rows = []
    for c, g in ks.groupby("column", sort=False):
        shares = [*res.loc[res["column"] == c, "pass %"].dropna(), *rec.loc[rec["column"] == c, "match at D %"].dropna()]
        has_rule, has_measure = g["kind"].isin(["formula", "recomputed"]).any(), g["measure"].notna().any()
        if any(s < GOOD for s in shares):
            status = "wrong"
        elif shares and (has_rule or not has_measure):
            status = "correct"
        else:
            status = "unconfirmed" if has_measure else "unchecked"
        rows.append({"column": c, "status": status, "item": ""})
    return pd.DataFrame(rows)


def _number(v) -> str:
    return "--" if pd.isna(v) else f"{v:.3g}"


def _tt(sql: str) -> str:
    """SQL in monospace; it may break after punctuation (a regular expression is a single word otherwise)."""
    escaped = "".join("\\" + ch if ch in "_%&#$" else ch for ch in sql)
    return "\\texttt{" + re.sub(r"(\\_|[(,\-\]])", r"\1\\allowbreak{}", escaped) + "}"


def _future(r) -> bool:
    """Later data: a day after D matches (at least half of the players) and clearly better than D."""
    return r["best shift"] > 0 and r["match at best %"] >= 50 and r["match at best %"] > r["match at D %"] + 10


def _issues(lang: str, cfg: dict, res, rec, status, limits: dict, names: list[str]) -> str:
    """The wrong columns, as in the October 2026 report: what fails with its numbers and why, the workbook rule, what
    to do and the evidence. First the columns where a row rule fails, then those that count days with a bet."""
    w = WORDS[lang]
    wrong = list(status.loc[status["status"] == "wrong", "column"])
    if not wrong:
        return w["nothing"]
    failed = res[res["pass %"] < GOOD]

    def block(c):
        lines = []
        for _, r in failed[failed["column"] == c].iterrows():
            lines.append(w["formula"].format(r["measure"], r["pass %"], int(r["n"]), r["higher %"], r["lower %"]) if r["kind"] == "formula"
                         else w["bound"].format(100 - r["pass %"], int(r["n"]), _tt(r["expr"])))
        for _, r in rec[(rec["column"] == c) & (rec["match at D %"] < GOOD)].iterrows():
            text = w["recomputed"].format(r["official %"], int(r["players"]), r["any wallet %"], _number(r["ratio"]), _number(r["diff"]))
            lines.append(text + (w["best"].format(int(r["best shift"]), r["match at best %"]) if _future(r) else w["no_future"]))
        e = cfg["explain"].get(c, {})
        if "why" in e:
            lines.append(f"\\textbf{{{w['why']}}} {e['why'][lang]}")
        error = "\\pts{" + "".join(f"\\item \\raggedright {x}" for x in lines) + "}"
        kinds = set(failed.loc[failed["column"] == c, "kind"])
        figs = [n for n, used in (("formulas", "formula" in kinds), ("limits", "bound" in kinds and c in limits),
                                  ("active_days", c in set(rec["column"]))) if used and n in names]
        proof = ", ".join([f"\\fig{{fig:{n}}}" for n in figs] + [w["annex"]])
        return (f"\\issue{{col:{c}}}{{\\prioE\\ \\code{{{c}}}}}{{{error}}}{{{e['what'][lang] if 'what' in e else ''}}}"
                f"{{{e['fix'][lang] if 'fix' in e else ''}}}{{{proof}}}")

    clear = [c for c in wrong if c in set(failed["column"])]
    out, first = [], True
    for (title, intro), cols in ((w["clear"], clear), (w["confirm"], [c for c in wrong if c not in clear])):
        if cols:
            series = "label=\\textbf{\\arabic*.},ref=\\arabic*,leftmargin=*,series=issues" if first else "resume*=issues"
            out += [f"\\section{{{title}}}", intro, f"\\begin{{enumerate}}[{series}]", *map(block, cols), "\\end{enumerate}", ""]
            first = False
    return "\n".join(out)


def _activity(lang: str, rec: pd.DataFrame) -> str:
    """Per column that counts days with a bet: % that matches on D with each definition of the active day, the
    best day, the definition it uses and whether it uses later data."""
    w = WORDS[lang]
    if not len(rec):
        return "--"
    lines = ["\\begin{center}\\small", "\\begin{tabular}{@{}lrrlll@{}}", "\\toprule", w["activity_head"], "\\midrule"]
    for _, r in rec.iterrows():
        uses = w["neither"] if r["match at D %"] < GOOD else w["official"] if r["variant"] == "official" else w["any"]
        best = f"$D+{int(r['best shift'])}$" if r["best shift"] else "$D$"
        later = f"{w['yes']} ({best})" if _future(r) else w["no"]
        lines.append(f"\\code{{{r['column']}}} & {r['official %']:.1f}\\% & {r['any wallet %']:.1f}\\% & {best} & {uses} & {later}\\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", "\\end{center}"])


def _figures(cfg: dict, ks: pd.DataFrame, res: pd.DataFrame, cov: pd.DataFrame, m: pd.DataFrame, rec: pd.DataFrame, limits: dict) -> list[str]:
    """The figures of the report into settings.FIGURES, each with its claim and how to read it; returns their names."""
    figures.OUT = settings.FIGURES
    names = []

    def draw(name, size, build, **kw):
        figures.one(name, size, build, **kw)
        names.append(name)

    if len(cov):
        per_day = cov.groupby("day")["rows"].sum()
        enough = 100 * (cov["rows"] * cov["sufficient"]).sum() / cov["rows"].sum()

        def coverage_fig(axes):
            plots.coverage(cov, axes)
            plots.head(axes[0].figure, f"The table has {per_day.min():,.0f} to {per_day.max():,.0f} rows a day, and {enough:.0f}% of them have enough history to be scored",
                       "Top: rows per day; the five largest brands in colour, the others added up in grey. Bottom: % of those rows with has_sufficient_data = 1, "
                       "the flag of the workbook for \"the player had enough history to score\".")
        draw("coverage", (10, 6.4), coverage_fig, nrows=2, sharex=True)

    formulas = res[(res["kind"] == "formula") & (res["pass %"] < GOOD)].merge(ks[["column", "n_check", "label"]], on=["column", "n_check"])
    if len(formulas):
        def formulas_fig(axes):
            for ax, (_, r) in zip(np.atleast_1d(axes).flat, formulas.iterrows()):
                plots.formula_vs_table(evidence.formula_pairs(cfg, r["column"], r["expr"]), evidence.PAIR_BINS, ax, r["column"], r["label"], r["pass %"])
            lo, hi = formulas["pass %"].min(), formulas["pass %"].max()
            plots.head(np.atleast_1d(axes).flat[0].figure,
                       f"Only {lo:.0f}% to {hi:.0f}% of the rows follow the workbook formula of {', '.join(formulas['column'])}: a correct column would lie on the red line",
                       "Each panel is one score. x: value that the workbook formula gives with the other columns of the same row. y: value stored in the table. "
                       "Colour: % of the rows in each cell (log scale). Red line: table = formula, where a correct column would lie; the number is the % of rows on it "
                       "(tolerance 0.01 + 0.1%).")
        draw("formulas", (3.6 * len(formulas) + 0.9, 5.6), formulas_fig, ncols=len(formulas))

    broken = res[(res["kind"] == "bound") & (res["pass %"] < GOOD) & res["column"].isin(limits)]
    if len(broken):
        def limits_fig(axes):
            for ax, (_, r) in zip(np.atleast_1d(axes).flat, broken.iterrows()):
                lo, hi = limits[r["column"]]
                plots.limit_hist(evidence.value_hist(cfg, r["column"], lo, hi), ax, r["column"], lo, hi, 100 - r["pass %"])
            claims = "; ".join(f"{r['column']}: {100 - r['pass %']:.1f}% of the rows" for _, r in broken.iterrows())
            plots.head(np.atleast_1d(axes).flat[0].figure, f"Values that the workbook rule makes impossible ({claims})",
                       "Each panel is one column. x: value in the table. y: % of the rows in each bar (log scale, because the bars differ a lot). Dashed line: limit of the rule. "
                       "Red zone and red bars: values the rule makes impossible. The edge bars add up everything beyond the edge of the plot.")
        draw("limits", (5.2 * len(broken) + 0.6, 5.4), limits_fig, ncols=len(broken))

    if len(m):
        later = rec[rec.apply(_future, axis=1)]["column"]

        def active_fig(axes):
            plots.active_day_lines(m, axes)
            claim = (f"{', '.join(later)} match better after day D: they use later data" if len(later) else
                     f"No column matches better after day D (no later data), and none follows the active day for all its players: at most {rec['match at D %'].max():.0f}% match")
            plots.head(np.atleast_1d(axes).flat[0].figure, claim,
                       "Each panel is one column that counts days with a bet. y: % of players where the table equals the value recomputed from the days with a bet "
                       "of player_daily (same database), with data up to day D + s. Blue: official active day (real money or released bonus, M053). Red: any wallet "
                       "(playable bonus too). Shaded: s > 0, data from after the date; a column that uses them would peak there. The number is the match on day D.")
        draw("active_days", (11, 7.6), active_fig, nrows=2, ncols=2, sharey=True)

    scores = {c: lim for c, lim in limits.items() if lim[1] is not None}
    if scores:
        def scores_fig(axes):
            flat = list(np.atleast_1d(axes).flat)
            for ax, (c, (lo, hi)) in zip(flat, scores.items()):
                share = 100 - res.loc[(res["column"] == c) & (res["kind"] == "bound"), "pass %"].min()
                plots.limit_hist(evidence.value_hist(cfg, c, lo, hi), ax, c, lo, hi, share, compact=True)
            for ax in flat[len(scores):]:
                ax.axis("off")
            plots.head(flat[0].figure, "Distribution of every score and rate that has limits in the workbook rule",
                       "Each panel is one column. x: value in the table. y: % of the rows in each bar (log scale only where the bars differ by more than 20 times). Dashed lines: limits of the rule; "
                       "red: values beyond them (none, unless the line above the panel says so). The line above says the share of rows exactly at a limit; "
                       "a large share is a default value or a cap, and an empty range or a flat shape is worth a question to the platform.")
        rows = (len(scores) + 3) // 4
        draw("score_limits", (14, 3.1 * rows + 1.6), scores_fig, nrows=rows, ncols=4, squeeze=False)
    return names


def write(lang: str, cfg: dict, res, rec, status, cov: pd.DataFrame, limits: dict, names: list[str]) -> Path:
    w = WORDS[lang]
    diff = clickhouse_mirror.schema_differences()
    per_day = cov.groupby("day")["rows"].sum() if len(cov) else pd.Series([np.nan])
    values = {"table": settings.RUN["table"], "today": report.day(dt.date.today(), lang),
              "brands": w["all_brands"].format(cov["brand_id"].nunique() if len(cov) else 0),
              "snapshots": report.days(settings.SNAPSHOTS, lang),
              "evidence": report.days(sorted(set(settings.EVIDENCE_DAYS) | set(settings.SNAPSHOTS)), lang),
              "catalogue": settings.CHECKS.name, "missing": report.codes(diff["missing"], lang),
              "extra": report.codes(diff["extra"], lang), "rows_lo": per_day.min(), "rows_hi": per_day.max(),
              "sufficient": 100 * (cov["rows"] * cov["sufficient"]).sum() / cov["rows"].sum() if len(cov) else np.nan,
              "failing": _issues(lang, cfg, res, rec, status, limits, names), "activity": _activity(lang, rec),
              "figures": "\n".join(f"\\evidence{{fig:{n}}}{{1.0}}{{{n}.png}}{{{w['captions'][n]}}}" for n in names)}
    out = settings.TEX / f"{w['file']}.tex"
    out.write_text(report.fill(report.template("checks", lang), values), encoding="utf-8")
    return out


def build(compile: bool = True) -> pd.DataFrame:
    """Checks, figures, tables and the report (es, en) of the run, from the cache. Returns the status per column."""
    settings.TEX.mkdir(parents=True, exist_ok=True)
    tables.TEX = settings.TEX
    cfg = fc.load()
    ks = fc.checks(cfg)
    res = fc.run(cfg)
    m = rs.match()
    rec = rs.summary(m)
    status = statuses(ks, res, rec)
    limits = {c: (lo, hi) for e in ks.loc[ks["kind"] == "bound", "expr"] if (found := _limit(e)) for c, lo, hi in [found]}
    cov = coverage()
    names = _figures(cfg, ks, res, cov, m, rec, limits)
    for lang in ("es", "en"):
        unconfirmed = ("Sin confirmar (falta la definición o sus entradas)" if lang == "es"
                       else "Unconfirmed (no definition, or its inputs are not in the table)")
        tables.status_tables(status, lang, rec.set_index("column")["match at D %"].to_dict(), unconfirmed)
        tex = write(lang, cfg, res, rec, status, cov, limits, names)
        print("saved", report.compile_pdf(tex) if compile else tex)
    return status


if __name__ == "__main__":
    s = build()
    print(s["status"].value_counts().to_string())
    print("wrong:", ", ".join(s.loc[s["status"] == "wrong", "column"]) or "none")
