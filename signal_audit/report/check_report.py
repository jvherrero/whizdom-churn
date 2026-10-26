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
from signal_audit.checks import evidence as ev
from signal_audit.checks import formula_check as fc
from signal_audit.checks import recompute_signal as rs
from signal_audit.data import audit, clickhouse_mirror
from signal_audit.report import figures, plots, report, tables

GOOD = 99   # % of rows (or players) for a check to pass
RANGE = re.compile(r"^(\w+) BETWEEN (\S+) AND (\S+)$")
WORDS = {
    "es": {"file": "auditoria",
           "formula": "\\textbf{{Fórmula del Excel ({}) con las columnas de la misma fila:}} se cumple en el {:.1f}\\% de {:,} filas; en las demás, la tabla queda por encima en el {:.1f}\\% y por debajo en el {:.1f}\\%.",
           "bound": "\\textbf{{Fuera de su rango:}} el {:.1f}\\% de {:,} filas no cumple {}.",
           "recomputed": "\\textbf{{Recalculada desde las tablas diarias del lake:}} coincide en $D$ en el {:.1f}\\% de {:,} jugadores ({}); mediana de tabla / recalculada: {}; mediana de la diferencia: {}.",
           "best": " Donde mejor coincide es en $D+{}$ ({:.1f}\\%).", "no_future": " No coincide mejor después de $D$: no usa datos del futuro.",
           "why": "Por qué:", "annex": "anexo~\\ref{sec:checks}",
           "clear": ("Errores claros", "La regla del Excel no se cumple con las columnas de la propia tabla, o la tabla tiene valores imposibles para su regla. No depende de ningún recálculo."),
           "confirm": ("A confirmar", "Solo fallan frente a sus entradas recalculadas desde las tablas diarias del lake. La tabla puede usar otras entradas (otra tabla, otra moneda, otra definición de día activo); antes de darlos como errores hay que confirmar con la plataforma qué entradas usa."),
           "nothing": "Todas las columnas comprobadas pasan.",
           "captions": {"coverage": "\\textbf{Cobertura.} Filas por día y marca, y \\% con datos suficientes.",
                        "check_results": "\\textbf{Reglas de fila que fallan.} \\% de filas que cumplen la regla y, si no, si la tabla queda por encima o por debajo, o fuera del rango.",
                        "recomputed_match": "\\textbf{Recalculadas.} \\% de jugadores en que la columna es igual a su regla con datos hasta $D+s$; en una columna correcta, solo $s = 0$ es oscuro.",
                        "score_limits": "\\textbf{Scores y proporciones.} \\% de filas en cada barra entre los límites de la regla."}},
    "en": {"file": "audit",
           "formula": "\\textbf{{Workbook formula ({}) with the columns of the same row:}} it holds in {:.1f}\\% of {:,} rows; in the others, the table is above in {:.1f}\\% and below in {:.1f}\\%.",
           "bound": "\\textbf{{Outside its range:}} {:.1f}\\% of {:,} rows do not meet {}.",
           "recomputed": "\\textbf{{Recomputed from the daily tables of the lake:}} it matches on $D$ for {:.1f}\\% of {:,} players ({}); median of table / recomputed: {}; median of the difference: {}.",
           "best": " It matches best on $D+{}$ ({:.1f}\\%).", "no_future": " It does not match better after $D$: it does not use later data.",
           "why": "Why:", "annex": "annex~\\ref{sec:checks}",
           "clear": ("Clear Errors", "The workbook rule does not hold with the columns of the table itself, or the table has values that its rule makes impossible. It does not depend on any recomputation."),
           "confirm": ("To Confirm", "They only fail against their inputs recomputed from the daily tables of the lake. The table may use other inputs (another table, another currency, another definition of an active day); before calling them errors, the platform must confirm which inputs it uses."),
           "nothing": "Every checked column passes.",
           "captions": {"coverage": "\\textbf{Coverage.} Rows per day and brand, and \\% with enough data.",
                        "check_results": "\\textbf{Row rules that fail.} \\% of rows that meet the rule and, if not, whether the table is above or below it, or outside the range.",
                        "recomputed_match": "\\textbf{Recomputed.} \\% of players where the column equals its rule with data up to $D+s$; for a correct column only $s = 0$ is dark.",
                        "score_limits": "\\textbf{Scores and shares.} \\% of rows in each bar between the limits of the rule."}},
}


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
    return "\\texttt{" + "".join("\\" + ch if ch in "_%&#$" else ch for ch in sql) + "}"


def _issues(lang: str, cfg: dict, res, rec, status, limits: dict, names: list[str]) -> str:
    """The wrong columns, as in the October 2026 report: what fails with its numbers and why, the workbook rule, what
    to do and the evidence. Clear errors first (a row rule fails), then those to confirm (only recomputed checks)."""
    w = WORDS[lang]
    wrong = list(status.loc[status["status"] == "wrong", "column"])
    if not wrong:
        return w["nothing"]
    failed = res[res["pass %"] < GOOD]

    def block(c, badge):
        lines = []
        for _, r in failed[failed["column"] == c].iterrows():
            lines.append(w["formula"].format(r["measure"], r["pass %"], int(r["n"]), r["higher %"], r["lower %"]) if r["kind"] == "formula"
                         else w["bound"].format(100 - r["pass %"], int(r["n"]), _tt(r["expr"])))
        for _, r in rec[(rec["column"] == c) & (rec["match at D %"] < GOOD)].iterrows():
            text = w["recomputed"].format(r["match at D %"], int(r["players"]), r["variant"], _number(r["ratio"]), _number(r["diff"]))
            better = r["best shift"] > 0 and r["match at best %"] > r["match at D %"] + 1
            lines.append(text + (w["best"].format(int(r["best shift"]), r["match at best %"]) if better else w["no_future"]))
        e = cfg["explain"].get(c, {})
        if "why" in e:
            lines.append(f"\\textbf{{{w['why']}}} {e['why'][lang]}")
        error = "\\pts{" + "".join(f"\\item {x}" for x in lines) + "}"
        what = e["what"][lang] if "what" in e else ""
        fix = e["fix"][lang] if "fix" in e else ""
        figs = [n for n, used in (("check_results", c in set(failed["column"])), ("recomputed_match", c in set(rec["column"])),
                                  ("score_limits", c in limits)) if used and n in names]
        evidence = ", ".join([f"\\fig{{fig:{n}}}" for n in figs] + [w["annex"]])
        return f"\\issue{{col:{c}}}{{{badge}\\ \\code{{{c}}}}}{{{error}}}{{{what}}}{{{fix}}}{{{evidence}}}"

    clear = [c for c in wrong if c in set(failed["column"])]
    groups = [(w["clear"], "\\prioE", clear), (w["confirm"], "\\prioC", [c for c in wrong if c not in clear])]
    out, first = [], True
    for (title, intro), badge, cols in groups:
        if cols:
            series = "label=\\textbf{\\arabic*.},ref=\\arabic*,leftmargin=*,series=issues" if first else "resume*=issues"
            out += [f"\\section{{{title}}}", intro, f"\\begin{{enumerate}}[{series}]", *[block(c, badge) for c in cols], "\\end{enumerate}", ""]
            first = False
    return "\n".join(out)


def _figures(cov: pd.DataFrame, res: pd.DataFrame, m: pd.DataFrame, limits: dict) -> list[str]:
    """The figures of the report into settings.FIGURES; returns their names."""
    figures.OUT = settings.FIGURES
    names = []
    if len(cov):
        figures.one("coverage", (10, 6), lambda axes: plots.coverage(cov, axes), nrows=2, sharex=True)
        names.append("coverage")
    failed = res[res["pass %"] < GOOD]
    if len(failed):
        figures.one("check_results", (10, 0.55 * len(failed) + 1.6), lambda ax: plots.check_results(failed, ax))
        names.append("check_results")
    if len(m):
        figures.one("recomputed_match", (11, 0.42 * m["column"].nunique() + 1.8), lambda ax: plots.match_heatmap(m, ax))
        names.append("recomputed_match")
    if not limits:
        return names
    _, hist = ev.index_limits(audit.load_config(), {c: (lo, hi, "") for c, (lo, hi) in limits.items()})
    cols = [c for c in limits if c in set(hist["column"])]
    if cols:
        def draw(axes):
            for ax, c in zip(axes.flat, cols):
                plots.index_limits(hist, c, ax)
                ax.set_title(c)
            for ax in axes.flat[len(cols):]:
                ax.axis("off")
        rows = (len(cols) + 2) // 3
        figures.one("score_limits", (11, 2.6 * rows), draw, nrows=rows, ncols=3, squeeze=False)
        names.append("score_limits")
    return names


def write(lang: str, cfg: dict, res, rec, status, cov: pd.DataFrame, limits: dict, names: list[str]) -> Path:
    w = WORDS[lang]
    diff = clickhouse_mirror.schema_differences()
    per_day = cov.groupby("day")["rows"].sum() if len(cov) else pd.Series([np.nan])
    values = {"table": settings.RUN["table"], "today": report.day(dt.date.today(), lang),
              "brands": ", ".join(map(str, cfg["brands"])), "snapshots": report.days(settings.SNAPSHOTS, lang),
              "evidence": report.days(sorted(set(settings.EVIDENCE_DAYS) | set(settings.SNAPSHOTS)), lang),
              "catalogue": settings.CHECKS.name, "missing": report.codes(diff["missing"], lang),
              "extra": report.codes(diff["extra"], lang), "rows_lo": per_day.min(), "rows_hi": per_day.max(),
              "sufficient": 100 * (cov["rows"] * cov["sufficient"]).sum() / cov["rows"].sum() if len(cov) else np.nan,
              "failing": _issues(lang, cfg, res, rec, status, limits, names),
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
    limits = {g[1]: (float(g[2]), float(g[3])) for e in ks.loc[ks["kind"] == "bound", "expr"] if (g := RANGE.match(e))}
    cov = coverage()
    names = _figures(cov, res, m, limits)
    for lang in ("es", "en"):
        tables.status_tables(status, lang, rec.set_index("column")["match at D %"].to_dict())
        tex = write(lang, cfg, res, rec, status, cov, limits, names)
        print("saved", report.compile_pdf(tex) if compile else tex)
    return status


if __name__ == "__main__":
    s = build()
    print(s["status"].value_counts().to_string())
    print("wrong:", ", ".join(s.loc[s["status"] == "wrong", "column"]) or "none")
