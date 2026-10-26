"""The automatic report of a signals table with the 165 columns of the lake: the 25 points of the error list,
each with its numbers and a verdict, in Spanish and English (runs/<run>/informe/auditoria.pdf and audit.pdf).

The points, their rules and texts are in report_points.yaml, the LaTeX in templates/points_*.tex, and the
figures and generated tables come from figures.py. From the cache only: run_audit.py fills it first.

    AUDIT_RUN=<run> RUN_S3=0 .venv/bin/python signal_audit/report.py
"""

from __future__ import annotations

import datetime as dt
import math
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import audit  # noqa: E402
import evidence as ev  # noqa: E402
import extra_checks as x  # noqa: E402
import figures  # noqa: E402
import formula_check as fc  # noqa: E402
import settings  # noqa: E402

POINTS = figures.POINTS
FIELD = re.compile(r"<<(\w+)(?::([^>]*))?>>")
MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
             "noviembre", "diciembre"]
WORDS = {
    "es": {"file": "auditoria", "source": {"lake": "el lake (S3)", "clickhouse": "ClickHouse, copia local por día con \\texttt{FINAL}"},
           "schema": "\\textbf{Columnas.} Faltan respecto al lake: {}. Sobran: {}.", "none": "ninguna",
           "groups": ("Puntos que fallan", "Puntos que pasan", "Sin confirmar: falta la definición"),
           "point": "Punto", "section": "Sección", "kind": ("Imposible", "Sospechoso")},
    "en": {"file": "audit", "source": {"lake": "the lake (S3)", "clickhouse": "ClickHouse, a local copy per day with \\texttt{FINAL}"},
           "schema": "\\textbf{Columns.} Missing against the lake: {}. Extra: {}.", "none": "none",
           "groups": ("Points that Fail", "Points that Pass", "Unconfirmed: No Definition"),
           "point": "Point", "section": "Section", "kind": ("Impossible", "Suspicious")},
}
VERDICT = {"fail": "\\stWrong", "pass": "\\stCorrect", "unconfirmed": "\\stUnconfirmed", "no data": "\\stUnchecked"}
PRIORITY = {"A": 0, "M": 1, "B": 2, "Q": 3}


# ---------------------------------------------------------------- text helpers, also used by check_report.py

def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def fill(text: str, values: dict) -> str:
    """Put the values into <<name:format>>; a missing number shows as --."""
    return FIELD.sub(lambda m: "--" if _missing(values[m[1]]) else format(values[m[1]], m[2] or ""), text)


def day(d: dt.date, lang: str) -> str:
    return f"{d.day} de {MONTHS_ES[d.month - 1]} de {d.year}" if lang == "es" else f"{d.day} {d:%b %Y}"


def days(ds, lang: str) -> str:
    return ", ".join(day(d, lang) for d in ds) or "--"


def codes(columns, lang: str) -> str:
    return ", ".join(f"\\code{{{c}}}" for c in columns) or WORDS[lang]["none"]


def span(r: dict, lang: str) -> str:
    """A period: the month when it is a whole month, otherwise its first and last day."""
    a, b = r["start"], r["end"]
    if a.day == 1 and (b + dt.timedelta(days=1)).day == 1 and (a.year, a.month) == (b.year, b.month):
        return f"{MONTHS_ES[a.month - 1]} de {a.year}" if lang == "es" else f"{a:%B %Y}"
    return f"{day(a, lang)} {'al' if lang == 'es' else 'to'} {day(b, lang)}"


def template(name: str, lang: str) -> str:
    """templates/<name>_<lang>.tex, with the preamble (packages and macros) of the October 2026 report."""
    source = "errores_tabla_senales.tex" if lang == "es" else "signal_table_errors.tex"
    preamble = (HERE / "informe" / source).read_text(encoding="utf-8").split("\\title{")[0].rstrip()
    return (HERE / "templates" / f"{name}_{lang}.tex").read_text(encoding="utf-8").replace("<<preamble>>", preamble)


def compile_pdf(tex: Path) -> Path:
    """xelatex twice (for the links); raises with the first LaTeX error."""
    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", tex.name], cwd=tex.parent, capture_output=True, text=True)
    errors = [line for line in r.stdout.splitlines() if line.startswith("!")]
    if errors or not tex.with_suffix(".pdf").exists():
        raise RuntimeError(f"{tex.name}: {errors[:3] or 'no PDF'} (see {tex.with_suffix('.log')})")
    return tex.with_suffix(".pdf")


# ---------------------------------------------------------------- numbers and verdicts

def _lo_hi(s) -> tuple[float, float]:
    s = pd.Series(s, dtype=float).dropna()
    return (float(s.min()), float(s.max())) if len(s) else (math.nan, math.nan)


def _pass(res: pd.DataFrame, columns, kinds) -> pd.Series:
    return res.loc[res["column"].isin(list(columns)) & res["kind"].isin(list(kinds)), "pass %"]


def facts(cfg: dict, rc: dict, flags: pd.DataFrame) -> dict:
    """Every number of the report (names as in report_points.yaml) and the verdict of each point."""
    f = {"table": settings.RUN["table"], "long_start": settings.LONG["start"]}
    for lang in ("es", "en"):
        f[f"period_{lang}"] = span(settings.PERIOD, lang)
        f[f"long_{lang}"] = span(settings.LONG, lang)
        f[f"period_start_{lang}"] = day(settings.PERIOD["start"], lang)

    lim, _ = ev.index_limits(cfg)
    a = lim[lim["brand"] == "all"].set_index("column")
    f |= {"vd_above": a.loc["value_decline", "above %"], "vd_max": a.loc["value_decline", "max"],
          "vd_at100": a.loc["value_decline", "at upper limit %"], "rp_at100": a.loc["recency_pressure", "at upper limit %"],
          "peak_hour0": a.loc["top_peak_hour", "zero %"], "peak_day0": a.loc["top_peak_day", "zero %"],
          "units_max": a.loc["churn_score", "max"]}
    for key, name in (("night", "night_index"), ("weekend", "weekend_index"), ("ent", "hourly_entropy")):
        p, _ = audit.metric_data(cfg, name)
        f[f"{key}_above"], f[f"{key}_max"] = 100 * p["above_range"] / p["n"], p["max"]

    holds = ev.session_checks(cfg).query("brand == 'all'").set_index("check")["holds %"]
    f |= {"se_over_day": holds["mean over 1 day"], "se_one_sec": holds["mean exactly 1 s"],
          "gap_neg": holds["gap between sessions < 0"], "ps_over_day": holds["play_secs today > 24 h"]}
    p, _ = audit.metric_data(cfg, "session_duration_avg_secs")
    f["se_max_days"], f["se_outside"] = p["max"] / 86400, 100 * (p["above_range"] + p["below_range"]) / p["n"]
    p, _ = audit.metric_data(rc, "rc_player_avg_session_secs")
    f["se_outside_rc"] = 100 * (p["above_range"] + p["below_range"]) / p["n"]
    f["gap_min"] = audit.metric_data(cfg, "session_gap_avg_hours")[0]["min"]
    sp = x.sessions_by_product(cfg)

    def share(rule, groups):
        g = sp[sp["rule"] == rule].groupby("grp")["impossible"].sum()
        return 100 * g.reindex(groups).fillna(0).sum() / g.sum() if g.sum() else math.nan
    f |= {"se_sports": share("mean session > 1 day", ["bets on sports"]),
          "ps_sports": share("play_secs_today > 24 h", ["sports only", "both"]),
          "gap_casino": share("gap between sessions < 0", ["casino only"])}

    frozen = ev.repeated_days(flags)
    tp = flags[(flags["metric"] == "night_index") & (flags["rows"] > 0)]["day"]
    first = tp.min() if len(tp) else None
    f |= {"tp_frozen_n": len(frozen), "tp_frozen": ", ".join(f"{d:%d/%m}" for d in frozen["day"]) or "--",
          "tp_first": first or dt.date.max}
    for lang in ("es", "en"):
        f[f"tp_first_{lang}"] = day(first, lang) if first else "--"

    e, shifts = ev.recency_evidence()
    med = shifts[shifts["shift"] >= 0].groupby("shift")["match %"].median() if len(shifts) else pd.Series(dtype=float)
    best = int(med.drop(0, errors="ignore").idxmax()) if len(med) > 1 else 0
    f |= {"rec_match0": med.get(0, math.nan), "rec_best_shift": best, "rec_best": med.get(best, math.nan)}
    auc = e.groupby("brand")[[c for c in e.columns if c.startswith("AUC")]].median() if len(e) else pd.DataFrame()
    f["rec_auc_table"] = auc["AUC next 7 days: table days_since_bet"].median() if len(auc) else math.nan
    f["rec_auc_real_lo"], f["rec_auc_real_hi"] = _lo_hi(auc.get("AUC next 7 days: recency as of the day"))
    f["churn_auc_lo"], f["churn_auc_hi"] = _lo_hi(auc.get("AUC next 7 days: churn_score"))
    zero = ev.recency_over_time()
    zero = zero.groupby("day")["table is 0 %"].median() if len(zero) else pd.Series(dtype=float)
    f |= {"rec_zero_months": int((zero >= 90).sum()), "rec_months": len(zero)}

    over = ev.active_days_over_time()
    by = over.groupby("column")[["= official %", "= any wallet %"]].median() if len(over) else {}
    f["ad_off_lo"], f["ad_off_hi"] = _lo_hi(by.get("= official %"))
    f["ad_any_lo"], f["ad_any_hi"] = _lo_hi(by.get("= any wallet %"))
    fa = ev.false_actives()
    f["fa_lo"], f["fa_hi"] = _lo_hi(fa["only playable bonus %"])
    f["fa_days_lo"], f["fa_days_hi"] = _lo_hi(fa["their mean table active days"])

    tg = ev.tenure_growth()
    share_ok = tg["share_growing_as_expected"] if len(tg) else pd.Series(dtype=float)
    f["ten_min"], f["ten_max"] = (100 * v for v in _lo_hi(share_ok))
    f["ten_last"] = 100 * share_ok[tg["day"] == tg["day"].max()].min() if len(tg) else math.nan
    f["ten_pairs"] = tg["day"].nunique() if len(tg) else 0

    res = fc.run(fc.load())
    f["churn_formula"] = _pass(res, ["churn_score"], ["formula"]).min()
    f["rgn_match"] = _pass(res, ["rg_night_play_ratio"], ["formula", "identity"]).min()
    f["deprec_formula"] = _pass(res, ["deposit_recency_score"], ["formula"]).min()
    f["ngr_lo"], f["ngr_hi"] = _lo_hi(_pass(res, ev.UNCONFIRMED["it:ngr"], ["formula", "identity"]))
    f["prior_lo"], f["prior_hi"] = _lo_hi(_pass(res, ev.UNCONFIRMED["it:prior"], ["lag"]))
    f["z_lo"], f["z_hi"] = _lo_hi(_pass(res, ev.UNCONFIRMED["it:z"], ["formula"]))

    nulls = audit.footer_nulls(settings.TABLE).loc["brand_id"]
    in_period = [d for d in nulls.index if settings.PERIOD["start"] <= pd.Timestamp(d).date() <= settings.PERIOD["end"]]
    f["brand_null"] = 100 * nulls[in_period].mean() if in_period else math.nan
    values = x.categories(cfg)
    for col in ("country", "language", "currency"):
        g = values[values["col"] == col]
        empty = g[g["value"].fillna("") == ""].groupby("brand_id")["rows"].sum()
        f[f"code_{col}"] = 100 * empty.sum() / g["rows"].sum()
        rows = g.groupby("brand_id")["rows"].sum()
        f[f"code_{col}_lo"], f[f"code_{col}_hi"] = _lo_hi(100 * empty.reindex(rows.index, fill_value=0) / rows)
    t = x.tax(cfg)
    f["tax_zero"], f["tax_above"] = 100 * t["tax_zero"].sum() / t["rows"].sum(), t["above_ggr"].sum()
    hold = audit.footer_range(settings.TABLE, "hold_pct_l30d")["max_num"]
    f |= {"hold_rows": flags.loc[flags["metric"] == "hold_pct_l30d", "impossible"].sum(),
          "hold_days": int((hold > 100).sum()), "hold_max": hold.max()}
    dep = ev.deposit_frequency_filled()
    f["dep_empty"], f["dep_months"] = int((dep == 0).sum().sum()), int(dep.notna().sum().sum())

    f["verdicts"] = verdicts(f)
    return f


def verdicts(f: dict) -> dict[str, str]:
    """fail or pass by the rule of each point; unconfirmed without a rule; no data when a number is missing."""
    out = {}
    for p in POINTS["points"]:
        rule = p.get("fail_if")
        names = set(re.findall(r"[a-z_][a-z0-9_]*", rule or "")) - {"or", "and", "not", "max", "min"}
        if not rule:
            out[p["id"]] = "unconfirmed"
        elif any(_missing(f.get(n)) for n in names):
            out[p["id"]] = "no data"
        else:
            out[p["id"]] = "fail" if eval(rule, {"__builtins__": {}, "max": max, "min": min}, f) else "pass"
    return out


# ---------------------------------------------------------------- the report

def _text(v, lang: str) -> str:
    return v[lang] if isinstance(v, dict) else v


def _figure_names() -> list[tuple[str, str]]:
    """(figure, point) of the figures made, in the order of the points, then the tiers and bands."""
    names = {}
    for p in POINTS["points"]:
        for n in p["figures"]:
            names.setdefault(n, p["id"])
    names["categories"] = ""
    return [(n, i) for n, i in names.items() if (figures.OUT / f"{n}.png").exists()]


def _flag_maps(cfg: dict, lang: str) -> list[str]:
    """The maps per day and brand of figures.flag_maps, with their captions."""
    w = WORDS[lang]
    rules, labels = (figures.RULE_ES, figures.LABEL_ES) if lang == "es" else (figures.RULE_EN, {})
    out = []
    for png in sorted(figures.OUT.glob("days_*.png")) + sorted(figures.OUT.glob("susp_*.png")):
        kind, name = png.stem.split("_", 1)
        rule = cfg["metrics"][name]["impossible" if kind == "days" else "suspicious"]
        label = labels.get(name) or (figures.LABEL[name] if "_" not in figures.LABEL[name] else f"\\code{{{figures.LABEL[name]}}}")
        out.append(f"\\evidence{{fig:{png.stem}}}{{0.72}}{{{png.name}}}{{\\textbf{{{w['kind'][kind != 'days']}: {label}, "
                   f"{rules.get(rule, rule)}.}} {w['section']}~\\ref{{sec:flags}}.}}")
    return out


def _points(f: dict, lang: str) -> str:
    """The points: those that fail by priority, then those that pass, then the unconfirmed; one numbering."""
    v = f["verdicts"]
    made = {n for n, _ in _figure_names()}

    def block(p):
        fails = v[p["id"]] == "fail"
        badge = f"\\prio{p['priority']}" if fails else VERDICT[v[p["id"]]]
        evidence = ", ".join(f"\\fig{{fig:{n}}}" for n in p["figures"] if n in made)
        return (f"\\auditpoint{{{p['id']}}}{{{badge}\\ {_text(p['title'], lang)}}}{{{fill(p['result'][lang], f)}}}"
                f"{{{p['definition'][lang]}}}{{{p['fix'][lang] if fails else ''}}}{{{evidence}}}")

    groups = [sorted((p for p in POINTS["points"] if v[p["id"]] == "fail"), key=lambda p: PRIORITY[p["priority"]]),
              [p for p in POINTS["points"] if v[p["id"]] in ("pass", "no data")],
              [p for p in POINTS["points"] if v[p["id"]] == "unconfirmed"]]
    lines, first = [], True
    for title, ps in zip(WORDS[lang]["groups"], groups):
        if ps:
            series = "label=\\textbf{\\arabic*.},ref=\\arabic*,leftmargin=*,series=issues" if first else "resume*=issues"
            lines += [f"\\section{{{title}}}", f"\\begin{{enumerate}}[{series}]", *map(block, ps), "\\end{enumerate}", ""]
            first = False
    return "\n".join(lines)


def write(cfg: dict, f: dict, lang: str) -> Path:
    """runs/<run>/informe/auditoria.tex (es) or audit.tex (en)."""
    w, v = WORDS[lang], f["verdicts"]
    rules = [p for p in POINTS["points"] if p.get("fail_if")]
    schema = ""
    if settings.RUN["source"] == "clickhouse":
        import clickhouse_mirror
        diff = clickhouse_mirror.schema_differences()
        schema = w["schema"].format(codes(diff["missing"], lang), codes(diff["extra"], lang))

    def row(p):
        badge = f"\\prio{p['priority']}{{}} " if v[p["id"]] == "fail" else ""
        return f"\\ref{{{p['id']}}} & {badge}{_text(p['name'], lang)} & {p['rule'][lang]} & {VERDICT[v[p['id']]]}\\\\"

    def figure(name, point):
        caption = POINTS["figures"][name]["caption"][lang] + (f" {w['point']}~\\ref{{{point}}}." if point else "")
        return f"\\evidence{{fig:{name}}}{{1.0}}{{{name}.png}}{{{caption}}}"

    months = settings.MONTH_STARTS
    values = f | {
        "today": day(dt.date.today(), lang), "source": w["source"][settings.RUN["source"]],
        "brands": ", ".join(map(str, cfg["brands"])), "snapshots": days(settings.SNAPSHOTS, lang),
        "evidence": days(settings.EVIDENCE_DAYS, lang),
        "months": f"{day(months[0], lang)} -- {day(months[-1], lang)}" if months else "--", "schema": schema,
        "n_fail": sum(v[p["id"]] == "fail" for p in rules), "n_pass": sum(v[p["id"]] == "pass" for p in rules),
        "n_rules": len(rules), "n_high": sum(v[p["id"]] == "fail" and p["priority"] == "A" for p in rules),
        "n_unconfirmed": sum(s == "unconfirmed" for s in v.values()),
        "points_table": "\n".join(map(row, POINTS["points"])), "points": _points(f, lang),
        "figures": "\n".join([figure(n, i) for n, i in _figure_names()] + _flag_maps(cfg, lang))}
    out = settings.TEX / f"{w['file']}.tex"
    out.write_text(fill(template("points", lang), values), encoding="utf-8")
    return out


def build(compile: bool = True) -> dict:
    """Figures, tables and the report (es, en) of the run, from the cache. Returns the facts."""
    figures.OUT, figures.TEX = settings.FIGURES, settings.TEX
    settings.TEX.mkdir(parents=True, exist_ok=True)
    cfg = audit.load_config()
    rc = audit.load_config(HERE / "metrics_recomputed.yaml")
    flags = figures.all_flags(cfg)
    f = facts(cfg, rc, flags)
    figures.all_figures(cfg, rc, flags, f)
    status = ev.column_status({k for k, s in f["verdicts"].items() if s == "fail"})
    for lang in ("es", "en"):
        figures.flag_table(cfg, flags, lang, existing={})
        figures.status_tables(status, lang)
        tex = write(cfg, f, lang)
        print("saved", compile_pdf(tex) if compile else tex)
    return f


if __name__ == "__main__":
    f = build()
    for p in POINTS["points"]:
        print(f"{f['verdicts'][p['id']]:12} {p['id']}")
