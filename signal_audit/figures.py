"""Every figure and table of the error list (informe/errores_tabla_senales.tex) and of the automatic report
(report.py), from the cache only (no S3). The claim on top of each figure and how to read it come from
report_points.yaml, with the numbers of report.facts(). Figures go to OUT and the tables to TEX: figures/ and
informe/ for the October 2026 report, runs/<run>/ for the automatic one. Run the notebooks (or run_audit.py)
once with RUN_S3 = True first.

    RUN_S3=0 .venv/bin/python signal_audit/figures.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "eda")]
import audit  # noqa: E402
import evidence as ev  # noqa: E402
import plots  # noqa: E402

OUT = HERE / "figures"
TEX = HERE / "informe"
POINTS = yaml.safe_load((HERE / "report_points.yaml").read_text(encoding="utf-8"))
SECONDS = [(1, "1 s"), (10, "10 s"), (60, "1 min"), (600, "10 min"), (3600, "1 h"), (4 * 3600, "4 h"),
           (86400, "1 day"), (7 * 86400, "7 days"), (28 * 86400, "28 days")]
SHARE = [(0, "0"), (0.5, "50%"), (1, "100%"), (2, "200%"), (3, "300%"), (4, "400%")]


def save(fig, name: str):
    for ax in fig.axes:   # readable numbers on log y axes (0.1, 1, 10 instead of 10^-1, 10^0, 10^1)
        if ax.get_yscale() == "log":
            ax.yaxis.set_major_formatter(plots.FuncFormatter(plots.plain))
            ax.yaxis.set_minor_formatter(plots.NullFormatter())
    fig.tight_layout(rect=(0, 0, 1, getattr(fig, "_top", 1)))   # room for plots.head()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=130)
    plt.close(fig)
    print("saved", f"{name}.png")


def head(fig, name: str, f: dict):
    """The claim of the figure (by the verdict of its point) and how to read it, from report_points.yaml."""
    import report
    t = POINTS["figures"][name]
    claim = t["claim"] if isinstance(t["claim"], str) else t["claim"][f["verdicts"].get(t["point"], "fail")]
    plots.head(fig, report.fill(claim, f), report.fill(t["how"], f))


def one(name: str, size: tuple, draw, nrows: int = 1, ncols: int = 1, **kw):
    """A figure of nrows x ncols axes drawn by draw(ax or axes)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=size, **kw)
    draw(axes)
    save(fig, name)


def table_vs_recomputed(cfg, rc, f):
    """The signals table against the signal recomputed from every bet."""
    t, m = plots.TABLE_C, plots.MINE_C
    day = [(86400, 3.5e6, "over 1 day:\nimpossible")], [(4 * 3600, 86400, "4 h to\n1 day")]
    above = [(1, 4.2, "above 1: impossible for a share")]
    specs = {
        "cmp_session_avg": ([("signals table", cfg, "session_duration_avg_secs", t), ("recomputed", rc, "rc_player_avg_session_secs", m)],
                            SECONDS, (0.8, 3.5e6), *day, "mean session length per player", "outside 10 s to 4 h"),
        "cmp_session_median": ([("signals table", cfg, "median_session_duration_secs", t), ("recomputed", rc, "rc_player_median_session_secs", m)],
                               SECONDS, (0.8, 3.5e6), *day, "median session length per player", "outside 10 s to 4 h"),
        "cmp_night": ([("tp_night_index", cfg, "night_index", t)], SHARE, (-0.05, 4.2), above, [], "share of the stake at night", "above 1"),
        "cmp_weekend": ([("tp_weekend_index", cfg, "weekend_index", t)],
                        SHARE, (-0.05, 4.2), above, [], "share of the stake at the weekend", "above 1"),
        "cmp_entropy": ([("tp_hourly_entropy", cfg, "hourly_entropy", t)],
                        [(0, "0"), (0.5, "0.5"), (1, "1"), (2, "2"), (3, "3"), (4, "4")], (-0.05, 4.2), above, [], "hourly entropy", "above 1"),
    }
    for name, (series, ticks, xlim, imp, susp, xlabel, outside) in specs.items():
        def draw(ax, series=series, ticks=ticks, xlim=xlim, imp=imp, susp=susp, xlabel=xlabel, outside=outside, name=name):
            if name == "cmp_entropy":
                ax.set_yscale("log")
            plots.compare(series, ax, ticks, xlim, imp, susp, xlabel=xlabel, outside=outside,
                          ylabel="% of players in each bar" if "session" in name else "% of rows in each bar")
            head(ax.figure, name, f)
        one(name, (10, 5.2), draw)


def time_of_play(cfg, flags, f):
    def scale(axes):
        for ax, (name, kind, title) in zip(axes, [("weekend_index", "weekend", "tp_weekend_index: masses at 3.5 / k"),
                                                  ("night_index", "night", "tp_night_index: masses at 4 / k"),
                                                  ("hourly_entropy", "entropy", "tp_hourly_entropy: masses at log$_2$ k")]):
            plots.scale(cfg, name, kind, title, ax)
        head(axes[0].figure, "tp_scale", f)
    one("tp_scale", (10, 10.5), scale, nrows=3)


    def above(ax):
        plots.above_one_brands(cfg, ax)
        head(ax.figure, "tp_above_brands", f)
    one("tp_above_brands", (10, 4.6), above)

    def daily(axes):
        plots.tp_daily(flags, ev.repeated_days(flags), axes)
        head(axes[0].figure, "tp_daily", f)
    one("tp_daily", (10, 7.4), daily, nrows=2, sharex=True)


def sessions(cfg, rc, f):
    def brands(axes):
        plots.session_brands(cfg, rc, list(axes.flat))
        axes.flat[-1].axis("off")
        head(axes.flat[0].figure, "session_brands", f)
    one("session_brands", (11, 7.6), brands, nrows=2, ncols=3, sharey=True)

    def rules(ax):
        plots.session_rules(ev.session_checks(cfg), ax)
        head(ax.figure, "session_rules", f)
    one("session_rules", (10, 4.6), rules)


def recency(f):
    evidence, shifts = ev.recency_evidence()

    def zero(ax):
        plots.zero_share(ev.recency_over_time(), ax)
        head(ax.figure, "recency_zero", f)
    one("recency_zero", (10, 4.8), zero)

    def shift(ax):
        plots.shift_match(shifts, ax)
        head(ax.figure, "recency_shift", f)
    one("recency_shift", (10, 5.8), shift)

    def auc(ax):
        plots.auc(evidence, ax)
        head(ax.figure, "recency_auc", f)
    one("recency_auc", (10, 5.2), auc)


def active_days_and_tenure(f):
    over = ev.active_days_over_time()

    def match(ax):
        plots.active_match_bars(over, ax)
        head(ax.figure, "active_match", f)
    one("active_match", (10, 6), match)

    def false(ax):
        plots.false_actives(ev.false_actives(), ax)
        head(ax.figure, "false_actives", f)
    one("false_actives", (10, 5), false)

    def ten(ax):
        plots.tenure(ev.tenure_growth(), ax)
        head(ax.figure, "tenure", f)
    one("tenure", (10, 5), ten)


def scores_hold_and_keys(cfg, f):
    summary, hist = ev.index_limits(cfg)
    a = summary[summary["brand"] == "all"].set_index("column")

    def limits(axes):
        notes = {"value_decline": f"must be 0 to 100: {a.loc['value_decline', 'above %']:.2f}% of rows above 100",
                 "recency_pressure": f"exactly 100: {a.loc['recency_pressure', 'at upper limit %']:.1f}% of rows",
                 "top_peak_hour": f"0 (also midnight): {a.loc['top_peak_hour', 'zero %']:.0f}% of rows",
                 "top_peak_day": f"0 (also a real day): {a.loc['top_peak_day', 'zero %']:.0f}% of rows"}
        red = {"value_decline": (), "recency_pressure": (ev.BINS,), "top_peak_hour": (0,), "top_peak_day": (0,)}
        v = a.loc["value_decline"]
        for ax, c in zip(axes.flat, notes):
            plots.index_limits(hist, c, ax, red=red[c], note=notes[c],
                               above=(v["at upper limit %"], v["above %"], v["max"]) if c == "value_decline" else None)
        head(axes[0, 0].figure, "index_limits", f)
    one("index_limits", (11, 8), limits, nrows=2, ncols=2)

    def history(axes):
        plots.footer_extremes(ev.footer_extremes(), axes)
        head(axes[0].figure, "hold_history", f)
    one("hold_history", (11, 5), history, ncols=2)

    def edge(axes):
        plots.hold_edge(cfg, axes)
        head(axes[0].figure, "hold_edge", f)
    one("hold_edge", (11, 5.2), edge, ncols=2)

    def bets(ax):
        plots.hold_by_bets(ev.hold_hist(cfg), ax)
        head(ax.figure, "hold_bets", f)
    one("hold_bets", (10, 5.2), bets)

    def nulls(ax):
        plots.brand_nulls(ev.brand_nulls(), ax)
        head(ax.figure, "missing_brand_id", f)
    one("missing_brand_id", (10, 4), nulls)


def codes_and_categories(cfg, f):
    """Figures of the checks of the columns with no formula (extra_checks.py, 06_extra_checks.ipynb)."""
    import extra_checks as x
    values = x.categories(cfg)

    def products(axes):
        plots.sessions_by_product(x.sessions_by_product(cfg), axes)
        head(axes[0].figure, "sessions_products", f)
    one("sessions_products", (12, 5.2), products, ncols=3)

    def codes(ax):
        plots.empty_codes(values, ax)
        head(ax.figure, "empty_codes", f)
    one("empty_codes", (10, 5), codes)

    def cats(axes):
        for ax, c in zip(axes.flat, plots.CATEGORY_ORDER):
            plots.category_shares(values, c, ax)
        head(axes.flat[0].figure, "categories", f)
    one("categories", (12, 8.6), cats, nrows=2, ncols=3)


# ---------------------------------------------------------------- impossible and suspicious values, per brand and day

LABEL = {"session_duration_avg_secs": "Mean session length", "median_session_duration_secs": "Median session length",
         "session_gap_avg_hours": "Gap between sessions", "night_index": "tp_night_index", "weekend_index": "tp_weekend_index",
         "hourly_entropy": "tp_hourly_entropy", "hold_pct_l30d": "hold_pct_l30d", "turnover_player_day": "Turnover per day",
         "wagered_eur_l30d": "Stake in 30 days", "avg_bet_eur_l30d": "Mean bet", "deposit_size_eur": "Mean deposit",
         "deposits_per_day": "Deposits per day", "bets_per_active_day": "Bets per day", "sessions_per_active_day": "Sessions per day",
         "active_days_l30d": "Active days in 30 days", "withdraw_settlement_hours": "Time to pay a withdrawal"}
# Spanish names and rules for the table of the PDF (column names stay as they are)
LABEL_ES = {"session_duration_avg_secs": "Duración media de sesión", "median_session_duration_secs": "Mediana de sesión",
            "session_gap_avg_hours": "Hueco entre sesiones", "turnover_player_day": "Turnover por día",
            "wagered_eur_l30d": "Apostado en 30 días", "avg_bet_eur_l30d": "Apuesta media", "deposit_size_eur": "Depósito medio",
            "deposits_per_day": "Depósitos por día", "bets_per_active_day": "Apuestas por día",
            "sessions_per_active_day": "Sesiones por día", "active_days_l30d": "Días activos en 30 días",
            "withdraw_settlement_hours": "Pago de retiradas"}
RULE_ES = {"x > 86400": "más de 1 día", "x > 1": "mayor que 1", "x > 100": "más de 100\\%", "x > 30": "más de 30",
           "x < 0": "negativo", "x = 1 OR (x > 14400 AND x <= 86400)": "1 s, o de 4 a 24 h", "x > 0 AND x < 0.5": "menos de 30 min",
           "x > 1000000": "más de 1 M€", "x > 10000000": "más de 10 M€", "x > 10000": "más de 10\\,000 €",
           "x > 100000": "más de 100\\,000 €", "x > 50": "más de 50", "x > 20000": "más de 20\\,000", "x > 20": "más de 20",
           "x > 168": "más de 1 semana", "x < -1000": "menos de $-1000$\\%"}
RULE_EN = {"x > 86400": "over 1 day", "x > 1": "above 1", "x > 100": "over 100\\%", "x > 30": "over 30",
           "x < 0": "negative", "x = 1 OR (x > 14400 AND x <= 86400)": "1 s, or 4 to 24 h", "x > 0 AND x < 0.5": "under 30 min",
           "x > 1000000": "over 1 M€", "x > 10000000": "over 10 M€", "x > 10000": "over 10\\,000 €",
           "x > 100000": "over 100\\,000 €", "x > 50": "over 50", "x > 20000": "over 20\\,000", "x > 20": "over 20",
           "x > 168": "over 1 week", "x < -1000": "below $-1000$\\%"}
# maps already in the annex under another label
EXISTING = {"days_session_duration_avg_secs": "session_days", "days_night_index": "night_days", "days_hold_pct_l30d": "hold_days"}


def all_flags(cfg) -> pd.DataFrame:
    """Impossible and suspicious values per day and brand: the signals table over the long period of the run,
    the other tables over the period."""
    tables = dict.fromkeys(m["table"] for m in cfg["metrics"].values())
    long = {k: str(v) for k, v in audit.settings.LONG.items()}
    return pd.concat([audit.flags_by_day(cfg, t, **(long if t == audit.settings.TABLE else {})) for t in tables],
                     ignore_index=True)


def flag_maps(cfg, flags):
    """One heatmap per metric and kind (impossible, suspicious) with any flagged value."""
    for kind, prefix in (("impossible", "days"), ("suspicious", "susp")):
        view = flags.drop(columns=[c for c in ("impossible", "suspicious") if c != kind]).rename(columns={kind: "impossible"})
        for name, g in view.groupby("metric"):
            if g["impossible"].sum() == 0:
                continue
            ax = plots.by_day(view, name, title=f"{LABEL[name]} ({re.sub(r'\\bx\\b', 'value', cfg['metrics'][name][kind])}): % of {kind} values per day and brand")
            save(ax.figure, f"{prefix}_{name}")


def _heat(v: float) -> str:
    """A share as a table cell, coloured by size: the darker, the larger (0.1%, 1%, 5%, 10%, 20%)."""
    shade = next((c for t, c in [(20, 75), (10, 55), (5, 40), (1, 25), (0.1, 12)] if v >= t), 0)
    if not shade:
        return f"{v:.2f}"
    text = f"{{\\color{{white}}{v:.2f}}}" if shade >= 55 else f"{v:.2f}"
    return f"\\cellcolor{{BrickRed!{shade}}}{text}"


def flag_table(cfg, flags, lang: str = "es", existing: dict = EXISTING):
    """informe/valores_raros.tex (Spanish) or informe/valores_raros_en.tex: per metric and kind, the rule,
    rows, the share in all brands and in each brand, and the days with any; impossible first, largest first.
    `existing`: maps already in the report under another label."""
    rules = RULE_ES if lang == "es" else RULE_EN
    rows = []
    for name, g in flags.groupby("metric"):
        m = cfg["metrics"][name]
        for kind in ("impossible", "suspicious"):
            if not m.get(kind) or g[kind].sum() == 0:
                continue
            by_b = g.groupby("brand_id")[["rows", kind]].sum()
            cells = " & ".join(_heat(100 * by_b.loc[int(b), kind] / max(by_b.loc[int(b), "rows"], 1)) if int(b) in by_b.index else ""
                               for b in plots.BRAND_ORDER)
            days = g[g[kind] > 0]["day"]
            label = f"{'days' if kind == 'impossible' else 'susp'}_{name}"
            plain = LABEL_ES.get(name) if lang == "es" else (LABEL[name] if "_" not in LABEL[name] else None)
            name_es = plain or f"\\code{{{LABEL[name]}}}"
            typ = "\\stWrongFlag" if kind == "impossible" else "\\stSuspFlag"
            n = f"{int(g[kind].sum()):,}".replace(",", "\\,")
            total = 100 * g[kind].sum() / g["rows"].sum()
            rows.append((kind != "impossible", -total,
                         f"{name_es} & {typ} & {rules[m[kind]]} & {n} & {_heat(total)} & {cells} & {days.nunique()} & "
                         f"{days.min():%d/%m}--{days.max():%d/%m} & \\ref{{fig:{existing.get(label, label)}}}\\\\"))
    out = TEX / ("valores_raros.tex" if lang == "es" else "valores_raros_en.tex")
    out.write_text("\n".join(r[2] for r in sorted(rows)) + "\n", encoding="utf-8")
    print("saved", out.name, len(rows), "rows")


# ---------------------------------------------------------------- status of every column

STATUS_TEX = {"wrong": "\\stWrong", "affected": "\\stAffected", "unconfirmed": "\\stUnconfirmed",
              "correct": "\\stCorrect", "unchecked": "\\stUnchecked"}
KIND_ES = {"formula": "fórmula", "identity": "identidad", "window": "ventana", "lag": "retardo", "bound": "límite",
           "band": "banda", "validated": "validada", "format": "formato", "distribution": "distribución",
           "share": "proporción", "recomputed": "recalculada"}
KIND_EN = {"formula": "formula", "identity": "identity", "window": "window", "lag": "lag", "bound": "limit",
           "band": "band", "validated": "validated", "format": "format", "distribution": "distribution",
           "share": "share", "recomputed": "recomputed"}
STATUS_EN = {"wrong": "Wrong", "affected": "Affected by a wrong input", "unconfirmed": "Unconfirmed (no full definition)",
             "correct": "Pass (they pass their checks)", "unchecked": "Not checkable (keys, text, metadata)"}
STATUS_ES = {"wrong": "Mal calculadas", "affected": "Afectadas por una entrada mal calculada",
             "unconfirmed": "Sin confirmar (falta la definición)", "correct": "Pass (cumplen sus comprobaciones)",
             "unchecked": "No comprobables (claves, texto, metadatos)"}


def status_tables(status: pd.DataFrame, lang: str = "es", extra: dict | None = None):
    """informe/resumen_columnas.tex (columns per status) and, through check_tables, the annex of every column
    with its checks; with lang="en", the same tables with the suffix _en. The priority of each point comes from
    report_points.yaml."""
    names, kinds, suffix = (STATUS_ES, KIND_ES, "") if lang == "es" else (STATUS_EN, KIND_EN, "_en")
    head = "Estado & & Columnas" if lang == "es" else "Status & & Columns"
    lines = ["\\begin{tabular}{@{}llr@{}}", "\\toprule", head + "\\\\", "\\midrule"]
    lines += [f"{STATUS_TEX[k]} & {label} & {int((status['status'] == k).sum())}\\\\" for k, label in names.items()]
    lines += ["\\midrule", f"Total & & {len(status)}\\\\", "\\bottomrule", "\\end{tabular}"]
    (TEX / f"resumen_columnas{suffix}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    order = {k: i for i, k in enumerate(STATUS_ES)}
    # the priority of each point: wrong columns go by priority
    prio = {p["id"]: p["priority"] for p in POINTS["points"]}
    rank = {"A": 0, "M": 1, "B": 2, "Q": 3}
    status = status.assign(o=status["status"].map(order), p=status["item"].map(lambda i: rank.get(prio.get(i), 4)))
    # status of each column and the point of the list it belongs to, with its priority
    state = {r["column"]: STATUS_TEX[r["status"]] + (f"\\newline \\prio{prio[r['item']]}{{}} \\ref{{{r['item']}}}" if r["item"] in prio
                                                    else f"\\newline \\ref{{{r['item']}}}" if r["item"] else "")
             for _, r in status.iterrows()}
    check_tables(status.sort_values(["o", "p", "column"])["column"].tolist(), lang, state, extra)
    print(f"saved resumen_columnas{suffix}.tex and comprobaciones{suffix}.tex:", status["status"].value_counts().to_dict())


# kind: (Spanish name, what it checks, rule), (English name, what it checks, rule)
CHECK_KINDS = {
    "formula": (("Fórmula", "El valor es la fórmula de su medida del Excel (código M) sobre otras columnas de la misma fila.",
                 "$|x - f| \\le 0.01 + 0.001\\,|f|$"),
                ("Formula", "The value is the formula of its workbook measure (M code) on other columns of the same row.",
                 "$|x - f| \\le 0.01 + 0.001\\,|f|$")),
    "identity": (("Identidad", "Una relación contable entre columnas, por ejemplo apostado = dinero real + bono.",
                  "$|x - f| \\le 0.01 + 0.001\\,|f|$"),
                 ("Identity", "An accounting relation between columns, for example stake = real money + bonus.",
                  "$|x - f| \\le 0.01 + 0.001\\,|f|$")),
    "window": (("Ventana", "La ventana de $w$ días es la suma de su valor diario en esos $w$ días.",
                "$x_w(D) = \\sum_{d \\in (D-w,\\, D]} x_{\\text{hoy}}(d)$"),
               ("Window", "The window of $w$ days is the sum of its daily value over those $w$ days.",
                "$x_w(D) = \\sum_{d \\in (D-w,\\, D]} x_{\\text{today}}(d)$")),
    "lag": (("Retardo", "Una columna \\code{prior_*} es el valor de su columna $k$ días antes.", "$x^{\\text{prior}}(D) = x(D - k)$"),
            ("Lag", "A \\code{prior_*} column is the value of its column $k$ days before.", "$x^{\\text{prior}}(D) = x(D - k)$")),
    "bound": (("Límite", "Una condición que siempre debe cumplirse: valores posibles (por ejemplo, de 0 a 100) y orden entre ventanas.",
               "la condición es cierta, p.~ej. $x_{7d} \\le x_{30d}$"),
              ("Limit", "A condition that must always hold: possible values (for example 0 to 100) and the order between windows.",
               "the condition is true, e.g. $x_{7d} \\le x_{30d}$")),
    "band": (("Banda", "La banda corresponde a su score según los cortes del Excel.", "banda $j$ si $c_j \\le s < c_{j+1}$"),
             ("Band", "The band matches its score with the cutoffs of the workbook.", "band $j$ if $c_j \\le s < c_{j+1}$")),
    "validated": (("Validada", "Comparada jugador a jugador con su valor recalculado desde las tablas diarias.", "mismo valor"),
                  ("Validated", "Compared player by player with its value recalculated from the daily tables.", "same value")),
    "format": (("Formato", "El texto de un id o de un código sigue su formato: UUID, solo dígitos, código en mayúsculas, nunca vacío.",
                "la condición es cierta"),
               ("Format", "The text of an id or a code follows its format: UUID, digits only, a code in upper case, never empty.",
                "the condition is true")),
    "distribution": (("Distribución", "Una columna de categorías no se reparte por igual; si sus niveles tienen orden (valor, riesgo), bajan del primero al último.",
                      "se revisa en su figura"),
                     ("Distribution", "A categorical column is not spread evenly; if its levels have an order (value, risk), they fall from the first to the last.",
                      "checked on its figure")),
    "share": (("Proporción", "Un importe es una parte de otro: el impuesto, una parte del GGR.", "$0 < \\text{tax} \\le \\text{GGR}$"),
              ("Share", "An amount is a part of another one: the tax, a part of the GGR.", "$0 < \\text{tax} \\le \\text{GGR}$")),
    "recomputed": (("Recalculada", "La regla del Excel con sus entradas recalculadas desde las tablas diarias, jugador a jugador, el día $D$.",
                    "$|x - f| \\le 0.01 + 0.001\\,|f|$ en $D$"),
                   ("Recomputed", "The workbook rule with its inputs recomputed from the daily tables, player by player, on day $D$.",
                    "$|x - f| \\le 0.01 + 0.001\\,|f|$ on $D$")),
    "none": (("Sin comprobación", "Claves, texto y metadatos.", "---"), ("No check", "Keys, text and metadata.", "---")),
}


def check_tables(order: list[str], lang: str = "es", state: dict | None = None, extra: dict | None = None):
    """informe/criterios.tex (the kinds of check, with their rule and how many columns and checks have each)
    and informe/comprobaciones.tex (every column, in `order`, with its status and point (`state`) and every
    check with the exact rule and the share of rows of the two snapshots that pass it); with lang="en", the
    suffix _en. Each column shows its name, status and point once, on a grey or white band, and the share that
    passes is green (passes), orange (almost) or red (fails)."""
    import extra_checks
    import formula_check as fc
    cfg = fc.load()
    ks = fc.checks(cfg)
    res = fc.run(cfg).set_index(["column", "n_check"])["pass %"]
    if extra is None:   # format, share, distribution, validated; a catalogue of its own passes its own results
        extra = extra_checks.check_results(audit.load_config())
    i, suffix = (0, "") if lang == "es" else (1, "_en")
    lines = [f"{v[i][0]} & {v[i][1]} & {v[i][2]} & {ks.loc[ks['kind'] == k, 'column'].nunique()} & {int((ks['kind'] == k).sum())}\\\\"
             for k, v in CHECK_KINDS.items() if (ks["kind"] == k).any()]
    lines += ["\\midrule", f"Total & & & {ks['column'].nunique()} & {len(ks)}\\\\", "\\bottomrule"]   # \\input in a tabular cannot be followed by a rule
    (TEX / f"criterios{suffix}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    words = {"es": ("suma de", "en los", "días hasta $D$", "días antes", "bandas de", "cortes", "sin cortes en el Excel",
                    "recalculada desde las tablas diarias"),
             "en": ("sum of", "over the", "days up to $D$", "days before", "bands of", "cutoffs", "no cutoffs in the workbook",
                    "recalculated from the daily tables")}[lang]
    def sql(e: str) -> str:   # SQL in monospace, with its spaces (\\code drops them inside a table)
        for ch in "\\_%&#${}":
            e = e.replace(ch, "\\" + ch)
        for ch in ("\\_", ",", "(", "-", "]"):   # long names, calls and patterns may break there in a narrow column
            e = e.replace(ch, ch + "\\allowbreak{}")
        return f"\\texttt{{{e}}}"

    body = []
    for n, c in enumerate(order):
        band = "\\rowcolor{gray!12}" if n % 2 == 0 else ""     # one band per column
        for j, (_, k) in enumerate(ks[ks["column"] == c].iterrows()):
            kind = k["kind"]
            if kind in ("formula", "identity") and not isinstance(k["expr"], str):   # z-scores against the brand of the day
                g = "marca" if lang == "es" else "brand"
                rule = f"$= (x - \\bar x_{{\\text{{{g}}}}}) / s_{{\\text{{{g}}}}}$ ({k['measure']})"
            elif kind in ("formula", "identity"):
                rule = f"$=$ {sql(k['expr'])}"
            elif kind in ("bound", "format", "share"):
                rule = sql(k["expr"])
            elif kind == "recomputed":
                rule = f"$=$ {sql(k['expr'])}"
            elif kind == "distribution":
                rule = ("no uniforme; si tiene niveles, bajan del primero al último" if lang == "es"
                        else "not uniform; if it has levels, they fall from the first to the last")
            elif kind == "validated" and c == "bonus_ggr_eur_today":
                rule = ("$=$ suma de \\code{bonus_ggr_eur} de \\code{gld_player_gaming_daily} del jugador y día (\\% sobre los jugador-día con juego)" if lang == "es"
                        else "$=$ sum of \\code{bonus_ggr_eur} of \\code{gld_player_gaming_daily} for the player and day (\\% of the player-days with play)")
            elif kind == "window":
                rule = f"$=$ {words[0]} \\code{{{k['base']}}} {words[1]} {int(k['days'])} {words[2]}"
            elif kind == "lag":
                rule = f"$=$ \\code{{{k['source']}}}, {int(k['lag_days'])} {words[3]}"
            elif kind == "band":
                cut = k["cutoffs"] if isinstance(k["cutoffs"], list) else None
                rule = f"{words[4]} \\code{{{k['score']}}}, " + (f"{words[5]} {', '.join(map(str, cut))}" if cut else words[6])
            elif kind == "validated":
                rule = words[7]
            else:
                rule = "---"
            pct = res.get((c, k["n_check"]))
            if (pct is None or pd.isna(pct)) and kind in ("format", "share", "validated", "recomputed"):
                pct = extra.get(c)
            if kind == "distribution":   # checked by eye on the figure of the tiers and bands
                pct = "Fig.~\\ref{fig:categories}"
            if not isinstance(pct, str):
                colour = "ForestGreen!70!black" if pct is not None and pct >= 99 else "BurntOrange" if pct is not None and pct >= 95 else "BrickRed"
                pct = "" if pct is None or pd.isna(pct) else f"\\textcolor{{{colour}}}{{{pct:.2f}}}"
            name, where = (f"\\code{{{c}}}", (state or {}).get(c, STATUS_TEX["unchecked"])) if j == 0 else ("", "")
            body.append(f"{band}{name} & {where} & {CHECK_KINDS[kind][i][0]} & {rule} & {pct}\\\\")
    (TEX / f"comprobaciones{suffix}.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- report of gld_player_gaming_daily

def _gaming_flag_rows(gcfg, flags, lang):
    """Rows of the table of impossible and suspicious values of the gaming report."""
    rules = RULE_ES if lang == "es" else RULE_EN
    extra = {"x > 43200 AND x <= 86400": ("12 a 24 h", "12 to 24 h"), "x > 57600 AND x <= 86400": ("16 a 24 h", "16 to 24 h"),
             "x > 1 OR x < 0": ("fuera de 0 a 1", "outside 0 to 1"), "x <= 0": ("0 o negativo", "0 or negative"),
             "x > 100": ("más de 100", "over 100"),
             "x < -10": ("menos de $-10$", "below $-10$"), "x > 10": ("más de 10", "over 10")}
    rows = []
    for name, g in flags.groupby("metric"):
        m = gcfg["metrics"][name]
        for kind in ("impossible", "suspicious"):
            if not m.get(kind) or g[kind].sum() == 0:
                continue
            rule = extra[m[kind]][0 if lang == "es" else 1] if m[kind] in extra else rules.get(m[kind], m[kind])
            by_b = g.groupby("brand_id")[["rows", kind]].sum()
            cells = " & ".join(_heat(100 * by_b.loc[int(b), kind] / max(by_b.loc[int(b), "rows"], 1)) if int(b) in by_b.index else ""
                               for b in plots.BRAND_ORDER)
            days = g[g[kind] > 0]["day"]
            total = 100 * g[kind].sum() / g["rows"].sum()
            n = f"{int(g[kind].sum()):,}".replace(",", "\\,")
            typ = "\\stWrongFlag" if kind == "impossible" else "\\stSuspFlag"
            rows.append((kind != "impossible", -total, f"\\code{{{name}}} & {typ} & {rule} & {n} & {_heat(total)} & {cells} & "
                                                        f"{days.nunique()} & \\ref{{fig:g_{kind}_{name}}}\\\\"))
    return "\n".join(r[2] for r in sorted(rows)) + "\n"


def gaming_report():
    """Figures (figures/gaming_*.png, figures/g_*.png) and tables (informe/gaming_*.tex) of the gaming report."""
    import gaming
    gcfg = gaming.load_config()
    # checks
    checks = gaming.row_checks(gcfg)
    summary = gaming.checks_summary(checks)
    one("gaming_checks", (12, 0.32 * len(summary) + 1.5), lambda ax: plots.checks_heatmap(summary, ax))
    verdict = {"passes": "\\stCorrect", "almost": "\\stAffected", "fails": "\\stWrong", "rejected": "\\stRejected", "no data": "---"}
    src = {"es": {"workbook": "Excel", "hypothesis": "hipótesis"}, "en": {"workbook": "workbook", "hypothesis": "hypothesis"}}
    for lang in ("es", "en"):
        lines = [f"\\code{{{r['check']}}} & {src[lang][r['source']]} & {r['pass %']:.2f} & {r['higher %']:.2f} & "
                 f"{r['lower %']:.2f} & {verdict[r['verdict']]}\\\\" for _, r in summary.iterrows()]
        (TEX / f"gaming_checks{'' if lang == 'es' else '_en'}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # sports betting (providers 58 and 50) against casino
    by = gaming.anomaly_by(gcfg, "play_secs > 86400", "provider_code", top=10_000)
    sport = by["value"].isin(["58", "50"])
    t = pd.DataFrame({"rows": [by.loc[sport, "rows"].sum(), by.loc[~sport, "rows"].sum()],
                      "bad_rows": [by.loc[sport, "bad_rows"].sum(), by.loc[~sport, "bad_rows"].sum()]},
                     index=["sports betting", "casino"])
    one("gaming_sports", (12, 3.6), lambda axes: plots.sports_split(t, axes), ncols=2)
    names = {"es": ["Deporte (proveedores 58 y 50)", "Casino", "Total"], "en": ["Sports betting (providers 58 and 50)", "Casino", "Total"]}
    full = pd.concat([t, t.sum().to_frame("total").T])
    for lang in ("es", "en"):
        lines = [f"{n} & {r['rows']:,.0f} & {100 * r['rows'] / t['rows'].sum():.1f} & {r['bad_rows']:,.0f} & "
                 f"{100 * r['bad_rows'] / r['rows']:.2f} & {100 * r['bad_rows'] / t['bad_rows'].sum():.0f}\\\\"
                 for n, (_, r) in zip(names[lang], full.iterrows())]
        lines[2:2] = ["\\midrule"]
        lines.append("\\bottomrule")   # in the file: \input in a tabular cannot be followed by a rule
        (TEX / f"gaming_sports{'' if lang == 'es' else '_en'}.tex").write_text("\n".join(lines).replace(",", "\\," if lang == "es" else ",") + "\n", encoding="utf-8")
    one("gaming_first_play_hours", (12, 3.6), lambda ax: plots.first_play_hours(gaming.first_play_hours(gcfg), ax))
    # history from the metadata
    one("gaming_rows_per_day", (12, 3.2), lambda ax: plots.daily_rows(gaming.daily_rows(), ax))
    limits = {"play_secs": ("max", 86400), "won_real_eur": ("min", 0), "fx_rate_to_eur": ("min_or_equal", 0), "max_win_eur": ("min", 0)}
    ext = gaming.daily_extremes(list(limits))
    one("gaming_extremes", (14, 6.5), lambda axes: plots.daily_extremes(ext, list(limits), limits, axes.flat), nrows=2, ncols=2)
    # distributions
    import stats
    for name in ("row_play_secs", "pd_play_secs", "pd_play_over_span", "row_loyalty_points", "row_hold_pct"):
        prof, hist = audit.metric_data(gcfg, name)
        res = stats.fit(gcfg, name, prof, hist)
        fig, ax = plt.subplots(figsize=(9, 3.8))
        plots.metric(gcfg, res, prof, ax=ax)
        save(fig, f"gaming_{name}")
    # impossible and suspicious per day and brand
    flags = pd.concat([audit.flags_by_day(gcfg, t) for t in gcfg["tables"]], ignore_index=True)
    for kind in ("impossible", "suspicious"):
        view = flags.drop(columns=[c for c in ("impossible", "suspicious") if c != kind]).rename(columns={kind: "impossible"})
        for name, g in view.groupby("metric"):
            if g["impossible"].sum() == 0:
                continue
            ax = plots.by_day(view, name, title=f"{name} ({gcfg['metrics'][name][kind]}): % of {kind} values per day and brand")
            save(ax.figure, f"g_{kind}_{name}")
    for lang in ("es", "en"):
        (TEX / f"gaming_flags{'' if lang == 'es' else '_en'}.tex").write_text(_gaming_flag_rows(gcfg, flags, lang), encoding="utf-8")
    print("saved the gaming report tables")


def all_figures(cfg, rc, flags, f):
    """Every figure of the signals table into OUT."""
    table_vs_recomputed(cfg, rc, f)
    time_of_play(cfg, flags, f)
    sessions(cfg, rc, f)
    recency(f)
    active_days_and_tenure(f)
    scores_hold_and_keys(cfg, f)
    flag_maps(cfg, flags)
    codes_and_categories(cfg, f)


if __name__ == "__main__":   # the October 2026 report of the lake table: informe/errores_tabla_senales.tex
    import report
    cfg = audit.load_config()
    rc = audit.load_config(HERE / "metrics_recomputed.yaml")
    flags = all_flags(cfg)
    f = report.facts(cfg, rc, flags)
    all_figures(cfg, rc, flags, f)
    gaming_report()
    status = ev.column_status()
    for lang in ("es", "en"):
        flag_table(cfg, flags, lang)
        status_tables(status, lang)
