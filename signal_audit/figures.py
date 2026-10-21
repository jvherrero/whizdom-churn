"""Every figure and table of the error list (informe/errores_tabla_senales.tex), from the cache only (no S3).
Figures go to figures/ (not in git); the tables go to informe/*.tex. Run the notebooks once with
RUN_S3 = True first.

    RUN_S3=0 .venv/bin/python signal_audit/figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "eda")]
import audit  # noqa: E402
import evidence as ev  # noqa: E402
import plots  # noqa: E402

OUT = HERE / "figures"
TEX = HERE / "informe"
LONG = {"start": "2026-06-01", "end": "2026-10-15"}   # the long period of the signals table
SECONDS = [(1, "1 s"), (10, "10 s"), (60, "1 min"), (600, "10 min"), (3600, "1 h"), (4 * 3600, "4 h"),
           (86400, "1 day"), (7 * 86400, "7 days"), (28 * 86400, "28 days")]
SHARE = [(0, "0"), (0.5, "50%"), (1, "100%"), (2, "200%"), (3, "300%"), (4, "400%")]


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=130)
    plt.close(fig)
    print("saved", f"{name}.png")


def one(name: str, size: tuple, draw, nrows: int = 1, ncols: int = 1, **kw):
    """A figure of nrows x ncols axes drawn by draw(ax or axes)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=size, **kw)
    draw(axes)
    save(fig, name)


def table_vs_recomputed(cfg, rc):
    """The signals table against the signal recomputed from every bet."""
    t, m, o = plots.TABLE_C, plots.MINE_C, plots.OTHER_C
    day = [(86400, 3.5e6)], [(4 * 3600, 86400)]
    specs = {
        "cmp_session_avg": ("Mean session length per player", [("signals table", cfg, "session_duration_avg_secs", t),
                            ("recomputed", rc, "rc_player_avg_session_secs", m)], SECONDS, (0.8, 3.5e6), *day),
        "cmp_session_median": ("Median session length per player", [("signals table", cfg, "median_session_duration_secs", t),
                               ("recomputed", rc, "rc_player_median_session_secs", m)], SECONDS, (0.8, 3.5e6), *day),
        "cmp_night": ("Share of play at night", [("table: tp_night_index", cfg, "night_index", t),
                      ("table: rg_night_play_ratio", cfg, "rg_night_play_ratio", o),
                      ("recomputed", rc, "rc_night_share", m)], SHARE, (-0.05, 4.2), [(1, 4.2)], []),
        "cmp_weekend": ("Share of play at the weekend", [("table: tp_weekend_index", cfg, "weekend_index", t),
                        ("recomputed", rc, "rc_weekend_share", m)], SHARE, (-0.05, 4.2), [(1, 4.2)], []),
        "cmp_entropy": ("Hourly entropy of play (0 = always the same hour, 1 = the same in every hour)",
                        [("table: tp_hourly_entropy", cfg, "hourly_entropy", t), ("recomputed", rc, "rc_hourly_entropy", m)],
                        [(0, "0"), (0.5, "0.5"), (1, "1"), (2, "2"), (3, "3"), (4, "4")], (-0.05, 4.2), [(1, 4.2)], []),
    }
    for name, (title, series, ticks, xlim, imp, susp) in specs.items():
        def draw(ax, series=series, ticks=ticks, xlim=xlim, imp=imp, susp=susp, title=title):
            plots.compare(series, ax, ticks, xlim, imp, susp)
            ax.set_title(title)
        one(name, (9.5, 4), draw)


def time_of_play(cfg, rc, flags):
    def scale(axes):
        for ax, (name, kind, title) in zip(axes, [("weekend_index", "weekend", "tp_weekend_index: values on 3.5/k (dotted lines)"),
                                                  ("night_index", "night", "tp_night_index: values on 4/k (dotted lines)"),
                                                  ("hourly_entropy", "entropy", "tp_hourly_entropy: values on log2 k (dotted lines)")]):
            plots.scale(cfg, name, kind, title, ax)
    one("tp_scale", (12, 9), scale, nrows=3)

    def rescaled(axes):
        for ax, (t, m, k, lab, title) in zip(axes, [("weekend_index", "rc_weekend_share", 2 / 7, "2/7", "Weekend share"),
                                                    ("hourly_entropy", "rc_hourly_entropy", 1 / 4.585, "1 / log2(24)", "Hourly entropy (0 to 1)"),
                                                    ("night_index", "rc_night_share_tod", 1 / 4, "6/24", "Night share")]):
            plots.rescaled(cfg, rc, t, m, k, lab, ax)
            ax.set_title(title)
    one("tp_rescaled", (15, 3.8), rescaled, ncols=3)
    one("tp_above_brands", (11, 3.6), lambda ax: plots.above_one_brands(cfg, ax))
    one("tp_daily", (12, 6.5), lambda axes: plots.tp_daily(flags, ev.repeated_days(flags), axes), nrows=2, sharex=True)


def sessions(cfg, rc):
    def brands(axes):
        plots.session_brands(cfg, rc, axes)
        axes[0].figure.suptitle("Mean session length per player, per brand (September 2026); outside = outside 10 s to 4 h", fontsize=11)
    one("session_brands", (17, 3.4), brands, ncols=5, sharey=True)
    one("session_rules", (11, 3.6), lambda ax: plots.session_rules(ev.session_checks(cfg), ax))


def recency():
    evidence, shifts = ev.recency_evidence()
    one("recency_zero", (12, 3.6), lambda ax: plots.zero_share(ev.recency_over_time(), ax))
    one("recency_shift", (12, 3.8), lambda ax: plots.shift_match(shifts, ax))
    one("recency_leak", (12, 3.9), lambda ax: plots.leak_proof(evidence, ax))
    one("recency_auc", (12, 3.9), lambda ax: plots.auc(evidence, ax))


def active_days_and_tenure():
    over = ev.active_days_over_time()

    def match(axes):
        for ax, col in zip(axes, ev.WINDOWS):
            plots.active_match(over, col, ax)
            ax.set_title(col)
    one("active_match", (16, 3.8), match, ncols=3)
    one("false_actives", (12, 3.8), lambda ax: plots.false_actives(ev.false_actives(), ax))
    one("tenure", (15, 3.8), lambda axes: plots.tenure(ev.tenure_growth(), axes), ncols=2)


def scores_hold_and_keys(cfg):
    _, hist = ev.index_limits(cfg)

    def limits(axes):
        for ax, c in zip(axes.flat, ["value_decline", "recency_pressure", "top_peak_hour", "top_peak_day"]):
            plots.index_limits(hist, c, ax)
    one("index_limits", (14, 6.5), limits, nrows=2, ncols=2)
    one("hold_history", (15, 3.6), lambda axes: plots.footer_extremes(ev.footer_extremes(), axes), ncols=2)
    one("hold_edge", (15, 4), lambda axes: plots.hold_edge(cfg, axes), ncols=2)
    one("hold_bets", (12, 4), lambda ax: plots.hold_by_bets(ev.hold_hist(cfg), ax))
    one("missing_brand_id", (10, 3.2), lambda ax: plots.brand_nulls(ev.brand_nulls(), ax))


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
    tables = dict.fromkeys(m["table"] for m in cfg["metrics"].values())
    return pd.concat([audit.flags_by_day(cfg, t, **(LONG if t == "gld_player_signals_daily" else {})) for t in tables],
                     ignore_index=True)


def flag_maps(cfg, flags):
    """One heatmap per metric and kind (impossible, suspicious) with any flagged value."""
    for kind, prefix in (("impossible", "days"), ("suspicious", "susp")):
        view = flags.drop(columns=[c for c in ("impossible", "suspicious") if c != kind]).rename(columns={kind: "impossible"})
        for name, g in view.groupby("metric"):
            if g["impossible"].sum() == 0:
                continue
            ax = plots.by_day(view, name, title=f"{LABEL[name]} ({cfg['metrics'][name][kind]}): % of {kind} values per day and brand")
            save(ax.figure, f"{prefix}_{name}")


def _heat(v: float) -> str:
    """A share as a table cell, coloured by size: the darker, the larger (0.1%, 1%, 5%, 10%, 20%)."""
    shade = next((c for t, c in [(20, 75), (10, 55), (5, 40), (1, 25), (0.1, 12)] if v >= t), 0)
    if not shade:
        return f"{v:.2f}"
    text = f"{{\\color{{white}}{v:.2f}}}" if shade >= 55 else f"{v:.2f}"
    return f"\\cellcolor{{BrickRed!{shade}}}{text}"


def flag_table(cfg, flags, lang: str = "es"):
    """informe/valores_raros.tex (Spanish) or informe/valores_raros_en.tex: per metric and kind, the rule,
    rows, the share in all brands and in each brand, and the days with any; impossible first, largest first."""
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
                         f"{days.min():%d/%m}--{days.max():%d/%m} & \\ref{{fig:{EXISTING.get(label, label)}}}\\\\"))
    out = TEX / ("valores_raros.tex" if lang == "es" else "valores_raros_en.tex")
    out.write_text("\n".join(r[2] for r in sorted(rows)) + "\n", encoding="utf-8")
    print("saved", out.name, len(rows), "rows")


# ---------------------------------------------------------------- status of every column

STATUS_TEX = {"wrong": "\\stWrong", "affected": "\\stAffected", "unconfirmed": "\\stUnconfirmed",
              "correct": "\\stCorrect", "unchecked": "\\stUnchecked"}
KIND_ES = {"formula": "fórmula", "identity": "identidad", "window": "ventana", "lag": "retardo", "bound": "límite",
           "band": "banda", "validated": "validada"}
KIND_EN = {"formula": "formula", "identity": "identity", "window": "window", "lag": "lag", "bound": "limit",
           "band": "band", "validated": "validated"}
STATUS_EN = {"wrong": "Wrong", "affected": "Affected by a wrong input", "unconfirmed": "Unconfirmed (no full definition)",
             "correct": "Correct (pass their checks)", "unchecked": "Not checkable (keys, text, metadata)"}
STATUS_ES = {"wrong": "Mal calculadas", "affected": "Afectadas por una entrada mal calculada",
             "unconfirmed": "Sin confirmar (falta la definición)", "correct": "Correctas (cumplen sus comprobaciones)",
             "unchecked": "No comprobables (claves, texto, metadatos)"}


def status_tables(status: pd.DataFrame, lang: str = "es"):
    """informe/resumen_columnas.tex (columns per status) and informe/columnas.tex (every column); with
    lang="en", the same tables with the suffix _en."""
    names, kinds, suffix = (STATUS_ES, KIND_ES, "") if lang == "es" else (STATUS_EN, KIND_EN, "_en")
    head = "Estado & & Columnas" if lang == "es" else "Status & & Columns"
    lines = ["\\begin{tabular}{@{}llr@{}}", "\\toprule", head + "\\\\", "\\midrule"]
    lines += [f"{STATUS_TEX[k]} & {label} & {int((status['status'] == k).sum())}\\\\" for k, label in names.items()]
    lines += ["\\midrule", f"Total & & {len(status)}\\\\", "\\bottomrule", "\\end{tabular}"]
    (TEX / f"resumen_columnas{suffix}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    order = {k: i for i, k in enumerate(STATUS_ES)}
    body = []
    for _, r in status.assign(o=status["status"].map(order)).sort_values(["o", "column"]).iterrows():
        pct = "" if pd.isna(r["worst %"]) else f"{r['worst %']:.1f}"
        checks = ", ".join(kinds[k] for k in r["checks"].split(", ")) if r["checks"] else "---"
        ref = f"\\ref{{{r['item']}}}" if r["item"] else ""
        body.append(f"\\code{{{r['column']}}} & {STATUS_TEX[r['status']]} & {checks} & {pct} & {ref}\\\\")
    (TEX / f"columnas{suffix}.tex").write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"saved resumen_columnas{suffix}.tex and columnas{suffix}.tex:", status["status"].value_counts().to_dict())


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    cfg = audit.load_config()
    rc = audit.load_config(HERE / "metrics_recomputed.yaml")
    flags = all_flags(cfg)
    table_vs_recomputed(cfg, rc)
    time_of_play(cfg, rc, flags)
    sessions(cfg, rc)
    recency()
    active_days_and_tenure()
    scores_hold_and_keys(cfg)
    flag_maps(cfg, flags)
    status = ev.column_status()
    for lang in ("es", "en"):
        flag_table(cfg, flags, lang)
        status_tables(status, lang)
