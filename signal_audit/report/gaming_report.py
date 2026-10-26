"""Figures (output/figures/gaming_*.png, g_*.png) and tables (reports/gaming_daily/gaming_*.tex) of the report of
gld_player_gaming_daily (reports/gaming_daily/gaming_daily_errors.tex), from the cache.

    .venv/bin/python -m signal_audit.report.gaming_report
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from signal_audit import settings
from signal_audit.checks import gaming, stats
from signal_audit.data import audit
from signal_audit.report import plots
from signal_audit.report.figures import one, save
from signal_audit.report.tables import RULE_EN, RULE_ES, _heat

TEX = settings.REPORTS / "gaming_daily"

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
    """Figures (output/figures/gaming_*.png, g_*.png) and tables (reports/gaming_daily/gaming_*.tex) of the gaming report."""
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



if __name__ == "__main__":
    gaming_report()
