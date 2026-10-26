"""Every figure of the signals table, from the cache only (no S3), for the October 2026 report
(reports/signals_daily/) and the automatic report (report.py). The claim on top of each figure and how to read it
come from config/report_points.yaml, with the numbers of report.facts(). Figures go to OUT: output/figures/ for
the October 2026 report, output/runs/<run>/figures/ for a run.

    RUN_S3=0 .venv/bin/python -m signal_audit.report.figures     # figures and tables of the October 2026 report
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from signal_audit import settings  # noqa: E402
from signal_audit.checks import evidence as ev  # noqa: E402
from signal_audit.data import audit  # noqa: E402
from signal_audit.report import plots, tables  # noqa: E402

OUT = settings.OUTPUT / "figures"                 # the figures of the October 2026 report; a run sets its own
POINTS = yaml.safe_load((settings.CONFIG / "report_points.yaml").read_text(encoding="utf-8"))
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
    from signal_audit.report import report
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
    from signal_audit.checks import extra_checks as x
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
            ax = plots.by_day(view, name, title=f"{tables.LABEL[name]} ({re.sub(r'\\bx\\b', 'value', cfg['metrics'][name][kind])}): % of {kind} values per day and brand")
            save(ax.figure, f"{prefix}_{name}")


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


if __name__ == "__main__":   # the October 2026 report of the lake table
    from signal_audit.report import report
    cfg = audit.load_config()
    rc = audit.load_config(settings.CONFIG / "metrics_recomputed.yaml")
    flags = all_flags(cfg)
    f = report.facts(cfg, rc, flags)
    all_figures(cfg, rc, flags, f)
    status = ev.column_status()
    for lang in ("es", "en"):
        tables.flag_table(cfg, flags, lang)
        tables.status_tables(status, lang)
