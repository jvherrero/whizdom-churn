"""Every figure of the signal audit, for the notebooks and for figures.py (the PDF). English labels.
Each function draws on an axis (or returns a figure) from aggregates; none reads S3 by itself."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.ndimage import gaussian_filter1d

from signal_audit.data import audit
from signal_audit.checks import evidence as ev
from signal_audit import settings
from signal_audit.checks import stats

IMPOSSIBLE = "#c0392b"
SUSPICIOUS = "#e08e2b"
# one meaning per colour in every figure: red = the signals table, blue = recomputed from each bet,
# yellow = a third series; grey = reference. Brands keep their colour in every figure.
TABLE_C, MINE_C, OTHER_C, LIGHT_C, REF_C = "#d03b2f", "#2a78d6", "#eda100", "#9fc3ea", "#8c8c8c"
BRAND_ORDER = ["14", "23", "64", "72", "73"]
BRAND_C = dict(zip(BRAND_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, MUTED = "#222222", "#5a5a5a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#cfcfcf", "grid.linewidth": 0.7, "axes.edgecolor": "#9a9a9a", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "axes.titlesize": 10.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 8.5})


def head(fig, claim: str, how: str) -> None:
    """The finding (bold) and how to read the figure (grey, may hold $math$) above the plots. save() of
    figures.py leaves the room: fig._top is the top of the plotting area."""
    import re
    import textwrap
    h, w = fig.get_figheight(), fig.get_figwidth()

    def wrap(text: str, width: int) -> str:   # wrap by words, never inside $math$
        text = re.sub(r"\$[^$]*\$", lambda m: m.group(0).replace(" ", "\u00a0"), text)
        return "\n".join(textwrap.fill(line, width) for line in text.split("\n")).replace("\u00a0", " ")
    claim, how = wrap(claim, int(w * 9.5)), wrap(how, int(w * 13.5))
    n_claim, n_how = claim.count("\n") + 1, how.count("\n") + 1
    fig._top = 1 - (0.3 + 0.27 * n_claim + 0.21 * n_how) / h
    fig.text(0.01, 1 - 0.08 / h, claim, ha="left", va="top", fontsize=13, weight="bold", color=INK, linespacing=1.3)
    fig.text(0.01, 1 - (0.16 + 0.27 * n_claim) / h, how, ha="left", va="top", fontsize=10, color=MUTED, linespacing=1.45)

def zone(ax, lo: float, hi: float, kind: str, label: str = "") -> None:
    """A shaded impossible (red) or suspicious (orange) zone of the x axis, named at the top."""
    colour = IMPOSSIBLE if kind == "impossible" else SUSPICIOUS
    ax.axvspan(lo, hi, color=colour, alpha=0.08 if kind == "impossible" else 0.10, lw=0)
    if label:
        x = (lo * hi) ** 0.5 if ax.get_xscale() == "log" and lo > 0 else (lo + hi) / 2
        ax.text(x, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5,
                color=colour, weight="bold")

def _month_axis(ax, interval: int = 2):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
DURATIONS = [(1, "1 s"), (10, "10 s"), (60, "1 min"), (600, "10 min"), (3600, "1 h"), (6 * 3600, "6 h"),
             (86400, "1 day"), (30 * 86400, "1 month"), (365 * 86400, "1 year"),
             (10 * 365 * 86400, "10 years"), (100 * 365 * 86400, "100 years")]


def plain(x: float, _=None) -> str:
    """0.001, 0.1, 1, 10, 1k instead of 10^-3, 10^-1... on a log axis."""
    return _short(x) if x >= 1 else f"{x:.6f}".rstrip("0")

def _short(x: float, _=None) -> str:
    """1, 10, 1k, 10M, 1bn instead of 10^0, 10^1, 10^3..."""
    for v, u in ((1e12, "tn"), (1e9, "bn"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= v:
            return f"{x / v:g}{u}"
    return f"{x:g}"

def _log_x(ax, secs: bool = False):
    """Log x axis with readable labels: durations (1 min, 1 h, 1 day) for seconds, 1k / 1M for the rest."""
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    if secs:
        lo, hi = ax.get_xlim()
        ticks = [(v, t) for v, t in DURATIONS if lo <= v <= hi]
        ax.set_xticks([v for v, _ in ticks], [t for _, t in ticks])
        ax.set_xlim(lo, hi)
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(_short))

def _outside(h: pd.DataFrame, m: dict) -> pd.Series:
    """Bins fully outside the real-life range (a bin that touches the limit counts as inside)."""
    lo, hi = m["range"]
    if not m.get("known_range", True):
        return pd.Series(False, index=h.index)
    if m["scale"] == "int":
        return (h["x"] < lo) | (h["x"] > hi)
    return (h["hi"] <= lo) | (h["lo"] > hi)

def _shade_impossible(ax, m: dict) -> None:
    """Shade the values outside the real-life range (none when `known_range: false`)."""
    if not m.get("known_range", True):
        return
    lo, hi = m["range"]
    left, right = ax.get_xlim()
    if lo > left:
        ax.axvspan(left, lo, color=IMPOSSIBLE, alpha=0.08, lw=0)
    if hi < right:
        ax.axvspan(hi, right, color=IMPOSSIBLE, alpha=0.08, lw=0)
    for v in (lo, hi):
        if left < v < right:
            ax.axvline(v, color=IMPOSSIBLE, ls=":", lw=1)
    ax.set_xlim(left, right)

def metric(cfg: dict, result: dict, prof: pd.Series | None = None, ax=None):
    """Observed histogram, fitted curves and peaks. Values outside the real-life range are red, on a
    shaded background; with `prof`, the exact number of impossible values is written on the plot."""

    name = result["metric"]
    m = cfg["metrics"][name]
    h = result["hist"]
    out = _outside(h, m)
    colours = np.where(out, IMPOSSIBLE, "#8fb3c7")
    ax = ax or plt.subplots(figsize=(8.5, 3.8))[1]
    if m["scale"] == "int":
        ax.bar(h["x"], h["share"], width=0.8, color=colours, label="observed")
        for f, colour in zip(result["fits"], ("#e08e2b", "#2f6f8f")):
            ax.plot(h["x"], f["pmf"], "o-", ms=3, color=colour, label=f"{f['family']} (KS {f['ks']:.3f})")
    else:
        ax.bar(h["x"] if m["scale"] != "log" else h["lo"], h["share"],
               width=(h["hi"] - h["lo"]) if m["scale"] == "log" else m["width"] * 0.95,
               align="edge" if m["scale"] == "log" else "center", color=colours, label="observed")
        for f, colour in zip(result["fits"], ("#e08e2b", "#2f6f8f")):
            lo0 = f["dist"].cdf(h["lo"].iloc[0])
            p = (f["dist"].cdf(h["hi"]) - f["dist"].cdf(h["lo"])) / max(1 - lo0, 1e-12)
            ax.plot(h["x"], p, color=colour, label=f"{f['family']} (KS {f['ks']:.3f})")
        if m["scale"] == "log":
            _log_x(ax, "secs" in name)
    for x in result["peaks"]:
        ax.axvline(x, color="black", ls="--", lw=0.8)
    _shade_impossible(ax, m)
    if m.get("log_y"):
        ax.set_yscale("log")
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    keep = [(hd, lb) for hd, lb in zip(handles, labels) if lb != "observed"]
    keep = [(Patch(color="#8fb3c7"), "observed")] + keep
    if out.any():
        keep.append((Patch(color=IMPOSSIBLE), "outside the real-life range"))
    if prof is not None and m.get("known_range", True):
        n_out = int(prof["below_range"] + prof["above_range"])
        lo, hi = m["range"]
        text = (f"impossible: {n_out:,} rows ({100 * n_out / max(prof['n'], 1):.2f}%)" if n_out
                else "no impossible values")
        ax.text(0.99, 0.97, f"range [{lo:g}, {hi:g}]\n{text}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=IMPOSSIBLE if n_out else "black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    ax.set(title=f"{name}: {len(result['peaks'])} peak(s)", ylabel="share of rows" + (" (log)" if m.get("log_y") else ""))
    ax.legend(*zip(*keep), fontsize=7, loc="upper left")
    return ax

def brands(cfg: dict, name: str, segment: str = "brand_id", ncols: int = 3):
    """One panel per brand: histogram, fits, peaks and values outside the range (as plot_metric)."""

    res = stats.brand_results(cfg, name, segment)
    nrows = int(np.ceil(len(res) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.6 * nrows), squeeze=False)
    for ax, (b, (prof, r)) in zip(axes.flat, res.items()):
        metric(cfg, r, prof, ax=ax)
        ax.set_title(f"{name}, brand {b}: {len(r['peaks'])} peak(s)", fontsize=10)
    for ax in list(axes.flat)[len(res):]:
        ax.axis("off")
    fig.tight_layout()
    return fig

def by_day(by_day: pd.DataFrame, name: str, ax=None, title: str | None = None,
                           all_label: str = "all brands", brand_label: str = "brand {b}"):
    """Heatmap of the share of impossible values per day (columns) and brand (rows), with a first row
    for all the brands together. White is 0%; grey is a day with no rows."""
    import matplotlib.dates as mdates

    g = by_day[by_day["metric"] == name]
    days = pd.date_range(g["day"].min(), g["day"].max()).date
    per = g.groupby(["brand_id", "day"])[["rows", "impossible"]].sum()
    total = g.groupby("day")[["rows", "impossible"]].sum()
    labels, grid = [all_label], [(total["impossible"] / total["rows"].where(total["rows"] > 0)).reindex(days)]
    for b in sorted(g["brand_id"].unique()):
        t = per.loc[b]
        labels.append(brand_label.format(b=int(b)))
        grid.append((t["impossible"] / t["rows"].where(t["rows"] > 0)).reindex(days))
    z = 100 * np.vstack([np.asarray(r, dtype=float) for r in grid])
    ax = ax or plt.subplots(figsize=(10, 0.45 * len(labels) + 1.4))[1]
    cmap = LinearSegmentedColormap.from_list("imp", ["#ffffff", "#f5b7a8", IMPOSSIBLE, "#6e1a10"])
    cmap.set_bad("#d9d9d9")
    x0, x1 = mdates.date2num(days[0]), mdates.date2num(days[-1]) + 1
    im = ax.imshow(np.ma.masked_invalid(z), aspect="auto", cmap=cmap, vmin=0, vmax=np.nanmax(z) or 1,
                   extent=(x0, x1, len(labels), 0), interpolation="nearest")
    ax.set_yticks(np.arange(len(labels)) + 0.5, labels)
    ax.axhline(1, color="black", lw=0.8)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    plt.colorbar(im, ax=ax, pad=0.01, label="% of the rows of that\nbrand and day")
    ax.set_xlabel("day (grey: no data that day)")
    ax.grid(False)
    ax.set_title(title or f"{name}: impossible values per day and brand (%)")
    return ax

def segments(cfg: dict, name: str, seg: pd.DataFrame, ax=None):
    """One curve per segment (largest ones), to see which group makes a second peak."""

    m = cfg["metrics"][name]
    ax = ax or plt.subplots(figsize=(8, 3.6))[1]
    top = seg.groupby("segment")["count"].sum().nlargest(cfg["thresholds"]["max_segments"]).index
    for s in top:
        h = stats.complete(seg[seg["segment"] == s][["bin", "count"]], m)
        n = len(stats.count_peaks(h, cfg))
        ax.plot(h["x"], gaussian_filter1d(h["share"].to_numpy(float), cfg["thresholds"]["smooth_bins"]),
                label=f"{s} ({h['count'].sum():,} rows, {n} peak(s))")
    if m["scale"] == "log":
        _log_x(ax, "secs" in name)
    _shade_impossible(ax, m)
    ax.set(title=f"{name} by segment", ylabel="share of rows (smoothed)")
    ax.legend(fontsize=7)
    return ax

# ---------------------------------------------------------------- evidence of the error list

def _bins(cfg: dict, name: str, scale: float = 1.0, drop_zero: bool = False):
    """Bin centres (times `scale`) and counts of a cached linear-scale metric."""
    m = cfg["metrics"][name]
    _, hist = audit.metric_data(cfg, name)
    h = stats.complete(hist, m | {"plot_range": [-1e9, 1e9]})
    x, c = h["x"].to_numpy(float) * scale, h["count"].to_numpy(float)
    if drop_zero:
        keep = h["bin"].to_numpy() != 0
        x, c = x[keep], c[keep]
    return x, c


def scale(cfg: dict, name: str, kind: str, title: str, ax):
    """Histogram of the table values (0 left out); bars above 1 (impossible for a share) in red, and the
    biggest masses named by the formula that gives them."""
    x, c = _bins(cfg, name, drop_zero=True)
    ax.bar(x, 100 * c / c.sum(), width=0.02, color=np.where(x > 1, TABLE_C, LIGHT_C))
    p, _ = audit.metric_data(cfg, name)
    zone(ax, 1, 4.7, "impossible")
    ax.text(0.99, 0.55, f"red zone, above 1:\nimpossible for a share\n({100 * p['above_range'] / p['n']:.1f}% of the rows)",
            transform=ax.transAxes, ha="right", va="center", fontsize=9, color=IMPOSSIBLE, weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    marks = {"weekend": [(3.5, "3.5 / 1"), (1.75, "3.5 / 2"), (3.5 / 3, "3.5 / 3")],
             "night": [(4, "4 / 1"), (2, "4 / 2"), (4 / 3, "4 / 3")],
             "entropy": [(1, "log$_2$ 2"), (np.log2(3), "log$_2$ 3"), (2, "log$_2$ 4")]}[kind]
    y = 100 * c / c.sum()
    for v, lab in marks:
        top = y[np.abs(x - v) < 0.015].max(initial=0)
        ax.annotate(lab, (v, top), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    ax.set(title=title, ylabel="% of rows (log scale)", yscale="log", xlim=(-0.05, 4.7), xlabel="value in the table")
    ax.set_ylim(top=y.max() * 40)


def live_line(ax, y: float, ha: str = "left"):
    """The day the daily process starts (settings.LIVE_FROM), when the run knows it: before it, the history
    was rebuilt in one go (backfill)."""
    if settings.LIVE_FROM is None:
        return
    day = pd.Timestamp(settings.LIVE_FROM)
    ax.axvline(day, color=INK, lw=0.9)
    ax.text(day, y, f"  {day:%-d %b %Y}: the daily process starts  \n  before: history rebuilt in one go (backfill)  ",
            fontsize=8, va="center", ha=ha, color=INK, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})

def zero_share(over_time: pd.DataFrame, ax):
    for b, g in over_time.groupby("brand"):
        g = g.sort_values("day")
        ax.plot(pd.to_datetime(g["day"]), g["table is 0 %"], marker="o", ms=3.5, lw=1.6, color=BRAND_C[str(b)], label=f"brand {b}")
    live_line(ax, 55)
    _month_axis(ax)
    ax.set(ylabel="% of active players\nwith days_since_bet = 0", xlabel="monthly snapshot (1st of each month)", ylim=(-2, 104))
    ax.legend(loc="lower left", ncol=5)

def shift_match(shifts: pd.DataFrame, ax):
    """% of players whose days_since_bet of day D equals the real recency of day D + s, for shifts s >= 0:
    s = 0 is the correct value, s > 0 needs data after D (the future)."""
    shifts = shifts[shifts["shift"] >= 0]
    med = shifts.groupby("shift")["match %"].median()
    hi = med.index.max() + 0.5
    ax.axvspan(0.5, hi, color=IMPOSSIBLE, alpha=0.07, lw=0)
    ax.text((0.5 + hi) / 2, 116, "shift > 0: needs data from after day D (the future)", ha="center", va="top", fontsize=9,
            color=IMPOSSIBLE, weight="bold")
    for b, g in shifts.groupby("brand"):
        m = g.groupby("shift")["match %"].median()
        ax.plot(m.index, m.to_numpy(), color=REF_C, lw=0.8, alpha=0.7)
    best = med.drop(0, errors="ignore").idxmax() if len(med) > 1 else 0
    ax.bar(med.index, med.to_numpy(), color=np.where(med.index == 0, MINE_C, np.where(med.index == best, TABLE_C, LIGHT_C)), width=0.75)
    for k, v in med.items():
        ax.text(k, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK)
    ax.annotate("shift = 0: the correct value\n(a correct column would be ~100%)", (0, med.get(0, 0) + 2), xytext=(0.6, 45),
                fontsize=8.5, color=MINE_C, arrowprops={"arrowstyle": "->", "color": MINE_C, "lw": 0.9})
    if med[best] > med.get(0, 0):
        ax.annotate(f"the table: shift = {best} days", (best - 0.4, med[best]), xytext=(max(best - 3.8, 1), med[best] - 8), fontsize=8.5,
                    color=TABLE_C, arrowprops={"arrowstyle": "->", "color": TABLE_C, "lw": 0.9})
    ax.set_xticks(med.index, [str(k) for k in med.index])
    ax.set_xlim(-0.5, hi)
    ax.set(xlabel="shift s (days): days_since_bet of day D is compared with the real recency of day D + s",
           ylabel="% of players where\ndays_since_bet(D) = R(D + s)", ylim=(0, 118))


def auc(evidence: pd.DataFrame, ax):
    cols = [("AUC next 7 days: table days_since_bet", TABLE_C, "days_since_bet of the table (uses later data)"),
            ("AUC next 7 days: recency as of the day", MINE_C, "real recency, data up to D"),
            ("AUC next 7 days: churn_score", OTHER_C, "churn_score of the table")]
    g = evidence.groupby("brand")[[c for c, _, _ in cols]].median()
    x, w = np.arange(len(g)), 0.27
    for k, (col, colour, label) in enumerate(cols):
        ax.bar(x + (k - 1) * w, g[col], width=w * 0.95, color=colour, label=label)
        for i, v in enumerate(g[col]):
            ax.text(i + (k - 1) * w, v + 0.005, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(0.5, color=INK, lw=0.8)
    ax.text(len(g) - 0.5, 0.495, "0.5 = no better than chance", ha="right", va="top", fontsize=8, color=MUTED)
    ax.set_xticks(x, [f"brand {b}" for b in g.index])
    ax.set(ylim=(0.45, 1.1), ylabel="AUC to predict a bet\nin the next 7 days", xlabel="brand")
    ax.legend(loc="upper center", ncol=3)


def active_match(over_time: pd.DataFrame, column: str, ax):
    g = over_time[over_time["column"] == column].groupby("day")[["= official %", "= any wallet %"]].median()
    ax.plot(g.index, g["= any wallet %"], marker="o", ms=3.5, color=TABLE_C, lw=1.8,
            label="days with a bet from any wallet (playable bonus included)")
    ax.plot(g.index, g["= official %"], marker="o", ms=3.5, color=MINE_C, lw=1.8,
            label="official active days, M053 (real money or released bonus)")
    _month_axis(ax, 4)
    ax.set(title=column, ylim=(0, 104), xlabel="monthly snapshot")

def active_match_bars(over_time: pd.DataFrame, ax):
    """Per window: % of players where the table equals the official count and the count with any wallet
    (median of brands and monthly snapshots; the line spans the lowest and highest snapshot), and the share
    of players where the table is above and below the official count."""
    x, w = np.arange(len(ev.WINDOWS)), 0.36
    for k, (col, colour, label) in enumerate([("= official %", MINE_C, "table = official active days, M053 (real money or released bonus)"),
                                              ("= any wallet %", TABLE_C, "table = days with a bet from any wallet (playable bonus included)")]):
        for i, window in enumerate(ev.WINDOWS):
            per_day = over_time[over_time["column"] == window].groupby("day")[col].median()
            v = over_time.loc[over_time["column"] == window, col].median()
            ax.bar(i + (k - 0.5) * w, v, width=w * 0.95, color=colour, label=label if i == 0 else None)
            ax.plot([i + (k - 0.5) * w] * 2, [per_day.min(), per_day.max()], color=INK, lw=1)
            ax.text(i + (k - 0.5) * w, v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=10, color="white", weight="bold")
    for i, window in enumerate(ev.WINDOWS):
        g = over_time[over_time["column"] == window]
        ax.text(i, -16, f"table above the official count: {g['above official %'].median():.0f}% of players\n"
                f"table below the official count: {g['below official %'].max():.0f}%", ha="center", va="top", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"{c}\n(last {d} days)" for c, d in ev.WINDOWS.items()])
    ax.set(ylim=(0, 118), ylabel="% of players where the table\nequals each count")
    ax.set_xlabel("window of the column", labelpad=34)
    ax.legend(loc="upper center", ncol=1)


def false_actives(fa: pd.DataFrame, ax):
    x = np.arange(len(fa))
    ax.bar(x, fa["only playable bonus %"], color=TABLE_C, width=0.6)
    for i, r in fa.iterrows():
        ax.text(i, r["only playable bonus %"] + 0.8, f"{r['only playable bonus %']:.0f}%  ({int(r['only playable bonus']):,} players)\n"
                f"table gives them {r['their mean table active days']:.1f} days", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"brand {b}" for b in fa["brand"]])
    ax.set(ylim=(0, max(fa["only playable bonus %"].max(), 1) * 1.35),
           ylabel=f"% of the players who bet in the\n30 days to {settings.PERIOD['start']:%-d %b %Y}", xlabel="brand")

def tp_daily(flags: pd.DataFrame, repeated: pd.DataFrame, axes):
    """Top: rows with a time-of-play value per day. Bottom: share of impossible values per day. Red lines:
    frozen days (the same rows and impossible values as the day before, in every brand)."""
    names = {"hourly_entropy": ("entropy above 1", "#2a78d6"), "weekend_index": ("weekend above 1", "#eb6834"),
             "night_index": ("night above 1", "#1baf7a"), "session_duration_avg_secs": ("mean session over 1 day", "#eda100")}
    d = flags[flags["metric"].isin(names)]
    d = d.groupby(["metric", "day"])[["rows", "impossible"]].sum().reset_index()
    d = d[d["rows"] > 0]
    n = d[d["metric"] == "night_index"].set_index("day")["rows"]
    axes[0].plot(pd.to_datetime(n.index), n.to_numpy() / 1000, color=INK, lw=1.6)
    axes[0].set(ylabel="thousand rows with\na time-of-play value")
    for name, (label, colour) in names.items():
        g = d[d["metric"] == name]
        axes[1].plot(pd.to_datetime(g["day"]), 100 * g["impossible"] / g["rows"], lw=1.6, color=colour, label=label)
    for ax in axes:
        for day in repeated["day"]:
            ax.axvline(pd.Timestamp(day), color=IMPOSSIBLE, lw=1, alpha=0.6)
        for day in settings.TP_EVENTS:
            ax.axvline(pd.Timestamp(day), color=INK, lw=0.8)
    for day, text in settings.TP_EVENTS.items():
        axes[0].text(pd.Timestamp(day), axes[0].get_ylim()[1] * 0.97, " " + text, fontsize=8, va="top", color=INK)
    if len(repeated):
        axes[0].text(pd.Timestamp(repeated["day"].iloc[0]), axes[0].get_ylim()[1] * 0.45, " red lines:\n frozen days",
                     fontsize=8, color=IMPOSSIBLE, va="top")
    axes[1].set(ylabel="% of the rows\nwith an impossible value", xlabel="day")
    import matplotlib.dates as mdates
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[1].legend(ncol=4, loc="upper right")

def index_limits(hist: pd.DataFrame, column: str, ax, red: tuple = (), note: str = "", above: tuple | None = None):
    """All brands: % of rows per bar between the limits; the bars `red` (bin numbers) in red. The last bar
    holds the upper limit and everything above it; with `above` = (% exactly at the limit, % above it,
    maximum), it is split: the limit stays in its bar and a red bar after the axis shows the values above."""
    h = hist[hist["column"] == column]
    lo, hi = h["lo"].iloc[0], h["hi"].iloc[0]
    width = (hi - lo) / ev.BINS
    g = h.groupby("bin")["count"].sum().reindex(range(-1, ev.BINS + 1), fill_value=0)
    x = lo + (g.index.to_numpy() + 0.5) * width
    y = 100 * g.to_numpy() / g.sum()
    if above is not None:
        y[-1] = above[0]
    ax.bar(x, y, width=width * 0.9, color=[TABLE_C if b in red else LIGHT_C for b in g.index])
    if above is not None:
        xa = hi + 9 * width
        ax.bar(xa, above[1], width=width * 2, color=TABLE_C)
        ax.text(xa, above[1] * 1.6, f"> {hi:g}: {above[1]:.2f}%\nup to {above[2]:,.0f}", ha="center", fontsize=8.5,
                color=TABLE_C, weight="bold")
        ticks = [t for t in ax.get_xticks() if lo <= t <= hi]
        ax.set_xticks(ticks + [xa], [f"{t:g}" for t in ticks] + [f"> {hi:g}"])
        ax.get_xticklabels()[-1].set_color(TABLE_C)
        ax.set_xlim(lo - 2 * width, xa + 2 * width)
    ax.set(title=column, ylabel="% of rows (log scale)", yscale="log", xlabel=f"value in the table (limits {lo:g} to {hi:g})")
    if note:
        ax.text(0.5, 0.97, note, transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=TABLE_C,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})

def hold_edge(cfg: dict, axes):
    """From the cached histogram of 01_distributions (no S3): left, the plot as presented, with every value
    below -100% in the edge bin; right, the same data with no edge bin, on bins of 1 point."""
    m = cfg["metrics"]["hold_pct_l30d"]
    _, hist = audit.metric_data(cfg, "hold_pct_l30d")
    h = hist.set_index("bin")["count"].sort_index()
    n = h.sum()
    shown = stats.complete(hist, m)
    axes[0].bar(shown["x"], 100 * shown["count"] / n, width=0.95, color=LIGHT_C)
    edge = shown["count"].iloc[0]
    axes[0].bar(shown["x"].iloc[0], 100 * edge / n, width=0.95, color=IMPOSSIBLE)
    axes[0].annotate(f"first bar: every value below -100%\nadded up, {int(edge):,} rows ({100 * edge / n:.2f}%)",
                     (shown["x"].iloc[0], 100 * edge / n), xytext=(-80, 100 * h.max() / n * 0.7), fontsize=8,
                     arrowprops={"arrowstyle": "->"})
    axes[0].set(title="As plotted before: everything below -100% in one bar", xlabel="hold %", ylabel="% of rows")
    full = h.loc[-300:100]
    colours = [MINE_C if b in (-100, 100) else LIGHT_C for b in full.index]
    axes[1].bar(full.index + 0.5, 100 * full.to_numpy() / n, width=1, color=colours)
    for v, lab, dx, fy in ((-100, "-100%: no peak", 10, 5), (7, "about 7%: house margin", -60, 0.4),
                           (100, "100%: lost every stake", -45, 0.08)):
        y = 100 * h.get(v, 0) / n
        axes[1].annotate(lab, (v + 0.5, y), xytext=(v + dx, y * fy), fontsize=8, ha="right" if dx < 0 else "left",
                         arrowprops={"arrowstyle": "->", "lw": 0.8})
    axes[1].set_xlim(-302, 104)
    axes[1].set(title=f"Bars of 1 point, -300% to 100% ({100 * h.loc[:-301].sum() / n:.2f}% of rows below -300%)",
                xlabel="hold %", ylabel="% of rows", yscale="log")

def hold_by_bets(hist: pd.DataFrame, ax):
    """Per group of bets in 30 days: % of rows with hold = 100% (won nothing) and below -100% (won more than
    twice the stake)."""
    order = ["1 bet", "2 to 5", "6 to 20", "21 to 100", "over 100"]
    rows = []
    for b in order:
        g = hist[hist["bets_30d"] == b].set_index("bin")["count"]
        n = g.sum()
        rows.append((b, n, 100 * g.get(100, 0) / n, 100 * g[g.index < -100].sum() / n))
    t = pd.DataFrame(rows, columns=["group", "rows", "at 100", "below -100"])
    x, w = np.arange(len(t)), 0.38
    for k, (col, colour, label) in enumerate([("at 100", TABLE_C, "hold = 100% (won nothing)"),
                                              ("below -100", OTHER_C, "hold below -100% (won over twice the stake)")]):
        ax.bar(x + (k - 0.5) * w, t[col], width=w * 0.95, color=colour, label=label)
        for i, v in enumerate(t[col]):
            ax.text(i + (k - 0.5) * w, v + 1, f"{v:.1f}%", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"{r.group}\n({int(r.rows):,} rows)" for r in t.itertuples()])
    ax.set(xlabel="bets of the player in 30 days", ylabel="% of the rows of the group", ylim=(0, t[["at 100", "below -100"]].max().max() * 1.2))
    ax.legend(loc="upper right")

def tenure(growth: pd.DataFrame, axes):
    """Left: share of players whose tenure grows by the days between two monthly snapshots (it must: the
    first active day does not move). Right, when two axes are given: the median growth against the expected one."""
    axes = list(np.atleast_1d(axes))
    for b, g in growth.groupby("brand"):
        axes[0].plot(g["day"], 100 * g["share_growing_as_expected"], marker="o", ms=3.5, lw=1.6, color=BRAND_C[str(b)], label=f"brand {b}")
        if len(axes) > 1:
            axes[1].plot(g["day"], g["median_growth"], marker="o", ms=3.5, lw=1.6, color=BRAND_C[str(b)])
    if len(axes) > 1:
        e = growth.groupby("day")["expected_growth"].first()
        axes[1].plot(e.index, e.to_numpy(), color=INK, lw=2.2, label="what it must grow\n(days between snapshots)")
        axes[1].set(ylabel="growth of the tenure (days, median)", title="Median growth against what it must grow")
        axes[1].legend(loc="upper left")
    for ax in axes:
        live_line(ax, ax.get_ylim()[1] * 0.62, ha="right")
        _month_axis(ax)
        ax.set_xlabel("monthly snapshot")
    axes[0].set(ylim=(-2, 104), ylabel="% of players whose tenure grew\nexactly the days in between (±1)",
                title="Players with a correct growth" if len(axes) > 1 else "")
    axes[0].legend(loc="center left")

def footer_extremes(ext: pd.DataFrame, axes):
    axes[0].plot(ext.index, ext["max hold %"], color=TABLE_C, lw=1.2)
    axes[0].axhline(100, color=INK, lw=1, label="100% = the maximum possible")
    axes[0].legend(loc="upper right")
    axes[0].set(yscale="log", ylabel="highest hold_pct_l30d\nof the day (%, log scale)",
                title=f"Above 100% on {int((ext['max hold %'] > 100).sum())} days")
    axes[1].plot(ext.index, ext["min won EUR"], color=TABLE_C, lw=1.2)
    axes[1].axhline(0, color=INK, lw=1, label="0 = the minimum possible")
    axes[1].legend(loc="lower right")
    axes[1].set(ylabel="lowest won_eur_l30d\nof the day (EUR)", title=f"Below 0 on {int((ext['min won EUR'] < 0).sum())} days")
    for ax in axes:
        _month_axis(ax)
        ax.set_xlabel("day")

def brand_nulls(nulls: pd.Series, ax):
    ax.plot(nulls.index, 100 * nulls.to_numpy(), color=TABLE_C, lw=1.8)
    _month_axis(ax)
    ax.set(ylabel="% of rows with\nan empty brand_id", xlabel="day", ylim=(-2, 104))

def session_brands(cfg: dict, rc_cfg: dict, axes):
    """Mean session length per player, the table against the recomputed one, one panel per brand."""
    outs = {}
    for conf, name in ((cfg, "session_duration_avg_secs"), (rc_cfg, "rc_player_avg_session_secs")):
        seg = audit.segment_histograms(conf, name, "brand_id")
        outs[name] = (seg, stats.outside_by_segment(conf, name, seg).set_index("segment"))
    for ax, b in zip(axes, BRAND_ORDER):
        title = []
        for conf, name, colour, label in ((cfg, "session_duration_avg_secs", TABLE_C, "signals table"),
                                          (rc_cfg, "rc_player_avg_session_secs", MINE_C, "recomputed")):
            seg, out = outs[name]
            m = conf["metrics"][name]
            h = stats.complete(seg[seg["segment"] == b][["bin", "count"]].reset_index(drop=True), m)
            o = out.loc[b, "below range %"] + out.loc[b, "above range %"]
            ax.step(h["x"], 100 * h["share"], where="mid", color=colour, lw=1.5, label=label)
            title.append(f"{o:.1f}%")
        _log_x(ax, secs=True)
        zone(ax, 4 * 3600, 86400, "suspicious")
        zone(ax, 86400, 3.5e6, "impossible")
        ticks = [(1, "1 s"), (60, "1 min"), (3600, "1 h"), (86400, "1 day"), (28 * 86400, "28 d")]
        ax.set_xticks([t for t, _ in ticks], [l for _, l in ticks])
        ax.minorticks_off()
        ax.set(xlim=(0.8, 3.5e6), xlabel="mean session length", title=f"Brand {b}")
        ax.text(0.98, 0.95, f"outside 10 s to 4 h\ntable {title[0]}\nrecomputed {title[1]}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=MUTED)
    axes = list(axes)
    for ax in axes[::3]:
        ax.set_ylabel("% of players in each bar")
    axes[0].legend(loc="upper left")

def session_rules(checks: pd.DataFrame, ax):
    """Share of rows that break each rule of a real session, per brand (it should be 0%)."""
    rules = {"mean over 1 day": "mean session > 1 day",
             "mean exactly 1 s": "mean session = 1 s (sessions of one bet)",
             "mean x sessions per active day > 24 h": "mean session × sessions in a day > 24 h",
             "gap between sessions < 0": "gap between sessions < 0 (overlap)",
             "play_secs today > 24 h": "play_secs_today > 24 h",
             "play_secs 30 days > 30 days": "play_secs_l30d > 30 days"}
    p = checks[checks["check"].isin(rules)].pivot_table(index="check", columns="brand", values="holds %").reindex(list(rules))
    p = p[BRAND_ORDER + ["all"]]
    ax.imshow(p.to_numpy(), cmap="Reds", aspect="auto", vmin=0, vmax=max(6, np.nanmax(p.to_numpy())))
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.iloc[i, j]
            ax.text(j, i, f"{v:.2f}%", ha="center", va="center", fontsize=9, color="white" if v > 3.5 else INK,
                    fontweight="bold" if j == p.shape[1] - 1 else None)
    ax.set_xticks(range(p.shape[1]), [f"brand {b}" for b in BRAND_ORDER] + ["all brands"])
    ax.set_yticks(range(p.shape[0]), [rules[r] for r in p.index])
    ax.set_xlabel("brand")
    ax.axvline(p.shape[1] - 1.5, color=INK, lw=1)
    ax.grid(False)
    ax.tick_params(length=0)

def above_one_brands(cfg: dict, ax):
    """Share of rows above 1 per brand, for the three time-of-play indices."""
    names = {"night_index": ("tp_night_index", "#2a78d6"), "weekend_index": ("tp_weekend_index", "#eb6834"),
             "hourly_entropy": ("tp_hourly_entropy", "#1baf7a")}
    x, w = np.arange(len(BRAND_ORDER) + 1), 0.27
    for k, (name, (label, colour)) in enumerate(names.items()):
        prof = audit.profile_by_segment(cfg, name).set_index("segment")
        allp, _ = audit.metric_data(cfg, name)
        vals = [100 * prof.loc[b, "above_range"] / prof.loc[b, "n"] for b in BRAND_ORDER] + [100 * allp["above_range"] / allp["n"]]
        ax.bar(x + (k - 1) * w, vals, width=w * 0.95, label=label, color=colour)
        for i, v in enumerate(vals):
            ax.text(i + (k - 1) * w, v + 0.4, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [f"brand {b}" for b in BRAND_ORDER] + ["all brands"])
    ax.set(ylabel="% of rows above 1\n(impossible for a share)", xlabel="brand")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
    ax.legend(loc="upper left", ncol=3)

def compare(series: list, ax, ticks=None, xlim=None, impossible=(), suspicious=(), xlabel: str = "",
            ylabel: str = "% of players in each bar", outside: str = "outside the real-life range"):
    """Histograms of several cached metrics with the same bins, [(label, cfg, metric, colour)], as % of
    rows per bar; the legend gives the median and the share outside the real-life range."""
    for label, cfg, name, colour in series:
        m = cfg["metrics"][name]
        prof, hist = audit.metric_data(cfg, name)
        h = stats.complete(hist, m)
        if m["scale"] == "log":
            _log_x(ax)
        if m.get("log_y"):
            ax.set_yscale("log")
        med = prof["quantiles"][4]
        med = (f"{med / 60:.0f} min" if med >= 60 else f"{med:.0f} s") if m["scale"] == "log" else f"{med:.2f}"
        out = 100 * (prof["below_range"] + prof["above_range"]) / max(prof["n"], 1)
        ax.step(h["x"], 100 * h["share"], where="mid", color=colour, lw=1.7,
                label=f"{label}: median {med}, {out:.1f}% {outside}")
    if xlim:
        ax.set_xlim(*xlim)
    for lo, hi, lab in suspicious:
        zone(ax, lo, hi, "suspicious", lab)
    for lo, hi, lab in impossible:
        zone(ax, lo, hi, "impossible", lab)
    if ticks:
        ax.set_xticks([t for t, _ in ticks], [lab for _, lab in ticks])
        ax.minorticks_off()
    ax.set(xlabel=xlabel, ylabel=ylabel + (" (log scale)" if ax.get_yscale() == "log" else ""))
    ax.legend(loc="center left") if xlim and xlim[1] > 10 else ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9))

def sessions_by_product(sp: pd.DataFrame, axes):
    """One panel per rule: % of the rows of each group (sports, casino) with an impossible value, and under
    each bar the share of all the impossible cases of that rule that come from the group."""
    rules = ["play_secs_today > 24 h", "mean session > 1 day", "gap between sessions < 0"]
    order = ["sports only", "both", "bets on sports", "casino only"]
    for ax, rule in zip(axes, rules):
        t = sp[sp["rule"] == rule].groupby("grp")[["rows", "impossible"]].sum()
        t = t.reindex([g for g in order if g in t.index])
        pct, share = 100 * t["impossible"] / t["rows"], 100 * t["impossible"] / t["impossible"].sum()
        colours = [MINE_C if g == "casino only" else OTHER_C for g in t.index]
        ax.bar(range(len(t)), pct, color=colours, width=0.6)
        for i, (v, (_, r)) in enumerate(zip(pct, t.iterrows())):
            ax.text(i, v, f"{v:.2f}%\n{int(r['impossible']):,} of {int(r['rows']):,}", ha="center", va="bottom", fontsize=8.5, color=INK)
        names = {"both": "sports and casino\nthe same day"}
        ax.set_xticks(range(len(t)), [f"{names.get(g, g)}\n{sh:.0f}% of the cases" for g, sh in zip(t.index, share)], fontsize=8.5)
        ax.set(title=rule, ylabel="% of the rows of the group\nwith the impossible value", ylim=(0, pct.max() * 1.3))

def empty_codes(values: pd.DataFrame, ax):
    """Per brand, % of rows with an empty country, language and currency."""
    cols = [("country", "#2a78d6"), ("language", "#eb6834"), ("currency", "#1baf7a")]
    x, w = np.arange(len(BRAND_ORDER)), 0.27
    for k, (c, colour) in enumerate(cols):
        g = values[values["col"] == c]
        empty = g[g["value"].fillna("") == ""].groupby("brand_id")["rows"].sum()
        pct = (100 * empty / g.groupby("brand_id")["rows"].sum()).reindex([int(b) for b in BRAND_ORDER]).fillna(0)
        ax.bar(x + (k - 1) * w, pct, width=w * 0.95, color=colour, label=c)
        for i, v in enumerate(pct):
            ax.text(i + (k - 1) * w, v + 1, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [f"brand {b}" for b in BRAND_ORDER])
    ax.set(ylabel="% of rows with an empty value", xlabel="brand", ylim=(0, 118))
    ax.legend(loc="upper center", ncol=3)
CATEGORY_ORDER = {"lifecycle_stage": ["Registered", "New", "Active", "Reactivated", "Dormant", "Churned"],
                  "value_tier": ["Not Scored", "Plankton", "Minnow", "Dolphin", "Shark", "Whale"],
                  "lifetime_value_tier": ["Not Scored", "Plankton", "Minnow", "Dolphin", "Shark", "Whale"],
                  "churn_band": ["Not Scored", "Low", "Moderate", "High", "Imminent", "Churned"],
                  "risk_band": ["Not Scored", "Low", "Medium", "High", "Critical"],
                  "prev_churn_band": ["Not Scored", "Low", "Moderate", "High", "Imminent", "Churned"]}


def category_shares(values: pd.DataFrame, col: str, ax):
    """% of rows of each value of a categorical column (all brands), in the order of its levels. Not Scored in grey."""
    g = values[values["col"] == col].groupby(values["value"].fillna("NULL").replace("", "(empty)"))["rows"].sum()
    order = [v for v in CATEGORY_ORDER.get(col, []) if v in g.index] + [v for v in g.index if v not in CATEGORY_ORDER.get(col, [])]
    g = 100 * g.reindex(order) / g.sum()
    ax.bar(range(len(g)), g, color=[REF_C if v in ("Not Scored", "NULL", "(empty)") else MINE_C for v in g.index], width=0.7)
    for i, v in enumerate(g):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(g)), g.index, rotation=30, ha="right", fontsize=8.5)
    ax.set(title=col, ylabel="% of rows", ylim=(0, 105))
# ---------------------------------------------------------------- gld_player_gaming_daily

def checks_heatmap(summary: pd.DataFrame, ax):
    """Pass % of each row check in each brand (red = fails; grey = rejected definition, not an error)."""
    cols = [c for c in summary.columns if c.startswith("brand ")] + ["pass %"]
    z = summary[cols].to_numpy(float)
    rejected = (summary["verdict"] == "rejected").to_numpy() if "verdict" in summary else np.zeros(len(summary), bool)
    ax.imshow(np.where(rejected[:, None], np.nan, 100 - z), cmap="Reds", aspect="auto", vmin=0, vmax=20)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if not np.isnan(z[i, j]):
                color = "grey" if rejected[i] else "white" if 100 - z[i, j] > 12 else "black"
                ax.text(j, i, f"{z[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color=color, fontweight="bold" if j == z.shape[1] - 1 else None)
    ax.set_xticks(range(len(cols)), [c.replace("pass %", "all") for c in cols])
    ax.set_yticks(range(len(summary)), [f"{s[:70]}  ({'rejected' if r else src})"
                                        for s, src, r in zip(summary["check"], summary["source"], rejected)], fontsize=7)
    for label, r in zip(ax.get_yticklabels(), rejected):
        if r:
            label.set_color("grey")
    ax.set_title("Row checks: % of rows that pass, per brand (red = fails, grey = rejected definition)", fontsize=10)

def sports_split(t: pd.DataFrame, axes):
    """Sports betting against casino (index): % of their rows with play_secs over 24 h (left), and their
    share of all rows against their share of the rows over 24 h (right)."""
    colors = [OTHER_C, LIGHT_C]
    pct = 100 * t["bad_rows"] / t["rows"]
    axes[0].bar(t.index, pct, color=colors)
    for x, v in enumerate(pct):
        axes[0].text(x, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    axes[0].set(ylabel="% of its rows", title="Rows with play_secs over 24 h")
    shares = pd.DataFrame({"all rows": 100 * t["rows"] / t["rows"].sum(),
                           "rows over 24 h": 100 * t["bad_rows"] / t["bad_rows"].sum()})
    bottom = np.zeros(2)
    for (name, row), c in zip(shares.iterrows(), colors):
        axes[1].bar(shares.columns, row, bottom=bottom, color=c, label=name)
        for x, (b, v) in enumerate(zip(bottom, row)):
            axes[1].text(x, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9)
        bottom += row.to_numpy()
    axes[1].set(ylabel="%", title="Share of sports betting and casino")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))

def first_play_hours(hours: pd.DataFrame, ax):
    """Rows per UTC hour of the first play (% of all rows): same day as event_date, and another day (red)."""
    total = hours["rows"].sum()
    same = hours[hours["day_shift"] == 0].groupby("hour")["rows"].sum().reindex(range(24), fill_value=0)
    other = hours[hours["day_shift"] != 0].groupby("hour")["rows"].sum().reindex(range(24), fill_value=0)
    ax.bar(same.index, 100 * same / total, color=LIGHT_C, label="first play on event_date (UTC)")
    ax.bar(other.index, 100 * other / total, bottom=100 * same / total, color=IMPOSSIBLE, label="first play on another day")
    ax.set(xlabel="UTC hour of the first play", ylabel="% of rows", xticks=range(24),
           title=f"First play by UTC hour: {100 * other.sum() / total:.1f}% of rows on another day than event_date")
    ax.legend(fontsize=8)

def daily_rows(rows: pd.Series, ax):
    """Rows per daily file; missing days in red."""
    ax.plot(rows.index, rows.to_numpy() / 1e3, color="black", lw=1.2)
    for day in rows.index[rows.isna()]:
        ax.axvline(day, color=IMPOSSIBLE, lw=0.8, alpha=0.6)
    ax.set(ylabel="thousand rows", title=f"Rows per daily file (all operators); red lines: missing days ({int(rows.isna().sum())})")

def daily_extremes(ext: pd.DataFrame, columns: list[str], limits: dict, axes):
    """Daily minimum or maximum of each column against its real-life limit (dashed)."""
    for ax, c in zip(np.atleast_1d(axes), columns):
        side, lim = limits[c]
        s = ext[f"{c}:{'max_num' if side == 'max' else 'min_num'}"]
        ax.plot(s.index, s.to_numpy(), color=TABLE_C, lw=1.1)
        ax.axhline(lim, color="black", ls="--", lw=1)
        bad = int((s > lim).sum() if side == "max" else (s < lim).sum() if side == "min" else (s <= lim).sum())
        word = {"max": f"above {lim:g}", "min": f"below {lim:g}", "min_or_equal": f"at {lim:g} or below"}[side]
        ax.set(title=f"Daily {side.split('_')[0]} of {c}: {word} on {bad} days", ylabel=c)
        if side == "max" and s.min() > 0 and s.max() > 100 * max(abs(lim), 1):
            ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
# ---------------------------------------------------------------- a table with its own catalogue (check_report.py)

def coverage(rows: pd.DataFrame, axes):
    """Top: rows per day, the five largest brands in colour and the others added up in grey. Bottom: % of those rows
    with has_sufficient_data = 1."""
    import matplotlib.dates as mdates
    top = rows.groupby("brand_id")["rows"].sum().nlargest(5).index
    colours = dict(zip(top, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
    groups = rows.assign(group=rows["brand_id"].where(rows["brand_id"].isin(top), -1), enough=rows["rows"] * rows["sufficient"])
    g = groups.groupby(["group", "day"])[["rows", "enough"]].sum().reset_index()
    for b, h in g.groupby("group"):
        day = pd.to_datetime(h["day"])
        name = f"brand {b}" if b != -1 else f"the other {rows['brand_id'].nunique() - len(top)} brands"
        style = {"color": colours.get(b, REF_C), "marker": "o", "ms": 3.5, "lw": 1.7}
        axes[0].plot(day, h["rows"] / 1000, label=name, **style)
        axes[1].plot(day, 100 * h["enough"] / h["rows"], **style)
    axes[0].set(ylabel="thousand rows", ylim=(0, None))
    axes[1].set(ylabel="% of the rows with\nhas_sufficient_data = 1", ylim=(0, 105), xlabel="day")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))


def formula_vs_table(pairs: pd.DataFrame, bins: int, ax, title: str, formula: str, share: float):
    """Rows per cell of (value of the formula, value in the table), on a log colour scale. The red line is what a
    column that follows its formula would draw."""
    import textwrap

    from matplotlib.colors import LogNorm
    grid = np.zeros((bins, bins))
    grid[pairs["y"].to_numpy(), pairs["x"].to_numpy()] = pairs["n"].to_numpy()
    pct = 100 * grid / grid.sum()
    im = ax.imshow(np.ma.masked_equal(pct, 0), origin="lower", extent=(0, 1, 0, 1), cmap="Blues", aspect="equal",
                   norm=LogNorm(vmin=pct[pct > 0].min(), vmax=pct.max()), interpolation="nearest")
    ax.plot([0, 1], [0, 1], color=TABLE_C, lw=2, label="table = formula")
    ax.text(0.03, 0.97, f"{share:.1f}% of the rows\nare on the red line", transform=ax.transAxes, va="top", fontsize=10.5,
            weight="bold", color=TABLE_C, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    ax.set_title(f"{title}\n" + textwrap.fill(f"= {formula}", 36), fontsize=10, loc="left")
    ax.set(xlabel="value of the formula, from the columns of the same row", ylabel="value in the table",
           xlim=(0, 1), ylim=(0, 1))
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="% of the rows in each cell")


def limit_hist(h: pd.DataFrame, ax, title: str, lo: float | None, hi: float | None, share_out: float, compact: bool = False):
    """Histogram of a column; the bars beyond a limit (impossible values) in red. The count of impossible rows is
    written above the plot, not over the bars. `compact`: for a panel of many, only the name and, if there are
    impossible rows, their share."""
    bars, a, b = int(h["bars"].iloc[0]), h["a"].iloc[0], h["b"].iloc[0]
    width = (b - a) / bars
    left = a + h["bin"].to_numpy() * width
    eps = width * 1e-6
    outside = ((left + width <= lo + eps) if lo is not None else np.zeros(len(left), bool)) | \
              ((left >= hi - eps) if hi is not None else np.zeros(len(left), bool))
    share = 100 * h["n"] / h["n"].sum()
    ax.bar(left + width / 2, share, width=width * 0.92, color=np.where(outside, TABLE_C, MINE_C))
    for edge in (lo, hi):
        if edge is not None:
            ax.axvline(edge, color=INK, lw=1.2, ls="--")
    if lo is not None:
        ax.axvspan(a, lo, color=IMPOSSIBLE, alpha=0.08, lw=0)
    if hi is not None:
        ax.axvspan(hi, b, color=IMPOSSIBLE, alpha=0.08, lw=0)
    log = share.max() / share.min() > 20   # a log scale only when the bars differ a lot: otherwise it exaggerates
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, share.max() * 1.1)
    ax.set_xlim(a, b)
    rule = (f"between {lo:g} and {hi:g}" if lo is not None and hi is not None
            else f"{lo:g} or more" if lo is not None else f"at most {hi:g}")
    main = f"{share_out:.1f}% of the rows are impossible" if share_out > 0 else "no row outside the limits"
    extra = [f"lowest {h['min'].iloc[0]:.3g}, highest {h['max'].iloc[0]:.3g}"] if share_out > 0 else []
    extra += [f"{100 * h[col].iloc[0]:.0f}% exactly at {edge:g}" for col, edge in (("at_lo", lo), ("at_hi", hi))
              if edge is not None and h[col].iloc[0] >= 0.01]
    size = (9.5, 8.5) if compact else (11, 10)
    ax.set_title(title, fontsize=size[0] + 1, pad=40 if compact else 44)
    ax.text(0, 1.095 if not compact else 1.115, main, transform=ax.transAxes, va="bottom", fontsize=size[1] + 1, weight="bold",
            color=TABLE_C if share_out > 0 else MUTED)
    ax.text(0, 1.025, "; ".join(extra), transform=ax.transAxes, va="bottom", fontsize=size[1], color=MUTED)
    if compact:
        ax.tick_params(labelsize=10.5)
    else:
        ax.set(xlabel=f"value in the table (the rule: {rule})", ylabel="% of the rows in each bar" + (" (log scale)" if log else ""))


def active_day_lines(m: pd.DataFrame, axes):
    """One panel per column that counts days with a bet: % of players where the table equals the recomputed value, with
    data up to D + s, for the official active day (blue) and for any wallet (red)."""
    g = m.groupby(["column", "variant", "shift"])[["n", "ok"]].sum()
    pct = (100 * g["ok"] / g["n"].where(g["n"] > 0)).rename("pct")
    for ax, (col, h) in zip(np.atleast_1d(axes).flat, pct.groupby(level="column")):
        for variant, colour, label in (("official", MINE_C, "official active day (real money or released bonus)"),
                                       ("any wallet", TABLE_C, "any wallet (playable bonus too)")):
            v = h.xs(variant, level="variant").droplevel("column")
            ax.plot(v.index, v.to_numpy(), marker="o", ms=4, lw=1.8, color=colour, label=label)
            ax.annotate(f"{v.iloc[0]:.0f}%", (0, v.iloc[0]), xytext=(-8, 4 if variant == "any wallet" else -12), textcoords="offset points",
                        fontsize=9.5, weight="bold", color=colour, ha="right")
        ax.axvspan(0.5, v.index.max() + 0.5, color=IMPOSSIBLE, alpha=0.06, lw=0)
        ax.set_title(col, fontsize=11)
        ax.set(xlim=(-1.6, v.index.max() + 0.5), ylim=(0, 105), xticks=list(v.index))
    for ax in np.atleast_1d(axes).flat:
        ax.set_xlabel("shift s: days after D used to compute the value")
        ax.set_ylabel("% of players where\ntable = recomputed")
    np.atleast_1d(axes).flat[0].legend(loc="upper right", fontsize=8.5)
