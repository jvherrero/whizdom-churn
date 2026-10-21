"""Every figure of the signal audit, for the notebooks and for figures.py (the PDF). English labels.
Each function draws on an axis (or returns a figure) from aggregates; none reads S3 by itself."""

from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

import audit
import evidence as ev
import stats

IMPOSSIBLE = "#c0392b"
SUSPICIOUS = "#e08e2b"
TABLE_C, MINE_C, OTHER_C, LIGHT_C = "#b5452f", "#2f6f8f", "#e08e2b", "#8fb3c7"
BRAND_ORDER = ["14", "23", "64", "72", "73"]

def _outside(h: pd.DataFrame, m: dict) -> pd.Series:
    """Bins fully outside the real-life range (a bin that touches the limit counts as inside)."""
    lo, hi = m["range"]
    if m["scale"] == "int":
        return (h["x"] < lo) | (h["x"] > hi)
    return (h["hi"] <= lo) | (h["lo"] > hi)


def _shade_impossible(ax, m: dict) -> None:
    """Shade the values outside the real-life range."""
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
            ax.set_xscale("log")
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
    if prof is not None:
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
    im = ax.imshow(np.ma.masked_invalid(z), aspect="auto", cmap=cmap, vmin=0, vmax=max(np.nanmax(z), 1),
                   extent=(x0, x1, len(labels), 0), interpolation="nearest")
    ax.set_yticks(np.arange(len(labels)) + 0.5, labels)
    ax.axhline(1, color="black", lw=0.8)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    plt.colorbar(im, ax=ax, pad=0.01, label="%")
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
        ax.set_xscale("log")
    _shade_impossible(ax, m)
    ax.set(title=f"{name} by segment", ylabel="share of rows (smoothed)")
    ax.legend(fontsize=7)
    return ax


def compare_results(pairs: list[tuple[str, dict, dict]], ax=None, title: str = ""):
    """Several versions of one signal on the same axes: (label, cfg, result) for each version. The
    shaded area is outside the real-life range of the first one."""

    ax = ax or plt.subplots(figsize=(8.5, 3.8))[1]
    for (label, cfg, res), colour in zip(pairs, ("#b5452f", "#2f6f8f", "#e08e2b", "#7a7a7a")):
        m = cfg["metrics"][res["metric"]]
        h = res["hist"]
        ax.step(h["x"], stats.density(h, m), where="mid", color=colour, lw=1.4,
                label=f"{label} ({h['count'].sum():,} rows, {len(res['peaks'])} peak(s))")
        if m["scale"] == "log":
            ax.set_xscale("log")
        if m.get("log_y"):
            ax.set_yscale("log")
    first = pairs[0][1]["metrics"][pairs[0][2]["metric"]]
    _shade_impossible(ax, first)
    ax.set(title=title or pairs[0][2]["metric"], ylabel="density")
    ax.legend(fontsize=7, loc="upper left")
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


def scale_marks(ax, kind: str):
    """Vertical marks where the values of a ratio to the uniform share (or of an entropy in bits) fall."""
    if kind == "weekend":
        marks = [3.5 / k for k in range(1, 8)]
    elif kind == "night":
        marks = [4 / k for k in range(1, 9)]
    else:
        marks = [np.log2(k) for k in range(2, 25)]
    for v in marks:
        ax.axvline(v, color="black", lw=0.6, ls=":", alpha=0.6)


def scale(cfg: dict, name: str, kind: str, title: str, ax):
    """Histogram of the table values (0 left out) with the marks of the scale hypothesis; the red zone
    is above 1, impossible for a share, with the share of rows that fall there."""
    x, c = _bins(cfg, name, drop_zero=True)
    ax.bar(x, c / c.sum(), width=0.02, color=np.where(x > 1, TABLE_C, LIGHT_C))
    scale_marks(ax, kind)
    ax.axvspan(1, max(x.max(), 1) + 0.1, color=TABLE_C, alpha=0.06, lw=0)
    p, _ = audit.metric_data(cfg, name)
    ax.text(0.62, 0.93, f"red zone: above 1, impossible for a share (0 to 1)\n{100 * p['above_range'] / p['n']:.1f}% of the rows",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color=IMPOSSIBLE,
            bbox={"facecolor": "white", "edgecolor": IMPOSSIBLE, "alpha": 0.9})
    ax.set(title=title, ylabel="share of rows (0 left out)", yscale="log", xlim=(-0.05, 4.7))


def rescaled(cfg: dict, rc_cfg: dict, table: str, mine: str, scale: float, label: str, ax):
    """The table values times `scale` against the recomputed share, both without the zeros."""
    edges = np.arange(0, 1.0201, 0.02)
    for name, conf, s, colour, lab in ((table, cfg, scale, TABLE_C, f"table x {label}"),
                                       (mine, rc_cfg, 1.0, MINE_C, "recomputed from each bet")):
        x, c = _bins(conf, name, s, drop_zero=True)
        h, _ = np.histogram(np.clip(x, 0, 1.0199), bins=edges, weights=c)
        ax.step(edges[:-1] + 0.01, h / h.sum(), where="mid", color=colour, lw=1.8, label=lab)
    ax.set(xlim=(0, 1.02), ylabel="share (0 left out)")
    ax.legend(fontsize=8)


def overlay(series: list, ax, log_x: bool = False):
    """Densities of several cached metrics on the same axes: [(cfg, metric, label, colour)]."""
    for conf, name, label, colour in series:
        m = conf["metrics"][name]
        prof, hist = audit.metric_data(conf, name)
        h = stats.complete(hist, m)
        y = stats.density(h, m)
        n_out = 100 * (prof["below_range"] + prof["above_range"]) / max(prof["n"], 1)
        ax.step(h["x"], y, where="mid", color=colour, lw=1.6, label=f"{label} (outside 10 s to 4 h: {n_out:.1f}%)")
    if log_x:
        ax.set_xscale("log")
    ax.legend(fontsize=8)


def _brand_lines(ax, frame: pd.DataFrame, x: str, y: str, **kw):
    for b, g in frame.groupby("brand"):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker="o", ms=3, lw=1.2, label=f"brand {b}", **kw)


def zero_share(over_time: pd.DataFrame, ax):
    _brand_lines(ax, over_time, "day", "table is 0 %")
    ax.axvline(ev.LIVE_FROM, color="black", ls="--", lw=1)
    ax.text(ev.LIVE_FROM, 50, "  daily job\n  from 1 Aug 2026", fontsize=8, va="center")
    ax.set(title="Monthly snapshots: players with days_since_bet = 0",
           ylabel="% of the players who bet\nin the 30 days before the snapshot", ylim=(-2, 102))
    ax.legend(fontsize=8)


def shift_match(shifts: pd.DataFrame, ax):
    med = shifts.groupby("shift")["match %"].median()
    for b, g in shifts.groupby("brand"):
        m = g.groupby("shift")["match %"].median()
        ax.plot(m.index, m.to_numpy(), color="#bbbbbb", lw=1)
    ax.bar(med.index, med.to_numpy(), color=np.where(med.index == 7, TABLE_C, LIGHT_C), alpha=0.9)
    for k, v in med.items():
        ax.text(k, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xticks(med.index, [f"{k:+d}" for k in med.index])
    ax.set(title="days_since_bet of day D equals the real recency of day D + k for ... of the players (median; grey: each brand)",
           xlabel="k = days after the snapshot date (0 = the date itself)", ylabel="% of players", ylim=(0, 105))


def leak_proof(evidence: pd.DataFrame, ax):
    """Players with a recency that only later data can give, how many of them bet in the next 7 days,
    and, to compare, the share of all players that bet in those 7 days."""
    cols = [("table shorter than any bet up to the day %", LIGHT_C, "recency of the table impossible with data up to D"),
            ("all players, bet in the next 7 days %", "#bbbbbb", "all players: bet between D and D + 7"),
            ("of those, bet in the next 7 days %", TABLE_C, "of the blue ones: bet between D and D + 7")]
    g = evidence.groupby("brand")[[c for c, _, _ in cols]].median()
    x, w = np.arange(len(g)), 0.27
    for k, (col, colour, label) in enumerate(cols):
        ax.bar(x + (k - 1) * w, g[col], width=w, color=colour, label=label)
        for i, v in enumerate(g[col]):
            ax.text(i + (k - 1) * w, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x, [f"brand {b}" for b in g.index])
    ax.set(ylim=(0, 150), ylabel="% of players (median of the snapshots)",
           title="When the table gives a last bet more recent than any bet up to D, the player bets after D almost always")
    ax.legend(fontsize=8, loc="upper left")


def auc(evidence: pd.DataFrame, ax):
    cols = [("AUC next 7 days: table days_since_bet", TABLE_C, "days_since_bet of the table"),
            ("AUC next 7 days: recency as of the day", MINE_C, "real recency up to D"),
            ("AUC next 7 days: churn_score", OTHER_C, "churn_score of the table")]
    g = evidence.groupby("brand")[[c for c, _, _ in cols]].median()
    x, w = np.arange(len(g)), 0.27
    for k, (col, colour, label) in enumerate(cols):
        ax.bar(x + (k - 1) * w, g[col], width=w, color=colour, label=label)
        for i, v in enumerate(g[col]):
            ax.text(i + (k - 1) * w, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="black", ls=":", lw=1)
    ax.set_xticks(x, [f"brand {b}" for b in g.index])
    ax.set(ylim=(0.45, 1.08), ylabel="AUC (0.5 = chance, 1 = perfect)",
           title="How well each value predicts a bet in the next 7 days")
    ax.legend(fontsize=8, loc="upper center", ncol=3)


def recency_differences(diff: pd.DataFrame, ax):
    for col, colour in (("as of the day", MINE_C), ("7 days later", TABLE_C)):
        ax.step(diff.index, 100 * diff[col] / diff[col].sum(), where="mid", color=colour, lw=1.8,
                label=f"table minus the recency {col}")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlim(-15, 15)
    ax.set(xlabel="difference in days", ylabel="% of players",
           title="days_since_bet minus the recomputed recency, 1 September 2026, all brands")
    ax.legend(fontsize=8)


def playable_share(share: pd.DataFrame, ax):
    for b in share.columns:
        ax.plot(share.index.to_timestamp(), share[b], marker="o", ms=3, label=f"brand {b}")
    ax.set(title="Player-days with bets only from playable bonus (not active days, M053)", ylabel="% of player-days")
    ax.legend(fontsize=8)


def active_match(over_time: pd.DataFrame, column: str, ax):
    g = over_time[over_time["column"] == column].groupby("day")[["= official %", "= any wallet %"]].median()
    ax.plot(g.index, g["= any wallet %"], marker="o", ms=3, color=LIGHT_C, lw=2, label="table = days with a bet from any wallet")
    ax.plot(g.index, g["= official %"], marker="o", ms=3, color=MINE_C, lw=2, label="table = official active days (M053)")
    ax.set(title=f"{column}: share of players where the table equals each count (median of the brands)",
           ylabel="% of players", ylim=(0, 102))
    ax.legend(fontsize=8, loc="lower left")


def active_differences(diff: pd.DataFrame, ax):
    x = diff.index.to_numpy(float)
    for k, (col, colour) in enumerate((("table - official", MINE_C), ("table - any wallet", LIGHT_C))):
        ax.bar(x + (k - 0.5) * 0.4, 100 * diff[col] / diff[col].sum(), width=0.4, color=colour, label=col)
    ax.set_xlim(-1.5, 10.5)
    ax.set(xlabel="difference in days", ylabel="% of players",
           title="active_days_l30d minus each recount, 1 September 2026, all brands: never below 0")
    ax.legend(fontsize=8)


def false_actives(fa: pd.DataFrame, ax):
    x = np.arange(len(fa))
    ax.bar(x, fa["only playable bonus %"], color=TABLE_C)
    for i, r in fa.iterrows():
        ax.text(i, r["only playable bonus %"] + 0.8,
                f"{r['only playable bonus %']:.0f}%\n({int(r['only playable bonus']):,})\ntable: {r['their mean table active days']:.1f} days",
                ha="center", fontsize=8)
    ax.set_xticks(x, [f"brand {b}" for b in fa["brand"]])
    ax.set(ylim=(0, fa["only playable bonus %"].max() * 1.45), ylabel="% of the players who bet in 30 days",
           title="Players with 0 official active days in 30 days, all shown as active by the table (1 Sep 2026)")


def tp_daily(flags: pd.DataFrame, repeated: pd.DataFrame, axes):
    """Top: rows with a time-of-play value per day. Bottom: share of impossible values per day. Red lines:
    frozen days (the same rows and impossible values as the day before, in every brand)."""
    names = {"session_duration_avg_secs": "session over 1 day", "night_index": "night above 1",
             "weekend_index": "weekend above 1", "hourly_entropy": "entropy above 1"}
    d = flags[flags["metric"].isin(names) & (flags["day"] >= dt.date(2026, 7, 1))]
    d = d.groupby(["metric", "day"])[["rows", "impossible"]].sum().reset_index()
    d = d[d["rows"] > 0]
    n = d[d["metric"] == "night_index"].set_index("day")["rows"]
    axes[0].plot(pd.to_datetime(n.index), n.to_numpy() / 1000, color="black", lw=1.5)
    axes[0].set(ylabel="thousand rows", title="Rows with a time-of-play value per day (five brands)")
    for name, g in d.groupby("metric"):
        axes[1].plot(pd.to_datetime(g["day"]), 100 * g["impossible"] / g["rows"], lw=1.5, label=names[name])
    for ax in axes:
        for day in repeated["day"]:
            ax.axvline(pd.Timestamp(day), color=IMPOSSIBLE, lw=0.9, alpha=0.7)
        for day in (ev.TP_FROM, dt.date(2026, 8, 1), dt.date(2026, 8, 19)):
            ax.axvline(pd.Timestamp(day), color="black", ls="--", lw=0.8)
    axes[1].set(ylabel="% of rows impossible",
                title="Impossible values per day; red lines: frozen days (the same as the day before, in every brand)")
    axes[1].legend(fontsize=8, ncol=4)


def index_limits(hist: pd.DataFrame, column: str, ax):
    """Share of rows per bin, one line per brand; the first and last bars are below and above the limits."""
    h = hist[hist["column"] == column]
    lo, hi = h["lo"].iloc[0], h["hi"].iloc[0]
    width = (hi - lo) / ev.BINS
    for b, g in h.groupby("brand"):
        g = g.set_index("bin")["count"].reindex(range(-1, ev.BINS + 1), fill_value=0)
        x = lo + (g.index.to_numpy() + 0.5) * width
        ax.step(x, 100 * g.to_numpy() / g.sum(), where="mid", lw=1.3, label=f"brand {b}")
    ax.axvspan(lo - width, lo, color=IMPOSSIBLE, alpha=0.12, lw=0)
    ax.axvspan(hi, hi + width, color=IMPOSSIBLE, alpha=0.12, lw=0)
    ax.set(title=f"{column} (red: outside [{lo:g}, {hi:g}])", ylabel="% of rows", yscale="log")
    ax.legend(fontsize=7)


def hold_edge(cfg: dict, axes):
    """From the cached histogram of 01_distributions (no S3): left, the plot as presented, with every value
    below -100% in the edge bin; right, the same data with no edge bin, on bins of 1 point."""
    m = cfg["metrics"]["hold_pct_l30d"]
    _, hist = audit.metric_data(cfg, "hold_pct_l30d")
    h = hist.set_index("bin")["count"].sort_index()
    n = h.sum()
    shown = stats.complete(hist, m)
    axes[0].bar(shown["x"], 100 * shown["count"] / n, width=0.95, color="#8fb3c7")
    edge = shown["count"].iloc[0]
    axes[0].bar(shown["x"].iloc[0], 100 * edge / n, width=0.95, color=IMPOSSIBLE)
    axes[0].annotate(f"edge bin: every value below -100%\n{int(edge):,} rows ({100 * edge / n:.2f}%)",
                     (shown["x"].iloc[0], 100 * edge / n), xytext=(-80, 100 * h.max() / n * 0.7), fontsize=8,
                     arrowprops={"arrowstyle": "->"})
    axes[0].set(title="As presented: the edge bin looks like a peak at -100%", xlabel="hold %", ylabel="% of rows")
    full = h.loc[-300:100]
    colours = ["#2f6f8f" if b in (-100, 100) else "#8fb3c7" for b in full.index]
    axes[1].bar(full.index + 0.5, 100 * full.to_numpy() / n, width=1, color=colours)
    for v, lab, dx, fy in ((-100, "-100%: no peak", 10, 5), (7, "about 7%: house margin", -60, 0.4),
                           (100, "100%: lost every stake", -45, 0.08)):
        y = 100 * h.get(v, 0) / n
        axes[1].annotate(lab, (v + 0.5, y), xytext=(v + dx, y * fy), fontsize=8, ha="right" if dx < 0 else "left",
                         arrowprops={"arrowstyle": "->", "lw": 0.8})
    axes[1].set_xlim(-302, 104)
    axes[1].set(title=f"No edge bin, -300% to 100% ({100 * h.loc[:-301].sum() / n:.2f}% of rows below -300%)",
                xlabel="hold %", ylabel="% of rows", yscale="log")


def hold_by_bets(hist: pd.DataFrame, ax):
    order = ["1 bet", "2 to 5", "6 to 20", "21 to 100", "over 100"]
    for b in order:
        g = hist[hist["bets_30d"] == b].set_index("bin")["count"].reindex(range(-301, 101), fill_value=0)
        if g.sum() == 0:
            continue
        ax.step(g.index + 0.5, 100 * g.to_numpy() / g.sum(), where="mid", lw=1.3, label=f"{b} ({int(g.sum()):,} rows)")
    ax.set(yscale="log", xlabel="hold % (first bin: below -300%)", ylabel="% of the rows of the group",
           title="hold_pct_l30d by bets in 30 days: the masses at the ends come from players with few bets")
    ax.legend(fontsize=8)


def tenure(growth: pd.DataFrame, axes):
    """Left: share of players whose tenure grows by the days between two monthly snapshots (it must: the
    first active day does not move). Right: the median growth against the expected one."""
    for b, g in growth.groupby("brand"):
        axes[0].plot(g["day"], 100 * g["share_growing_as_expected"], marker="o", ms=3, label=f"brand {b}")
        axes[1].plot(g["day"], g["median_growth"], marker="o", ms=3, label=f"brand {b}")
    e = growth.groupby("day")["expected_growth"].first()
    axes[1].plot(e.index, e.to_numpy(), color="black", ls="--", lw=1.5, label="expected (days between snapshots)")
    for ax in axes:
        ax.axvline(pd.Timestamp(ev.LIVE_FROM), color="black", ls=":", lw=1)
        ax.legend(fontsize=7)
    axes[0].set(ylim=(-2, 102), ylabel="% of players (±1 day)",
                title="Players whose tenure grows exactly\nthe days between the previous snapshot and this one")
    axes[1].set(ylabel="days", title="How much the tenure grows between two snapshots (median)\nagainst what it must grow")


def footer_extremes(ext: pd.DataFrame, axes):
    axes[0].plot(ext.index, ext["max hold %"], color="#b5452f", lw=1.2)
    axes[0].axhline(100, color="black", ls="--", lw=1)
    axes[0].set(yscale="log", ylabel="%", title=f"Daily maximum of hold_pct_l30d: above 100% on {int((ext['max hold %'] > 100).sum())} days")
    axes[1].plot(ext.index, ext["min won EUR"], color="#b5452f", lw=1.2)
    axes[1].axhline(0, color="black", ls="--", lw=1)
    axes[1].set(ylabel="EUR", title=f"Daily minimum of won_eur_l30d: below 0 on {int((ext['min won EUR'] < 0).sum())} days")


def brand_nulls(nulls: pd.Series, ax):
    ax.plot(nulls.index, 100 * nulls.to_numpy(), color="#b5452f", lw=1.5)
    ax.set(title="gld_player_signals_daily: rows with an empty brand_id per day", ylabel="%", ylim=(-2, 102))


def session_brands(cfg: dict, rc_cfg: dict, axes):
    """Mean session length per player, the table against the recomputed one, one panel per brand."""
    outs = {}
    for conf, name in ((cfg, "session_duration_avg_secs"), (rc_cfg, "rc_player_avg_session_secs")):
        seg = audit.segment_histograms(conf, name, "brand_id")
        outs[name] = (seg, stats.outside_by_segment(conf, name, seg).set_index("segment"))
    for ax, b in zip(axes, BRAND_ORDER):
        for conf, name, colour, label in ((cfg, "session_duration_avg_secs", TABLE_C, "table"),
                                          (rc_cfg, "rc_player_avg_session_secs", MINE_C, "recomputed")):
            seg, out = outs[name]
            m = conf["metrics"][name]
            h = stats.complete(seg[seg["segment"] == b][["bin", "count"]].reset_index(drop=True), m)
            o = out.loc[b, "below range %"] + out.loc[b, "above range %"]
            ax.step(h["x"], gaussian_filter1d(stats.density(h, m), 1), where="mid", color=colour, lw=1.4,
                    label=f"{label}: {o:.1f}% outside")
        ax.set_xscale("log")
        ax.axvspan(86400, 3.5e6, color=IMPOSSIBLE, alpha=0.08, lw=0)
        ax.axvspan(4 * 3600, 86400, color=SUSPICIOUS, alpha=0.10, lw=0)
        ticks = [(1, "1 s"), (60, "1 min"), (3600, "1 h"), (86400, "1 d"), (28 * 86400, "28 d")]
        ax.set_xticks([t for t, _ in ticks], [l for _, l in ticks])
        ax.minorticks_off()
        ax.set(xlim=(0.8, 3.5e6), title=f"Brand {b}")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("density")


def session_rules(checks: pd.DataFrame, ax):
    """Share of rows that break each rule of a real session, per brand (it should be 0%)."""
    rules = {"mean over 1 day": "Mean session over 1 day", "mean exactly 1 s": "Mean session of exactly 1 s",
             "mean x sessions per active day > 24 h": "Sessions of a day over 24 h",
             "gap between sessions < 0": "Negative gap between sessions", "play_secs today > 24 h": "play_secs of a day over 24 h",
             "play_secs 30 days > 30 days": "play_secs of 30 days over 30 days"}
    p = checks[checks["check"].isin(rules)].pivot_table(index="check", columns="brand", values="holds %").reindex(list(rules))
    p = p[BRAND_ORDER + ["all"]]
    ax.imshow(p.to_numpy(), cmap="Reds", aspect="auto", vmin=0, vmax=max(6, np.nanmax(p.to_numpy())))
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.iloc[i, j]
            ax.text(j, i, f"{v:.2f}%", ha="center", va="center", fontsize=9, color="white" if v > 3.5 else "black",
                    fontweight="bold" if j == p.shape[1] - 1 else None)
    ax.set_xticks(range(p.shape[1]), [f"brand {b}" for b in BRAND_ORDER] + ["all"])
    ax.set_yticks(range(p.shape[0]), [rules[r] for r in p.index])
    ax.axvline(p.shape[1] - 1.5, color="black", lw=1)
    ax.set_title("Rows that break each rule of a real session, per brand (September 2026). It should be 0%.", fontsize=10)


def above_one_brands(cfg: dict, ax):
    """Share of rows above 1 per brand, for the three time-of-play indices."""
    names = {"night_index": "tp_night_index", "weekend_index": "tp_weekend_index", "hourly_entropy": "tp_hourly_entropy"}
    x, w = np.arange(len(BRAND_ORDER) + 1), 0.27
    for k, (name, label) in enumerate(names.items()):
        prof = audit.profile_by_segment(cfg, name).set_index("segment")
        allp, _ = audit.metric_data(cfg, name)
        vals = [100 * prof.loc[b, "above_range"] / prof.loc[b, "n"] for b in BRAND_ORDER] + [100 * allp["above_range"] / allp["n"]]
        ax.bar(x + (k - 1) * w, vals, width=w, label=label)
        for i, v in enumerate(vals):
            ax.text(i + (k - 1) * w, v + 0.4, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x, [f"brand {b}" for b in BRAND_ORDER] + ["all"])
    ax.set(ylabel="% of rows above 1", title="Impossible values (above 1) per brand, September 2026: every brand")
    ax.legend(fontsize=8)


def compare(series: list, ax, ticks=None, xlim=None, impossible=(), suspicious=()):
    """Densities of several cached metrics on the same axes, [(label, cfg, metric, colour)], with the median
    and the share outside the real-life range in the legend, and shaded impossible and suspicious zones."""
    for label, cfg, name, colour in series:
        m = cfg["metrics"][name]
        prof, hist = audit.metric_data(cfg, name)
        h = stats.complete(hist, m)
        y = stats.density(h, m)
        if m["scale"] == "log":
            y = gaussian_filter1d(y, 1)
            ax.set_xscale("log")
        if m.get("log_y"):
            ax.set_yscale("log")
        med = prof["quantiles"][4]
        med = (f"{med / 60:.0f} min" if med >= 60 else f"{med:.0f} s") if m["scale"] == "log" else f"{med:.2f}"
        out = 100 * (prof["below_range"] + prof["above_range"]) / max(prof["n"], 1)
        ax.step(h["x"], y, where="mid", color=colour, lw=1.6, label=f"{label}: median {med}, outside the range {out:.2f}%")
    if xlim:
        ax.set_xlim(*xlim)
    for lo, hi in suspicious:
        ax.axvspan(lo, hi, color=SUSPICIOUS, alpha=0.10, lw=0)
    for lo, hi in impossible:
        ax.axvspan(lo, hi, color=IMPOSSIBLE, alpha=0.08, lw=0)
        ax.text((lo * hi) ** 0.5 if ax.get_xscale() == "log" else (lo + hi) / 2, 0.04, "impossible",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color=IMPOSSIBLE, fontweight="bold")
    if ticks:
        ax.set_xticks([t for t, _ in ticks], [lab for _, lab in ticks])
        ax.minorticks_off()
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper right" if xlim and xlim[1] < 10 else "upper left")
