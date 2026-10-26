"""Statistics of the binned aggregates of audit.py: bins, peaks, the fits of the expected families, and the
scorecards (all brands and brand by brand).

The fits use the binned data: with millions of rows every test rejects, so the result is a distance
(binned Kolmogorov-Smirnov, the largest gap between the two cumulative curves) and not a p-value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from signal_audit.data import audit

def bin_edges(m: dict, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper edge of each bin, in the units of the metric."""
    bins = np.asarray(bins, dtype=float)
    if m["scale"] == "log":
        return 10 ** (bins * m["width"]), 10 ** ((bins + 1) * m["width"])
    if m["scale"] == "int":
        return bins - 0.5, np.where(bins >= m["cap"], np.inf, bins + 0.5)
    return bins * m["width"], (bins + 1) * m["width"]

def complete(hist: pd.DataFrame, m: dict) -> pd.DataFrame:
    """Every bin between the first and the last one, with 0 where empty, and its edges and centre.

    Linear scale: bins outside `plot_range` (default `range`) are added to the first or last bin, so
    a few extreme values do not create millions of empty bins. `share` is the density used for the
    peaks and the plots; `count` stays the real count, used by the fits. For an integer metric on a
    log scale (`integer: true`), small bins hold one integer or none, which makes false peaks at 1, 2,
    3: `share` is then the count spread over the width of the bin, as a continuous value would be."""
    hist = hist[["bin", "count"]]
    overflow = set()
    if m["scale"] == "linear":
        lo_r, hi_r = m.get("plot_range", m["range"])
        first, last = int(np.floor(lo_r / m["width"])), int(np.floor(hi_r / m["width"]))
        overflow = {b for b, out in ((first, (hist["bin"] < first).any()), (last, (hist["bin"] > last).any())) if out}
        hist = hist.assign(bin=hist["bin"].clip(first, last)).groupby("bin", as_index=False)["count"].sum()
    full = np.arange(hist["bin"].min(), hist["bin"].max() + 1)
    h = hist.set_index("bin")["count"].reindex(full, fill_value=0).rename_axis("bin").reset_index()
    h["lo"], h["hi"] = bin_edges(m, h["bin"])
    density = h["count"].astype(float)
    if m.get("integer") and m["scale"] == "log":
        n_int = np.floor(h["hi"] - 1e-9) - np.ceil(h["lo"]) + 1
        h = h[n_int > 0].reset_index(drop=True)
        density = h["count"] * (h["hi"] - h["lo"]) / n_int[n_int > 0].to_numpy()
    h["share"] = density / density.sum()
    h["overflow"] = h["bin"].isin(overflow)   # an edge bin that also holds the values outside the plot range
    if m["scale"] == "log":
        h["x"] = 10 ** ((h["bin"] + 0.5) * m["width"])
    elif m["scale"] == "int":
        h["x"] = h["bin"].astype(float)
    else:
        h["x"] = (h["bin"] + 0.5) * m["width"]
    return h

def count_peaks(h: pd.DataFrame, cfg: dict) -> list[float]:
    """Positions (x) of the peaks of the smoothed histogram. The density is per log10 unit on a log
    scale, so a log-normal has one peak."""
    th = cfg["thresholds"]
    share = h["share"].where(~h["overflow"], 0) if "overflow" in h else h["share"]
    density = gaussian_filter1d(share.to_numpy(float), th["smooth_bins"], mode="constant")
    if density.max() <= 0:
        return []
    idx, _ = find_peaks(np.r_[0, density, 0], prominence=th["peak_prominence"] * density.max())
    return h["x"].to_numpy()[idx - 1].tolist()

def _binned_ks(h: pd.DataFrame, cdf) -> float:
    """Largest gap between the observed and the fitted cumulative share at the bin edges."""
    observed = h["count"].cumsum().to_numpy() / h["count"].sum()
    lo0 = cdf(np.array([h["lo"].iloc[0]]))[0]
    fitted = (cdf(h["hi"].to_numpy(float)) - lo0) / max(1 - lo0, 1e-12)
    return float(np.max(np.abs(observed - np.clip(fitted, 0, 1))))

def _binned_mle(h: pd.DataFrame, dist, start: list[float], truncate_below: float | None = None):
    """Parameters of a scipy distribution fitted to the binned counts (maximum likelihood)."""
    lo, hi, n = h["lo"].to_numpy(float), h["hi"].to_numpy(float), h["count"].to_numpy(float)

    def nll(params):
        try:
            d = dist(*params)
            p = d.cdf(hi) - d.cdf(lo)
            if truncate_below is not None:
                p = p / max(1 - d.cdf(truncate_below), 1e-300)
        except (ValueError, FloatingPointError):
            return np.inf
        return -np.sum(n * np.log(np.clip(p, 1e-300, None)))

    return optimize.minimize(nll, start, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6}).x

def _count_fits(h: pd.DataFrame, prof: pd.Series, m: dict) -> list[dict]:
    """Poisson and negative binomial (a Poisson whose rate changes between players), zero-truncated
    when the population has no zeros."""
    k = h["bin"].to_numpy(int)
    n = h["count"].to_numpy(float)
    trunc = k.min() >= 1
    open_last = m["scale"] == "int" and k.max() >= m["cap"]

    def pmf_table(pmf, sf):
        p = pmf(k).astype(float)
        if open_last:
            p[-1] = sf(k[-1] - 1)
        if trunc:
            p = p / max(1 - pmf(0), 1e-300)
        return p

    def fit_one(make, start):
        def nll(params):
            d = make(params)
            if d is None:
                return np.inf
            return -np.sum(n * np.log(np.clip(pmf_table(d.pmf, d.sf), 1e-300, None)))
        res = optimize.minimize(nll, start, method="Nelder-Mead", options={"maxiter": 4000})
        d = make(res.x)
        p = pmf_table(d.pmf, d.sf)
        return d, p

    mean, var = float(prof["mean"]), float(prof["var"])
    out = []
    pois, p_pois = fit_one(lambda a: sps.poisson(a[0]) if a[0] > 0 else None, [max(mean, 0.1)])
    out.append({"family": "Poisson", "params": f"lambda={pois.args[0]:.2f}",
                "ks": float(np.max(np.abs(np.cumsum(n) / n.sum() - np.cumsum(p_pois)))), "pmf": p_pois})
    r0 = mean ** 2 / max(var - mean, 1e-3)
    nb, p_nb = fit_one(lambda a: sps.nbinom(a[0], a[1]) if a[0] > 0 and 0 < a[1] < 1 else None,
                       [max(r0, 0.1), min(max(r0 / (r0 + mean), 0.01), 0.99)])
    out.append({"family": "Negative binomial", "params": f"r={nb.args[0]:.2f}, p={nb.args[1]:.3f}",
                "ks": float(np.max(np.abs(np.cumsum(n) / n.sum() - np.cumsum(p_nb)))), "pmf": p_nb})
    return out

def fit(cfg: dict, name: str, prof: pd.Series, hist: pd.DataFrame) -> dict:
    """Compare the metric with its expected family (and a close alternative)."""
    m = cfg["metrics"][name]
    th = cfg["thresholds"]
    h = complete(hist, m)
    peaks = count_peaks(h, cfg)
    fits = []
    if m["expected"] == "count" and m["scale"] == "int":
        fits = _count_fits(h, prof, m)
    elif m["expected"] in ("lognormal", "delay", "count"):
        positive = h[h["lo"] > 0] if m["scale"] != "log" else h
        if m["expected"] == "lognormal" or m["scale"] == "log":
            mu, sd = float(prof["mean_ln"]), float(prof["sd_ln"])
            d = sps.lognorm(sd, scale=np.exp(mu))
            fits.append({"family": "Log-normal", "params": f"median={np.exp(mu):.4g}, sigma={sd:.2f}",
                         "ks": _binned_ks(h, d.cdf), "dist": d})
        if m["expected"] == "delay":
            loc, scale = _binned_mle(positive, lambda a, b: sps.gumbel_r(a, abs(b)),
                                     [float(prof["mean"]) - 0.45 * np.sqrt(prof["var"]), 0.78 * np.sqrt(prof["var"])],
                                     truncate_below=float(positive["lo"].iloc[0]))
            d = sps.gumbel_r(loc, abs(scale))
            fits.append({"family": "Gumbel", "params": f"loc={loc:.3g}, scale={abs(scale):.3g}",
                         "ks": _binned_ks(positive, d.cdf), "dist": d})
    for f in fits:
        f["verdict"] = "close" if f["ks"] < th["ks_good"] else "acceptable" if f["ks"] < th["ks_fair"] else "different"
    best = min(fits, key=lambda f: f["ks"]) if fits else None
    return {"metric": name, "expected": m["expected"], "hist": h, "peaks": peaks, "fits": fits, "best": best}

def summary_row(cfg: dict, name: str, prof: pd.Series, result: dict) -> dict:
    """One line of the scorecard."""
    m = cfg["metrics"][name]
    best = result["best"]
    q = list(prof["quantiles"])
    impossible = (prof["below_range"] + prof["above_range"]) / max(prof["n"], 1)
    return {"metric": name, "table": m["table"], "expected": m["expected"], "rows": int(prof["rows"]),
            "null %": 100 * prof["nulls"] / max(prof["rows"], 1), "p1": q[1], "median": q[4], "p99": q[7],
            "max": prof["max"], "outside range %": 100 * impossible, "peaks": len(result["peaks"]),
            "best fit": best["family"] if best else "", "KS distance": best["ks"] if best else np.nan,
            "fit": best["verdict"] if best else ""}

def impossible_summary(by_day: pd.DataFrame) -> pd.DataFrame:
    """Per metric: rows, impossible share, first and last day with an impossible value, the worst day,
    the share in each brand, and the share in the rows that had no brand_id."""
    rows = []
    for name, g in by_day.groupby("metric"):
        hit = g[g["impossible"] > 0]
        day = g.groupby("day")[["rows", "impossible"]].sum()
        worst = (day["impossible"] / day["rows"].where(day["rows"] > 0)).idxmax() if len(hit) else None
        row = {"metric": name, "rows": int(g["rows"].sum()), "impossible": int(g["impossible"].sum()),
               "impossible %": 100 * g["impossible"].sum() / max(g["rows"].sum(), 1),
               "first day": hit["day"].min() if len(hit) else None, "last day": hit["day"].max() if len(hit) else None,
               "worst day": worst}
        for b, gb in g.groupby("brand_id"):
            row[f"brand {b} %"] = 100 * gb["impossible"].sum() / max(gb["rows"].sum(), 1)
        nb = g[g["brand_filled"]]
        row["no brand_id %"] = 100 * nb["impossible"].sum() / max(nb["rows"].sum(), 1)
        rows.append(row)
    return pd.DataFrame(rows)

def brand_results(cfg: dict, name: str, segment: str = "brand_id") -> dict:
    """Profile and fit of a metric in each brand: {brand: (profile, fit result)}."""
    profs = audit.profile_by_segment(cfg, name, segment).set_index("segment")
    hists = audit.segment_histograms(cfg, name, segment)
    out = {}
    for b, g in hists.groupby("segment"):
        if b not in profs.index:
            continue
        prof = profs.loc[b]
        out[b] = (prof, fit(cfg, name, prof, g[["bin", "count"]].reset_index(drop=True)))
    return out

def brand_scorecard(cfg: dict, names: list[str], segment: str = "brand_id") -> pd.DataFrame:
    """The scorecard of section 3 of 01_distributions with one line per metric and brand."""
    rows = []
    for name in names:
        for b, (prof, res) in brand_results(cfg, name, segment).items():
            rows.append({"brand": b, **summary_row(cfg, name, prof, res),
                         "peak positions": ", ".join(f"{x:.3g}" for x in res["peaks"])})
    return pd.DataFrame(rows)

def segment_peaks(cfg: dict, name: str, seg: pd.DataFrame) -> pd.DataFrame:
    """Rows and peaks of each segment."""
    m = cfg["metrics"][name]
    rows = []
    for s, g in seg.groupby("segment"):
        h = complete(g[["bin", "count"]], m)
        p = count_peaks(h, cfg)
        rows.append({"segment": s, "rows": int(g["count"].sum()), "peaks": len(p),
                     "peak positions": ", ".join(f"{x:.3g}" for x in p)})
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def outside_by_segment(cfg: dict, name: str, seg: pd.DataFrame) -> pd.DataFrame:
    """Share of rows outside the real-life range in each segment (at bin precision)."""
    m = cfg["metrics"][name]
    lo, hi = m["range"]
    mm = m | {"plot_range": [min(lo, -1e12), max(hi, 1e12)]} if m["scale"] == "linear" else m
    rows = []
    for s, g in seg.groupby("segment"):
        h = complete(g[["bin", "count"]], mm)
        total = h["count"].sum()
        rows.append({"segment": s, "rows": int(total),
                     "below range %": 100 * h.loc[h["hi"] <= lo, "count"].sum() / total,
                     "above range %": 100 * h.loc[h["lo"] > hi, "count"].sum() / total})   # a bin starting at hi also holds hi
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)

def density(h: pd.DataFrame, m: dict) -> np.ndarray:
    """Share per unit of the x axis (per log10 unit on a log scale), so two histograms with different
    bins can be drawn together."""
    if m["scale"] == "log":
        return h["share"].to_numpy(float) / np.log10(h["hi"] / h["lo"]).to_numpy(float)
    if m["scale"] == "linear":
        return h["share"].to_numpy(float) / m["width"]
    return h["share"].to_numpy(float)


# ================================================================ used by the notebooks only

def point_masses(cfg: dict, name: str, hist: pd.DataFrame, values=(0, 1, 2, 3, 4, 5, 10)) -> pd.DataFrame:
    """Share of rows at single exact values of an integer metric (or of a log bin holding one integer).
    A value much more frequent than its neighbours is a point mass: often a default or a rule of the
    computation (a session of one event lasts 1 second; a player who lost every stake has hold 100)."""
    m = cfg["metrics"][name]
    h = complete(hist, m)
    total = h["count"].sum()
    rows = []
    for v in values:
        hit = h[(h["lo"] <= v) & (h["hi"] > v)]
        if hit.empty:
            continue
        b = hit.iloc[0]
        single = m["scale"] == "int" or (m["scale"] == "log" and np.floor(b["hi"] - 1e-9) == np.ceil(b["lo"]))
        rows.append({"value": v, "share %": 100 * b["count"] / total, "exact value": bool(single)})
    return pd.DataFrame(rows)


def hist_quantile(h: pd.DataFrame, q: float) -> float:
    """Quantile from a completed histogram (upper edge of the bin where the cumulative share passes q)."""
    cum = h["count"].cumsum() / h["count"].sum()
    return float(h.loc[cum >= q, "hi"].iloc[0])
