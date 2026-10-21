"""Evidence of the errors of gld_player_signals_daily, for the notebooks and figures.py.

Part A (03_three_errors.ipynb): the three main errors, against the daily activity, on local files only
(the snapshots of the signals table per brand in data/02_intermediate/daily_cache/signals and the
daily activity cache):
1. time of play (`tp_*`): impossible session lengths, indices on the wrong scale, days not recomputed;
2. `days_since_bet` uses data from after its date;
3. `active_days_*` counts the days played only with playable bonus.
The activity cache has one row per player and day with a bet from any wallet; `active` marks the
official active day (measure M053: a stake from real money or released bonus).

Part B (01_distributions.ipynb and 02_validation.ipynb): SQL aggregates per brand on S3 (no sampling,
cached like the rest of the audit): how the session length is computed, the columns with index limits
and the peaks of hold_pct_l30d.

Player rows stay inside these functions; they return aggregates.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(PROJECT_ROOT / "src" / "features"), str(PROJECT_ROOT / "eda")]
import audit  # noqa: E402
import daily_cache  # noqa: E402
import signal_snapshots  # noqa: E402

# ================================================================ A. the three main errors

BRANDS = (14, 23, 64, 72, 73)
KEY = ["tenant_id", "player_id"]
LIVE_FROM = dt.date(2026, 8, 1)       # first snapshot written by the daily job (the earlier ones are the backfill)
TP_FROM = dt.date(2026, 7, 6)         # first day with time-of-play values


# ---------------------------------------------------------------- data

def activity(brand: int) -> pd.DataFrame:
    a = daily_cache.read("activity", brand)
    a["day"] = pd.to_datetime(a["day"])
    return a


def snapshot_days(brand: int) -> list[dt.date]:
    return sorted(dt.date.fromisoformat(p.stem) for p in signal_snapshots.cache_dir(brand).glob("*.parquet"))


def snapshot(brand: int, day: dt.date) -> pd.DataFrame:
    return signal_snapshots.snapshots([day], brand, build_missing=False).set_index(KEY)


def _recency(act: pd.DataFrame, end: pd.Timestamp) -> pd.Series:
    """Days from the last day with a bet up to `end` (inclusive) to `end`."""
    return (end - act[act["day"] <= end].groupby(KEY)["day"].max()).dt.days


def _days(act: pd.DataFrame, end: pd.Timestamp, window: int) -> pd.Series:
    """Distinct days in (end - window, end]."""
    part = act[(act["day"] > end - pd.Timedelta(window, unit="D")) & (act["day"] <= end)]
    return part.groupby(KEY)["day"].nunique()


# ---------------------------------------------------------------- 2. days_since_bet

def recency_players(brand: int, day: dt.date, act: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per player of the snapshot with a bet in (day - 30, day]: days_since_bet of the table, the
    recency as of the day and on the days around it, whether they bet in the next 7 days, and
    churn_score. Any wallet, as the table does (it matches the any-wallet recency best); the official
    recency as of the day is kept for reference. Internal: only aggregates leave this module."""
    act = activity(brand) if act is None else act
    ts = pd.Timestamp(day)
    pop = act[(act["day"] > ts - pd.Timedelta(30, unit="D")) & (act["day"] <= ts)][KEY].drop_duplicates()
    s = snapshot(brand, day).join(pop.set_index(KEY), how="inner")
    mine = act.merge(pop, on=KEY)                    # only the players of the snapshot: much faster
    mine = mine[mine["day"] <= ts + pd.Timedelta(10, unit="D")]
    out = pd.DataFrame({"table": s["days_since_bet"].astype(float), "churn_score": s["churn_score"].astype(float)})
    out["as_of_official"] = _recency(mine[mine["active"]], ts).reindex(out.index)
    # last day with a bet up to each date around the snapshot, from the sorted days of each player
    days = mine.groupby(KEY)["day"].apply(lambda d: np.sort(d.to_numpy())).reindex(out.index)
    for k in range(-3, 11):
        end = np.datetime64(ts + pd.Timedelta(k, unit="D"))
        last = days.map(lambda d: d[np.searchsorted(d, end, side="right") - 1]
                        if isinstance(d, np.ndarray) and np.searchsorted(d, end, side="right") > 0 else np.datetime64("NaT"))
        out[f"recency_{k:+d}"] = ((end - pd.to_datetime(last)).dt.days).to_numpy()
    out["as_of"] = out["recency_+0"]
    nxt = mine[(mine["day"] > ts) & (mine["day"] <= ts + pd.Timedelta(7, unit="D"))]
    out["bets_next_7d"] = out.index.isin(pd.MultiIndex.from_frame(nxt[KEY].drop_duplicates()))
    return out


def recency_live_days(brand: int, act: pd.DataFrame) -> list[dt.date]:
    """Live snapshots with 10 days of activity after them (to test later days)."""
    last = act["day"].max().date()
    return [d for d in snapshot_days(brand) if d >= LIVE_FROM and d + dt.timedelta(days=10) <= last]


def _match(a: pd.Series, b: pd.Series) -> float:
    both = pd.concat([a, b], axis=1).dropna()
    return float(np.isclose(both.iloc[:, 0], both.iloc[:, 1], atol=0.5).mean())


def _auc(score: pd.Series, label: pd.Series) -> float:
    ok = score.notna()
    return float(roc_auc_score(label[ok], score[ok])) if label[ok].nunique() == 2 else np.nan


def recency_evidence(brands=BRANDS, refresh: bool = False):
    """Cached in cache/local (slow: player rows of local files); `refresh` computes it again."""
    return audit.cached_local("recency_evidence", lambda: _recency_evidence(brands), refresh)


def _recency_evidence(brands=BRANDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per brand and live snapshot: match with the recency as of the day and N days later, the share
    of players whose table recency is shorter than any bet up to the day (impossible without later
    data) and, of those, the share that bet in the next 7 days; AUC of each recency to predict a bet
    in the next 7 days. Second frame: match per shift, to see on which day the table matches best."""
    rows, shifts = [], []
    for b in brands:
        act = activity(b)
        for day in recency_live_days(b, act):
            p = recency_players(b, day, act)
            short = p["table"] < p["as_of"]                 # last bet more recent than any bet up to the day
            rows.append({
                "brand": b, "day": day, "players": len(p),
                "match as of the day %": 100 * _match(p["table"], p["as_of"]),
                "match 7 days later %": 100 * _match(p["table"], p["recency_+7"]),
                "table shorter than any bet up to the day %": 100 * short.mean(),
                "of those, bet in the next 7 days %": 100 * p.loc[short, "bets_next_7d"].mean(),
                "all players, bet in the next 7 days %": 100 * p["bets_next_7d"].mean(),
                "AUC next 7 days: table days_since_bet": _auc(-p["table"], p["bets_next_7d"]),
                "AUC next 7 days: recency as of the day": _auc(-p["as_of"], p["bets_next_7d"]),
                "AUC next 7 days: churn_score": _auc(-p["churn_score"], p["bets_next_7d"]),
            })
            for k in range(-3, 11):
                shifts.append({"brand": b, "day": day, "shift": k, "match %": 100 * _match(p["table"], p[f"recency_{k:+d}"])})
    return pd.DataFrame(rows), pd.DataFrame(shifts)


def recency_differences(brand: int, day: dt.date) -> pd.DataFrame:
    """Counts of (table - recency as of the day) and (table - recency 7 days later), clipped to +-30."""
    p = recency_players(brand, day)
    out = {}
    for name, col in (("as of the day", "as_of"), ("7 days later", "recency_+7")):
        d = (p["table"] - p[col]).dropna().clip(-30, 30).round().astype(int)
        out[name] = d.value_counts()
    return pd.DataFrame(out).reindex(range(-30, 31)).fillna(0).astype(int)


def recency_over_time(brands=BRANDS) -> pd.DataFrame:
    """Every month-start snapshot (backfill and live): share with days_since_bet = 0 and the match as
    of the day and 7 days later. From the saved results of eda/signal_validation.py."""
    import signal_validation as sv
    res = sv.saved_results(brands)
    rows = []
    for b, r in res.items():
        g = r["recomputed"][r["recomputed"]["column"] == "days_since_bet"]
        rows.append(pd.DataFrame({"brand": b, "day": g["day"], "players": g["players"],
                                  "table is 0 %": 100 * g["table_zero"], "match as of the day %": 100 * g["match"],
                                  "match 7 days later %": 100 * g["match_7d_ahead"]}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 3. active days

WINDOWS = {"active_days_l7d": 7, "active_days_l30d": 30, "active_days_l90d": 90}


def active_days_players(brand: int, day: dt.date, act: pd.DataFrame | None = None, window: int = 30) -> pd.DataFrame:
    """Per player with a bet from any wallet in (day - 30, day]: the table value and both recounts.
    Internal: only aggregates leave this module."""
    act = activity(brand) if act is None else act
    ts = pd.Timestamp(day)
    pop = act[(act["day"] > ts - pd.Timedelta(30, unit="D")) & (act["day"] <= ts)][KEY].drop_duplicates()
    s = snapshot(brand, day).join(pop.set_index(KEY), how="inner")
    col = f"active_days_l{window}d"
    return pd.DataFrame({"table": s[col].astype(float),
                         "official": _days(act[act["active"]], ts, window).reindex(s.index).fillna(0),
                         "any wallet": _days(act, ts, window).reindex(s.index).fillna(0)})


def active_days_over_time(brands=BRANDS, refresh: bool = False):
    """Cached in cache/local (slow: player rows of local files); `refresh` computes it again."""
    return audit.cached_local("active_days_over_time", lambda: _active_days_over_time(brands), refresh)


def _active_days_over_time(brands=BRANDS) -> pd.DataFrame:
    """Per brand, month-start snapshot and window: share of players where the table equals each recount,
    and where it is above or below the official one."""
    rows = []
    for b in brands:
        act = activity(b)
        for day in [d for d in snapshot_days(b) if d.day == 1 and d >= dt.date(2025, 7, 1)]:
            for col, w in WINDOWS.items():
                p = active_days_players(b, day, act, w)
                rows.append({"brand": b, "day": day, "column": col, "players": len(p),
                             "= official %": 100 * (p["table"] == p["official"]).mean(),
                             "= any wallet %": 100 * (p["table"] == p["any wallet"]).mean(),
                             "above official %": 100 * (p["table"] > p["official"]).mean(),
                             "below official %": 100 * (p["table"] < p["official"]).mean()})
    return pd.DataFrame(rows)


def active_days_differences(brand: int, day: dt.date) -> pd.DataFrame:
    """Counts of (table - official) and (table - any wallet) for active_days_l30d."""
    p = active_days_players(brand, day)
    return pd.DataFrame({"table - official": (p["table"] - p["official"]).value_counts(),
                         "table - any wallet": (p["table"] - p["any wallet"]).value_counts()}).fillna(0).astype(int).sort_index()


def false_actives(brands=BRANDS, day: dt.date = dt.date(2026, 9, 1)) -> pd.DataFrame:
    """Players who bet in the last 30 days only with playable bonus: 0 official active days, but the
    table gives them active days. For churn they are inactive; the table shows them as active."""
    rows = []
    for b in brands:
        p = active_days_players(b, day)
        only_bonus = p["official"] == 0
        rows.append({"brand": b, "players with a bet in 30 days": len(p),
                     "only playable bonus": int(only_bonus.sum()),
                     "only playable bonus %": 100 * only_bonus.mean(),
                     "of those, table active days > 0 %": 100 * (p.loc[only_bonus, "table"] > 0).mean(),
                     "their mean table active days": p.loc[only_bonus, "table"].mean()})
    return pd.DataFrame(rows)


def playable_only_share(brands=BRANDS) -> pd.DataFrame:
    """Share of player-days with bets only from playable bonus, per brand and month."""
    rows = []
    for b in brands:
        a = activity(b)
        m = a.groupby(a["day"].dt.to_period("M"))["active"].agg(lambda s: 100 * (~s).mean())
        rows.append(m.rename(b))
    return pd.concat(rows, axis=1).rename_axis("month")


# ---------------------------------------------------------------- 1. time of play


def repeated_days(by_day: pd.DataFrame, metric: str = "night_index", control: str = "active_days_l30d") -> pd.DataFrame:
    """Days where a tp_* metric has exactly the rows and impossible values of the day before, while a
    control column of the same table changes."""
    def daily(name):
        d = by_day[by_day["metric"] == name].groupby("day")[["rows", "impossible"]].sum()
        return d[d["rows"] > 0]
    t, c = daily(metric), daily(control)
    same = (t.diff().abs().sum(axis=1, min_count=1) == 0)
    control_same = (c.diff().abs().sum(axis=1, min_count=1) == 0).reindex(t.index)
    out = t[same].assign(**{"same as the day before": True, f"{control} changes": ~control_same[same]})
    return out.reset_index()


# ================================================================ B. session length, index limits, hold

TABLE = "gld_player_signals_daily"

# ---------------------------------------------------------------- 3. session length

TP = "tp_avg_session_duration_secs > 0"   # rows with time-of-play data (0 = no data)
# name: (rows it applies to, condition, what it tests). Each one is a hypothesis about how the table
# computes the session length, or a rule that a real session must follow.
SESSION_CHECKS = {
    "mean = play_secs / sessions (30 days)": (
        f"{TP} AND sessions_l30d > 0",
        "abs(tp_avg_session_duration_secs - play_secs_l30d / sessions_l30d) <= 0.02 * tp_avg_session_duration_secs + 1",
        "The mean session is the play time of 30 days over the sessions of 30 days"),
    "mean = play_secs / sessions (7 days)": (
        f"{TP} AND sessions_l7d > 0",
        "abs(tp_avg_session_duration_secs - play_secs_l7d / sessions_l7d) <= 0.02 * tp_avg_session_duration_secs + 1",
        "The same with 7 days"),
    "mean = median": (TP, "tp_avg_session_duration_secs = tp_median_session_duration_secs",
                      "Only one session in the window, or every session the same length"),
    "median > mean": (TP, "tp_median_session_duration_secs > tp_avg_session_duration_secs",
                      "Rare for a right-skewed length (log-normal): the median is usually below the mean"),
    "mean exactly 1 s": (TP, "tp_avg_session_duration_secs = 1", "Sessions of one event: the start equals the end"),
    "mean exactly 1 s, one session in 30 days": (f"{TP} AND tp_avg_session_duration_secs = 1", "sessions_l30d <= 1",
                                                 "Of the 1-second rows: a single session"),
    "mean over 1 day": (TP, "tp_avg_session_duration_secs > 86400", "A session that never closes"),
    "mean x sessions per active day > 24 h": (
        f"{TP} AND tp_avg_sessions_per_active_day > 0",
        "tp_avg_session_duration_secs * tp_avg_sessions_per_active_day > 86400",
        "The sessions of one day last longer than the day"),
    "mean > window": (f"{TP} AND tp_window_days > 0", "tp_avg_session_duration_secs > tp_window_days * 86400",
                      "One session longer than the whole window it is computed on"),
    "gap between sessions < 0": ("tp_avg_session_gap_hours <> 0", "tp_avg_session_gap_hours < 0",
                                 "Negative gap: a session starts before the previous one ends"),
    "play_secs today > 24 h": ("play_secs_today > 0", "play_secs_today > 86400", "More play time than the day has"),
    "play_secs 30 days > 30 days": ("play_secs_l30d > 0", "play_secs_l30d > 30 * 86400",
                                    "More play time than the 30 days have"),
}
SESSION_COLUMNS = ["tp_avg_session_duration_secs", "tp_median_session_duration_secs", "tp_avg_session_gap_hours",
                   "tp_avg_sessions_per_active_day", "tp_window_days", "sessions_l7d", "sessions_l30d",
                   "play_secs_today", "play_secs_l7d", "play_secs_l30d"]


def _key(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _by_brand(cfg: dict, name: str, select: str, columns: list[str]) -> pd.DataFrame:
    """One aggregate query over the signals table of the period, per brand and for all brands
    (GROUPING SETS), cached."""
    def compute():
        src = audit.source(cfg, TABLE, columns=columns)
        return audit.connection().sql(
            f"SELECT COALESCE(CAST(brand_id AS VARCHAR), 'all') AS brand, {select} FROM {src} "
            f"GROUP BY GROUPING SETS ((brand_id), ()) ORDER BY brand").df()
    return audit.cached(cfg, f"feedback__{name}__{_key([select, columns, cfg['brands']])}", compute)


def session_checks(cfg: dict) -> pd.DataFrame:
    """Share of rows where each session check holds, per brand. Long format."""
    parts = []
    for k, (pop, cond, _) in enumerate(SESSION_CHECKS.values()):
        parts += [f"count_if({pop}) AS n{k}", f"count_if(({pop}) AND ({cond})) AS h{k}"]
    wide = _by_brand(cfg, "session_checks", ", ".join(parts), SESSION_COLUMNS)
    rows = []
    for k, (name, (_, _, why)) in enumerate(SESSION_CHECKS.items()):
        for _, r in wide.iterrows():
            rows.append({"check": name, "what it tests": why, "brand": r["brand"], "rows": int(r[f"n{k}"]),
                         "holds %": 100 * r[f"h{k}"] / r[f"n{k}"] if r[f"n{k}"] else np.nan})
    return pd.DataFrame(rows)


def session_ratio(cfg: dict) -> pd.DataFrame:
    """Quantiles of the mean session over play_secs / sessions (30 days), and of the window, per brand."""
    q = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    ratio = "tp_avg_session_duration_secs / (play_secs_l30d / sessions_l30d)"
    sel = (f"count_if({TP} AND sessions_l30d > 0 AND play_secs_l30d > 0) AS rows, "
           f"quantile_cont({ratio}, {q}) FILTER (WHERE {TP} AND sessions_l30d > 0 AND play_secs_l30d > 0) AS ratio, "
           f"quantile_cont(tp_window_days, {q}) FILTER (WHERE {TP}) AS window_days")
    out = _by_brand(cfg, "session_ratio", sel, SESSION_COLUMNS)
    for col in ("ratio", "window_days"):
        out[[f"{col} p{int(100 * x)}" for x in q]] = pd.DataFrame(out.pop(col).tolist(), index=out.index)
    return out


def window_days(cfg: dict) -> pd.DataFrame:
    """Rows per value of tp_window_days (rows with time-of-play data), all brands."""
    def compute():
        src = audit.source(cfg, TABLE, columns=SESSION_COLUMNS)
        return audit.connection().sql(f"SELECT tp_window_days AS window_days, count(*) AS rows FROM {src} "
                                      f"WHERE {TP} GROUP BY ALL ORDER BY ALL").df()
    return audit.cached(cfg, f"feedback__window_days__{_key(cfg['brands'])}", compute)


# ---------------------------------------------------------------- 4. columns with index limits

# column: (lower limit, upper limit, what the workbook says). Scores are 0 to 100 in production; the
# workbook writes the unit 0-1 for all of them.
INDEX_LIMITS = {
    "churn_score": (0, 100, "score, workbook unit 0-1"),
    "recency_pressure": (0, 100, "recency_score, workbook unit 0-1"),
    "frequency_decline": (0, 100, "frequency_score, workbook unit 0-1"),
    "value_decline": (0, 100, "monetary_score, workbook unit 0-1; the formula has no upper clamp"),
    "engagement_score": (0, 100, "score, workbook unit 0-1"),
    "risk_score": (0, 100, "rg_risk_score, workbook unit 0-1"),
    "risk_appetite_score": (0, 100, "score, workbook unit 0-1"),
    "betting_aggression_score": (0, 100, "score, workbook unit 0-1"),
    "session_intensity_score": (0, 100, "min(100, ...)"),
    "deposit_frequency_score": (0, 100, "score, workbook unit 0-1"),
    "deposit_recency_score": (0, 100, "score, workbook unit 0-1"),
    "prior_churn_score": (0, 100, "score of the previous window"),
    "prior_engagement_score": (0, 100, "score of the previous window"),
    "prior_risk_score": (0, 100, "score of the previous window"),
    "prior_betting_aggression_score": (0, 100, "score of the previous window"),
    "confidence_score": (0, 1, "workbook unit 0-1"),
    "rg_loss_chasing_30d": (0, 1, "rg_loss_chasing, workbook unit 0-1"),
    "rg_night_play_ratio": (0, 1, "night_index, a share"),
    "tp_night_index": (0, 1, "a share"),
    "tp_weekend_index": (0, 1, "a share"),
    "tp_hourly_entropy": (0, 1, "entropy, workbook unit 0-1"),
    "tp_business_hours_ratio": (0, 1, "a share"),
    "hold_pct_l30d": (None, 100, "GGR / stake x 100: at most 100"),
    "stake_escalation": (-1, None, "(avg bet 7d - avg bet 30d) / avg bet 30d: at least -1"),
    "top_peak_hour": (0, 23, "an hour"),
    "top_peak_day": (0, 6, "a day of the week"),
}
BINS = 50   # bins of each histogram, between the two limits (plus one below and one above)


def index_limits(cfg: dict, columns: dict = INDEX_LIMITS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each column with limits, per brand: rows, nulls, min, max, median, p99, share below the
    lower limit, above the upper one, exactly at each limit and at 0; and a histogram of 50 bins
    between the limits (open limits use the 1st or 99th percentile of the period). One scan."""
    cols = list(columns)
    parts = []
    for k, (c, (lo, hi, _)) in enumerate(columns.items()):
        x = f"CAST({c} AS DOUBLE)"
        parts += [f"count({x}) AS n{k}", f"count_if({c} IS NULL) AS null{k}", f"min({x}) AS min{k}", f"max({x}) AS max{k}",
                  f"quantile_cont({x}, [0.01, 0.5, 0.99]) AS q{k}", f"count_if({x} = 0) AS zero{k}",
                  f"count_if({x} < {lo}) AS below{k}" if lo is not None else f"0 AS below{k}",
                  f"count_if({x} > {hi}) AS above{k}" if hi is not None else f"0 AS above{k}",
                  f"count_if({x} = {lo}) AS atlo{k}" if lo is not None else f"0 AS atlo{k}",
                  f"count_if({x} = {hi}) AS athi{k}" if hi is not None else f"0 AS athi{k}"]
    wide = _by_brand(cfg, "index_limits", ", ".join(parts), cols)
    rows = []
    for k, (c, (lo, hi, note)) in enumerate(columns.items()):
        for _, r in wide.iterrows():
            n = r[f"n{k}"]
            q = r[f"q{k}"] if r[f"q{k}"] is not None else [np.nan] * 3
            rows.append({"column": c, "brand": r["brand"], "limits": f"[{'' if lo is None else lo}, {'' if hi is None else hi}]",
                         "note": note, "rows": int(n), "nulls": int(r[f"null{k}"]), "min": r[f"min{k}"], "max": r[f"max{k}"],
                         "p1": q[0], "median": q[1], "p99": q[2],
                         "below %": 100 * r[f"below{k}"] / n if n else np.nan,
                         "above %": 100 * r[f"above{k}"] / n if n else np.nan,
                         "at lower limit %": 100 * r[f"atlo{k}"] / n if n else np.nan,
                         "at upper limit %": 100 * r[f"athi{k}"] / n if n else np.nan,
                         "zero %": 100 * r[f"zero{k}"] / n if n else np.nan})
    summary = pd.DataFrame(rows)

    # histograms: the limits come from the config, an open limit from the percentiles of all brands
    allq = summary[summary["brand"] == "all"].set_index("column")
    edges = {c: (lo if lo is not None else allq.loc[c, "p1"], hi if hi is not None else allq.loc[c, "p99"])
             for c, (lo, hi, _) in columns.items()}
    edges = {c: (a, b) for c, (a, b) in edges.items() if pd.notna(a) and pd.notna(b) and b > a}

    def compute_hist():
        src = audit.source(cfg, TABLE, columns=cols)
        sel = " UNION ALL ".join(
            f"SELECT '{c}' AS col, CAST(brand_id AS VARCHAR) AS brand, "
            f"CAST(least(greatest(floor((CAST({c} AS DOUBLE) - CAST({float(a)!r} AS DOUBLE)) / CAST({float(b) - float(a)!r} AS DOUBLE) * {BINS}), -1), {BINS}) AS INTEGER) AS bin "
            f"FROM r0 WHERE {c} IS NOT NULL" for c, (a, b) in edges.items())
        return audit.connection().sql(f"WITH r0 AS (SELECT * FROM {src}) SELECT col AS \"column\", brand, bin, count(*) AS count "
                                      f"FROM ({sel}) GROUP BY ALL ORDER BY ALL").df()
    hist = audit.cached(cfg, f"feedback__index_hist__{_key([edges, cfg['brands']])}", compute_hist)
    hist["lo"] = hist["column"].map(lambda c: edges[c][0])
    hist["hi"] = hist["column"].map(lambda c: edges[c][1])
    return summary, hist


# ---------------------------------------------------------------- 5. hold_pct_l30d

HOLD_BUCKETS = "CASE WHEN bets_l30d <= 1 THEN '1 bet' WHEN bets_l30d <= 5 THEN '2 to 5' WHEN bets_l30d <= 20 THEN '6 to 20' " \
               "WHEN bets_l30d <= 100 THEN '21 to 100' ELSE 'over 100' END"
HOLD_COLUMNS = ["hold_pct_l30d", "bets_l30d", "wagered_eur_l30d", "won_eur_l30d", "ggr_eur_l30d"]


def hold_masses(cfg: dict) -> pd.DataFrame:
    """Rows of hold_pct_l30d at the values that look like peaks, per brand and bets in 30 days."""
    sel = ("count(*) AS rows, count_if(hold_pct_l30d < -100) AS \"below -100\", "
           "count_if(hold_pct_l30d >= -101 AND hold_pct_l30d < -99) AS \"-101 to -99\", "
           "count_if(hold_pct_l30d >= -100.01 AND hold_pct_l30d <= -99.99) AS \"exactly -100\", "
           "count_if(hold_pct_l30d >= 0 AND hold_pct_l30d < 20) AS \"0 to 20\", "
           "count_if(hold_pct_l30d >= 99.99) AS \"exactly 100\", "
           "count_if(won_eur_l30d = 0) AS \"won nothing\"")

    def compute():
        src = audit.source(cfg, TABLE, columns=HOLD_COLUMNS)
        return audit.connection().sql(
            f"SELECT COALESCE(CAST(brand_id AS VARCHAR), 'all') AS brand, COALESCE({HOLD_BUCKETS}, 'all') AS bets_30d, {sel} "
            f"FROM {src} WHERE wagered_eur_l30d > 0 "
            f"GROUP BY GROUPING SETS ((brand_id, {HOLD_BUCKETS}), (brand_id), ({HOLD_BUCKETS}), ()) ORDER BY ALL").df()
    return audit.cached(cfg, f"feedback__hold_masses__{_key([sel, HOLD_BUCKETS, cfg['brands']])}", compute)


def hold_hist(cfg: dict) -> pd.DataFrame:
    """Histogram of hold_pct_l30d per bets bucket, bins of 1 point from -300 to 100; the first bin
    holds everything below -300."""
    def compute():
        src = audit.source(cfg, TABLE, columns=HOLD_COLUMNS)
        return audit.connection().sql(
            f"SELECT {HOLD_BUCKETS} AS bets_30d, CAST(greatest(floor(hold_pct_l30d), -301) AS INTEGER) AS bin, count(*) AS count "
            f"FROM {src} WHERE wagered_eur_l30d > 0 GROUP BY ALL ORDER BY ALL").df()
    return audit.cached(cfg, f"feedback__hold_hist__{_key([HOLD_BUCKETS, cfg['brands']])}", compute)


# ================================================================ C. status of every column

# Columns with an error of their own, or affected through a wrong input, or with a definition to confirm,
# and the point of the error list that explains it. Every other column is "correct" when it passes its
# checks of signal_checks.yaml, or "not checkable" (keys, text, metadata).
WRONG = {
    "it:session": ["tp_avg_session_duration_secs", "tp_median_session_duration_secs"],
    "it:gap": ["tp_avg_session_gap_hours"],
    "it:playsecs": ["play_secs_today", "play_secs_l7d", "play_secs_l30d", "play_secs_l90d", "play_secs_l180d"],
    "it:night": ["tp_night_index"], "it:rgnight": ["rg_night_play_ratio"], "it:weekend": ["tp_weekend_index"],
    "it:entropy": ["tp_hourly_entropy"], "it:recency": ["days_since_bet"],
    "it:activedays": ["active_days_today", "active_days_l7d", "active_days_l30d", "active_days_l90d", "active_days_l180d",
                      "prior_active_days_l7d", "prior_active_days_l30d"],
    "it:tenure": ["tenure_days"], "it:valuedecline": ["value_decline"], "it:recencypressure": ["recency_pressure"],
    "it:peak": ["top_peak_hour", "top_peak_day"], "it:hold": ["hold_pct_l30d"], "it:depfreq": ["deposit_frequency_score"],
    "it:depositrecency": ["deposit_recency_score"], "it:brand": ["brand_id"],
}
AFFECTED = {
    "it:tpdays": ["tp_avg_sessions_per_active_day", "tp_max_sessions_single_day", "tp_business_hours_ratio", "tp_window_days"],
    "it:churnscore": ["churn_score", "churn_propensity_raw", "churn_band"],
    "it:activedays": ["frequency_decline", "engagement_score"],
    "it:playsecs": ["session_intensity_score"],
    "it:rgnight": ["risk_score", "risk_band"],
}
UNCONFIRMED = {
    "it:ngr": ["ngr_eur_today", "ngr_eur_l7d", "ngr_eur_l30d", "ngr_eur_l90d", "ngr_eur_l180d"],
    "it:prior": ["prior_churn_score", "prior_engagement_score", "prior_risk_score", "prior_betting_aggression_score"],
    "it:z": ["bet_z_score", "session_len_z"],
}


def column_status(refresh: bool = False):
    """Cached in cache/local (slow: player rows of local files); `refresh` computes it again."""
    return audit.cached_local("column_status", lambda: _column_status(), refresh)


def _column_status() -> pd.DataFrame:
    """One row per column of the signals table: status (wrong, affected, unconfirmed, correct or
    unchecked), its checks of signal_checks.yaml, the worst pass share of those checks on the two
    snapshot days, the point of the error list, and the expression of its formula or identity (which
    shows the inputs an affected column takes from a wrong one). Wrong, affected and unconfirmed come
    from the groups above (the evidence is in the notebooks); every other column is correct when it
    passes its checks, or unchecked (keys, text, metadata, bands without cutoffs)."""
    import formula_check as fc
    cfg = fc.load()
    ks = fc.checks(cfg)
    res = fc.run(cfg)
    worst = res.groupby("column")["pass %"].min()
    kinds = ks.groupby("column")["kind"].agg(lambda k: ", ".join(sorted(set(k))))
    exprs = ks[ks["kind"].isin(["formula", "identity"])].groupby("column")["expr"].first()
    point = {}
    for status, groups in (("unconfirmed", UNCONFIRMED), ("affected", AFFECTED), ("wrong", WRONG)):
        for item, cols in groups.items():
            for c in cols:
                point[c] = (status, item)
    rows = []
    for c in dict.fromkeys(ks["column"]):
        default = "unchecked" if kinds[c] in ("none", "band") else "correct"
        status, item = point.get(c, (default, ""))
        rows.append({"column": c, "status": status, "checks": "" if kinds[c] == "none" else kinds[c],
                     "worst %": worst.get(c, np.nan), "item": item, "expression": exprs.get(c, "")})
    return pd.DataFrame(rows)


# ================================================================ D. the other points of the error list

VALIDATION = PROJECT_ROOT / "data/03_output/signal_validation"   # saved by eda/signal_validation.py (notebook eda/08)


def tenure_growth(brands=BRANDS) -> pd.DataFrame:
    """tenure_days must grow by the days between two monthly snapshots: share of players for which it
    does, per brand and snapshot (saved results of eda/signal_validation.py)."""
    return pd.concat([pd.read_csv(VALIDATION / f"brand{b}" / "tenure_growth.csv", parse_dates=["day"]).assign(brand=b)
                      for b in brands], ignore_index=True)


def deposit_frequency_filled(brands=BRANDS) -> pd.DataFrame:
    """Snapshots where deposit_frequency_score is not 0 for every player (1) or is (0), per brand."""
    rows = []
    for b in brands:
        p = pd.read_csv(VALIDATION / f"brand{b}" / "plausibility.csv")
        p = p[(p["column"] == "deposit_frequency_score") & p["rule"].str.startswith("filled")]
        rows.append(p.set_index("day")["holds"].rename(b))
    return pd.concat(rows, axis=1).rename_axis("snapshot")


def footer_extremes() -> pd.DataFrame:
    """From the Parquet metadata of every daily file (all operators, no S3): daily maximum of
    hold_pct_l30d and minimum of won_eur_l30d. Above 100% and below 0 are impossible."""
    hold = audit.footer_range(TABLE, "hold_pct_l30d")["max_num"].rename("max hold %")
    won = audit.footer_range(TABLE, "won_eur_l30d")["min_num"].rename("min won EUR")
    out = pd.concat([hold, won], axis=1)
    out.index = pd.to_datetime(out.index)
    return out


def brand_nulls() -> pd.Series:
    """Share of rows with an empty brand_id per day (Parquet metadata, all operators)."""
    n = audit.footer_nulls(TABLE).loc["brand_id"]
    n.index = pd.to_datetime(n.index)
    return n


