"""Validation of gld_player_signals_daily: is every signal column what its name says, as of its date?

    .venv/bin/python eda/signal_validation.py                 # brands 14, 64 and 73, monthly snapshots
    make validate-signals [BRANDS="14 64 73"] [BUILD=1]       # BUILD=1: build missing snapshots from S3

The signals table is used for the first time, and nobody guarantees its columns, so every column the
features read is checked on the monthly snapshots (the players with a bet in the 30 days up to each
day), with local files only (no S3):

1. Recomputation. A column that can be rebuilt from the daily tables (activity, financial, payments
   caches of src/features/daily_cache.py) is compared player by player: it matches when it is within
   2% (or 0.5 absolute) of the recomputed value; the column is OK on a day when 95% of the players match.
2. Point in time. The same recomputation on a window moved 7 days forward: if the column matches the
   future window better than its own one, it was built with data after its date (a leak).
3. Plausibility. A column that cannot be rebuilt (sessions, play time, games, night play, the composite
   scores) is checked for what must hold: sessions >= active days, engagement_score = its documented
   formula on the table's own columns, valid ranges, filled on each day, and point in time: a 7-day
   column's window must end on its date (where its correlation with daily bets drops), and a 30-day
   column must follow the bet days of its own 30 days more than those of the 30 days ending 7 days later.
4. Recency and tenure, the two columns the pipeline replaces, in detail.

Outputs: docs/signal_validation.md (one short report for every brand: method, failed checks, figures),
the tables and figures of each brand in data/03_output/signal_validation/brand{id}/. Notebook, one brand
in detail: eda/08_signal_validation.ipynb.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
import daily_cache  # noqa: E402
import signal_snapshots  # noqa: E402

KEY = ["tenant_id", "player_id"]
TOLERANCE, ATOL, MIN_MATCH = 0.02, 0.5, 0.95
SHIFT_DAYS = 7
BACKFILL_DATE = dt.date(2026, 8, 8)
LIVE_FROM = dt.date(2026, 8, 1)  # the first snapshot already built the live way (days_since_bet, tenure_days)  # the source tables were recomputed up to this day (data card)
BRANDS = (14, 23, 64, 72, 73)
S3_HINT = "org/40-gold"
PER_ROW = 3  # brands per row in the report figures
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
DOC_PATH = PROJECT_ROOT / "docs/signal_validation.md"
SUMMARY_DIR = PROJECT_ROOT / "data/03_output/signal_validation"
OUTPUT_DIR = SUMMARY_DIR / "brand{brand}"

# column: (cache, source column, how, window): the window is (day - window, day], or for "prior" columns
# the window before it, (day - 2 * window, day - window]. how: days = distinct active days, sum, recency.
RECOMPUTABLE = {
    "active_days_l7d": ("activity", "day", "days", 7), "active_days_l30d": ("activity", "day", "days", 30),
    "active_days_l90d": ("activity", "day", "days", 90),
    "prior_active_days_l7d": ("activity", "day", "days_prior", 7), "prior_active_days_l30d": ("activity", "day", "days_prior", 30),
    "bets_l7d": ("activity", "bets", "sum", 7), "bets_l30d": ("activity", "bets", "sum", 30),
    "prior_bets_l7d": ("activity", "bets", "sum_prior", 7),
    "wagered_eur_l7d": ("activity", "turnover_eur", "sum", 7), "wagered_eur_l30d": ("activity", "turnover_eur", "sum", 30),
    "wagered_eur_l90d": ("activity", "turnover_eur", "sum", 90),
    "prior_wagered_eur_l7d": ("activity", "turnover_eur", "sum_prior", 7),
    "ggr_eur_l7d": ("activity", "ggr_eur", "sum", 7), "ggr_eur_l30d": ("activity", "ggr_eur", "sum", 30),
    "ggr_eur_l90d": ("activity", "ggr_eur", "sum", 90), "prior_ggr_eur_l30d": ("activity", "ggr_eur", "sum_prior", 30),
    "wagered_bonus_eur_l30d": ("financial", "wagered_bonus_eur", "sum", 30),
    "deposits_eur_l30d": ("payments", "deposit_eur", "sum", 30), "withdrawals_eur_l30d": ("payments", "withdraw_eur", "sum", 30),
    "avg_bet_eur_l30d": ("activity", None, "avg_bet", 30),
    "days_since_bet": ("activity", "day", "recency", None),
}
# Which feature reads each column (docs/feature_dictionary.md); "replaced" = recomputed by the pipeline.
USED_BY = {
    "active_days_l7d": "active_days_l7d", "active_days_l30d": "active_days_l30d", "active_days_l90d": "active_days_l90d",
    "prior_active_days_l7d": "prior_active_days_l7d", "bets_l7d": "bets_l7d", "bets_l30d": "bets_l30d",
    "wagered_eur_l7d": "wagered_eur_l7d", "wagered_eur_l30d": "wagered_eur_l30d", "wagered_eur_l90d": "wagered_eur_l90d",
    "prior_wagered_eur_l7d": "prior_wagered_eur_l7d", "ggr_eur_l7d": "net_loss_7d", "ggr_eur_l30d": "ggr_eur_l30d",
    "ggr_eur_l90d": "heavy_loss_multiple", "prior_ggr_eur_l30d": "prior_ggr_eur_l30d",
    "wagered_bonus_eur_l30d": "bonus_stake_share_30d", "deposits_eur_l30d": "deposits_eur_l30d",
    "avg_bet_eur_l30d": "avg_bet_eur_l30d", "days_since_bet": "replaced by days_since_last_bet",
    "tenure_days": "replaced by days_since_first_bet", "sessions_l7d": "sessions_l7d", "sessions_l30d": "sessions_l30d",
    "engagement_score": "engagement_score", "games_breadth_l30d": "games_breadth_30d", "tp_night_index": "night_play_index",
    "deposit_frequency_score": "deposit_frequency_score",
}
ENGAGEMENT_WEIGHTS = (0.4, 0.3, 0.3)  # active days, sessions, play time (gld-schemas.xlsx)
# Columns that cannot be rebuilt from the cached daily tables: the activity they summarise (bet days, or
# deposit days for the deposit score) and their window, for the point-in-time checks. The deposit score
# weighs the last 7 days (its daily profile jumps there), so it is tested as a 7-day column.
NOT_RECOMPUTABLE = {"sessions_l7d": ("activity", 7), "sessions_l30d": ("activity", 30), "engagement_score": ("activity", 30),
                    "games_breadth_l30d": ("activity", 30), "tp_night_index": ("activity", 30),
                    "deposit_frequency_score": ("payments", 7)}
PROFILE_BEFORE, PROFILE_AHEAD = 30, 14  # days before and after the snapshot in the daily profile
# Point in time. 7-day columns: the day where the daily profile drops most (7 days before minus 7 days
# after) is the end of the window, at most EDGE_SLACK days after the date (honest columns: 0 to 2; a
# leak of 3 days: 3 to 4). 30-day columns have no sharp edge (habit keeps the correlation after the
# date), so they are tested on whole windows instead.
EDGE_SLACK = 2
EDGE_RULE = "its window ends on its date (no later data)"
WINDOW_RULE = "follows its own 30 days more than the 30 days ending 7 later (no later data)"
WHOLE_SNAPSHOT = ("filled", EDGE_RULE, WINDOW_RULE)  # rules that hold or not for the whole snapshot
# Why a check fails, from the investigation in notebook 08: (column, start of the check) -> reason.
ACTIVE_DAYS_WHY = ("Higher than the bet days, never lower: it counts some days with no bet (not the days with a payment). "
                   "Definition to confirm.")
WHY = {
    **{(f"{prior}active_days_l{w}d", "recomputed"): ACTIVE_DAYS_WHY for prior in ("", "prior_") for w in (7, 30, 90)},
    ("days_since_bet", "recomputed"): "0 for almost every player in the backfilled months (never filled); from August 2026 it is the "
                                      "recency 7 days after its date (built with later data).",
    ("tenure_days", ">="): "Shorter than the activity history in the backfilled months.",
    ("tenure_days", "grows"): "Does not grow day by day in the backfilled months; jumps on 2026-08-01, correct after that.",
    ("tp_night_index", "filled"): "Not computed before August 2026 (not backfilled).",
    ("tp_night_index", "between"): "Reaches values above 1 once filled: not a share of play. Definition to confirm.",
    ("deposit_frequency_score", "filled"): "The payments table has almost no deposits before 2026.",
}


def snapshot_days(brand_id: int, last_day: dt.date | None = None) -> list[dt.date]:
    """The month-start snapshots with 90 days of history before them: the cached ones, or with `last_day`
    every month start up to it (the missing ones are then built from S3)."""
    first = dt.date(2025, 4, 1) + dt.timedelta(days=90)
    if last_day is not None:
        return [d.date() for d in pd.date_range(first, last_day, freq="MS")]
    cached = sorted(dt.date.fromisoformat(p.stem) for p in signal_snapshots.cache_dir(brand_id).glob("*.parquet"))
    return [d for d in cached if d.day == 1 and d >= first]


def load(brand_id: int) -> dict[str, pd.DataFrame]:
    out = {}
    for cache in ("activity", "financial", "payments"):
        frame = daily_cache.read(cache, brand_id)
        frame["day"] = pd.to_datetime(frame["day"])
        out[cache] = frame
    return out


def _recompute(data: dict, column: str, day: pd.Timestamp, shift: int = 0) -> pd.Series:
    """The recomputed value of `column` per player as of `day` + `shift` days."""
    cache, source, how, window = RECOMPUTABLE[column]
    frame, end = data[cache], day + pd.Timedelta(shift, unit="D")
    if how == "recency":
        return (end - frame[frame["day"] <= end].groupby(KEY)["day"].max()).dt.days
    lo, hi = (2 * window, window) if how.endswith("_prior") else (window, 0)
    part = frame[(frame["day"] > end - pd.Timedelta(lo, unit="D")) & (frame["day"] <= end - pd.Timedelta(hi, unit="D"))]
    if how.startswith("days"):
        return part.groupby(KEY)["day"].nunique()
    if how == "avg_bet":
        g = part.groupby(KEY)[["turnover_eur", "bets"]].sum()
        return g["turnover_eur"] / g["bets"].where(g["bets"] > 0)
    return part.groupby(KEY)[source].sum()


def _compare(table: pd.Series, mine: pd.Series) -> tuple[float, float, float]:
    """(share matching, share where the table is higher, share where it is lower)."""
    both = pd.concat([table.astype(float), mine.reindex(table.index).fillna(0).astype(float)], axis=1).dropna()
    t, m = both.iloc[:, 0], both.iloc[:, 1]
    close = np.isclose(t, m, rtol=TOLERANCE, atol=ATOL)
    return float(close.mean()), float(((t > m) & ~close).mean()), float(((t < m) & ~close).mean())


def _match(table: pd.Series, mine: pd.Series) -> float:
    return _compare(table, mine)[0]


def recomputation_checks(brand_id: int, days: list[dt.date], data: dict) -> pd.DataFrame:
    """One row per (day, column): match share with its own window and with the window 7 days ahead."""
    snaps = signal_snapshots.snapshots(days, brand_id, build_missing=False)
    last_day = data["activity"]["day"].max()
    rows = []
    for day in days:
        ts = pd.Timestamp(day)
        act = data["activity"]
        population = act[(act["day"] > ts - pd.Timedelta(30, unit="D")) & (act["day"] <= ts)][KEY].drop_duplicates()
        snap = snaps[snaps["snapshot_date"] == day].merge(population, on=KEY).set_index(KEY)
        future_known = ts + pd.Timedelta(SHIFT_DAYS, unit="D") <= last_day
        for column in RECOMPUTABLE:
            past, higher, lower = _compare(snap[column], _recompute(data, column, ts))
            ahead = _match(snap[column], _recompute(data, column, ts, SHIFT_DAYS)) if future_known else np.nan
            rows.append({"day": day, "column": column, "players": len(snap), "table_zero": float((snap[column].astype(float) == 0).mean()),
                         "match": past, "table_higher": higher,
                         "table_lower": lower, "match_7d_ahead": ahead})
    out = pd.DataFrame(rows)
    out["period"] = np.where(pd.to_datetime(out["day"]).dt.date < BACKFILL_DATE, "backfill", "live")
    return out


def plausibility_checks(brand_id: int, days: list[dt.date], data: dict) -> pd.DataFrame:
    """Columns that cannot be rebuilt from the daily tables: what must hold for each, share of players it holds."""
    snaps = signal_snapshots.snapshots(days, brand_id, build_missing=False)
    act = data["activity"]
    rows = []
    for day in days:
        ts = pd.Timestamp(day)
        population = act[(act["day"] > ts - pd.Timedelta(30, unit="D")) & (act["day"] <= ts)][KEY].drop_duplicates()
        s = snaps[snaps["snapshot_date"] == day].merge(population, on=KEY)
        w = ENGAGEMENT_WEIGHTS
        formula = np.clip(w[0] * s["active_days_l30d"] / 30 * 100 + w[1] * np.minimum(1, s["sessions_l30d"] / 60) * 100
                          + w[2] * np.minimum(1, s["play_secs_l30d"] / 108000) * 100, 0, 100)
        checks = {
            ("sessions_l7d", "sessions >= active days (7d)"): s["sessions_l7d"] >= s["active_days_l7d"],
            ("sessions_l30d", "sessions >= active days (30d)"): s["sessions_l30d"] >= s["active_days_l30d"],
            ("engagement_score", "= documented formula on the table's own columns (within 1 point)"):
                (s["engagement_score"] - formula).abs() <= 1,
            ("engagement_score", "between 0 and 100"): s["engagement_score"].between(0, 100),
            ("games_breadth_l30d", ">= 1 game when the player was active in 30 days"):
                (s["games_breadth_l30d"] >= 1) | (s["active_days_l30d"] == 0),
            ("tp_night_index", "between 0 and 1 (a share of play)"): s["tp_night_index"].between(0, 1),
            ("deposit_frequency_score", "between 0 and 100"): s["deposit_frequency_score"].between(0, 100),
            ("tenure_days", ">= days since the first bet seen in the activity"):
                s["tenure_days"] >= (ts - act[act["day"] <= ts].groupby(KEY)["day"].min()).dt.days.reindex(
                    pd.MultiIndex.from_frame(s[KEY])).to_numpy() - 1,
        }
        for (column, rule), ok in checks.items():
            rows.append({"day": day, "column": column, "rule": rule, "holds": float(np.mean(ok))})
        # Filled: a column that is 0 for every player on a day was not computed for that day.
        for column in NOT_RECOMPUTABLE:
            rows.append({"day": day, "column": column, "rule": "filled (not 0 for every player)",
                         "holds": float((s[column].astype(float) != 0).any())})
        # 30-day columns: more correlated with the bet days of their own window than of the window 7 days later.
        if ts + pd.Timedelta(SHIFT_DAYS, unit="D") <= act["day"].max():
            idx = pd.MultiIndex.from_frame(s[KEY])
            bet_days = lambda end: act[(act["day"] > end - pd.Timedelta(30, unit="D")) & (act["day"] <= end)].groupby(KEY)["day"].nunique()
            own, later = (pd.Series(bet_days(ts + pd.Timedelta(k, unit="D")).reindex(idx).fillna(0).to_numpy()) for k in (0, SHIFT_DAYS))
            for column, (cache, window) in NOT_RECOMPUTABLE.items():
                x = pd.Series(s[column].astype(float).to_numpy())
                if window == 30 and x.nunique() > 1:
                    rows.append({"day": day, "column": column, "rule": WINDOW_RULE,
                                 "holds": float(x.corr(own, method="spearman") >= x.corr(later, method="spearman"))})
    return pd.DataFrame(rows)


def day_profiles(brand_id: int, days: list[dt.date], data: dict) -> pd.DataFrame:
    """The daily profile of the columns that cannot be recomputed: the rank correlation of each column with
    "bet (or deposited) on day d", from 30 days before the snapshot to 14 days after it. The correlation
    drops where the column's window ends."""
    snaps = signal_snapshots.snapshots(days, brand_id, build_missing=False)
    act = data["activity"]
    frames = {"activity": act, "payments": data["payments"][data["payments"]["deposit_count"] > 0]}
    rows = []
    for day in days:
        ts = pd.Timestamp(day)
        if ts + pd.Timedelta(PROFILE_AHEAD, unit="D") > act["day"].max():
            continue
        population = act[(act["day"] > ts - pd.Timedelta(30, unit="D")) & (act["day"] <= ts)][KEY].drop_duplicates()
        s = snaps[snaps["snapshot_date"] == day].merge(population, on=KEY)
        idx = pd.MultiIndex.from_frame(s[KEY])
        for column, (cache, window) in NOT_RECOMPUTABLE.items():
            x = s[column].astype(float)
            if x.nunique() < 2:
                continue
            ranks = x.rank().to_numpy()
            frame = frames[cache]
            for offset in range(1 - PROFILE_BEFORE, PROFILE_AHEAD + 1):
                on_day = frame[frame["day"] == ts + pd.Timedelta(offset, unit="D")]
                flag = idx.isin(pd.MultiIndex.from_frame(on_day[KEY])).astype(float)
                rows.append({"day": day, "column": column, "offset": offset,
                             "corr": float(np.corrcoef(ranks, flag)[0, 1]) if 0 < flag.mean() < 1 else np.nan})
    return pd.DataFrame(rows)


def window_end(profile: pd.Series) -> int:
    """The day (relative to the snapshot) where a daily profile drops most: mean of the 7 days up to it
    minus the mean of the 7 days after it."""
    profile = profile.sort_index()
    drops = {t: profile.loc[t - 6:t].mean() - profile.loc[t + 1:t + 7].mean()
             for t in profile.index if t - 6 >= profile.index.min() and t + 7 <= profile.index.max()}
    return int(max(drops, key=drops.get))


def edge_rule(profiles: pd.DataFrame) -> pd.DataFrame:
    """Plausibility rows for the 7-day columns: holds (1) when the window ends at most EDGE_SLACK days after the date."""
    short = [c for c, (_, window) in NOT_RECOMPUTABLE.items() if window == 7]
    ends = profiles[profiles["column"].isin(short)].groupby(["column", "day"]).apply(
        lambda g: window_end(g.set_index("offset")["corr"]))
    return pd.DataFrame({"column": ends.index.get_level_values("column"), "day": ends.index.get_level_values("day"),
                         "rule": EDGE_RULE, "holds": (ends <= EDGE_SLACK).astype(float).to_numpy(), "window_end": ends.to_numpy()})


def tenure_growth(brand_id: int, days: list[dt.date]) -> pd.DataFrame:
    """Between two consecutive snapshots, tenure_days must grow by the days in between."""
    snaps = signal_snapshots.snapshots(days, brand_id, build_missing=False)
    rows = []
    for prev, day in zip(days, days[1:]):
        both = snaps[snaps["snapshot_date"] == prev].merge(snaps[snaps["snapshot_date"] == day], on=KEY, suffixes=("_a", "_b"))
        growth = both["tenure_days_b"].astype(float) - both["tenure_days_a"].astype(float)
        rows.append({"day": day, "players": len(both), "expected_growth": (day - prev).days,
                     "median_growth": float(growth.median()),
                     "share_growing_as_expected": float(((growth - (day - prev).days).abs() <= 1).mean())})
    return pd.DataFrame(rows)


def recency_detail(brand_id: int, data: dict, day: dt.date) -> pd.DataFrame:
    """days_since_bet against the recency recomputed as of the day, and as of 7 days later."""
    ts = pd.Timestamp(day)
    act = data["activity"]
    population = act[(act["day"] > ts - pd.Timedelta(30, unit="D")) & (act["day"] <= ts)][KEY].drop_duplicates()
    s = signal_snapshots.snapshots([day], brand_id, build_missing=False).merge(population, on=KEY).set_index(KEY)
    out = pd.DataFrame({"table": s["days_since_bet"].astype(float)})
    out["recomputed"] = _recompute(data, "days_since_bet", ts).reindex(out.index)
    out["recomputed_7d_later"] = _recompute(data, "days_since_bet", ts, SHIFT_DAYS).reindex(out.index)
    return out


def _segments(days, status, failed_only: bool = False) -> str:
    """The runs of a per-snapshot status, by month: 'mismatch 2025-07 to 2026-07; leak 2026-08 to 2026-10'."""
    days, status, out, start = [str(d)[:7] for d in days], list(status), [], 0
    for i in range(1, len(status) + 1):
        if i == len(status) or status[i] != status[start]:
            if not (failed_only and status[start] == "OK"):
                out.append(f"{status[start]} {days[start]}" + (f" to {days[i - 1]}" if i - 1 > start else ""))
            start = i
    return "; ".join(out)


def summary(recomputed: pd.DataFrame, plausible: pd.DataFrame, growth: pd.DataFrame) -> pd.DataFrame:
    """One verdict per column: OK, BROKEN, LEAK (built with later data) or a failed plausibility rule."""
    rows = []
    for column, g in recomputed.groupby("column", sort=False):
        live, back = g[g["period"] == "live"], g[g["period"] == "backfill"]
        leak_days = int((g["match_7d_ahead"] > g["match"] + 0.05).sum())
        ok_days = int((g["match"] >= MIN_MATCH).sum())
        verdict = ("LEAK" if leak_days else "OK" if ok_days == len(g) else "BROKEN" if ok_days == 0
                   else f"MOSTLY OK ({ok_days}/{len(g)} days >= 95%)")
        # Per snapshot: OK, LEAK or WRONG; the last snapshots cannot be tested for a leak and keep the status before.
        ok = g["match"] >= MIN_MATCH
        status = pd.Series(np.where(ok, "OK", np.where(g["match_7d_ahead"] > g["match"] + 0.05, "leak", "mismatch")))
        status = status.where(ok.to_numpy() | g["match_7d_ahead"].notna().to_numpy()).ffill().fillna("mismatch")
        rows.append({"column": column, "check": "recomputed from the daily tables", "days": len(g),
                     "median_match": g["match"].median(), "worst_match": g["match"].min(),
                     "backfill_median": back["match"].median() if len(back) else np.nan,
                     "live_median": live["match"].median() if len(live) else np.nan,
                     "table_higher": g["table_higher"].median(), "table_lower": g["table_lower"].median(),
                     "days_matching_7d_ahead_better": leak_days, "verdict": verdict, "when": _segments(g["day"], status),
                     "fails_on": _segments(g["day"], status, failed_only=True),
                     "failing_median": g.loc[~ok.to_numpy(), "match"].median(), "used_by": USED_BY.get(column, "not used")})
    for (column, rule), g in plausible.groupby(["column", "rule"], sort=False):
        g = g.dropna(subset=["holds"])
        holds = g["holds"] >= (1 if rule.startswith(WHOLE_SNAPSHOT) else MIN_MATCH)
        label = "empty" if rule.startswith("filled") else "later data" if rule in (EDGE_RULE, WINDOW_RULE) else "fails"
        rows.append({"column": column, "check": rule, "days": len(g), "median_match": g["holds"].median(),
                     "worst_match": g["holds"].min(), "verdict": "OK" if holds.all() else "FAILS",
                     "when": _segments(g["day"], np.where(holds, "OK", label)),
                     "fails_on": _segments(g["day"], np.where(holds, "OK", label), failed_only=True),
                     "failing_median": g.loc[~holds, "holds"].median(),
                     "used_by": USED_BY.get(column, "not used")})
    rows.append({"column": "tenure_days", "check": "grows by the days between two snapshots", "days": len(growth),
                 "median_match": growth["share_growing_as_expected"].median(),
                 "worst_match": growth["share_growing_as_expected"].min(),
                 "verdict": "OK" if growth["share_growing_as_expected"].min() >= MIN_MATCH else "BROKEN",
                 "when": _segments(growth["day"], np.where(growth["share_growing_as_expected"] >= MIN_MATCH, "OK", "fails")),
                 "fails_on": _segments(growth["day"], np.where(growth["share_growing_as_expected"] >= MIN_MATCH, "OK", "fails"),
                                       failed_only=True),
                 "failing_median": growth.loc[growth["share_growing_as_expected"] < MIN_MATCH, "share_growing_as_expected"].median(),
                 "used_by": USED_BY["tenure_days"]})
    return pd.DataFrame(rows)


def figures(recomputed, plausible, profiles, growth, recency, recency_day, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / f"{k}.png" for k in ("match_heatmap", "point_in_time", "recency", "tenure", "plausibility", "day_profile")}
    pivot = recomputed.pivot(index="column", columns="day", values="match")
    fig, ax = plt.subplots(figsize=(13, 0.38 * len(pivot) + 1.5))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), [str(d)[:7] for d in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.0%}", ha="center", va="center", fontsize=6.5)
    ax.axvline(sum(pd.Index(pivot.columns) < BACKFILL_DATE) - 0.5, color="black", lw=1.5, ls="--")
    ax.set_title("Share of players whose value matches the recomputation from the daily tables (dashed: end of the backfill)")
    fig.colorbar(im, ax=ax, fraction=0.02); fig.tight_layout(); fig.savefig(paths["match_heatmap"], dpi=110); plt.close(fig)

    g = recomputed.groupby("column")[["match", "match_7d_ahead"]].median().sort_values("match")
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(g) + 1.5))
    y = np.arange(len(g))
    ax.barh(y - 0.2, g["match"], height=0.4, color="#1e6b57", label="own window (as of the date)")
    ax.barh(y + 0.2, g["match_7d_ahead"], height=0.4, color="#c9a24a", label="window 7 days later")
    ax.set_yticks(y, g.index); ax.set_xlim(0, 1); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Point in time: a column built with later data matches the later window better")
    fig.tight_layout(); fig.savefig(paths["point_in_time"], dpi=110); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    clip = lambda s: s.clip(upper=60)
    axes[0].hist([clip(recency["table"]), clip(recency["recomputed"])], bins=30, label=["signals table days_since_bet", "recomputed"],
                 color=["#c9a24a", "#1e6b57"])
    axes[0].set_title(f"Recency on {recency_day}, capped at 60 days"); axes[0].legend(fontsize=8)
    diff = (recency["table"] - recency["recomputed"]).clip(-30, 30)
    axes[1].hist(diff, bins=61, color="#c9a24a"); axes[1].axvline(0, color="black", lw=1)
    axes[1].set_title("days_since_bet minus the recomputed recency")
    fig.tight_layout(); fig.savefig(paths["recency"], dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(growth["day"].astype(str), growth["share_growing_as_expected"], color=np.where(growth["share_growing_as_expected"] >= MIN_MATCH, "#1e6b57", "#b0413e"))
    ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=45)
    ax.set_title("tenure_days: share of players whose tenure grew by the days between two snapshots")
    fig.tight_layout(); fig.savefig(paths["tenure"], dpi=110); plt.close(fig)

    p = plausible[~plausible["rule"].str.startswith(WHOLE_SNAPSHOT)].groupby(["column", "rule"])["holds"].median().sort_values()
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(p) + 1.5))
    ax.barh([f"{c}: {r}" for c, r in p.index], p.to_numpy(), color=np.where(p.to_numpy() >= MIN_MATCH, "#1e6b57", "#b0413e"))
    ax.set_xlim(0, 1); ax.axvline(MIN_MATCH, color="black", ls="--", lw=1)
    ax.set_title("Columns that cannot be recomputed: share of players for which each rule holds (median over days)")
    fig.tight_layout(); fig.savefig(paths["plausibility"], dpi=110); plt.close(fig)

    m = profiles.groupby(["column", "offset"])["corr"].median().unstack(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    for column in m.columns:
        ax.plot(m.index, m[column], marker=".", lw=1.2, label=column)
    ax.axvline(0, color="black", lw=1); ax.axvspan(0.5, m.index.max() + 0.5, color="#b0413e", alpha=0.07)
    ax.set_xlabel("day relative to the snapshot date (shaded: after the date)"); ax.set_ylabel("rank correlation")
    ax.set_title("Point in time: correlation of each column with activity on each day (median over snapshots)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(paths["day_profile"], dpi=110); plt.close(fig)
    return paths


def validate(brand_id: int, build_missing: bool = False) -> dict:
    """Every check for one brand; saves its tables and figures in OUTPUT_DIR."""
    data = load(brand_id)
    last_day = data["activity"]["day"].max()
    days = snapshot_days(brand_id, last_day.date() if build_missing else None)
    if build_missing:
        signal_snapshots.snapshots(days, brand_id, build_missing=True)
    recomputed = recomputation_checks(brand_id, days, data)
    profiles = day_profiles(brand_id, days, data)
    plausible = pd.concat([plausibility_checks(brand_id, days, data), edge_rule(profiles)], ignore_index=True)
    growth = tenure_growth(brand_id, days)
    recency_day = max(d for d in days if pd.Timestamp(d) + pd.Timedelta(SHIFT_DAYS, unit="D") <= last_day)
    recency = recency_detail(brand_id, data, recency_day)
    table = summary(recomputed, plausible, growth)
    out_dir = Path(str(OUTPUT_DIR).format(brand=brand_id))
    figures(recomputed, plausible, profiles, growth, recency, recency_day, out_dir)
    for name, frame in (("recomputation", recomputed), ("plausibility", plausible), ("day_profiles", profiles),
                        ("tenure_growth", growth), ("summary", table)):
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"brand {brand_id}: {len(days)} snapshots ({days[0]} to {days[-1]}), {int((table['verdict'] != 'OK').sum())} check(s) not OK")
    return {"days": days, "recomputed": recomputed, "plausible": plausible, "growth": growth, "table": table}


def model_features(brands) -> dict[int, set[str]]:
    """The features of each brand's latest optimised LightGBM run in MLflow (empty when there is none)."""
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
    import train  # noqa: E402  (heavy: only for the report)
    out = {}
    for brand in brands:
        try:
            out[brand] = set(train.run_info(train.latest_run("lightgbm_classifier", str(brand))["run_id"])["features"])
        except LookupError:
            out[brand] = set()
    return out


def _daily(result: dict) -> pd.DataFrame:
    """Per (column, check, day): the share that passes (players matching or holding, or 1/0 for whole-snapshot rules)."""
    r = result["recomputed"].assign(check="recomputed from the daily tables", value=lambda d: d["match"])
    p = result["plausible"].rename(columns={"rule": "check", "holds": "value"})
    g = result["growth"].assign(column="tenure_days", check="grows by the days between two snapshots",
                                value=lambda d: d["share_growing_as_expected"])
    return pd.concat([x[["column", "check", "day", "value"]] for x in (r, p, g)], ignore_index=True)


def report_figures(results: dict[int, dict], failed: pd.DataFrame) -> dict[str, Path]:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    paths = {"failed_checks": SUMMARY_DIR / "failed_checks.png", "recency": SUMMARY_DIR / "days_since_bet_by_brand.png"}
    keys = list(failed[["column", "check"]].itertuples(index=False, name=None))
    # Up to PER_ROW brands per row, so the figure stays readable at page width.
    n_rows = -(-len(results) // PER_ROW)
    n_days = max(len(r["days"]) for r in results.values())
    fig, axes = plt.subplots(n_rows, PER_ROW, figsize=(PER_ROW * 0.42 * n_days + 4, n_rows * (0.45 * len(keys) + 1.6)),
                             squeeze=False)
    need = np.array([1.0 if k.startswith(WHOLE_SNAPSHOT) else MIN_MATCH for _, k in keys])[:, None]
    colours = matplotlib.colors.ListedColormap(["#d9867f", "#7fbf9f"])  # fails, passes
    for i, ax in enumerate(axes.flat):
        if i >= len(results):
            ax.axis("off")
            continue
        brand, result = list(results.items())[i]
        daily = _daily(result).set_index(["column", "check", "day"])["value"]
        grid = np.array([[daily.get((c, k, d), np.nan) for d in result["days"]] for c, k in keys])
        ax.imshow(np.where(np.isnan(grid), np.nan, (grid >= need - 1e-9).astype(float)), cmap=colours, vmin=0, vmax=1, aspect="auto")
        for r, c in np.argwhere(~np.isnan(grid)):
            ax.text(c, r, f"{grid[r, c]:.0%}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(result["days"])), [str(d)[:7] for d in result["days"]], rotation=90, fontsize=7)
        ax.set_yticks(range(len(keys)), [f"{c}: {k[:40]}" for c, k in keys] if i % PER_ROW == 0 else [], fontsize=8)
        ax.set_title(f"Brand {brand}")
    fig.suptitle("Failed checks by snapshot: share of players that pass (green: passes, red: fails; "
                 "rules on the whole snapshot are 0% or 100%)", fontsize=10)
    fig.tight_layout(); fig.savefig(paths["failed_checks"], dpi=150); plt.close(fig)

    fig, axes = plt.subplots(n_rows, PER_ROW, figsize=(5 * PER_ROW, 3.6 * n_rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= len(results):
            ax.axis("off")
            continue
        brand, result = list(results.items())[i]
        d = result["recomputed"][result["recomputed"]["column"] == "days_since_bet"]
        x = d["day"].astype(str).str[:7]
        ax.plot(x, d["match"], marker="o", color="#1e6b57", label="= recency as of the date")
        ax.plot(x, d["match_7d_ahead"], marker="o", color="#c9a24a", label="= recency 7 days later")
        ax.plot(x, d["table_zero"], marker=".", ls="--", color="#888888", label="= 0")
        ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=90, labelsize=7); ax.set_title(f"Brand {brand}")
        if i % PER_ROW == 0:
            ax.set_ylabel("share of players")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("days_since_bet: what it equals, by snapshot", fontsize=10)
    fig.tight_layout(); fig.savefig(paths["recency"], dpi=150); plt.close(fig)
    return paths


# The gold column behind each cached column (daily_cache.CACHES), for the appendix.
GOLD_COLUMN = {"bets": "bets", "turnover_eur": "turnover_eur", "ggr_eur": "ggr_eur",
               "wagered_bonus_eur": "wagered_bonus_eur_today", "deposit_eur": "deposit_amount_eur",
               "withdraw_eur": "withdraw_amount_eur"}


def definitions() -> list[dict]:
    """How each recomputable column is rebuilt, grouped by calculation: one row per (table, value)."""
    value = {"days": "distinct days with `bets` > 0", "days_prior": "distinct days with `bets` > 0",
             "recency": "D minus the last day with `bets` > 0, up to D",
             "avg_bet": "sum of `turnover_eur` / sum of `bets`"}
    groups = {}
    for column, (cache, source, how, window) in RECOMPUTABLE.items():
        what = value.get(how, f"sum of `{GOLD_COLUMN.get(source, source)}`")
        span = "all history up to D" if window is None else (f"(D-{2 * window}, D-{window}]" if how.endswith("_prior")
                                                             else f"(D-{window}, D]")
        groups.setdefault((daily_cache.CACHES[cache]["table"], what), []).append((column, span))
    return [{"Columns": ", ".join(f"`{c}`" for c, _ in cols), "Gold table": f"`{table}`", "Value": what,
             "Window": ", ".join(dict.fromkeys(w for _, w in cols))} for (table, what), cols in groups.items()]


def examples(results: dict[int, dict], failed: pd.DataFrame) -> list[dict]:
    """One player per failing recomputed column (two for days_since_bet: one per regime), in the brand and
    snapshot where it fails most, plus tenure_days and tp_night_index: the table's value next to the recomputed one."""
    data, rows = {}, []
    recomputed = pd.concat([r["recomputed"].assign(brand=b) for b, r in results.items()], ignore_index=True)

    def snapshot(brand, day):
        if brand not in data:
            data[brand] = load(brand)
        act, ts = data[brand]["activity"], pd.Timestamp(day)
        population = act[(act["day"] > ts - pd.Timedelta(30, unit="D")) & (act["day"] <= ts)][KEY].drop_duplicates()
        return signal_snapshots.snapshots([day], brand, build_missing=False).merge(population, on=KEY).set_index(KEY), ts

    def add(column, brand, day, table, mine, later=None):
        if table.empty:
            return
        rows.append({"Column": f"`{column}`", "Brand": brand, "Snapshot": str(day), "tenant_id": table.index[0][0],
                     "player_id": table.index[0][1], "Table": f"{round(table.iloc[0], 2):g}",
                     "Recomputed": "" if mine is None else f"{round(mine, 2):g}",
                     "Recomputed 7 days later": "" if later is None else f"{round(later, 2):g}"})

    failing = set(failed["column"])
    for column in [c for c in RECOMPUTABLE if c in failing]:
        g = recomputed[recomputed["column"] == column]
        picks = ([g[g["day"] < LIVE_FROM].nsmallest(1, "match"), g[g["day"] >= LIVE_FROM].nsmallest(1, "match")]
                 if column == "days_since_bet" else [g.nsmallest(1, "match")])
        for pick in picks:
            if pick.empty:
                continue
            brand, day = int(pick["brand"].iloc[0]), pick["day"].iloc[0]
            snap, ts = snapshot(brand, day)
            table = snap[column].astype(float)
            mine = _recompute(data[brand], column, ts).reindex(table.index).fillna(0)
            off = table[~np.isclose(table, mine, rtol=TOLERANCE, atol=ATOL)].sort_index().head(1)
            later = (_recompute(data[brand], column, ts, SHIFT_DAYS).reindex(off.index).iloc[0]
                     if column == "days_since_bet" and not off.empty else None)
            if not off.empty:
                add(column, brand, day, off, mine.reindex(off.index).iloc[0], later)
    if "tenure_days" in failing:
        g = pd.concat([r["plausible"].assign(brand=b) for b, r in results.items()])
        pick = g[(g["column"] == "tenure_days") & g["rule"].str.startswith(">=")].nsmallest(1, "holds")
        brand, day = int(pick["brand"].iloc[0]), pick["day"].iloc[0]
        snap, ts = snapshot(brand, day)
        act = data[brand]["activity"]
        first = (ts - act[act["day"] <= ts].groupby(KEY)["day"].min()).dt.days.reindex(snap.index)
        table = snap["tenure_days"].astype(float)
        off = table[table < first - 1].sort_index().head(1)
        if not off.empty:
            add("tenure_days (vs days since the first bet)", brand, day, off, first.reindex(off.index).iloc[0])
    if "tp_night_index" in failing:
        brand = BRANDS[0] if BRANDS[0] in results else next(iter(results))
        day = max(results[brand]["days"])
        snap, _ = snapshot(brand, day)
        night = snap["tp_night_index"].astype(float)
        add("tp_night_index (above 1)", brand, day, night[night > 1].sort_index().head(1), None)
    return rows


def _md(rows: list[dict]) -> str:
    cols = list(rows[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows])


def report(results: dict[int, dict], features: dict[int, set[str]]) -> Path:
    brands = list(results)
    tables = pd.concat([r["table"].assign(brand=b) for b, r in results.items()], ignore_index=True)
    failed = tables[tables["verdict"] != "OK"][["column", "check"]].drop_duplicates().sort_values("column", kind="stable")
    paths = report_figures(results, failed)
    rows = []
    for column, check in failed.itertuples(index=False, name=None):
        row = {"Column": f"`{column}`", "Check that fails": check}
        for b in brands:
            t = results[b]["table"]
            f = t[(t["column"] == column) & (t["check"] == check)]
            if f.empty:
                row[f"Brand {b}"] = "n/a"
            elif f["verdict"].iloc[0] == "OK":
                row[f"Brand {b}"] = "OK"
            else:
                share = "" if check.startswith(WHOLE_SNAPSHOT) else f" ({f['failing_median'].iloc[0]:.0%} pass)"
                row[f"Brand {b}"] = f["fails_on"].iloc[0] + share
        row["Why"] = next((w for (c, start), w in WHY.items() if c == column and check.startswith(start)), "See notebook 08.")
        used = USED_BY.get(column, "not used")
        in_models = [str(b) for b in brands if used in features.get(b, set())]
        row["Impact on the models"] = (f"None: {used}" if used.startswith("replaced") else
                                       "None: no feature reads it" if used == "not used" else
                                       f"`{used}` is in the model of brand{'s' if len(in_models) > 1 else ''} {', '.join(in_models)}" if in_models
                                       else f"`{used}` is in no current model")
        rows.append(row)
    columns = sorted(set(tables["column"]))
    bad = sorted(set(failed["column"]))
    in_a_model = sorted({c for c in bad if not USED_BY.get(c, "").startswith("replaced")
                         and any(USED_BY.get(c) in features.get(b, set()) for b in brands)})
    rel = lambda p: os.path.relpath(p, DOC_PATH.parent)
    span = "; ".join(f"brand {b}: {r['days'][0]:%Y-%m} to {r['days'][-1]:%Y-%m}" for b, r in results.items())
    method = [
        {"Check": "Recomputation", "Columns": f"{len(RECOMPUTABLE)} sums and counts (active days, bets, wagered, GGR, deposits, recency)",
         "How": "Rebuild each player's value from the daily tables (activity, financial, payments) with the window of its name",
         "Passes when": f"{MIN_MATCH:.0%} of the players within {TOLERANCE:.0%} (or {ATOL}) on every snapshot"},
        {"Check": "Later data (leak)", "Columns": "The same",
         "How": f"The same recomputation as of {SHIFT_DAYS} days after the date",
         "Passes when": "The table does not match the later value better (by 5 points)"},
        {"Check": "Plausibility", "Columns": f"{len(NOT_RECOMPUTABLE)} that cannot be rebuilt (sessions, games, night play, scores)",
         "How": "Valid ranges, sessions >= active days, `engagement_score` = its documented formula",
         "Passes when": f"{MIN_MATCH:.0%} of the players on every snapshot"},
        {"Check": "Filled", "Columns": "The same", "How": "Not 0 for every player", "Passes when": "Every snapshot"},
        {"Check": "Later data, 7-day columns", "Columns": "`sessions_l7d`, `deposit_frequency_score`",
         "How": "Rank correlation with \"bet (deposit) on day d\", day by day: it drops where the window ends",
         "Passes when": f"The window ends at most {EDGE_SLACK} days after the date, on every snapshot"},
        {"Check": "Later data, 30-day columns", "Columns": "Sessions, engagement, games, night play (30 days)",
         "How": "Rank correlation with the bet days of its own 30 days and of the 30 days ending 7 days later",
         "Passes when": "Its own 30 days win, on every snapshot"},
        {"Check": "Tenure", "Columns": "`tenure_days`",
         "How": "Grows by the days between two snapshots; >= the days since the first bet in the activity",
         "Passes when": f"{MIN_MATCH:.0%} of the players on every snapshot"},
    ]
    text = f"""# Signal Validation: gld_player_signals_daily

- **What**: every column of the signals table that the features read, checked against the daily tables, for brands {", ".join(map(str, brands))}.
- **Data**: month-start snapshots ({span}); on each, the players with a bet in the 30 days before. Local files only.


## 1. Result

**{len(columns) - len(bad)} of {len(columns)} columns pass every check in every brand. {len(bad)} fail{"; the models read " + ", ".join(f"`{c}`" for c in in_a_model) if in_a_model else ""}.**

Cells: the months where the check fails and how (mismatch: the table differs from the recomputation; leak: it equals the value of 7 days later; empty: 0 for every player), and the median share of players that pass in those months.

{_md(rows)}

![Failed checks by brand and snapshot]({rel(paths['failed_checks'])})

![days_since_bet by brand]({rel(paths['recency'])})

Pass in every brand: {", ".join(f"`{c}`" for c in columns if c not in bad)}.

## 2. Method

{_md(method)}

## 3. Actions

- **Pipeline**: it never reads `days_since_bet` or `tenure_days`; it recomputes recency (`days_since_last_bet`) and tenure (`days_since_first_bet`) from the daily activity. Tenure is correct since September 2026, but the models train on the earlier months.

## 4. Limitations

- It has only been tested for {NUMBER_WORDS.get(len(brands), len(brands))} brands.

## Appendix: How Each Column Was Recomputed

**A. Definitions.** Snapshot of day D: data up to and including D, for the players with a day with bets > 0 in (D-30, D]. A value matches when it is within {TOLERANCE:.0%} (or {ATOL}) of the recomputed one. Cached daily tables, one row per player and day, read from S3 (`{S3_HINT}`).

{_md(definitions())}

**B. Examples.** One player per failing column, in the brand and snapshot where it fails most.

{_md(examples(results, failed))}
"""
    DOC_PATH.write_text(text, encoding="utf-8")
    print(f"saved {os.path.relpath(DOC_PATH, PROJECT_ROOT)} | {len(bad)} of {len(columns)} column(s) fail a check")
    return DOC_PATH


def saved_results(brands=BRANDS) -> dict[int, dict]:
    """The checks of each brand as the last run saved them (OUTPUT_DIR), without recomputing them."""
    out = {}
    for brand in brands:
        folder = Path(str(OUTPUT_DIR).format(brand=brand))
        load = lambda name: pd.read_csv(folder / f"{name}.csv").assign(day=lambda x: pd.to_datetime(x["day"]).dt.date)
        recomputed, plausible, growth = load("recomputation"), load("plausibility"), load("tenure_growth")
        out[brand] = {"days": sorted(recomputed["day"].unique()), "recomputed": recomputed, "plausible": plausible,
                      "growth": growth, "table": summary(recomputed, plausible, growth)}
    return out


def run(brands=BRANDS, build_missing: bool = False) -> Path:
    results = {b: validate(b, build_missing) for b in brands}
    return report(results, model_features(brands))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the columns of gld_player_signals_daily against the daily tables.")
    parser.add_argument("--brands", type=int, nargs="+", default=list(BRANDS))
    parser.add_argument("--build-missing", action="store_true", help="build the missing month-start snapshots from S3")
    parser.add_argument("--report-only", action="store_true", help="rewrite the report from the saved tables, no recomputation")
    args = parser.parse_args()
    if args.report_only:
        report(saved_results(args.brands), model_features(args.brands))
    else:
        run(args.brands, args.build_missing)
