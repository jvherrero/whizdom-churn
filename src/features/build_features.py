"""Feature store builder on the gold layer (T6): one row per player, as of one cutoff date.

    df = build_feature_store(cutoff_date="2026-08-01", brand_id=64)       # the final feature vector
    df = build_feature_store(cutoff_date="2026-08-01", brand_id="basel")  # every brand
    raw = raw_features(cutoff_date="2026-08-01", brand_id=64)             # every signal, real units

    .venv/bin/python src/features/build_features.py --as-of 2026-08-01 [--brand-id 64|basel]

The features are those of docs/feature_dictionary.md (FEATURES, by family in FEATURE_COLUMNS). The
population of a cutoff is the brand's players with a bet in the 30 days up to it. Every feature uses
data up to and including the cutoff day (UTC days, as in gold) and nothing after it:

- gld_player_signals_daily on the cutoff day, and 7 and 30 days before it (src/features/gold_signals.py);
- the local daily caches (src/features/gold_cache.py): activity (bets), financial (stake and GGR per
  day) and payments (deposits, withdrawals, failed deposits).

Data limits (docs/dq_reports/gold/semantic_checks_brand64.md):
- gold's days_since_bet is broken, so recency is computed from the activity (days_since_last_bet);
- gold's tenure_days is not consistent across days (it does not grow with the calendar), so tenure is
  the days since the first bet seen in the activity, capped at TENURE_CAP_DAYS (days_since_first_bet);
- completed deposits only exist from DEPOSITS_VALID_FROM: a deposit feature is empty (unknown, not 0)
  at a cutoff whose window starts before it.

build_feature_store() caps the monetary features at each brand's p99.5 (configs/winsorisation_features_
brand{id}.yaml), then applies a signed log1p to the counts, amounts, days and ratios in SCALED.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gold_cache  # noqa: E402
import gold_labels  # noqa: E402
import gold_signals  # noqa: E402
from gold import last_closed_day  # noqa: E402
from winsorisation import winsorise  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_START = dt.date(2025, 4, 1)      # first day of the gold daily tables
LABEL_HORIZON_DAYS = 60
LOOKBACK_DAYS = 30                       # population: a bet in the 30 days up to the cutoff
MIN_HISTORY_DAYS = 90                    # the longest gold window the features read (l90d)
MIN_CUTOFF = HISTORY_START + dt.timedelta(days=MIN_HISTORY_DAYS)
HISTORY_DAYS = 180                       # how far back the history-based signals look
DEPOSITS_VALID_FROM = dt.date(2026, 3, 1)
# Tenure cap: every training cutoff has at least this much activity history before it (from 2025-04-01
# to the first cutoff, 2025-09-01, is 153 days), so a capped tenure means the same at every cutoff.
TENURE_CAP_DAYS = 150
ALL_BRANDS = "basel"  # sentinel meaning "every brandId", not a real brand
KEY = ["tenant_id", "player_id"]
ID_COLUMNS = ["cutoff_date", "tenant_id", "brandId", "player_id"]

# docs/feature_dictionary.md, by family.
FEATURE_COLUMNS = {
    "recency": ["days_since_last_bet", "days_since_last_deposit"],
    "frequency": ["active_days_l7d", "active_days_l30d", "active_days_l90d", "bets_l7d", "bets_l30d",
                  "sessions_l7d", "sessions_l30d", "n_deposit_days_7d", "n_deposit_days_30d"],
    "monetary": ["wagered_eur_l7d", "wagered_eur_l30d", "wagered_eur_l90d", "net_loss_7d", "ggr_eur_l30d",
                 "deposits_eur_l30d", "withdrawals_30d_share"],
    "lag": ["prior_wagered_eur_l7d", "prior_active_days_l7d", "prior_ggr_eur_l30d"],
    "trend": ["wagered_trend_7", "heavy_loss_multiple", "deposits_30_vs_prior30"],
    "tenure": ["days_since_first_bet", "prior_dormancy_spells_14d", "days_since_last_return"],
    "mix": ["engagement_score", "games_breadth_30d", "games_breadth_ratio", "bonus_stake_share_30d",
            "losing_streak", "avg_bet_eur_l30d", "night_play_index"],
    "deposit": ["deposited_within_3d", "deposited_within_7d", "deposited_within_14d", "deposit_frequency_score",
                "failed_deposits_14d"],
}
FEATURES = [f for cols in FEATURE_COLUMNS.values() for f in cols]
# Not sign-log scaled: flags, 0-100 scores, shares and the day of the week (already on a small scale).
NOT_SCALED = {"deposited_within_3d", "deposited_within_7d", "deposited_within_14d", "deposit_frequency_score",
              "engagement_score", "bonus_stake_share_30d", "night_play_index", "withdrawals_30d_share"}
SCALED = [f for f in FEATURES if f not in NOT_SCALED]
# Deposit and withdrawal signals and the days of history each needs (empty before DEPOSITS_VALID_FROM).
DEPOSIT_WINDOWS = {
    "n_deposit_days_7d": 7, "n_deposit_days_30d": 30, "deposit_days_ratio_30_vs_prior30": 60,
    "days_since_last_deposit": 30, "deposited_within_3d": 30, "deposited_within_7d": 30, "deposited_within_14d": 30,
    "deposit_size_trend": 120, "deposits_30_vs_prior30": 60, "deposit_frequency_score": 30, "deposit_recency_score": 30,
    "deposits_eur_l30d": 30, "withdrawals_30d_share": 30, "big_win_then_withdrawal": 30,
    "withdrawal_no_redeposit": 14, "failed_deposits_14d": 14,
}


def parse_brand_id(value: str | int) -> int | str:
    """A command-line brand: an integer brandId, or ALL_BRANDS ("basel") for every brand."""
    return ALL_BRANDS if str(value).lower() == ALL_BRANDS else int(value)


def brand_label(brand_ids) -> str:
    """How a set of brands is named in files and MLflow: the brandId when there is one, else "basel"."""
    brands = sorted({int(b) for b in brand_ids})
    return str(brands[0]) if len(brands) == 1 else ALL_BRANDS


def data_available_through() -> dt.date:
    """The last day with data: DATA_AVAILABLE_THROUGH if set, else the last cached activity day,
    else yesterday (UTC)."""
    override = os.environ.get("DATA_AVAILABLE_THROUGH")
    if override:
        return dt.date.fromisoformat(override)
    days = gold_cache.cached_days("activity")
    return days[-1] if days else last_closed_day()


def _ratio(a, b):
    return (a + 1) / (b + 1)


def history_signals(c: dt.date, fin: pd.DataFrame, act: pd.DataFrame, pay: pd.DataFrame | None,
                    players: pd.DataFrame) -> pd.DataFrame:
    """The signals that need the day-by-day history up to cutoff c (days as datetime64)."""
    out = players[KEY].copy().set_index(KEY)
    ts = pd.Timestamp(c)
    f = fin[(fin["day"] > ts - pd.Timedelta(HISTORY_DAYS, unit="D")) & (fin["day"] <= ts)]
    before = lambda df, lo, hi: df[(df["day"] > ts - pd.Timedelta(lo, unit="D")) & (df["day"] <= ts - pd.Timedelta(hi, unit="D"))]
    if pay is None:  # no payments data for these days (the table starts on 2025-09-18): a typed empty frame
        pay = pd.DataFrame({"tenant_id": pd.Series(dtype=object), "player_id": pd.Series(dtype="int32"),
                            "day": pd.Series(dtype="datetime64[us]"),
                            **{c: pd.Series(dtype=float) for c in ("deposit_count", "deposit_eur", "withdraw_count",
                                                                   "withdraw_eur", "failed_deposit_count")}})
    pay = pay[(pay["day"] > ts - pd.Timedelta(HISTORY_DAYS, unit="D")) & (pay["day"] <= ts)]
    dep = pay[pay["deposit_count"] > 0]
    wd = pay[pay["withdraw_count"] > 0]

    # Bet recency, from the activity cache (gold's days_since_bet is broken).
    bets = act[act["day"] <= ts].groupby(KEY)["day"]
    out["days_since_last_bet"] = (ts - bets.max()).dt.days
    # Tenure from the first bet seen, capped (gold's tenure_days is not consistent across days).
    out["days_since_first_bet"] = (ts - bets.min()).dt.days.clip(upper=TENURE_CAP_DAYS)

    out["n_deposit_days_7d"] = before(dep, 7, 0).groupby(KEY).size()
    out["n_deposit_days_30d"] = before(dep, 30, 0).groupby(KEY).size()
    n_prior = before(dep, 60, 30).groupby(KEY).size()
    out[["n_deposit_days_7d", "n_deposit_days_30d"]] = out[["n_deposit_days_7d", "n_deposit_days_30d"]].fillna(0)
    out["deposit_days_ratio_30_vs_prior30"] = _ratio(out["n_deposit_days_30d"], n_prior.reindex(out.index).fillna(0))
    last_dep = dep.groupby(KEY)["day"].max()
    out["days_since_last_deposit"] = ((ts - last_dep).dt.days).reindex(out.index).clip(upper=31).fillna(31)
    for d in (3, 7, 14):
        out[f"deposited_within_{d}d"] = (out["days_since_last_deposit"] < d).astype(int)
    per_day = lambda df: df["deposit_eur"] / df["deposit_count"]
    recent, earlier = before(dep, 30, 0), before(dep, 120, 30)
    out["deposit_size_trend"] = per_day(recent).groupby([recent["tenant_id"], recent["player_id"]]).mean() / \
        per_day(earlier).groupby([earlier["tenant_id"], earlier["player_id"]]).mean()

    # Big win then withdrawal: a day in the last 30 with net win (-GGR) > 5x the median daily stake
    # of the last 90 days, and a withdrawal on that day or within the next 3 (up to the cutoff).
    playing = before(f[f["wagered_eur"] > 0], 90, 0)
    median_stake = playing.groupby(KEY)["wagered_eur"].median().rename("median_stake")
    last30 = before(f, 30, 0).merge(median_stake, on=KEY)
    wins = last30[-last30["ggr_eur"] > 5 * last30["median_stake"]][KEY + ["day"]]
    pair = wins.merge(before(wd, 30, 0)[KEY + ["day"]], on=KEY, suffixes=("_win", "_wd"))
    gap = (pair["day_wd"] - pair["day_win"]).dt.days
    hit = pair[(gap >= 0) & (gap <= 3)].groupby(KEY).size()
    out["big_win_then_withdrawal"] = (hit.reindex(out.index).fillna(0) > 0).astype(int)
    w30 = before(pay, 30, 0).groupby(KEY)[["withdraw_eur", "deposit_eur"]].sum() if len(pay) else None
    out["withdrawals_30d_share"] = (w30["withdraw_eur"] / w30["deposit_eur"].replace(0, np.nan)) if w30 is not None else np.nan

    # Withdrawal without redeposit: the last withdrawal in the last 14 days, no deposit on a later day.
    last_wd = before(wd, 14, 0).groupby(KEY)["day"].max()
    redeposit = last_dep.reindex(last_wd.index) > last_wd
    out["withdrawal_no_redeposit"] = (~redeposit).astype(int).reindex(out.index).fillna(0).astype(int)

    # Losing streak: consecutive latest playing days (last 90) with a net loss for the player (GGR > 0).
    p = playing.sort_values(KEY + ["day"], ascending=[True, True, False])
    broken = (p["ggr_eur"] <= 0).astype(int).groupby([p["tenant_id"], p["player_id"]]).cumsum()
    out["losing_streak"] = p[broken == 0].groupby(KEY).size().reindex(out.index).fillna(0)

    # Prior dormancy: returns after 14+ / 30+ silent days in the last 180 days, from the activity history.
    a = act[act["day"] <= ts].sort_values(KEY + ["day"])
    nxt = a.groupby(KEY)["day"].shift(-1)
    silent = (nxt - a["day"]).dt.days - 1
    returns = a.assign(return_day=nxt, silent=silent).dropna(subset=["return_day"])
    returns = returns[returns["return_day"] > ts - pd.Timedelta(HISTORY_DAYS, unit="D")]
    for n in (14, 30):
        out[f"prior_dormancy_spells_{n}d"] = returns[returns["silent"] >= n].groupby(KEY).size().reindex(out.index).fillna(0)
    last_return = returns[returns["silent"] >= 14].groupby(KEY)["return_day"].max()
    out["days_since_last_return"] = ((ts - last_return).dt.days).reindex(out.index).fillna(HISTORY_DAYS + 1)

    out["failed_deposits_14d"] = before(pay, 14, 0).groupby(KEY)["failed_deposit_count"].sum() \
        .reindex(out.index).fillna(0)
    return out.reset_index()


def snapshot_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Signals from the gold snapshots at the cutoff (no suffix), 7 days before (_p7) and 30 days
    before (_p30)."""
    s = frame
    s["net_loss_7d"] = s["ggr_eur_l7d"]
    weekly90 = s["ggr_eur_l90d"] / 90 * 7
    s["heavy_loss_multiple"] = s["ggr_eur_l7d"] / weekly90.where(weekly90 > 0)
    s["heavy_loss_flag"] = ((s["ggr_eur_l7d"] > 0) & (s["heavy_loss_multiple"] > 2)).astype(int)
    s["deposits_30_vs_prior30"] = s["deposits_eur_l30d"] / s["prior_deposits_eur_l30d"].where(s["prior_deposits_eur_l30d"] > 0)
    s["session_ratio_7_vs_prior7"] = _ratio(s["sessions_l7d"], s["sessions_l7d_p7"].fillna(0))
    length_now = s["play_secs_l7d"] / s["sessions_l7d"].where(s["sessions_l7d"] > 0)
    length_before = s["play_secs_l7d_p7"] / s["sessions_l7d_p7"].where(s["sessions_l7d_p7"] > 0)
    s["session_length_ratio"] = length_now / length_before
    s["active_days_7_vs_prior7"] = _ratio(s["active_days_l7d"], s["prior_active_days_l7d"])
    s["bonus_stake_share_30d"] = s["wagered_bonus_eur_l30d"] / s["wagered_eur_l30d"].where(s["wagered_eur_l30d"] > 0)
    s["bonus_granted_30d"] = s["bonus_granted_eur_l30d"]
    s["games_breadth_30d"] = s["games_breadth_l30d"]
    s["games_breadth_ratio"] = _ratio(s["games_breadth_l30d"], s["games_breadth_l30d_p30"].fillna(0))
    s["wagered_trend_7"] = _ratio(s["wagered_eur_l7d"], s["prior_wagered_eur_l7d"])
    s["night_play_index"] = s["tp_night_index"]
    return s


def _one_brand(c: dt.date, brand: int, players: pd.DataFrame) -> pd.DataFrame:
    """Every signal of one brand's players at cutoff c, in real units."""
    snaps = gold_signals.snapshots([c, c - dt.timedelta(days=7), c - dt.timedelta(days=30)], brand)
    fin = gold_cache.read("financial", brand, start=c - dt.timedelta(days=HISTORY_DAYS), end=c)
    act = gold_cache.read("activity", brand, end=c, columns="tenant_id, player_id, day")
    try:
        pay = gold_cache.read("payments", brand, start=c - dt.timedelta(days=HISTORY_DAYS), end=c)
    except FileNotFoundError:
        pay = None
    for frame in (fin, act, pay):
        if frame is not None:
            frame["day"] = pd.to_datetime(frame["day"])
    keep_prior = ["sessions_l7d", "play_secs_l7d", "games_breadth_l30d"]
    snap = snaps[snaps["snapshot_date"] == c].drop(columns="snapshot_date")
    p7 = snaps[snaps["snapshot_date"] == c - dt.timedelta(days=7)][KEY + keep_prior].add_suffix("_p7")
    p30 = snaps[snaps["snapshot_date"] == c - dt.timedelta(days=30)][KEY + keep_prior].add_suffix("_p30")
    return (players.merge(snap, on=KEY, how="left")
            .merge(p7.rename(columns={"tenant_id_p7": "tenant_id", "player_id_p7": "player_id"}), on=KEY, how="left")
            .merge(p30.rename(columns={"tenant_id_p30": "tenant_id", "player_id_p30": "player_id"}), on=KEY, how="left")
            .merge(history_signals(c, fin, act, pay, players), on=KEY, how="left"))


def raw_features(cutoff_date: str | dt.date, brand_id: int | str) -> pd.DataFrame:
    """Every signal (the features and the ones T4 rejected), one row per player of the population,
    in real units (days, counts, EUR): not winsorised, not scaled."""
    c = pd.to_datetime(cutoff_date).date() if isinstance(cutoff_date, str) else cutoff_date
    if c < MIN_CUTOFF:
        raise ValueError(f"cutoff_date must be {MIN_CUTOFF} or later ({MIN_HISTORY_DAYS} days of gold history)")
    pop = gold_labels.population(c, brand_id, LOOKBACK_DAYS)
    if pop.empty:
        raise ValueError(f"no player of brand_id={brand_id!r} with a bet in the {LOOKBACK_DAYS} days up to {c}")
    parts = [_one_brand(c, int(b), rows[KEY]).assign(brandId=int(b)) for b, rows in pop.groupby("brand_id")]
    frame = snapshot_signals(pd.concat(parts, ignore_index=True))
    numeric = frame.select_dtypes("number").columns
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)  # a ratio over 0: not computable
    for name, days in DEPOSIT_WINDOWS.items():
        if name in frame and c - dt.timedelta(days=days - 1) < DEPOSITS_VALID_FROM:
            frame[name] = np.nan  # deposits unknown before DEPOSITS_VALID_FROM: empty, not 0
    frame = frame.copy()  # one block of memory before adding a column
    frame.insert(0, "cutoff_date", c)
    return frame


def _scale(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [c for c in SCALED if c in df]:
        df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
    return df


def _unscale(df: pd.DataFrame) -> pd.DataFrame:
    """Undo the signed log1p (for a saved, scaled dataset: plots in days / EUR / counts)."""
    df = df.copy()
    for col in [c for c in SCALED if c in df]:
        df[col] = (np.sign(df[col]) * np.expm1(np.abs(df[col]))).round(6)
    return df


def finalise(raw: pd.DataFrame) -> pd.DataFrame:
    """raw_features() rows (any cutoffs) -> the final feature vector (ID_COLUMNS + FEATURES): each
    brand's EUR caps, then the signed log1p."""
    out = _scale(winsorise(raw))[ID_COLUMNS + FEATURES]
    return out.astype({f: "float64" for f in FEATURES} | {"brandId": "int64"})


def build_feature_store(cutoff_date: str | dt.date, brand_id: int | str) -> pd.DataFrame:
    """The final feature vector of one cutoff."""
    return finalise(raw_features(cutoff_date, brand_id))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final feature vector as of one cutoff date.")
    parser.add_argument("--as-of", required=True, help="cutoff date, YYYY-MM-DD")
    parser.add_argument("--brand-id", default="64", help=f"brandId, or '{ALL_BRANDS}' for every brand")
    parser.add_argument("--output", help="parquet path (default data/processed/features_{brand}_{as_of}_{unix_ts}.parquet)")
    args = parser.parse_args()

    start = time.time()
    df = build_feature_store(args.as_of, parse_brand_id(args.brand_id))
    output = Path(args.output) if args.output else (
        PROJECT_ROOT / f"data/processed/features_{args.brand_id}_{args.as_of}_{int(time.time())}.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"saved {os.path.relpath(output.resolve(), PROJECT_ROOT)} ({len(df):,} rows, {df.shape[1]} columns) "
          f"in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
