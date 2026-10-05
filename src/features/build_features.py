"""Feature store builder: one row per player, as of a single cutoff date.

Call signature:

    df_features = build_feature_store(cutoff_date="2026-08-28", brand_id=64)
    df_features = build_feature_store(cutoff_date="2026-08-28", brand_id="basel")  # every brandId

`brand_id="basel"` is this module's sentinel for "run every brandId found under
both operators", not a real brand. The whole pipeline (extraction, null handling,
EUR conversion, sign-log scaling) runs once per real brandId, never pooled across
brands.

"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import boto3
import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fx_rates import convert_amount_to_eur, load_fx_rates_eur

AWS_PROFILE = "javier-whizdom-prod-ds"
BUCKET = "whizdomai-eu-central-1-650177547431-datalake-gmntc-prod"
LANDING_LEVEL = "org/10-landing/kafka-sink"
LANDING_TABLES = {
    "player": "whizdomai-players",
    "bet": "whizdomai-transactions",
    "transaction": "whizdomai-payments",
    "bonus": "whizdomai-bonuses",
}
OPERATORS = ["primus", "secundus"]

HISTORY_START = dt.date(2026, 6, 18)  
DATA_AVAILABLE_THROUGH = dt.date(2026, 9, 29)  
LABEL_HORIZON_DAYS = 60  

ALL_BRANDS = "basel"  # sentinel meaning "every brandId", not a real brand
FX_RATES_CSV = Path(__file__).resolve().parent.parent.parent / "data/01_raw/fx_rates.csv"

FEATURE_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data/02_intermediate/feature_cache"

COUNT_WINDOWS = (7, 14, 30)   
TREND_WINDOWS = (7, 14)       
DORMANCY_GAP_WINDOWS = (7, 14, 30)  


def _s3_duckdb() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with the SSO profile's credentials already loaded."""
    creds = boto3.Session(profile_name=AWS_PROFILE).get_credentials().get_frozen_credentials()
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs;")
    con.sql("SET memory_limit='24GB'")
    con.sql("SET threads=32")

    con.sql("SET max_temp_directory_size='20GB'")
    con.sql(f"""
        CREATE OR REPLACE SECRET s3_secret (
            TYPE S3, KEY_ID '{creds.access_key}', SECRET '{creds.secret_key}',
            SESSION_TOKEN '{creds.token}', REGION 'eu-central-1'
        );
    """)
    return con


def _landing_paths(operator: str, landing_table: str, days: list[dt.date]) -> list[str]:
    return [
        f"s3://{BUCKET}/{LANDING_LEVEL}/{operator}/{landing_table}/{d.year}/{d.month:02d}/{d.day:02d}/*.snappy.parquet"
        for d in days
    ]


def _date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def _feature_cache_path(operator: str, brand_id: int, cutoff: dt.date) -> Path:
    return FEATURE_CACHE_DIR / f"{operator}_{brand_id}_{cutoff}.parquet"


def _resolve_operators_for_brand(brand_id: int, as_of: dt.date) -> list[str]:
    """Which operator(s) a brandId belongs to, checked on `as_of` (one snapshot day,
    same cheap existence check used throughout this project, not a full-history scan)."""
    con = _s3_duckdb()
    found = []
    for operator in OPERATORS:
        paths = _landing_paths(operator, LANDING_TABLES["player"], [as_of])
        n = con.sql(f"""
            SELECT COUNT(*) AS n FROM read_parquet({paths}, union_by_name=True)
            WHERE brandId = {brand_id}
        """).df()["n"].iloc[0]
        if n > 0:
            found.append(operator)
    con.close()
    if not found:
        raise ValueError(f"brandId={brand_id} not found under any operator on {as_of}")
    return found


def _discover_brand_ids(as_of: dt.date) -> list[tuple[str, int]]:
    """
    Every (operator, brandId) pair that exists on `as_of`, for brand_id='basel'
    """
    con = _s3_duckdb()
    pairs = []
    for operator in OPERATORS:
        paths = _landing_paths(operator, LANDING_TABLES["player"], [as_of])
        brand_ids = con.sql(f"""
            SELECT DISTINCT brandId FROM read_parquet({paths}, union_by_name=True)
            WHERE brandId IS NOT NULL ORDER BY brandId
        """).df()["brandId"].tolist()
        pairs.extend((operator, int(b)) for b in brand_ids)
    con.close()
    return pairs


def _signed_log1p(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


def _windowed(event_days: pd.DataFrame, cutoff: dt.date, start_offset: int, end_offset: int) -> pd.DataFrame:
    start = cutoff - dt.timedelta(days=start_offset)
    end = cutoff - dt.timedelta(days=end_offset)
    return event_days[(event_days["activity_date"] > start) & (event_days["activity_date"] <= end)]


def _bet_events(
    con: duckdb.DuckDBPyConnection, operator: str, brand_id: int, days: list[dt.date]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = _landing_paths(operator, LANDING_TABLES["bet"], days)
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE bet_events AS
        SELECT * FROM read_parquet({paths}, union_by_name=True) WHERE brandId = {brand_id}
    """)
    activity = con.sql("""
        WITH active_days AS (
            SELECT DISTINCT partyId, CAST(dateTime AS DATE) AS activity_date
            FROM bet_events
            WHERE partyId IS NOT NULL AND dateTime IS NOT NULL
        )
        SELECT partyId, activity_date,
               LEAD(activity_date) OVER (PARTITION BY partyId ORDER BY activity_date) AS next_active_date
        FROM active_days
    """).df()
    amounts = con.sql("""
        SELECT partyId, CAST(dateTime AS DATE) AS activity_date, currency,
               -SUM(CASE WHEN tranType = 'GAME_BET' THEN amountReal ELSE 0 END) AS stake_local,
               SUM(CASE WHEN tranType = 'GAME_WIN' THEN amountReal ELSE 0 END) AS win_local
        FROM bet_events
        WHERE partyId IS NOT NULL AND dateTime IS NOT NULL AND rolledBack = False
          AND tranType IN ('GAME_BET', 'GAME_WIN')
        GROUP BY partyId, CAST(dateTime AS DATE), currency
    """).df()
    activity["gap_to_next_days"] = (
        pd.to_datetime(activity["next_active_date"]) - pd.to_datetime(activity["activity_date"])
    ).dt.days
    return activity, amounts


def _deposit_amounts_daily(
    con: duckdb.DuckDBPyConnection, operator: str, brand_id: int, days: list[dt.date], fallback_currency: str
) -> pd.DataFrame:
    paths = _landing_paths(operator, LANDING_TABLES["transaction"], days)
    con.sql(f"CREATE OR REPLACE VIEW txn AS SELECT * FROM read_parquet({paths}, union_by_name=True)")
    has_currency = "currency" in con.sql("SELECT * FROM txn LIMIT 0").columns
    currency_select = "currency," if has_currency else ""
    currency_group_by = ", currency" if has_currency else ""
    df = con.sql(f"""
        SELECT partyId, CAST(requestDate AS DATE) AS activity_date, {currency_select}
               SUM(processedAmount) AS deposit_amount_local
        FROM txn
        WHERE transactionType = 'DEPOSIT' AND status = 'COMPLETED' AND brandId = {brand_id}
          AND partyId IS NOT NULL AND requestDate IS NOT NULL
        GROUP BY partyId, CAST(requestDate AS DATE){currency_group_by}
    """).df()
    if not has_currency:
        df["currency"] = fallback_currency
    df["currency"] = df["currency"].fillna(fallback_currency)
    return df


def _player_regdate(con: duckdb.DuckDBPyConnection, operator: str, brand_id: int, days: list[dt.date]) -> pd.DataFrame:
    paths = _landing_paths(operator, LANDING_TABLES["player"], days)
    con.sql(f"CREATE OR REPLACE VIEW player_events AS SELECT * FROM read_parquet({paths}, union_by_name=True)")
    return con.sql(f"""
        SELECT DISTINCT partyId, CAST(regdate AS DATE) AS regdate
        FROM player_events WHERE brandId = {brand_id}
    """).df()


def _bonus_events(
    con: duckdb.DuckDBPyConnection, operator: str, party_ids: set[int], days: list[dt.date], fallback_currency: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = _landing_paths(operator, LANDING_TABLES["bonus"], days)
    con.sql(f"CREATE OR REPLACE TEMP TABLE bonus_events AS SELECT * FROM read_parquet({paths}, union_by_name=True)")
    event_days = con.sql("""
        SELECT DISTINCT partyId, CAST(changeStatusTimestamp AS DATE) AS activity_date
        FROM bonus_events WHERE partyId IS NOT NULL AND changeStatusTimestamp IS NOT NULL
    """).df()
    amounts = con.sql("""
        SELECT partyId, CAST(changeStatusTimestamp AS DATE) AS activity_date, SUM(amount) AS bonus_amount_local
        FROM bonus_events
        WHERE partyId IS NOT NULL AND changeStatusTimestamp IS NOT NULL AND status = 'ACTIVE'
        GROUP BY partyId, CAST(changeStatusTimestamp AS DATE)
    """).df()
    event_days = event_days[event_days["partyId"].isin(party_ids)].copy()
    amounts = amounts[amounts["partyId"].isin(party_ids)].copy()
    amounts["currency"] = fallback_currency
    return event_days, amounts




def _add_recency(snap: pd.DataFrame, event_df: pd.DataFrame, prefix: str, cutoff: dt.date) -> pd.DataFrame:
    """
    days_since_last_<prefix>, plus a missingness flag for "never happened by cutoff"
    """
    last = event_df.groupby("partyId")["activity_date"].max().rename(f"last_{prefix}_date")
    snap = snap.merge(last, on="partyId", how="left")
    never_seen_sentinel = (cutoff - HISTORY_START).days + 1
    snap[f"days_since_last_{prefix}_missing"] = snap[f"last_{prefix}_date"].isna().astype(int)
    snap[f"days_since_last_{prefix}"] = snap[f"last_{prefix}_date"].apply(
        lambda d: (cutoff - d).days if pd.notna(d) else never_seen_sentinel
    )
    return snap.drop(columns=[f"last_{prefix}_date"])


def _add_window_counts(snap: pd.DataFrame, event_df: pd.DataFrame, prefix: str, cutoff: dt.date, windows=COUNT_WINDOWS) -> pd.DataFrame:
    for w in windows:
        counts = _windowed(event_df, cutoff, w, 0).groupby("partyId").size().rename(f"n_{prefix}_last_{w}d")
        snap = snap.merge(counts, on="partyId", how="left")
        snap[f"n_{prefix}_last_{w}d"] = snap[f"n_{prefix}_last_{w}d"].fillna(0).astype(int)
    return snap


def _add_count_trend(snap: pd.DataFrame, event_df: pd.DataFrame, prefix: str, cutoff: dt.date, windows=TREND_WINDOWS) -> pd.DataFrame:
    """<prefix>_drop_{w}_vs_prior{w} for each window pair w in `windows` (Lag+trend)."""
    for w in windows:
        last_col, prior_col = f"n_{prefix}_last_{w}d", f"_n_{prefix}_prior_{w}d"
        if last_col not in snap.columns:
            last = _windowed(event_df, cutoff, w, 0).groupby("partyId").size().rename(last_col)
            snap = snap.merge(last, on="partyId", how="left")
            snap[last_col] = snap[last_col].fillna(0).astype(int)
        prior = _windowed(event_df, cutoff, 2 * w, w).groupby("partyId").size().rename(prior_col)
        snap = snap.merge(prior, on="partyId", how="left")
        snap[prior_col] = snap[prior_col].fillna(0).astype(int)
        snap[f"{prefix}_drop_{w}_vs_prior{w}"] = snap[last_col] - snap[prior_col]
        snap = snap.drop(columns=[prior_col])
    return snap


def _add_amount_windows(snap: pd.DataFrame, amounts_df: pd.DataFrame, value_cols: list[str], cutoff: dt.date, windows=COUNT_WINDOWS) -> pd.DataFrame:
    """sum(<value_col>)_last_{w}d for each value column and window (Monetary / GGR decomposition)."""
    for w in windows:
        window_df = _windowed(amounts_df, cutoff, w, 0)
        agg = window_df.groupby("partyId")[value_cols].sum()
        agg.columns = [f"{c}_last_{w}d" for c in value_cols]
        snap = snap.merge(agg, on="partyId", how="left")
        for c in agg.columns:
            snap[c] = snap[c].fillna(0.0)
    return snap


def _add_amount_trend(snap: pd.DataFrame, amounts_df: pd.DataFrame, value_cols: list[str], cutoff: dt.date, windows=TREND_WINDOWS) -> pd.DataFrame:
    """<value_col>_trend_{w} = sum_last_{w}d - sum_prior_{w}d, for each window pair (Lag+trend)."""
    for w in windows:
        last_df = _windowed(amounts_df, cutoff, w, 0).groupby("partyId")[value_cols].sum()
        prior_df = _windowed(amounts_df, cutoff, 2 * w, w).groupby("partyId")[value_cols].sum()
        last_df = last_df.reindex(snap["partyId"].unique(), fill_value=0.0)
        prior_df = prior_df.reindex(snap["partyId"].unique(), fill_value=0.0)
        trend = (last_df - prior_df).rename(columns={c: f"{c}_trend_{w}" for c in value_cols})
        snap = snap.merge(trend, on="partyId", how="left")
        for c in trend.columns:
            snap[c] = snap[c].fillna(0.0)
    return snap


def _build_single_brand_features(operator: str, brand_id: int, cutoff: dt.date, use_cache: bool = True) -> pd.DataFrame:
    cache_path = _feature_cache_path(operator, brand_id, cutoff)
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    days = _date_range(HISTORY_START, cutoff)

    con = _s3_duckdb()
    try:
        return _build_single_brand_features_with_con(con, operator, brand_id, cutoff, days, use_cache, cache_path)
    finally:
        con.close()


def _build_single_brand_features_with_con(
    con: duckdb.DuckDBPyConnection,
    operator: str,
    brand_id: int,
    cutoff: dt.date,
    days: list[dt.date],
    use_cache: bool,
    cache_path: Path,
) -> pd.DataFrame:
    activity, bet_amounts_df = _bet_events(con, operator, brand_id, days)
    if activity.empty:
        return pd.DataFrame()
    activity["activity_date"] = pd.to_datetime(activity["activity_date"]).dt.date
    activity["next_active_date"] = pd.to_datetime(activity["next_active_date"]).dt.date
    bet_amounts_df["activity_date"] = pd.to_datetime(bet_amounts_df["activity_date"]).dt.date

    activity = activity[activity["activity_date"] <= cutoff]
    bet_amounts_df = bet_amounts_df[bet_amounts_df["activity_date"] <= cutoff]
    dominant_currency = bet_amounts_df["currency"].mode().iloc[0]
    player_days = activity[["partyId", "activity_date"]].drop_duplicates()

    snap = player_days[["partyId"]].drop_duplicates().reset_index(drop=True)

    snap = _add_recency(snap, player_days, "active", cutoff)
    snap = _add_window_counts(snap, player_days, "active_days", cutoff)
    snap = _add_count_trend(snap, player_days, "freq", cutoff)

    prior_gaps = activity[activity["next_active_date"].notna() & (activity["next_active_date"] <= cutoff)]
    for gap_w in DORMANCY_GAP_WINDOWS:
        n_dormancy = (
            prior_gaps[prior_gaps["gap_to_next_days"] >= gap_w]
            .groupby("partyId").size().rename(f"n_prior_dormancy_episodes_{gap_w}d")
        )
        snap = snap.merge(n_dormancy, on="partyId", how="left")
        snap[f"n_prior_dormancy_episodes_{gap_w}d"] = snap[f"n_prior_dormancy_episodes_{gap_w}d"].fillna(0).astype(int)
    max_gap = prior_gaps.groupby("partyId")["gap_to_next_days"].max().rename("max_prior_gap_days")
    snap = snap.merge(max_gap, on="partyId", how="left")
    snap["max_prior_gap_days"] = snap["max_prior_gap_days"].fillna(0).astype(int)

    regdate_df = _player_regdate(con, operator, brand_id, days)
    regdate_df["regdate"] = pd.to_datetime(regdate_df["regdate"]).dt.date
    snap = snap.merge(regdate_df[["partyId", "regdate"]], on="partyId", how="left")
    snap["tenure_days_missing"] = snap["regdate"].isna().astype(int)
    snap["tenure_days"] = (cutoff - snap["regdate"]).apply(lambda x: x.days if pd.notna(x) else np.nan)
    median_tenure = snap["tenure_days"].median()
    snap["tenure_days"] = snap["tenure_days"].fillna(median_tenure if pd.notna(median_tenure) else 0)
    snap = snap.drop(columns=["regdate"])


    deposit_amounts_df = _deposit_amounts_daily(con, operator, brand_id, days, dominant_currency)
    deposit_amounts_df["activity_date"] = pd.to_datetime(deposit_amounts_df["activity_date"]).dt.date
    deposit_amounts_df = deposit_amounts_df[deposit_amounts_df["activity_date"] <= cutoff]

    fx_rates_df = load_fx_rates_eur(FX_RATES_CSV, HISTORY_START - dt.timedelta(days=1), cutoff + dt.timedelta(days=1))
    bet_amounts_df["stake"] = convert_amount_to_eur(bet_amounts_df, "stake_local", "currency", "activity_date", fx_rates_df)
    bet_amounts_df["win"] = convert_amount_to_eur(bet_amounts_df, "win_local", "currency", "activity_date", fx_rates_df)
    bet_amounts_df["net_loss"] = bet_amounts_df["stake"] - bet_amounts_df["win"]
    deposit_amounts_df["deposit_amount"] = convert_amount_to_eur(
        deposit_amounts_df, "deposit_amount_local", "currency", "activity_date", fx_rates_df
    )

    deposit_days_df = deposit_amounts_df[["partyId", "activity_date"]].drop_duplicates()
    snap = _add_recency(snap, deposit_days_df, "deposit", cutoff)
    snap = _add_window_counts(snap, deposit_days_df, "deposit", cutoff)
    snap = _add_count_trend(snap, deposit_days_df, "deposit", cutoff)

    party_ids = set(regdate_df["partyId"])
    bonus_event_days_df, bonus_amounts_df = _bonus_events(con, operator, party_ids, days, dominant_currency)
    bonus_event_days_df["activity_date"] = pd.to_datetime(bonus_event_days_df["activity_date"]).dt.date
    bonus_event_days_df = bonus_event_days_df[bonus_event_days_df["activity_date"] <= cutoff]
    snap = _add_recency(snap, bonus_event_days_df, "bonus", cutoff)
    snap = _add_window_counts(snap, bonus_event_days_df, "bonus", cutoff)
    snap = _add_count_trend(snap, bonus_event_days_df, "bonus", cutoff)

    bonus_amounts_df["activity_date"] = pd.to_datetime(bonus_amounts_df["activity_date"]).dt.date
    bonus_amounts_df = bonus_amounts_df[bonus_amounts_df["activity_date"] <= cutoff]
    bonus_amounts_df["bonus_amount"] = convert_amount_to_eur(
        bonus_amounts_df, "bonus_amount_local", "currency", "activity_date", fx_rates_df
    )

    snap = _add_amount_windows(snap, bet_amounts_df, ["stake", "win", "net_loss"], cutoff)
    snap = _add_amount_windows(snap, deposit_amounts_df, ["deposit_amount"], cutoff)
    snap = _add_amount_windows(snap, bonus_amounts_df, ["bonus_amount"], cutoff)
    snap = _add_amount_trend(snap, bet_amounts_df, ["stake", "win", "net_loss"], cutoff)
    snap = _add_amount_trend(snap, deposit_amounts_df, ["deposit_amount"], cutoff)
    snap = _add_amount_trend(snap, bonus_amounts_df, ["bonus_amount"], cutoff)


    for w in TREND_WINDOWS:
        snap[f"already_dormant_{w}"] = (
            (snap[f"n_active_days_last_{w}d"] == 0) & (snap[f"stake_last_{w}d"] == 0)
        ).astype(int)

    for w in COUNT_WINDOWS:
        stake_col, win_col, rtp_col = f"stake_last_{w}d", f"win_last_{w}d", f"rtp_last_{w}d"
        snap[f"{rtp_col}_missing"] = (snap[stake_col] <= 0).astype(int)
        snap[rtp_col] = np.where(snap[stake_col] > 0, snap[win_col] / snap[stake_col], np.nan)
        median_rtp = snap[rtp_col].median()
        snap[rtp_col] = snap[rtp_col].fillna(median_rtp if pd.notna(median_rtp) else 0.0)

    sign_log_cols = [c for c in snap.columns if c not in NEVER_SIGN_LOG and snap[c].dtype.kind in "if"]
    for col in sign_log_cols:
        snap[col] = _signed_log1p(snap[col])


    label_available = cutoff + dt.timedelta(days=LABEL_HORIZON_DAYS) <= DATA_AVAILABLE_THROUGH

    snap.insert(0, "cutoff_date", cutoff)
    snap.insert(1, "operator", operator)
    snap.insert(2, "brandId", brand_id)
    snap.insert(3, "label_available_60d", label_available)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        snap.to_parquet(cache_path)
    return snap



NEVER_SIGN_LOG = (
    {"partyId", "brandId"}
    | {f"rtp_last_{w}d" for w in COUNT_WINDOWS}
    | {f"rtp_last_{w}d_missing" for w in COUNT_WINDOWS}
    | {f"already_dormant_{w}" for w in TREND_WINDOWS}
    | {f"days_since_last_{p}_missing" for p in ("active", "deposit", "bonus")}
    | {"tenure_days_missing"}
)


def _feature_columns() -> dict[str, list[str]]:
    recency = ["days_since_last_active", "days_since_last_deposit", "days_since_last_bonus"]
    frequency = [f"n_{p}_last_{w}d" for p in ("active_days", "deposit", "bonus") for w in COUNT_WINDOWS]
    monetary = [
        f"{c}_last_{w}d" for c in ("stake", "win", "net_loss", "deposit_amount", "bonus_amount") for w in COUNT_WINDOWS
    ]
    lag_trend = (
        [f"{p}_drop_{w}_vs_prior{w}" for p in ("freq", "deposit", "bonus") for w in TREND_WINDOWS]
        + [f"{c}_trend_{w}" for c in ("stake", "win", "net_loss", "deposit_amount", "bonus_amount") for w in TREND_WINDOWS]
        + [f"already_dormant_{w}" for w in TREND_WINDOWS]
    )
    tenure = ["tenure_days", "max_prior_gap_days"] + [f"n_prior_dormancy_episodes_{w}d" for w in DORMANCY_GAP_WINDOWS]
    mix = [f"rtp_last_{w}d" for w in COUNT_WINDOWS]
    return {
        "recency": recency, "frequency": frequency, "monetary": monetary,
        "lag_trend": lag_trend, "tenure": tenure, "mix": mix,
    }


FEATURE_COLUMNS = _feature_columns()


BASELINE_FEATURES = [
    "days_since_last_active", "days_since_last_deposit", "days_since_last_bonus",
    "n_active_days_last_7d", "n_active_days_last_30d",
    "n_deposit_last_7d", "n_deposit_last_30d",
    "n_bonus_last_7d",
    "stake_last_7d", "stake_last_30d",
    "win_last_7d", "win_last_30d",
    "net_loss_last_7d", "net_loss_last_30d",
    "deposit_amount_last_7d", "deposit_amount_last_30d",
    "bonus_amount_last_7d",
    "freq_drop_7_vs_prior7", "deposit_drop_7_vs_prior7", "bonus_drop_7_vs_prior7",
    "stake_trend_7", "win_trend_7", "net_loss_trend_7", "deposit_amount_trend_7", "bonus_amount_trend_7",
    "already_dormant_7",
    "tenure_days", "max_prior_gap_days",
    "rtp_last_7d",
]


ID_COLUMNS = ["cutoff_date", "operator", "brandId", "partyId", "label_available_60d"]


def _unscale(df: pd.DataFrame) -> pd.DataFrame:
    """Undo the signed log1p on exactly the columns it was applied to."""
    # The cache stores scaled values; signed log1p is exactly invertible, so the
    # unscaled view is derived here instead of invalidating every cached snapshot.
    df = df.copy()
    for col in [c for c in df.columns if c not in NEVER_SIGN_LOG and df[c].dtype.kind in "if"]:
        df[col] = (np.sign(df[col]) * np.expm1(np.abs(df[col]))).round(6)
    return df


def build_feature_store(
    cutoff_date: str | dt.date,
    brand_id: int | str,
    use_cache: bool = True,
    full: bool = False,
    scaled: bool = True,
) -> pd.DataFrame:
    """One row per player, as of `cutoff_date`, for `brand_id` (or every brandId if `brand_id="basel"`)
    """
    cutoff = pd.to_datetime(cutoff_date).date() if isinstance(cutoff_date, str) else cutoff_date
    if cutoff <= HISTORY_START:
        raise ValueError(f"cutoff_date must be after {HISTORY_START} (earliest available landing data)")

    if isinstance(brand_id, str) and brand_id.lower() == ALL_BRANDS:
        brand_pairs = _discover_brand_ids(cutoff)
    else:
        brand_id = int(brand_id)
        brand_pairs = [(operator, brand_id) for operator in _resolve_operators_for_brand(brand_id, cutoff)]

    frames = [_build_single_brand_features(operator, bid, cutoff, use_cache) for operator, bid in brand_pairs]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError(f"no data found for brand_id={brand_id!r} as of {cutoff}")
    result = pd.concat(frames, ignore_index=True)
    if not scaled:
        result = _unscale(result)

    if full:
        return result
    return result[ID_COLUMNS + BASELINE_FEATURES]
