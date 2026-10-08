"""EDA stage 2, outliers and anomalies: the study of one table, and its outputs.

    .venv/bin/python eda/anomalies.py --source daily
    .venv/bin/python eda/anomalies.py --source features
    .venv/bin/python eda/anomalies.py --source features --cutoff-dates 2026-07-18 2026-07-25
    .venv/bin/python eda/anomalies.py --source features --brand-id basel   # one study per brand
    .venv/bin/python eda/anomalies.py --source features --input some_unscaled_features.parquet

Any table described by an AnomalySpec can be studied. Two specs ship with it:
    DAILY_SPEC      one row per (tenant_id, player_id, day) with a bet or a payment, EUR amounts, from the
                   local daily caches (activity + payments, src/features/daily_cache.py) by load_player_day()
    FEATURES_SPEC  one row per (player_id, cutoff_date), the output of raw_features(); every
                   detector uses only the spec columns actually present

Detectors: univariate (IQR and robust z-score), multivariate (Isolation Forest), temporal
(daily totals vs a rolling band) and structural (impossible sequences). Outliers are flagged,
never removed. Outputs per brand:
    data/03_output/anomaly_table_{source}_brand{brand_id}.csv
    data/02_intermediate/anomaly_flags_{source}_brand{brand_id}.parquet
    configs/winsorisation_features_brand{brand_id}.yaml   (features only: p99.5 caps, src/features/winsorisation.py)

The pipeline does not run this study: it only needs the caps, which it computes with the same
code (winsorisation.write_feature_caps) on every run.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
import daily_cache  # noqa: E402
from build_features import (  # noqa: E402
    ALL_BRANDS, DEPOSITS_VALID_FROM, LOOKBACK_DAYS, data_available_through, parse_brand_id, raw_features,
)
from winsorisation import MONETARY_FEATURES, SIGNED_MONETARY_FEATURES  # noqa: E402
from build_survival_dataset import DEFAULT_CUTOFF_DATES  # noqa: E402
from winsorisation import WINSOR_CONFIG_PATH, build_winsorisation_config, write_winsorisation_config  # noqa: E402


@dataclass(frozen=True)
class AnomalySpec:
    name: str
    grain: str
    key_columns: tuple[str, ...]
    date_column: str
    monetary_columns: tuple[str, ...]
    count_columns: tuple[str, ...]
    ratio_columns: tuple[str, ...]
    signed_columns: frozenset[str]
    temporal_sum_columns: tuple[str, ...]
    temporal_active_column: str | None
    structural_rules: tuple[Callable, ...]
    winsor_applies_to: str


TOP_N = 20

ANOMALY_COLUMNS = [
    "column", "method", "n_evaluated", "n_flagged", "share",
    "threshold", "note", "top_20_examples",
]


def _present_columns(df, columns):
    """Return unique configured columns present in the frame."""
    return [column for column in dict.fromkeys(columns) if column in df.columns]


def _evaluated_mask(s, column, spec):
    """Select nonmissing nonzero signed values or positive unsigned values."""
    return (
        s.notna() & (s.ne(0) if column in spec.signed_columns else s.gt(0))
    ).fillna(False)


def _result(column, method, n_evaluated, n_flagged, threshold, note="", examples=""):
    n_evaluated, n_flagged = int(n_evaluated), int(n_flagged)
    return {
        "column": column,
        "method": method,
        "n_evaluated": n_evaluated,
        "n_flagged": n_flagged,
        "share": float(n_flagged / n_evaluated) if n_evaluated else 0.0,
        "threshold": threshold,
        "note": note,
        "top_20_examples": examples,
    }


def _examples(frame, values, severity, flagged, spec, date_values=False,
              key_columns=None):
    """Format up to TOP_N flagged examples in descending severity order."""
    positions = np.flatnonzero(
        pd.Series(flagged).fillna(False).to_numpy(dtype=bool)
    )
    if not len(positions):
        return ""
    scores = pd.Series(severity).to_numpy(dtype=float, na_value=np.nan)
    positions = positions[np.argsort(-scores[positions], kind="stable")[:TOP_N]]
    selected = frame.iloc[positions]
    keys = spec.key_columns if key_columns is None else key_columns
    parts = []
    for column in keys:
        if column not in selected.columns:
            parts.append(np.full(len(selected), "<missing>", dtype=object))
        elif column == spec.date_column:
            parts.append(
                pd.to_datetime(selected[column])
                .dt.strftime("%Y-%m-%d").fillna("NaT").to_numpy(dtype=str)
            )
        else:
            parts.append(selected[column].astype(str).to_numpy(dtype=str))
    selected_values = pd.Series(values).iloc[positions]
    if date_values:
        formatted = (
            pd.to_datetime(selected_values)
            .dt.strftime("%Y-%m-%d").fillna("NaT").to_numpy(dtype=str)
        )
    else:
        formatted = np.char.mod(
            "%.2f", selected_values.to_numpy(dtype=float, na_value=np.nan)
        )
    parts.append(formatted)
    text = np.asarray(parts[0], dtype=str)
    for part in parts[1:]:
        text = np.char.add(np.char.add(text, "|"), np.asarray(part, dtype=str))
    return "; ".join(text.tolist())


def univariate_outliers(df, spec) -> tuple[list[dict], dict[str, pd.Series]]:
    """Flag IQR and robust-MAD outliers using only evaluated values."""
    rows, flags = [], {}
    columns = _present_columns(
        df, spec.monetary_columns + spec.count_columns + spec.ratio_columns
    )
    for column in columns:
        s = df[column].astype(float)
        evaluated = _evaluated_mask(s, column, spec)
        values = s[evaluated]
        n = len(values)
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flag = (evaluated & (s.lt(lower) | s.gt(upper))).fillna(False)
        flag = flag.astype(bool).rename(f"flag_{column}_iqr")
        flags[flag.name] = flag
        severity = np.maximum(lower - s, s - upper)
        rows.append(_result(
            column, "iqr_1.5", n, flag.sum(),
            f"x < {lower:.6g} or x > {upper:.6g}" if n else "not available",
            "" if n else "No evaluated values",
            _examples(df, s, severity, flag, spec),
        ))

        median = values.median()
        mad = (values - median).abs().median()
        z = pd.Series(0.0, index=df.index)
        note = ""
        if not n:
            note = "No evaluated values"
        elif mad == 0:
            note = "MAD is 0, not applicable"
        else:
            z = 0.6745 * (s - median) / mad
        flag = (evaluated & z.abs().gt(3.5)).fillna(False)
        flag = flag.astype(bool).rename(f"flag_{column}_mad")
        flags[flag.name] = flag
        rows.append(_result(
            column, "robust_z_mad_3.5", n, flag.sum(),
            "|0.6745 * (x - median) / MAD| > 3.5",
            note, _examples(df, s, z.abs(), flag, spec),
        ))
    return rows, flags


def isolation_forest_outliers(df, spec, random_state=42) -> tuple[list[dict], pd.Series]:
    """Flag unusual signed-log-transformed vectors with IsolationForest."""
    columns = _present_columns(
        df, spec.monetary_columns + spec.count_columns + spec.ratio_columns
    )
    flag = pd.Series(False, index=df.index, name="flag_isolation_forest")
    vector_name = f"{spec.grain}_vector"
    if not columns or df.empty:
        note = "No input rows" if df.empty else "No configured numeric columns present"
        return [_result(
            vector_name, "isolation_forest_c0.005", 0, 0,
            "not available", note,
        )], flag

    matrix = df[columns].fillna(0).to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("IsolationForest inputs must contain only finite numeric values.")
    transformed = np.sign(matrix) * np.log1p(np.abs(matrix))
    model = IsolationForest(
        n_estimators=200, contamination=0.005, random_state=random_state,
    ).fit(transformed)
    scores = model.score_samples(transformed)
    flag.iloc[:] = model.predict(transformed) == -1
    return [_result(
        vector_name, "isolation_forest_c0.005", len(df), flag.sum(),
        f"score_samples < {model.offset_:.6g}",
        examples=_examples(df, scores, -scores, flag, spec),
    )], flag


def temporal_outliers(df, spec) -> list[dict]:
    """Flag daily deviations using strictly preceding rolling medians and residual MADs."""
    if not spec.temporal_sum_columns or spec.date_column not in df.columns:
        return []
    sums = _present_columns(df, spec.temporal_sum_columns)
    include_active = (
        spec.temporal_active_column is not None
        and spec.temporal_active_column in df.columns
    )
    columns = (["active_players"] if include_active else []) + sums
    if not columns:
        return []
    method = "rolling_median_28d_3mad"
    threshold = "|x - previous-28-day median| > 3 * MAD; MAD > 0"
    work = df.loc[df[spec.date_column].notna()].copy()
    if work.empty:
        return [
            _result(column, method, 0, 0, threshold, "No calendar days")
            for column in columns
        ]

    work[spec.date_column] = pd.to_datetime(work[spec.date_column])
    daily = work.groupby(spec.date_column)[sums].sum()
    if include_active:
        # Player-day grain: one row per player and day, so active players = rows with activity.
        active = work.loc[work[spec.temporal_active_column].gt(0)].groupby(spec.date_column).size()
        daily["active_players"] = active.reindex(daily.index, fill_value=0)
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar, fill_value=0).rename_axis(spec.date_column)
    frame = daily.reset_index()
    rows = []
    for column in columns:
        s = daily[column]
        baseline = s.shift(1).rolling(28, min_periods=14).median()
        residual = (s - baseline).abs()
        mad = residual.shift(1).rolling(28, min_periods=14).median()
        usable = baseline.notna() & mad.notna() & mad.gt(0)
        flag = usable & residual.gt(3 * mad)
        severity = residual / mad.where(mad.gt(0))
        rows.append(_result(
            column, method, usable.sum(), flag.sum(), threshold,
            "MAD uses preceding absolute residuals; evaluation requires MAD > 0.",
            _examples(
                frame, s, severity, flag, spec,
                key_columns=(spec.date_column,),
            ),
        ))
    return rows


def withdrawal_without_prior_deposit(df, history_start):
    """Flag players whose first withdrawal has no deposit on or before that day. Only players first
    seen on or after `history_start` are evaluated (the source has completed deposits only from
    DEPOSITS_VALID_FROM, and no reliable registration date: an older player may have deposited before)."""
    required = ("tenant_id", "player_id", "day", "withdraw_count", "deposit_count", "withdraw_eur")
    if any(column not in df.columns for column in required):
        return [], {}
    key = df["tenant_id"].astype(str) + ":" + df["player_id"].astype(str)
    first_seen = df.groupby(key)["day"].min()
    first_withdrawal = df.loc[df["withdraw_count"].gt(0)].groupby(key)["day"].min()
    first_deposit = df.loc[df["deposit_count"].gt(0)].groupby(key)["day"].min().reindex(first_withdrawal.index)
    eligible = first_seen.reindex(first_withdrawal.index).ge(pd.Timestamp(history_start))
    flagged = eligible & (first_deposit.isna() | first_deposit.gt(first_withdrawal))
    first_flagged = first_withdrawal[flagged]
    flag = (key.isin(first_flagged.index) & df["day"].eq(key.map(first_flagged))) \
        .fillna(False).astype(bool).rename("flag_withdrawal_without_prior_deposit")
    row = _result(
        "withdraw_count", "structural:withdrawal_without_prior_deposit", eligible.sum(), flagged.sum(),
        f"No deposit on or before the first withdrawal; player first seen >= {pd.Timestamp(history_start).date()}",
        f"Players with withdrawals not evaluated (first seen before {pd.Timestamp(history_start).date()}): "
        f"{int((~eligible).sum())}. Examples ranked by first-day withdrawal EUR.",
        _examples(df, df["withdraw_eur"], df["withdraw_eur"], flag, DAILY_SPEC),
    )
    return [row], {flag.name: flag}


def _bound_rule(rule_name, column, bound, upper=False, keys=("player_id", "cutoff_date")):
    """Create a structural rule checking one column against a bound."""
    example_spec = SimpleNamespace(key_columns=keys, date_column=keys[-1])

    def rule(df, history_start):
        """Evaluate the configured bound on nonmissing values."""
        if column not in df.columns:
            return [], {}
        s = df[column]
        evaluated = s.notna()
        violation = s.gt(bound) if upper else s.lt(bound)
        flag = (
            evaluated & violation
        ).fillna(False).astype(bool).rename(f"flag_{rule_name}_{column}")
        severity = s - bound if upper else bound - s
        row = _result(
            column, f"structural:{rule_name}", evaluated.sum(), flag.sum(),
            f"{column} {'>' if upper else '<'} {bound}",
            examples=_examples(df, s, severity, flag, example_spec),
        )
        return [row], {flag.name: flag}

    return rule


def _feature_monotonic_rule(base, tolerance=0.0):
    """Create a structural rule comparing consecutive available cumulative windows."""
    def rule(df, history_start):
        """Flag decreasing cumulative values across available nonmissing window pairs."""
        columns = _present_columns(
            df, tuple(f"{base}_last_{window}d" for window in (7, 14, 30))
        )
        if len(columns) < 2:
            return [], {}
        evaluated = pd.Series(False, index=df.index)
        flag = pd.Series(False, index=df.index)
        severity = pd.Series(-np.inf, index=df.index)
        offending_value = pd.Series(np.nan, index=df.index)
        thresholds = []
        for smaller, larger in zip(columns, columns[1:]):
            valid = df[smaller].notna() & df[larger].notna()
            difference = df[smaller].astype(float) - df[larger].astype(float)
            violation = (valid & difference.gt(tolerance)).fillna(False)
            replace = violation & difference.gt(severity)
            offending_value.loc[replace] = df.loc[replace, smaller].astype(float)
            severity.loc[replace] = difference.loc[replace]
            evaluated |= valid
            flag |= violation
            thresholds.append(f"{smaller} > {larger} + {tolerance:g}")
        flag = flag.astype(bool).rename(f"flag_window_not_monotonic_{base}")
        row = _result(
            base, "structural:window_not_monotonic",
            evaluated.sum(), flag.sum(), " or ".join(thresholds),
            "Examples show the smaller-window value from the largest violating difference.",
            _examples(df, offending_value, severity, flag, FEATURES_SPEC),
        )
        return [row], {flag.name: flag}

    return rule


DAILY_KEYS = ("tenant_id", "player_id", "day")
DAILY_SPEC = AnomalySpec(
    name="daily",
    grain="player_day",
    key_columns=DAILY_KEYS,
    date_column="day",
    monetary_columns=("turnover_eur", "ggr_eur", "deposit_eur", "withdraw_eur"),
    count_columns=("bets", "rounds", "deposit_count", "withdraw_count", "failed_deposit_count"),
    ratio_columns=(),
    signed_columns=frozenset({"ggr_eur"}),
    temporal_sum_columns=("bets", "turnover_eur", "ggr_eur", "deposit_eur"),
    temporal_active_column="bets",
    structural_rules=(
        withdrawal_without_prior_deposit,
        *(_bound_rule("negative_amount", column, 0, keys=DAILY_KEYS)
          for column in ("turnover_eur", "deposit_eur", "withdraw_eur")),
    ),
    winsor_applies_to="daily EUR amounts (study only: the pipeline caps the features)",
)

FEATURES_SPEC = AnomalySpec(
    name="features",
    grain="player_cutoff",
    key_columns=("player_id", "cutoff_date"),
    date_column="cutoff_date",
    monetary_columns=MONETARY_FEATURES,
    count_columns=(
        "days_since_last_bet", "days_since_last_deposit", "active_days_l7d", "active_days_l30d", "active_days_l90d",
        "bets_l7d", "bets_l30d", "sessions_l7d", "sessions_l30d", "n_deposit_days_7d", "n_deposit_days_30d",
        "days_since_first_bet", "prior_dormancy_spells_14d", "days_since_last_return", "games_breadth_30d", "losing_streak",
        "failed_deposits_14d",
    ),
    ratio_columns=("wagered_trend_7", "heavy_loss_multiple", "deposits_30_vs_prior30", "games_breadth_ratio",
                   "withdrawals_30d_share", "bonus_stake_share_30d"),
    signed_columns=SIGNED_MONETARY_FEATURES,
    temporal_sum_columns=(),
    temporal_active_column=None,
    structural_rules=(
        tuple(_bound_rule("negative_recency", c, 0) for c in ("days_since_last_bet", "days_since_last_deposit"))
        + (_bound_rule("negative_tenure", "days_since_first_bet", 0),)
        # The population bet in the 30 days up to the cutoff: recency above 29 days is impossible.
        + (_bound_rule("exceeds_lookback", "days_since_last_bet", LOOKBACK_DAYS - 1, upper=True),)
        + tuple(_bound_rule("window_exceeds_days", c, w, upper=True)
                for c, w in (("active_days_l7d", 7), ("active_days_l30d", 30), ("active_days_l90d", 90),
                             ("n_deposit_days_7d", 7), ("n_deposit_days_30d", 30)))
        + tuple(_bound_rule("negative_amount", c, 0)
                for c in ("wagered_eur_l7d", "wagered_eur_l30d", "wagered_eur_l90d", "deposits_eur_l30d", "avg_bet_eur_l30d"))
    ),
    winsor_applies_to="feature values before sign-log scaling",
)


def structural_outliers(df, spec, history_start) -> tuple[list[dict], dict[str, pd.Series]]:
    """Combine anomaly records and flags from the configured structural rules."""
    rows, flags = [], {}
    for rule in spec.structural_rules:
        rule_rows, rule_flags = rule(df, history_start)
        rows.extend(rule_rows)
        flags.update(rule_flags)
    return rows, flags


def save_anomaly_outputs(table, flagged, config, source, brand_id) -> dict[str, Path]:
    """Write the anomaly table and the flagged rows; for the features, also the winsorisation YAML
    that build_feature_store() reads (the daily study's caps are only reported in the table)."""
    suffix = f"{source}_brand{brand_id}"
    paths = {
        "table": PROJECT_ROOT / f"data/03_output/anomaly_table_{suffix}.csv",
        "flags": PROJECT_ROOT / f"data/02_intermediate/anomaly_flags_{suffix}.parquet",
    }
    if source == "features":
        paths["config"] = Path(str(WINSOR_CONFIG_PATH).format(brand_id=brand_id))
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths["table"], index=False)
    flagged.to_parquet(paths["flags"], index=False)
    if "config" in paths:
        write_winsorisation_config(config, paths["config"])
    return paths


def build_anomaly_table(rows: list[dict]) -> pd.DataFrame:
    """Assemble anomaly records with ordered columns, rounded shares, and stable sorting."""
    table = pd.DataFrame(rows, columns=ANOMALY_COLUMNS)
    table["share"] = table["share"].astype(float).round(6)
    return table.sort_values(["method", "column"], kind="stable").reset_index(drop=True)


def detect_anomalies(
    df, spec, history_start=None, history_end=None, brand_id=None, random_state=42
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return anomaly records, a flagged copy of the input, and monetary winsorisation caps."""
    frame = df.reset_index(drop=True).copy()
    if spec.date_column in frame.columns:
        dates = pd.to_datetime(frame[spec.date_column])
        if history_start is None:
            history_start = dates.min()
        if history_end is None:
            history_end = dates.max()
    if brand_id is None and "brandId" in frame.columns:
        brands = frame["brandId"].unique()
        if len(brands) == 1 and pd.notna(brands[0]):
            brand_id = brands[0]

    rows, flags = univariate_outliers(frame, spec)
    forest_rows, forest_flag = isolation_forest_outliers(
        frame, spec, random_state=random_state
    )
    rows.extend(forest_rows)
    flags[forest_flag.name] = forest_flag
    rows.extend(temporal_outliers(frame, spec))
    structural_rows, structural_flags = structural_outliers(
        frame, spec, history_start
    )
    rows.extend(structural_rows)
    flags.update(structural_flags)

    config = build_winsorisation_config(
        frame, spec.monetary_columns, spec.signed_columns, brand_id, spec.name, spec.grain,
        spec.winsor_applies_to, history_start, history_end,
    )
    flag_frame = pd.DataFrame(flags, index=frame.index).fillna(False).astype(bool)
    frame = pd.concat(
        [frame.drop(columns=list(flags) + ["flag_any"], errors="ignore"), flag_frame],
        axis=1,
    )
    frame["flag_any"] = flag_frame.any(axis=1)
    return build_anomaly_table(rows), frame, config




def load_player_day(brand_id: int) -> pd.DataFrame:
    """Player-day table of one brand from the local daily caches: bets (activity) and payments, EUR."""
    activity = daily_cache.read("activity", brand_id, columns="tenant_id, brand_id, player_id, day, bets, rounds, "
                                                              "turnover_eur, ggr_eur")
    payments = daily_cache.read("payments", brand_id, columns="tenant_id, player_id, day, deposit_count, deposit_eur, "
                                                              "withdraw_count, withdraw_eur, failed_deposit_count")
    player_day = activity.drop(columns="brand_id").merge(payments, on=list(DAILY_KEYS), how="outer")
    player_day[player_day.columns[3:]] = player_day[player_day.columns[3:]].fillna(0)
    player_day["day"] = pd.to_datetime(player_day["day"])
    return player_day.assign(brandId=brand_id)


SPECS = {"daily": DAILY_SPEC, "features": FEATURES_SPEC}


def load_table(source: str, brand_id: int | str, cutoff_dates: list[str], input_path: str | None):
    """The table to study: a parquet given on the command line, or built for the source."""
    if input_path:
        return pd.read_parquet(input_path)
    if source == "daily":
        brands = (sorted(daily_cache.read("activity", None, columns="DISTINCT brand_id")["brand_id"])
                  if brand_id == ALL_BRANDS else [brand_id])
        return pd.concat([load_player_day(int(b)) for b in brands], ignore_index=True)
    # Unscaled and unwinsorised on purpose: caps and IQR/MAD must be in raw EUR.
    return pd.concat(
        [raw_features(c, brand_id)
         for c in cutoff_dates],
        ignore_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=SPECS, required=True)
    parser.add_argument("--brand-id", type=parse_brand_id, default=64, help="brandId, or 'basel' for every brand")
    parser.add_argument("--cutoff-dates", nargs="+", default=DEFAULT_CUTOFF_DATES)
    parser.add_argument("--input", help="parquet to study instead of building the table (features must be unscaled)")
    args = parser.parse_args()

    start = time()
    spec = SPECS[args.source]
    df = load_table(args.source, args.brand_id, args.cutoff_dates, args.input)
    print(f"{spec.name}: {len(df):,} rows, {len(df[['tenant_id', 'player_id']].drop_duplicates()):,} players")

    # Daily: withdrawal_without_prior_deposit only judges players first seen once deposits are complete.
    history = (DEPOSITS_VALID_FROM, data_available_through()) if spec is DAILY_SPEC else (None, None)
    # One study per brand: thresholds and winsorisation caps are never pooled across brands.
    for brand, rows in df.groupby("brandId"):
        table, flagged, config = detect_anomalies(rows, spec, *history, brand_id=int(brand))
        paths = save_anomaly_outputs(table, flagged, config, spec.name, int(brand))
        print(f"brandId {brand}:")
        print(table[["column", "method", "n_evaluated", "n_flagged", "share"]].to_string(index=False))
        for path in paths.values():
            print(f"saved {path.relative_to(PROJECT_ROOT)}")
    print(f"execution time: {time() - start:.1f}s")


if __name__ == "__main__":
    main()
