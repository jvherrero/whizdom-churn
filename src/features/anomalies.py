"""Anomaly detection that works on any table described by an AnomalySpec.

Two specs ship with it:
    LANDING_SPEC   one row per (partyId, activity_date), EUR amounts, built from
                   the raw landing tables by load_player_day()
    FEATURES_SPEC  one row per (partyId, cutoff_date), the output of
                   build_feature_store(..., scaled=False); every detector uses
                   only the spec columns actually present, so the full 53
                   features or the 29 baseline ones both work

    table, flagged, config = detect_anomalies(df, FEATURES_SPEC)

Outliers are flagged, never removed. The winsorisation config caps every
monetary column at p99.5 (and p0.5 for signed columns), computed on non-zero
values; apply_winsorisation(df, config) applies it.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from time import time
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import (
    data_available_through,
    FX_RATES_CSV, HISTORY_START, LANDING_TABLES, LOOKBACK_DAYS, WINSOR_CONFIG_PATH,
    _date_range, _landing_paths, _resolve_operators_for_brand, _s3_duckdb,
)
from fx_rates import convert_amount_to_eur, load_fx_rates_eur

PLAYER_DAY_CACHE = PROJECT_ROOT / "data/02_intermediate/player_day_brand{brand_id}_through{end}.parquet"


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


WINSOR_UPPER_Q = 0.995
WINSOR_LOWER_Q = 0.005
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
        and "partyId" in df.columns
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
        active = (
            work.loc[work[spec.temporal_active_column].gt(0)]
            .groupby(spec.date_column)["partyId"].nunique()
        )
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


def bet_before_registration(df, history_start):
    """Flag bets occurring before the player's registration date."""
    required = ("n_bets", "regdate", "activity_date", "partyId")
    if any(column not in df.columns for column in required):
        return [], {}
    evaluated = df["n_bets"].gt(0) & df["regdate"].notna()
    flag = (
        evaluated & df["activity_date"].lt(df["regdate"])
    ).fillna(False).astype(bool).rename("flag_bet_before_registration")
    severity = (df["regdate"] - df["activity_date"]).dt.total_seconds()
    row = _result(
        "activity_date", "structural:bet_before_registration",
        evaluated.sum(), flag.sum(), "activity_date < regdate",
        f"Distinct players involved: {df.loc[flag, 'partyId'].nunique()}.",
        _examples(
            df, df["regdate"], severity, flag, LANDING_SPEC, date_values=True
        ),
    )
    return [row], {flag.name: flag}


def withdrawal_without_prior_deposit(df, history_start):
    """Flag eligible players' first withdrawals without a deposit on or before that day."""
    required = (
        "partyId", "regdate", "activity_date",
        "n_withdrawals", "n_deposits", "withdrawal_eur",
    )
    if any(column not in df.columns for column in required):
        return [], {}
    history_start = pd.Timestamp(history_start) if history_start is not None else pd.NaT
    registrations = df.groupby("partyId")["regdate"].min()
    withdrawals = df.loc[df["n_withdrawals"].gt(0)]
    players = (
        withdrawals.groupby("partyId")["activity_date"]
        .min().to_frame("first_withdrawal")
    )
    players["regdate"] = registrations.reindex(players.index)
    deposits = (
        df.loc[df["n_deposits"].gt(0)]
        .groupby("partyId")["activity_date"].min()
    )
    players["first_deposit"] = deposits.reindex(players.index)
    eligible = players["regdate"].ge(history_start)
    flagged_players = eligible & (
        players["first_deposit"].isna()
        | players["first_deposit"].gt(players["first_withdrawal"])
    )
    excluded_early = int(players["regdate"].lt(history_start).sum())
    excluded_unknown = int(players["regdate"].isna().sum())
    first_flagged = players.loc[flagged_players, "first_withdrawal"]
    flag = (
        df["partyId"].isin(first_flagged.index)
        & df["activity_date"].eq(df["partyId"].map(first_flagged))
    ).fillna(False).astype(bool).rename("flag_withdrawal_without_prior_deposit")
    row = _result(
        "n_withdrawals", "structural:withdrawal_without_prior_deposit",
        eligible.sum(), flagged_players.sum(),
        "No deposit on or before first withdrawal; regdate >= history_start",
        f"Players with withdrawals excluded because registered before history_start: "
        f"{excluded_early}; excluded because regdate is missing: {excluded_unknown}. "
        "Examples ranked by first-day withdrawal EUR.",
        _examples(
            df, df["withdrawal_eur"], df["withdrawal_eur"], flag, LANDING_SPEC
        ),
    )
    return [row], {flag.name: flag}


def _bound_rule(rule_name, column, bound, upper=False, keys=("partyId", "cutoff_date")):
    """Create a structural rule checking one column against a bound."""
    example_spec = SimpleNamespace(key_columns=keys, date_column=keys[1])

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


LANDING_SPEC = AnomalySpec(
    name="landing",
    grain="player_day",
    key_columns=("partyId", "activity_date"),
    date_column="activity_date",
    monetary_columns=(
        "stake_eur", "win_eur", "ggr_eur",
        "deposit_eur", "withdrawal_eur", "bonus_eur",
    ),
    count_columns=("n_bets", "n_deposits", "n_withdrawals", "n_bonus_events"),
    ratio_columns=(),
    signed_columns=frozenset({"ggr_eur"}),
    temporal_sum_columns=("n_bets", "stake_eur", "ggr_eur", "deposit_eur"),
    temporal_active_column="n_bets",
    structural_rules=(
        bet_before_registration,
        withdrawal_without_prior_deposit,
        *(
            _bound_rule("negative_amount", column, 0, keys=("partyId", "activity_date"))
            for column in ("stake_eur", "win_eur", "deposit_eur", "withdrawal_eur", "bonus_eur")
        ),
    ),
    winsor_applies_to="daily EUR amounts, before any window aggregation",
)

FEATURES_SPEC = AnomalySpec(
    name="features",
    grain="player_cutoff",
    key_columns=("partyId", "cutoff_date"),
    date_column="cutoff_date",
    monetary_columns=(
        tuple(
            f"{base}_last_{window}d"
            for base in ("stake", "win", "net_loss", "deposit_amount", "bonus_amount")
            for window in (7, 14, 30)
        )
        + tuple(
            f"{base}_trend_{window}"
            for base in ("stake", "win", "net_loss", "deposit_amount", "bonus_amount")
            for window in (7, 14)
        )
    ),
    count_columns=(
        ("days_since_last_active", "days_since_last_deposit", "days_since_last_bonus")
        + tuple(
            f"{base}_last_{window}d"
            for base in ("n_active_days", "n_deposit", "n_bonus")
            for window in (7, 14, 30)
        )
        + tuple(
            f"{base}_drop_{window}_vs_prior{window}"
            for base in ("freq", "deposit", "bonus")
            for window in (7, 14)
        )
        + ("tenure_days", "max_prior_gap_days")
        + tuple(f"n_prior_dormancy_episodes_{window}d" for window in (7, 14, 30))
    ),
    ratio_columns=tuple(f"rtp_last_{window}d" for window in (7, 14, 30)),
    signed_columns=frozenset(
        tuple(f"net_loss_last_{window}d" for window in (7, 14, 30))
        + tuple(
            f"{base}_trend_{window}"
            for base in ("stake", "win", "net_loss", "deposit_amount", "bonus_amount")
            for window in (7, 14)
        )
        + tuple(
            f"{base}_drop_{window}_vs_prior{window}"
            for base in ("freq", "deposit", "bonus")
            for window in (7, 14)
        )
    ),
    temporal_sum_columns=(),
    temporal_active_column=None,
    structural_rules=(
        tuple(
            _bound_rule("negative_recency", column, 0)
            for column in (
                "days_since_last_active",
                "days_since_last_deposit",
                "days_since_last_bonus",
            )
        )
        + (_bound_rule("negative_tenure", "tenure_days", 0),)
        + tuple(
            _bound_rule("exceeds_lookback", column, LOOKBACK_DAYS, upper=True)
            for column in (
                "days_since_last_active", "days_since_last_deposit", "days_since_last_bonus",
                "max_prior_gap_days",
            )
        )
        + tuple(
            _bound_rule(
                "window_exceeds_days", f"{base}_last_{window}d", window, upper=True
            )
            for base in ("n_active_days", "n_deposit", "n_bonus")
            for window in (7, 14, 30)
        )
        + tuple(
            _feature_monotonic_rule(base)
            for base in ("n_active_days", "n_deposit", "n_bonus")
        )
        + tuple(
            _feature_monotonic_rule(base, tolerance=1e-9)
            for base in ("stake", "win", "deposit_amount", "bonus_amount")
        )
        + tuple(
            _bound_rule("negative_rtp", f"rtp_last_{window}d", 0)
            for window in (7, 14, 30)
        )
        + tuple(
            _bound_rule("negative_amount", f"{base}_last_{window}d", 0)
            for base in ("stake", "win", "deposit_amount", "bonus_amount")
            for window in (7, 14, 30)
        )
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


def _format_date(value):
    """Format a supplied date or return None for a missing date."""
    return None if value is None or pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def build_winsorisation_config(df, spec, brand_id, history_start, history_end) -> dict:
    """Build monetary caps, using infinite no-op caps and NaN extrema for empty samples."""
    columns = {}
    for column in _present_columns(df, spec.monetary_columns):
        s = df[column]
        values = s[_evaluated_mask(s, column, spec)]
        n = len(values)
        upper = round(float(values.quantile(WINSOR_UPPER_Q)), 4) if n else float("inf")
        lower = None
        if column in spec.signed_columns:
            lower = round(float(values.quantile(WINSOR_LOWER_Q)), 4) if n else float("-inf")
        columns[column] = {
            "lower": lower,
            "upper": upper,
            "n_values": int(n),
            "n_above_upper": int(values.gt(upper).sum()),
            "n_below_lower": int(values.lt(lower).sum()) if lower is not None else 0,
            "max_observed": round(float(values.max()), 4) if n else float("nan"),
            "min_observed": round(float(values.min()), 4) if n else float("nan"),
        }
    if isinstance(brand_id, np.generic):
        brand_id = brand_id.item()
    if brand_id is not None and pd.isna(brand_id):
        brand_id = None
    return {
        "version": 1,
        "brand_id": brand_id,
        "source": spec.name,
        "grain": spec.grain,
        "applies_to": spec.winsor_applies_to,
        "upper_quantile": WINSOR_UPPER_Q,
        "lower_quantile_signed_only": WINSOR_LOWER_Q,
        "computed_on": "non-zero values (non-negative columns: > 0, signed columns: != 0)",
        "history_start": _format_date(history_start),
        "history_end": _format_date(history_end),
        "generated_at_unix": int(time()),
        "columns": columns,
    }


def write_winsorisation_config(config, path):
    """Write a YAML configuration to the explicitly supplied output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def apply_winsorisation(df, config) -> pd.DataFrame:
    """Return a copy with configured columns clipped to their stored bounds."""
    result = df.copy()
    for column, bounds in config["columns"].items():
        if column in result.columns:
            result[column] = result[column].clip(
                lower=bounds["lower"], upper=bounds["upper"],
            )
    return result


def save_anomaly_outputs(table, flagged, config, source, brand_id) -> dict[str, Path]:
    """Write the anomaly table, the flagged rows and the winsorisation YAML for one source."""
    suffix = f"{source}_brand{brand_id}"
    paths = {
        "table": PROJECT_ROOT / f"data/03_output/anomaly_table_{suffix}.csv",
        "flags": PROJECT_ROOT / f"data/02_intermediate/anomaly_flags_{suffix}.parquet",
        # The features YAML is the one build_feature_store() reads.
        "config": (Path(str(WINSOR_CONFIG_PATH).format(brand_id=brand_id)) if source == "features"
                   else PROJECT_ROOT / f"configs/winsorisation_{suffix}.yaml"),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths["table"], index=False)
    flagged.to_parquet(paths["flags"], index=False)
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
        frame, spec, brand_id, history_start, history_end
    )
    flag_frame = pd.DataFrame(flags, index=frame.index).fillna(False).astype(bool)
    frame = pd.concat(
        [frame.drop(columns=list(flags) + ["flag_any"], errors="ignore"), flag_frame],
        axis=1,
    )
    frame["flag_any"] = flag_frame.any(axis=1)
    return build_anomaly_table(rows), frame, config




def _extract_operator(con, operator: str, brand_id: int, days: list) -> pd.DataFrame:
    """Bets, completed payments and bonus events per player-day, in local currency."""
    bet_paths = _landing_paths(operator, LANDING_TABLES["bet"], days)
    bets = con.sql(f"""
        SELECT partyId, CAST(dateTime AS DATE) AS activity_date, currency,
               COUNT(*) FILTER (WHERE tranType = 'GAME_BET') AS n_bets,
               -SUM(CASE WHEN tranType = 'GAME_BET' THEN amountReal ELSE 0 END) AS stake_local,
               SUM(CASE WHEN tranType = 'GAME_WIN' THEN amountReal ELSE 0 END) AS win_local
        FROM read_parquet({bet_paths}, union_by_name=True)
        WHERE brandId = {brand_id} AND partyId IS NOT NULL AND dateTime IS NOT NULL
          AND rolledBack = False
        GROUP BY ALL
    """).df()
    dominant_currency = bets["currency"].mode().iloc[0]


    txn_paths = _landing_paths(operator, LANDING_TABLES["transaction"], days)
    con.sql(f"CREATE OR REPLACE VIEW txn AS SELECT * FROM read_parquet({txn_paths}, union_by_name=True)")
    currency_expr = "currency" if "currency" in con.sql("SELECT * FROM txn LIMIT 0").columns else "NULL::VARCHAR"
    txns = con.sql(f"""
        SELECT partyId, CAST(requestDate AS DATE) AS activity_date, {currency_expr} AS currency,
               COUNT(*) FILTER (WHERE transactionType = 'DEPOSIT') AS n_deposits,
               COALESCE(SUM(processedAmount) FILTER (WHERE transactionType = 'DEPOSIT'), 0) AS deposit_local,
               COUNT(*) FILTER (WHERE transactionType = 'WITHDRAWAL') AS n_withdrawals,
               COALESCE(SUM(processedAmount) FILTER (WHERE transactionType = 'WITHDRAWAL'), 0) AS withdrawal_local
        FROM txn
        WHERE status = 'COMPLETED' AND brandId = {brand_id}
          AND partyId IS NOT NULL AND requestDate IS NOT NULL
        GROUP BY ALL
    """).df()
    txns["currency"] = txns["currency"].fillna(dominant_currency)

    player_paths = _landing_paths(operator, LANDING_TABLES["player"], days)
    regdates = con.sql(f"""
        SELECT partyId, MIN(CAST(regdate AS DATE)) AS regdate
        FROM read_parquet({player_paths}, union_by_name=True)
        WHERE brandId = {brand_id} AND partyId IS NOT NULL
        GROUP BY partyId
    """).df()


    con.register("brand_players", regdates[["partyId"]])
    bonus_paths = _landing_paths(operator, LANDING_TABLES["bonus"], days)
    bonuses = con.sql(f"""
        SELECT b.partyId, CAST(b.changeStatusTimestamp AS DATE) AS activity_date,
               COUNT(*) AS n_bonus_events,
               COALESCE(SUM(b.amount) FILTER (WHERE b.status = 'ACTIVE'), 0) AS bonus_local
        FROM read_parquet({bonus_paths}, union_by_name=True) AS b
        JOIN brand_players AS p USING (partyId)
        WHERE b.changeStatusTimestamp IS NOT NULL
        GROUP BY ALL
    """).df()
    con.unregister("brand_players")
    bonuses["currency"] = dominant_currency

    fx_rates = load_fx_rates_eur(FX_RATES_CSV, HISTORY_START, data_available_through())
    parts = []
    for df, amounts in [
        (bets, {"stake_local": "stake_eur", "win_local": "win_eur"}),
        (txns, {"deposit_local": "deposit_eur", "withdrawal_local": "withdrawal_eur"}),
        (bonuses, {"bonus_local": "bonus_eur"}),
    ]:
        df["activity_date"] = pd.to_datetime(df["activity_date"])

        df = df[df["activity_date"].between(pd.Timestamp(HISTORY_START), pd.Timestamp(data_available_through()))].copy()
        for local, eur in amounts.items():
            df[eur] = convert_amount_to_eur(df, local, "currency", "activity_date", fx_rates)
        keep = [c for c in df.columns if c.startswith("n_")] + list(amounts.values())
        parts.append(df.groupby(["partyId", "activity_date"])[keep].sum())

    player_day = pd.concat(parts, axis=1).fillna(0).reset_index()
    count_cols = [c for c in player_day.columns if c.startswith("n_")]
    player_day[count_cols] = player_day[count_cols].astype("int64")
    player_day["ggr_eur"] = player_day["stake_eur"] - player_day["win_eur"]
    regdates["regdate"] = pd.to_datetime(regdates["regdate"])
    player_day = player_day.merge(regdates, on="partyId", how="left")
    player_day.insert(0, "operator", operator)
    player_day.insert(1, "brandId", brand_id)
    return player_day


def load_player_day(brand_id: int, use_cache: bool = True) -> pd.DataFrame:
    """Player-day table for one brand, from cache when available, else from S3."""
    end = data_available_through()
    cache = Path(str(PLAYER_DAY_CACHE).format(brand_id=brand_id, end=end))
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    days = _date_range(HISTORY_START, end)
    operators = _resolve_operators_for_brand(brand_id, end)
    con = _s3_duckdb()
    try:
        player_day = pd.concat(
            [_extract_operator(con, op, brand_id, days) for op in operators], ignore_index=True
        )
    finally:
        con.close()
    cache.parent.mkdir(parents=True, exist_ok=True)
    player_day.to_parquet(cache, index=False)
    return player_day
