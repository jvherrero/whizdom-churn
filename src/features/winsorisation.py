"""Winsorisation caps of the monetary features: what the pipeline needs from the anomaly study.

Heavy-tailed amounts are capped, never deleted: p99.5 of each monetary column (and p0.5 for
signed ones), computed on the non-zero values of the training snapshots, per brand, in EUR.

    configs = write_feature_caps(raw)      # raw = raw_features() rows of the training cutoffs
    capped = winsorise(features)           # each brand's YAML applied, before the sign-log

The full anomaly study (IQR / MAD, Isolation Forest, temporal and structural checks) is EDA:
eda/anomalies.py. It writes the same YAML through build_winsorisation_config().
"""

from __future__ import annotations

from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINSOR_CONFIG_PATH = PROJECT_ROOT / "configs/winsorisation_features_brand{brand_id}.yaml"
WINSOR_UPPER_Q = 0.995
WINSOR_LOWER_Q = 0.005

# The monetary features (EUR) of docs/feature_dictionary.md.
MONETARY_FEATURES = (
    "wagered_eur_l7d", "wagered_eur_l30d", "wagered_eur_l90d", "net_loss_7d", "ggr_eur_l30d",
    "deposits_eur_l30d", "prior_wagered_eur_l7d", "prior_ggr_eur_l30d", "avg_bet_eur_l30d",
)
# Monetary features that can be negative (the player won): capped at both ends.
SIGNED_MONETARY_FEATURES = frozenset({"net_loss_7d", "ggr_eur_l30d", "prior_ggr_eur_l30d"})


def _format_date(value):
    return None if value is None or pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def build_winsorisation_config(df, monetary_columns, signed_columns, brand_id, source, grain, applies_to,
                               history_start, history_end) -> dict:
    """Caps of every monetary column present in `df`, on its non-zero values (signed columns: != 0,
    the others: > 0). An empty column gets infinite (no-op) caps."""
    columns = {}
    for column in [c for c in dict.fromkeys(monetary_columns) if c in df.columns]:
        s = df[column]
        values = s[(s.notna() & (s.ne(0) if column in signed_columns else s.gt(0))).fillna(False)]
        n = len(values)
        upper = round(float(values.quantile(WINSOR_UPPER_Q)), 4) if n else float("inf")
        lower = None
        if column in signed_columns:
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
        "source": source,
        "grain": grain,
        "applies_to": applies_to,
        "upper_quantile": WINSOR_UPPER_Q,
        "lower_quantile_signed_only": WINSOR_LOWER_Q,
        "computed_on": "non-zero values (non-negative columns: > 0, signed columns: != 0)",
        "history_start": _format_date(history_start),
        "history_end": _format_date(history_end),
        "generated_at_unix": int(time()),
        "columns": columns,
    }


def write_winsorisation_config(config: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def feature_caps(raw: pd.DataFrame, brand_id: int) -> dict:
    """The winsorisation config of one brand's raw feature rows (raw_features(), several cutoffs)."""
    dates = pd.to_datetime(raw["cutoff_date"])
    return build_winsorisation_config(
        raw, MONETARY_FEATURES, SIGNED_MONETARY_FEATURES, brand_id, source="features", grain="player_cutoff",
        applies_to="feature values before sign-log scaling", history_start=dates.min(), history_end=dates.max(),
    )


def write_feature_caps(raw: pd.DataFrame) -> list[Path]:
    """One YAML per brand (configs/winsorisation_features_brand{id}.yaml): caps are never pooled
    across brands."""
    return [write_winsorisation_config(feature_caps(rows, int(brand)),
                                       str(WINSOR_CONFIG_PATH).format(brand_id=int(brand)))
            for brand, rows in raw.groupby("brandId")]


def winsorise(df: pd.DataFrame) -> pd.DataFrame:
    """Clip the monetary columns (in EUR, before scaling) to each brand's caps."""
    df = df.copy()
    for brand_id, rows in df.groupby("brandId").groups.items():
        path = Path(str(WINSOR_CONFIG_PATH).format(brand_id=brand_id))
        if not path.exists():
            raise FileNotFoundError(f"{path} not found, create it with: make anomalies-features BRAND={brand_id}")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for col, bounds in config["columns"].items():
            if col in df.columns:
                df.loc[rows, col] = df.loc[rows, col].clip(lower=bounds["lower"], upper=bounds["upper"])
    return df
