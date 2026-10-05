"""Run the anomaly study on one table and write its outputs.

    .venv/bin/python eda/anomalies.py --source landing
    .venv/bin/python eda/anomalies.py --source features
    .venv/bin/python eda/anomalies.py --source features --cutoff-dates 2026-07-18 2026-07-25
    .venv/bin/python eda/anomalies.py --source features --input some_unscaled_features.parquet

The detection logic lives in src/features/anomalies.py, so the feature pipeline
can call detect_anomalies(df, FEATURES_SPEC) directly; this script only picks
the table, runs it and saves:
    data/03_output/anomaly_table_{source}_brand{brand_id}.csv
    data/02_intermediate/anomaly_flags_{source}_brand{brand_id}.parquet
    configs/winsorisation_{source}_brand{brand_id}.yaml
"""

import argparse
import sys
from pathlib import Path
from time import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
from anomalies import (
    FEATURES_SPEC, LANDING_SPEC, detect_anomalies, load_player_day, write_winsorisation_config,
)
from build_features import DATA_AVAILABLE_THROUGH, HISTORY_START, build_feature_store
from build_training_features import DEFAULT_CUTOFF_DATES

SPECS = {"landing": LANDING_SPEC, "features": FEATURES_SPEC}


def load_table(source: str, brand_id: int, cutoff_dates: list[str], input_path: str | None, use_cache: bool):
    """The table to study: a parquet given on the command line, or built for the source."""
    if input_path:
        return pd.read_parquet(input_path)
    if source == "landing":
        return load_player_day(brand_id, use_cache)
    # Unscaled on purpose: caps and IQR/MAD must be in EUR, not on the sign-log scale.
    return pd.concat(
        [build_feature_store(c, brand_id, use_cache=use_cache, full=True, scaled=False) for c in cutoff_dates],
        ignore_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=SPECS, required=True)
    parser.add_argument("--brand-id", type=int, default=64)
    parser.add_argument("--cutoff-dates", nargs="+", default=DEFAULT_CUTOFF_DATES)
    parser.add_argument("--input", help="parquet to study instead of building the table (features must be unscaled)")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    start = time()
    spec = SPECS[args.source]
    df = load_table(args.source, args.brand_id, args.cutoff_dates, args.input, not args.no_cache)
    print(f"{spec.name}: {len(df):,} rows, {df['partyId'].nunique():,} players")

    history = (HISTORY_START, DATA_AVAILABLE_THROUGH) if spec is LANDING_SPEC else (None, None)
    table, flagged, config = detect_anomalies(df, spec, *history, brand_id=args.brand_id)

    suffix = f"{spec.name}_brand{args.brand_id}"
    table_path = PROJECT_ROOT / f"data/03_output/anomaly_table_{suffix}.csv"
    flags_path = PROJECT_ROOT / f"data/02_intermediate/anomaly_flags_{suffix}.parquet"
    config_path = PROJECT_ROOT / f"configs/winsorisation_{suffix}.yaml"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    flagged.to_parquet(flags_path, index=False)
    write_winsorisation_config(config, config_path)

    print(table[["column", "method", "n_evaluated", "n_flagged", "share"]].to_string(index=False))
    for path in (table_path, flags_path, config_path):
        print(f"saved {path.relative_to(PROJECT_ROOT)}")
    print(f"execution time: {time() - start:.1f}s")


if __name__ == "__main__":
    main()
