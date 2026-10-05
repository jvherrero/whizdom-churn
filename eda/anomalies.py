"""Run the anomaly study on one table and write its outputs.

    .venv/bin/python eda/anomalies.py --source landing
    .venv/bin/python eda/anomalies.py --source features
    .venv/bin/python eda/anomalies.py --source features --cutoff-dates 2026-07-18 2026-07-25
    .venv/bin/python eda/anomalies.py --source features --brand-id basel   # one study per brand
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
from anomalies import FEATURES_SPEC, LANDING_SPEC, detect_anomalies, load_player_day, save_anomaly_outputs
from build_features import (
    ALL_BRANDS, DATA_AVAILABLE_THROUGH, HISTORY_START, _discover_brand_ids, build_feature_store, parse_brand_id,
)
from build_training_features import DEFAULT_CUTOFF_DATES

SPECS = {"landing": LANDING_SPEC, "features": FEATURES_SPEC}


def load_table(source: str, brand_id: int | str, cutoff_dates: list[str], input_path: str | None, use_cache: bool):
    """The table to study: a parquet given on the command line, or built for the source."""
    if input_path:
        return pd.read_parquet(input_path)
    if source == "landing":
        brands = ([b for _, b in _discover_brand_ids(DATA_AVAILABLE_THROUGH)] if brand_id == ALL_BRANDS
                  else [brand_id])
        return pd.concat([load_player_day(b, use_cache) for b in sorted(set(brands))], ignore_index=True)
    # Unscaled and unwinsorised on purpose: caps and IQR/MAD must be in raw EUR.
    return pd.concat(
        [build_feature_store(c, brand_id, use_cache=use_cache, full=True, scaled=False, winsorise=False)
         for c in cutoff_dates],
        ignore_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=SPECS, required=True)
    parser.add_argument("--brand-id", type=parse_brand_id, default=64, help="brandId, or 'basel' for every brand")
    parser.add_argument("--cutoff-dates", nargs="+", default=DEFAULT_CUTOFF_DATES)
    parser.add_argument("--input", help="parquet to study instead of building the table (features must be unscaled)")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    start = time()
    spec = SPECS[args.source]
    df = load_table(args.source, args.brand_id, args.cutoff_dates, args.input, not args.no_cache)
    print(f"{spec.name}: {len(df):,} rows, {df['partyId'].nunique():,} players")

    history = (HISTORY_START, DATA_AVAILABLE_THROUGH) if spec is LANDING_SPEC else (None, None)
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
