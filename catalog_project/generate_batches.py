"""Phase 1.3: split output/missing_games.csv into equal batches to label by hand (batches and columns in config.yaml).

The games go from most to least rounds, one to each batch in turn, so the sizes differ by one at most and every
batch gets the same mix of big and small games.

    .venv/bin/python catalog_project/generate_batches.py
"""

from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).parent
OUT = HERE / "output"
KEEP = ["game_uid", "game_id", "provider_code", "game_name", "provider_name", "vertical", "game_type", "theme",
        "enrichment_status", "rounds_6m"]


def split(games: pd.DataFrame, names: list[str]) -> dict[str, pd.DataFrame]:
    """The games sorted by rounds (ties by game_uid) dealt to the batches in turn."""
    games = games.sort_values(["rounds_6m", "game_uid"], ascending=[False, True])
    return {name: games.iloc[i::len(names)] for i, name in enumerate(names)}


def main() -> None:
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    missing = pd.read_csv(OUT / "missing_games.csv", dtype=str, keep_default_na=False)
    missing["rounds_6m"] = missing["rounds_6m"].astype(float)
    for name, batch in split(missing[KEEP], config["batches"]).items():
        batch = batch.assign(**{c: "" for c in config["fill_columns"]})
        batch.to_csv(OUT / f"batch_{name}.csv", index=False)
        print(f"batch_{name}.csv: {len(batch)} games, {batch['rounds_6m'].sum():,.0f} rounds, "
              f"{(batch['game_name'] == '').sum()} without name")


if __name__ == "__main__":
    main()
