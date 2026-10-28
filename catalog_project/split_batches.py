"""Split the games of output/priority_games.xlsx into one file per person (names in config.yaml, splits).

    .venv/bin/python catalog_project/split_batches.py
"""

from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).parent
OUT = HERE / "output"


def assign(games: pd.DataFrame, names: list[str]) -> pd.Series:
    """The batch name of each game, by provider."""
    sizes = {name: 0 for name in names}
    owner = {}
    for provider, count in games["provider_name"].fillna("(no provider)").value_counts().items():
        name = min(sizes, key=sizes.get)
        owner[provider] = name
        sizes[name] += count
    return games["provider_name"].fillna("(no provider)").map(owner)


def main() -> None:
    names = yaml.safe_load((HERE / "config.yaml").read_text())["splits"]
    text = {"game_id": str, "is_sports": str, "ca_found": str, "ca_fills_game_type": str}   # keep true/false as text
    games = pd.read_excel(OUT / "priority_games.xlsx", sheet_name="games", dtype=text)
    games["batch"] = assign(games, names)
    for name in names:
        batch = games[games["batch"] == name].drop(columns="batch")
        batch.to_excel(OUT / f"batch_{name}.xlsx", sheet_name="games", index=False)
        print(f"{name}: {len(batch)} games, {batch['provider_name'].nunique()} providers, "
              f"{batch['ggr_eur_6m'].sum() / games['ggr_eur_6m'].sum():.1%} of the GGR in the list")
    print(f"total: {len(games)} games")


if __name__ == "__main__":
    main()
