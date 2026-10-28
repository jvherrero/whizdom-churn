"""Split the games of output/priority_games.xlsx into one file per person (names in config.yaml, splits).

    .venv/bin/python catalog_project/split_batches.py
"""

from pathlib import Path

import pandas as pd
import yaml

from priority_list import column_notes, link_urls, tidy

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
    # dtype=object and no default NaN: the cells stay as they are (`N/A` and text numbers are not converted)
    games = pd.read_excel(OUT / "priority_games.xlsx", sheet_name="games", dtype=object, keep_default_na=False, na_values=[""])
    games["batch"] = assign(games, names)
    for name in names:
        batch = games[games["batch"] == name].drop(columns="batch")
        with pd.ExcelWriter(OUT / f"batch_{name}.xlsx") as writer:
            batch.to_excel(writer, sheet_name="games", index=False)
            link_urls(writer.sheets["games"])
            column_notes().to_excel(writer, sheet_name="columns", index=False)
            tidy(writer)
        print(f"{name}: {len(batch)} games, {batch['provider_name'].nunique()} providers, "
              f"{batch['ggr_eur_6m'].astype(float).sum() / games['ggr_eur_6m'].astype(float).sum():.1%} of the GGR in the list")
    print(f"total: {len(games)} games")


if __name__ == "__main__":
    main()
