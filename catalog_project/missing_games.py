"""Phase 1.2: the games with 99% of the rounds of the last 6 months that are not enriched yet.

The activity (slv_game_daily) is aggregated by game in ClickHouse and joined to the catalog (ref_games) through
the resolution table. A game is highly active if it is in the smallest set that holds `coverage` of the rounds.
It is labelled if its enrichment_status is 'enriched'. Read only; the files go to output/.

    .venv/bin/python catalog_project/missing_games.py
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run as a script: the repo root
from signal_audit.data.clickhouse_mirror import client  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "output"


def active_set(games: pd.DataFrame, total_rounds: float, coverage: float) -> pd.DataFrame:
    """The games with most rounds that hold `coverage` of the total, the one that crosses the limit included."""
    games = games.sort_values(["rounds_6m", "game_uid"], ascending=[False, True])
    before = games["rounds_6m"].cumsum() - games["rounds_6m"]
    return games[before < coverage * total_rounds]


def main() -> None:
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    c = client()
    sql = lambda name: (HERE / "sql" / f"{name}.sql").read_text()
    activity = c.query_df(sql("active_games"), parameters={"months": config["months"]})
    catalog = c.query_df(sql("ref_games"))
    total = activity["rounds_6m"].sum()
    resolved = activity[activity["game_uid"] != ""]

    active = active_set(resolved, total, config["coverage"]).merge(catalog, on="game_uid", how="left")
    active["enrichment_status"] = active["enrichment_status"].fillna("not_in_ref_games")
    missing = active[active["enrichment_status"] != "enriched"]

    OUT.mkdir(exist_ok=True)
    active.to_csv(OUT / "active_games.csv", index=False)
    missing.to_csv(OUT / "missing_games.csv", index=False)

    labelled = active.loc[active["enrichment_status"] == "enriched", "rounds_6m"].sum()
    print(f"rounds in the window: {total:,.0f}; do not resolve to a game: {1 - resolved['rounds_6m'].sum() / total:.4%}")
    print(f"highly active games: {len(active):,} of {len(resolved):,} with rounds ({config['coverage']:.0%} of the rounds)")
    print(f"labelled (enriched): {len(active) - len(missing):,}; missing: {len(missing):,}")
    print(f"share of the rounds of the active games already labelled: {labelled / active['rounds_6m'].sum():.2%}")
    print("\nmissing by enrichment_status and status:")
    print(missing.groupby(["enrichment_status", "status"], dropna=False)
          .agg(games=("game_uid", "count"), rounds_6m=("rounds_6m", "sum")).to_string())
    print(f"\nwrote {OUT.relative_to(HERE.parent)}/active_games.csv and missing_games.csv")


if __name__ == "__main__":
    main()
