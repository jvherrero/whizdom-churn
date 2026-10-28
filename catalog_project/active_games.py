"""The highly active games of the last 6 months: the smallest set with 99% of the positive GGR.

    .venv/bin/python catalog_project/active_games.py
    
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run as a script: the repo root
from signal_audit.data.clickhouse_mirror import client  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "output"


def query(c, name: str, **params) -> pd.DataFrame:
    return c.query_df((HERE / "sql" / f"{name}.sql").read_text(), parameters=params or None)


def active_set(games: pd.DataFrame, total_ggr: float, coverage: float) -> pd.DataFrame:
    """The games with most GGR (then most rounds) that hold `coverage` of the positive total, the one that crosses
    the limit included."""
    games = games[games["ggr_eur_6m"] > 0].sort_values(["ggr_eur_6m", "rounds_6m", "game_uid"], ascending=[False, False, True])
    before = games["ggr_eur_6m"].cumsum() - games["ggr_eur_6m"]
    return games[before < coverage * total_ggr]


def unique_players(c) -> pd.DataFrame:
    """Lifetime players from gold; 30 and 90 day players from the rounds, one query per tenant (the time limit)."""
    tenants = query(c, "tenants")["tenant_id"]
    recent = []
    for i, tenant in enumerate(tenants, 1):
        print(f"unique players of the last 90 days: tenant {i} of {len(tenants)}", flush=True)
        recent.append(query(c, "game_players_90d", tenant=tenant))
    recent = pd.concat(recent).groupby("game_uid", as_index=False).sum()
    return query(c, "game_players_lifetime").merge(recent, on="game_uid", how="outer")


def add_factors(games: pd.DataFrame, today: pd.Timestamp, placeholder: str, data_start: str) -> pd.DataFrame:
    """Stickiness over 30 and 90 days, and the days since launch: release_date, or else first_seen. The placeholder
    release date, and a first_seen on the first day of the data, are not launch dates: the launch is then unknown."""
    for days in (30, 90):
        players = games[f"unique_players_{days}d"].where(games[f"unique_players_{days}d"] > 0)
        games[f"stickiness_{days}d"] = games[f"daily_players_{days}d"] / days / players
    release = pd.to_datetime(games["release_date"]).where(lambda d: d != pd.Timestamp(placeholder))
    first_seen = pd.to_datetime(games["first_seen"]).where(lambda d: d > pd.Timestamp(data_start))
    games["days_since_launch"] = (today - release.fillna(first_seen)).dt.days
    return games


def main() -> None:
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    c = client()
    activity = query(c, "active_games", months=config["months"])
    resolved = activity[activity["game_uid"] != ""]
    total_ggr = activity["ggr_eur_6m"].clip(lower=0).sum()

    active = (active_set(resolved, total_ggr, config["coverage"])
              .merge(unique_players(c), on="game_uid", how="left")
              .merge(query(c, "ref_games"), on="game_uid", how="left"))
    active = add_factors(active, pd.Timestamp.today().normalize(), str(config["placeholder_release_date"]), str(config["data_start"]))
    OUT.mkdir(exist_ok=True)
    active.to_csv(OUT / "active_games.csv", index=False)

    print(f"positive GGR of the window: {total_ggr:,.0f} EUR; games with GGR 0 or below: {(resolved['ggr_eur_6m'] <= 0).sum():,}")
    print(f"highly active games: {len(active):,} ({config['coverage']:.0%} of the positive GGR)")
    print(f"wrote {OUT.relative_to(HERE.parent)}/active_games.csv")


if __name__ == "__main__":
    main()
