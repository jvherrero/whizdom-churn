"""The priority list: the highly active games (output/active_games.csv, 99% of the GGR) that are not already classified
in games_enriched.xlsx, sorted by GGR and then rounds, with their Slot Catalog URL and what the Client
Area already has for them (reason says if the URL or the game type is missing). A sheet by provider shows how much of its GGR is covered. Local only, no ClickHouse.

The three sources join on the normalized id (lower case, letters and digits only, as in ref_games) and the provider
code: game_id + provider_code in ref_games, game_src_id or core_id_desktop + platform in the SA file, Core Id Desktop
or Core Id Mobile + Platform in the Client Area, game_src_id, core_desktop_id or core_mobile_id + platform_code in
games_enriched. The URL comes from sc_url of the SA file, or else from External
Resource of the Client Area when it is a Slot Catalog link.

    .venv/bin/python catalog_project/priority_list.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).parent
OUT = HERE / "output"
CA_COLUMNS = {"Game type": "ca_game_type", "Provider": "ca_provider", "Theme": "ca_theme", "Volatility": "ca_volatility",
              "RTP": "ca_rtp", "Release Date": "ca_release_date", "Game slug": "ca_slug",
              "External Resource": "ca_external_resource"}
COLUMNS = ["rank", "game_uid", "game_id", "provider_code", "provider_name", "game_name", "enrichment_status", "status",
           "reason", "is_sports", "ggr_eur_6m", "share_of_ggr", "cumulative_share_of_ggr", "rounds_6m", "share_of_rounds",
           "wagered_eur_6m", "unique_players_30d", "unique_players_90d", "unique_players_lifetime", "stickiness_30d",
           "stickiness_90d", "days_since_launch", "vertical", "game_type", "theme", "sc_url", "sc_url_source", "ca_found",
           "ca_fills_game_type", "ca_game_type", "ca_provider", "ca_theme", "ca_volatility", "ca_rtp", "ca_release_date",
           "ca_slug"]


def key(ids: pd.Series, provider: pd.Series) -> pd.Series:
    """normalized id|provider code."""
    return ids.fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True) + "|" + provider.fillna("").astype(str)


def by_key(table: pd.DataFrame, id_columns: list[str], provider: str, columns: list[str]) -> pd.DataFrame:
    """`columns` of the table by key, one row per key: the first id column has priority, then the first row."""
    stacked = pd.concat([table.assign(key=key(table[c], table[provider])) for c in id_columns])
    return stacked.drop_duplicates("key").set_index("key")[columns]


def build(active: pd.DataFrame, sa: pd.DataFrame, ca: pd.DataFrame, enriched: pd.DataFrame,
          sports_codes: list[str] = ()) -> pd.DataFrame:
    """The active games with their URL, what the Client Area has, what is missing (reason, '' if nothing), if they
    are already classified (games_enriched), if they are in the work list (not classified) and if they are sports."""
    sa_urls = by_key(sa[sa["sc_url"].notna()], ["game_src_id", "core_id_desktop"], "platform", ["sc_url"])
    ca_values = by_key(ca, ["Core Id Desktop", "Core Id Mobile"], "Platform", list(CA_COLUMNS)).rename(columns=CA_COLUMNS)
    games = active.assign(key=key(active["game_id"], active["provider_code"])).join(sa_urls, on="key").join(ca_values, on="key")
    games["ca_found"] = games["key"].isin(ca_values.index)

    ca_url = games["ca_external_resource"].where(games["ca_external_resource"].astype(str).str.contains("slotcatalog"))
    games["sc_url_source"] = np.select([games["sc_url"].notna(), ca_url.notna()], ["sa_file", "client_area"], "")
    games["sc_url"] = games["sc_url"].fillna(ca_url)

    no_url, no_type = games["sc_url"].isna(), games["game_type"].fillna("") == ""
    games["reason"] = np.select([no_url & no_type, no_url, no_type], ["no_sc_url+no_game_type", "no_sc_url", "no_game_type"], "")
    games["ca_fills_game_type"] = no_type & games["ca_game_type"].notna()

    enriched_keys = pd.concat([key(enriched[c], enriched["platform_code"]) for c in ["game_src_id", "core_desktop_id", "core_mobile_id"]])
    games["in_games_enriched"] = games["key"].isin(enriched_keys)
    games["in_list"] = ~games["in_games_enriched"]
    games["is_sports"] = games["provider_code"].isin(sports_codes)
    return games


def priority(games: pd.DataFrame) -> pd.DataFrame:
    """The games of the work list, sorted by GGR and then rounds, with their share of the GGR and of the rounds of all
    the active games."""
    listed = games[games["in_list"]].sort_values(["ggr_eur_6m", "rounds_6m", "game_uid"], ascending=[False, False, True])
    listed = listed.assign(rank=range(1, len(listed) + 1), share_of_ggr=listed["ggr_eur_6m"] / games["ggr_eur_6m"].sum(),
                           share_of_rounds=listed["rounds_6m"] / games["rounds_6m"].sum())
    listed["cumulative_share_of_ggr"] = listed["share_of_ggr"].cumsum()
    return listed[[c for c in COLUMNS if c in listed]]


def providers(games: pd.DataFrame) -> pd.DataFrame:
    """By provider_name: active games, their GGR, how many are in the list and how much of their GGR is covered."""
    out = games.assign(ggr_in_list=games["ggr_eur_6m"].where(games["in_list"], 0)).groupby("provider_name").agg(
        active_games=("game_uid", "count"), ggr_eur_6m=("ggr_eur_6m", "sum"),
        games_in_list=("in_list", "sum"), ggr_in_list=("ggr_in_list", "sum"))
    out["share_of_ggr"] = out["ggr_eur_6m"] / games["ggr_eur_6m"].sum()
    out["covered_share_of_ggr"] = 1 - out["ggr_in_list"] / out["ggr_eur_6m"]
    out = out.sort_values("ggr_eur_6m", ascending=False).reset_index()
    return out.round({"ggr_eur_6m": 0, "ggr_in_list": 0, "share_of_ggr": 3, "covered_share_of_ggr": 3})


def summary(games: pd.DataFrame) -> pd.DataFrame:
    """How many games, and what share of the GGR and of the rounds, in each group."""
    listed = games["in_list"]
    groups = {
        "highly active games (99% of the GGR)": games["game_uid"].notna(),
        "with Slot Catalog URL from the SA file": games["sc_url_source"] == "sa_file",
        "with Slot Catalog URL from the Client Area only": games["sc_url_source"] == "client_area",
        "found in the Client Area file": games["ca_found"],
        "already classified (in games_enriched)": games["in_games_enriched"],
        "IN THE PRIORITY LIST": listed,
        "  with Slot Catalog URL and game type": listed & (games["reason"] == ""),
        "  no Slot Catalog URL only": listed & (games["reason"] == "no_sc_url"),
        "  no game type only": listed & (games["reason"] == "no_game_type"),
        "  no Slot Catalog URL and no game type": listed & (games["reason"] == "no_sc_url+no_game_type"),
        "  sports betting (is_sports)": listed & games["is_sports"],
        "  Client Area has the missing game type": listed & games["ca_fills_game_type"],
        "  Client Area has a theme": listed & games["ca_theme"].notna(),
        "  Client Area has volatility": listed & games["ca_volatility"].notna(),
        "  Client Area has RTP": listed & games["ca_rtp"].notna()}
    return pd.DataFrame({
        "group": list(groups), "games": [m.sum() for m in groups.values()],
        "share_of_ggr": [games.loc[m, "ggr_eur_6m"].sum() / games["ggr_eur_6m"].sum() for m in groups.values()],
        "share_of_rounds": [games.loc[m, "rounds_6m"].sum() / games["rounds_6m"].sum() for m in groups.values()]}).round(3)


def main() -> None:
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    active = pd.read_csv(OUT / "active_games.csv", dtype={"game_id": str}, keep_default_na=False, na_values=[""])
    sa = pd.read_excel(HERE / config["sa_urls"], dtype=str)
    ca = pd.read_csv(HERE / config["client_area"], dtype={"Core Id Desktop": str, "Core Id Mobile": str})
    enriched = pd.read_excel(HERE / config["games_enriched"], dtype=str)

    games = build(active, sa, ca, enriched, config["sports_provider_codes"])
    games["ca_release_date"] = games["ca_release_date"].str[:10]
    result, table, by_provider = priority(games), summary(games), providers(games)

    with pd.ExcelWriter(OUT / "priority_games.xlsx") as writer:
        result.to_excel(writer, sheet_name="games", index=False)
        table.to_excel(writer, sheet_name="summary", index=False)
        by_provider.to_excel(writer, sheet_name="providers", index=False)
    print(table.to_string(index=False))
    print("\nthe 15 providers with most GGR:")
    print(by_provider.head(15).to_string(index=False))
    print(f"\nwrote {(OUT / 'priority_games.xlsx').relative_to(HERE.parent)}: {len(result)} games")


if __name__ == "__main__":
    main()
