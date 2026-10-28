"""The priority list: the highly active games (output/active_games.csv, 99% of the GGR) that are not already classified
in games_enriched.xlsx, sorted by GGR and then rounds.

    .venv/bin/python catalog_project/priority_list.py
"""

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import slotcatalog_json

HERE = Path(__file__).parent
OUT = HERE / "output"
CA_COLUMNS = {"Game type": "ca_game_type", "Provider": "ca_provider", "Theme": "ca_theme", "Volatility": "ca_volatility",
              "RTP": "ca_rtp", "Game slug": "ca_slug",
              "External Resource": "ca_external_resource"}
COLUMNS = ["rank", "game_uid", "game_id", "provider_code", "provider_name", "game_name", "status",
           "is_sports", "ggr_eur_6m", "share_of_ggr", "cumulative_share_of_ggr", "rounds_6m", "share_of_rounds",
           "wagered_eur_6m", "unique_players_30d", "unique_players_90d", "unique_players_lifetime", "stickiness_30d",
           "stickiness_90d", "days_since_launch", "vertical", "game_type", "theme", "sc_url", "sc_url_source",
           "ca_fills_game_type", "ca_game_type", "ca_provider", "ca_theme", "ca_volatility", "ca_rtp",
           "ca_slug", *[c for c in slotcatalog_json.COLUMNS if c not in ("sc_uuid", "sc_extracted_at")]]   # sc_uuid only joins the JSON


def key(ids: pd.Series, provider: pd.Series) -> pd.Series:
    """normalized id|provider code."""
    return ids.fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True) + "|" + provider.fillna("").astype(str)


def by_key(table: pd.DataFrame, id_columns: list[str], provider: str, columns: list[str]) -> pd.DataFrame:
    """`columns` of the table by key, one row per key: the first id column has priority, then the first row."""
    stacked = pd.concat([table.assign(key=key(table[c], table[provider])) for c in id_columns])
    return stacked.drop_duplicates("key").set_index("key")[columns]


def build(active: pd.DataFrame, sa: pd.DataFrame, ca: pd.DataFrame, enriched: pd.DataFrame,
          sports_codes: list[str] = (), info: pd.DataFrame | None = None) -> pd.DataFrame:
    """The active games with their URL, what the Client Area has, if they are already classified (games_enriched),
    if they are in the work list (not classified) and if they are sports betting. `info` is the table of the raw
    Slot Catalog JSON (slotcatalog_json.py), joined by the UUID that the SA file gives to each game."""
    sa_urls = by_key(sa[sa["sc_url"].notna()], ["game_src_id", "core_id_desktop"], "platform", ["sc_url", "game_id"])
    sa_urls = sa_urls.rename(columns={"game_id": "sc_uuid"})
    ca_values = by_key(ca, ["Core Id Desktop", "Core Id Mobile"], "Platform", list(CA_COLUMNS)).rename(columns=CA_COLUMNS)
    games = active.assign(key=key(active["game_id"], active["provider_code"])).join(sa_urls, on="key").join(ca_values, on="key")
    games["ca_found"] = games["key"].isin(ca_values.index)
    if info is not None:
        games = games.join(info.set_index("sc_uuid"), on="sc_uuid")
        # third source of the launch date, after release_date and first_seen of ref_games (active_games.py)
        launch = pd.to_datetime(games["sc_release_date"], format="%d/%m/%Y", errors="coerce")
        days = games.get("days_since_launch", pd.Series(np.nan, index=games.index))
        games["days_since_launch"] = days.fillna((pd.Timestamp.today().normalize() - launch).dt.days)

    ca_url = games["ca_external_resource"].where(games["ca_external_resource"].astype(str).str.contains("slotcatalog"))
    games["sc_url_source"] = np.select([games["sc_url"].notna(), ca_url.notna()], ["slotcatalog_file", "client_area"], "")
    games["sc_url"] = games["sc_url"].fillna(ca_url)

    no_type = games["game_type"].fillna("") == ""
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
    no_url, no_type = games["sc_url"].isna(), games["game_type"].fillna("") == ""
    groups = {
        "highly active games (99% of the GGR)": games["game_uid"].notna(),
        "with Slot Catalog URL from the slotcatalog file": games["sc_url_source"] == "slotcatalog_file",
        "with Slot Catalog URL from the Client Area only": games["sc_url_source"] == "client_area",
        "found in the Client Area file": games["ca_found"],
        "already classified (in games_enriched)": games["in_games_enriched"],
        "IN THE PRIORITY LIST": listed,
        "  with Slot Catalog URL and game type": listed & ~no_url & ~no_type,
        "  no Slot Catalog URL": listed & no_url,
        "  no game type": listed & no_type,
        "  no Slot Catalog URL and no game type": listed & no_url & no_type,
        "  sports betting (is_sports)": listed & games["is_sports"],
        "  Client Area has the missing game type": listed & games["ca_fills_game_type"],
        "  Client Area has a theme": listed & games["ca_theme"].notna(),
        "  Client Area has volatility": listed & games["ca_volatility"].notna(),
        "  Client Area has RTP": listed & games["ca_rtp"].notna(),
        "  with Slot Catalog JSON data": listed & games.get("sc_title", pd.Series(index=games.index, dtype=object)).notna()}
    return pd.DataFrame({
        "group": list(groups), "games": [m.sum() for m in groups.values()],
        "share_of_ggr": [games.loc[m, "ggr_eur_6m"].sum() / games["ggr_eur_6m"].sum() for m in groups.values()],
        "share_of_rounds": [games.loc[m, "rounds_6m"].sum() / games["rounds_6m"].sum() for m in groups.values()]}).round(3)


def column_notes() -> pd.DataFrame:
    """The sheet `columns`: meaning and source of each column of the games sheet (columns.yaml)."""
    notes = yaml.safe_load((HERE / "columns.yaml").read_text())
    return pd.DataFrame([{"column": c, **notes[c]} for c in COLUMNS if c in notes])


def tidy(writer) -> None:
    """Every table sheet: the header row stays visible and every column has a filter. The summary has a block of
    dates above its table, so it is left as it is."""
    for name, sheet in writer.sheets.items():
        if name != "summary":
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions


def data_dates() -> pd.DataFrame:
    """When the data were made: the activity window (the same as active_games.sql) and the date of the active file."""
    today = pd.Timestamp.today().normalize()
    built = pd.Timestamp(os.path.getmtime(OUT / "active_games.csv"), unit="s").normalize()
    return pd.DataFrame({"item": ["list generated on", "activity window (6 full months)", "active games file built on"],
                         "value": [today.date(), f"{(today - pd.DateOffset(months=6)).date()} to {(today.date() - dt.timedelta(days=1))}",
                                   built.date()]})


def link_urls(sheet, column: str = "sc_url") -> None:
    """Make the cells of `column` clickable links (the text stays the same)."""
    col = [c.value for c in sheet[1]].index(column) + 1
    for row in range(2, sheet.max_row + 1):
        cell = sheet.cell(row=row, column=col)
        if cell.value:
            cell.hyperlink = cell.value
            cell.style = "Hyperlink"


def main() -> None:
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    active = pd.read_csv(OUT / "active_games.csv", dtype={"game_id": str}, keep_default_na=False, na_values=[""])
    sa = pd.read_excel(HERE / config["sa_urls"], dtype=str)
    ca = pd.read_csv(HERE / config["client_area"], dtype={"Core Id Desktop": str, "Core Id Mobile": str})
    enriched = pd.read_excel(HERE / config["games_enriched"], dtype=str)

    info = slotcatalog_json.load(HERE / config["slotcatalog_json"])
    games = build(active, sa, ca, enriched, config["sports_provider_codes"], info)
    result, table, by_provider = priority(games), summary(games), providers(games)

    for flag in ["is_sports", "ca_fills_game_type"]:     # true/false as text, so Excel does not translate them
        result[flag] = result[flag].map({True: "true", False: "false"})
    with pd.ExcelWriter(OUT / "priority_games.xlsx") as writer:
        result.to_excel(writer, sheet_name="games", index=False)
        link_urls(writer.sheets["games"])
        dates = data_dates()
        dates.to_excel(writer, sheet_name="summary", index=False)
        table.to_excel(writer, sheet_name="summary", index=False, startrow=len(dates) + 2)
        by_provider.to_excel(writer, sheet_name="providers", index=False)
        column_notes().to_excel(writer, sheet_name="columns", index=False)
        tidy(writer)
    print(table.to_string(index=False))
    print("\nthe 15 providers with most GGR:")
    print(by_provider.head(15).to_string(index=False))
    print(f"\nwrote {(OUT / 'priority_games.xlsx').relative_to(HERE.parent)}: {len(result)} games")


if __name__ == "__main__":
    main()
