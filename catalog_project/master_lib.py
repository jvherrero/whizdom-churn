"""The master catalog of games: one Whizdom Game ID per game, and the functions master.ipynb uses to build it.

A game is its name plus its provider (the studio that made it, not the platform that sells it). Both are written as a
slug, the provider goes through the rebranding dictionary (providers.yaml), and the ID is
uuid5(NAMESPACE, "<provider slug>/<game slug>"): the same game and provider always give the same ID. The ID depends on
slugify() and on providers.yaml, so a change in either can change IDs (ID_VERSION); the notebook reports any change.
"""

import difflib
import json
import re
import unicodedata
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment

HERE = Path(__file__).parent
ID_VERSION = 1
NAMESPACE = uuid.UUID("907a5265-b739-542f-8906-b9b0c91432a8")   # uuid5(NAMESPACE_DNS, "whizdom.ai"), fixed for ever
SYMBOLS = re.compile(r"[™®©]")                                   # NFKD would turn ™ into "TM"
APOSTROPHES = re.compile(r"['’`´]")                              # "Joker's" and "Jokers" are the same name
# The source of each column of the catalog, by its prefix; a column with no prefix is made by master.ipynb.
SOURCES = {"ref_": "ClickHouse ref_games", "ca_": "Client Area export", "sa_": "Slot Catalog URL file of SA",
           "sc_": "raw Slot Catalog JSON", "ge_": "games_enriched.xlsx"}
# How the columns of each source are made (the prefix tells the source). {original} is the column or field of the source.
HOW = {"ref_": "Copied as it is from ClickHouse (SELECT * FROM ref_games FINAL). One row per game_uid; the catalog shows the row of the best match of the ID.",
       "ca_": "Copied as it is from the Client Area export, column '{original}'. Joined to ref_games by key = game id normalized + platform code: Core Id Desktop first, then Core Id Mobile.",
       "sa_": "Copied as it is from the Slot Catalog URL file of SA, column '{original}'. Joined by key = game id normalized + platform: game_src_id first, then core_id_desktop.",
       "ge_": "Copied as it is from games_enriched, column '{original}'. Joined by key = game id normalized + platform_code: game_src_id, core_desktop_id, core_mobile_id.",
       "sc_": "Field {original} of the raw Slot Catalog JSON, cleaned (the page marks [ i ] are removed). The JSON is the one found by the UUID of the SA file, else by name + provider."}
SC_EXTRA = {"sc_title": " The Play at the start of the title is removed.",
            "sc_rtp": " Converted to a number; a value outside 50 to 100 is an error of the page and is left empty.",
            **dict.fromkeys(["sc_theme", "sc_features", "sc_tags", "sc_technology"], " Several values are written as a, b, c.")}
STUDIO_RANK = {"ref_games": 0, "client_area": 1, "slotcatalog_json": 2, "platform": 3}   # where the studio of a game came from
# Ids of a platform, a brand or a source (the name of a source column without its prefix): the catalog leaves them out.
PLATFORM_ID = re.compile(r"^(game_uid|game_id|game_src_id|id|uuid|catalog_id|external_game_id|game_platform_id|game_record_seq"
                         r"|provider_id|brand_id)$|core_.*id|launch_id|start_id")


def slugify(text) -> str:
    """Lower case, no accents, no symbols, letters and digits joined by one hyphen. '' if there is no text."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = APOSTROPHES.sub("", SYMBOLS.sub("", str(text)))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return "-".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def load_providers(path: Path = HERE / "providers.yaml") -> dict:
    """Rebranding dictionary as {alias slug: standard provider slug}. A standard name is also its own alias."""
    aliases = {}
    for standard, names in (yaml.safe_load(path.read_text()) or {}).items():
        for name in [standard, *(names or [])]:
            aliases[slugify(name)] = slugify(standard)
    return aliases


def canonical_provider(names: pd.Series, aliases: dict) -> tuple[pd.Series, pd.Series]:
    """The standard slug of each provider name, and whether the dictionary knew it (if not, its own slug)."""
    slugs = names.map(slugify)
    return slugs.map(lambda s: aliases.get(s, s)), slugs.isin(aliases.keys()) | slugs.eq("")


def provider_candidates(names: pd.Series, aliases: dict) -> pd.DataFrame:
    """Provider names that look like the same company without words like 'gaming' or 'casino', and are not yet one
    standard provider in the dictionary. Only a proposal: a person decides what goes into providers.yaml."""
    stop = {"gaming", "games", "game", "casino", "live", "dealer", "studios", "studio", "interactive", "entertainment",
            "group", "ltd", "feed", "the"}
    counts = names.dropna().map(slugify).value_counts()
    table = pd.DataFrame({"slug": counts.index, "rows": counts.to_numpy()})
    # without the common words and without hyphens, so "3 Oaks" and "3Oaks" or "PG Soft" and "PGSoft" meet
    table["core"] = table["slug"].map(lambda s: "".join(w for w in s.split("-") if w not in stop) or s.replace("-", ""))
    table["standard"] = table["slug"].map(lambda s: aliases.get(s, s))
    groups = table.groupby("core").filter(lambda d: d["standard"].nunique() > 1)
    return groups.sort_values(["core", "rows"], ascending=[True, False]).reset_index(drop=True)


def whizdom_id(game_slug: str, provider_slug: str) -> str | None:
    """The Whizdom Game ID of a game; None if the name or the provider is missing."""
    if not game_slug or not provider_slug:
        return None
    return str(uuid.uuid5(NAMESPACE, f"{provider_slug}/{game_slug}"))


def fix_possessive(name):
    """ref_games writes "Santa's Gifts" as "Santa S Gifts": the apostrophe becomes a space. A lone capital S between two
    words joins the word before it again ("Santa's Gifts"). A name that ends in S is left as it is."""
    if not isinstance(name, str):
        return name
    return re.sub(r"\b([A-Za-z]{2,}) S (?=\S)", r"\1's ", name)


def studio_from_same_name(games: pd.DataFrame, tolerance: float = 0.05) -> pd.DataFrame:
    """A game takes the studio of the same game name (game_slug) from a better source: studio_source ranks ref_games,
    then the Client Area (often the distributor), then the Slot Catalog JSON, then the platform. For each name the best
    source that has it gives the provider, if all its games have the same one. Generic names (European Roulette,
    Blackjack) belong to several studios, so the game must also have the same `rtp` (both known, within `tolerance`) as
    one game of that provider. Returns provider_slug and studio, NaN where there is nothing to take, so
    the same game gets one Whizdom Game ID on every platform."""
    rank = games["studio_source"].map(STUDIO_RANK)
    known = games[rank.notna() & rank.lt(len(STUDIO_RANK) - 1) & games["game_slug"].ne("")].assign(rank=rank)
    best = known[known["rank"].eq(known.groupby("game_slug")["rank"].transform("min"))]
    best = best[best.groupby("game_slug")["provider_slug"].transform("nunique").eq(1)]
    donors = best[["game_slug", "provider_slug", "studio", "rank", "rtp"]].rename(columns={"rank": "donor_rank", "rtp": "donor_rtp"})
    targets = pd.DataFrame({"game_slug": games["game_slug"], "rtp": games["rtp"], "rank": rank, "row": games.index})
    found = targets.merge(donors, on="game_slug")
    found = found[found["rank"].gt(found["donor_rank"]) & (found["rtp"] - found["donor_rtp"]).abs().le(tolerance)]
    return found.drop_duplicates("row").set_index("row")[["provider_slug", "studio"]].reindex(games.index)


def provider_id(provider_slug: str) -> str | None:
    """The Provider ID: uuid5 of the standard provider slug. A provider slug has no '/', so it never gives a game ID."""
    return str(uuid.uuid5(NAMESPACE, provider_slug)) if provider_slug else None


def platform_id_columns(columns) -> list[str]:
    """The source columns that are ids of a platform, a brand or a source (game_id, core ids, launch ids...), so the
    catalog does not depend on the old ids. Columns made by master.ipynb (no prefix) are never in the list."""
    out = []
    for column in columns:
        prefix = next((p for p in SOURCES if column.startswith(p)), None)
        if prefix and PLATFORM_ID.search(column.removeprefix(prefix)):
            out.append(column)
    return out


def first_of(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.Series]:
    """Per row, the first column with a value, and the name of that column ('' if none has a value)."""
    value = pd.Series(np.nan, index=frame.index, dtype=object)
    source = pd.Series("", index=frame.index, dtype=object)
    for column in columns:
        fill = value.isna() & frame[column].notna() & frame[column].astype(str).str.strip().ne("")
        value[fill], source[fill] = frame.loc[fill, column], column
    return value, source


def first_labelled(frame: pd.DataFrame, labels: dict) -> tuple[pd.Series, pd.Series]:
    """Like first_of over the columns of `labels`, and the source as its label ('none' if no column has a value)."""
    value, source = first_of(frame, list(labels))
    return value, source.map({**labels, "": "none"})


def load_raw(folder: Path) -> pd.DataFrame:
    """The raw Slot Catalog JSON, one row per file: the UUID of SA, the page, the date of extraction and the whole JSON
    as text, so the scraping never has to be repeated."""
    rows = []
    for path in sorted(Path(folder).glob("*.json")):
        text = path.read_text()
        data = json.loads(text)
        if "raw_data" not in data:               # mapping_summary.json is not a game
            continue
        rows.append({"source_id": data["game_id"], "source_url": data.get("source_url"),
                     "extracted_at": (data.get("extraction_timestamp") or "")[:10] or None, "raw_json": text})
    return pd.DataFrame(rows, columns=["source_id", "source_url", "extracted_at", "raw_json"])


def close_names(games: pd.DataFrame, reference: pd.DataFrame, cutoff: float = 0.9) -> pd.Series:
    """For each game, a name of `reference` from the same provider that is close but not equal (similarity >= cutoff),
    to send to review. Both frames have provider_slug and game_slug."""
    by_provider = reference.groupby("provider_slug")["game_slug"].apply(lambda s: sorted(set(s))).to_dict()
    def closest(provider, name):
        options = [n for n in by_provider.get(provider, []) if n != name]
        found = difflib.get_close_matches(name, options, n=1, cutoff=cutoff) if name else []
        return found[0] if found else None
    return pd.Series([closest(p, n) for p, n in zip(games["provider_slug"], games["game_slug"])], index=games.index)


def match_status(game_ids: pd.Series, by_name: pd.Series, id_record_ids: pd.Series, close: pd.Series) -> pd.Series:
    """exact: the record found by the platform ids has the same name + provider (the same Whizdom Game ID) as the game.
    review: only one way finds a record, they disagree, or only a close name exists. no_match: nothing.
    `by_name` is the record found by name + provider, `id_record_ids` the Whizdom Game ID of the record found by ids.
    Comparing IDs and not records: one game can be twice in a source with two record ids."""
    exact = id_record_ids.notna() & (id_record_ids == game_ids)
    one = by_name.notna() | id_record_ids.notna()
    return pd.Series(np.select([exact, one, close.notna()], ["exact", "review", "review"], "no_match"), index=game_ids.index)


def compare_ids(new: pd.DataFrame, old_path: Path) -> pd.DataFrame:
    """Against the last run (game_identities.parquet): games that are new, gone, or whose Whizdom Game ID changed."""
    if not old_path.exists():
        return pd.DataFrame(columns=["game_uid", "change", "old_id", "new_id"])
    old = pd.read_parquet(old_path).rename(columns={"wisdom_game_id": "whizdom_game_id"})[["game_uid", "whizdom_game_id"]]   # the first runs called it wisdom_game_id
    both = old.merge(new[["game_uid", "whizdom_game_id"]], on="game_uid", how="outer", suffixes=("_old", "_new"), indicator=True)
    both["change"] = np.select([both["_merge"].eq("right_only"), both["_merge"].eq("left_only"),
                                both["whizdom_game_id_old"].ne(both["whizdom_game_id_new"])], ["new", "gone", "changed"], "")
    out = both[both["change"] != ""].rename(columns={"whizdom_game_id_old": "old_id", "whizdom_game_id_new": "new_id"})
    return out[["game_uid", "change", "old_id", "new_id"]].reset_index(drop=True)


def snake(name: str) -> str:
    """A column name as lower case words joined by '_' ('Core Id Desktop' is core_id_desktop)."""
    return slugify(name).replace("-", "_")


def column_notes(master: pd.DataFrame, notes: dict, originals: dict) -> pd.DataFrame:
    """The sheet `columns`: for every column of the catalog, its source, the columns or fields it comes from (origin),
    how it is made, its meaning, the percent of rows that have a value and one example. `notes` is {column: {meaning,
    how, origin, source}} (master_columns.yaml); the columns of a source (prefix) get source, origin and how from
    SOURCES, `originals` and HOW. What no file explains is said so, not guessed."""
    rows = []
    for column in master.columns:
        prefix = next((p for p in SOURCES if column.startswith(p)), "")
        note, original = notes.get(column, {}), originals.get(column, "")
        meaning = note.get("meaning") or (f"Column '{original}' of this source. There is no description of it in the repository." if prefix else "")
        values = master[column]
        filled = (values.notna() & values.astype(str).str.strip().ne("")).mean() * 100
        example = values[values.notna() & values.astype(str).str.strip().ne("")].head(1).astype(str).str[:80]
        rows.append({"column": column, "source": note.get("source") or SOURCES.get(prefix, "made in master.ipynb"),
                     "origin": note.get("origin") or original,
                     "how": note.get("how") or (HOW.get(prefix, "").format(original=original) + SC_EXTRA.get(column, "")),
                     "meaning": meaning, "filled_pct": round(filled, 1), "example": example.iloc[0] if len(example) else ""})
    return pd.DataFrame(rows)


def source_notes(path: Path, here: Path, rows: dict) -> pd.DataFrame:
    """The sheet `sources`: one row per source (master_sources.yaml) with the text of its query, read from the file in
    sql/ when the notebook runs, and how many rows or entries it has (`rows` is {prefix: count})."""
    table = pd.DataFrame(yaml.safe_load(path.read_text()))
    table["query"] = table["sql"].map(lambda f: (here / f).read_text().strip() if isinstance(f, str) else "")
    table["rows"] = table["prefix"].map(rows)
    return table[["prefix", "source", "where", "how", "joined", "query", "rows"]]


def _cell(value):
    """One value as Excel can show it: true/false as text, lists as 'a, b', text without control characters and no
    longer than an Excel cell (32,767 characters)."""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, np.ndarray)):
        value = ", ".join(str(v) for v in value)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        return ILLEGAL_CHARACTERS_RE.sub("", value)[:32767]
    return value


def excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy of the table that Excel shows as it is (see _cell); dates lose their time zone, which Excel has not."""
    out = frame.copy()
    for column in out.columns:
        if isinstance(out[column].dtype, pd.DatetimeTZDtype):
            out[column] = out[column].dt.tz_localize(None)
        elif out[column].dtype == bool or out[column].dtype == object:
            out[column] = out[column].map(_cell).astype(object)
    return out


def write_excel(path: Path, sheets: dict, freeze: dict | None = None, links: dict | None = None, wide: dict | None = None) -> None:
    """One sheet per table, in order. Every sheet keeps its header visible (or the cell in `freeze`, so the first
    columns stay too), has a filter on every column and readable widths; the column in `links` {sheet: column} is
    made of clickable links; the columns in `wide` {sheet: {column: width}} have that width and wrap their long text."""
    freeze, links, wide = freeze or {}, links or {}, wide or {}
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in sheets.items():
            table = excel_safe(table)
            table.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = freeze.get(name, "A2")
            sheet.auto_filter.ref = sheet.dimensions
            for i, column in enumerate(table.columns, start=1):
                longest = table[column].head(200).astype(str).str.len().max() if len(table) else 0
                sheet.column_dimensions[sheet.cell(row=1, column=i).column_letter].width = min(max(len(str(column)), longest or 0, 8) + 2, 50)
            for column, width in wide.get(name, {}).items():
                i = list(table.columns).index(column) + 1
                sheet.column_dimensions[sheet.cell(row=1, column=i).column_letter].width = width
                for row in range(2, len(table) + 2):
                    sheet.cell(row=row, column=i).alignment = Alignment(wrap_text=True, vertical="top")
            if name in links:
                col = list(table.columns).index(links[name]) + 1
                for row in range(2, len(table) + 2):
                    cell = sheet.cell(row=row, column=col)
                    if isinstance(cell.value, str) and cell.value.startswith("http"):
                        cell.hyperlink, cell.style = cell.value, "Hyperlink"
