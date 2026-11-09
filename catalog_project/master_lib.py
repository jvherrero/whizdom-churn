"""The master catalog of games: one Whizdom Game ID per game, and the functions master.ipynb uses to build it.

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
ID_VERSION = 2                                                    # 2: variants by Client Area name, RTP or type
NAMESPACE = uuid.UUID("907a5265-b739-542f-8906-b9b0c91432a8")   # uuid5(NAMESPACE_DNS, "whizdom.ai"), fixed for ever
SYMBOLS = re.compile(r"[™®©]")                                   # NFKD would turn ™ into "TM"
APOSTROPHES = re.compile(r"['’`´]")                              # "Joker's" and "Jokers" are the same name
# The source of each column of the catalog, by its prefix; a column with no prefix is built for the catalog. These texts go
# to the Excel, so they name the files that the team has, never a local file or a person.
SOURCES = {"ref_": "ref_games (ClickHouse)", "ca_": "Client Area export", "scu_": "Slot Catalog URL file (game_slotcatalog_urls.xlsx)",
           "sc_": "Slot Catalog JSON", "ge_": "games_enriched.xlsx", "gm_": "games matched (games_matched_final.xlsx)"}
NO_PREFIX = "built for the catalog"
# How the columns of each source are made (the prefix tells the source). {original} is the column or field of the source.
HOW = {"ref_": "Copied as it is from ClickHouse (SELECT * FROM ref_games FINAL). One row per game_uid; the catalog shows the row of the best match of the ID.",
       "ca_": "Copied as it is from the Client Area export, column '{original}'. Joined to ref_games by key = game id normalized + platform code: Core Id Desktop first, then Core Id Mobile.",
       "scu_": "Copied as it is from the Slot Catalog URL file, column '{original}'. Joined by key = game id normalized + platform: game_src_id first, then core_id_desktop.",
       "ge_": "Copied as it is from games_enriched, column '{original}'. Joined by key = game id normalized + platform_code: game_src_id, core_desktop_id, core_mobile_id.",
       "gm_": "From games matched (games_matched_final.xlsx), by game_uid.",
       "sc_": "Field {original} of the Slot Catalog JSON, cleaned (the page marks [ i ] are removed). The JSON is the one found by the UUID of the Slot Catalog URL file, else by name + provider."}
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


def whizdom_id(game_slug: str, provider_slug: str, variant: str = "") -> str | None:
    """The Whizdom Game ID of a game; None if the name or the provider is missing. A variant (two games with the same
    name and provider, told apart by their RTP or type) adds its slug: '<provider>/<game>/rtp-94'."""
    if not game_slug or not provider_slug:
        return None
    return str(uuid.uuid5(NAMESPACE, f"{provider_slug}/{game_slug}" + (f"/{slugify(variant)}" if variant else "")))


def name_from_client_area(games: pd.DataFrame) -> pd.Series:
    """ref_games can give the same name to two games: 'Lady Wolf Moon Megaways' for 'Lady Wolf Moon' too. If the
    game_uid of one name and provider have different names in the Client Area, each takes its Client Area name.
    Returns the new names (empty where the name stays). Needs game_slug, provider_slug and ca_game_name."""
    ca_slug = games["ca_game_name"].map(slugify).replace("", np.nan)
    group = games["provider_slug"] + "/" + games["game_slug"]
    differs = games["game_slug"].ne("") & group.notna() & ca_slug.groupby(group).transform("nunique").gt(1)
    return games["ca_game_name"].where(differs & ca_slug.notna())


def variant_of(games: pd.DataFrame, rtp_tol: float = 0.1) -> pd.Series:
    """Two game_uid with the same name and provider are two games if the casino gives them a different RTP (by more
    than rtp_tol points: Great Blue Jackpot 95.02 and 94.03) or a different game type (Roulette and Slots). Returns
    'rtp 95.02' or 'type Roulette' for each game_uid of such a name, '' for the rest. A game_uid with no value keeps ''.
    Needs whizdom_game_id (the ID without variant), rtp and game_type. The volatility is not used: it differs between
    platforms for the same game (European Roulette is low on one and medium on another)."""
    out = pd.Series("", index=games.index)
    rtp = games["rtp"].round(2)
    kind = games["game_type"].astype("string").str.strip().replace("", pd.NA)      # '' is no value, not a type
    by_id = games.assign(kind=kind.str.lower()).groupby("whizdom_game_id")
    by_rtp = (by_id["rtp"].transform("max") - by_id["rtp"].transform("min")).gt(rtp_tol)
    by_type = ~by_rtp & by_id["kind"].transform("nunique").gt(1)
    out[by_rtp & rtp.notna()] = "rtp " + rtp[by_rtp & rtp.notna()].map("{:g}".format).astype(str)
    out[by_type & kind.notna()] = "type " + kind[by_type & kind.notna()].astype(str)
    return out


NO_TERM = {"", "na", "n/a", "none", "-", "unknown", "null", "nan"}


def split_terms(text) -> list:
    """The words of a list of themes or features: split by , | or ; but not inside brackets, so 'Burst (Crash|Bustabit
    like) Mechanic' stays one word. '/' is not a separator: 'Risk/Gamble (Double) game' is one feature."""
    if not isinstance(text, str):
        return []
    terms, word, depth = [], "", 0
    for char in text:
        depth += (char == "(") - (char == ")")
        if char in ",|;" and depth <= 0:
            terms.append(word)
            word, depth = "", 0
        else:
            word += char
    terms.append(word)
    terms = [re.sub(r"(?i)^feature:\s*", "", t.strip()) for t in terms]          # 'Feature: Random reward' is 'Random reward'
    return [t for t in terms if t.lower() not in NO_TERM]


def term_key(term: str) -> str:
    """Two words are the same if they only differ in case, spaces, hyphens, a final s, the end ' theme' or the start
    'Feature:': 'Free Spins' and 'FreeSpins', 'Animals' and 'Animal', 'Fruit theme' and 'Fruit'. 'Gold' and 'Gold-color' differ."""
    key = re.sub(r"^feature:\s*", "", term.strip().lower())
    key = re.sub(r"\s+theme$", "", key)
    key = re.sub(r"[^a-z0-9]", "", key)
    return key[:-1] if len(key) > 3 and key.endswith("s") else key


def unify_terms(frame: pd.DataFrame, columns: list) -> pd.Series:
    """One list of themes (or features) per row out of several columns: every word of every source, the same word only
    once (term_key), sorted, joined with ', '. Each word is written as most sources write it, with capitals first
    (Slot Catalog writes 'Free Spins', ref_games 'free spins'). Empty if no source has a word."""
    words = {c: frame[c].map(split_terms) for c in columns}
    spellings = pd.Series([t for s in words.values() for terms in s for t in terms])
    if spellings.empty:
        return pd.Series(None, index=frame.index, dtype=object)
    table = pd.DataFrame({"term": spellings, "key": spellings.map(term_key), "capital": spellings.str.contains(r"[A-Z]")})
    table = table.groupby(["key", "term", "capital"]).size().rename("n").reset_index()
    best = table.sort_values(["key", "capital", "n", "term"], ascending=[True, False, False, True]).drop_duplicates("key").set_index("key")["term"]
    rows = pd.concat(words.values(), axis=1).apply(lambda r: sorted({best[term_key(t)] for terms in r for t in terms}, key=str.lower), axis=1)
    return rows.map(lambda terms: ", ".join(terms) if terms else None)


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


def unify_attribute(frame: pd.DataFrame, sources: list, numeric: bool = False, tol_abs: float = 0.0, tol_rel: float = 0.0,
                    conflict_columns: list | None = None) -> pd.DataFrame:
    """One value of an attribute out of several columns. `sources` is [(column, normalizer)] in the order of priority: each value is
    normalized (None if it cannot be read) and the first one that exists is the value. Returns value, source (the column that gave it)
    and conflict (true if two sources have a different value: numbers differ by more than tol_abs + tol_rel * value). The
    columns in `conflict_columns` (default all) are the ones compared. A conflict is solved: the value is the one that most of
    the sources give (all of them, not only conflict_columns), and a tie goes to the first source in the order."""
    clean = pd.DataFrame({c: frame[c].map(lambda v, f=f: f(v) if pd.notna(v) else None) for c, f in sources}, index=frame.index)
    if numeric:
        clean = clean.apply(pd.to_numeric, errors="coerce")
    has = clean.notna()
    compared = clean[conflict_columns or list(clean.columns)]
    if numeric:
        high, low = compared.max(axis=1), compared.min(axis=1)
        differs = (high - low) > tol_abs + tol_rel * high.abs()
    else:
        differs = compared.nunique(axis=1) > 1
    conflict = compared.notna().sum(axis=1).ge(2) & differs
    value, source = clean.bfill(axis=1).iloc[:, 0], has.idxmax(axis=1).where(has.any(axis=1))

    def same(a, b):
        return abs(a - b) <= tol_abs + tol_rel * max(abs(a), abs(b)) if numeric else a == b

    # a conflict is solved by the majority of all the sources (the value that most of them give); a tie goes to the first in order
    for i, row in clean[conflict].iterrows():
        given = [(c, v) for c, v in row.items() if pd.notna(v)]
        votes = [sum(same(v, w) for _, w in given) for _, v in given]
        source[i], value[i] = given[votes.index(max(votes))]
    return pd.DataFrame({"value": value, "source": source, "conflict": conflict})


def attributes_from(in_enriched: pd.Series, status: pd.Series) -> pd.Series:
    """Where the attributes of each game come from. games_enriched wins: its classification is the one that the project follows,
    so a game that is also in games matched (the same game with another game_uid) stays games_enriched. Else games matched
    (matched or fuzzy); 'games matched, unmatched (pending)' if it has no match yet; else 'not in games_enriched or games matched'
    (its attributes come only from ref_games, the Client Area and Slot Catalog). status is gm_status."""
    names = status.map({"matched": "games matched, matched", "fuzzy": "games matched, fuzzy",
                        "unmatched": "games matched, unmatched (pending)"}).fillna("not in games_enriched or games matched")
    return pd.Series(np.where(in_enriched, "games_enriched", names), index=status.index)


def tracker(rows: pd.DataFrame) -> pd.DataFrame:
    """The Games Catalog Tracker: one row per game, with its ID, name, provider, URL and four states.
    Raw Data: Yes if there is raw page data (a Slot Catalog JSON, or games matched matched or fuzzy).
    Mapping: Mapped (Slot Catalog exact, games_enriched or games matched matched), To review (Slot Catalog review or
    games matched fuzzy) or Not mapped. Status: the commercial status and 'ready to use' (released, Mapped, with an
    RTP and a volatility), 'needs review' (released and Mapped or To review, but a check fails) or
    'not ready'. Scope: In scope if the game is in the 99% GGR list. Notes: why the game is not ready.
    `rows` has the columns of the catalog; a game with no name has no whizdom_game_id and its id in `id_note`. A game
    that is only in another source says so in `source_note`, which goes first in the notes."""
    has_id = rows["whizdom_game_id"].notna()
    state = rows["status"].replace("", np.nan).fillna("unknown")
    released = state.eq("released")
    mapped = rows["match_status"].eq("exact") | rows["in_games_enriched"].eq(True) | rows["gm_status"].eq("matched")
    review = ~mapped & (rows["match_status"].eq("review") | rows["gm_status"].eq("fuzzy"))
    attributes = pd.DataFrame({"no RTP": rows["rtp"].isna(), "no volatility": rows["volatility"].isna()})   # conflicts are solved in the catalog
    ready = has_id & released & mapped & ~attributes.any(axis=1)
    readiness = np.select([ready, has_id & released & (mapped | review)], ["ready to use", "needs review"], "not ready")
    # why the game is not ready: one text per failed check, in this order
    id_note = rows["id_note"].fillna("") if "id_note" in rows else pd.Series("", index=rows.index)
    source_note = rows["source_note"].fillna("") if "source_note" in rows else pd.Series("", index=rows.index)
    texts = pd.DataFrame({"source": source_note, "name": np.where(rows["game_name"].notna(), "", "no name (" + id_note + ")"),
                          "status": np.where(released, "", "status " + state),
                          "variant": ("variant of a game with the same name: " + rows["variant"]).where(rows["variant"].fillna("").ne(""), "")
                          if "variant" in rows else "",
                          "mapping": np.select([review, ~mapped], ["mapping to review", "not mapped"], ""),
                          **{k: np.where(v, k, "") for k, v in attributes.items()}}, index=rows.index)
    notes = texts.apply(lambda r: ", ".join(t for t in r if t), axis=1)
    url = rows["url"]                          # the best link of the game (see url in the catalog)
    raw = rows["sc_title"].notna() | rows["gm_status"].isin(["matched", "fuzzy"])
    return pd.DataFrame({"whizdom_game_id": rows["whizdom_game_id"], "Game Name": rows["game_name"], "Game Provider": rows["provider"],
                         "External URL": url, "Raw Data": np.where(raw, "Yes", "No"),
                         "Mapping": np.select([mapped, review], ["Mapped", "To review"], "Not mapped"),
                         "Status": state + ", " + readiness, "Scope": np.where(rows["ggr_eur_6m"].notna(), "In scope", "Out of scope"),
                         "Notes": notes}).reset_index(drop=True)


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
        rows.append({"column": column, "source": note.get("source") or SOURCES.get(prefix, NO_PREFIX),
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


# Words that must not be in the Excel: names of people and local files that only the author of the catalog has.
FORBIDDEN = re.compile(r"(?i)\bnavid(?![a-z])|\bjavier(?![a-z])|sh?udhamshu|\bnv_|\brv_|\bsa_|active_games|output/|\bsql/|check_games_matched_final"
                       r"|\.(?:py|ipynb|yaml|parquet)\b|(?-i:\bSA\b)")


def forbidden_cells(sheets: dict, pattern: re.Pattern = FORBIDDEN) -> list:
    """Every header or text cell of the sheets that has a forbidden word, as (sheet, column, text). Empty if all is clean."""
    hits = []
    for sheet, frame in sheets.items():
        frame = frame.to_frame() if isinstance(frame, pd.Series) else frame
        for column in frame.columns:
            if pattern.search(str(column)):
                hits.append((sheet, "header", str(column)))
            text = frame[column].dropna().astype(str)
            hits += [(sheet, column, t[:120]) for t in text[text.str.contains(pattern)].unique()]
    return hits


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
