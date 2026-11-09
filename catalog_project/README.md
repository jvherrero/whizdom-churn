# Game catalog V0: priority list of games to label

I find the games that make 99% of the GGR of the last 6 months, remove the ones already classified, and give the
rest as an Excel list, split by provider for the three people who label.

At the time of writing (2026-10-28): 3,720 highly active games, 769 already classified, **2,951 in the list**.

## Pipeline

```mermaid
flowchart LR
    subgraph CH["ClickHouse (read only)"]
        T["slv_game_daily<br/>slv_games_discovered_resolved<br/>slv_game_rounds<br/>gld_game_lifetime<br/>ref_games"]
    end
    subgraph LOCAL["internal_taxonomy/ (local files)"]
        F1["Slot Catalog URLs (SA)"]
        F2["Client Area export"]
        F3["games_enriched.xlsx"]
        F4["sa_raw/: raw Slot Catalog JSON (SA)"]
    end

    T --> S1["1. active_games.py<br/>games with 99% of the GGR"]
    S1 --> O1[("active_games.csv")]
    O1 --> S2["2. priority_list.py<br/>remove classified games,<br/>add URLs and Client Area data"]
    F1 --> S2
    F2 --> S2
    F3 --> S2
    F4 --> S2
    S2 --> O2[("priority_games.xlsx<br/>games, summary, providers")]
    O2 --> S3["3. split_batches.py<br/>split by provider"]
    S3 --> O3[("batch_NAME.xlsx<br/>one file per person")]
```

```bash
.venv/bin/python catalog_project/active_games.py      # ClickHouse, slow
.venv/bin/python catalog_project/priority_list.py     # local, seconds
.venv/bin/python catalog_project/split_batches.py     # local
.venv/bin/python -m pytest catalog_project/tests -q   # 13 tests, no ClickHouse needed
```

If only a local input changes (URL file, Client Area, `games_enriched`), run the last two scripts only.
`audit_ref_games.py` is a separate check: it prints the freshness of the catalog and how well the activity joins to it.

## Decisions

| Decision | Why |
|---|---|
| **The metric is GGR** (wagered minus won, real money and bonus, EUR), not rounds | The boss asked for it |
| **99% cut:** sort by GGR, then rounds; keep games until the running total reaches 99% of the positive GGR. The game that crosses the limit stays. GGR of 0 or below never enters | The smallest set that covers the activity. The other 16,000 games hold about 1% |
| **Join activity to catalog by normalised id** (lower case, letters and digits only) **and provider, through `slv_games_discovered_resolved`** | The catalog and the activity write ids differently (`coinstrike` and `coin_strike`). Without this, 30% of the rounds did not match. Now 99.9995% do. One id can also belong to two providers |
| **The list is every active game not in `games_enriched.xlsx`**, even if it already has a URL or a game type | `games_enriched` is the file of games SA already classified. `enrichment_status` of `ref_games` is not used to filter, because it only says what the pipeline has |
| **URL:** the Slot Catalog file first, then the Client Area `External Resource` if it is a Slot Catalog link. `sc_url_source` says which | SA has checked its file. The Client Area is a fallback |
| **The raw Slot Catalog JSON of SA add columns at the end** (`sc_*`: features, themes, tags, RTP, volatility, paylines, description...). The old columns do not change | They keep the multiple mechanics and the tags that the summary Excel files dropped. Only games with a URL from the SA file have a JSON (925 of the list). `slotcatalog_json.py` reads them. They join through the UUID that `game_slotcatalog_urls.xlsx` gives to each game |
| **Sports games stay in the list** with `is_sports = true` (providers BETTECH and DIGITAIN_SPORTBOOK) | They hold a large share of the GGR but have no Slot Catalog page. The boss decides how to label them |
| **Players:** lifetime from `gld_game_lifetime`; 30 and 90 days from `slv_game_rounds`, one query per tenant | A 6 month query hit the 300 s limit. A player belongs to one tenant, so the sum is exact. The 30 day figure of `gld_game_signals_daily` is a sum of daily players, not distinct players |
| **Launch date:** `release_date`, else `first_seen` (both from `ref_games`), else `sc_release_date` (raw JSON). The placeholder `2022-01-01` and a `first_seen` on the first day of data (`2025-04-01`) are ignored. If no source has a real date, the days stay empty | 1,812 games share the placeholder. A wrong number is worse than an empty one. The JSON date fills 709 of the 1,085 empty values of the list |
| **Split by provider, balanced by number of games** | Each person works on whole providers and nobody overlaps. The GGR per batch is uneven |
| **`true` and `false` are written as text** | Excel in Spanish would show VERDADERO and FALSO |

## Inputs and outputs

- **ClickHouse** (`whizdomai_org_gmntc_medallion`): `slv_game_daily`, `slv_games_discovered_resolved`, `slv_game_rounds`,
  `gld_game_lifetime`, `ref_games`. Queries are in `sql/`.
- **Local files** in `internal_taxonomy/`: Slot Catalog URLs (SA), Client Area export, `games_enriched.xlsx`, and the folder
  `sa_raw/` with the raw Slot Catalog JSON (one file per game; ignored by git).
- **Outputs** in `output/` (ignored by git): `active_games.csv`, `priority_games.xlsx`, `batch_<name>.xlsx`.
- **Settings** (paths, `months`, `coverage`, sports providers, batch names) are in `config.yaml`.

The `sc_*` columns come from the raw JSON. `sc_title` is the web page title, so it has extra text: use `game_name` as the name.
The `ca_*` columns come from the Client Area. They are a starting point to autocomplete, and they are not checked.

## Open points

- `games_enriched.xlsx` has 1,071 games. About 2,200 more are `enriched` in `ref_games` but not in that file, so the
  list treats them as not classified. The boss has to confirm that `games_enriched` is the full list of classified games.
- `gld_game_unique_players` is empty (checked 2026-10-28). The gold layer needs a check after the node 3 outage.
- `days_since_launch` is empty for about 45% of the games. A real launch date would have to come from SA.
- A few games have more 90 day players than lifetime players (for example `baccarat`, VIVO). The difference is small and
  does not change the list.
- Check the spelling of the batch names in `splits` of `config.yaml` before sharing the files.

`PROGRESS.md` has the day by day log, `DATA_FLOW.md` how the data moves from the client area to the catalog, and
`PLAN.md` the original plan.

## Master catalog (Whizdom Game ID)

`master.ipynb` builds the master list of games with `master_lib.py` (tests in `tests/test_master_lib.py`). It starts
from all the games of ref_games (no filter by date), with every
column, and joins all the data of the other sources to them:

- **Sources and prefixes:** `ref_` ref_games (`sql/ref_games_master.sql`, all columns), `ca_` Client Area, `scu_` Slot
  Catalog URL file, `sc_` raw Slot Catalog JSON (all fields, also `pros_cons`, `how_to_play` and
  `bonuses_features`), `ge_` games_enriched, `gm_` games matched (`games_matched_final.xlsx`). The Client Area, the Slot
  Catalog URL file and games_enriched are joined by game id and platform; the JSON by the UUID of the Slot Catalog URL
  file, else by name + provider; games matched by `game_uid`. Columns with no prefix are made by the notebook.
- **A game is its name + provider**, and the provider is the **studio** that made it (not the platform). The studio
  comes from `studios` of ref_games, then the Client Area (often the distributor), then the raw JSON; if none is known,
  the platform is used and `studio_source = platform`.
- **Rows of the catalog:** one row per `game_uid` of ref_games (the list of ClickHouse, 20078), plus the games that are only in
  other sources (`only_in`, at the end). A game on several platforms has one Whizdom Game ID in several rows. The Games
  Catalog Tracker (`output/master/Games Catalog Tracker.xlsx`, section 11) is made from the same rows, in the same order.
- **Order of the catalog:** the same as `active_games.py`: GGR of the last 6 months (highest first), then rounds, then
  `game_uid`. `ggr_eur_6m` and `rounds_6m` come from `output/active_games.csv` (`master_ggr` in `config.yaml`, the frozen list
  of 2026-10-28), by `game_uid`. The first rows are the 99% list, already sorted; the rows with no GGR go after.
- **games matched and games_enriched by row:** each `game_uid` takes its own row of games matched; a `game_uid` that is not in
  it takes the row of its game (the best status of the ID). The ge_ data of the other `game_uid` of an ID fill the empty cells.
- **Sports betting** has no name in any source: `explicit_names` of `config.yaml` gives the six games a name on purpose
  (`name_source = explicit`), `providers.yaml` maps the GT Sports feeds to `betby` and `digitain`, and `is_sports` marks
  them. Any other game with no name in any source has no ID (sheet `no name`, with its GGR).
- **Name:** `ref_games` first; if it is empty, the Client Area, the Slot Catalog URL file or games_enriched (`name_source`). The Client
  Area name is not first because it is sometimes more generic (`Curse of the Pharaoh` for `... Bonus Buy`). `ref_games`
  loses apostrophes (`Santa S Gifts`): `fix_possessive` puts them back.
- **Studio from the same name:** a game takes the provider of the same game name from a better source (ref_games, then
  Client Area, then JSON, then platform) if that source has one provider for the name and the RTP is the same
  (`studio_source = same_name`). Generic names (`European Roulette`) belong to several studios, so a different RTP is a
  different game. This joins, for example, ORYX (distributor) and Gamomat (studio) into one game.
- **Rebranding dictionary:** `providers.yaml` maps every spelling of a provider to one standard name. A person adds names
  to it; the notebook only proposes groups that look alike.
- **Whizdom Game ID:** `uuid5(NAMESPACE, "<provider slug>/<game slug>")`, with a fixed namespace in the code. The same
  game and provider always give the same ID. **Variants (ID version 2):** one name and provider can be two games. If the
  Client Area gives them two names, each takes its Client Area name (Lady Wolf Moon / Lady Wolf Moon Megaways); else, if
  the RTP of the casino differs by more than 0.1 points or the game type differs, the variant goes into the ID
  (`<provider>/<game>/rtp-94`) and into the column `variant`. The volatility is not used: it differs between platforms for
  the same game. A game with no name gets no ID (sheet `no name`): an ID from the platform
  id would tie the catalog to the old ids. **Provider ID:** `uuid5(NAMESPACE, "<provider slug>")` (sheet `providers`).
- **No old ids:** the catalog leaves out the ids of platforms, brands and sources (game_uid, game_id, core ids, launch
  and start ids, the UUID of the Slot Catalog URL file...). `game_identities.parquet` keeps the link from each game_uid to its Whizdom Game ID,
  to join the activity. A change in `slugify()` or in `providers.yaml` can change IDs, so every run
  compares them with the last one (sheet `id changes`).
- **Flags:** `match_status` (exact, review, no_match against the raw JSON), `no_url`, `missing_data`,
  `similar_provider` (another provider has almost the same name, not merged in `providers.yaml`; the sheet `providers`
  shows it in `similar_to`), `id_shared` and
  `same_platform_conflict` (one game with two game_uid on the same platform, so far always the same game).
  `status` is ref_games, else the Client Area.
- **Final columns and full file:** the sheet `catalog` of `master_catalog.xlsx` has the 39 columns to read (ID, name, GGR, first and last seen, vertical,
  provider, status, the unified attributes with one fixed value each, `theme`, `features`, URL and checks). When the sources
  disagree, the value is the one that most of them give (a tie goes to the first in the order).
  `master_catalog_full.xlsx` has the same rows with every column of every source (also the `_source` columns).
- **Themes and features:** one column each (`theme`, `features`), the words of every source joined, each word once
  (`ml.unify_terms`: Free Spins = FreeSpins, Animals = Animal, Fruit theme = Fruit; `Gold` and `Gold-color` stay apart).
- **Final file:** `output/master/master_catalog.xlsx`, for the team. The sheet `read me` says what each sheet is and what
  to expect in it: `catalog`, `columns`, `order of preference`, `sources`, `coverage 99%`, `pending unmatched`,
  `providers`, `flags`, `review` and `no name`. Columns that are empty in every row are left out.
- **No names and no local files in the Excel:** the texts name the files that the team has (the Slot Catalog JSON, the
  Excels, the Client Area export, `games_matched_final.xlsx`), never a person, `.py`, `.yaml`, `.ipynb`, `.parquet` or
  `output/` file. The batches of games matched are `batch 1`, `batch 2` and `batch 3` (alphabetical order of their
  name). The last cell stops before writing if `ml.forbidden_cells` finds one of these words.
- **Other outputs** in `output/master/`: `master_games.parquet` (the catalog), `game_identities.parquet` (every
  game_uid and its ID) and `raw_sources.parquet` (every raw JSON, whole, also the games that are not in ref_games).
- **games matched and games_enriched (version 1):** `games_enriched` is joined by game id (`ge_` columns; `in_games_enriched`). games matched
  joins by `game_uid` (`gm_` columns): its status, its links and one clean value per attribute (`check_games_matched_final/unify.ipynb`, with the
  source and a flag for what was read from a sentence). The texts of third party websites stay out until the team decides. The review of the
  batches is not joined yet (it waits for my check). `attributes_from` says which file gives the attributes of each game; the sheet
  `coverage 99%` counts it on the 99% list, and `pending unmatched` lists the games of games matched with no match yet, most GGR first.
- **Unified attributes:** RTP, volatility, minimum and maximum bet, max win, hit frequency, layout, release date, buy bonus and free spins have one
  value, its source (`<attribute>_source`) and a conflict mark (`<attribute>_conflict`). The sheet `order of preference` lists the sources of
  each one in order, why, and when two sources are a conflict. The order of the sources comes from comparing them:
  games_enriched, ref_games and the Client Area agree 100% with each other (they are the casino: the variant that is played), and games matched and
  Slot Catalog agree with each other but only 80 to 88% with the casino (they describe the base game, so in a `94 Low Rtp` variant their RTP is
  the base one). So the casino first, then games matched, then Slot Catalog. The release date goes the other way (games matched, Slot Catalog, then the
  platform), because the date of the casino is when the game was added, and agrees with Slot Catalog in only 24% of the games.
  `game_type`, theme and features are not unified: the casino and Slot Catalog use different lists of words.
- **games_enriched wins:** a game that is in games_enriched and also in games matched (the same game with another game_uid) stays
  `games_enriched` in `attributes_from`, and `also_in_games_matched` marks it.
- Cells marked MANUAL read ClickHouse or S3. The notebook needs `output/master/ref_games.parquet`: run the MANUAL cell
  once to export it. games matched is `navid_final` in `config.yaml`.
