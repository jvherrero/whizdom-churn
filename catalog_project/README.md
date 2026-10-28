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
