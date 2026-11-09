# check_games_matched_final

Analysis of the data completeness of `games_matched_final.xlsx`, the final file of Navid.

I want to know which attributes have data, which are mostly empty, and where the same information is written in more
than one column, so each attribute can later be unified in one clean column.

## Files
- `fields.yaml`: the map of the attributes (which columns hold each one) and the proposed rule to unify each one.
  It also lists the columns that are not attributes. Edit it to change a rule or add a field.
- `completeness.py`: the functions (tested in `tests/test_completeness.py`).
- `analysis.ipynb`: calls the functions and writes the report.
- `output/completeness_report.xlsx`: the report (ignored by git).
- `unify.py` and `unify.ipynb`: one value per field and game, without losing information (tested in `tests/test_unify.py`).
- `output/games_unified.xlsx` and `output/games_unified.parquet`: the unified table.
- `PROBLEMS.md`: the problems I found in the file.

## How to run
Put `games_matched_final.xlsx` in `catalog_project/` and run `analysis.ipynb` from the top (about 20 seconds).
There is no cell that reads ClickHouse or S3.

## What counts as missing
Empty, or written as N/A, NA, N-A, none, -, null, unknown, [] or {}. A `0` is counted apart: it can be a real value.

## Statuses
- matched: Navid confirmed the game.
- fuzzy: there are candidates but they are not confirmed.
- unmatched: no candidate.

The levels are computed on matched + fuzzy (the data that can be used). The fuzzy values are marked as not confirmed.

## Report
| Sheet | What it shows |
|---|---|
| `fields` | One row per attribute: coverage of the column of Navid (`attr_`) and of all the places, games that have the value only outside `attr_`, conflicts, level |
| `columns` | Every column of the file with its coverage by status, values written as N/A and zeros |
| `dispersion` | Games with a value outside `attr_` while `attr_` is empty, by place and status |
| `dispersion_examples` | Real games to check by hand |
| `formats` | How the values are written (values or shapes), to write the normalization |
| `unification_rules` | The proposed rule of each field, to validate with Navid |

## Unify
`unify.ipynb` makes one row per matched or fuzzy game (`confirmed` is false for fuzzy). For each field:
- `<field>`: the clean value. The first one found: `attr_` of Navid, then the other columns (order of `fields.yaml`), then `crawler_records`.
- `<field>_raw`: the original value, `<field>_source`: where it comes from, `<field>_ok`: false if it could not be normalized.
- `<field>_all`: every value of every place (not for long texts, which stay in the file of Navid).
The normalization is strict: a value is changed only if it has one exact, known format (`x5000.00`, `96.5%`, `28/03/2024`,
`High volatility`). Free text is never read, because a number taken from a sentence can be the one of another version of the
game (`96.25% base; no 94% version found`). Those values stay in `<field>_raw` and the sheet `not normalized` lists them.
The RTP and the max win of some games are written in a sentence (`96.25% base; no 94% version found`): the first number is taken
and the game is marked in `rtp_from_text` or `max_win_from_text`. In a variant (`94 Low Rtp`) that number can be the RTP of the base game.
An amount in money in the max win (`EUR 250,000`) is not a multiple of the bet, so it is not mixed with it: it goes to `max_win_amount` and `max_win_currency`.
The sheet `not normalized` lists the doubtful values to discuss with the team, each with the clean value that was put and the reason: read
from a sentence (the first value of the text), a layout kept as it is written, an amount in money apart from the multiples, or no amount
written. They are in the clean table, because they go to the master catalog.
In the volatility, a sentence that says adjustable or selectable gives `High-Adjustable`, and it is marked in `volatility_from_text`.
The notebook stops if a value of the file is lost, if a value of Navid is changed or if a number changes its value. The sheet
`changes` lists every value whose writing changed, to check by eye. The sheet `rtp variants` lists the games whose RTP is not
the one in their name. The sheet `clean` of the Excel has only the
clean values and their source; `full` has everything.
Two sheets give the ratio of empty fields: `empty by field` (one row per attribute) and `empty by group` (one row per category: math,
bets, gameplay, classification, dates, texts, studio_website, community, technical; the group of each field is in `fields.yaml`).
Empty means no clean value. `no_data_%` is a game that has no value in any place, and `not_normalized_%` is a game that has a value
that could not be normalized (kept in `<field>_raw`), for example a layout written as text.
The hit rate and the bonus frequency of the community are written `1/3.6` (one every 3.6 spins) and are converted to percent (100 / 3.6 = 27.78). The ratio of a
category is the empty cells of all its attributes together, not the average of the percents. It is calculated on matched and fuzzy games.
The rules are still a proposal: Navid and I have to validate them.
Texts of third party websites (description, reviews, pros, cons, FAQs) are measured, but the team has not decided if they enter the catalog.
