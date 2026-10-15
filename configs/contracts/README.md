# Table contracts

One file per table. Each file says what a correct table looks like, and the data checks in `src/quality/` read it with `contracts.load()`.

| Folder | Tables | Made by |
|---|---|---|
| `lake/` | The 30 tables that exist in the lake today: 29 in gold and `transactions` in bronze. The checks use these. | `src/quality/inventory_to_contracts.py`, from `docs/dq_reports/layer_inventory.json` and `configs/quality.yaml` |
| `design/` | The 21 gold tables of the rebuild design (2026-09-01). Reference only: most names and columns differ from the lake. | `src/quality/excel_to_contracts.py`, run once on 2026-10-14 from the design workbook `gld-schemas.xlsx` (not in the repository) |

The lake contracts reuse the design where a column has the same name: group, unit, range and additive come from there.

To change a lake contract, I edit `configs/quality.yaml` (grains, personal data columns) and run the script again. A hand edit in a lake YAML file is lost on the next run.

## Table fields

| Field | Meaning |
|---|---|
| `table` | Table name. |
| `layer` | Lake contracts only: layer of the table, e.g. `org/40-gold`. |
| `grain` | Columns that identify one row, so they must be unique together. In design contracts, the `ORDER BY`. In lake contracts: set by hand in `configs/quality.yaml`, else the design grain of the table with the same name, else a proposal from the id and date columns. A table with no date column has one file per day, so its grain holds within each day. |
| `grain_checked` | `true` only after the grain is shown to be unique on real data. All lake grains start as `false`. |
| `partition_by` | Design: `PARTITION BY` expression. Lake: folder layout, e.g. `YYYY/MM/DD folders`. |
| `engine` | ClickHouse engine (design only). With `ReplacingMergeTree`, rows with the same grain and a newer `_version` replace older ones. |
| `columns` | One entry per column, in table order. |

## Column fields

Only `name` and `type` are required. The other fields appear only when they apply.

| Field | Meaning | Check it gives |
|---|---|---|
| `name` | Column name. | The column exists. |
| `type` | Design: ClickHouse type. Lake: Arrow type of the Parquet file. | Type. |
| `nullable` | Lake: written in the file (`false` for the grain columns). Design: `true` when the type is `Nullable(...)`. | No nulls when `false`. |
| `key` | `true` if the column is part of the grain. | Grain is unique. |
| `group` | Block of the column: key, meta, money, count, fx, ... | None. Used to group the reports. |
| `unit` | Unit of the measure: currency, days, players, 0-1, ... | None. Used to pick the histogram bins. |
| `range` | `[min, max]`, both included. | Values inside the range. |
| `additive` | `true` if the measure can be summed across days and players. | Only additive measures are compared as sums between layers. |
| `status` | Design only: `exists` on prod, `renamed`, `new`, `in_silver`, `needs_pipeline`, `needs_source`. | None. |
| `pii` | `true` for personal data (`pii_columns` in `configs/quality.yaml`). | Counted, but its values are never printed. |

## Open points

- Six scores of the design table `gld_player_signal` have unit 0-1, but their rule clamps them to 0 to 100. Lake columns with the same name, such as `engagement_score` in `gld_player_signals_daily`, take the range `[0, 100]`, until I check the values.
- In the design, `provider_code` of `gld_player_gaming_daily` is a key column but not part of the grain.
- The design grain of `gld_player_signal` uses `event_date`. An older note in the design used `snapshot_date`.
- All lake grains are still to be checked on the data (phase 1).
