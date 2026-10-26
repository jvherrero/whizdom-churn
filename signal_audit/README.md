# Signal Audit

Audit of the gold table `gld_player_signals_daily` (and the daily tables it comes from): distributions, impossible and suspicious values, its own formulas, and the signals recomputed from the raw transactions. The deliverable is the error list `informe/errores_tabla_senales.pdf`: each wrong column, the fix and a link to the evidence.

| File | What it is |
|---|---|
| `01_distributions.ipynb` | Shape, peaks, missing, impossible and suspicious values, all brands and brand by brand; index limits; hold |
| `02_validation.ipynb` | The table against its workbook formulas (A), against the signals recomputed from every bet (B), and how the session length is computed (C) |
| `03_three_errors.ipynb` | The three main errors: time of play, `days_since_bet` (later data) and `active_days_*` (playable bonus). No S3 |
| `04_gaming_daily.ipynb` | Audit of `gld_player_gaming_daily`: schema against the workbook, history, keys, row checks, distributions, impossible values, categories, production definition |
| `05_anomalies.ipynb` | Who, where and what of two anomalies of `gld_player_gaming_daily` (`play_secs` over 24 h, negative `won_real_eur`), with aggregates only |
| `06_extra_checks.ipynb`, `extra_checks.py` | Checks of the columns with no formula: load times, country and currency, categorical distributions, ids, snapshot date, tax |
| `gaming.py`, `gaming.yaml` | Checks and metrics of `gld_player_gaming_daily` |
| `audit.py` | Data: the config, the cache (S3 aggregates and local ones), every SQL aggregate, impossible and suspicious values per day and brand |
| `stats.py` | Statistics of the aggregates: bins, peaks, fits of the expected families, scorecards (all brands and per brand) |
| `plots.py` | Every figure, for the notebooks and the PDF |
| `evidence.py` | Evidence of the error list: recency, active days, session length, index limits, hold, status of every column |
| `recompute.py` | Sessions and time of play rebuilt from every bet of the landing transactions, one day at a time |
| `formula_check.py` | Runs the checks of `signal_checks.yaml` on snapshot days |
| `figures.py` | Writes the figures and tables of the error list and of the automatic report, from the cache (no S3) |
| `run_audit.py`, `make audit RUN=<run>` | The whole audit of a signals table in one command, with its report in Spanish and English |
| `settings.py`, `runs/*.yaml` | A run: which table (lake or ClickHouse), its dates, and its folders (`runs/<run>/`) |
| `clickhouse_mirror.py` | Local copy of a ClickHouse table, one Parquet file per day with `FINAL`, read as the lake table |
| `check_report.py`, `recompute_signal.py`, `player_signal_checks.yaml` | The audit of a table with its own catalogue of checks (player_signal) |
| `templates/` | The LaTeX of the automatic reports, with `<<name>>` for the numbers (`points_*`: the 25 points; `checks_*`: a table with its own catalogue) |
| `report.py`, `report_points.yaml` | The automatic report: the numbers, the rule of each of the 25 points, the texts (es, en) and the claims of the figures |
| `metrics.yaml` | The metrics of the table: expected shape, real-life range, `impossible` and `suspicious` rules |
| `metrics_recomputed.yaml` | The recomputed signals |
| `signal_checks.yaml` | Every column of the signals table and how it is checked |
| `informe/errores_tabla_senales.tex` | The error list (Spanish) |
| `informe/errores_tabla_juego.tex`, `informe/gaming_daily_errors.tex` | The error list of `gld_player_gaming_daily` (Spanish, English) |
| `tests/` | Tests on synthetic data (no S3) |
| `cache/`, `figures/` | Cached aggregates and figures (not in git) |

## How to run

```bash
aws sso login                                              # only to fill the cache
.venv/bin/python -m pytest signal_audit/tests -q
```

1. Open the notebooks from the `signal_audit/` folder with `RUN_S3 = True` the first time: the aggregates are saved in `cache/`, and after that `RUN_S3 = False` reads only the cache. The period is `period` in `metrics.yaml`; each period has its own cache folder.
2. Figures and the error list:

```bash
RUN_S3=0 .venv/bin/python signal_audit/figures.py
cd signal_audit/informe && xelatex errores_tabla_senales.tex && xelatex errores_tabla_senales.tex   # twice, for the links
```

## Audit of another signals table

A run is one file of `runs/`. To check another table with the same columns:

1. Copy `runs/example_clickhouse.yaml` to `runs/<run>.yaml`, and set the table (`database.table` in ClickHouse) and the period. The other dates have a default value.
2. Open the AWS SSO session and, for ClickHouse, the tunnel (host, user and password in `.env`).
3. `make audit RUN=<run>`.

The report is in `runs/<run>/informe/auditoria.pdf` (Spanish) and `audit.pdf` (English): the 25 points of the error list, each with its rule, its numbers and a verdict, plus the checks of the 165 columns, the impossible values and the figures. Each step keeps what is already done, so a run that stops can be started again. `runs/<run>/` is not in git; delete `runs/<run>/mirror/` (the local copy of the table) when the run is done.

A table with other columns gets its own catalogue of checks (`checks:` in the run): `player_signal_checks.yaml` for `whizdomai_org_gmntc_gld.player_signal`, the design table `gld_player_signal` of the workbook. Then the report lists every column against its workbook rule (`check_report.py`), and the rules on inputs the table does not keep are checked with those inputs recomputed from the daily caches, on day D and on D + 1 to D + 10 (`recompute_signal.py`). `make audit RUN=<run> COPY_ONLY=1` copies only the snapshot days and lists the columns that are missing or extra.

The rules of the points (thresholds) and their texts are in `report_points.yaml`. `runs/lake.yaml` is the lake table of the October 2026 report: with it, the automatic report gives the same numbers as `informe/errores_tabla_senales.tex`.

## Rules

- No sampling: every number is an aggregate over all the rows of the period.
- Only aggregates leave DuckDB and only aggregates are cached or shown; never player rows. The personal columns of the transactions are never read.
- S3 and ClickHouse are read only. The local copy of a ClickHouse table holds player rows: it stays outside git, and only aggregates leave it.
- To add a metric, add an entry to `metrics.yaml`; no code change is needed.
