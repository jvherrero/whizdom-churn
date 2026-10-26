# Signal Audit

Audit of the gold signals tables: does each column follow its definition in the data model workbook, and does it give the same value when it is computed again from the daily tables and the transactions? The results are error lists, in English (in git) and Spanish (local).

## Folders

| Folder | What is in it |
|---|---|
| `run_audit.py` | The whole audit of one table in one command (`make audit RUN=<run>`) |
| `settings.py` | The folders, and the run: which table, which dates, which catalogue (from `config/runs/<run>.yaml`) |
| `config/` | Every YAML: the metrics and their ranges (`metrics.yaml`, `metrics_recomputed.yaml`, `gaming.yaml`), the checks of each column (`signal_checks.yaml` for `gld_player_signals_daily`, `player_signal_checks.yaml` for `player_signal`), the 25 points of the automatic report (`report_points.yaml`) and the runs (`runs/`) |
| `data/` | Reading the data: SQL aggregates and their cache (`audit.py`), the local copy of a ClickHouse table (`clickhouse_mirror.py`), the sessions rebuilt from the landing transactions (`recompute.py`) |
| `checks/` | The checks: the rules of each column (`formula_check.py`), the evidence of the errors (`evidence.py`), the columns with no formula (`extra_checks.py`), the rules on recomputed inputs (`recompute_signal.py`), the gaming table (`gaming.py`), the statistics of the distributions (`stats.py`) |
| `report/` | The reports: the automatic report of the 25 points (`report.py`) and by checks (`check_report.py`), the figures (`figures.py`, `plots.py`), the LaTeX tables (`tables.py`), the gaming report (`gaming_report.py`) and the LaTeX templates (`templates/`) |
| `reports/` | The documents: `signals_daily/` (the error list of `gld_player_signals_daily` and the review script) and `gaming_daily/` (the error list of `gld_player_gaming_daily`) |
| `notebooks/` | The exploration behind the October 2026 reports (01 to 06) |
| `tests/` | Tests on synthetic data, with no S3 and no ClickHouse |
| `output/` | Everything generated, not in git: `cache/` (aggregates of the lake run), `figures/` (figures of the October reports), `runs/<run>/` (cache, figures, automatic report and, for ClickHouse, the local copy of the table) |

`signal_audit` is a package: the modules import each other as `from signal_audit.data import audit`, and `__init__.py` puts `src/features` and `eda` on the path.

## Commands

From the repo root:

```bash
aws sso login                                   # S3 (and, for ClickHouse, open the tunnel: .env)
make audit RUN=lake                             # the lake table of the October 2026 report
make audit RUN=player_signal                    # whizdomai_org_gmntc_gld.player_signal in ClickHouse
make audit RUN=player_signal COPY_ONLY=1        # only copy the snapshot days, to see the columns
.venv/bin/python -m pytest signal_audit/tests -q
```

The automatic report goes to `output/runs/<run>/informe/` (`auditoria.pdf`, `audit.pdf`). Each step keeps what is already done, so a run that stops can be started again.

To audit another table, copy `config/runs/example_clickhouse.yaml` to `config/runs/<run>.yaml` and set the table and the period. A table with the 165 columns of `gld_player_signals_daily` gets the report of the 25 points; a table with other columns needs its own catalogue (`checks:` in the run, like `player_signal_checks.yaml`), and then the report lists every column against its workbook rule.

The October 2026 reports, from the cache (no S3):

```bash
RUN_S3=0 .venv/bin/python -m signal_audit.report.figures         # figures and tables of reports/signals_daily/
RUN_S3=0 .venv/bin/python -m signal_audit.report.gaming_report   # figures and tables of reports/gaming_daily/
cd signal_audit/reports/signals_daily && xelatex signal_table_errors.tex && xelatex signal_table_errors.tex
```

The notebooks run from `notebooks/` with `RUN_S3 = False` once the cache is filled.

## Rules

- No sampling: every number is an aggregate over all the rows of the period.
- Only aggregates leave DuckDB and only aggregates are cached or shown; never player rows. The personal columns of the transactions are never read.
- S3 and ClickHouse are read only. The local copy of a ClickHouse table holds player rows: it stays in `output/`, outside git, and only aggregates leave it.
