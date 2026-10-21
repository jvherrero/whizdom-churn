# Signal Audit

Audit of the gold table `gld_player_signals_daily` (and the daily tables it comes from): distributions, impossible and suspicious values, its own formulas, and the signals recomputed from the raw transactions. The deliverable is the error list `informe/errores_tabla_senales.pdf`: each wrong column, the fix and a link to the evidence.

| File | What it is |
|---|---|
| `01_distributions.ipynb` | Shape, peaks, missing, impossible and suspicious values, all brands and brand by brand; index limits; hold |
| `02_validation.ipynb` | The table against its workbook formulas (A), against the signals recomputed from every bet (B), and how the session length is computed (C) |
| `03_three_errors.ipynb` | The three main errors: time of play, `days_since_bet` (later data) and `active_days_*` (playable bonus). No S3 |
| `audit.py` | Data: the config, the cache (S3 aggregates and local ones), every SQL aggregate, impossible and suspicious values per day and brand |
| `stats.py` | Statistics of the aggregates: bins, peaks, fits of the expected families, scorecards (all brands and per brand) |
| `plots.py` | Every figure, for the notebooks and the PDF |
| `evidence.py` | Evidence of the error list: recency, active days, session length, index limits, hold, status of every column |
| `recompute.py` | Sessions and time of play rebuilt from every bet of the landing transactions, one day at a time |
| `formula_check.py` | Runs the checks of `signal_checks.yaml` on snapshot days |
| `figures.py` | Writes the figures and tables of the error list, from the cache (no S3) |
| `metrics.yaml` | The metrics of the table: expected shape, real-life range, `impossible` and `suspicious` rules |
| `metrics_recomputed.yaml` | The recomputed signals |
| `signal_checks.yaml` | Every column of the signals table and how it is checked |
| `informe/errores_tabla_senales.tex` | The error list (Spanish) |
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

## Rules

- No sampling: every number is an aggregate over all the rows of the period.
- Only aggregates leave DuckDB and only aggregates are cached or shown; never player rows. The personal columns of the transactions are never read.
- S3 is read only.
- To add a metric, add an entry to `metrics.yaml`; no code change is needed.
