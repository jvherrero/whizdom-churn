"""The run of the audit (AUDIT_RUN, default lake): runs/<run>.yaml gives the table, its dates and its catalogue.

The audit always calls the table gld_player_signals_daily; a run changes where its files come from (the lake,
or a local copy of a ClickHouse table) and where the results go (runs/<run>/; the lake run keeps its cache in
cache/, shared with the notebooks). Importing it sets LOCAL_TABLES and SIGNALS_CACHE_DIR for src/features.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TABLE = "gld_player_signals_daily"
NAME = os.environ.get("AUDIT_RUN", "lake")
RUN = yaml.safe_load((HERE / "runs" / f"{NAME}.yaml").read_text(encoding="utf-8"))


def _day(v) -> dt.date:
    return pd.Timestamp(v).date()


def _range(v) -> dict[str, dt.date]:
    return {k: _day(v[k]) for k in ("start", "end")}


PERIOD = _range(RUN["period"])                                   # the month of the checks
SNAPSHOTS = [_day(d) for d in RUN.get("snapshots") or [PERIOD["start"] + dt.timedelta(days=14), PERIOD["end"]]]
LONG = _range(RUN.get("long") or RUN["period"])                  # frozen days and the maps per day
EVIDENCE_DAYS = [_day(d) for d in RUN.get("evidence_days") or pd.date_range(PERIOD["start"], PERIOD["end"], freq="7D")]
HISTORY_START = _day(RUN.get("history_start") or PERIOD["start"])
MONTH_STARTS = [d.date() for d in pd.date_range(HISTORY_START, max(PERIOD["end"], *SNAPSHOTS), freq="MS")]
LIVE_FROM = _day(RUN["live_from"]) if RUN.get("live_from") else None   # start of the daily process, when known
TP_EVENTS = {_day(d): text for d, text in (RUN.get("tp_events") or {}).items()}   # notes on the time-of-play figure

CHECKS = HERE / RUN.get("checks", "signal_checks.yaml")          # the checks of each column
CATALOGUE = "checks" in RUN       # a table with its own catalogue: a report by checks, not the 25 points
CONTRACT = PROJECT_ROOT / RUN.get("contract", "configs/contracts/lake/gld_player_signals_daily.yaml")

ROOT = HERE / "runs" / NAME
FIGURES, TEX = ROOT / "figures", ROOT / "informe"
RECOMPUTED = HERE / "cache" / "recomputed"                       # sessions from landing: the same for every run
if NAME == "lake":
    CACHE = HERE / "cache"
    FOOTERS = PROJECT_ROOT / "data/02_intermediate/profile_tables/footers"
else:
    CACHE = ROOT / "cache"
    FOOTERS = CACHE / "footers"
    os.environ["SIGNALS_CACHE_DIR"] = str(ROOT / "signals")
MIRROR = ROOT / "mirror"
if RUN["source"] == "clickhouse":
    os.environ["LOCAL_TABLES"] = f"{TABLE}={MIRROR}"
