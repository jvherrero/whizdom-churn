"""Signals recomputed from the raw transactions, to compare with the signals table.

Used by 02_validation.ipynb (part B). The session and time-of-play signals of gld_player_signals_daily
are wrong (sessions of 1 second and of 28 days, a night share up to 4), and the daily tables only hold
daily totals, so they are rebuilt here from every bet of the landing transactions (org/10-landing, one
row per Kafka message, from 2026-06-18).

Definitions (signal_audit/metrics_recomputed.yaml, section `recompute`):
    bet       a GAME_BET transaction of the five brands, not rolled back, one per uuid
    session   the bets of one player with no gap longer than the timeout (30 minutes; 15 and 60 to test it)
    duration  last bet minus first bet of a session with two bets or more; one-bet sessions are counted apart
    night     00:00 to 06:00 in the local time of the currency of the bet; night share = night stake / stake

Each event day d is built from the landing folders d - 1, d and d + 1 (folders follow the arrival day,
and a few events arrive one day late), with the bets from d - 2 hours to d + 1 day + 12 hours. Only the
sessions that start on d are kept, so a session that crosses midnight is counted once and whole.

Per day, local files in signal_audit/cache/recomputed/ (not in git):
    sessions_<d>.parquet     one row per session and timeout: brand and player ids, currency, start,
                             stop, bets, duration (the player id is needed for the gaps between sessions)
    player_day_<d>.parquet   one row per player and day: ids, bets, stake, night stake, sessions.
    time_of_play_<d>.parquet one row per player, local hour and local weekday with a stake (build_time_of_play):
                             the base of the weekend share and the hourly entropy.
These are local caches, like the activity cache of the churn project: numeric ids, no personal data.
Only aggregates are shown. The personal columns of the transactions (name, email, ip...) are never read:
the query selects only the columns it needs.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import sys
from pathlib import Path

import pandas as pd
from pyarrow import fs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "features"))
from datalake import s3_filesystem, uri  # noqa: E402

import audit  # noqa: E402

LANDING = "org/10-landing"
TRANSACTIONS = ["kafka-sink/primus/whizdomai-transactions", "kafka-sink/secundus/whizdomai-transactions"]
# Only these columns are read from the transactions: no personal data.
COLUMNS = ["uuid", "accountId", "brandId", "dateTime", "tranType", "currency", "rolledBack",
           "amountReal", "amountReleasedBonus", "amountPlayableBonus"]
_DAY = re.compile(r"(\d{4})/(\d{2})/(\d{2})/")


def out_dir() -> Path:
    return audit.local_root() / "recomputed"


def landing_files(table: str) -> dict[dt.date, list[str]]:
    """Parquet files of each landing folder day (several files per day)."""
    pfs, base = s3_filesystem(LANDING)
    out: dict[dt.date, list[str]] = {}
    for info in pfs.get_file_info(fs.FileSelector(f"{base}/{table}", recursive=True, allow_not_found=True)):
        m = _DAY.search(info.path[len(f"{base}/{table}"):] + "/") if info.path.endswith(".parquet") else None
        if m:
            out.setdefault(dt.date(*map(int, m.groups())), []).append(uri(pfs, info.path))
    return out


def bets_sql(cfg: dict, day: dt.date, files: dict[str, dict], id_column: str = "accountId") -> str:
    """Bets of the five brands around `day`, one row per uuid, stake in all wallets."""
    folders = [day - dt.timedelta(days=1), day, day + dt.timedelta(days=1)]
    paths = [p for t in TRANSACTIONS for d in folders for p in files[t].get(d, [])]
    if not paths:
        raise FileNotFoundError(f"no landing transactions around {day}")
    listing = ", ".join(f"'{p}'" for p in paths)
    lo = dt.datetime.combine(day, dt.time()) - dt.timedelta(hours=2)
    hi = dt.datetime.combine(day + dt.timedelta(days=1), dt.time()) + dt.timedelta(hours=12)
    brands = ", ".join(map(str, cfg["brands"]))
    return f"""
        SELECT uuid, brandId AS brand_id, {id_column} AS player_id, currency, dateTime AS ts,
               abs(COALESCE(amountReal, 0) + COALESCE(amountReleasedBonus, 0) + COALESCE(amountPlayableBonus, 0))::DOUBLE AS stake
        FROM read_parquet([{listing}], union_by_name = true)
        WHERE tranType = 'GAME_BET' AND NOT COALESCE(rolledBack, false) AND brandId IN ({brands})
          AND dateTime >= TIMESTAMPTZ '{lo}+00' AND dateTime < TIMESTAMPTZ '{hi}+00'
        QUALIFY row_number() OVER (PARTITION BY uuid) = 1"""


def _local_hour(cfg: dict) -> str:
    zones = cfg["recompute"]["timezones"]
    cases = " ".join(f"WHEN '{c}' THEN hour(timezone('{z}', ts))" for c, z in zones.items())
    return f"CASE currency {cases} ELSE hour(timezone('UTC', ts)) END"


def build_day(cfg: dict, day: dt.date, files: dict[str, dict], con=None) -> None:
    """Sessions and player-day totals of one event day, saved locally."""
    rc = cfg["recompute"]
    con = con or audit.connection()
    night_lo, night_hi = rc["night_hours"]
    timeouts = rc["timeouts_min"]
    d0, d1 = f"TIMESTAMPTZ '{day} 00:00:00+00'", f"TIMESTAMPTZ '{day + dt.timedelta(days=1)} 00:00:00+00'"
    con.sql(f"CREATE OR REPLACE TEMP TABLE bets AS {bets_sql(cfg, day, files)}")
    con.sql("""CREATE OR REPLACE TEMP TABLE gaps AS
               SELECT *, epoch(ts) - epoch(lag(ts) OVER (PARTITION BY brand_id, player_id ORDER BY ts)) AS gap_s
               FROM bets""")
    session_ids = ", ".join(
        f"sum(CASE WHEN gap_s IS NULL OR gap_s > {60 * t} THEN 1 ELSE 0 END) "
        f"OVER (PARTITION BY brand_id, player_id ORDER BY ts ROWS UNBOUNDED PRECEDING) AS s{t}" for t in timeouts)
    con.sql(f"CREATE OR REPLACE TEMP TABLE marked AS SELECT *, {session_ids} FROM gaps")
    parts = []
    for t in timeouts:
        parts.append(f"""
            SELECT {t} AS timeout_min, brand_id, player_id, any_value(currency) AS currency, s{t} AS session,
                   min(ts) AS start, max(ts) AS stop, count(*) AS bets
            FROM marked GROUP BY brand_id, player_id, s{t}""")
    con.sql(f"CREATE OR REPLACE TEMP TABLE sess AS {' UNION ALL '.join(parts)}")
    con.sql("CREATE OR REPLACE TEMP TABLE sess_day AS SELECT *, epoch(stop) - epoch(start) AS duration_s FROM sess")
    out_dir().mkdir(parents=True, exist_ok=True)
    # only the sessions that start on d
    con.sql(f"""SELECT DATE '{day}' AS day, timeout_min, brand_id, player_id, currency, start, stop, bets, duration_s
                FROM sess_day WHERE start >= {d0} AND start < {d1}""").write_parquet(str(out_dir() / f"sessions_{day}.parquet"))
    hour = _local_hour(cfg)
    per_timeout = ", ".join(
        f"count(*) FILTER (WHERE timeout_min = {t}) AS sessions_{t}, "
        f"count(*) FILTER (WHERE timeout_min = {t} AND bets = 1) AS single_bet_sessions_{t}, "
        f"sum(duration_s) FILTER (WHERE timeout_min = {t} AND bets > 1) AS duration_sum_{t}" for t in timeouts)
    con.sql(f"""
        WITH b AS (SELECT brand_id, player_id, any_value(currency) AS currency, count(*) AS bets, sum(stake) AS stake,
                          sum(stake) FILTER (WHERE {hour} >= {night_lo} AND {hour} < {night_hi}) AS night_stake
                   FROM bets WHERE ts >= {d0} AND ts < {d1} GROUP BY ALL),
             s AS (SELECT brand_id, player_id, {per_timeout}
                   FROM sess_day WHERE start >= {d0} AND start < {d1} GROUP BY ALL)
        SELECT DATE '{day}' AS day, b.*, COALESCE(s.sessions_30, 0) AS sessions,
               {', '.join(f's.sessions_{t}, s.single_bet_sessions_{t}, s.duration_sum_{t}' for t in timeouts)}
        FROM b LEFT JOIN s USING (brand_id, player_id)""").write_parquet(str(out_dir() / f"player_day_{day}.parquet"))
    for table in ("bets", "gaps", "marked", "sess", "sess_day"):
        con.sql(f"DROP TABLE IF EXISTS {table}")


def local_time_sql(cfg: dict) -> str:
    """The local time of a bet, from the time zone of its currency (UTC when not listed)."""
    zones = cfg["recompute"]["timezones"]
    cases = " ".join(f"WHEN '{c}' THEN timezone('{z}', ts)" for c, z in zones.items())
    return f"CASE currency {cases} ELSE timezone('UTC', ts) END"


def build_time_of_play_day(cfg: dict, day: dt.date, files: dict[str, dict], con=None) -> None:
    """Stake of each player by local hour and local day of the week, for the bets of one UTC day.
    Long format: one row per player, local hour and local weekday with a stake. No session is needed,
    so the bets of the day are enough (folders d and d + 1, for the late ones)."""
    con = con or audit.connection()
    d0, d1 = f"TIMESTAMPTZ '{day} 00:00:00+00'", f"TIMESTAMPTZ '{day + dt.timedelta(days=1)} 00:00:00+00'"
    local = local_time_sql(cfg)
    out_dir().mkdir(parents=True, exist_ok=True)
    con.sql(f"""SELECT DATE '{day}' AS day, brand_id, player_id, any_value(currency) AS currency,
                       hour({local}) AS local_hour, isodow({local}) AS local_dow, sum(stake) AS stake, count(*) AS bets
                FROM ({bets_sql(cfg, day, files)}) WHERE ts >= {d0} AND ts < {d1}
                GROUP BY ALL""").write_parquet(str(out_dir() / f"time_of_play_{day}.parquet"))


def build_time_of_play(cfg: dict, days: list[dt.date] | None = None) -> list[dt.date]:
    """Build the missing days of the time-of-play files (RUN_S3=1 needed)."""
    days = days or audit.period_days(cfg)
    todo = [d for d in days if not (out_dir() / f"time_of_play_{d}.parquet").exists()]
    if not todo:
        return []
    if os.environ.get("RUN_S3") != "1":
        raise FileNotFoundError(f"{len(todo)} day(s) not built: run with RUN_S3=1 (needs the AWS SSO session)")
    files = {t: landing_files(t) for t in TRANSACTIONS}
    for d in todo:
        build_time_of_play_day(cfg, d, files)
        print(f"built time of play {d}", flush=True)
    return todo


def build(cfg: dict, days: list[dt.date] | None = None) -> list[dt.date]:
    """Build the missing days (RUN_S3=1 needed). Returns the days built."""
    days = days or audit.period_days(cfg)
    out = out_dir()
    todo = [d for d in days if not ((out / f"sessions_{d}.parquet").exists() and (out / f"player_day_{d}.parquet").exists())]
    if not todo:
        return []
    if os.environ.get("RUN_S3") != "1":
        raise FileNotFoundError(f"{len(todo)} day(s) not built: run with RUN_S3=1 (needs the AWS SSO session)")
    files = {t: landing_files(t) for t in TRANSACTIONS}
    for d in todo:
        build_day(cfg, d, files)
        print(f"built {d}", flush=True)
    return todo


def key_check(cfg: dict, day: dt.date, id_column: str = "accountId") -> pd.DataFrame:
    """Is `id_column` of the transactions (accountId or partyId) the player_id of the gold tables?
    Players with a bet on `day` in each source, per brand, and how many are in both. Aggregates only.
    On 2026-09-15 accountId matched almost no player_id: it is not the player id."""
    def compute():
        con = audit.connection()
        files = {t: landing_files(t) for t in TRANSACTIONS}
        con.sql(f"CREATE OR REPLACE TEMP TABLE kb AS SELECT DISTINCT brand_id, player_id FROM ({bets_sql(cfg, day, files, id_column)}) "
                f"WHERE CAST(ts AS DATE) = DATE '{day}'")
        pfs, base = s3_filesystem()
        gold = audit.daily_files(pfs, base, "gld_player_gaming_daily")[day]
        con.sql(f"CREATE OR REPLACE TEMP TABLE kg AS SELECT DISTINCT brand_id, player_id "
                f"FROM read_parquet('{uri(pfs, gold)}') WHERE bets > 0 AND brand_id IN ({', '.join(map(str, cfg['brands']))})")
        out = con.sql("""
            SELECT COALESCE(l.brand_id, g.brand_id) AS brand_id,
                   count(l.player_id) AS landing_players, count(g.player_id) AS gold_players,
                   count(*) FILTER (WHERE l.player_id IS NOT NULL AND g.player_id IS NOT NULL) AS in_both
            FROM kb l FULL JOIN kg g ON l.brand_id = g.brand_id AND l.player_id = CAST(g.player_id AS BIGINT)
            GROUP BY ALL ORDER BY brand_id""").df()
        con.sql("DROP TABLE kb; DROP TABLE kg")
        return out
    return audit.cached(cfg, f"key_check_{day}" + ("" if id_column == "accountId" else f"_{id_column}"), compute)
