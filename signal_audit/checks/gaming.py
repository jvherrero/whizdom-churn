"""Audit of gld_player_gaming_daily (04_gaming_daily.ipynb): schema against the workbook, daily volume,
keys, row checks (formulas, identities and limits) per day and brand, categories, and the production
definition from ClickHouse.

Every number is an SQL aggregate over all the rows of the period (DuckDB on S3, no sampling), cached
like the rest of the audit; the schema and the daily extremes come from the Parquet metadata (no S3).
Only aggregates leave the database. Distributions and impossible values use audit.py with gaming.yaml.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from signal_audit import settings
from signal_audit.data import audit

CONFIG = settings.CONFIG / "gaming.yaml"
TABLE = "gld_player_gaming_daily"
WORKBOOK = settings.PROJECT_ROOT / "gld-schemas.xlsx"
KEY = ["tenant_id", "player_id", "game_id", "provider_code", "event_date", "currency"]


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def _key(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]
# ---------------------------------------------------------------- schema

def workbook_schema(sheet: str = TABLE, path: Path = WORKBOOK) -> pd.DataFrame:
    """The columns of the table in the workbook (data type, group, description, rule). Standard library only."""
    z = zipfile.ZipFile(path)
    strings = [html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", s, re.S)))
               for s in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode(), re.S)]
    book = z.read("xl/workbook.xml").decode()
    rels = z.read("xl/_rels/workbook.xml.rels").decode()
    rid = dict(re.findall(r'<sheet name="([^"]+)"[^>]*r:id="([^"]+)"', book))
    target = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))
    xml = z.read("xl/" + target[rid[sheet]].lstrip("/").replace("xl/", "")).decode()
    rows = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", xml, re.S):
        vals = []
        for c in re.findall(r"<c [^>]*?(?:/>|>.*?</c>)", row, re.S):
            v = re.search(r"<v>(.*?)</v>", c)
            t = re.search(r"<t[^>]*>(.*?)</t>", c, re.S)
            vals.append(strings[int(v.group(1))] if v and 't="s"' in c else v.group(1) if v else html.unescape(t.group(1)) if t else "")
        if vals and vals[0].isdigit() and len(vals) > 2:
            rows.append({"column": vals[1], "workbook type": vals[2], "group": vals[3] if len(vals) > 3 else "",
                         "description": next((x for x in vals[4:] if len(x) > 12), "")})
    return pd.DataFrame(rows)

def deployed_schema(table: str = TABLE) -> pd.DataFrame:
    """The columns of the deployed table from the Parquet metadata: type, first and last day, nulls."""
    f = pd.read_parquet(audit.FOOTERS / f"{table}.parquet")
    return f.groupby("column").agg(type=("type", "first"), first_day=("day", "min"), last_day=("day", "max"),
                                   nulls=("nulls", "sum")).reset_index()

# ---------------------------------------------------------------- history from the metadata (no S3)

def daily_rows(table: str = TABLE) -> pd.Series:
    """Rows per daily file (all operators): gaps and jumps in the volume."""
    f = pd.read_parquet(audit.FOOTERS / f"{table}.parquet")
    s = f.groupby("day")["n_rows"].first()
    s.index = pd.to_datetime(s.index)
    return s.asfreq("D")   # missing days show as NaN

def daily_extremes(columns: list[str], table: str = TABLE) -> pd.DataFrame:
    """Daily minimum and maximum of each column, from the metadata (all operators)."""
    out = []
    for c in columns:
        r = audit.footer_range(table, c)
        r.index = pd.to_datetime(r.index)
        out.append(r[["min_num", "max_num"]].add_prefix(f"{c}:"))
    return pd.concat(out, axis=1)
# ---------------------------------------------------------------- keys and versions

def key_check(cfg: dict) -> pd.DataFrame:
    """Per day and brand, on the raw files (before keeping the latest version): rows, distinct keys,
    keys with more than one version, and players in more than one brand or currency that day."""
    def compute():
        pfs, base = audit.s3_filesystem()
        files = audit.daily_files(pfs, base, TABLE)
        paths = ", ".join(f"'{audit.uri(pfs, files[d])}'" for d in audit.period_days(cfg) if d in files)
        brands = ", ".join(map(str, cfg["brands"]))
        k = ", ".join(KEY)
        return audit.connection().sql(f"""
            WITH r AS (SELECT {k}, brand_id, _version FROM read_parquet([{paths}], union_by_name = true)
                       WHERE brand_id IN ({brands})),
                 keys AS (SELECT {k}, any_value(brand_id) AS brand_id, count(*) AS n, count(DISTINCT _version) AS versions
                          FROM r GROUP BY ALL),
                 players AS (SELECT tenant_id, player_id, event_date, count(DISTINCT brand_id) AS brands,
                                    count(DISTINCT currency) AS currencies FROM r GROUP BY ALL),
                 per_key AS (SELECT event_date AS day, brand_id, sum(n) AS rows, count(*) AS keys,
                                    count_if(n > 1) AS keys_with_copies, count_if(versions > 1) AS keys_with_versions
                             FROM keys GROUP BY ALL),
                 per_day AS (SELECT event_date AS day, count_if(brands > 1) AS players_in_2_brands,
                                    count_if(currencies > 1) AS players_in_2_currencies FROM players GROUP BY ALL)
            SELECT * FROM per_key LEFT JOIN per_day USING (day) ORDER BY day, brand_id""").df()
    return audit.cached(cfg, f"gaming_keys__{_key(cfg['brands'])}", compute)
# ---------------------------------------------------------------- row checks

def _as_double(src: str) -> str:
    """The same rows with every DECIMAL column as DOUBLE. A product of two decimals (an amount in
    DECIMAL(18, 4) times an FX rate in DECIMAL(18, 8)) overflows DECIMAL(18) for large amounts; in
    DOUBLE it does not, and the checks compare within a tolerance anyway."""
    types = audit.connection().sql(f"DESCRIBE SELECT * FROM {src}").df()
    dec = types.loc[types["column_type"].str.startswith("DECIMAL"), "column_name"].tolist()
    if not dec:
        return src
    return f"(SELECT * REPLACE ({', '.join(f'CAST({c} AS DOUBLE) AS {c}' for c in dec)}) FROM {src})"

def _num(expr: str) -> str:
    return f"CAST(({expr}) AS DOUBLE)"

def row_checks(cfg: dict) -> pd.DataFrame:
    """Every check of gaming.yaml per day and brand, in one scan of the latest version of each row:
    rows where it applies (n), where it holds (ok), and where the column is above or below the expected
    value (identity checks). Long format: day, brand_id, check, n, ok, higher, lower."""
    checks = cfg["checks"]
    tol = cfg["tolerance"]
    names = sorted({w for c in checks for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f"{c['column']} {c['expr']} {c.get('where', '')}")})

    def compute():
        parts = []
        for i, c in enumerate(checks):
            w = c.get("where", "TRUE")
            if c["kind"] == "identity":
                t, e = _num(c["column"]), _num(c["expr"])
                both = f"({w}) AND {t} IS NOT NULL AND {e} IS NOT NULL"
                close = f"abs({t} - {e}) <= {tol['atol']} + {tol['rtol']} * abs({e})"
                parts += [f"count_if({both}) AS n_{i}", f"count_if({both} AND {close}) AS ok_{i}",
                          f"count_if({both} AND NOT {close} AND {t} > {e}) AS hi_{i}",
                          f"count_if({both} AND NOT {close} AND {t} < {e}) AS lo_{i}"]
            else:
                parts += [f"count_if(({w}) AND ({c['expr']}) IS NOT NULL) AS n_{i}", f"count_if(({w}) AND ({c['expr']})) AS ok_{i}",
                          f"0 AS hi_{i}", f"0 AS lo_{i}"]
        src = _as_double(audit.source(cfg, "gaming_rows", columns=names))
        return audit.connection().sql(f"SELECT event_date AS day, brand_id, {', '.join(parts)} FROM {src} "
                                      f"GROUP BY ALL ORDER BY ALL").df()

    wide = audit.cached(cfg, f"gaming_checks__{_key([checks, tol, cfg['brands']])}", compute)
    rows = []
    for i, c in enumerate(checks):
        label = c.get("label") or (f"{c['column']} = {c['expr']}" if c["kind"] == "identity" else c["expr"])
        group = c.get("group") or (c["column"] if c["kind"] == "identity" else "")
        rows.append(wide[["day", "brand_id"]].assign(check=label, column=c["column"], kind=c["kind"], source=c["source"], group=group,
                                                     note=c.get("note", ""), n=wide[f"n_{i}"], ok=wide[f"ok_{i}"],
                                                     higher=wide[f"hi_{i}"], lower=wide[f"lo_{i}"]))
    return pd.concat(rows, ignore_index=True)

def checks_summary(checks: pd.DataFrame, good: float = 99.0, fair: float = 95.0) -> pd.DataFrame:
    """Per check: pass % in all brands and in each brand, higher and lower %, days where it fails, verdict.
    A hypothesis that fails, or almost passes while another check of its group passes, is "rejected":
    the guess is wrong, not the table."""
    out = []
    for (label, col, kind, src, group, note), g in checks.groupby(["check", "column", "kind", "source", "group", "note"], sort=False):
        n, ok = g["n"].sum(), g["ok"].sum()
        row = {"check": label, "column": col, "kind": kind, "source": src, "group": group, "rows": int(n),
               "pass %": 100 * ok / n if n else np.nan,
               "higher %": 100 * g["higher"].sum() / n if n else np.nan, "lower %": 100 * g["lower"].sum() / n if n else np.nan}
        for b, gb in g.groupby("brand_id"):
            row[f"brand {b}"] = 100 * gb["ok"].sum() / gb["n"].sum() if gb["n"].sum() else np.nan
        day = g.groupby("day")[["n", "ok"]].sum()
        row["days failing"] = int(((day["ok"] / day["n"].where(day["n"] > 0)) < good / 100).sum())
        p = row["pass %"]
        row["verdict"] = "no data" if n == 0 else "passes" if p >= good else "almost" if p >= fair else "fails"
        row["note"] = note
        out.append(row)
    out = pd.DataFrame(out)
    passed = set(out.loc[out["verdict"] == "passes", "group"]) - {""}
    hyp = out["source"] == "hypothesis"
    out.loc[hyp & ((out["verdict"] == "fails") | ((out["verdict"] == "almost") & out["group"].isin(passed))), "verdict"] = "rejected"
    return out

def first_play_hours(cfg: dict) -> pd.DataFrame:
    """Rows per brand, UTC hour of the first play and days between its UTC date and event_date (0 = same
    day): if the rows of another day pile up in a few hours, event_date uses another time zone."""
    def compute():
        src = audit.source(cfg, "gaming_rows", columns=["first_played_at"])
        return audit.connection().sql(f"SELECT brand_id, hour(first_played_at) AS hour, "
                                      f"event_date - CAST(first_played_at AS DATE) AS day_shift, count(*) AS rows "
                                      f"FROM {src} WHERE first_played_at IS NOT NULL GROUP BY ALL ORDER BY ALL").df()
    return audit.cached(cfg, f"gaming_first_play_hours__{_key([cfg['brands'], 'utc'])}", compute)
# ---------------------------------------------------------------- anomalies (aggregates only, no player id)

def _rows(cfg: dict, *exprs: str) -> str:
    """Latest version of each row, with the columns that the expressions use."""
    names = sorted({w for e in exprs for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", e)})
    return _as_double(audit.source(cfg, "gaming_rows", columns=names))

def _anomaly_key(cfg: dict, *parts) -> str:
    return _key([*parts, cfg["brands"], [str(d) for d in audit.period_days(cfg)]])

def anomaly_by(cfg: dict, cond: str, column: str, top: int = 20) -> pd.DataFrame:
    """Per value of `column`: rows, anomalous rows (cond), players and anomalous players, and the share
    of the anomalies against the share of all rows (lift > 1 = over-represented). Top values only."""
    def compute():
        return audit.connection().sql(f"""
            SELECT CAST({column} AS VARCHAR) AS value, count(*) AS rows, count_if({cond}) AS bad_rows,
                   count(DISTINCT player_id) AS players, count(DISTINCT player_id) FILTER (WHERE {cond}) AS bad_players
            FROM {_rows(cfg, cond, column)} GROUP BY ALL""").df()
    t = audit.cached(cfg, f"gaming_anomaly_by__{_anomaly_key(cfg, cond, column)}", compute)
    t = t.assign(bad_pct=100 * t["bad_rows"] / t["rows"],
                 share_of_bad=100 * t["bad_rows"] / max(t["bad_rows"].sum(), 1),
                 share_of_rows=100 * t["rows"] / t["rows"].sum())
    t["lift"] = t["share_of_bad"] / t["share_of_rows"]
    return t.sort_values("bad_rows", ascending=False).head(top).reset_index(drop=True)

def anomaly_players(cfg: dict, cond: str) -> pd.DataFrame:
    """How concentrated the anomalies are: per player (inside DuckDB) the anomalous rows and days, then
    only aggregates leave: players, share of the top 1 / 10 / 100, and players by days with an anomaly."""
    def compute():
        return audit.connection().sql(f"""
            WITH p AS (SELECT player_id, count(*) AS bad_rows, count(DISTINCT event_date) AS bad_days
                       FROM {_rows(cfg, cond)} WHERE {cond} GROUP BY player_id),
                 r AS (SELECT *, row_number() OVER (ORDER BY bad_rows DESC) AS rk FROM p)
            SELECT count(*) AS players, sum(bad_rows) AS bad_rows,
                   100 * sum(bad_rows) FILTER (WHERE rk <= 1) / sum(bad_rows) AS top1_pct,
                   100 * sum(bad_rows) FILTER (WHERE rk <= 10) / sum(bad_rows) AS top10_pct,
                   100 * sum(bad_rows) FILTER (WHERE rk <= 100) / sum(bad_rows) AS top100_pct,
                   count_if(bad_days = 1) AS players_1_day, count_if(bad_days BETWEEN 2 AND 5) AS players_2_5_days,
                   count_if(bad_days BETWEEN 6 AND 15) AS players_6_15_days, count_if(bad_days > 15) AS players_over_15_days
            FROM r""").df()
    return audit.cached(cfg, f"gaming_anomaly_players__{_anomaly_key(cfg, cond)}", compute)

def anomaly_values(cfg: dict, cond: str, expr: str, top: int = 20) -> pd.DataFrame:
    """Most frequent values of `expr` in the anomalous rows (repeated or round values point to a default
    or a bug), with rows and players."""
    def compute():
        return audit.connection().sql(f"""
            SELECT {expr} AS value, count(*) AS rows, count(DISTINCT player_id) AS players
            FROM {_rows(cfg, cond, expr)} WHERE {cond} GROUP BY ALL ORDER BY rows DESC LIMIT {top}""").df()
    return audit.cached(cfg, f"gaming_anomaly_values__{_anomaly_key(cfg, cond, expr, top)}", compute)

def anomaly_quantiles(cfg: dict, cond: str, exprs: dict[str, str]) -> pd.DataFrame:
    """Quantiles (1, 10, 50, 90, 99 %) of each expression in the anomalous rows and in the normal rows."""
    def compute():
        sel = ", ".join(f"quantile_cont({e}, [0.01, 0.1, 0.5, 0.9, 0.99]) AS \"{n}\"" for n, e in exprs.items())
        return audit.connection().sql(f"SELECT ({cond}) AS anomalous, {sel} FROM {_rows(cfg, cond, *exprs.values())} "
                                      f"GROUP BY ALL").df()
    t = audit.cached(cfg, f"gaming_anomaly_quantiles__{_anomaly_key(cfg, cond, exprs)}", compute)
    q = ["p1", "p10", "p50", "p90", "p99"]
    return pd.concat({n: pd.DataFrame(t[n].tolist(), index=t["anomalous"], columns=q) for n in exprs}, names=["expr", "anomalous"])

# ---------------------------------------------------------------- categories

def categories(cfg: dict, columns=("dq_status", "fx_is_provisional", "currency", "provider_code", "source_system")) -> pd.DataFrame:
    """Rows per brand and value of each categorical column (latest version of each row)."""
    cols = list(columns)

    def compute():
        src = audit.source(cfg, "gaming_rows", columns=cols)
        sel = " UNION ALL ".join(f"SELECT '{c}' AS col, CAST({c} AS VARCHAR) AS value, brand_id, count(*) AS rows "
                                 f"FROM r0 GROUP BY ALL" for c in cols)
        return audit.connection().sql(f"WITH r0 AS (SELECT * FROM {src}) {sel} ORDER BY col, rows DESC").df()
    return audit.cached(cfg, f"gaming_categories__{_key([cols, cfg['brands']])}", compute)
# ---------------------------------------------------------------- the production definition (ClickHouse)


# ================================================================ used by the notebooks only

def schema_diff(table: str = TABLE) -> pd.DataFrame:
    """Every column of the workbook or of the deployed table, and where it is."""
    wb, dep = workbook_schema(table), deployed_schema(table)
    out = wb.merge(dep, on="column", how="outer", indicator=True)
    out["where"] = out.pop("_merge").map({"both": "both", "left_only": "workbook only", "right_only": "deployed only"})
    return out.sort_values(["where", "column"]).reset_index(drop=True)


def game_verticals(cfg: dict) -> pd.DataFrame:
    """Rows, players and anomalies per vertical and game type of the game catalogue (gld_game_profile of
    the last day of the period, joined on tenant and game_id). `verticals_per_game` > 1 means a game_id
    with two verticals in the catalogue (it has no platform here, so the join cannot tell them apart)."""
    def compute():
        pfs, base = audit.s3_filesystem()
        files = audit.daily_files(pfs, base, "gld_game_profile")
        day = max(d for d in files if d <= audit.period_days(cfg)[-1])
        return audit.connection().sql(f"""
            WITH g AS (SELECT tenant_id, game_id, count(DISTINCT vertical) AS n_vertical,
                              any_value(vertical) AS vertical, any_value(game_type) AS game_type
                       FROM read_parquet('{audit.uri(pfs, files[day])}') GROUP BY ALL)
            SELECT COALESCE(g.vertical, '(not in catalogue)') AS vertical, COALESCE(g.game_type, '') AS game_type,
                   max(g.n_vertical) AS verticals_per_game, count(*) AS rows, count(DISTINCT r.player_id) AS players,
                   count(DISTINCT r.game_id) AS games, count_if(r.play_secs > 86400) AS play_over_day,
                   count_if(r.won_real_eur < 0) AS negative_wins, sum(r.turnover_eur) AS turnover_eur
            FROM {_rows(cfg, "play_secs won_real_eur turnover_eur game_id tenant_id")} r
            LEFT JOIN g ON r.tenant_id = g.tenant_id AND r.game_id = g.game_id
            GROUP BY ALL ORDER BY rows DESC""").df()
    return audit.cached(cfg, f"gaming_verticals__{_anomaly_key(cfg)}", compute)


def clickhouse_definition(table: str = TABLE, database: str | None = None) -> pd.DataFrame:
    """The CREATE statement of the table and of every view or materialized view that writes to it or
    reads from it, from ClickHouse system tables (metadata only, never data). Needs CLICKHOUSE_HOST,
    CLICKHOUSE_PORT, CLICKHOUSE_USER and CLICKHOUSE_PASSWORD in the environment or in .env."""
    import clickhouse_connect
    from dotenv import load_dotenv

    load_dotenv(settings.PROJECT_ROOT / ".env")
    client = clickhouse_connect.get_client(host=os.environ["CLICKHOUSE_HOST"], port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
                                           username=os.environ["CLICKHOUSE_USER"], password=os.environ["CLICKHOUSE_PASSWORD"],
                                           secure=os.getenv("CLICKHOUSE_SECURE", "false").lower() == "true")
    where = f"AND database = '{database}'" if database else ""
    return client.query_df(f"""
        SELECT database, name, engine, create_table_query
        FROM system.tables
        WHERE (name = '{table}' OR create_table_query ILIKE '%{table}%') {where}
        ORDER BY engine, database, name""")
