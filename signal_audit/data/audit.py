"""Data of the signal audit: the config, the cache, and every SQL aggregate over the gold tables.

Every number comes from SQL aggregates over all the rows of the period (DuckDB on S3, no sampling); only
aggregates leave the database, never player rows. They are cached as small Parquet files in output/cache/, so
the notebooks run again without S3. Statistics are in stats.py, figures in plots.py.

    from signal_audit.data import audit
    cfg = audit.load_config()
    prof, hist = audit.metric_data(cfg, "turnover_player_day")      # needs RUN_S3=1 the first time
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from signal_audit import settings
# src/features and eda are on the path through signal_audit/__init__.py
from datalake import daily_files, s3_duckdb, s3_filesystem, uri

CONFIG = settings.CONFIG / "metrics.yaml"
CACHE = settings.CACHE                 # one folder per run (settings.py)
FOOTERS = settings.FOOTERS
# Local activity cache of the churn project: gives the brand of each (tenant_id, player_id) for the
# tables without brand_id. tenant_id is the operator (a UUID), not the brand.
ACTIVITY = settings.PROJECT_ROOT / "data/02_intermediate/daily_cache/activity"
QUANTILES = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]


# ---------------------------------------------------------------- config and cache

def load_config(path: Path = CONFIG) -> dict:
    """A config; the signals configs (metrics.yaml, metrics_recomputed.yaml) take the period of the run."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if path.name.startswith("metrics"):
        cfg["period"] = dict(settings.PERIOD)
    return cfg


def local_root() -> Path:
    """Folder of the local files of the audit (the signals recomputed from landing, recompute.py). They do not
    depend on the signals table, so every run reads them from the lake cache."""
    return CACHE if settings.NAME == "lake" else settings.RECOMPUTED.parent


def period_days(cfg: dict) -> list[dt.date]:
    if cfg.get("days"):   # a list of days instead of the whole period
        return sorted(pd.Timestamp(d).date() for d in cfg["days"])
    start, end = (pd.Timestamp(cfg["period"][k]).date() for k in ("start", "end"))
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


# Fields of a metric that change its SQL. A change gives a new cache file, so old results are never reused.
SQL_FIELDS = ("table", "expr", "where", "scale", "width", "cap", "range")


def spec_hash(cfg: dict, name: str) -> str:
    m = cfg["metrics"][name]
    t = cfg["tables"][m["table"]]
    spec = {k: m.get(k) for k in SQL_FIELDS} | {"base": t.get("base"), "brands": cfg["brands"]}
    if t.get("source_table"):
        spec["source_table"] = t["source_table"]
    return hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _cache_path(cfg: dict, name: str) -> Path:
    p = cfg["period"]
    return CACHE / f"{p['start']}_{p['end']}" / f"{name}.parquet"


def cached(cfg: dict, name: str, compute, local: bool = False) -> pd.DataFrame:
    """Read an aggregate from the cache, or compute it on S3 when RUN_S3=1 and save it. Aggregates of
    local files are computed every time (fast, and the local files can be rebuilt)."""
    if local:
        return compute()
    path = _cache_path(cfg, name)
    if path.exists():
        return pd.read_parquet(path)
    if os.environ.get("RUN_S3") != "1":
        raise FileNotFoundError(f"{path.name} is not cached: run with RUN_S3=1 (needs the AWS SSO session)")
    out = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def cached_local(name: str, compute, refresh: bool = False):
    """A frame, or a tuple of frames, computed from local files only (no S3) and saved in output/cache/local/ (the cache of the run)."""
    folder = CACHE / "local"
    paths = sorted(folder.glob(f"{name}__*.parquet"))
    if paths and not refresh:
        frames = [pd.read_parquet(p) for p in paths]
        return frames[0] if len(frames) == 1 else tuple(frames)
    out = compute()
    folder.mkdir(parents=True, exist_ok=True)
    for p in paths:
        p.unlink()
    for k, f in enumerate(out if isinstance(out, tuple) else (out,)):
        f.to_parquet(folder / f"{name}__{k}.parquet", index=False)
    return out


# ---------------------------------------------------------------- reading the rows

_CON = None


def connection():
    """DuckDB connection. With RUN_S3=1 it can read S3 (needs the AWS SSO session); otherwise it reads
    local files only, so the cached results can be used with no AWS session."""
    global _CON
    if _CON is None:
        _CON = s3_duckdb() if os.environ.get("RUN_S3") == "1" or os.environ.get("DATALAKE_ROOT") else duckdb.connect()
        _CON.sql("SET TimeZone = 'UTC'")   # the lake timestamps are UTC: CAST(ts AS DATE) must give the UTC day
    return _CON


def _columns_used(cfg: dict, table: str) -> list[str]:
    """Columns of the table that the metrics, segments and base of this table need."""
    t = cfg["tables"][table]
    text = " ".join([t.get("base", ""), *(f"{m['expr']} {m.get('where', '')} {' '.join(m.get('segments', []))}"
                                          for m in cfg["metrics"].values() if m["table"] == table)])
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    return sorted(words | set(t["key"]) | {t["date"], "tenant_id", "player_id", "brand_id"})


def is_local(cfg: dict, table: str) -> bool:
    return bool(cfg["tables"][table].get("local"))


def source(cfg: dict, table: str, fill_brand: bool = False, columns: list[str] | None = None) -> str:
    """SQL of the rows of the period: brands only, latest _version of each key, then the base grain.

    A table with `local` reads local files instead (a path with {day}, inside the cache folder), with
    no deduplication or brand filter (they are built that way). A table with `source_table` reads the
    files of that lake table, so one lake table can have several grains (`base`). With `fill_brand`, a
    row with an empty brand_id takes the brand of the player from the activity cache (and is kept),
    and `brand_filled` says which rows got it that way. `columns` adds columns that no metric uses."""
    t = cfg["tables"][table]
    if t.get("local"):
        paths = [local_root() / t["local"].format(day=d) for d in period_days(cfg)]
        paths = [str(p) for p in paths if p.exists()]
        if not paths:
            raise FileNotFoundError(f"no local file for {table}: build it first")
        rows = f"(SELECT * FROM read_parquet({paths}, union_by_name = true))"
        return f"({t['base'].format(src=rows)})" if t.get("base") else rows
    pfs, base = s3_filesystem()
    files = daily_files(pfs, base, t.get("source_table", table))
    picked = [uri(pfs, files[d]) for d in period_days(cfg) if d in files]
    if not picked:
        raise FileNotFoundError(f"no {t.get('source_table', table)} file in the period")
    paths = ", ".join(f"'{p}'" for p in picked)
    schema = set(connection().sql(f"DESCRIBE SELECT * FROM read_parquet(['{picked[0]}'])").df()["column_name"])
    cols = [c for c in dict.fromkeys(_columns_used(cfg, table) + list(columns or [])) if c in schema]
    order = "ORDER BY _version DESC" if "_version" in schema else ""
    keep = ", ".join(f'r."{c}"' for c in dict.fromkeys(cols + (["_version"] if order else [])))
    brands = ", ".join(map(str, cfg["brands"]))
    files_sql = f"read_parquet([{paths}], union_by_name = true)"
    players = (f"(SELECT tenant_id, player_id, any_value(brand_id) AS brand_id "
               f"FROM read_parquet('{ACTIVITY}/*.parquet') WHERE brand_id IN ({brands}) GROUP BY ALL)")
    if "brand_id" in schema and fill_brand:
        keep = keep.replace('r."brand_id"', "COALESCE(r.brand_id, b.brand_id) AS brand_id, r.brand_id IS NULL AS brand_filled")
        frm = f"{files_sql} r LEFT JOIN {players} b USING (tenant_id, player_id)"
        cond = f"COALESCE(r.brand_id, b.brand_id) IN ({brands})"
    elif "brand_id" in schema:
        frm, cond = f"{files_sql} r", f"r.brand_id IN ({brands})"
    else:  # the brand of each player comes from the local activity cache
        frm, cond = f"{files_sql} r JOIN {players} b USING (tenant_id, player_id)", "TRUE"
        keep += ", b.brand_id" + (", FALSE AS brand_filled" if fill_brand else "")
    rows = (f"(SELECT {keep} FROM {frm} WHERE {cond} "
            f"QUALIFY row_number() OVER (PARTITION BY {', '.join('r.' + k for k in t['key'])} {order}) = 1)")
    return f"({t['base'].format(src=rows)})" if t.get("base") else rows


def _values(cfg: dict, name: str, extra: str = "") -> str:
    m = cfg["metrics"][name]
    return (f"(SELECT CAST({m['expr']} AS DOUBLE) AS x{extra} FROM {source(cfg, m['table'])} "
            f"WHERE {m.get('where', 'TRUE')})")


# ---------------------------------------------------------------- aggregates

def profile_sql(values: str, lo: float, hi: float, by: str | None = None) -> str:
    """One profile row, or one per value of `by` (a column of `values`, returned as `segment`)."""
    ln = "CASE WHEN x > 0 THEN ln(x) END"
    group = f"{by} AS segment, " if by else ""
    return f"""
        SELECT {group}count(*) AS rows, count(x) AS n, count_if(x IS NULL) AS nulls, count_if(x = 0) AS zeros,
               count_if(x < 0) AS negatives, count_if(x < {lo}) AS below_range, count_if(x > {hi}) AS above_range,
               min(x) AS min, max(x) AS max, avg(x) AS mean, var_samp(x) AS var,
               skewness(x) AS skew, kurtosis(x) AS kurt,
               avg({ln}) AS mean_ln, stddev_samp({ln}) AS sd_ln, skewness({ln}) AS skew_ln, kurtosis({ln}) AS kurt_ln,
               quantile_cont(x, {QUANTILES}) AS quantiles
        FROM {values}{" GROUP BY ALL ORDER BY ALL" if by else ""}"""


def histogram_sql(values: str, m: dict, by: str | None = None) -> str:
    """Counts per bin. log: bin k covers [10^(k w), 10^((k+1) w)); int: one bin per value, the last
    one is `cap` or more; linear: bin k covers [k w, (k+1) w)."""
    scale = m["scale"]
    if scale == "log":
        b, cond = f"floor(log10(x) / {m['width']})", "x > 0"
    elif scale == "int":
        b, cond = f"least(round(x), {m['cap']})", "x IS NOT NULL"
    else:
        b, cond = f"floor(x / {m['width']})", "x IS NOT NULL"
    group = f"{by} AS segment, " if by else ""
    return (f"SELECT {group}CAST({b} AS BIGINT) AS bin, count(*) AS count FROM {values} WHERE {cond} "
            f"GROUP BY ALL ORDER BY ALL")


def metric_data(cfg: dict, name: str) -> tuple[pd.Series, pd.DataFrame]:
    """Profile (one row) and histogram of a metric, cached."""
    m = cfg["metrics"][name]
    lo, hi = m["range"]

    def profile():
        return connection().sql(profile_sql(_values(cfg, name), lo, hi)).df()

    def histogram():
        return connection().sql(histogram_sql(_values(cfg, name), m)).df()

    key = f"{name}__{spec_hash(cfg, name)}"
    local = is_local(cfg, m["table"])
    prof = cached(cfg, f"{key}__profile", profile, local).iloc[0]
    return prof, cached(cfg, f"{key}__hist", histogram, local)


def segment_histograms(cfg: dict, name: str, segment: str) -> pd.DataFrame:
    """Histogram of a metric split by one segment column, cached."""
    m = cfg["metrics"][name]

    def compute():
        values = _values(cfg, name, extra=f", CAST({segment} AS VARCHAR) AS seg")
        return connection().sql(histogram_sql(values, m, by="seg")).df()

    return cached(cfg, f"{name}__{spec_hash(cfg, name)}__by_{segment}", compute, is_local(cfg, m["table"]))


def footer_nulls(table: str) -> pd.DataFrame:
    """Null share per column and day, from the cached Parquet footers (no S3)."""
    f = pd.read_parquet(FOOTERS / f"{table}.parquet")
    f["null_share"] = f["nulls"] / f["n_rows"].where(f["n_rows"] > 0)
    return f.pivot_table(index="column", columns="day", values="null_share")


def footer_range(table: str, column: str) -> pd.DataFrame:
    """Daily minimum and maximum of a column, from the cached Parquet footers (no S3)."""
    f = pd.read_parquet(FOOTERS / f"{table}.parquet")
    return f.loc[f["column"] == column, ["day", "min_num", "max_num", "n_rows"]].set_index("day").sort_index()


# ---------------------------------------------------------------- per brand

def profile_by_segment(cfg: dict, name: str, segment: str = "brand_id") -> pd.DataFrame:
    """Profile of a metric per value of a segment column (one row each), cached."""
    m = cfg["metrics"][name]
    lo, hi = m["range"]

    def compute():
        values = _values(cfg, name, extra=f", CAST({segment} AS VARCHAR) AS seg")
        return connection().sql(profile_sql(values, lo, hi, by="seg")).df()

    return cached(cfg, f"{name}__{spec_hash(cfg, name)}__profile_by_{segment}", compute, is_local(cfg, m["table"]))


# ---------------------------------------------------------------- impossible and suspicious values

def _rule(m: dict, kind: str) -> str | None:
    return re.sub(r"\bx\b", f"({m['expr']})", m[kind]) if m.get(kind) else None


def flags_by_day(cfg: dict, table: str, start=None, end=None) -> pd.DataFrame:
    """Rows, impossible and suspicious values of each metric of a table with an `impossible` or a
    `suspicious` rule, per day and brand, in one scan. Long format: day, brand_id, brand_filled,
    metric, rows, impossible, suspicious. A row with an empty brand_id takes the brand of its player
    (tables with a `base` keep their brand as it is)."""
    names = [n for n, m in cfg["metrics"].items() if m["table"] == table and (m.get("impossible") or m.get("suspicious"))]
    sub = cfg | {"period": {"start": start or cfg["period"]["start"], "end": end or cfg["period"]["end"]}}
    t = cfg["tables"][table]
    date = t["date"]
    fill = not t.get("base")

    def compute():
        parts = []
        for n in names:
            m = cfg["metrics"][n]
            w = m.get("where", "TRUE")
            parts.append(f"count_if({w}) AS \"rows__{n}\"")
            for kind in ("impossible", "suspicious"):
                rule = _rule(m, kind)
                parts.append(f"count_if(({w}) AND ({rule})) AS \"{kind}__{n}\"" if rule else f"0 AS \"{kind}__{n}\"")
        filled = "brand_filled" if fill else "FALSE AS brand_filled"
        return connection().sql(f"SELECT CAST({date} AS DATE) AS day, brand_id, {filled}, {', '.join(parts)} "
                                f"FROM {source(sub, table, fill_brand=fill)} GROUP BY ALL ORDER BY ALL").df()

    spec = {n: {k: cfg["metrics"][n].get(k) for k in ("expr", "where", "impossible", "suspicious")} for n in names}
    spec |= {"base": t.get("base"), "brands": cfg["brands"]}
    key = hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:8]
    wide = cached(sub, f"flags_by_day__{table}__{key}", compute)
    long = wide.melt(id_vars=["day", "brand_id", "brand_filled"])
    long[["kind", "metric"]] = long["variable"].str.split("__", n=1, expand=True)
    out = long.pivot_table(index=["day", "brand_id", "brand_filled", "metric"], columns="kind", values="value",
                           aggfunc="sum").reset_index()
    out["day"] = pd.to_datetime(out["day"]).dt.date
    return out[["day", "brand_id", "brand_filled", "metric", "rows", "impossible", "suspicious"]]


def flags_summary(flags: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per metric: the two rules, rows, impossible and suspicious counts and shares, the days with
    flagged values and the share in each brand."""
    rows = []
    for name, g in flags.groupby("metric"):
        m = cfg["metrics"][name]
        row = {"metric": name, "impossible if": m.get("impossible", ""), "suspicious if": m.get("suspicious", ""),
               "rows": int(g["rows"].sum())}
        for kind in ("impossible", "suspicious"):
            hit = g[g[kind] > 0]
            row[kind] = int(g[kind].sum())
            row[f"{kind} %"] = 100 * g[kind].sum() / max(g["rows"].sum(), 1)
            row[f"days with {kind}"] = hit["day"].nunique()
        for b, gb in g.groupby("brand_id"):
            row[f"brand {b}: susp. %"] = 100 * gb["suspicious"].sum() / max(gb["rows"].sum(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def flagged_values(cfg: dict, name: str, n: int = 15) -> pd.DataFrame:
    """The most frequent impossible or suspicious values of a metric, with their brand, currency (when
    the metric is split by it) and day range: aggregates, never player rows."""
    m = cfg["metrics"][name]
    rules = [f"({r})" for r in (m.get("impossible"), m.get("suspicious")) if r]
    if not rules:
        return pd.DataFrame()
    t = cfg["tables"][m["table"]]
    cur = ", CAST(currency AS VARCHAR) AS currency" if "currency" in m.get("segments", []) else ", NULL AS currency"
    imp = m.get("impossible") or "FALSE"

    def compute():
        values = _values(cfg, name, extra=f", CAST(brand_id AS VARCHAR) AS brand, CAST({t['date']} AS DATE) AS day{cur}")
        return connection().sql(f"""
            SELECT round(x, 4) AS value, CASE WHEN {imp} THEN 'impossible' ELSE 'suspicious' END AS kind,
                   brand, currency, count(*) AS rows, min(day) AS first_day, max(day) AS last_day
            FROM {values} WHERE {' OR '.join(rules)}
            GROUP BY ALL ORDER BY rows DESC, value DESC LIMIT {int(n)}""").df()

    key = hashlib.sha1(json.dumps([m.get("impossible"), m.get("suspicious"), n], default=str).encode()).hexdigest()[:6]
    return cached(cfg, f"{name}__{spec_hash(cfg, name)}__flagged_values_{key}", compute, is_local(cfg, m["table"]))


