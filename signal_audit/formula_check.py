"""Does gld_player_signals_daily follow its own definitions? Used by 02_validation.ipynb (part A).

Each column of the table has its checks in signal_checks.yaml (see the header of that file for the
kinds). For one snapshot day, every check becomes a few aggregates of one SQL pass:
    formula, identity   share of rows where the stored value equals the expression (within tolerance),
                        and where it is higher or lower
    bound               share of rows where the condition is false
    window              the 7- or 30-day column against the sum of its daily column over those days
    lag                 a prior_* column against its source column lag_days before
    band                which band each score range falls in

Rows: the five brands, through the local activity cache (brand_id of the table is null in 31% of
its rows), latest _version of each key. Only aggregates leave DuckDB.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

import audit

HERE = Path(__file__).resolve().parent
CHECKS = HERE / "signal_checks.yaml"
KEY = ["tenant_id", "player_id", "snapshot_date"]


def load(path: Path = CHECKS) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def checks(cfg: dict) -> pd.DataFrame:
    """One row per check: column, kind and its fields."""
    return pd.DataFrame([{"column": c, "n_check": i, **k} for c, ks in cfg["columns"].items() for i, k in enumerate(ks)])


GROUPS = {"rows": ("formula", "identity", "bound"), "history": ("window", "lag"), "bands": ("band",)}


def _hash(cfg: dict, group: str) -> str:
    """Fingerprint of the checks of one group: editing a bound recomputes the row checks only."""
    kinds = GROUPS[group]
    cols = {c: [k for k in ks if k["kind"] in kinds] for c, ks in cfg["columns"].items()}
    cols = {c: ks for c, ks in cols.items() if ks}
    return hashlib.sha1(json.dumps({"columns": cols, **{k: cfg[k] for k in ("brands", "tolerance", "table")}},
                                   sort_keys=True, default=str).encode()).hexdigest()[:8]


def rows_sql(cfg: dict, days: list[dt.date], columns: list[str]) -> str:
    """Rows of the table on `days`, five brands (from the activity cache), latest version of each key."""
    pfs, base = audit.s3_filesystem()
    files = audit.daily_files(pfs, base, cfg["table"])
    paths = [audit.uri(pfs, files[d]) for d in days if d in files]
    if not paths:
        raise FileNotFoundError(f"no {cfg['table']} file for {days}")
    listing = ", ".join(f"'{p}'" for p in paths)
    brands = ", ".join(map(str, cfg["brands"]))
    schema = set(audit.connection().sql(f"DESCRIBE SELECT * FROM read_parquet(['{paths[0]}'])").df()["column_name"])
    keep = ", ".join(f'r."{c}"' for c in dict.fromkeys(KEY + ["_version"] + columns) if c in schema and c != "brand_id")
    con = audit.connection()
    con.sql("SET preserve_insertion_order = false")   # less memory; the order of the rows does not matter here
    con.sql(f"CREATE TEMP TABLE IF NOT EXISTS players AS SELECT tenant_id, player_id, any_value(brand_id) AS brand "
            f"FROM read_parquet('{audit.ACTIVITY}/*.parquet') WHERE brand_id IN ({brands}) GROUP BY ALL")
    return (f"SELECT {keep}, b.brand FROM read_parquet([{listing}], union_by_name = true) r "
            f"JOIN players b USING (tenant_id, player_id) "
            f"QUALIFY row_number() OVER (PARTITION BY r.tenant_id, r.player_id, r.snapshot_date ORDER BY r._version DESC) = 1")


def _num(col: str) -> str:
    return f'CAST("{col}" AS DOUBLE)'


def _close(t: str, e: str, tol: dict) -> str:
    return f"abs(({t}) - ({e})) <= {tol['atol']} + {tol['rtol']} * abs({e})"


def _compare_parts(i: int, t: str, e: str, tol: dict) -> list[str]:
    both = f"({t}) IS NOT NULL AND ({e}) IS NOT NULL"
    return [f"count(*) FILTER (WHERE {both}) AS n_{i}",
            f"count(*) FILTER (WHERE {both} AND {_close(t, e, tol)}) AS ok_{i}",
            f"count(*) FILTER (WHERE {both} AND NOT {_close(t, e, tol)} AND ({t}) > ({e})) AS hi_{i}",
            f"count(*) FILTER (WHERE {both} AND NOT {_close(t, e, tol)} AND ({t}) < ({e})) AS lo_{i}"]


def _expr_columns(cfg: dict) -> list[str]:
    """Every name in a check (SQL words included; rows_sql keeps only the columns of the table)."""
    import re
    used = set()
    for col, ks in cfg["columns"].items():
        for k in ks:
            if k["kind"] in ("formula", "identity", "bound", "band"):
                used.add(col)
                text = " ".join(str(k.get(f, "")) for f in ("expr", "score", "tenant_z"))
                used |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))   # filtered against the schema later
    return sorted(used)


def day_checks(cfg: dict, day: dt.date) -> pd.DataFrame:
    """Every row check (formula, identity, bound) of one snapshot day, as one aggregate row per check."""
    def compute():
        con = audit.connection()
        tol = cfg["tolerance"]
        cols = _expr_columns(cfg)
        con.sql(f"CREATE OR REPLACE TEMP TABLE sig AS {rows_sql(cfg, [day], cols)}")
        # tenant z-scores: mean and deviation over the brand's players with a value above 0 that day
        z = [(c, k["tenant_z"]) for c, ks in cfg["columns"].items() for k in ks if k.get("tenant_z")]
        zcols = ", ".join(
            f"CASE WHEN {_num(src)} > 0 AND stddev_pop({_num(src)}) FILTER (WHERE {_num(src)} > 0) OVER w > 0 "
            f"THEN ({_num(src)} - avg({_num(src)}) FILTER (WHERE {_num(src)} > 0) OVER w) "
            f"/ stddev_pop({_num(src)}) FILTER (WHERE {_num(src)} > 0) OVER w ELSE 0 END AS \"_z_{col}\"" for col, src in z)
        if zcols:
            con.sql(f"CREATE OR REPLACE TEMP TABLE sig AS SELECT *, {zcols} FROM sig WINDOW w AS (PARTITION BY brand)")
        parts, meta = ["count(*) AS rows"], []
        for _, k in checks(cfg).iterrows():
            i = len(meta)
            if k["kind"] in ("formula", "identity"):
                e = f'"_z_{k["column"]}"' if isinstance(k.get("tenant_z"), str) else f"CAST(({k['expr']}) AS DOUBLE)"
                parts += _compare_parts(i, _num(k["column"]), e, tol)
            elif k["kind"] == "bound":
                parts += [f"count(*) FILTER (WHERE ({k['expr']}) IS NOT NULL) AS n_{i}",
                          f"count(*) FILTER (WHERE ({k['expr']})) AS ok_{i}",
                          f"0 AS hi_{i}", f"0 AS lo_{i}"]
            else:
                continue
            meta.append(k)
        r = con.sql(f"SELECT {', '.join(parts)} FROM sig").df().iloc[0]
        con.sql("DROP TABLE sig")
        out = pd.DataFrame([{"column": k["column"], "n_check": k["n_check"], "kind": k["kind"],
                             "rows": int(r["rows"]), "n": int(r[f"n_{i}"]), "ok": int(r[f"ok_{i}"]),
                             "higher": int(r[f"hi_{i}"]), "lower": int(r[f"lo_{i}"])} for i, k in enumerate(meta)])
        return out.assign(day=str(day))
    return audit.cached(_cache_cfg(cfg), f"formula_check_rows_{day}_{_hash(cfg, 'rows')}", compute)


def history_checks(cfg: dict, day: dt.date) -> pd.DataFrame:
    """Window and lag checks of one snapshot day: they need the days before it. The days are read one
    at a time and added to one row per player, so memory holds one day of the table, not 31."""
    def compute():
        con = audit.connection()
        tol = cfg["tolerance"]
        ks = checks(cfg)
        win, lag = ks[ks["kind"] == "window"], ks[ks["kind"] == "lag"]
        meta = [k for _, k in win.iterrows()] + [k for _, k in lag.iterrows()]
        depth = int(max(win["days"].max() if len(win) else 0, lag["lag_days"].max() if len(lag) else 0))
        # the stored values on the snapshot day
        now_cols = sorted({k["column"] for k in meta})
        con.sql(f"CREATE OR REPLACE TEMP TABLE day_rows AS {rows_sql(cfg, [day], now_cols)}")
        t = ", ".join(f"{_num(k['column'])} AS t_{i}" for i, k in enumerate(meta))
        w = ", ".join(f"CAST(NULL AS DOUBLE) AS w_{i}" for i in range(len(meta)))
        con.sql(f"CREATE OR REPLACE TEMP TABLE acc AS SELECT tenant_id, player_id, {t}, {w} FROM day_rows")
        for back in range(depth + 1):
            d = day - dt.timedelta(days=back)
            inputs = {}
            for i, k in enumerate(meta):
                if k["kind"] == "window" and back < int(k["days"]):          # the window is (day - days, day]
                    inputs[i] = k["base"]
                elif k["kind"] == "lag" and back == int(k["lag_days"]):
                    inputs[i] = k["source"]
            if not inputs:
                continue
            try:
                con.sql(f"CREATE OR REPLACE TEMP TABLE day_rows AS {rows_sql(cfg, [d], sorted(set(inputs.values())))}")
            except FileNotFoundError:
                continue
            sets = []
            for i in range(len(meta)):
                if i not in inputs:
                    sets.append(f"a.w_{i}")
                elif meta[i]["kind"] == "window":   # a sum: stays NULL until a day has a value
                    v = _num(inputs[i]).replace('"', 'r."', 1)
                    sets.append(f"CASE WHEN {v} IS NULL THEN a.w_{i} ELSE COALESCE(a.w_{i}, 0) + {v} END AS w_{i}")
                else:
                    sets.append(f"{_num(inputs[i]).replace(chr(34), 'r.' + chr(34), 1)} AS w_{i}")
            tcols = ", ".join(f"a.t_{i}" for i in range(len(meta)))
            con.sql(f"""CREATE OR REPLACE TEMP TABLE acc AS
                        SELECT a.tenant_id, a.player_id, {tcols}, {', '.join(sets)}
                        FROM acc a LEFT JOIN day_rows r USING (tenant_id, player_id)""")
        agg = ["count(*) AS rows"] + [p for i in range(len(meta)) for p in _compare_parts(i, f"t_{i}", f"w_{i}", tol)]
        r = con.sql(f"SELECT {', '.join(agg)} FROM acc").df().iloc[0]
        con.sql("DROP TABLE day_rows; DROP TABLE acc")
        out = pd.DataFrame([{"column": k["column"], "n_check": k["n_check"], "kind": k["kind"], "rows": int(r["rows"]),
                             "n": int(r[f"n_{i}"]), "ok": int(r[f"ok_{i}"]), "higher": int(r[f"hi_{i}"]),
                             "lower": int(r[f"lo_{i}"])} for i, k in enumerate(meta)])
        return out.assign(day=str(day))
    return audit.cached(_cache_cfg(cfg), f"formula_check_history_{day}_{_hash(cfg, 'history')}", compute)


def band_checks(cfg: dict, day: dt.date) -> pd.DataFrame:
    """Score range of each band, and the share of each band inside one cutoff bucket."""
    def compute():
        con = audit.connection()
        ks = checks(cfg)
        bands = ks[ks["kind"] == "band"]
        cols = sorted(set(bands["column"]) | set(bands["score"]))
        con.sql(f"CREATE OR REPLACE TEMP TABLE sig AS {rows_sql(cfg, [day], cols)}")
        out = []
        for _, k in bands.iterrows():
            cut = k.get("cutoffs") if isinstance(k.get("cutoffs"), list) else []
            bucket = " + ".join(f"CASE WHEN {_num(k['score'])} >= {c} THEN 1 ELSE 0 END" for c in cut) or "0"
            df = con.sql(f"""SELECT CAST("{k['column']}" AS VARCHAR) AS band, {bucket} AS bucket, count(*) AS rows,
                                    min({_num(k['score'])}) AS score_min, max({_num(k['score'])}) AS score_max
                             FROM sig GROUP BY ALL ORDER BY ALL""").df()
            out.append(df.assign(column=k["column"], score=k["score"]))
        con.sql("DROP TABLE sig")
        return pd.concat(out, ignore_index=True).assign(day=str(day)) if out else pd.DataFrame()
    return audit.cached(_cache_cfg(cfg), f"formula_check_bands_{day}_{_hash(cfg, 'bands')}", compute)


def _cache_cfg(cfg: dict) -> dict:
    """The cache folder is named after the snapshot days."""
    return {"period": {"start": cfg["snapshots"][0], "end": cfg["snapshots"][-1]}}


def run(cfg: dict) -> pd.DataFrame:
    """Every row, window and lag check of every snapshot day, with a verdict."""
    parts = []
    for d in cfg["snapshots"]:
        day = pd.Timestamp(d).date()
        parts += [day_checks(cfg, day), history_checks(cfg, day)]
    res = pd.concat(parts, ignore_index=True)
    return verdicts(cfg, res)


def verdicts(cfg: dict, res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby(["column", "n_check", "kind"], as_index=False)[["n", "ok", "higher", "lower"]].sum()
    g["pass %"] = 100 * g["ok"] / g["n"].where(g["n"] > 0)
    g["higher %"] = 100 * g["higher"] / g["n"].where(g["n"] > 0)
    g["lower %"] = 100 * g["lower"] / g["n"].where(g["n"] > 0)
    good, fair = 100 * cfg["verdict"]["good"], 100 * cfg["verdict"]["fair"]
    g["verdict"] = pd.cut(g["pass %"], [-1, fair, good, 101], labels=["fails", "almost", "passes"], right=False).astype(str)
    g.loc[g["n"] == 0, "verdict"] = "no data"
    ks = checks(cfg).set_index(["column", "n_check"])
    info = ks.reindex(pd.MultiIndex.from_frame(g[["column", "n_check"]]))
    for f in ("expr", "note", "measure"):
        g[f] = info[f].to_numpy() if f in info else ""
    return g
