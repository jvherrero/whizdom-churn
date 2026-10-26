"""Active day and data from the future: the checks of kind `recomputed` of a catalogue (player_signal_checks.yaml).

The columns that count days with a bet (active_streak_days, gap_days, frequency_score, recency_score) are
computed again from the daily player table of the same database (the copy of `activity_table`, see
clickhouse_mirror.copy_activity), player by player, with data up to day D + s, s = 0 ... 10:
- with the official active days (M053: real money or released bonus) and with the days of any wallet: the one
  that matches says which definition the table uses;
- at each shift s: a correct column matches at s = 0; one that matches better at s > 0 uses later data.
Player rows stay inside this module; it returns how many players match.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import numpy as np
import pandas as pd

from signal_audit import settings
from signal_audit.checks import evidence as ev
from signal_audit.checks import formula_check as fc
from signal_audit.data import audit, clickhouse_mirror

KEY = ev.KEY
SHIFTS = range(11)
FUNCTIONS = {"minimum": np.minimum, "maximum": np.maximum, "clip": np.clip, "where": np.where}
VARIANTS = {"official": True, "any wallet": False}


def snapshot(day: dt.date) -> pd.DataFrame:
    """Rows of one day of the local copy of the table."""
    return duckdb.sql(f"SELECT * FROM read_parquet('{clickhouse_mirror.path(day)}')").df().set_index(KEY)


def activity() -> pd.DataFrame:
    """Every copied day of the daily player table: one row per player and day with a bet, and `active`."""
    files = sorted(str(p) for p in (settings.ROOT / "activity").glob("*.parquet"))
    a = duckdb.sql(f"SELECT tenant_id, player_id, day, CAST(active AS BOOLEAN) AS active FROM read_parquet({files})").df()
    a["day"] = pd.to_datetime(a["day"])
    return a


def inputs(days: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """Per player, with the days with a bet up to `end`: the inputs of the rules (see the catalogue header)."""
    days = days[days["day"] <= end]
    one = days.assign(one=1)
    window = lambda w: one[one["day"] > end - pd.Timedelta(w, unit="D")].groupby(KEY)["one"].sum()   # noqa: E731
    prev = days[days["day"] < end].groupby(KEY)["day"].max()
    back = days.assign(back=(end - days["day"]).dt.days).sort_values("back")
    streak = back[back["back"] == back.groupby(KEY).cumcount()].groupby(KEY).size()   # 0, 1, 2 ... days back with no hole
    return pd.DataFrame({"days_since_bet": ev._recency(days, end), "active_days_7": window(7), "active_days_30": window(30),
                         "active_streak": streak, "gap_prev": (end - prev).dt.days})


def match_day(day: dt.date, act: pd.DataFrame, cfg: dict) -> list[dict]:
    """For each recomputed check, definition of the active day and shift: players compared and players that match,
    and the medians of table / recomputed and table - recomputed."""
    rows = snapshot(day)
    act = act.merge(rows.index.to_frame(index=False), on=KEY)   # only the players of the day
    ks = fc.checks(cfg)
    ks = ks[(ks["kind"] == "recomputed") & ks["column"].isin(rows.columns)]
    table = {c: rows[c].astype(float).to_numpy() for c in rows.select_dtypes("number").columns}
    out = []
    for variant, official in VARIANTS.items():
        days = act[act["active"]] if official else act
        for s in SHIFTS:
            x = inputs(days, pd.Timestamp(day) + pd.Timedelta(s, unit="D")).reindex(rows.index)
            x[["active_days_7", "active_days_30", "active_streak"]] = x[["active_days_7", "active_days_30", "active_streak"]].fillna(0)
            env = {**FUNCTIONS, **table, **{c: x[c].to_numpy(float) for c in x}}
            for _, k in ks.iterrows():
                with np.errstate(divide="ignore", invalid="ignore"):
                    e = np.asarray(eval(k["expr"], {"__builtins__": {}}, env), dtype=float) * np.ones(len(rows))
                t = table[k["column"]]
                both = np.isfinite(t) & np.isfinite(e)
                ratio = both & (e != 0)
                ok = np.abs(t[both] - e[both]) <= cfg["tolerance"]["atol"] + cfg["tolerance"]["rtol"] * np.abs(e[both])
                out.append({"day": day, "column": k["column"], "n_check": k["n_check"], "variant": variant,
                            "shift": s, "n": int(both.sum()), "ok": int(ok.sum()),
                            "ratio": np.median(t[ratio] / e[ratio]) if ratio.any() else np.nan,
                            "diff": np.median(t[both] - e[both]) if both.any() else np.nan})
    return out


def match(refresh: bool = False) -> pd.DataFrame:
    """Cached in the run's cache/local (player rows of local files); `refresh` computes it again."""
    return audit.cached_local("recomputed_match", _match, refresh)


def _match() -> pd.DataFrame:
    """Every evidence day (or snapshot) that is copied and has 10 days of activity after it, all brands."""
    cfg = fc.load()
    act = activity()
    last = act["day"].max().date()
    days = sorted(set(settings.EVIDENCE_DAYS) | set(settings.SNAPSHOTS))
    rows = [r for d in days if clickhouse_mirror.path(d).exists() and d + dt.timedelta(days=max(SHIFTS)) <= last
            for r in match_day(d, act, cfg)]
    return pd.DataFrame(rows, columns=["day", "column", "n_check", "variant", "shift", "n", "ok", "ratio", "diff"])


def summary(m: pd.DataFrame) -> pd.DataFrame:
    """Per recomputed check, all brands and days together: % that matches at s = 0 with each definition of the
    active day, the definition that matches best, the medians of table / recomputed and table - recomputed with
    it, and the best shift with that definition."""
    g = m.groupby(["column", "n_check", "variant", "shift"])[["n", "ok"]].sum()
    g["match %"] = 100 * g["ok"] / g["n"].where(g["n"] > 0)
    g[["ratio", "diff"]] = m.groupby(["column", "n_check", "variant", "shift"])[["ratio", "diff"]].median()
    out = []
    for (col, n), h in g.groupby(level=["column", "n_check"]):
        h = h.reset_index()
        at0 = h[h["shift"] == 0].set_index("variant")
        variant = at0["match %"].idxmax()
        best = h[h["variant"] == variant].sort_values("match %", ascending=False).iloc[0]
        out.append([col, n, at0.loc["official", "match %"], at0.loc["any wallet", "match %"], variant, int(at0.loc[variant, "n"]),
                    at0.loc[variant, "match %"], at0.loc[variant, "ratio"], at0.loc[variant, "diff"], int(best["shift"]), best["match %"]])
    return pd.DataFrame(out, columns=["column", "n_check", "official %", "any wallet %", "variant", "players", "match at D %",
                                      "ratio", "diff", "best shift", "match at best %"])
