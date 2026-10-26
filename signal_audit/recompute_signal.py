"""The checks of kind `recomputed` of a catalogue (player_signal_checks.yaml): the rule of the workbook with its
inputs recomputed from the daily caches, player by player, on day D and on D + 1 ... D + 10.

The table keeps the scores but not their inputs (days since the last bet, active days, GGR, deposits), so the
inputs are recomputed from gld_player_gaming_daily (daily cache `activity`) and gld_player_payments_daily
(daily cache `payments`), up to day D + s. A correct column matches at s = 0; a column that matches better at
s > 0 uses data from after its date. Each input is computed twice: with the days of any wallet and with the
official active days (M053). Player rows stay inside this module; it returns the share of players that match.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import numpy as np
import pandas as pd

import audit
import clickhouse_mirror
import daily_cache
import evidence as ev
import formula_check as fc
import settings

KEY = ev.KEY
SHIFTS = range(11)
FUNCTIONS = {"minimum": np.minimum, "maximum": np.maximum, "clip": np.clip, "where": np.where}
VARIANTS = {"any wallet": False, "official": True}
SUMS = ["active_days_7", "active_days_30", "ggr_today", "ggr_7", "ggr_30", "ggr_90", "wagered_30", "deposits_today",
        "deposits_7", "deposits_30", "deposits_90", "active_streak"]   # 0 for a player with nothing in the window


def snapshot(brand: int, day: dt.date) -> pd.DataFrame:
    """Rows of one brand and day of the local copy of the table."""
    return duckdb.sql(f"SELECT * FROM read_parquet('{clickhouse_mirror.path(day)}') WHERE brand_id = {int(brand)}").df().set_index(KEY)


def payments(brand: int) -> pd.DataFrame:
    p = daily_cache.read("payments", brand, columns="tenant_id, player_id, day, deposit_eur")
    p["day"] = pd.to_datetime(p["day"])
    return p


def inputs(act: pd.DataFrame, pay: pd.DataFrame, end: pd.Timestamp, official: bool) -> pd.DataFrame:
    """The inputs of the rules for every player, with data up to `end` (see the catalogue header)."""
    act = act[act["day"] <= end]
    days = act[act["active"]] if official else act      # the days that count as days with a bet
    pay = pay[pay["day"] <= end]

    def window(frame, col, w):
        return frame[frame["day"] > end - pd.Timedelta(w, unit="D")].groupby(KEY)[col].sum()

    one = days.assign(one=1)
    prev = days[days["day"] < end].groupby(KEY)["day"].max()
    ggr_prev = act.set_index(KEY + ["day"])["ggr_eur"].reindex(pd.MultiIndex.from_frame(prev.reset_index()))
    back = days.assign(back=(end - days["day"]).dt.days).sort_values("back")
    streak = back[back["back"] == back.groupby(KEY).cumcount()].groupby(KEY).size()   # 0, 1, 2 ... days back with no hole
    return pd.DataFrame({
        "days_since_bet": ev._recency(days, end), "active_days_7": window(one, "one", 7), "active_days_30": window(one, "one", 30),
        "ggr_today": window(act, "ggr_eur", 1), "ggr_7": window(act, "ggr_eur", 7), "ggr_30": window(act, "ggr_eur", 30),
        "ggr_90": window(act, "ggr_eur", 90), "wagered_30": window(act, "turnover_eur", 30),
        "avg_bet_7": window(act, "turnover_eur", 7) / window(act, "bets", 7),
        "avg_bet_30": window(act, "turnover_eur", 30) / window(act, "bets", 30),
        "deposits_today": window(pay, "deposit_eur", 1), "deposits_7": window(pay, "deposit_eur", 7),
        "deposits_30": window(pay, "deposit_eur", 30), "deposits_90": window(pay, "deposit_eur", 90),
        "ggr_prev": pd.Series(ggr_prev.to_numpy(), index=prev.index),
        "active_streak": streak, "gap_prev": (end - prev).dt.days, "tenure": (end - days.groupby(KEY)["day"].min()).dt.days})


def _close(t: np.ndarray, e: np.ndarray, tol: dict) -> np.ndarray:
    return np.abs(t - e) <= tol["atol"] + tol["rtol"] * np.abs(e)


def match_day(brand: int, day: dt.date, act: pd.DataFrame, pay: pd.DataFrame, cfg: dict) -> list[dict]:
    """For each recomputed check, variant and shift: players compared and players that match."""
    rows = snapshot(brand, day)
    players = rows.index.to_frame(index=False)
    act, pay = act.merge(players, on=KEY), pay.merge(players, on=KEY)   # only the players of the day: much faster
    ks = fc.checks(cfg)
    ks = ks[ks["kind"] == "recomputed"]
    table = {c: rows[c].astype(float).to_numpy() for c in rows.select_dtypes("number").columns}
    out = []
    for variant, official in VARIANTS.items():
        for s in SHIFTS:
            x = inputs(act, pay, pd.Timestamp(day) + pd.Timedelta(s, unit="D"), official).reindex(rows.index)
            x[SUMS] = x[SUMS].fillna(0)
            env = {**FUNCTIONS, **table, **{c: x[c].to_numpy(float) for c in x}}
            for _, k in ks.iterrows():
                if k["column"] not in table:   # a column the table does not have
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    e = np.asarray(eval(k["expr"], {"__builtins__": {}}, env), dtype=float) * np.ones(len(rows))
                t = table[k["column"]]
                both = np.isfinite(t) & np.isfinite(e)
                ratio = both & (e != 0)
                out.append({"brand": brand, "day": day, "column": k["column"], "n_check": k["n_check"], "variant": variant,
                            "shift": s, "n": int(both.sum()), "ok": int(_close(t[both], e[both], cfg["tolerance"]).sum()),
                            "ratio": np.median(t[ratio] / e[ratio]) if ratio.any() else np.nan,
                            "diff": np.median(t[both] - e[both]) if both.any() else np.nan})
    return out


def match(refresh: bool = False) -> pd.DataFrame:
    """Cached in cache/local (player rows of local files); `refresh` computes it again."""
    return audit.cached_local("recomputed_match", _match, refresh)


def _match() -> pd.DataFrame:
    """Every brand and evidence day (or snapshot) that is copied and has 10 days of activity after it."""
    cfg = fc.load()
    days = sorted(set(settings.EVIDENCE_DAYS) | set(settings.SNAPSHOTS))
    rows = []
    for b in cfg["brands"]:
        act, pay = ev.activity(b), payments(b)
        last = act["day"].max().date()
        for day in days:
            if clickhouse_mirror.path(day).exists() and day + dt.timedelta(days=max(SHIFTS)) <= last:
                rows += match_day(b, day, act, pay, cfg)
    return pd.DataFrame(rows, columns=["brand", "day", "column", "n_check", "variant", "shift", "n", "ok", "ratio", "diff"])


def summary(m: pd.DataFrame) -> pd.DataFrame:
    """Per recomputed check, all brands and days together: % that matches at s = 0 (with the variant that matches
    best there), the median of table / recomputed and of table - recomputed there (medians of the brands and days),
    and the % at the best shift."""
    g = m.groupby(["column", "n_check", "variant", "shift"])[["n", "ok"]].sum()
    g[["ratio", "diff"]] = m.groupby(["column", "n_check", "variant", "shift"])[["ratio", "diff"]].median()
    g["match %"] = 100 * g["ok"] / g["n"].where(g["n"] > 0)
    out = []
    for (col, n), h in g.groupby(level=["column", "n_check"]):
        h = h.reset_index().sort_values("match %", ascending=False)
        at0, best = h[h["shift"] == 0].iloc[0], h.iloc[0]
        out.append([col, n, at0["variant"], int(at0["n"]), at0["match %"], at0["ratio"], at0["diff"], int(best["shift"]), best["match %"]])
    return pd.DataFrame(out, columns=["column", "n_check", "variant", "players", "match at D %", "ratio", "diff", "best shift",
                                      "match at best %"])
