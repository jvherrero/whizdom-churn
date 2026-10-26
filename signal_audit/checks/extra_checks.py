"""Checks of the columns of gld_player_signals_daily that have no formula (06_extra_checks.ipynb): load
times (with the snapshot date), country and currency, the categorical columns, the ids and the tax. The
snapshot date against the transactions is the recency shift of evidence.py (figure recency_shift).

Every function is one SQL aggregate over all the rows of the period (latest _version of each key, the
five brands, brand_id filled from the activity cache when it is empty), cached like the rest of the audit.
Only aggregates leave DuckDB: no player id, no player row.
"""

from __future__ import annotations

import pandas as pd

from signal_audit.data import audit

TABLE = "gld_player_signals_daily"
CATEGORIES = ["country", "currency", "language", "lifecycle_stage", "value_tier", "lifetime_value_tier", "churn_band",
              "risk_band", "prev_lifecycle_stage", "prev_value_tier", "prev_churn_band", "prev_risk_band", "param_version"]
UUID = "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"


def _src(cfg: dict, columns: list[str]) -> str:
    return audit.source(cfg, TABLE, fill_brand=True, columns=columns)


def load_times(cfg: dict) -> pd.DataFrame:
    """Per brand and snapshot day: minutes from _ingested_at to _processed_at, and hours from the end of the
    snapshot day (snapshot_date + 1 day, 00:00 UTC) to each of them, as quantiles."""
    def compute():
        q = "[0.01, 0.5, 0.99]"
        end = "CAST(snapshot_date + INTERVAL 1 DAY AS TIMESTAMPTZ)"
        return audit.connection().sql(f"""
            SELECT brand_id, snapshot_date, count(*) AS rows,
                   quantile_cont((epoch(_processed_at) - epoch(_ingested_at)) / 60, {q}) AS process_minus_ingest_min,
                   count_if(epoch(_processed_at) - epoch(_ingested_at) BETWEEN 0 AND 600) AS within_10_min,
                   count_if(_processed_at < _ingested_at) AS processed_before_ingested,
                   quantile_cont((epoch(_ingested_at) - epoch({end})) / 3600, {q}) AS ingest_after_day_end_h,
                   quantile_cont((epoch(_processed_at) - epoch({end})) / 3600, {q}) AS process_after_day_end_h
            FROM {_src(cfg, ['_ingested_at', '_processed_at'])} GROUP BY ALL ORDER BY ALL""").df()
    return audit.cached(cfg, "extra_load_times", compute)


def categories(cfg: dict) -> pd.DataFrame:
    """Rows per brand and value of each categorical column (NULL kept as its own value)."""
    def compute():
        sel = " UNION ALL ".join(f"SELECT '{c}' AS col, CAST({c} AS VARCHAR) AS value, brand_id, count(*) AS rows "
                                 f"FROM r0 GROUP BY ALL" for c in CATEGORIES)
        return audit.connection().sql(f"WITH r0 AS (SELECT * FROM {_src(cfg, CATEGORIES)}) {sel}").df()
    return audit.cached(cfg, "extra_categories", compute)


def formats(values: pd.DataFrame) -> pd.DataFrame:
    """Format of each categorical column from categories(): distinct values, rows in upper case, in lower
    case and in mixed case, the lengths, and the share of NULL."""
    out = []
    for col, g in values.groupby("col"):
        v, n = g["value"], g["rows"]
        known = v.notna()
        out.append({"column": col, "distinct": v[known].nunique(), "rows": int(n.sum()),
                    "null %": 100 * n[~known].sum() / n.sum(),
                    "upper %": 100 * n[known & (v == v.str.upper())].sum() / n.sum(),
                    "lower %": 100 * n[known & (v == v.str.lower()) & (v != v.str.upper())].sum() / n.sum(),
                    "lengths": ", ".join(map(str, sorted(v[known].str.len().unique())))})
    return pd.DataFrame(out)


def ids(cfg: dict) -> pd.DataFrame:
    """Format of player_id and tenant_id: type, rows, NULL, length of the text, digits only, UUID, case."""
    def compute():
        p, t = "CAST(player_id AS VARCHAR)", "CAST(tenant_id AS VARCHAR)"
        return audit.connection().sql(f"""
            SELECT typeof(any_value(player_id)) AS player_id_type, count(*) AS rows,
                   count_if(player_id IS NULL) AS player_id_null,
                   min(length({p})) AS player_id_min_len, max(length({p})) AS player_id_max_len,
                   count_if(NOT regexp_full_match({p}, '[0-9]+')) AS player_id_not_digits,
                   min(player_id) AS player_id_min, max(player_id) AS player_id_max,
                   typeof(any_value(tenant_id)) AS tenant_id_type, count(DISTINCT tenant_id) AS tenant_ids,
                   count_if(tenant_id IS NULL) AS tenant_id_null,
                   count_if(NOT regexp_full_match({t}, '{UUID}')) AS tenant_id_not_uuid,
                   count_if({t} <> lower({t})) AS tenant_id_not_lower
            FROM {_src(cfg, ['player_id', 'tenant_id'])}""").df()
    return audit.cached(cfg, "extra_ids", compute)


def id_lengths(cfg: dict) -> pd.DataFrame:
    """Rows per brand and length of player_id as text."""
    def compute():
        return audit.connection().sql(f"""
            SELECT brand_id, length(CAST(player_id AS VARCHAR)) AS length, count(*) AS rows
            FROM {_src(cfg, ['player_id'])} GROUP BY ALL ORDER BY ALL""").df()
    return audit.cached(cfg, "extra_id_lengths", compute)


def tax(cfg: dict) -> pd.DataFrame:
    """Per brand and window: tax against GGR and stake. Rows with NULL, zero and non-zero tax, negative tax, tax above GGR (with GGR
    above 0), tax above the stake, and quantiles of tax / GGR and tax / stake."""
    def compute():
        parts = []
        for w in ("today", "l90d", "l180d"):
            t, g, s = f"tax_eur_{w}", f"ggr_eur_{w}", f"wagered_eur_{w}"
            parts.append(f"""
                SELECT brand_id, '{w}' AS window, count(*) AS rows, count_if({t} IS NULL) AS tax_null,
                       count_if({t} = 0) AS tax_zero, count_if({t} <> 0) AS with_tax,
                       count_if({t} < 0) AS negative, count_if({g} > 0 AND {t} > {g}) AS above_ggr,
                       count_if({t} > {s}) AS above_stake,
                       quantile_cont({t} / {g}, [0.01, 0.5, 0.99]) FILTER (WHERE {g} > 0 AND {t} <> 0) AS tax_over_ggr,
                       quantile_cont({t} / {s}, [0.01, 0.5, 0.99]) FILTER (WHERE {s} > 0 AND {t} <> 0) AS tax_over_stake
                FROM r0 GROUP BY ALL""")
        cols = [f"{c}_{w}" for c in ("tax_eur", "ggr_eur", "wagered_eur") for w in ("today", "l90d", "l180d")]
        return audit.connection().sql(f"WITH r0 AS (SELECT * FROM {_src(cfg, cols)}) {' UNION ALL '.join(parts)} "
                                      f"ORDER BY ALL").df()
    return audit.cached(cfg, "extra_tax_v2", compute)   # v2: with NULL and zero counts


SPORTS = (58, 50)   # providers of sports betting in gld_player_gaming_daily (gaming report)


def sessions_by_product(cfg: dict) -> pd.DataFrame:
    """Are the impossible session values of the signals table sports bets? Per brand and group, rows and rows
    with an impossible value:
    - day level, play_secs_today > 24 h, by what the player played that day: sports only, casino only, both;
    - player level (90-day windows), mean session over 1 day and negative gap between sessions, by whether
      the player bet on sports in the period at all.
    Sports = providers 58 and 50 of gld_player_gaming_daily. Aggregates only."""
    def compute():
        from signal_audit.checks import gaming
        g = {**gaming.load_config(), "period": cfg["period"], "brands": cfg["brands"]}
        rows = audit.source(g, "gaming_rows", columns=["provider_code"])
        sports = ", ".join(map(str, SPORTS))
        cols = ["play_secs_today", "tp_avg_session_duration_secs", "tp_avg_session_gap_hours"]
        return audit.connection().sql(f"""
            WITH d AS (SELECT tenant_id, player_id, event_date,
                              bool_or(provider_code IN ({sports})) AS sports, bool_or(provider_code NOT IN ({sports})) AS casino
                       FROM {rows} WHERE bets > 0 GROUP BY ALL),
                 p AS (SELECT tenant_id, player_id, bool_or(sports) AS sports_player FROM d GROUP BY ALL),
                 s AS (SELECT * FROM {_src(cfg, cols)})
            SELECT s.brand_id, 'day' AS level,
                   CASE WHEN d.sports AND d.casino THEN 'both' WHEN d.sports THEN 'sports only' ELSE 'casino only' END AS grp,
                   count(*) AS rows, count_if(s.play_secs_today > 86400) AS impossible, 'play_secs_today > 24 h' AS rule
            FROM s JOIN d ON d.tenant_id = s.tenant_id AND d.player_id = s.player_id AND d.event_date = s.snapshot_date
            WHERE s.play_secs_today > 0 GROUP BY ALL
            UNION ALL
            SELECT s.brand_id, 'player', CASE WHEN p.sports_player THEN 'bets on sports' ELSE 'casino only' END,
                   count(*), count_if(s.tp_avg_session_duration_secs > 86400), 'mean session > 1 day'
            FROM s JOIN p ON p.tenant_id = s.tenant_id AND p.player_id = s.player_id
            WHERE s.tp_avg_session_duration_secs > 0 GROUP BY ALL
            UNION ALL
            SELECT s.brand_id, 'player', CASE WHEN p.sports_player THEN 'bets on sports' ELSE 'casino only' END,
                   count(*), count_if(s.tp_avg_session_gap_hours < 0), 'gap between sessions < 0'
            FROM s JOIN p ON p.tenant_id = s.tenant_id AND p.player_id = s.player_id
            WHERE s.tp_avg_session_gap_hours <> 0 GROUP BY ALL
            ORDER BY ALL""").df()
    return audit.cached(cfg, "extra_sessions_by_product", compute)


def bonus_vs_gaming(cfg: dict) -> pd.DataFrame:
    """Per brand: player-days where bonus_ggr_eur_today of the signals table equals the sum of bonus_ggr_eur of
    gld_player_gaming_daily of that player and day (within 0.01 + 0.1%), and where it is higher or lower.
    Player-days with no row in the gaming table count as 0 there."""
    def compute():
        from signal_audit.checks import gaming
        g = gaming.load_config()
        g = {**g, "period": cfg["period"], "brands": cfg["brands"]}
        rows = audit.source(g, "gaming_rows", columns=["bonus_ggr_eur"])
        e, t = "COALESCE(d.bonus_ggr, 0)", "CAST(s.bonus_ggr_eur_today AS DOUBLE)"
        close = f"abs({t} - {e}) <= 0.01 + 0.001 * abs({e})"
        return audit.connection().sql(f"""
            WITH d AS (SELECT tenant_id, player_id, event_date, sum(CAST(bonus_ggr_eur AS DOUBLE)) AS bonus_ggr
                       FROM {rows} GROUP BY ALL),
                 s AS (SELECT * FROM {_src(cfg, ['bonus_ggr_eur_today'])} WHERE bonus_ggr_eur_today IS NOT NULL)
            SELECT s.brand_id, count(*) AS rows, count_if({close}) AS ok,
                   count_if(NOT {close} AND {t} > {e}) AS higher, count_if(NOT {close} AND {t} < {e}) AS lower,
                   count_if(d.bonus_ggr IS NULL) AS not_in_gaming
            FROM s LEFT JOIN d ON d.tenant_id = s.tenant_id AND d.player_id = s.player_id AND d.event_date = s.snapshot_date
            GROUP BY ALL ORDER BY ALL""").df()
    return audit.cached(cfg, "extra_bonus_vs_gaming", compute)



def check_results(cfg: dict) -> dict:
    """The result of each check of kind format, share and validated in signal_checks.yaml:
    the % of rows that pass it. From the cache. The distributions are checked by eye on their figure."""
    out = {}
    i = ids(cfg).iloc[0]
    out["tenant_id"] = 100 * (1 - max(i["tenant_id_not_uuid"], i["tenant_id_not_lower"]) / i["rows"])
    out["player_id"] = 100 * (1 - i["player_id_not_digits"] / i["rows"]) if 5 <= i["player_id_min_len"] and i["player_id_max_len"] <= 8 else 0.0
    values = categories(cfg)
    for col, case in (("country", str.upper), ("currency", str.upper), ("language", str.lower)):
        g = values[values["col"] == col]
        ok = g["value"].fillna("").map(lambda v: v != "" and v == case(v))
        out[col] = 100 * g.loc[ok, "rows"].sum() / g["rows"].sum()
    tx = tax(cfg)
    for w, g in tx.groupby("window"):
        out[f"tax_eur_{w}"] = 100 * g["with_tax"].sum() / g["rows"].sum()
    b = bonus_vs_gaming(cfg)
    played = b["rows"].sum() - b["not_in_gaming"].sum()          # player-days with a row in the gaming table
    out["bonus_ggr_eur_today"] = 100 * (1 - (b["higher"].sum() + b["lower"].sum()) / played)
    return out
