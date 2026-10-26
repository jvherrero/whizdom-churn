"""The evidence behind a failing row check of a catalogue, as aggregates (never player rows):
- formula_pairs: for a formula, how many rows have each pair (value of the formula with the other columns of the
  row, value in the table), in a grid of bins of 0.02 on the scale 0 to 1; a correct column lies on the diagonal;
- value_hist: for a limit, the histogram of the column and the rows beyond the limits.
Cached in the cache of the run, like the checks themselves."""

from __future__ import annotations

import math

import pandas as pd

from signal_audit.checks import formula_check as fc
from signal_audit.data import audit

PAIR_BINS = 50       # bins of 0.02 on 0 to 1
HIST_BINS = 50       # bars of a histogram; with two limits, this many between them


def _sig(cfg: dict, columns: list[str]) -> None:
    """The rows of the snapshot days of the catalogue in the temporary table `sig`."""
    days = [pd.Timestamp(d).date() for d in cfg["snapshots"]]
    audit.connection().sql(f"CREATE OR REPLACE TEMP TABLE sig AS {fc.rows_sql(cfg, days, columns)}")


def formula_pairs(cfg: dict, column: str, expr: str) -> pd.DataFrame:
    """Rows per cell (x: the formula, y: the table), both clipped to 0 to 1; columns x, y (bin numbers) and n."""
    def compute():
        _sig(cfg, fc._expr_columns(cfg))
        cell = lambda v: f"CAST(least(floor(greatest({v}, 0) * {PAIR_BINS}), {PAIR_BINS - 1}) AS INTEGER)"   # noqa: E731
        return audit.connection().sql(f"""
            SELECT {cell(f'CAST(({expr}) AS DOUBLE)')} AS x, {cell(fc._num(column))} AS y, count(*) AS n
            FROM sig WHERE ({expr}) IS NOT NULL AND {fc._num(column)} IS NOT NULL GROUP BY ALL""").df()
    return audit.cached(fc._cache_cfg(cfg), f"row_pairs_{column}_{fc._hash(cfg, 'rows')}", compute)


def value_hist(cfg: dict, column: str, lo: float | None, hi: float | None) -> pd.DataFrame:
    """Rows per bar of the column between the 1st and 99th percentile, widened to hold the limits, with the bars
    aligned to the limits (a limit is the edge of a bar); the edge bars hold every value beyond them. Columns bin, n and,
    repeated in each row, the left edge a, the right edge b, the number of bars, the minimum, the maximum and the
    share of the rows exactly at each limit."""
    def compute():
        _sig(cfg, fc._expr_columns(cfg))
        con, x = audit.connection(), fc._num(column)
        q1, q99, mn, mx = con.sql(f"SELECT quantile_cont({x}, 0.01), quantile_cont({x}, 0.99), min({x}), max({x}) FROM sig").fetchone()
        a, b = min(q1, lo if lo is not None else q1), max(q99, hi if hi is not None else q99)
        pad = (b - a) * 0.05 or 1.0
        a, b = a - pad, b + pad
        width = (hi - lo) / HIST_BINS if lo is not None and hi is not None else (b - a) / HIST_BINS
        edge = lo if lo is not None else hi
        if edge is not None:
            a = edge - math.ceil((edge - a) / width) * width
        bars = math.ceil((b - a) / width)
        index = f"floor(({x} - {a!r}) / {width!r} + 1e-9)"
        if hi is not None:   # a value exactly at the upper limit is valid: it belongs to the bar below the limit
            index = f"CASE WHEN abs({x} - {hi!r}) < {width * 1e-6!r} THEN {round((hi - a) / width) - 1} ELSE {index} END"
        h = con.sql(f"""SELECT CAST(least(greatest({index}, 0), {bars - 1}) AS INTEGER) AS bin,
                               count(*) AS n FROM sig WHERE {x} IS NOT NULL GROUP BY ALL ORDER BY ALL""").df()
        at = lambda edge: 0.0 if edge is None else con.sql(  # noqa: E731
            f"SELECT count_if(abs({x} - {edge!r}) < {width * 1e-6!r}) / count({x}) FROM sig").fetchone()[0]
        return h.assign(a=a, b=a + bars * width, bars=bars, min=mn, max=mx, at_lo=at(lo), at_hi=at(hi))
    return audit.cached(fc._cache_cfg(cfg), f"row_hist_{column}_{fc._hash(cfg, 'rows')}_v4", compute)
