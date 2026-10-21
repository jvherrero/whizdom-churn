"""Signal audit on a small synthetic lake (no S3): each check finds the shape it was built with.

    .venv/bin/python -m pytest signal_audit/tests -q
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit  # noqa: E402
import stats  # noqa: E402

DAYS = [dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(3)]
N = 4000  # players per brand and day


def _write(root: Path, table: str, frame: pd.DataFrame, date_col: str) -> None:
    for d, rows in frame.groupby(date_col):
        folder = root / table / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}"
        folder.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(folder / "part-0.parquet", index=False)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for d in DAYS:
        for brand, mu in ((14, 2.0), (64, 5.0)):  # two brands with very different stakes: two peaks together
            rows.append(pd.DataFrame({
                "tenant_id": "8132d267-operator", "brand_id": brand, "player_id": brand * 100_000 + np.arange(N),
                "snapshot_date": d,
                "turnover_eur": rng.lognormal(3.0, 1.0, N),        # one log-normal for both brands
                "mixed_eur": rng.lognormal(mu, 0.4, N),            # a different median per brand
                "sessions_today": rng.poisson(2.0, N),
                "delay_hours": rng.gumbel(30, 6, N),
                "night_share": rng.uniform(0, 1, N) * np.where(rng.uniform(size=N) < 0.02, 4, 1),
                "_version": 1}))
    frame = pd.concat(rows, ignore_index=True)
    dup = frame.iloc[:50].assign(turnover_eur=1e9, _version=0)  # an older version that must be dropped
    other = frame.assign(brand_id=99, player_id=frame["player_id"] + 10_000_000)  # a brand out of scope
    _write(tmp_path / "lake", "gld_player_test_daily", pd.concat([frame, dup, other]), "snapshot_date")
    # the same rows without brand_id: the brand comes from the activity cache
    _write(tmp_path / "lake", "gld_player_nobrand_daily", pd.concat([frame, other]).drop(columns="brand_id"),
           "snapshot_date")
    # brand_id empty for brand 64: impossible_by_day takes the brand of the player from the activity cache
    _write(tmp_path / "lake", "gld_player_nullbrand_daily",
           pd.concat([frame, other]).assign(brand_id=lambda f: f["brand_id"].where(f["brand_id"] != 64)),
           "snapshot_date")
    activity = tmp_path / "activity"
    activity.mkdir()
    pd.concat([frame, other])[["tenant_id", "brand_id", "player_id"]].drop_duplicates().to_parquet(
        activity / "2026-09-01.parquet", index=False)
    monkeypatch.setattr(audit, "ACTIVITY", activity)

    monkeypatch.setenv("DATALAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.setenv("RUN_S3", "1")
    monkeypatch.setattr(audit, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(audit, "_CON", None)
    base = {"table": "gld_player_test_daily", "segments": ["brand_id"]}
    return {
        "period": {"start": DAYS[0], "end": DAYS[-1]}, "brands": [14, 64],
        "thresholds": {"ks_good": 0.03, "ks_fair": 0.08, "peak_prominence": 0.05, "smooth_bins": 2, "max_segments": 8},
        "tables": {t: {"date": "snapshot_date", "key": ["tenant_id", "player_id", "snapshot_date"]}
                   for t in ("gld_player_test_daily", "gld_player_nobrand_daily", "gld_player_nullbrand_daily")},
        "metrics": {
            "turnover": {**base, "expr": "turnover_eur", "where": "turnover_eur > 0", "expected": "lognormal",
                         "scale": "log", "width": 0.05, "range": [0.01, 1e6]},
            "mixed": {**base, "expr": "mixed_eur", "where": "mixed_eur > 0", "expected": "lognormal",
                      "scale": "log", "width": 0.05, "range": [0.01, 1e6]},
            "sessions": {**base, "expr": "sessions_today", "where": "sessions_today > 0", "expected": "count",
                         "scale": "int", "cap": 20, "range": [1, 50]},
            "delay": {**base, "expr": "delay_hours", "where": "TRUE", "expected": "delay",
                      "scale": "linear", "width": 1, "range": [0, 240]},
            "turnover_nobrand": {**base, "table": "gld_player_nobrand_daily", "expr": "turnover_eur",
                                 "where": "turnover_eur > 0", "expected": "lognormal", "scale": "log", "width": 0.05,
                                 "range": [0.01, 1e6]},
            "night": {**base, "expr": "night_share", "where": "TRUE", "expected": "share",
                      "scale": "linear", "width": 0.02, "range": [0, 1]},
            "night_nullbrand": {**base, "table": "gld_player_nullbrand_daily", "expr": "night_share", "where": "TRUE",
                                "expected": "share", "scale": "linear", "width": 0.02, "range": [0, 1],
                                "impossible": "x > 1"},
        },
    }


def _run(cfg, name):
    prof, hist = audit.metric_data(cfg, name)
    return prof, stats.fit(cfg, name, prof, hist)


def test_older_versions_are_dropped(cfg):
    prof, _ = _run(cfg, "turnover")
    assert prof["n"] == 2 * N * len(DAYS)
    assert prof["max"] < 1e9


def test_a_lognormal_metric_has_one_peak_and_a_close_fit(cfg):
    _, res = _run(cfg, "turnover")
    assert len(res["peaks"]) == 1
    assert res["best"]["family"] == "Log-normal" and res["best"]["verdict"] == "close"


def test_two_mixed_groups_give_two_peaks_and_one_each_by_segment(cfg):
    _, res = _run(cfg, "mixed")
    assert len(res["peaks"]) == 2
    by = stats.segment_peaks(cfg, "mixed", audit.segment_histograms(cfg, "mixed", "brand_id"))
    assert by["peaks"].tolist() == [1, 1]


def test_zero_truncated_poisson_counts_fit_poisson(cfg):
    _, res = _run(cfg, "sessions")
    pois = next(f for f in res["fits"] if f["family"] == "Poisson")
    assert pois["verdict"] == "close"
    assert abs(float(pois["params"].split("=")[1]) - 2.0) < 0.1


def test_a_gumbel_delay_fits_gumbel(cfg):
    _, res = _run(cfg, "delay")
    g = next(f for f in res["fits"] if f["family"] == "Gumbel")
    assert g["verdict"] == "close"


def test_values_outside_the_range_are_counted(cfg):
    prof, res = _run(cfg, "night")
    share = prof["above_range"] / prof["n"]
    assert 0.005 < share < 0.03
    row = stats.summary_row(cfg, "night", prof, res)
    assert row["outside range %"] == pytest.approx(100 * share)


def test_cached_results_are_read_without_s3(cfg, monkeypatch):
    first, _ = audit.metric_data(cfg, "turnover")
    monkeypatch.setenv("RUN_S3", "0")
    monkeypatch.setenv("DATALAKE_ROOT", "/nonexistent")
    again, _ = audit.metric_data(cfg, "turnover")
    assert again["n"] == first["n"]
    with pytest.raises(FileNotFoundError, match="RUN_S3=1"):
        audit.metric_data(cfg, "mixed")


def test_only_the_brands_in_scope_with_or_without_brand_id(cfg):
    with_column, _ = _run(cfg, "turnover")
    from_activity, _ = _run(cfg, "turnover_nobrand")
    assert from_activity["n"] == with_column["n"] == 2 * N * len(DAYS)


def test_integer_metric_on_a_log_scale_has_no_false_peaks_at_small_integers():
    rng = np.random.default_rng(1)
    values = np.maximum(1, np.round(rng.lognormal(1.2, 1.0, 200_000)))
    m = {"scale": "log", "width": 0.05, "range": [1, 1e6]}
    hist = pd.DataFrame({"bin": np.floor(np.log10(values) / 0.05).astype(int)}).value_counts("bin").rename("count").reset_index()
    cfg = {"thresholds": {"peak_prominence": 0.05, "smooth_bins": 2}}
    assert len(stats.count_peaks(stats.complete(hist, m), cfg)) > 1          # raw: spikes at 1, 2, 3...
    assert len(stats.count_peaks(stats.complete(hist, m | {"integer": True}), cfg)) == 1


def test_linear_bins_outside_the_plot_range_go_to_the_edges():
    hist = pd.DataFrame({"bin": [-368_385, -3, 0, 5, 5000], "count": [1, 2, 3, 4, 5]})
    h = stats.complete(hist, {"scale": "linear", "width": 1, "range": [-1e5, 100], "plot_range": [-10, 10]})
    assert len(h) == 21 and h["count"].sum() == 15
    assert h.iloc[0]["count"] == 1 and h.iloc[-1]["count"] == 5


def test_a_changed_definition_is_not_read_from_the_old_cache(cfg, monkeypatch):
    audit.metric_data(cfg, "turnover")
    cfg["metrics"]["turnover"]["range"] = [0.01, 10.0]
    monkeypatch.setenv("RUN_S3", "0")
    with pytest.raises(FileNotFoundError):
        audit.metric_data(cfg, "turnover")


def test_impossible_values_per_day_and_brand_keep_the_rows_without_brand_id(cfg):
    prof, _ = audit.metric_data(cfg, "night")
    by = audit.flags_by_day(cfg, "gld_player_nullbrand_daily")
    assert by["rows"].sum() == 2 * N * len(DAYS)                       # brand 99 out, brand 64 kept
    assert by["impossible"].sum() == prof["above_range"]               # the same rows as the full table
    assert set(by.loc[by["brand_filled"], "brand_id"]) == {64}
    assert sorted(by["day"].unique()) == DAYS
    s = stats.impossible_summary(by).iloc[0]
    assert s["first day"] == DAYS[0] and s["impossible %"] == pytest.approx(100 * prof["above_range"] / prof["n"])


def test_scorecard_per_brand_splits_the_mixed_metric(cfg):
    card = stats.brand_scorecard(cfg, ["mixed", "turnover"]).set_index(["metric", "brand"])
    assert card.loc[("mixed", "14"), "peaks"] == 1 and card.loc[("mixed", "64"), "peaks"] == 1
    assert card.loc[("turnover", "64"), "fit"] == "close"
    assert card.loc[("turnover", "14"), "rows"] == N * len(DAYS)


def test_suspicious_values_per_day_and_brand(cfg):
    cfg["metrics"]["night_nullbrand"]["suspicious"] = "x > 0.9 AND x <= 1"
    flags = audit.flags_by_day(cfg, "gld_player_nullbrand_daily")
    s = audit.flags_summary(flags, cfg).iloc[0]
    assert 0.05 < s["suspicious %"] / 100 < 0.15 and s["days with suspicious"] == len(DAYS)
    top = audit.flagged_values(cfg, "night_nullbrand", n=5)
    assert len(top) == 5 and set(top["kind"]) <= {"impossible", "suspicious"}
