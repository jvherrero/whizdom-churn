"""Leakage test (T6): features at a cutoff must not depend on data after the cutoff.

    .venv/bin/python -m pytest src/features/tests -q

Two small synthetic copies of the source tables are built. The second one changes every day after
the cutoff. The features must be the same in both, and the labels must change.
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CUTOFF = dt.date(2025, 7, 15)
START, END = dt.date(2025, 4, 1), dt.date(2025, 10, 31)
N_PLAYERS = 120


def _write(root: Path, table: str, day: dt.date, frame: pd.DataFrame) -> None:
    folder = root / table / f"{day:%Y/%m/%d}"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / "data.parquet", index=False)


def make_source(root: Path, future_changed: bool) -> None:
    """Synthetic source tables. With future_changed, every day after CUTOFF is different."""
    import signal_snapshots

    days = [START + dt.timedelta(days=i) for i in range((END - START).days + 1)]
    for day in days:
        late = day > CUTOFF
        alt = np.random.default_rng(day.toordinal() + (1000 if (late and future_changed) else 0))
        n = N_PLAYERS + (40 if (late and future_changed) else 0)  # new players appear after the cutoff
        ids = np.arange(n, dtype="int32")
        ids = ids[alt.random(n) < 0.3]
        wager = alt.uniform(1, 200, len(ids))
        playable_only = alt.random(len(ids)) < 0.1      # stakes only from playable bonus: not an active day
        _write(root, "gld_player_gaming_daily", day, pd.DataFrame({
            "tenant_id": "t1", "brand_id": np.int32(64), "player_id": ids, "bets": alt.integers(1, 50, len(ids)).astype("uint32"),
            "rounds": np.uint32(10), "turnover_eur": wager, "ggr_eur": alt.normal(2, 20, len(ids)),
            "wagered_real": np.where(playable_only, 0.0, wager), "wagered_released_bonus": 0.0,
            "wagered_playable_bonus": np.where(playable_only, wager, 0.0)}))
        _write(root, "gld_player_financial_daily", day, pd.DataFrame({
            "tenant_id": "t1", "player_id": ids, "snapshot_date": day, "wagered_eur_today": wager,
            "won_eur_today": wager * 0.9, "ggr_eur_today": alt.normal(2, 20, len(ids)), "deposits_eur_today": 0.0,
            "withdrawals_eur_today": 0.0, "bonus_ggr_eur_today": 0.0, "wagered_bonus_eur_today": 0.0}))
        _write(root, "gld_player_payments_daily", day, pd.DataFrame({
            "tenant_id": "t1", "brand_id": np.int32(64), "player_id": ids,
            "deposit_count": (alt.random(len(ids)) < 0.3).astype("uint32"), "deposit_attempts": np.uint32(1),
            "failed_deposit_count": alt.integers(0, 3, len(ids)).astype("uint32"),
            "withdraw_count": (alt.random(len(ids)) < 0.1).astype("uint32"), "failed_withdraw_count": np.uint32(0),
            "deposit_amount_eur": alt.uniform(10, 300, len(ids)), "withdraw_amount_eur": alt.uniform(10, 300, len(ids))}))
        n_all = N_PLAYERS + 40
        sig = pd.DataFrame({"tenant_id": "t1", "player_id": np.arange(n_all, dtype="int32"), "snapshot_date": day})
        for col in signal_snapshots.MONEY:
            sig[col] = alt.uniform(0, 500, n_all)
        for col in signal_snapshots.OTHER:
            sig[col] = alt.integers(0, 30, n_all)
        sig["churn_band"], sig["lifecycle_stage"] = "low", "active"
        _write(root, "gld_player_signals_daily", day, sig)


def features_and_labels(root: Path, cache: Path, monkeypatch) -> tuple[pd.DataFrame, pd.DataFrame]:
    monkeypatch.setenv("DATALAKE_ROOT", str(root))
    monkeypatch.setenv("DATA_AVAILABLE_THROUGH", str(END))
    import build_features
    import daily_cache
    import churn_labels

    monkeypatch.setattr(daily_cache, "CACHE_ROOT", cache)
    # The synthetic deposits are valid from the start, so the deposit features are tested too.
    monkeypatch.setattr(build_features, "DEPOSITS_VALID_FROM", START)
    daily_cache.build("activity", end=END)
    daily_cache.build("financial", 64, end=END)
    daily_cache.build("payments", 64, end=END)
    raw = build_features.raw_features(CUTOFF, 64)
    labels = churn_labels.churn_targets(raw[["tenant_id", "player_id"]], CUTOFF, 64, END)
    key = ["tenant_id", "player_id"]
    return raw.sort_values(key).reset_index(drop=True), labels.sort_values(key).reset_index(drop=True)


@pytest.fixture(scope="module")
def two_worlds(tmp_path_factory):
    base = tmp_path_factory.mktemp("source")
    make_source(base / "original", future_changed=False)
    make_source(base / "future_changed", future_changed=True)
    return base


def test_features_ignore_everything_after_the_cutoff(two_worlds, tmp_path, monkeypatch):
    raw_a, _ = features_and_labels(two_worlds / "original", tmp_path / "cache_a", monkeypatch)
    raw_b, _ = features_and_labels(two_worlds / "future_changed", tmp_path / "cache_b", monkeypatch)
    import build_features

    assert len(raw_a) > 0
    assert raw_a[["tenant_id", "player_id"]].equals(raw_b[["tenant_id", "player_id"]]), "the population changed"
    for col in build_features.FEATURES:
        a, b = raw_a[col].astype(float), raw_b[col].astype(float)
        same = np.isclose(a, b, rtol=0, atol=1e-12) | (a.isna() & b.isna())
        assert same.all(), f"{col} changed when only data after the cutoff changed"


def test_labels_do_look_after_the_cutoff(two_worlds, tmp_path, monkeypatch):
    _, labels_a = features_and_labels(two_worlds / "original", tmp_path / "cache_a", monkeypatch)
    _, labels_b = features_and_labels(two_worlds / "future_changed", tmp_path / "cache_b", monkeypatch)
    assert not labels_a["duration_days"].equals(labels_b["duration_days"]), \
        "the labels did not change with the future: the leakage test would prove nothing"


def test_no_label_column_is_a_feature():
    import build_features

    import churn_labels

    assert {"event_60d", "duration_days", "event_observed", "churn_within_30d"} <= set(churn_labels.LABEL_COLUMNS)
    assert not set(churn_labels.LABEL_COLUMNS) & set(build_features.FEATURES)
