"""Backtest additions for the Phase 1 review (P0 4, P0 5, P1 7), on synthetic players.

    .venv/bin/python -m pytest src/models/tests/test_backtest_review.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest  # noqa: E402


def players(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Recent bettors churn less; value is independent of churn."""
    rng = np.random.default_rng(seed)
    recency = rng.integers(0, 30, n).astype(float)
    churn = (rng.random(n) < 0.1 + recency / 40).astype(int)
    return pd.DataFrame({"cutoff_date": "2026-01-01", "tenant_id": "t", "player_id": np.arange(n),
                         "days_since_last_bet": recency, "active_days_l30d": 30 - recency,
                         "active_days_l90d": 60 - recency, "days_since_first_bet": rng.integers(10, 150, n),
                         "ggr_eur_l90d": rng.gamma(1.0, 300.0, n), backtest.TARGET: churn})


def test_baselines_rank_like_recency_and_give_probabilities():
    train, test = players(seed=1), players(seed=2)
    out = backtest.fit_baselines(train, test)
    assert list(out.columns) == backtest.KEY + ["recency_prob", "simple_rft"]
    assert out[["recency_prob", "simple_rft"]].stack().between(0, 1).all()
    # isotonic: never lower for a longer silence
    order = test["days_since_last_bet"].argsort()
    assert np.all(np.diff(out["recency_prob"].to_numpy()[order]) >= -1e-12)


def scored(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["recency_prob"] = rows["days_since_last_bet"] / 30
    rows["simple_rft"] = rows["recency_prob"]
    rows["lgbm_optimised"] = rows["recency_prob"]
    rows["cox30"] = rows["recency_prob"] / 2
    return rows


def test_value_aware_catches_more_ggr_than_value_blind():
    out = pd.DataFrame(backtest.policies(scored(players()), 50)).set_index(["score", "weighting"])
    assert len(out) == 2 * len(backtest.POLICY_SCORES)
    assert (out.xs("value-aware", level="weighting")["ggr_at_risk_captured"]
            > out.xs("value-blind", level="weighting")["ggr_at_risk_captured"]).all()


def test_lists_that_rank_alike_overlap_fully():
    o = backtest.list_overlap(scored(players()), 50)
    assert o["top_n_overlap"] == 1.0 and o["spearman_probability"] > 0.999


def test_cox_horizons_use_only_known_labels():
    rows = players().assign(cox_p7=0.1, cox_p14=0.2, cox_p30=0.3)
    rows["cox_p30"] = rows["days_since_last_bet"] / 30
    rows["label_7d"], rows["label_14d"] = np.nan, np.nan                 # unknown: skipped
    rows["label_30d"] = rows[backtest.TARGET].astype(float)
    rows.loc[:9, "label_30d"] = np.nan
    out = backtest.cox_horizon_metrics(rows)
    assert [r["horizon_days"] for r in out] == [30] and out[0]["n_players"] == len(rows) - 10


def test_review_sections_are_empty_for_older_backtests():
    assert backtest.review_sections(None, 1000) == ""
    policies = pd.DataFrame([{"cutoff": "2026-01-01", **r} for r in backtest.policies(scored(players()), 50)])
    text = backtest.review_sections({"policies": policies, "overlap": pd.DataFrame(), "cox_horizons": pd.DataFrame()}, 50)
    assert "Predictive uplift" in text and "`cox_30d`" in text


def test_nowcast_replay_only_uses_cohorts_complete_on_the_replay_day(monkeypatch):
    import datetime as dt
    cutoffs = [dt.date(2026, m, 1) for m in range(1, 7)]
    features = pd.DataFrame([{"cutoff_date": c, "tenant_id": "t", "player_id": i, backtest.BRAND: 64}
                             for c in cutoffs for i in range(3)])
    seen = []

    def fake_labels(rows, data_end):
        known = rows["cutoff_date"].iloc[0] + dt.timedelta(days=60) <= data_end
        return rows.assign(**{backtest.TARGET: 1}), 0 if known else len(rows)

    def fake_rate(early, ref, day, brand):
        seen.append((early["cutoff_date"].iloc[0], ref["cutoff_date"].iloc[0], day))
        return 0.5

    monkeypatch.setattr(backtest, "with_labels", fake_labels)
    monkeypatch.setattr(backtest.calibration_level, "early_rate", fake_rate)
    out = backtest.nowcast_checks(features, dt.date(2026, 6, 15))
    assert out and {r["days_observed"] for r in out} <= set(backtest.NOWCAST_DAYS)
    for cohort, ref, day in seen:
        assert ref + dt.timedelta(days=60) <= day < cohort + dt.timedelta(days=60)   # complete reference, open cohort


def test_months_vs_recency_counts_wins_and_the_worst_month():
    rows = []
    for k, (model, rec) in enumerate([(0.85, 0.75), (0.80, 0.82), (0.88, 0.70)]):
        c = f"2026-0{k + 1}-01"
        rows += [{"cutoff": c, "comparator": backtest.CANDIDATE, "auc": model, "top_decile_precision": 0.9},
                 {"cutoff": c, "comparator": "recency", "auc": rec, "top_decile_precision": 0.8 if k else 0.85}]
    out = backtest.months_vs_recency(pd.DataFrame(rows), threshold=1.10).set_index("comparator").loc[backtest.CANDIDATE]
    assert out["months"] == 3 and out["months_auc_above_recency"] == 2
    assert out["auc_worst"] == 0.80 and out["auc_worst_month"] == "2026-02-01"
    assert out["months_ratio_at_threshold"] == 2 and np.isclose(out["ratio_worst"], 0.9 / 0.85)

