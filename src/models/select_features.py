"""Feature selection in four steps, using the training months only.

1. drop features with more than 40% missing values, almost constant, or a function of another
2. for each pair with |correlation| > 0.95, keep the one with the higher AUC
3. adversarial validation (training vs test months, no labels): drop features that drift
4. permutation importance: keep the smallest set within 0.002 of the full model's AUC

    selected, report, summary = select_features(data, split, features, adapter, target)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

MAX_MISSING_SHARE = 0.40
# Almost constant: the most common value is more than 95% of the known values.
NZV_MAX_MODE_SHARE = 0.95
# Only discrete features are checked for functional dependence. Continuous ones are caught in step 2.
DISCRETE_MAX_UNIQUE = 50
MAX_ABS_CORR = 0.95
# Drifting features are dropped while the adversarial AUC is above this. 0.60 removed the best signals.
ADV_AUC_THRESHOLD = 0.70
# A feature is drifting when it has more than this share of the adversarial importance.
ADV_IMPORTANCE_SHARE = 0.10
MAX_ADV_ROUNDS = 5
MAX_METRIC_LOSS = 0.002
PERMUTATION_REPEATS = 5
SEED = 42


@dataclass
class ModelAdapter:
    """How step 4 trains and scores the model (higher score is better)."""
    fit: Callable[[list[str]], object]
    score: Callable[[object, pd.DataFrame, list[str]], float]
    metric: str
    params: dict = field(default_factory=dict)  # for logging


def _near_zero_variance(s: pd.Series) -> bool:
    counts = s.value_counts()
    return len(counts) < 2 or counts.iloc[0] / counts.sum() > NZV_MAX_MODE_SHARE


def single_feature_auc(x: pd.Series, y: pd.Series) -> float:
    """AUC of one feature on its own, in any direction. Returns 0.5 if it cannot be measured."""
    known = x.notna()
    if known.sum() < 2 or y[known].nunique() < 2 or x[known].nunique() < 2:
        return 0.5
    auc = roc_auc_score(y[known], x[known])
    return max(auc, 1 - auc)


def _is_function_of(a: pd.Series, b: pd.Series) -> bool:
    """True if each value of b maps to only one value of a."""
    return bool(pd.DataFrame({"a": a, "b": b}).groupby("b")["a"].nunique().max() == 1)


def _step1(train: pd.DataFrame, features: list[str], auc: dict) -> tuple[list[str], list[dict]]:
    rows, kept = [], []
    for f in features:
        share = train[f].isna().mean()
        if share > MAX_MISSING_SHARE:
            rows.append({"feature": f, "step": 1, "reason": f"{share:.0%} missing"})
        elif _near_zero_variance(train[f]):
            rows.append({"feature": f, "step": 1, "reason": "near-zero variance"})
        else:
            kept.append(f)

    discrete = [f for f in kept if train[f].nunique() <= DISCRETE_MAX_UNIQUE]
    dropped = set()
    for a in discrete:
        for b in discrete:
            if a == b or a in dropped or b in dropped or not _is_function_of(train[a], train[b]):
                continue
            # If both carry the same information, keep the better one.
            if _is_function_of(train[b], train[a]) and auc[a] > auc[b]:
                continue
            dropped.add(a)
            rows.append({"feature": a, "step": 1, "reason": f"deterministic function of {b}"})
    return [f for f in kept if f not in dropped], rows


def _step2(train: pd.DataFrame, features: list[str], auc: dict) -> tuple[list[str], list[dict]]:
    corr = train[features].corr().abs()
    pairs = [
        (corr.loc[a, b], a, b) for i, a in enumerate(features) for b in features[i + 1:]
        if corr.loc[a, b] > MAX_ABS_CORR
    ]
    rows, dropped = [], set()
    for value, a, b in sorted(pairs, reverse=True):
        if a in dropped or b in dropped:
            continue
        loser, winner = (a, b) if auc[a] < auc[b] else (b, a)
        dropped.add(loser)
        rows.append({"feature": loser, "step": 2,
                     "reason": f"|corr| {value:.3f} with {winner} (AUC {auc[loser]:.3f} < {auc[winner]:.3f})"})
    return [f for f in features if f not in dropped], rows


def adversarial_validation(train_months: pd.DataFrame, test_months: pd.DataFrame, features: list[str]):
    """AUC of a LightGBM that tells training rows from test rows, and each feature's importance share."""
    X = pd.concat([train_months[features], test_months[features]], ignore_index=True)
    y = np.r_[np.zeros(len(train_months)), np.ones(len(test_months))]
    oof, importance = np.zeros(len(X)), np.zeros(len(features))
    for fit_idx, pred_idx in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        clf = lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbose=-1, importance_type="gain")
        clf.fit(X.iloc[fit_idx], y[fit_idx])
        oof[pred_idx] = clf.predict_proba(X.iloc[pred_idx])[:, 1]
        importance += clf.feature_importances_
    return roc_auc_score(y, oof), pd.Series(importance / importance.sum(), index=features)


def _step3(train_months, test_months, features) -> tuple[list[str], list[dict], float, float, pd.Series]:
    """Drop drifting features until the two periods look alike."""
    rows, kept = [], list(features)
    first_auc, first_share = adversarial_validation(train_months, test_months, kept)
    adv_auc, share = first_auc, first_share
    for _ in range(MAX_ADV_ROUNDS):
        drifting = list(share[share > ADV_IMPORTANCE_SHARE].index)
        # Compare with 3 decimals: at exactly 0.700 it used to drop the recency (2026-10-07).
        if round(adv_auc, 3) <= ADV_AUC_THRESHOLD or not drifting:
            break
        for f in drifting:
            rows.append({"feature": f, "step": 3,
                         "reason": f"drifting: {share[f]:.0%} of adversarial importance (AUC {adv_auc:.3f})"})
        kept = [f for f in kept if f not in drifting]
        adv_auc, share = adversarial_validation(train_months, test_months, kept)
    # The report shows the importance from the first round.
    return kept, rows, first_auc, adv_auc, first_share


def permutation_importance(adapter: ModelAdapter, model, valid: pd.DataFrame, features: list[str]) -> pd.Series:
    """Mean drop in the validation score when a feature is shuffled."""
    rng = np.random.default_rng(SEED)
    base = adapter.score(model, valid, features)
    drops = {}
    for f in features:
        scores = []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = valid.copy()
            shuffled[f] = rng.permutation(shuffled[f].to_numpy())
            scores.append(adapter.score(model, shuffled, features))
        drops[f] = base - float(np.mean(scores))
    return pd.Series(drops).sort_values(ascending=False)


def _step4(adapter: ModelAdapter, valid: pd.DataFrame, features: list[str]):
    full_model = adapter.fit(features)
    full_score = adapter.score(full_model, valid, features)
    importance = permutation_importance(adapter, full_model, valid, features)
    ranked = list(importance.index)
    # Smallest top-k set within MAX_METRIC_LOSS of the full model.
    for k in range(1, len(ranked) + 1):
        score = adapter.score(adapter.fit(ranked[:k]), valid, ranked[:k])
        if score >= full_score - MAX_METRIC_LOSS:
            break
    kept = ranked[:k]
    rows = [{"feature": f, "step": 4, "reason": f"permutation importance {importance[f]:.4f}, outside the top {k}"}
            for f in ranked[k:]]
    return kept, rows, full_score, score, importance


def select_features(
    data: pd.DataFrame, split: pd.Series, features: list[str], adapter: ModelAdapter, target: str
) -> tuple[list[str], pd.DataFrame, dict]:
    """Run the four steps. `split` is 'train', 'valid' or 'test'."""
    train, valid = data[split == "train"], data[split == "valid"]
    training_months, test_months = data[split != "test"], data[split == "test"]
    # Only the train labels are used here, so step 4 is scored on clean data.
    auc = {f: single_feature_auc(train[f], train[target]) for f in features}

    kept, rows1 = _step1(train, features, auc)
    kept, rows2 = _step2(train, kept, auc)
    kept, rows3, adv_auc, adv_auc_after, adv_share = _step3(training_months, test_months, kept)
    kept, rows4, full_score, selected_score, importance = _step4(adapter, valid, kept)

    report = pd.DataFrame({"feature": features})
    report["single_feature_auc"] = report["feature"].map(auc).round(4)
    report["adversarial_importance"] = report["feature"].map(adv_share).round(4)
    report["permutation_importance"] = report["feature"].map(importance).round(5)
    decisions = pd.DataFrame(rows1 + rows2 + rows3 + rows4, columns=["feature", "step", "reason"])
    report = report.merge(decisions, on="feature", how="left")
    report["selected"] = report["feature"].isin(kept)
    report = report.sort_values(["selected", "step", "permutation_importance"], ascending=[False, True, False])

    summary = {
        "n_features_candidate": len(features),
        "n_dropped_step1": len(rows1), "n_dropped_step2": len(rows2),
        "n_dropped_step3": len(rows3), "n_dropped_step4": len(rows4),
        "n_features_selected": len(kept),
        "adversarial_auc": adv_auc,
        "adversarial_auc_after_step3": adv_auc_after,
        f"selection_full_valid_{adapter.metric}": full_score,
        f"selection_selected_valid_{adapter.metric}": selected_score,
    }
    return kept, report.reset_index(drop=True), summary
