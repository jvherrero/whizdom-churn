"""Train one model from catalog/model_library_catalog.json on a training dataset, logged to MLflow.

    .venv/bin/python src/models/train.py --model-id lightgbm_classifier
    .venv/bin/python src/models/train.py --model-id cox_ph
    .venv/bin/python src/models/train.py --model-id lightgbm_classifier --params '{"n_estimators": 500}'

The model class and its default parameters come from the catalog; --params only overrides them.
The catalog interface decides how the model is trained:
    sklearn_estimator    classification on event_60d
    lifelines_survival   survival on duration_days + event_observed (Cox PH)

Split: the last cutoff is the test month (out of time). The earlier cutoffs are the training
months, split by partyId into train/valid (85/15), so one player is never in both.
Before the final fit, the Stage 2 feature selection (select_features.py) runs on the training
months only. --no-selection skips it.

A classifier with a `class_balance` list in the catalog (none / scale_pos_weight / is_unbalance)
is fitted once per option; the one with the lowest log-loss after calibration (cross-fitted on the
validation rows) is kept, not the best AUC. The catalog's class_balance_preferred option (class
weights) is kept unless another one beats it by more than class_balance_min_gain. Stage 2 selection runs before, with plain log-loss.

A classifier with calibration methods in the catalog (isotonic, Platt) gets each one fitted on the
validation rows, per brand; the one with the lowest expected calibration error (ECE) on the test
month is kept, and the reliability curve shows them all. churn_probability() applies it.

Brands: features are computed per brand upstream; here one model covers every brand. LightGBM
gets brandId as a categorical feature, Cox PH is stratified by brand, calibration is per brand,
and metrics are reported per brand. brandId never goes through Stage 2 selection.

k-means player segments (segments.py) are fitted on the training months and added as one-hot
input features before Stage 2, which decides whether the model keeps them. --no-segments skips it.
"""

from __future__ import annotations

import argparse
import ast
import getpass
import importlib
import json
import os
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
from build_features import brand_label
from build_training_features import _git_commit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segments as segmentation
import select_features as selection
from calibration import BrandCalibrator

CATALOG_PATH = PROJECT_ROOT / "catalog/model_library_catalog.json"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
MLFLOW_EXPERIMENT = "whizdom-churn-training"

TARGET = "event_60d"
DURATION, EVENT = "duration_days", "event_observed"
BRAND = "brandId"
# brandId is not a selectable feature: the model adds it itself (catalog brand_feature / brand_strata).
NON_FEATURES = ["cutoff_date", "operator", BRAND, "partyId", TARGET, DURATION, EVENT]
SPLITS = ("train", "valid", "test")
N_TEST_CUTOFFS = 1  # the latest cutoff(s) are the test months
VALID_SHARE = 0.15  # of the training-month players
SEED = 42
# Single-feature reference: how far the model gets beyond "days since the last bet".
BENCHMARK_FEATURE = "days_since_last_active"
ECE_BINS = 10
CALIBRATOR_ARTIFACT = "calibration/isotonic.joblib"
# Isotonic regression gives exactly 0 or 1 at the tails (a validation bin with only one class).
# A sure 0 or 1 is never true for a player and blows up log-loss on any miss, so bound it.
CALIBRATION_BOUND = 0.005
CALIBRATION_FOLDS = 5  # cross-fitting folds to score "log-loss after calibration" on the validation rows


def load_catalog_entry(model_id: str) -> tuple[dict, dict, str]:
    """The catalog entry for `model_id`, its interface and the catalog version."""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = {m["id"]: m for m in catalog["models"]}
    if model_id not in entries:
        raise KeyError(f"{model_id!r} is not in {CATALOG_PATH.name}; available: {sorted(entries)}")
    entry = entries[model_id]
    return entry, catalog["interfaces"][entry["interface"]], catalog["catalog_version"]


def instantiate(entry: dict, overrides: dict | None = None):
    module_name, class_name = entry["import_path"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**{**entry["default_params"], **(overrides or {})})


def player_key(df: pd.DataFrame) -> pd.Series:
    """One id per player: partyId is only unique within a brand (operators may reuse numbers)."""
    return df[BRAND].astype("int64") * 10**10 + df["partyId"].astype("int64")


def split_rows(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """'test' for the latest cutoff(s); 'train' / 'valid' by player for the earlier ones."""
    cutoffs = sorted(df["cutoff_date"].unique())
    if len(cutoffs) <= N_TEST_CUTOFFS:
        raise ValueError(f"need more than {N_TEST_CUTOFFS} cutoff(s) for an out-of-time test, got {cutoffs}")
    is_test = df["cutoff_date"].isin(cutoffs[-N_TEST_CUTOFFS:])
    keys = player_key(df)
    players = np.random.default_rng(seed).permutation(np.sort(keys[~is_test].unique()))
    valid_players = set(players[: int(len(players) * VALID_SHARE)])
    split = pd.Series(np.where(keys.isin(valid_players), "valid", "train"), index=df.index)
    split[is_test] = "test"
    return split


def model_columns(entry: dict, features: list[str]) -> list[str]:
    """The columns the model gets: the selected features, plus brandId when the catalog says the
    model takes it as a categorical feature. brandId never goes through Stage 2 selection."""
    if entry.get("brand_feature") == "categorical":
        return list(dict.fromkeys([*features, BRAND]))
    return list(features)


def model_input(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """What a classifier sees: brandId as a category, never as a number (64 is not "near" 63).
    LightGBM maps the categories to the ones it was trained with; an unknown brand becomes missing."""
    X = rows[columns]
    if BRAND in X:
        X = X.assign(**{BRAND: X[BRAND].astype("int64").astype("category")})
    return X


def cox_strata(entry: dict) -> list[str]:
    """Cox PH stratified by brand: one baseline hazard per brand, shared covariate effects."""
    return [BRAND] if entry.get("brand_strata") else []


def cox_input(model, rows: pd.DataFrame) -> pd.DataFrame:
    """The covariates a fitted Cox model uses, plus its strata columns (needed to predict)."""
    strata = model.strata or []
    strata = [strata] if isinstance(strata, str) else list(strata)
    X = rows[list(model.params_.index) + strata]
    return X.assign(**{s: X[s].astype("int64") for s in strata})


def class_balance_options(entry: dict, train: pd.DataFrame) -> dict[str, dict]:
    """The catalog's candidate class-imbalance settings, as model parameters."""
    names = entry.get("class_balance", [])
    names = [names] if isinstance(names, str) else names
    n_pos = int(train[TARGET].sum())
    options = {
        "none": {},  # plain log-loss
        "scale_pos_weight": {"scale_pos_weight": round((len(train) - n_pos) / n_pos, 6)},
        "is_unbalance": {"is_unbalance": True},
    }
    return {name: options[name] for name in names}


def make_adapter(
    entry: dict, params: dict | None, train: pd.DataFrame, valid: pd.DataFrame | None = None
) -> selection.ModelAdapter:
    """How to fit the catalog model on the train rows with a given feature list, and score it.
    With `valid` and the catalog's fit_params, LightGBM stops early on the validation rows.
    brandId is added by the model itself (categorical feature or Cox strata), never selected."""
    if entry["interface"] == "sklearn_estimator":
        model_params = {**entry["default_params"], **(params or {})}
        fit_params = entry.get("fit_params", {})
        early_stopping = (
            valid is not None and "early_stopping_rounds" in fit_params
            and entry["import_path"].startswith("lightgbm.")
        )

        def fit(features):
            model = instantiate(entry, model_params)
            columns = model_columns(entry, features)
            if not early_stopping:
                return model.fit(model_input(train, columns), train[TARGET])
            # Stops when AUC or log-loss on the validation rows has not improved for
            # early_stopping_rounds; predictions then use the best iteration.
            return model.fit(
                model_input(train, columns), train[TARGET],
                eval_X=model_input(valid, columns), eval_y=valid[TARGET], eval_metric=fit_params["eval_metric"],
                callbacks=[lgb.early_stopping(fit_params["early_stopping_rounds"], first_metric_only=fit_params.get("first_metric_only", False), verbose=False)],
            )

        def score(model, rows, features):
            proba = model.predict_proba(model_input(rows, model_columns(entry, features)))[:, 1]
            return roc_auc_score(rows[TARGET], proba)

        logged = {**model_params, **({f"fit__{k}": v for k, v in fit_params.items() if k != "notes"}
                                     if early_stopping else {})}
        return selection.ModelAdapter(fit, score, "roc_auc", logged)

    strata = cox_strata(entry)

    def fit(features):
        model = instantiate(entry, params)
        # A constant column makes the Cox fit singular. cluster_col: a player shows up at
        # several cutoffs, so use robust standard errors (one cluster per brand + partyId).
        features = [f for f in features if f != BRAND and train[f].nunique() > 1]
        frame = train[features + [DURATION, EVENT] + strata].assign(player_key=player_key(train))
        frame = frame.assign(**{s: frame[s].astype("int64") for s in strata})
        model.fit(frame, duration_col=DURATION, event_col=EVENT, cluster_col="player_key",
                  strata=strata or None)
        return model

    def score(model, rows, features):
        # Higher hazard means an earlier churn, so the risk score is the negative hazard.
        return concordance_index(rows[DURATION], -model.predict_partial_hazard(cox_input(model, rows)), rows[EVENT])

    return selection.ModelAdapter(fit, score, "c_index", {**entry["default_params"], **(params or {})})


def expected_calibration_error(y, p, n_bins: int = ECE_BINS) -> float:
    """Mean |observed rate - mean prediction| over equal-size bins of the prediction, weighted by size."""
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    frame = pd.DataFrame({"y": np.asarray(y), "p": p, "bin": bins})
    per_bin = frame.groupby("bin").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((per_bin["n"] * (per_bin["y"] - per_bin["p"]).abs()).sum() / per_bin["n"].sum())


def raw_probability(entry: dict, model, rows: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(model_input(rows, model_columns(entry, features)))[:, 1]


def calibration_methods(entry: dict) -> list[str]:
    """The catalog's calibration candidates: a list under calibration.methods (or one method name)."""
    cal = entry.get("calibration")
    if not cal:
        return []
    return [cal] if isinstance(cal, str) else list(cal.get("methods", []))


def fit_calibrator(entry: dict, model, valid: pd.DataFrame, features: list[str],
                   method: str | None = None) -> BrandCalibrator | None:
    """A calibrator per brand (all brands as the fallback) from the raw validation predictions to
    the observed churn, with `method` (default: the catalog's first one)."""
    methods = calibration_methods(entry)
    if not methods:
        return None
    raw = raw_probability(entry, model, valid, features)
    return BrandCalibrator(CALIBRATION_BOUND, method=method or methods[0]).fit(raw, valid[TARGET], valid[BRAND])


def choose_calibrator(entry: dict, model, data: pd.DataFrame, split: pd.Series, features: list[str]):
    """Every catalog calibration method fitted on the validation rows; the one with the lowest
    expected calibration error on the test month is kept (catalog: calibration.choose_by)."""
    candidates = {m: fit_calibrator(entry, model, data[split == "valid"], features, m)
                  for m in calibration_methods(entry)}
    if not candidates:
        return None, None, {}
    columns = model_columns(entry, features)
    rows = []
    for method, calibrator in candidates.items():
        row = {"method": method}
        for name in ("valid", "test"):
            part = data[split == name]
            p = churn_probability(model, calibrator, part, columns)
            row.update({f"{name}_ece": expected_calibration_error(part[TARGET], p),
                        f"{name}_log_loss": log_loss(part[TARGET], p), f"{name}_brier": brier_score_loss(part[TARGET], p)})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("test_ece").reset_index(drop=True)
    table["chosen"] = table.index == 0
    return candidates[table.loc[0, "method"]], table, candidates


def cross_fitted_calibration(raw: np.ndarray, valid: pd.DataFrame, method: str = "isotonic") -> np.ndarray:
    """Out-of-fold calibrated probabilities on the validation rows: each fold is calibrated (per
    brand) on the other folds, grouped by player, so the score is honest."""
    out = np.zeros(len(raw))
    y, brands = valid[TARGET].to_numpy(), valid[BRAND].to_numpy()
    folds = StratifiedGroupKFold(n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=SEED)
    for fit_idx, pred_idx in folds.split(raw, y, player_key(valid)):
        calibrator = BrandCalibrator(CALIBRATION_BOUND, method=method).fit(raw[fit_idx], y[fit_idx], brands[fit_idx])
        out[pred_idx] = calibrator.predict(raw[pred_idx], brands[pred_idx])
    return out


def compare_class_balance(entry: dict, params: dict | None, data: pd.DataFrame, split: pd.Series,
                          features: list[str]):
    """Fit one model per class-imbalance option and keep the lowest log-loss after calibration
    (cross-fitted on the validation rows), not the best AUC: reweighting distorts probabilities.
    The test month is reported, never used to choose."""
    train, valid, test = data[split == "train"], data[split == "valid"], data[split == "test"]
    rows, fitted = [], {}
    for name, balance in class_balance_options(entry, train).items():
        adapter = make_adapter(entry, {**balance, **(params or {})}, train, valid)
        fitted_model = adapter.fit(features)
        raw_valid = raw_probability(entry, fitted_model, valid, features)
        oof = cross_fitted_calibration(raw_valid, valid, (calibration_methods(entry) or ["isotonic"])[0])
        calibrator = fit_calibrator(entry, fitted_model, valid, features)
        p_test = churn_probability(fitted_model, calibrator, test, model_columns(entry, features))
        fitted[name] = (adapter, fitted_model)
        rows.append({
            "class_balance": name,
            "valid_calibrated_log_loss_cv": log_loss(valid[TARGET], oof),
            "valid_calibrated_ece_cv": expected_calibration_error(valid[TARGET], oof),
            "valid_raw_log_loss": log_loss(valid[TARGET], raw_valid),
            "valid_raw_mean_prediction": float(raw_valid.mean()),
            "valid_roc_auc": roc_auc_score(valid[TARGET], raw_valid),
            "best_iteration": getattr(fitted_model, "best_iteration_", None),
            "test_calibrated_log_loss": log_loss(test[TARGET], p_test),
            "test_calibrated_roc_auc": roc_auc_score(test[TARGET], p_test),
        })
    table = pd.DataFrame(rows).sort_values("valid_calibrated_log_loss_cv").reset_index(drop=True)
    best = table.loc[0, "class_balance"]
    # The catalog's preferred option (class weights by default, which matter for brands with a very
    # small churn class) is kept unless another one beats it by a clear margin; a smaller gain is noise.
    preferred = entry.get("class_balance_preferred", "none")
    min_gain = entry.get("class_balance_min_gain", 0.0)
    if best != preferred and preferred in fitted:
        loss = table.set_index("class_balance")["valid_calibrated_log_loss_cv"]
        if loss[preferred] - loss[best] <= min_gain:
            best = preferred
    table["chosen"] = table["class_balance"] == best
    return best, *fitted[best], table, fitted


def tune_hyperparameters(entry: dict, base_params: dict, train: pd.DataFrame, valid: pd.DataFrame,
                         features: list[str], n_trials: int | None = None) -> tuple[dict, int, pd.DataFrame]:
    """Optuna Bayesian search (TPE) over the catalog's search space. Objective: validation log-loss
    of an early-stopped model. Trials that fall behind are cut by the median pruner, which gets the
    validation log-loss after every boosting round. Returns the best parameters, the early-stopped
    tree count of the best trial and one row per trial."""
    import optuna

    search = entry["search"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fit_params = entry.get("fit_params", {})
    columns = model_columns(entry, features)
    X_train, X_valid = model_input(train, columns), model_input(valid, columns)

    def suggest(trial):
        out = {}
        for name, spec in search["space"].items():
            if spec["type"] == "int":
                out[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
            else:
                out[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        return out

    def objective(trial):
        params = {**base_params, **search.get("fixed_params", {}), **suggest(trial)}
        model = instantiate(entry, params)

        def report(env):  # the median pruner sees the validation log-loss after each round
            loss = next(r[2] for r in env.evaluation_result_list if r[1] == "binary_logloss")
            trial.report(loss, env.iteration)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model.fit(X_train, train[TARGET], eval_X=X_valid, eval_y=valid[TARGET],
                  eval_metric=fit_params.get("eval_metric", ["binary_logloss"]),
                  callbacks=[lgb.early_stopping(fit_params.get("early_stopping_rounds", 100), first_metric_only=fit_params.get("first_metric_only", False), verbose=False), report])
        trial.set_user_attr("best_iteration", int(model.best_iteration_ or model.n_estimators))
        return float(model.best_score_["valid_0"]["binary_logloss"])

    pruner = search.get("pruner", {})
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=search.get("seed", SEED)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=pruner.get("n_startup_trials", 5),
                                           n_warmup_steps=pruner.get("n_warmup_steps", 0)),
    )
    study.optimize(objective, n_trials=n_trials or search["n_trials"])
    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    best = {**search.get("fixed_params", {}), **study.best_params}
    return best, int(study.best_trial.user_attrs["best_iteration"]), trials


def regularisation_check(entry: dict, params: dict, train: pd.DataFrame, valid: pd.DataFrame,
                         features: list[str], model) -> tuple[object, dict, pd.DataFrame]:
    """If the train - validation AUC gap is over the catalog's max_gap, raise min_child_samples and
    lambda_l2 (reg_lambda) and lower num_leaves, one round at a time, until the gap closes. A round
    that costs more than max_valid_auc_loss of validation AUC is rejected and the previous model kept.
    Each round refits with early stopping on the validation rows. Returns the model, its parameters
    and one row per round."""
    rule = entry["regularisation_check"]
    columns = model_columns(entry, features)

    def aucs(m):
        return (roc_auc_score(train[TARGET], m.predict_proba(model_input(train, columns))[:, 1]),
                roc_auc_score(valid[TARGET], m.predict_proba(model_input(valid, columns))[:, 1]))

    current = {**model.get_params(), **params}
    train_auc, valid_auc = aucs(model)
    start_valid_auc = valid_auc
    rows = [{"round": 0, "train_auc": train_auc, "valid_auc": valid_auc, "gap": train_auc - valid_auc,
             "accepted": True, **{k: current.get(k) for k in rule["steps"]}}]
    for round_ in range(1, rule["max_rounds"] + 1):
        if train_auc - valid_auc <= rule["max_gap"]:
            break
        candidate = dict(current)
        for name, step in rule["steps"].items():
            value = max(float(candidate.get(name) or 0.0), step.get("start_at_least", 0.0)) * step["factor"]
            value = min(value, step["max"]) if "max" in step else max(value, step["min"])
            candidate[name] = int(round(value)) if name in ("min_child_samples", "num_leaves") else value
        # The tree count is found again by early stopping (up to the catalog's maximum).
        candidate["n_estimators"] = entry["default_params"].get("n_estimators", candidate["n_estimators"])
        refit_params = {k: v for k, v in candidate.items() if k in params or k in rule["steps"] or k == "n_estimators"}
        new_model = make_adapter(entry, refit_params, train, valid).fit(features)
        new_train_auc, new_valid_auc = aucs(new_model)
        accepted = new_valid_auc >= start_valid_auc - rule["max_valid_auc_loss"]
        rows.append({"round": round_, "train_auc": new_train_auc, "valid_auc": new_valid_auc,
                     "gap": new_train_auc - new_valid_auc, "accepted": accepted,
                     **{k: candidate[k] for k in rule["steps"]}})
        if not accepted:
            break  # closing the gap further would cost validation AUC: keep the previous model
        model, train_auc, valid_auc = new_model, new_train_auc, new_valid_auc
        current = {**candidate, "n_estimators": int(new_model.best_iteration_ or candidate["n_estimators"])}
        params = {**refit_params, "n_estimators": current["n_estimators"]}
    return model, params, pd.DataFrame(rows)


def final_pruning(entry: dict, params: dict, model, train: pd.DataFrame, valid: pd.DataFrame,
                  features: list[str]) -> tuple[object, list[str], pd.DataFrame, dict]:
    """The permutation-importance rule of Stage 2, applied again to the final (tuned, regularised)
    model: rank the features by how much shuffling each one costs in validation AUC, then keep the
    smallest top-k whose retrained model stays within max_valid_auc_loss of the full model. Each
    retrain uses the final parameters, with early stopping on the validation rows for the tree count.
    The pruned set is what the run logs as feature_columns, so it is what scoring (serving) uses."""
    rule = entry["final_pruning"]
    refit = make_adapter(entry, {**params, "n_estimators": entry["default_params"].get("n_estimators", 2000)},
                         train, valid)
    full_auc = refit.score(model, valid, features)
    importance = selection.permutation_importance(refit, model, valid, features)
    ranked = list(importance.index)
    kept, pruned_model, pruned_auc = ranked, model, full_auc
    for k in range(1, len(ranked)):
        candidate = refit.fit(ranked[:k])
        auc = refit.score(candidate, valid, ranked[:k])
        if auc >= full_auc - rule["max_valid_auc_loss"]:
            kept, pruned_model, pruned_auc = ranked[:k], candidate, auc
            break
    table = pd.DataFrame({"feature": ranked, "permutation_importance": importance.values,
                          "kept": [f in kept for f in ranked]})
    summary = {"pruning_n_features_before": len(ranked), "pruning_n_features_after": len(kept),
               "pruning_full_valid_roc_auc": full_auc, "pruning_pruned_valid_roc_auc": pruned_auc,
               "pruning_valid_auc_change": pruned_auc - full_auc}
    return pruned_model, kept, table, summary


def _timed(fn, repeats: int = 3) -> float:
    """Best of `repeats` wall-clock runs, in seconds (the minimum is the least noisy estimate)."""
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def inference_seconds_per_100k(entry: dict, model, rows: pd.DataFrame, features: list[str]) -> float:
    """Scoring time for 100k players: the rows are resampled to that size and scored the way
    score.py does it (LightGBM probability, or Cox median survival)."""
    n = entry["inference_budget"].get("benchmark_players", 100_000)
    sample = rows.sample(n=n, replace=True, random_state=SEED)
    if entry["interface"] == "sklearn_estimator":
        X = model_input(sample, model_columns(entry, features))
        seconds = _timed(lambda: model.predict_proba(X))
    else:
        X = cox_input(model, sample)
        seconds = _timed(lambda: model.predict_median(X), repeats=1)
    return seconds * 100_000 / n


def inference_budget(entry: dict, params: dict, model, data: pd.DataFrame, split: pd.Series,
                     features: list[str]) -> tuple[object, dict]:
    """Project the scoring time of a full-population run on the serving container; if it goes over
    the budget, cap the LightGBM tree count (time is about proportional to trees) and refit."""
    rule = entry["inference_budget"]
    population = rule.get("population_players") or int((data["cutoff_date"] == data["cutoff_date"].max()).sum())
    per_100k = inference_seconds_per_100k(entry, model, data[split == "test"], features)
    projected = per_100k * population / 100_000 * rule.get("container_slowdown", 1.0)
    info = {"inference_seconds_per_100k": per_100k, "inference_population_players": population,
            "inference_projected_full_run_seconds": projected, "inference_tree_cap": 0}
    trees = getattr(model, "n_estimators_", None) or getattr(model, "n_estimators", None)
    if projected > rule["max_seconds_full_run"] and entry["interface"] == "sklearn_estimator" and trees:
        cap = max(1, int(trees * rule["max_seconds_full_run"] / projected * rule.get("safety_margin", 0.8)))
        model = make_adapter(entry, {**params, "n_estimators": cap}, data[split == "train"]).fit(features)
        per_100k = inference_seconds_per_100k(entry, model, data[split == "test"], features)
        projected = per_100k * population / 100_000 * rule.get("container_slowdown", 1.0)
        info.update(inference_tree_cap=cap, inference_seconds_per_100k=per_100k,
                    inference_projected_full_run_seconds=projected)
    info["inference_within_budget"] = int(projected <= rule["max_seconds_full_run"])
    return model, info


STEP_NAMES = {1: "defaults", 2: "class_imbalance", 3: "hyperparameter_search", 4: "regularisation_check",
              5: "feature_pruning", 6: "calibration", 9: "inference_cost"}


def step_metrics(entry: dict, model, features: list[str], calibrator, data: pd.DataFrame, split: pd.Series) -> dict:
    """The same yardstick for every optimisation step: validation AUC (where decisions are made) and,
    on the test month, AUC, log-loss and ECE of the delivered probability. A step before the
    calibration step gets the catalog's first calibration method fitted on validation, so log-loss and
    ECE are compared like for like."""
    calibrator = calibrator or fit_calibrator(entry, model, data[split == "valid"], features)
    columns = model_columns(entry, features)
    valid, test = data[split == "valid"], data[split == "test"]
    p_test = churn_probability(model, calibrator, test, columns)
    return {"valid_roc_auc": roc_auc_score(valid[TARGET], raw_probability(entry, model, valid, features)),
            "test_roc_auc": roc_auc_score(test[TARGET], p_test),
            "test_log_loss": log_loss(test[TARGET], p_test),
            "test_ece": expected_calibration_error(test[TARGET], p_test),
            "n_features": len(features),
            "n_trees": int(getattr(model, "n_estimators_", 0) or 0)}


def log_optimisation_steps(entry: dict, steps: list[dict], data: pd.DataFrame, split: pd.Series,
                           parent_name: str, brand: str) -> pd.DataFrame:
    """One nested MLflow run per optimisation step, with its metrics and the change (delta) against
    the previous step. Called inside the training run, which becomes their parent."""
    rows, previous = [], None
    for snap in steps:
        m = step_metrics(entry, snap["model"], snap["features"], snap.get("calibrator"), data, split)
        delta = {f"delta_{k}": m[k] - previous[k] for k in ("valid_roc_auc", "test_roc_auc", "test_log_loss", "test_ece")} \
            if previous else {}
        name = f"step{snap['step']}_{STEP_NAMES[snap['step']]}"
        with mlflow.start_run(run_name=f"{parent_name}_{name}", nested=True):
            mlflow.set_tags({"step": snap["step"], "step_name": STEP_NAMES[snap["step"]], "brand_id": brand,
                             "role": "optimisation_step"})
            mlflow.log_param("note", snap.get("note", ""))
            mlflow.log_param("feature_columns", model_columns(entry, snap["features"]))
            mlflow.log_params({f"model__{k}": v for k, v in snap["params"].items() if not k.startswith("fit__")})
            mlflow.log_metrics({**m, **delta, **snap.get("extra", {})})
        rows.append({"step": snap["step"], "name": STEP_NAMES[snap["step"]], **m, **delta, "note": snap.get("note", "")})
        previous = m
    return pd.DataFrame(rows)


def search_penalizer(entry: dict, params: dict | None, train: pd.DataFrame, valid: pd.DataFrame,
                     features: list[str]):
    """Cox PH: the only tuning is the L2 penalty. Each value of the catalog grid is fitted on train
    and the one with the highest validation concordance (c-index) is kept."""
    rows, fitted = [], {}
    for penalizer in entry["search"]["grid"]:
        adapter = make_adapter(entry, {**(params or {}), "penalizer": penalizer}, train)
        model = adapter.fit(features)
        fitted[penalizer] = (adapter, model)
        rows.append({"penalizer": penalizer, "valid_c_index": adapter.score(model, valid, features)})
    table = pd.DataFrame(rows).sort_values("valid_c_index", ascending=False).reset_index(drop=True)
    table["chosen"] = table.index == 0
    adapter, model = fitted[table.loc[0, "penalizer"]]
    return model, adapter, table


def churn_probability(model, calibrator, rows: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """The model's churn probability for `rows` (with a brandId column), calibrated per brand when
    there is a calibrator. `columns` are the model's input columns (feature_columns of the run)."""
    raw = model.predict_proba(model_input(rows, columns))[:, 1]
    if calibrator is None:
        return raw
    if isinstance(calibrator, IsotonicRegression):  # runs from before the per-brand calibration
        return calibrator.predict(raw)
    return calibrator.predict(raw, rows[BRAND])


def load_calibrator(run_id: str):
    """The calibrator logged with a training run, or None for a run without one."""
    try:
        return joblib.load(mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{CALIBRATOR_ARTIFACT}"))
    except Exception:
        return None


def _classification_metrics(y, p) -> dict:
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "mean_prediction": float(np.mean(p)),
        "churn_rate": float(np.mean(y)),
    }


def _reliability(entry, data: pd.DataFrame, split: pd.Series, model, calibrator, features,
                 candidates: dict | None = None) -> tuple[Figure, pd.DataFrame]:
    """Reliability curve (observed churn rate vs mean prediction per bin): raw and every calibration
    candidate, the chosen one marked."""
    fig = Figure(figsize=(11, 4.8))
    axes = fig.subplots(1, 2)
    rows = []
    for ax, name in zip(axes, ("valid", "test")):
        part = data[split == name]
        curves = {"raw": raw_probability(entry, model, part, features)}
        for method, cal in (candidates or ({"calibrated": calibrator} if calibrator is not None else {})).items():
            label = f"{method} (chosen)" if cal is calibrator and candidates else method
            curves[label] = churn_probability(model, cal, part, model_columns(entry, features))
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="perfect")
        for label, p in curves.items():
            observed, predicted = calibration_curve(part[TARGET], p, n_bins=ECE_BINS, strategy="quantile")
            ece = expected_calibration_error(part[TARGET], p)
            ax.plot(predicted, observed, marker="o", label=f"{label} (ECE {ece:.3f})")
            rows += [{"split": name, "probabilities": label, "bin": i, "mean_prediction": pp, "observed_rate": oo}
                     for i, (pp, oo) in enumerate(zip(predicted, observed))]
        ax.set_title(f"{name} ({'fit here' if name == 'valid' else 'out of time'})")
        ax.set_xlabel("mean predicted churn probability"); ax.set_ylabel("observed churn rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left")
    fig.suptitle("Reliability curve, LightGBM: raw vs calibrated (fitted on validation, chosen by test ECE)")
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


def _roc(entry, data: pd.DataFrame, split: pd.Series, model, features) -> tuple[Figure, pd.DataFrame]:
    """ROC curves of train and validation together (plus the out-of-time test): a train curve far
    above the validation one means the model memorises the training players (overfitting).
    Raw probabilities: calibration does not change the ranking, so it does not change the ROC."""
    fig = Figure(figsize=(6.5, 6))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1, label="random (AUC 0.5)")
    rows, auc = [], {}
    styles = {"train": dict(linewidth=2), "valid": dict(linewidth=2), "test": dict(linestyle="--", linewidth=1.5)}
    for name in SPLITS:
        part = data[split == name]
        p = raw_probability(entry, model, part, features)
        fpr, tpr, _ = roc_curve(part[TARGET], p)
        auc[name] = roc_auc_score(part[TARGET], p)
        label = {"train": "train", "valid": "validation", "test": "test (out of time)"}[name]
        ax.plot(fpr, tpr, label=f"{label} (AUC {auc[name]:.3f})", **styles[name])
        # A thinned copy of the curve, enough to redraw it.
        step = max(1, len(fpr) // 200)
        rows += [{"split": name, "false_positive_rate": f, "true_positive_rate": t}
                 for f, t in zip(fpr[::step], tpr[::step])]
    ax.set_xlabel("false positive rate (non-churners flagged)")
    ax.set_ylabel("true positive rate (churners caught)")
    ax.set_title(f"ROC curve, LightGBM\ntrain - validation AUC gap: {auc['train'] - auc['valid']:+.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


def _metrics_by_brand(data: pd.DataFrame, split: pd.Series, scores: dict[str, np.ndarray], kind: str) -> pd.DataFrame:
    """The same metrics, brand by brand: a good global number can hide a brand where the model fails."""
    out = []
    for name, values in scores.items():
        part = data[split == name].assign(_score=values)
        for brand, rows in part.groupby(BRAND):
            row = {"split": name, "brandId": int(brand), "n_rows": len(rows)}
            if kind == "classifier":
                row.update(churn_rate=rows[TARGET].mean(), mean_prediction=rows["_score"].mean(),
                           log_loss=log_loss(rows[TARGET], rows["_score"], labels=[0, 1]),
                           ece=expected_calibration_error(rows[TARGET], rows["_score"].to_numpy()))
                if rows[TARGET].nunique() == 2:
                    row["roc_auc"] = roc_auc_score(rows[TARGET], rows["_score"])
            elif rows[EVENT].sum() > 0:
                row["c_index"] = concordance_index(rows[DURATION], -rows["_score"], rows[EVENT])
            out.append(row)
    return pd.DataFrame(out)


def evaluate(entry: dict, model, data: pd.DataFrame, split: pd.Series, features: list[str], calibrator=None,
             calibration_candidates: dict | None = None):
    """`features` are the selected features; the model may add brandId (see model_columns)."""
    metrics, tables, figures = {}, {}, {}
    if entry["interface"] == "sklearn_estimator":
        columns = model_columns(entry, features)
        delivered = {}
        for name in SPLITS:
            part = data[split == name]
            p = raw_probability(entry, model, part, features)
            metrics.update({f"{name}_{k}": v for k, v in _classification_metrics(part[TARGET], p).items()})
            delivered[name] = p
            if calibrator is not None:
                p_cal = churn_probability(model, calibrator, part, columns)
                metrics.update({f"{name}_calibrated_{k}": v
                                for k, v in _classification_metrics(part[TARGET], p_cal).items()})
                delivered[name] = p_cal
        by_brand = _metrics_by_brand(data, split, {s: delivered[s] for s in ("valid", "test")}, "classifier")
        tables["metrics_by_brand.csv"] = by_brand
        for r in by_brand.itertuples():
            for key in ("roc_auc", "log_loss", "ece"):
                if pd.notna(getattr(r, key, np.nan)):
                    metrics[f"{r.split}_brand{r.brandId}_{key}"] = getattr(r, key)
        figures["reliability_curve.png"], tables["reliability_curve.csv"] = _reliability(
            entry, data, split, model, calibrator, features, calibration_candidates)
        figures["roc_curve.png"], tables["roc_curve.csv"] = _roc(entry, data, split, model, features)
        # Overfitting check: how much better the model ranks the players it was trained on.
        metrics["overfit_gap_train_valid_roc_auc"] = metrics["train_roc_auc"] - metrics["valid_roc_auc"]
        test = data[split == "test"]
        if BENCHMARK_FEATURE in test:
            auc = roc_auc_score(test[TARGET], test[BENCHMARK_FEATURE])
            metrics[f"test_roc_auc_benchmark_{BENCHMARK_FEATURE}"] = max(auc, 1 - auc)
        if getattr(model, "best_iteration_", None):
            metrics["best_iteration"] = model.best_iteration_
            for key, value in model.best_score_["valid_0"].items():
                metrics[f"early_stopping_valid_{key}"] = value
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({"feature": columns, "importance": model.feature_importances_})
            tables["feature_importance.csv"] = importance.sort_values("importance", ascending=False)
        if calibrator is not None and hasattr(calibrator, "describe"):
            tables["calibration_by_brand.csv"] = calibrator.describe()
        return metrics, tables, figures

    hazards = {}
    for name in SPLITS:
        part = data[split == name]
        hazards[name] = np.asarray(model.predict_partial_hazard(cox_input(model, part)))
        metrics[f"{name}_c_index"] = concordance_index(part[DURATION], -hazards[name], part[EVENT])
        metrics[f"{name}_event_rate"] = float(part[EVENT].mean())
        metrics[f"{name}_max_event_day"] = int(part.loc[part[EVENT] == 1, DURATION].max())
    by_brand = _metrics_by_brand(data, split, {s: hazards[s] for s in ("valid", "test")}, "survival")
    tables["metrics_by_brand.csv"] = by_brand
    for r in by_brand.itertuples():
        if pd.notna(getattr(r, "c_index", np.nan)):
            metrics[f"{r.split}_brand{r.brandId}_c_index"] = r.c_index
    metrics["train_log_likelihood"] = float(model.log_likelihood_)
    tables["hazard_ratios.csv"] = model.summary.reset_index()
    return metrics, tables, figures


ADAPTED_INTERFACES = ("sklearn_estimator", "lifelines_survival")
MLFLOW_FLAVORS = {"lightgbm": mlflow.lightgbm, "xgboost": mlflow.xgboost}


def latest_run(model_id: str) -> pd.Series:
    """The most recent training run of `model_id` in MLFLOW_EXPERIMENT."""
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT], filter_string=f"params.model_id = '{model_id}'",
        order_by=["attributes.start_time DESC"], max_results=1,
    )
    if runs.empty:
        raise LookupError(f"no {model_id} run in {MLFLOW_EXPERIMENT}: run `make train-baseline` first")
    return runs.iloc[0]


def run_info(run_id: str) -> dict:
    """What a training run used: features, dataset, model parameters, metrics."""
    run = mlflow.get_run(run_id)
    params = run.data.params
    model_params = {}
    for key, value in params.items():
        # model__fit__* are fit settings (early stopping), not constructor parameters.
        if key.startswith("model__") and not key.startswith("model__fit__"):
            try:
                model_params[key.removeprefix("model__")] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                model_params[key.removeprefix("model__")] = value
    return {
        "run_id": run_id, "run_name": run.info.run_name, "model_id": params["model_id"],
        "features": ast.literal_eval(params["feature_columns"]), "dataset_path": params["dataset_path"],
        "params": model_params, "metrics": run.data.metrics,
        # Brands the run was trained on (None for runs from before brands were logged).
        "brands": ast.literal_eval(params["brands"]) if "brands" in params else None,
    }


def train(
    model_id: str, dataset_path: str | Path | None = None, params: dict | None = None, select: bool = True,
    segments: bool = True, tune: bool = True, n_trials: int | None = None, reference: bool = False,
    features_from_run: str | None = None,
) -> str:
    """Train `model_id` on a train_dataset file (default: the latest one). Returns the MLflow run id."""
    start = time.time()
    entry, interface, catalog_version = load_catalog_entry(model_id)
    if entry["interface"] not in ADAPTED_INTERFACES:
        raise NotImplementedError(f"interface {entry['interface']!r} has no trainer yet: {ADAPTED_INTERFACES}")

    if dataset_path is None:
        dataset_path = max(PROCESSED_DIR.glob("train_dataset_*.parquet"), key=lambda p: p.stat().st_mtime)
    dataset_path = Path(dataset_path).resolve()
    data = pd.read_parquet(dataset_path)
    needed = [DURATION, EVENT] if entry["task_type"] == "survival" else [TARGET]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"{dataset_path.name} has no {missing}: rebuild it with `make labels dataset`")

    split = split_rows(data)
    segment_model = None
    if segments:
        # Unsupervised, but still fitted on the training months only, never the test month.
        segment_model = segmentation.fit_segments(data[split != "test"])
        data = segmentation.add_segment_features(data, segment_model)
    features = [c for c in data.columns if c not in NON_FEATURES]
    feature_source = None
    if entry.get("feature_set"):
        # Same feature set as another model (Cox PH uses LightGBM's): taken from that model's run on this
        # very dataset (`features_from_run`, default its latest run), with no Stage 2 selection of its own.
        source = run_info(features_from_run or latest_run(entry["feature_set"]).run_id)
        if source["dataset_path"] != os.path.relpath(dataset_path, PROJECT_ROOT):
            raise ValueError(f"{entry['feature_set']} run {source['run_name']} was trained on "
                             f"{source['dataset_path']}, not on this dataset: train it first")
        features = [f for f in source["features"] if f != BRAND]
        missing = [f for f in features if f not in data.columns]
        if missing:
            raise ValueError(f"{source['run_name']} uses {missing}, not in this dataset")
        select, feature_source = False, source["run_name"]
    adapter = make_adapter(entry, params, data[split == "train"], data[split == "valid"])
    summary = {}
    penalizer_table = None
    if select:
        features, report, summary = selection.select_features(data, split, features, adapter, TARGET)

    balance_table, trials, tuning = None, None, {}
    steps = []  # one snapshot per optimisation step, logged as nested MLflow runs with deltas
    params_of = lambda a: {k: v for k, v in a.params.items() if not k.startswith("fit__")}
    if reference:
        # Reference model = optimisation step 1 only: the catalog defaults with its preferred class
        # weights and early stopping. No imbalance comparison, search, regularisation or pruning.
        tune = False
        if entry["interface"] == "sklearn_estimator":
            balance = entry.get("class_balance_preferred", "none")
            fixed = class_balance_options(entry, data[split == "train"]).get(balance, {})
            adapter = make_adapter(entry, {**fixed, **(params or {})}, data[split == "train"], data[split == "valid"])
        model = adapter.fit(features)
    elif entry["interface"] == "sklearn_estimator" and class_balance_options(entry, data[split == "train"]):
        balance, adapter, model, balance_table, fitted = compare_class_balance(entry, params, data, split, features)
        if "none" in fitted:
            steps.append({"step": 1, "model": fitted["none"][1], "features": list(features),
                          "params": params_of(fitted["none"][0]), "note": "catalog defaults, plain log-loss"})
        steps.append({"step": 2, "model": model, "features": list(features), "params": params_of(adapter),
                      "note": f"chosen: {balance}"})
    elif tune and entry.get("search", {}).get("param") == "penalizer":
        model, adapter, penalizer_table = search_penalizer(entry, params, data[split == "train"],
                                                           data[split == "valid"], features)
    else:
        model = adapter.fit(features)
    if tune and "search" in entry and entry["interface"] == "sklearn_estimator":
        # Hyperparameter search with the selected features and the chosen class weights, then the best
        # trial refit on train with the tree count its early stopping found (no early stopping now).
        train_rows, valid_rows = data[split == "train"], data[split == "valid"]
        base = {**(class_balance_options(entry, train_rows).get(balance, {}) if balance_table is not None else {}),
                **(params or {})}
        best, best_iteration, trials = tune_hyperparameters(entry, base, train_rows, valid_rows, features, n_trials)
        final = {**base, **best, "n_estimators": best_iteration}
        adapter = make_adapter(entry, final, train_rows)  # no `valid`: no early stopping in the refit
        model = adapter.fit(features)
        states = trials["state"].value_counts()
        tuning = {"tuning_n_trials": len(trials), "tuning_n_pruned": int(states.get("PRUNED", 0)),
                  "tuning_n_complete": int(states.get("COMPLETE", 0)),
                  "tuning_best_valid_log_loss": float(trials["value"].min()), "tuning_best_iteration": best_iteration}
        steps.append({"step": 3, "model": model, "features": list(features), "params": params_of(adapter),
                      "note": f"best of {len(trials)} Optuna trials", "extra": tuning})
    regularisation = None
    if entry["interface"] == "sklearn_estimator" and "regularisation_check" in entry and not reference:
        model, final_params, regularisation = regularisation_check(
            entry, {k: v for k, v in adapter.params.items() if not k.startswith("fit__")},
            data[split == "train"], data[split == "valid"], features, model)
        if len(regularisation) > 1 and regularisation["accepted"].iloc[1:].any():
            adapter = make_adapter(entry, final_params, data[split == "train"])  # logs the regularised parameters
        if steps:
            rounds = int(regularisation["accepted"].iloc[1:].sum())
            steps.append({"step": 4, "model": model, "features": list(features), "params": params_of(adapter),
                          "note": f"{rounds} regularisation round(s), gap {regularisation['gap'].iloc[0]:.4f} -> "
                                  f"{regularisation[regularisation['accepted']]['gap'].iloc[-1]:.4f}"})
    pruning, pruning_summary = None, {}
    if entry["interface"] == "sklearn_estimator" and "final_pruning" in entry and select and not reference:
        clean = {k: v for k, v in adapter.params.items() if not k.startswith("fit__")}
        model, pruned, pruning, pruning_summary = final_pruning(
            entry, clean, model, data[split == "train"], data[split == "valid"], features)
        if len(pruned) < len(features):
            features = pruned
            adapter = make_adapter(entry, {**clean, "n_estimators": int(model.best_iteration_ or model.n_estimators)},
                                   data[split == "train"])  # logs the parameters of the pruned model
        if steps:
            steps.append({"step": 5, "model": model, "features": list(features), "params": params_of(adapter),
                          "note": f"{pruning_summary['pruning_n_features_before']} -> "
                                  f"{pruning_summary['pruning_n_features_after']} features"})
    inference = {}
    if "inference_budget" in entry:
        # Cost of inference: scoring time per 100k players, and a tree cap if a full-population run
        # would go over the budget on the serving container. Before calibration, so it fits the final model.
        clean = {k: v for k, v in adapter.params.items() if not k.startswith("fit__")}
        model, inference = inference_budget(entry, clean, model, data, split, features)
        if inference["inference_tree_cap"]:
            adapter = make_adapter(entry, {**clean, "n_estimators": inference["inference_tree_cap"]},
                                   data[split == "train"])
    # The columns the model actually uses: Cox covariates, or the selected features + brandId.
    used_features = (list(model.params_.index) if entry["interface"] == "lifelines_survival"
                     else model_columns(entry, features))
    calibrator, calibration_table, candidates = (
        choose_calibrator(entry, model, data, split, features) if entry["interface"] == "sklearn_estimator"
        else (None, None, {}))
    metrics, tables, figures = evaluate(entry, model, data, split, features, calibrator, candidates)
    if steps:
        steps.append({"step": 6, "model": model, "features": list(features), "params": params_of(adapter),
                      "calibrator": calibrator, "note": f"chosen: {calibrator.method} (lowest test ECE)"})
        steps.append({"step": 9, "model": model, "features": list(features), "params": params_of(adapter),
                      "calibrator": calibrator, "extra": inference,
                      "note": (f"tree cap {inference['inference_tree_cap']}" if inference.get("inference_tree_cap")
                               else "within budget, no cap")})
    if calibration_table is not None:
        tables["calibration_comparison.csv"] = calibration_table
        for r in calibration_table.itertuples():
            metrics[f"calibration_{r.method}_test_ece"] = r.test_ece
    if select:
        tables["feature_selection.csv"] = report
    if penalizer_table is not None:
        tables["penalizer_search.csv"] = penalizer_table
    metrics.update(inference)
    if pruning is not None:
        tables["final_pruning.csv"] = pruning
        metrics.update(pruning_summary)
    if regularisation is not None:
        tables["regularisation_check.csv"] = regularisation
        kept = regularisation[regularisation["accepted"]].iloc[-1]
        metrics.update({"regularisation_rounds": int(regularisation["accepted"].iloc[1:].sum()),
                        "regularisation_gap_before": float(regularisation["gap"].iloc[0]),
                        "regularisation_gap_after": float(kept["gap"])})
    if trials is not None:
        tables["optuna_trials.csv"] = trials
        metrics.update(tuning)
    if balance_table is not None:
        tables["class_imbalance.csv"] = balance_table
        for r in balance_table.itertuples():
            metrics[f"imbalance_{r.class_balance}_valid_calibrated_log_loss_cv"] = r.valid_calibrated_log_loss_cv
    execution_time_s = round(time.time() - start, 1)

    timestamp_unix = int(time.time())
    brand = brand_label(data[BRAND].unique())  # the brandId, or "basel" for several brands
    run_name = f"{model_id}{'_reference' if reference else ''}_brand{brand}_{timestamp_unix}"
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("timestamp_unix", timestamp_unix)
        mlflow.set_tag("timestamp_iso", datetime.fromtimestamp(timestamp_unix, tz=timezone.utc).isoformat())
        mlflow.set_tag("git_commit", _git_commit())
        mlflow.set_tag("user", getpass.getuser())
        mlflow.set_tag("host", socket.gethostname())
        mlflow.set_tag("source_script", "src/models/train.py")
        mlflow.set_tag("brand_id", brand)
        mlflow.set_tag("role", "reference" if reference else "optimised")
        mlflow.log_param("seed", SEED)
        mlflow.log_dict(entry, "config/catalog_entry.json")  # the exact model configuration used
        # The training dataset, linked to the run under its brand-named MLflow dataset name.
        mlflow.log_input(mlflow.data.from_pandas(
            data, source=os.path.relpath(dataset_path, PROJECT_ROOT),
            name=f"train_dataset_brand{dataset_path.stem.removeprefix('train_dataset_')}",
        ), context="training")

        mlflow.log_param("model_id", model_id)
        mlflow.log_param("import_path", entry["import_path"])
        mlflow.log_param("interface", entry["interface"])
        mlflow.log_param("task_type", entry["task_type"])
        mlflow.log_param("catalog_version", catalog_version)
        mlflow.log_params({f"model__{k}": v for k, v in adapter.params.items()})
        mlflow.log_param("dataset_path", os.path.relpath(dataset_path, PROJECT_ROOT))
        mlflow.log_param("cutoff_dates", sorted(str(c) for c in data["cutoff_date"].unique()))
        mlflow.log_param("feature_columns", used_features)
        mlflow.log_param("brands", sorted(int(b) for b in data[BRAND].unique()))
        mlflow.log_param("brand_handling", (
            "categorical feature" if entry.get("brand_feature") == "categorical"
            else "Cox strata" if cox_strata(entry) else "none"))
        mlflow.log_param("test_cutoffs", sorted(str(c) for c in data.loc[split == "test", "cutoff_date"].unique()))
        mlflow.log_param("split", f"test = last {N_TEST_CUTOFFS} cutoff(s); train/valid by partyId, "
                                  f"valid {VALID_SHARE}, seed {SEED}")
        mlflow.log_param("feature_selection", select)
        if feature_source:
            mlflow.log_param("feature_source", feature_source)
        if regularisation is not None:
            mlflow.log_param("regularisation_check", f"max train-valid AUC gap {entry['regularisation_check']['max_gap']}")
        if inference:
            mlflow.log_param("inference_budget", f"{entry['inference_budget']['max_seconds_full_run']}s per full run, "
                                                 f"container slowdown {entry['inference_budget'].get('container_slowdown', 1.0)}")
        mlflow.log_param("hyperparameter_search", f"optuna TPE, {len(trials)} trials, median pruner, objective "
                                                  "valid log-loss" if trials is not None else "none")
        if balance_table is not None:
            mlflow.log_param("class_balance", balance)
            mlflow.log_param("class_balance_min_gain", entry.get("class_balance_min_gain", 0.0))
            mlflow.log_param("class_balance_preferred", entry.get("class_balance_preferred", "none"))
        mlflow.log_param("segments_k", segment_model.k if segment_model else 0)
        if segment_model:
            mlflow.log_metric("segments_silhouette", segment_model.silhouette[segment_model.k])
        mlflow.log_param("calibration", f"{calibrator.method} (lowest test ECE of {list(candidates)}), fitted on "
                                        f"valid, per brand with >= {calibrator.min_rows} rows "
                                        f"(all brands as fallback), bounded to [{CALIBRATION_BOUND}, "
                                        f"{1 - CALIBRATION_BOUND}]" if calibrator is not None else "none")
        if select:
            mlflow.log_params({f"selection__{k.lower()}": getattr(selection, k) for k in (
                "MAX_MISSING_SHARE", "NZV_FREQ_RATIO", "NZV_UNIQUE_SHARE", "MAX_ABS_CORR",
                "ADV_AUC_THRESHOLD", "ADV_IMPORTANCE_SHARE", "MAX_ADV_ROUNDS", "MAX_METRIC_LOSS")})

        mlflow.log_metric("n_features", len(used_features))
        for name in SPLITS:
            mlflow.log_metric(f"n_rows_{name}", int((split == name).sum()))
        mlflow.log_metrics({k: round(float(v), 6) for k, v in {**metrics, **summary}.items()})
        mlflow.log_metric("execution_time_s", execution_time_s)
        if steps:
            # Optimisation steps as nested runs (step 1 defaults ... step 9 inference cost), each with
            # the change in AUC, log-loss and ECE against the previous step.
            step_table = log_optimisation_steps(entry, steps, data, split, run_name, brand)
            tables["optimisation_steps.csv"] = step_table

        with tempfile.TemporaryDirectory() as tmp:
            for filename, table in tables.items():
                table.to_csv(Path(tmp) / filename, index=False)
                mlflow.log_artifact(str(Path(tmp) / filename))
            for filename, figure in figures.items():
                mlflow.log_figure(figure, filename)
            if segment_model is not None:
                joblib.dump(segment_model, Path(tmp) / Path(segmentation.ARTIFACT).name)
                mlflow.log_artifact(str(Path(tmp) / Path(segmentation.ARTIFACT).name),
                                    artifact_path=str(Path(segmentation.ARTIFACT).parent))
            if calibrator is not None:
                joblib.dump(calibrator, Path(tmp) / Path(CALIBRATOR_ARTIFACT).name)
                mlflow.log_artifact(str(Path(tmp) / Path(CALIBRATOR_ARTIFACT).name),
                                    artifact_path=str(Path(CALIBRATOR_ARTIFACT).parent))
            if entry["interface"] == "sklearn_estimator":
                # Each library's own MLflow flavor: the sklearn one rejects non-sklearn classes.
                flavor = MLFLOW_FLAVORS.get(entry["import_path"].split(".")[0], mlflow.sklearn)
                # The signature is given, not inferred from an input example: MLflow would cast the
                # categorical brandId to an integer and LightGBM would reject it.
                example = model_input(data, used_features).head(100)
                signature = infer_signature(example.assign(**{BRAND: example[BRAND].astype("int64")})
                                            if BRAND in example else example, model.predict_proba(example)[:, 1])
                # Also registered as a versioned model named after the brand (Models tab in MLflow).
                flavor.log_model(model, name="model", signature=signature,
                                 registered_model_name=f"churn_{model_id}_brand{brand}")
            else:
                joblib.dump(model, Path(tmp) / "model.joblib")
                mlflow.log_artifact(str(Path(tmp) / "model.joblib"), artifact_path="model")

    print(f"{model_id}: {len(used_features)} features, train/valid/test rows "
          f"{[int((split == s).sum()) for s in SPLITS]}, {execution_time_s}s")
    if select:
        print(report[["feature", "step", "reason", "selected"]].to_string(index=False))
        for key, value in summary.items():
            print(f"  {key:45s} {value:.4f}" if isinstance(value, float) else f"  {key:45s} {value}")
    if balance_table is not None:
        print(f"class imbalance, chosen by log-loss after calibration (preferred "
              f"{entry.get('class_balance_preferred', 'none')} unless beaten by more than "
              f"{entry.get('class_balance_min_gain', 0.0)}): {balance}")
        print(balance_table.round(4).to_string(index=False))
    if "optimisation_steps.csv" in tables:
        print("optimisation steps (each one a nested MLflow run):")
        cols = ["step", "name", "valid_roc_auc", "test_roc_auc", "test_log_loss", "test_ece",
                "delta_test_roc_auc", "delta_test_log_loss", "delta_test_ece", "note"]
        print(tables["optimisation_steps.csv"].reindex(columns=cols).round(4).to_string(index=False))
    for key in sorted(k for k in metrics if k.startswith(("valid_", "test_"))):
        print(f"  {key:45s} {metrics[key]:.4f}")
    print(f"MLflow run '{run_name}' logged in experiment '{MLFLOW_EXPERIMENT}'")
    return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one catalog model on a training dataset.")
    parser.add_argument("--model-id", required=True, help="catalog id, e.g. lightgbm_classifier or cox_ph")
    parser.add_argument("--dataset-path", help="default: the latest train_dataset file")
    parser.add_argument("--params", type=json.loads, help="JSON that overrides the catalog default_params")
    parser.add_argument("--no-selection", action="store_true", help="train on every feature, skip Stage 2")
    parser.add_argument("--no-segments", action="store_true", help="do not add the k-means segment features")
    parser.add_argument("--no-tuning", action="store_true", help="skip the Optuna search, use the catalog parameters")
    parser.add_argument("--n-trials", type=int, help="Optuna trials (default: the catalog's)")
    parser.add_argument("--features-from-run", help="run id whose features to use (catalog feature_set; default its latest run)")
    args = parser.parse_args()
    train(args.model_id, args.dataset_path, args.params, select=not args.no_selection,
          segments=not args.no_segments, tune=not args.no_tuning, n_trials=args.n_trials,
          features_from_run=args.features_from_run)
