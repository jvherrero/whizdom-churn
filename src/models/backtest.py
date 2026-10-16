"""Rolling-origin backtest (T10): train on the past, score on a date and check what happened next,
for every month of a year.

    .venv/bin/python src/models/backtest.py --as-of 2026-10-06
    .venv/bin/python src/models/backtest.py --as-of 2026-10-06 --config configs/backtest.yaml --brand-id 64
    .venv/bin/python src/models/backtest.py --report-only backtest_brand64_2026-10-06_1791391493

"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
import churn_labels  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration_level  # noqa: E402
import daily_cache  # noqa: E402
from build_features import (  # noqa: E402
    LABEL_HORIZON_DAYS,
    MIN_CUTOFF,
    brand_label,
    finalise,
    parse_brand_id,
    raw_features,
)
from build_survival_dataset import with_labels  # noqa: E402
from segments import load_segment_model  # noqa: E402
from train import (  # noqa: E402
    BRAND,
    DURATION,
    EVENT,
    HORIZONS,
    TARGET,
    churn_probability,
    cox_input,
    horizon_label,
    horizon_probability,
    load_calibrator,
    load_horizon_calibrators,
    run_info,
    train,
)

from evaluation.evaluate import run_data  # noqa: E402
from evaluation.metrics import block_bootstrap_ci, c_index, expected_calibration_error  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs/backtest.yaml"
SCENARIO_DIR = PROJECT_ROOT / "data/02_intermediate/backtest"
OUTPUT_DIR = PROJECT_ROOT / "data/03_output/backtest"
DOC_PATH = PROJECT_ROOT / "docs/backtest_v0_brand{brand}.md"
MLFLOW_EXPERIMENT = "whizdom-churn-backtest"
KEY = ["cutoff_date", "tenant_id", "player_id"]
CANDIDATE = "lgbm_optimised"
PROBABILITY_MODELS = ("lgbm_optimised", "lgbm_reference")
# Simple baselines (review item P0 4), fitted at every cutoff on the same training rows as the models:
# recency_prob turns the days since the last bet into a probability (isotonic regression), simple_rft
# is a logistic regression on recency, frequency and tenure.
SIMPLE_FEATURES = ["days_since_last_bet", "active_days_l30d", "active_days_l90d", "days_since_first_bet"]
BASELINES = ("recency_prob", "simple_rft")
COMPARATORS = ("lgbm_optimised", "lgbm_reference", "cox", "recency", "incumbent", *BASELINES)
# One colour per comparator in all the panels.
COLOURS = {"lgbm_optimised": "#1f77b4", "lgbm_reference": "#ff7f0e", "cox": "#2ca02c", "recency": "#d62728",
           "incumbent": "#9467bd", "random": "#7f7f7f", "recency_prob": "#e377c2", "simple_rft": "#8c564b"}
# Value-matched policies (review items P0 4 and P0 5): each risk score ranks the players alone
# (value-blind) and multiplied by the expected 30-day GGR (value-aware). cox30 is p_churn_30d exactly
# as scoring writes it (the Cox 30-day probability, never below p_churn_60d), so cox30 x value is
# the value_at_risk_30d that scoring wrote until 2026-10-16.
POLICY_SCORES = {"recency": "recency_prob", "simple_rft": "simple_rft", "lgbm_60d": "lgbm_optimised",
                 "cox_30d": "cox30"}


def month_cutoffs(as_of: dt.date) -> list[dt.date]:
    """Month starts from MIN_CUTOFF with a complete 60-day label by `as_of`."""
    last = as_of - dt.timedelta(days=LABEL_HORIZON_DAYS)
    return [d.date() for d in pd.date_range(MIN_CUTOFF, last, freq="MS")]


def train_cutoffs(cutoffs: list[dt.date], test: dt.date) -> list[dt.date]:
    """Cutoffs whose label was already known on the test date."""
    return [c for c in cutoffs if c + dt.timedelta(days=LABEL_HORIZON_DAYS) <= test]


def early_cutoffs(test: dt.date, past: list[dt.date]) -> list[dt.date]:
    """Recent months used by the calibration level on the test date."""
    return [c for c in calibration_level.cutoffs(test)[1] if c > past[-1]]


def level_shifts(run_id: str, matured: pd.DataFrame, early: pd.DataFrame, test: dt.date) -> dict[int, float]:
    """Calibration level on the test cutoff, the same way scoring does it."""
    info = run_info(run_id)
    predict = calibration_level.predictor(mlflow.lightgbm.load_model(f"runs:/{run_id}/model"), load_calibrator(run_id),
                                          load_segment_model(run_id), info["features"])
    return calibration_level.fit(predict, matured, early if len(early) else None, test)


def _predictions(run_id: str) -> pd.DataFrame:
    """Score of each test row: calibrated probability (LightGBM) or partial hazard (Cox)."""
    info = run_info(run_id)
    data, split = run_data(info)
    test = data[split == "test"]
    if info["model_id"] == "cox_ph":
        model = __import__("joblib").load(mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model/model.joblib"))
        score = np.asarray(model.predict_partial_hazard(cox_input(model, test)))
    else:
        model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
        score = churn_probability(model, load_calibrator(run_id), test, info["features"])
    return test[KEY + [BRAND, TARGET, DURATION, EVENT]].assign(score=score)


def cutoff_metrics(rows: pd.DataFrame, score: str, probability: bool, top_share: float) -> dict:
    """Metrics of one score on one cutoff."""
    y, s = rows[TARGET].to_numpy(), rows[score].to_numpy(dtype=float)
    top = np.argsort(-s, kind="stable")[: max(1, int(round(len(s) * top_share)))]
    out = {"auc": roc_auc_score(y, s), "c_index": c_index(rows[DURATION], s, rows[EVENT]),
           "top_decile_precision": y[top].mean(), "top_decile_recall": y[top].sum() / y.sum()}
    if probability:
        out.update(log_loss=log_loss(y, s), ece=expected_calibration_error(y, s), brier=brier_score_loss(y, s))
    return out


def fit_baselines(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> pd.DataFrame:
    """recency_prob and simple_rft on the test rows, fitted on the training rows (labels known on C)."""
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(train_rows["days_since_last_bet"], train_rows[TARGET])
    rft = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    rft.fit(train_rows[SIMPLE_FEATURES].fillna(0), train_rows[TARGET])
    return test_rows[KEY].assign(recency_prob=iso.predict(test_rows["days_since_last_bet"]),
                                 simple_rft=rft.predict_proba(test_rows[SIMPLE_FEATURES].fillna(0))[:, 1])


def cox_horizons(run_id: str) -> pd.DataFrame:
    """Calibrated Cox P(churn within h days) on the test rows, and the h-day labels."""
    info = run_info(run_id)
    data, split = run_data(info)
    test = data[split == "test"]
    model = __import__("joblib").load(mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model/model.joblib"))
    p = horizon_probability(model, load_horizon_calibrators(run_id), test)
    out = test[KEY].copy()
    for h in HORIZONS:
        out[f"cox_p{h}"] = p[h].to_numpy()
        out[f"label_{h}d"] = horizon_label(test, h).to_numpy()
    return out


def value_30d(rows: pd.DataFrame) -> pd.Series:
    """Expected 30-day GGR, as scoring computes it (expected_ggr_30d, never below 0)."""
    return (rows["ggr_eur_l90d"].clip(lower=0) / 90 * 30).fillna(0)


def _capture(rows: pd.DataFrame, rank: pd.Series, n: int, realised: pd.Series) -> dict:
    top = rank.fillna(-np.inf).to_numpy().argsort(kind="stable")[::-1][:n]
    return {"churn_captured": rows[TARGET].iloc[top].sum() / rows[TARGET].sum(),
            "ggr_at_risk_captured": realised.iloc[top].sum() / realised.sum(),
            "ggr_at_risk_per_1000": realised.iloc[top].sum() / n * 1000}


def policies(rows: pd.DataFrame, n: int) -> list[dict]:
    """Every risk score, value-blind and value-aware, on the same players and the same N contacts."""
    value = value_30d(rows)
    realised = value * rows[TARGET]
    n = min(n, len(rows))
    out = []
    for name, col in POLICY_SCORES.items():
        for weighting, rank in (("value-blind", rows[col]), ("value-aware", value * rows[col])):
            out.append({"score": name, "weighting": weighting, **_capture(rows, rank, n, realised)})
    return out


def list_overlap(rows: pd.DataFrame, n: int) -> dict:
    """How much the LightGBM 60-day and the Cox 30-day value-at-risk lists agree."""
    value = value_30d(rows)
    a, b = value * rows["lgbm_optimised"], value * rows["cox30"]
    n = min(n, len(rows))
    top_a = set(np.argsort(-a.to_numpy(), kind="stable")[:n])
    top_b = set(np.argsort(-b.to_numpy(), kind="stable")[:n])
    return {"top_n_overlap": len(top_a & top_b) / n, "spearman_value_at_risk": spearmanr(a, b).statistic,
            "spearman_probability": spearmanr(rows["lgbm_optimised"], rows["cox30"]).statistic}


NOWCAST_DAYS = (14, 21, 30, 45)   # days observed when the early churn estimate is replayed


def nowcast_checks(features: pd.DataFrame, as_of: dt.date) -> list[dict]:
    """Review item P0 3: replay the early churn estimate of the calibration level for every monthly
    cohort, k days after its cutoff, with data up to that day only: the share of the cohort with no
    bet yet, times the share of the last cohort complete on that day that, silent after k days,
    stayed silent for 60 (calibration_level.early_rate). Each estimate is set against the rate known
    today, and against the rate of that last complete cohort (a level without the early signal)."""
    out = []
    cutoffs = sorted(pd.to_datetime(features["cutoff_date"]).dt.date.unique())
    for brand, rows in features.groupby(BRAND):
        by_cutoff = {pd.Timestamp(c).date(): g for c, g in rows.groupby("cutoff_date")}
        for c in cutoffs:
            if c not in by_cutoff or c + dt.timedelta(days=LABEL_HORIZON_DAYS) > as_of:
                continue
            realised, n_unknown = with_labels(by_cutoff[c], as_of)
            if n_unknown or realised.empty:
                continue
            for k in NOWCAST_DAYS:
                day = c + dt.timedelta(days=k)
                complete = [r for r in by_cutoff if r + dt.timedelta(days=LABEL_HORIZON_DAYS) <= day]
                if not complete:
                    continue
                ref, _ = with_labels(by_cutoff[max(complete)], day)
                out.append({"cutoff": str(c), "brand": int(brand), "days_observed": k,
                            "estimated_rate": calibration_level.early_rate(by_cutoff[c], ref, day, int(brand)),
                            "realised_rate": float(realised[TARGET].mean()),
                            "last_complete_rate": float(ref[TARGET].mean())})
    return out


def cox_horizon_metrics(rows: pd.DataFrame) -> list[dict]:
    """Cox PH at each horizon (review item P1 7): AUC, Brier and ECE of the calibrated probability
    against the h-day label, on the players whose h-day label is known."""
    out = []
    for h in HORIZONS:
        known = rows[rows[f"label_{h}d"].notna()]
        y, p = known[f"label_{h}d"].to_numpy(dtype=int), known[f"cox_p{h}"].to_numpy(dtype=float)
        if len(known) == 0 or y.min() == y.max():
            continue
        out.append({"horizon_days": h, "n_players": len(known), "churn_rate": y.mean(), "auc": roc_auc_score(y, p),
                    "brier": brier_score_loss(y, p), "ece": expected_calibration_error(y, p)})
    return out


def top_decile_ratio(precision: pd.DataFrame) -> float:
    """Mean top-decile precision of the model divided by the recency rule's."""
    return float(precision["model"].mean() / precision["recency"].mean())


def replay(rows: pd.DataFrame, n: int) -> list[dict]:
    """Campaign replay on one cutoff: take the top `n` players of each ranking and measure the GGR
    at risk of those who really churned. The model ranks by value at risk."""
    value = (rows["ggr_eur_l90d"].clip(lower=0) / 90 * 30).fillna(0)
    realised = value * rows[TARGET]
    rankings = {"lgbm_optimised": value * rows["lgbm_optimised"], "recency": rows["recency"],
                "incumbent": rows["incumbent"]}
    n = min(n, len(rows))
    out = []
    for name, rank in rankings.items():
        top = rank.fillna(-np.inf).to_numpy().argsort(kind="stable")[::-1][:n]
        out.append({"ranking": name, "churn_captured": rows[TARGET].iloc[top].sum() / rows[TARGET].sum(),
                    "ggr_at_risk_captured": realised.iloc[top].sum() / realised.sum(),
                    "ggr_at_risk_per_1000": realised.iloc[top].sum() / n * 1000})
    out.append({"ranking": "random", "churn_captured": n / len(rows), "ggr_at_risk_captured": n / len(rows),
                "ggr_at_risk_per_1000": realised.mean() * 1000})  # expected value of a random pick
    return out


def run(config: str | Path = CONFIG_PATH, as_of: str | None = None, brand_id: int | str = 64,
        top_up: bool = False) -> Path:
    start = time.time()
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    as_of_date = dt.date.fromisoformat(as_of)
    if top_up:
        daily_cache.top_up(as_of_date, brand_id)
    cutoffs = month_cutoffs(as_of_date)
    tests = cutoffs[-cfg["n_test_cutoffs"]:]
    n_valid = cfg.get("valid_cutoffs", 1)
    skipped = [c for c in tests if len(train_cutoffs(cutoffs, c)) < max(cfg["min_train_cutoffs"], n_valid + 1)]
    tests = [c for c in tests if c not in skipped]
    needed = sorted({c for t in tests for c in train_cutoffs(cutoffs, t) + early_cutoffs(t, train_cutoffs(cutoffs, t))}
                    | set(tests))
    label = brand_label([brand_id]) if brand_id != "basel" else "basel"
    backtest_id = f"backtest_brand{label}{churn_labels.VARIANT}_{as_of}_{int(start)}"
    print(f"{backtest_id}: {len(tests)} test cutoffs {[str(c) for c in tests]}, skipped {[str(c) for c in skipped]}")

    print("[1] features of every cutoff (raw once, then winsorised + sign-log)")
    raw = pd.concat([raw_features(c, brand_id) for c in needed], ignore_index=True)
    features = finalise(raw)
    extra = raw[KEY + ["churn_score", "ggr_eur_l90d"]]
    m = cfg["models"]
    params = {"n_estimators": m["max_trees"]}

    scenario_dir = SCENARIO_DIR / backtest_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, replay_rows, policy_rows, overlap_rows, horizon_rows = [], [], [], [], []
    for i, test in enumerate(tests, 1):
        past = train_cutoffs(cutoffs, test)
        print(f"\n== {i}/{len(tests)} test {test}: train {past[0]}..{past[-n_valid - 1]}, "
              f"validation {[str(c) for c in past[-n_valid:]]}")
        # Train labels as known on the test date; test labels as known on AS_OF.
        train_part, _ = with_labels(features[features["cutoff_date"].isin(past)], test)
        test_part, _ = with_labels(features[features["cutoff_date"] == test], as_of_date)
        path = scenario_dir / f"test{test}.parquet"
        pd.concat([train_part, test_part], ignore_index=True).to_parquet(path, index=False)
        tags = {"backtest_id": backtest_id, "test_cutoff": str(test)}
        common = {"n_test_cutoffs": 1, "n_valid_cutoffs": n_valid, "tags": tags, "seed": m["seed"]}
        runs = {"lgbm_optimised": train("lightgbm_classifier", path, params, n_trials=m["n_trials"], **common),
                "lgbm_reference": train("lightgbm_classifier", path, params, reference=True, **common)}
        runs["cox"] = train("cox_ph", path, features_from_run=runs["lgbm_optimised"], **common)

        rows = _predictions(runs["lgbm_optimised"]).rename(columns={"score": "lgbm_optimised"})
        for name in ("lgbm_reference", "cox"):
            rows = rows.merge(_predictions(runs[name])[KEY + ["score"]].rename(columns={"score": name}), on=KEY)
        # Calibration level on the test date, as in scoring.
        matured = train_part[train_part["cutoff_date"].isin(past[-calibration_level.N_MATURED:])]
        early = features[features["cutoff_date"].isin(early_cutoffs(test, past))]
        shifts = {}
        for name in PROBABILITY_MODELS:
            shifts[name] = level_shifts(runs[name], matured, early, test)
            rows[f"{name}_static"] = rows[name]
            rows[name] = calibration_level.apply(rows[name], rows[BRAND], shifts[name])
        rows = rows.merge(test_part[KEY + ["days_since_last_bet"]].rename(columns={"days_since_last_bet": "recency"}),
                          on=KEY).merge(extra.rename(columns={"churn_score": "incumbent"}), on=KEY, how="left")
        rows = rows.merge(fit_baselines(train_part, test_part), on=KEY).merge(cox_horizons(runs["cox"]), on=KEY)
        rows["cox30"] = np.maximum(rows["cox_p30"], rows["lgbm_optimised"])   # p_churn_30d as scoring writes it
        for name in COMPARATORS:
            known = rows[rows[name].notna()]
            level = ({"level_shift": float(np.mean(list(shifts[name].values()))),
                      "ece_static": expected_calibration_error(known[TARGET], known[f"{name}_static"])}
                     if name in PROBABILITY_MODELS else {})
            metric_rows.append({"cutoff": str(test), "comparator": name, "n_players": len(known),
                                "churn_rate": known[TARGET].mean(), "n_train_cutoffs": len(past),
                                **cutoff_metrics(known, name, name in PROBABILITY_MODELS, cfg["top_share"]), **level})
        n_contacts = cfg["targeting_replay"]["contacts"]
        replay_rows += [{"cutoff": str(test), **r} for r in replay(rows, n_contacts)]
        policy_rows += [{"cutoff": str(test), **r} for r in policies(rows, n_contacts)]
        overlap_rows.append({"cutoff": str(test), **list_overlap(rows, n_contacts)})
        horizon_rows += [{"cutoff": str(test), **r} for r in cox_horizon_metrics(rows)]

        print(pd.DataFrame(metric_rows[-len(COMPARATORS):]).round(4).to_string(index=False))

    metrics, replays = pd.DataFrame(metric_rows), pd.DataFrame(replay_rows)
    extras = {"policies": pd.DataFrame(policy_rows), "overlap": pd.DataFrame(overlap_rows),
              "cox_horizons": pd.DataFrame(horizon_rows),
              "early_rates": pd.DataFrame(nowcast_checks(features, as_of_date))}
    doc = _report(metrics, replays, cfg, backtest_id, label, as_of, skipped, time.time() - start, extras=extras)
    print(f"\nsaved {os.path.relpath(doc, PROJECT_ROOT)} | {(time.time() - start) / 60:.1f} min")
    return doc


def _summary(table: pd.DataFrame, by: str, columns: list[str], n_boot: int, block: int = 1) -> pd.DataFrame:
    """Mean, median, min, max and block bootstrap CI of the mean across cutoffs (in cutoff order)."""
    out = []
    for name, rows in table.groupby(by, sort=False):
        rows = rows.sort_values("cutoff")
        for col in columns:
            v = rows[col].dropna().to_numpy()
            if not len(v):
                continue
            low, high = (block_bootstrap_ci(len(v), lambda i: v[i].mean(), block, n_boot) if len(v) > 1
                         else (np.nan, np.nan))
            out.append({by: name, "metric": col, "mean": v.mean(), "ci_low": low, "ci_high": high,
                        "median": float(np.median(v)), "min": v.min(), "max": v.max()})
    return pd.DataFrame(out)


def months_vs_recency(metrics: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Month by month against the recency rule (review item P1 11): the median and the worst month of
    the AUC and of the top-decile ratio, and the months each comparator beats recency."""
    wide = metrics.pivot_table(index="cutoff", columns="comparator", values=["auc", "top_decile_precision"])
    rec_auc, rec_prec = wide[("auc", "recency")], wide[("top_decile_precision", "recency")]
    out = []
    for name in [c for c in COMPARATORS if c != "recency" and ("auc", c) in wide]:
        auc = wide[("auc", name)].dropna()
        ratio = (wide[("top_decile_precision", name)] / rec_prec).dropna()
        out.append({"comparator": name, "months": len(auc), "auc_median": auc.median(), "auc_worst": auc.min(),
                    "auc_worst_month": auc.idxmin(), "months_auc_above_recency": int((auc > rec_auc[auc.index]).sum()),
                    "ratio_median": ratio.median(), "ratio_worst": ratio.min(), "ratio_worst_month": ratio.idxmin(),
                    "months_ratio_at_threshold": int((ratio >= threshold).sum())})
    return pd.DataFrame(out)


def figure(metrics: pd.DataFrame, replays: pd.DataFrame, cfg: dict, label: str, as_of: str, path: Path) -> Path:
    """Figure: AUC and top-decile precision per cutoff, and the campaign replay."""
    rule = cfg["pass_conditions"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, col, title in ((axes[0], "auc", "AUC by cutoff"), (axes[1], "top_decile_precision", "Top-decile precision")):
        for name, rows in metrics.groupby("comparator", sort=False):
            ax.plot(rows["cutoff"], rows[col], marker="o", label=name, color=COLOURS[name],
                    linewidth=2.5 if name == CANDIDATE else 1.2)
        ax.set_title(title); ax.tick_params(axis="x", rotation=45); ax.legend(fontsize=8)
    axes[0].axhline(rule["min_auc"], color="grey", linestyle="--", linewidth=1)
    for name, rows in replays.groupby("ranking", sort=False):
        axes[2].plot(rows["cutoff"], rows["ggr_at_risk_per_1000"], marker="o", label=name, color=COLOURS[name],
                     linewidth=2.5 if name == CANDIDATE else 1.2, linestyle=":" if name == "random" else "-")
    axes[2].set_title(f"GGR at risk caught per 1,000 contacts (top {cfg['targeting_replay']['contacts']:,})")
    axes[2].tick_params(axis="x", rotation=45); axes[2].legend(fontsize=8)
    fig.suptitle(f"Rolling-origin backtest, brandId {label}, as of {as_of}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def review_sections(extras: dict, n: int) -> str:
    """Report sections for the Phase 1 review items P0 4, P0 5 and P1 7 (empty for older backtests)."""
    if not extras or extras.get("policies") is None or extras["policies"].empty:
        return ""
    pol = extras["policies"].groupby(["score", "weighting"], sort=False)[["churn_captured", "ggr_at_risk_captured"]].mean()
    won = extras["policies"].pivot_table(index="cutoff", columns=["score", "weighting"], values="ggr_at_risk_captured")
    lines = ["## Value-Matched Policies (review items P0 4 and P0 5)", "",
             f"Every risk score ranks the same players with the same {n:,} contacts, alone (value-blind) and multiplied by "
             "the expected 30-day GGR (value-aware). `recency` is the days since the last bet turned into a probability, "
             "`simple_rft` a logistic regression on recency, frequency and tenure (both fitted on the training months of "
             "each cutoff), `lgbm_60d` the candidate, and `cox_30d` the `p_churn_30d` that scoring writes, so "
             "`cox_30d` value-aware is the `value_at_risk_30d` that scoring wrote until 2026-10-16 (now `value_at_risk` uses `lgbm_60d`). Mean over the cutoffs.", "",
             "| Score | Churners caught, value-blind | GGR at risk caught, value-blind | Churners caught, value-aware | "
             "GGR at risk caught, value-aware | Months value-aware beats `simple_rft` value-aware |",
             "|---|---|---|---|---|---|"]
    for score in dict.fromkeys(pol.index.get_level_values(0)):
        b, v = pol.loc[(score, "value-blind")], pol.loc[(score, "value-aware")]
        beats = int((won[(score, "value-aware")] > won[("simple_rft", "value-aware")]).sum()) if score != "simple_rft" else ""
        lines.append(f"| `{score}` | {100 * b['churn_captured']:.1f}% | {100 * b['ggr_at_risk_captured']:.1f}% | "
                     f"{100 * v['churn_captured']:.1f}% | {100 * v['ggr_at_risk_captured']:.1f}% | "
                     f"{f'{beats} of {len(won)}' if beats != '' else ''} |")
    g = lambda score, w: pol.loc[(score, w), "ggr_at_risk_captured"]
    lines += ["", "Split of the GGR at risk caught by the final policy (`lgbm_60d` value-aware) against value-blind recency:", "",
              f"- **Predictive uplift** (`lgbm_60d` against `simple_rft`, both value-aware): {100 * (g('lgbm_60d', 'value-aware') - g('simple_rft', 'value-aware')):+.1f} points.",
              f"- **Value-weighting uplift** (`lgbm_60d` value-aware against value-blind): {100 * (g('lgbm_60d', 'value-aware') - g('lgbm_60d', 'value-blind')):+.1f} points.",
              f"- **Combined** (`lgbm_60d` value-aware against `recency` value-blind): {100 * (g('lgbm_60d', 'value-aware') - g('recency', 'value-blind')):+.1f} points.",
              ""]
    ov = extras["overlap"]
    if not ov.empty:
        lines += ["### Cox 30-day against LightGBM 60-day value at risk", "",
                  f"Share of the top {n:,} players in both lists: mean {100 * ov['top_n_overlap'].mean():.0f}% "
                  f"(range {100 * ov['top_n_overlap'].min():.0f}% to {100 * ov['top_n_overlap'].max():.0f}%). Spearman correlation "
                  f"of the two values at risk: {ov['spearman_value_at_risk'].mean():.3f}; of the two probabilities: "
                  f"{ov['spearman_probability'].mean():.3f}.", ""]
    er = extras.get("early_rates")
    if er is not None and not er.empty:
        er = er.assign(err_estimate=(er["estimated_rate"] - er["realised_rate"]).abs(),
                       err_last=(er["last_complete_rate"] - er["realised_rate"]).abs())
        lines += ["## Calibration Level: the Early Churn Estimate (review item P0 3)", "",
                  "On a scoring date the calibration level estimates the 60-day churn rate of the recent cohorts, whose "
                  "label is not complete yet, from data up to that date only: the share of the cohort with no bet so far, "
                  "times the share of the last complete cohort that, silent after the same number of days, stayed silent "
                  "for 60 days (`calibration_level.early_rate`). Here the estimate is replayed for every monthly cohort "
                  f"{', '.join(map(str, NOWCAST_DAYS))} days after its cutoff, with data up to that day only, and set "
                  "against the rate known today and against simply using the rate of the last complete cohort. Mean "
                  "absolute error, in points of churn rate.", "",
                  "| Days observed | Cohorts | Mean realised rate | Error of the early estimate | Error of the last complete month |",
                  "|---|---|---|---|---|"]
        for band, g in er.groupby("days_observed"):
            lines.append(f"| {band} days | {len(g)} | {100 * g['realised_rate'].mean():.1f}% | {100 * g['err_estimate'].mean():.2f} | "
                         f"{100 * g['err_last'].mean():.2f} |")
        lines += ["", f"The early estimate is closer to the realised rate in {int((er['err_estimate'] < er['err_last']).sum())} "
                  f"of {len(er)} cohorts.", ""]
    hz = extras["cox_horizons"]
    if not hz.empty:
        lines += ["## Cox PH per Horizon (review item P1 7)", "",
                  "The calibrated Cox probability of churn starting within h days, against the h-day label, on the players "
                  "whose label is known. Mean over the cutoffs (range).", "",
                  "| Horizon | Players per cutoff | Churn rate | AUC | Brier | ECE |", "|---|---|---|---|---|---|"]
        for h, r in hz.groupby("horizon_days"):
            lines.append(f"| {h} days | {r['n_players'].mean():,.0f} | {r['churn_rate'].mean():.3f} | "
                         f"{r['auc'].mean():.3f} ({r['auc'].min():.3f} to {r['auc'].max():.3f}) | {r['brier'].mean():.3f} | "
                         f"{r['ece'].mean():.3f} ({r['ece'].min():.3f} to {r['ece'].max():.3f}) |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _report(metrics, replays, cfg, backtest_id, label, as_of, skipped, seconds, run_id: str | None = None,
            extras: dict | None = None) -> Path:
    """Write the report and log it to MLflow. With `run_id`, update that run instead."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = Path(str(DOC_PATH).format(brand=f"{label}{churn_labels.VARIANT}"))
    paths = {"metrics": OUTPUT_DIR / f"{backtest_id}_metrics.csv", "replay": OUTPUT_DIR / f"{backtest_id}_replay.csv",
             "figure": OUTPUT_DIR / f"{backtest_id}.png"}
    metrics.to_csv(paths["metrics"], index=False)
    replays.to_csv(paths["replay"], index=False)
    for name, frame in (extras or {}).items():
        paths[name] = OUTPUT_DIR / f"{backtest_id}_{name}.csv"
        frame.to_csv(paths[name], index=False)
    n_boot = cfg["bootstrap_resamples"]
    metric_cols = ["auc", "c_index", "top_decile_precision", "top_decile_recall", "log_loss", "ece", "brier"]
    block = cfg.get("bootstrap_block_months", 1)
    summary = _summary(metrics, "comparator", metric_cols, n_boot, block)
    replay_summary = _summary(replays, "ranking", ["churn_captured", "ggr_at_risk_captured", "ggr_at_risk_per_1000"],
                              n_boot, block)

    # Pass conditions.
    rule = cfg["pass_conditions"]
    cand, rec = metrics[metrics["comparator"] == CANDIDATE], metrics[metrics["comparator"] == "recency"]
    precision = cand.set_index("cutoff")["top_decile_precision"].to_frame("model").join(
        rec.set_index("cutoff")["top_decile_precision"].rename("recency")).sort_index()
    ratio = top_decile_ratio(precision)
    ratio_ci = block_bootstrap_ci(len(precision), lambda i: top_decile_ratio(precision.iloc[i]), block, n_boot)
    monthly = months_vs_recency(metrics, rule["top_decile_precision_vs_recency"])
    checks = [
        (f"mean top-decile precision >= {rule['top_decile_precision_vs_recency']:.2f}x the recency rule",
         f"{precision['model'].mean():.3f} vs {precision['recency'].mean():.3f}: {ratio:.2f}x [{ratio_ci[0]:.2f}, {ratio_ci[1]:.2f}]",
         ratio >= rule["top_decile_precision_vs_recency"]),
        (f"AUC never below {rule['min_auc']}", f"min {cand['auc'].min():.3f} ({cand.loc[cand['auc'].idxmin(), 'cutoff']})",
         cand["auc"].min() >= rule["min_auc"]),
        (f"ECE never above {rule['max_ece']}", f"max {cand['ece'].max():.3f} ({cand.loc[cand['ece'].idxmax(), 'cutoff']})",
         cand["ece"].max() <= rule["max_ece"]),
    ]
    passed = all(ok for *_, ok in checks)

    figure(metrics, replays, cfg, label, as_of, paths["figure"])

    def table(df, cols, digits=3):
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in df[cols].itertuples(index=False):
            lines.append("| " + " | ".join(f"{v:,.{digits}f}" if isinstance(v, float) else str(v) for v in r) + " |")
        return "\n".join(lines)

    def fmt_summary(df, key):
        df = df.assign(value=df.apply(lambda r: f"{r['mean']:,.3f} [{r['ci_low']:,.3f}, {r['ci_high']:,.3f}] "
                                                f"(median {r['median']:,.3f}, range {r['min']:,.3f} to {r['max']:,.3f})",
                                      axis=1))
        wide = df.pivot(index=key, columns="metric", values="value").reset_index().fillna("n/a")
        return table(wide, [key] + [c for c in df["metric"].unique()])

    per_cutoff = metrics.pivot(index="cutoff", columns="comparator", values="auc").reset_index()
    level = [c for c in ("ece_static", "level_shift") if c in metrics]  # missing in older backtests
    per_cutoff = per_cutoff.merge(metrics[metrics["comparator"] == CANDIDATE][["cutoff", "n_players", "churn_rate",
                                                                               "n_train_cutoffs", "ece", *level]], on="cutoff")
    per_cutoff = per_cutoff.rename(columns={"ece": f"ECE {CANDIDATE}", "ece_static": "ECE without level", "level_shift": "level shift"})
    n = cfg["targeting_replay"]["contacts"]
    text = f"""# Backtest v0: rolling-origin (brandId {label})

Does the model predict the way it will be used: trained on the past, scored on a date, judged on what the players did next, repeated over a year. Backtest id `{backtest_id}`, data as of {as_of}, {metrics['cutoff'].nunique()} test cutoffs, {seconds / 60:.0f} min. Config: `configs/backtest.yaml`.

**Verdict: {"PASS" if passed else "FAIL"}** (a model that fails is not promoted, whatever its single-split scores).

| Pass condition (delivery plan) | Candidate (`{CANDIDATE}`) | Result |
|---|---|---|
""" + "\n".join(f"| {c} | {v} | {'pass' if ok else 'FAIL'} |" for c, v, ok in checks) + f"""

![Backtest]({os.path.relpath(paths['figure'], doc.parent)})

## How It Runs

- **Test cutoffs**: the first day of each month, {', '.join(sorted(metrics['cutoff'].unique()))}.{f" Skipped (fewer than {cfg['min_train_cutoffs']} earlier months with a known label): {', '.join(map(str, skipped))}." if skipped else ""}
- **Training at each cutoff C**: every earlier monthly cutoff whose 60-day label was already known on C (cutoff + 60 days <= C), labels computed from data up to C only; the latest {cfg.get('valid_cutoffs', 1)} of them are the validation months (early stopping, tuning, calibration). So the training data grows from {metrics['n_train_cutoffs'].min()} to {metrics['n_train_cutoffs'].max()} months, and the models never see a label C did not know.
- **Scored**: every player with a bet in the 30 days up to C, judged on the 60 days after it (data up to {as_of}).
- **Calibration level on C**, as scoring sets it (`src/models/calibration_level.py`): one log-odds shift per brand, the median of the shifts that set the mean probability right on the last {calibration_level.N_MATURED} months with a known label and on the later months' early signal (who has bet again by C). It changes no ranking, only the ECE, log-loss and Brier; the ECE without it is in the per-cutoff table.
- **Models**: trained exactly as in normal training, with {cfg['models']['n_trials']} Optuna trials (not 100) and at most {cfg['models']['max_trees']} trees, as the plan allows for the backtest. Each training run is in MLflow with the tag `backtest_id`.
- **Comparators**: `lgbm_reference` (optimisation step 1), `cox` (partial hazard), `recency` (days since the last bet alone) and `incumbent` (the platform's `churn_score` in `gld_player_signals_daily`, a weighted heuristic: 0.5 recency + 0.3 frequency + 0.2 value). The plan's 7/14/30-day labels are replaced by the 60-day churn of `docs/churn_definition_v0.md`.

## Results Across Cutoffs

Mean, 95% CI of the mean, median and range. The CI is a circular block bootstrap: each of the {n_boot} resamples joins blocks of {block} consecutive cutoffs, because neighbouring months move together (season, players who repeat). Log-loss, ECE and Brier only for the calibrated probabilities.

{fmt_summary(summary, "comparator")}

## Month by Month against Recency (review item P1 11)

The cutoffs are not independent, so the interval alone can look too sure. This table does not depend on any resampling: the median and the worst month of each comparator, and the months where it beats the recency rule (AUC above recency; top-decile ratio at or above {rule['top_decile_precision_vs_recency']:.2f}x).

{table(monthly, ["comparator", "months", "auc_median", "auc_worst", "auc_worst_month", "months_auc_above_recency", "ratio_median", "ratio_worst", "ratio_worst_month", "months_ratio_at_threshold"])}

## AUC per Cutoff

{table(per_cutoff, ["cutoff", "n_players", "churn_rate", "n_train_cutoffs"] + [c for c in COMPARATORS if c in per_cutoff] + [c for c in (f"ECE {CANDIDATE}", "ECE without level", "level shift") if c in per_cutoff])}

## Targeting Replay (would it have paid?)

A campaign that contacts the top **{n:,}** players at every cutoff (N to agree with the CRM team). The candidate ranks by **value at risk** = trailing-90-day mean daily GGR x 30 x its churn probability; `recency` and `incumbent` rank by their own score; `random` is the expected value of a random pick. **Realised GGR at risk** = the 30-day value of the players who did churn. No treatment was applied, so this measures targeting quality, not uplift.

{fmt_summary(replay_summary, "ranking")}

{review_sections(extras, n)}
## How to Read It

- **Top-decile condition at {rule['top_decile_precision_vs_recency']:.2f}x, not the plan's 1.5x** (`docs/decision_top_decile_condition.md`): precision cannot exceed 1 and the recency rule already reaches {precision['recency'].mean():.3f}, so the most any model could reach is {1 / precision['recency'].mean():.2f}x, and this ceiling falls in the months with more churn. {rule['top_decile_precision_vs_recency']:.2f}x is the strictest round value the model meets with 95% confidence in brands 14, 64 and 73, and at today's precisions it is the plan's own demand (1.5 times the churners found per wasted contact). Interval: 95%, circular block bootstrap of {block} consecutive cutoffs, {n_boot} resamples.
- **Calibration lags the churn rate**: the calibration month is the latest one whose 60-day label was known on C, 2 to 3 months before it. When the monthly churn rate moves between them (around 50% from November to January, 30 to 40% the rest of the year), every probability is off by about the same amount: the ECE rises while the AUC does not move. The level shift on C removes most of it. What is left comes from the 60 days after C (a holiday season, a sports calendar), which no data before C shows: in the analysis of 2026-10-09 even the exact level of the month before C would not have kept every month of brand 14 under 0.08 (`eda/07_calibration_seasonality.ipynb`).
- **The replay trades churners for value**: the candidate ranks by value at risk, so it contacts fewer players who churn than recency does, but the ones it contacts carry most of the GGR at risk. Ranking by churn probability alone is the top-decile columns above.
- **The cutoffs are not independent**: the players repeat and the training months overlap. The CIs resample blocks of consecutive months, which keeps the dependence between neighbouring months, but not players, so they can still be a little optimistic; the month-by-month counts do not depend on any resampling.
- **The winsorisation caps** come from the main pipeline's configuration (they may have seen later months' features, never labels).
- **Early cutoffs train on few months**, and on no deposit data before March 2026 (the deposit features are empty then). Training cutoffs before 2025-09 have under 150 days of history, so `days_since_first_bet` (capped at 150) is lower there for the same players.

Generated by `src/models/backtest.py` (`make backtest`). Tables: `{os.path.relpath(paths['metrics'], PROJECT_ROOT)}`, `{os.path.relpath(paths['replay'], PROJECT_ROOT)}`.
"""
    doc.write_text(text, encoding="utf-8")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    if run_id is not None:
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("verdict", "pass" if passed else "fail")
            mlflow.log_metric("passed", int(passed))
            for p in (paths["figure"], doc, CONFIG_PATH):
                mlflow.log_artifact(str(p))
        return doc
    with mlflow.start_run(run_name=backtest_id):
        mlflow.set_tags({"brand_id": label, "backtest_id": backtest_id, "verdict": "pass" if passed else "fail",
                          "churn_days": churn_labels.HORIZON_DAYS})
        mlflow.log_params({"as_of": as_of, "test_cutoffs": sorted(metrics["cutoff"].unique()), "skipped": [str(s) for s in skipped],
                           **{f"config__{k}": v for k, v in cfg.items()}})
        for step, (cutoff, rows) in enumerate(metrics.groupby("cutoff")):
            for r in rows.itertuples(index=False):  # one step per cutoff
                for col in metric_cols:
                    if pd.notna(getattr(r, col, np.nan)):
                        mlflow.log_metric(f"{r.comparator}_{col}", float(getattr(r, col)), step=step)
        for r in summary.itertuples(index=False):
            mlflow.log_metric(f"mean_{r.comparator}_{r.metric}", float(r.mean))
        for r in replay_summary.itertuples(index=False):
            mlflow.log_metric(f"replay_mean_{r.ranking}_{r.metric}", float(r.mean))
        mlflow.log_metric("passed", int(passed))
        mlflow.log_metric("execution_time_s", seconds)
        for p in (*paths.values(), doc, CONFIG_PATH):
            mlflow.log_artifact(str(p))
    return doc


def rewrite_report(backtest_id: str, config: str | Path = CONFIG_PATH) -> Path:
    """Rewrite the report of a finished backtest with the current pass conditions, without training."""
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT], filter_string=f"tags.backtest_id = '{backtest_id}'")
    if runs.empty:
        raise LookupError(f"no backtest run {backtest_id!r} in {MLFLOW_EXPERIMENT}")
    r = runs.iloc[0]
    metrics = pd.read_csv(OUTPUT_DIR / f"{backtest_id}_metrics.csv", dtype={"cutoff": str})
    replays = pd.read_csv(OUTPUT_DIR / f"{backtest_id}_replay.csv", dtype={"cutoff": str})
    extras = {name: pd.read_csv(OUTPUT_DIR / f"{backtest_id}_{name}.csv", dtype={"cutoff": str})
              for name in ("policies", "overlap", "cox_horizons", "early_rates")
              if (OUTPUT_DIR / f"{backtest_id}_{name}.csv").exists()}
    doc = _report(metrics, replays, cfg, backtest_id, r["tags.brand_id"], r["params.as_of"],
                  ast.literal_eval(r["params.skipped"]), r["metrics.execution_time_s"], run_id=r["run_id"],
                  extras=extras or None)
    print(f"rewrote {os.path.relpath(doc, PROJECT_ROOT)}")
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling-origin backtest (T10).")
    parser.add_argument("--report-only", metavar="BACKTEST_ID",
                        help="rewrite the report of a finished backtest with the current pass conditions (no training)")
    parser.add_argument("--as-of", help="the backtest's 'today' (YYYY-MM-DD)")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--brand-id", type=parse_brand_id, default=64)
    parser.add_argument("--top-up", action="store_true", help="top up the daily caches first (needs AWS)")
    args = parser.parse_args()
    if args.report_only:
        rewrite_report(args.report_only, args.config)
    elif not args.as_of:
        parser.error("--as-of is required (or --report-only BACKTEST_ID)")
    else:
        run(args.config, args.as_of, args.brand_id, args.top_up)
