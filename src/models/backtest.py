"""Rolling-origin backtest (T10): does the model predict, the way it will be used? Trained on the past,
scored on a date, judged on what the players did next, repeated over a year.

    .venv/bin/python src/models/backtest.py --as-of 2026-10-06                       # configs/backtest.yaml
    .venv/bin/python src/models/backtest.py --as-of 2026-10-06 --config configs/backtest.yaml --brand-id 64
    .venv/bin/python src/models/backtest.py --report-only backtest_brand64_2026-10-06_1791391493   # rewrite a report

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
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration_level  # noqa: E402
import daily_cache  # noqa: E402
from build_features import (  # noqa: E402
    LABEL_HORIZON_DAYS, MIN_CUTOFF, brand_label, finalise, parse_brand_id, raw_features,
)
from build_survival_dataset import with_labels  # noqa: E402
from evaluation.evaluate import run_data  # noqa: E402
from evaluation.metrics import bootstrap_ci, c_index, expected_calibration_error  # noqa: E402
from segments import load_segment_model  # noqa: E402
from train import (  # noqa: E402
    BRAND, DURATION, EVENT, TARGET, churn_probability, cox_input, load_calibrator, run_info, train,
)

CONFIG_PATH = PROJECT_ROOT / "configs/backtest.yaml"
SCENARIO_DIR = PROJECT_ROOT / "data/02_intermediate/backtest"
OUTPUT_DIR = PROJECT_ROOT / "data/03_output/backtest"
DOC_PATH = PROJECT_ROOT / "docs/backtest_v0_brand{brand}.md"
MLFLOW_EXPERIMENT = "whizdom-churn-backtest"
KEY = ["cutoff_date", "tenant_id", "player_id"]
CANDIDATE = "lgbm_optimised"
PROBABILITY_MODELS = ("lgbm_optimised", "lgbm_reference")
COMPARATORS = ("lgbm_optimised", "lgbm_reference", "cox", "recency", "incumbent")
# One colour per comparator in every panel of the figure (random only appears in the replay).
COLOURS = {"lgbm_optimised": "#1f77b4", "lgbm_reference": "#ff7f0e", "cox": "#2ca02c", "recency": "#d62728",
           "incumbent": "#9467bd", "random": "#7f7f7f"}


def month_cutoffs(as_of: dt.date) -> list[dt.date]:
    """Every month start from MIN_CUTOFF whose 60-day label is complete by `as_of`."""
    last = as_of - dt.timedelta(days=LABEL_HORIZON_DAYS)
    return [d.date() for d in pd.date_range(MIN_CUTOFF, last, freq="MS")]


def train_cutoffs(cutoffs: list[dt.date], test: dt.date) -> list[dt.date]:
    """The cutoffs whose label was already known on `test`: no label window reaches past it."""
    return [c for c in cutoffs if c + dt.timedelta(days=LABEL_HORIZON_DAYS) <= test]


def early_cutoffs(test: dt.date, past: list[dt.date]) -> list[dt.date]:
    """The month starts after the last training cutoff whose early signal the level uses on `test`."""
    return [c for c in calibration_level.cutoffs(test)[1] if c > past[-1]]


def level_shifts(run_id: str, matured: pd.DataFrame, early: pd.DataFrame, test: dt.date) -> dict[int, float]:
    """The calibration level of a LightGBM run on the test cutoff, as scoring would set it."""
    info = run_info(run_id)
    predict = calibration_level.predictor(mlflow.lightgbm.load_model(f"runs:/{run_id}/model"), load_calibrator(run_id),
                                          load_segment_model(run_id), info["features"])
    return calibration_level.fit(predict, matured, early if len(early) else None, test)


def _predictions(run_id: str) -> pd.DataFrame:
    """The run's score of every test row: calibrated probability (LightGBM) or partial hazard (Cox)."""
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
    """The plan's metrics of one score on one cutoff's players."""
    y, s = rows[TARGET].to_numpy(), rows[score].to_numpy(dtype=float)
    top = np.argsort(-s, kind="stable")[: max(1, int(round(len(s) * top_share)))]
    out = {"auc": roc_auc_score(y, s), "c_index": c_index(rows[DURATION], s, rows[EVENT]),
           "top_decile_precision": y[top].mean(), "top_decile_recall": y[top].sum() / y.sum()}
    if probability:
        out.update(log_loss=log_loss(y, s), ece=expected_calibration_error(y, s), brier=brier_score_loss(y, s))
    return out


def top_decile_ratio(precision: pd.DataFrame) -> float:
    """Mean top-decile precision of the model over the recency rule's (columns model, recency)."""
    return float(precision["model"].mean() / precision["recency"].mean())


def replay(rows: pd.DataFrame, n: int) -> list[dict]:
    """Targeting replay on one cutoff: the top `n` players by each ranking, and what they went on to do.
    value_30d = trailing-90-day mean daily GGR x 30 (a player who won is worth 0); the candidate ranks by
    value at risk = value_30d x its churn probability, recency and incumbent by their own score. Realised
    GGR at risk = value_30d of the players who did churn."""
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
                "ggr_at_risk_per_1000": realised.mean() * 1000})  # the expected value of a random pick
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
    backtest_id = f"backtest_brand{label}_{as_of}_{int(start)}"
    print(f"{backtest_id}: {len(tests)} test cutoffs {[str(c) for c in tests]}, skipped {[str(c) for c in skipped]}")

    print("[1] features of every cutoff (raw once, then winsorised + sign-log)")
    raw = pd.concat([raw_features(c, brand_id) for c in needed], ignore_index=True)
    features = finalise(raw)
    extra = raw[KEY + ["churn_score", "ggr_eur_l90d"]]
    m = cfg["models"]
    params = {"n_estimators": m["max_trees"]}

    scenario_dir = SCENARIO_DIR / backtest_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, replay_rows = [], []
    for i, test in enumerate(tests, 1):
        past = train_cutoffs(cutoffs, test)
        print(f"\n== {i}/{len(tests)} test {test}: train {past[0]}..{past[-n_valid - 1]}, "
              f"validation {[str(c) for c in past[-n_valid:]]}")
        # Train labels as seen on the test cutoff; test labels as seen on AS_OF (what happened next).
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
        # Calibration level re-estimated on the test cutoff, as scoring does (calibration_level.py).
        matured = train_part[train_part["cutoff_date"].isin(past[-calibration_level.N_MATURED:])]
        early = features[features["cutoff_date"].isin(early_cutoffs(test, past))]
        shifts = {}
        for name in PROBABILITY_MODELS:
            shifts[name] = level_shifts(runs[name], matured, early, test)
            rows[f"{name}_static"] = rows[name]
            rows[name] = calibration_level.apply(rows[name], rows[BRAND], shifts[name])
        rows = rows.merge(test_part[KEY + ["days_since_last_bet"]].rename(columns={"days_since_last_bet": "recency"}),
                          on=KEY).merge(extra.rename(columns={"churn_score": "incumbent"}), on=KEY, how="left")
        for name in COMPARATORS:
            known = rows[rows[name].notna()]
            level = ({"level_shift": float(np.mean(list(shifts[name].values()))),
                      "ece_static": expected_calibration_error(known[TARGET], known[f"{name}_static"])}
                     if name in PROBABILITY_MODELS else {})
            metric_rows.append({"cutoff": str(test), "comparator": name, "n_players": len(known),
                                "churn_rate": known[TARGET].mean(), "n_train_cutoffs": len(past),
                                **cutoff_metrics(known, name, name in PROBABILITY_MODELS, cfg["top_share"]), **level})
        replay_rows += [{"cutoff": str(test), **r} for r in replay(rows, cfg["targeting_replay"]["contacts"])]
        print(pd.DataFrame(metric_rows[-len(COMPARATORS):]).round(4).to_string(index=False))

    metrics, replays = pd.DataFrame(metric_rows), pd.DataFrame(replay_rows)
    doc = _report(metrics, replays, cfg, backtest_id, label, as_of, skipped, time.time() - start)
    print(f"\nsaved {os.path.relpath(doc, PROJECT_ROOT)} | {(time.time() - start) / 60:.1f} min")
    return doc


def _summary(table: pd.DataFrame, by: str, columns: list[str], n_boot: int) -> pd.DataFrame:
    """Mean, min, max across cutoffs and a bootstrap CI of the mean (cutoffs resampled)."""
    out = []
    for name, rows in table.groupby(by, sort=False):
        for col in columns:
            v = rows[col].dropna().to_numpy()
            if not len(v):
                continue
            low, high = bootstrap_ci(len(v), lambda i: v[i].mean(), n_resamples=n_boot) if len(v) > 1 else (np.nan, np.nan)
            out.append({by: name, "metric": col, "mean": v.mean(), "ci_low": low, "ci_high": high,
                        "min": v.min(), "max": v.max()})
    return pd.DataFrame(out)


def figure(metrics: pd.DataFrame, replays: pd.DataFrame, cfg: dict, label: str, as_of: str, path: Path) -> Path:
    """AUC and top-decile precision by cutoff for every comparator, and the targeting replay."""
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


def _report(metrics, replays, cfg, backtest_id, label, as_of, skipped, seconds, run_id: str | None = None) -> Path:
    """Write the report; log a new backtest run to MLflow, or with `run_id` (a rewrite) update that run's
    verdict and report only."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = Path(str(DOC_PATH).format(brand=label))
    paths = {"metrics": OUTPUT_DIR / f"{backtest_id}_metrics.csv", "replay": OUTPUT_DIR / f"{backtest_id}_replay.csv",
             "figure": OUTPUT_DIR / f"{backtest_id}.png"}
    metrics.to_csv(paths["metrics"], index=False)
    replays.to_csv(paths["replay"], index=False)
    n_boot = cfg["bootstrap_resamples"]
    metric_cols = ["auc", "c_index", "top_decile_precision", "top_decile_recall", "log_loss", "ece", "brier"]
    summary = _summary(metrics, "comparator", metric_cols, n_boot)
    replay_summary = _summary(replays, "ranking", ["churn_captured", "ggr_at_risk_captured", "ggr_at_risk_per_1000"], n_boot)

    # Pass conditions of the delivery plan, on the candidate.
    rule = cfg["pass_conditions"]
    cand, rec = metrics[metrics["comparator"] == CANDIDATE], metrics[metrics["comparator"] == "recency"]
    precision = cand.set_index("cutoff")["top_decile_precision"].to_frame("model").join(
        rec.set_index("cutoff")["top_decile_precision"].rename("recency"))
    ratio = top_decile_ratio(precision)
    ratio_ci = bootstrap_ci(len(precision), lambda i: top_decile_ratio(precision.iloc[i]), n_boot)
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
                                                f"(range {r['min']:,.3f} to {r['max']:,.3f})", axis=1))
        wide = df.pivot(index=key, columns="metric", values="value").reset_index().fillna("n/a")
        return table(wide, [key] + [c for c in df["metric"].unique()])

    per_cutoff = metrics.pivot(index="cutoff", columns="comparator", values="auc").reset_index()
    level = [c for c in ("ece_static", "level_shift") if c in metrics]  # not in backtests before the calibration level
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

Mean, 95% bootstrap CI of the mean (cutoffs resampled, {n_boot} times) and range. Log-loss, ECE and Brier only for the calibrated probabilities.

{fmt_summary(summary, "comparator")}

## AUC per Cutoff

{table(per_cutoff, ["cutoff", "n_players", "churn_rate", "n_train_cutoffs"] + [c for c in COMPARATORS if c in per_cutoff] + [c for c in (f"ECE {CANDIDATE}", "ECE without level", "level shift") if c in per_cutoff])}

## Targeting Replay (would it have paid?)

A campaign that contacts the top **{n:,}** players at every cutoff (N to agree with the CRM team). The candidate ranks by **value at risk** = trailing-90-day mean daily GGR x 30 x its churn probability; `recency` and `incumbent` rank by their own score; `random` is the expected value of a random pick. **Realised GGR at risk** = the 30-day value of the players who did churn. No treatment was applied, so this measures targeting quality, not uplift.

{fmt_summary(replay_summary, "ranking")}

## How to Read It

- **Top-decile condition at {rule['top_decile_precision_vs_recency']:.2f}x, not the plan's 1.5x** (`docs/decision_top_decile_condition.md`): precision cannot exceed 1 and the recency rule already reaches {precision['recency'].mean():.3f}, so the most any model could reach is {1 / precision['recency'].mean():.2f}x, and this ceiling falls in the months with more churn. {rule['top_decile_precision_vs_recency']:.2f}x is the strictest round value the model meets with 95% confidence in brands 14, 64 and 73, and at today's precisions it is the plan's own demand (1.5 times the churners found per wasted contact). Interval: 95%, cutoffs resampled {n_boot} times.
- **Calibration lags the churn rate**: the calibration month is the latest one whose 60-day label was known on C, 2 to 3 months before it. When the monthly churn rate moves between them (around 50% from November to January, 30 to 40% the rest of the year), every probability is off by about the same amount: the ECE rises while the AUC does not move. The level shift on C removes most of it. What is left comes from the 60 days after C (a holiday season, a sports calendar), which no data before C shows: in the analysis of 2026-10-09 even the exact level of the month before C would not have kept every month of brand 14 under 0.08 (`eda/07_calibration_seasonality.ipynb`).
- **The replay trades churners for value**: the candidate ranks by value at risk, so it contacts fewer players who churn than recency does, but the ones it contacts carry most of the GGR at risk. Ranking by churn probability alone is the top-decile columns above.
- **The cutoffs are not independent**: the players repeat and the training months overlap, so the CIs are optimistic.
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
        mlflow.set_tags({"brand_id": label, "backtest_id": backtest_id, "verdict": "pass" if passed else "fail"})
        mlflow.log_params({"as_of": as_of, "test_cutoffs": sorted(metrics["cutoff"].unique()), "skipped": [str(s) for s in skipped],
                           **{f"config__{k}": v for k, v in cfg.items()}})
        for step, (cutoff, rows) in enumerate(metrics.groupby("cutoff")):
            for r in rows.itertuples(index=False):  # one point per cutoff: MLflow charts them over time
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
    """The report of a finished backtest, written again from its saved tables with the current pass
    conditions (no training): its MLflow run gets the new verdict and report."""
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT], filter_string=f"tags.backtest_id = '{backtest_id}'")
    if runs.empty:
        raise LookupError(f"no backtest run {backtest_id!r} in {MLFLOW_EXPERIMENT}")
    r = runs.iloc[0]
    metrics = pd.read_csv(OUTPUT_DIR / f"{backtest_id}_metrics.csv", dtype={"cutoff": str})
    replays = pd.read_csv(OUTPUT_DIR / f"{backtest_id}_replay.csv", dtype={"cutoff": str})
    doc = _report(metrics, replays, cfg, backtest_id, r["tags.brand_id"], r["params.as_of"],
                  ast.literal_eval(r["params.skipped"]), r["metrics.execution_time_s"], run_id=r["run_id"])
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
