"""Feature importance of the trained baseline models, three measures reported together
in docs/feature_importance_brand{brand}.md (gain importance alone overstates high-cardinality and
correlated features):

    permutation   drop in AUC and calibration (Cox: c-index) when one feature is shuffled,
                  on the validation rows, 10 repeats, 95% CI
    SHAP          what each feature does to each player's score, and in which direction:
                  global mean |SHAP| + dependence plots for the top 10
    ablation      retrain without a whole feature family, AUC (Cox: c-index) delta

    .venv/bin/python src/models/feature_importance.py                  # latest run of each model
    .venv/bin/python src/models/feature_importance.py --lgbm-run-id <id> --cox-run-id <id>

It uses the models logged by train.py, so the report always describes one exact run. The
figures and tables also go into each model's MLflow run, under feature_importance/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import FEATURE_COLUMNS, _unscale, brand_label
from segments import SEGMENT_PREFIX, add_segment_features, load_segment_model
from train import (
    BRAND, DURATION, ECE_BINS, EVENT, TARGET, churn_probability, cox_input, expected_calibration_error,
    latest_run, load_calibrator, load_catalog_entry, make_adapter, model_input, run_info, split_rows,
)

# One report per brand ("basel" when the runs cover several brands), so runs never overwrite each other.
DOC_PATH = PROJECT_ROOT / "docs/feature_importance_brand{brand}.md"
FIGURE_DIR = PROJECT_ROOT / "data/03_output/feature_importance/brand{brand}"
N_REPEATS = 10
TOP_N_DEPENDENCE = 10
SEED = 42
# Below this |rank correlation| between a feature and its SHAP value, one direction would mislead.
MONOTONIC_MIN_RHO = 0.5


def family_of(feature: str) -> str:
    """Feature family from build_features.FEATURE_COLUMNS; a _missing flag goes with its feature."""
    if feature.startswith(SEGMENT_PREFIX):
        return "segment"
    base = feature.removesuffix("_missing")
    return next((fam for fam, cols in FEATURE_COLUMNS.items() if base in cols), "other")


class Scorer:
    """Every metric one model is judged on, for any frame of rows."""

    def __init__(self, kind: str, model, features: list[str], train_mean: pd.Series | None = None,
                 calibrator=None):
        self.kind, self.model, self.features, self.train_mean = kind, model, features, train_mean
        self.calibrator = calibrator

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.kind == "classifier":
            # The delivered probability: calibrated when the run has a calibrator.
            return churn_probability(self.model, self.calibrator, rows, self.features)
        return np.asarray(self.model.predict_partial_hazard(cox_input(self.model, rows)))

    def metrics(self, rows: pd.DataFrame) -> dict:
        p = self.predict(rows)
        if self.kind == "classifier":
            return {"roc_auc": roc_auc_score(rows[TARGET], p), "brier": brier_score_loss(rows[TARGET], p),
                    "ece": expected_calibration_error(rows[TARGET], p)}
        # Higher hazard means an earlier churn, so the risk score is the negative hazard.
        return {"c_index": concordance_index(rows[DURATION], -p, rows[EVENT])}

    def shap(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Per-row SHAP values: log-odds for LightGBM (TreeSHAP), log-hazard for Cox."""
        if self.kind == "classifier":
            contrib = self.model.predict(model_input(rows, self.features), pred_contrib=True)[:, :-1]
            return pd.DataFrame(contrib, columns=self.features, index=rows.index)
        X = rows[self.features]
        # A linear model's exact SHAP value: beta * (x - training mean).
        return (X - self.train_mean[self.features]) * self.model.params_[self.features]


def permutation_importance(scorer: Scorer, rows: pd.DataFrame) -> pd.DataFrame:
    """Score drop per shuffled feature: mean and 95% CI over N_REPEATS shuffles.
    For AUC / c-index a positive drop means the feature helps; for Brier / ECE the
    metric rises, so the drop is reported as the (positive) rise."""
    rng = np.random.default_rng(SEED)
    base = scorer.metrics(rows)
    t = stats.t.ppf(0.975, N_REPEATS - 1)
    out = []
    for feature in scorer.features:
        draws = {k: [] for k in base}
        for _ in range(N_REPEATS):
            shuffled = rows.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            for k, v in scorer.metrics(shuffled).items():
                draws[k].append(base[k] - v if k in ("roc_auc", "c_index") else v - base[k])
        row = {"feature": feature, "family": family_of(feature)}
        for k, values in draws.items():
            values = np.asarray(values)
            half = t * values.std(ddof=1) / np.sqrt(N_REPEATS)
            row[f"{k}_drop"], row[f"{k}_ci_low"], row[f"{k}_ci_high"] = values.mean(), values.mean() - half, values.mean() + half
        out.append(row)
    main = "roc_auc_drop" if scorer.kind == "classifier" else "c_index_drop"
    return pd.DataFrame(out).sort_values(main, ascending=False).reset_index(drop=True)


def shap_summary(scorer: Scorer, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = scorer.shap(rows)
    direction = {
        f: stats.spearmanr(rows[f], values[f]).statistic if rows[f].nunique() > 1 else np.nan
        for f in scorer.features
    }
    table = pd.DataFrame({
        "feature": scorer.features,
        "family": [family_of(f) for f in scorer.features],
        "mean_abs_shap": [values[f].abs().mean() for f in scorer.features],
        "rank_corr": [direction[f] for f in scorer.features],
        "direction": [
            "categorical (brand)" if f == BRAND
            else "non-monotonic, see the plot" if not abs(direction[f]) >= MONOTONIC_MIN_RHO
            else "higher value, higher churn risk" if direction[f] > 0 else "higher value, lower churn risk"
            for f in scorer.features
        ],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return table, values


def ablation(adapter, features: list[str], valid: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Refit without each family that the model uses; delta = ablated - full (negative: the family helps)."""
    def scores(feats):
        model = adapter.fit(feats)
        return adapter.score(model, valid, feats), adapter.score(model, test, feats)

    full_valid, full_test = scores(features)
    rows = []
    for fam in [*FEATURE_COLUMNS, "segment"]:
        members = [f for f in features if family_of(f) == fam]
        if not members:
            rows.append({"family": fam, "features_in_model": "", "valid_delta": np.nan, "test_delta": np.nan})
            continue
        rest = [f for f in features if f not in members]
        if not rest:
            rows.append({"family": fam, "features_in_model": ", ".join(members),
                         "valid_delta": np.nan, "test_delta": np.nan})
            continue
        v, t = scores(rest)
        rows.append({"family": fam, "features_in_model": ", ".join(members),
                     "valid_delta": v - full_valid, "test_delta": t - full_test})
    table = pd.DataFrame(rows)
    table.attrs.update(full_valid=full_valid, full_test=full_test, metric=adapter.metric)
    return table


def _plot_permutation(perm: pd.DataFrame, metric: str, title: str, path: Path):
    perm = perm.sort_values(f"{metric}_drop")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(perm) + 1.2))
    err = [perm[f"{metric}_drop"] - perm[f"{metric}_ci_low"], perm[f"{metric}_ci_high"] - perm[f"{metric}_drop"]]
    ax.barh(perm["feature"], perm[f"{metric}_drop"], xerr=err, color="tab:blue", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"{metric} drop when shuffled (mean, 95% CI, {N_REPEATS} repeats)")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_shap_bar(table: pd.DataFrame, unit: str, title: str, path: Path):
    table = table.sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(table) + 1.2))
    ax.barh(table["feature"], table["mean_abs_shap"], color="tab:purple")
    ax.set_xlabel(f"mean |SHAP| ({unit})"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_dependence(rows: pd.DataFrame, values: pd.DataFrame, features: list[str], unit: str, title: str, path: Path):
    real = _unscale(rows[features])  # x axis in days / EUR / counts, not on the sign-log scale
    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), squeeze=False)
    for ax, f in zip(axes.flat, features):
        ax.scatter(real[f], values[f], s=4, alpha=0.3, color="tab:purple")
        ax.axhline(0, color="black", linewidth=0.8)
        if real[f].abs().max() > 1000:
            ax.set_xscale("symlog")
        ax.set_xlabel(f); ax.set_ylabel(f"SHAP ({unit})")
    for ax in list(axes.flat)[len(features):]:
        ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _md_table(df: pd.DataFrame, floats: int = 4) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.{floats}f}"
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _ci(df: pd.DataFrame, metric: str, digits: int = 4) -> pd.Series:
    return df.apply(lambda r: f"{r[f'{metric}_drop']:.{digits}f} [{r[f'{metric}_ci_low']:.{digits}f}, "
                              f"{r[f'{metric}_ci_high']:.{digits}f}]", axis=1)


def run(lgbm_run_id: str | None = None, cox_run_id: str | None = None) -> Path:
    lgbm = run_info(lgbm_run_id or latest_run("lightgbm_classifier").run_id)
    cox = run_info(cox_run_id or latest_run("cox_ph").run_id)
    if lgbm["dataset_path"] != cox["dataset_path"]:
        raise ValueError(f"the two runs use different datasets: {lgbm['dataset_path']} vs {cox['dataset_path']}")

    data = pd.read_parquet(PROJECT_ROOT / lgbm["dataset_path"])
    split = split_rows(data)
    # The k-means segments each run was trained with (same dataset and seed, so the same segments).
    segments = {name: load_segment_model(info["run_id"]) for name, info in (("lgbm", lgbm), ("cox", cox))}
    if segments["lgbm"] is not None:
        data = add_segment_features(data, segments["lgbm"])
        if segments["cox"] is not None and not np.array_equal(segments["lgbm"].assign(data), segments["cox"].assign(data)):
            raise ValueError("the two runs were trained with different player segments")
    train, valid, test = data[split == "train"], data[split == "valid"], data[split == "test"]

    lgbm_model = mlflow.lightgbm.load_model(f"runs:/{lgbm['run_id']}/model")
    cox_model = joblib.load(mlflow.artifacts.download_artifacts(f"runs:/{cox['run_id']}/model/model.joblib"))
    lgbm_calibrator = load_calibrator(lgbm["run_id"])
    lgbm_scorer = Scorer("classifier", lgbm_model, lgbm["features"], calibrator=lgbm_calibrator)
    cox_scorer = Scorer("survival", cox_model, cox["features"], train_mean=train[cox["features"]].mean())

    # Sanity check: the loaded models reproduce the metrics logged at training time.
    lgbm_key = "calibrated_roc_auc" if lgbm_calibrator is not None else "roc_auc"
    for name, scorer, info, key in (("LightGBM", lgbm_scorer, lgbm, lgbm_key), ("Cox", cox_scorer, cox, "c_index")):
        logged, now = info["metrics"][f"test_{key}"], scorer.metrics(test)[key.removeprefix("calibrated_")]
        if abs(logged - now) > 1e-4:
            raise ValueError(f"{name}: test {key} {now:.4f} differs from the logged {logged:.4f}")

    perm_lgbm = permutation_importance(lgbm_scorer, valid)
    perm_cox = permutation_importance(cox_scorer, valid)
    shap_lgbm, shap_values_lgbm = shap_summary(lgbm_scorer, valid)
    shap_cox, _ = shap_summary(cox_scorer, valid)
    abl_lgbm = ablation(make_adapter(load_catalog_entry(lgbm["model_id"])[0], lgbm["params"], train, valid),
                        lgbm["features"], valid, test)
    abl_cox = ablation(make_adapter(load_catalog_entry(cox["model_id"])[0], cox["params"], train, valid),
                       cox["features"], valid, test)

    label = brand_label(data[BRAND].unique())
    doc_path = Path(str(DOC_PATH).format(brand=label))
    fig_dir = Path(str(FIGURE_DIR).format(brand=label))
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig = {
        "perm_lgbm": fig_dir / "permutation_lightgbm.png",
        "perm_cox": fig_dir / "permutation_cox.png",
        "shap_lgbm": fig_dir / "shap_global_lightgbm.png",
        "shap_cox": fig_dir / "shap_global_cox.png",
        "dep_lgbm": fig_dir / "shap_dependence_lightgbm.png",
    }
    _plot_permutation(perm_lgbm, "roc_auc", "LightGBM: permutation importance (validation)", fig["perm_lgbm"])
    _plot_permutation(perm_cox, "c_index", "Cox PH: permutation importance (validation)", fig["perm_cox"])
    _plot_shap_bar(shap_lgbm, "log-odds", "LightGBM: global SHAP (validation)", fig["shap_lgbm"])
    _plot_shap_bar(shap_cox, "log-hazard", "Cox PH: global SHAP (validation)", fig["shap_cox"])
    top = [f for f in shap_lgbm["feature"] if f != BRAND][:TOP_N_DEPENDENCE]  # brandId is categorical
    _plot_dependence(valid, shap_values_lgbm, top, "log-odds", "LightGBM: SHAP dependence, top features", fig["dep_lgbm"])

    tables = {
        "permutation_lightgbm.csv": perm_lgbm, "permutation_cox.csv": perm_cox,
        "shap_lightgbm.csv": shap_lgbm, "shap_cox.csv": shap_cox,
        "ablation_lightgbm.csv": abl_lgbm, "ablation_cox.csv": abl_cox,
    }
    for name, table in tables.items():
        table.to_csv(fig_dir / name, index=False)

    doc = _write_doc(lgbm, cox, data, split, perm_lgbm, perm_cox, shap_lgbm, shap_cox, abl_lgbm, abl_cox, fig,
                     calibrated=lgbm_calibrator is not None, doc_path=doc_path)
    for info in (lgbm, cox):
        with mlflow.start_run(run_id=info["run_id"]):
            mlflow.log_artifact(str(doc), artifact_path="feature_importance")
            mlflow.log_artifacts(str(fig_dir), artifact_path="feature_importance")
    print(f"saved {os.path.relpath(doc, PROJECT_ROOT)} and {os.path.relpath(fig_dir, PROJECT_ROOT)}/, "
          f"logged to runs {lgbm['run_name']} and {cox['run_name']}")
    return doc


def _write_doc(lgbm, cox, data, split, perm_lgbm, perm_cox, shap_lgbm, shap_cox, abl_lgbm, abl_cox, fig,
               calibrated: bool, doc_path: Path) -> Path:
    rel = lambda p: os.path.relpath(p, doc_path.parent)
    test_cutoff = sorted(str(c) for c in data.loc[split == "test", "cutoff_date"].unique())
    n = {s: int((split == s).sum()) for s in ("train", "valid", "test")}

    perm_l = pd.DataFrame({
        "feature": perm_lgbm["feature"], "family": perm_lgbm["family"],
        "AUC drop [95% CI]": _ci(perm_lgbm, "roc_auc"),
        "Brier rise [95% CI]": _ci(perm_lgbm, "brier"),
        "ECE rise [95% CI]": _ci(perm_lgbm, "ece"),
    })
    perm_c = pd.DataFrame({"feature": perm_cox["feature"], "family": perm_cox["family"],
                           "c-index drop [95% CI]": _ci(perm_cox, "c_index")})
    abl = abl_lgbm[["family", "features_in_model"]].rename(columns={"features_in_model": "LightGBM features"})
    abl["LightGBM AUC delta, valid"] = abl_lgbm["valid_delta"]
    abl["LightGBM AUC delta, test"] = abl_lgbm["test_delta"]
    abl["Cox features"] = abl_cox["features_in_model"]
    abl["Cox c-index delta, valid"] = abl_cox["valid_delta"]
    abl["Cox c-index delta, test"] = abl_cox["test_delta"]

    best_family = lambda t: t.dropna(subset=["test_delta"]).sort_values("test_delta").iloc[0]
    bl, bc = best_family(abl_lgbm), best_family(abl_cox)

    brands = ", ".join(str(b) for b in sorted(data[BRAND].unique()))
    text = f"""# Feature Importance v0 (brandId {brands})

This document reports three importance measures together for the two baseline models. Gain-based importance alone overstates features with many distinct values and features that are correlated with others, so I do not use it here.

| Measure | Question it answers |
|---|---|
| Permutation importance | How much do the model's AUC and calibration fall when this feature is shuffled? |
| SHAP values | What does this feature do to each player's score, and in which direction? |
| Ablation by family | What is a whole family of features worth to the model? |

## 0. Models and Data

| | LightGBM | Cox PH |
|---|---|---|
| MLflow run | `{lgbm['run_name']}` | `{cox['run_name']}` |
| Target | `event_60d` (churn in the next 60 days) | `duration_days` + `event_observed` (churn day) |
| Features (after Stage 2 selection) | {len(lgbm['features'])} | {len(cox['features'])} |
| Test score (last cutoff) | AUC {lgbm['metrics']['test_roc_auc']:.4f} | c-index {cox['metrics']['test_c_index']:.4f} |

- Dataset: `{lgbm['dataset_path']}`.
- Rows: train {n['train']:,}, validation {n['valid']:,}, test {n['test']:,}. The test month is {', '.join(test_cutoff)}.
- With only 3 cutoffs there is no separate validation month. The validation rows are 15% of the players of the training months, never seen in training. Permutation importance and SHAP use them.
- Brands: LightGBM gets `brandId` as a categorical feature and Cox PH is stratified by brand (one baseline per brand), so `brandId` never appears in the Cox tables. With one brand it is constant and its importance is 0. Ablation does not drop it: the model always adds it.
- Every number here comes from the models logged in those two runs. The script checks that they reproduce the logged test score before it measures anything.

## 1. Permutation Importance

I shuffle one feature at a time on the validation rows, {N_REPEATS} times, and measure how much the score changes. The interval is a 95% t-interval over the {N_REPEATS} shuffles, so it shows how stable the shuffle is, not the sampling error of the validation set.

- **AUC drop**: positive means the feature helps the model rank players.
- **Brier rise** and **ECE rise**: positive means the probabilities get less reliable. ECE is the expected calibration error over {ECE_BINS} equal-size bins.
- The probabilities are the ones the model delivers: {"isotonic-calibrated" if calibrated else "not calibrated (this run has no calibrator)"}.

### LightGBM

{_md_table(perm_l)}

![Permutation importance, LightGBM]({rel(fig['perm_lgbm'])})

### Cox PH

The Cox model gives a risk ranking, not a probability, so only the c-index is measured.

{_md_table(perm_c)}

![Permutation importance, Cox PH]({rel(fig['perm_cox'])})

## 2. SHAP Values

SHAP splits each player's score into one part per feature. The parts add up to the score.

- **LightGBM**: exact TreeSHAP values from LightGBM itself (`pred_contrib=True`, the same algorithm as `shap.TreeExplainer`), in log-odds of churn.
- **Cox PH**: the model is linear, so the exact SHAP value is `beta * (x - training mean)`, in log-hazard. Positive means a higher risk of churning sooner.
- **Direction** is the sign of the rank correlation (`rank_corr`) between the feature value and its SHAP value. Under {MONOTONIC_MIN_RHO} in absolute value one direction would mislead, so it says "non-monotonic" and the dependence plot shows the real shape.

### LightGBM

{_md_table(shap_lgbm)}

![Global SHAP, LightGBM]({rel(fig['shap_lgbm'])})

Dependence plots for the top {min(TOP_N_DEPENDENCE, len(shap_lgbm))} features (all the features the model uses). The x axis is in real units (days, EUR, counts), not on the sign-log scale. Each dot is one validation player.

![SHAP dependence, LightGBM]({rel(fig['dep_lgbm'])})

### Cox PH

{_md_table(shap_cox)}

![Global SHAP, Cox PH]({rel(fig['shap_cox'])})

A dependence plot for a linear model is a straight line with slope `beta`, so I do not draw it. The hazard ratios are in `hazard_ratios.csv` in the Cox MLflow run.

## 3. Ablation by Family

I retrain the model without every feature of one family and compare it with the full model (same features otherwise, same parameters, same rows). A negative delta means the family helps. A family with no feature in the model is left empty.

Full model: LightGBM AUC valid {abl_lgbm.attrs['full_valid']:.4f}, test {abl_lgbm.attrs['full_test']:.4f}. Cox c-index valid {abl_cox.attrs['full_valid']:.4f}, test {abl_cox.attrs['full_test']:.4f}.

{_md_table(abl)}

The most valuable family on the test month is **{bl['family']}** for LightGBM (AUC {bl['test_delta']:+.4f} without it) and **{bc['family']}** for Cox (c-index {bc['test_delta']:+.4f} without it).

## 4. Known Limits

- **The Cox test month only has churn on day 0.** With data through 2026-10-04, a churn day after the cutoff can only be confirmed up to 60 days before the data ends, which is day 0 for the last cutoff. So on the test month the c-index measures "who is already gone", not "which day".
- **The intervals only cover the shuffle.** A different validation sample would move the numbers more than the intervals show.
- **Correlated features share credit.** Stage 2 already removed pairs over 0.95, but features from the same family (for example `n_active_days_last_30d` and `max_prior_gap_days`) can still hide each other in permutation importance. Ablation by family shows the joint value.

Generated by `src/models/feature_importance.py` (`make importance`). The figures and tables are in `{os.path.relpath(fig['perm_lgbm'].parent, PROJECT_ROOT)}/` and in the `feature_importance/` folder of both MLflow runs.
"""
    doc_path.write_text(text, encoding="utf-8")
    return doc_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Permutation importance, SHAP and family ablation for the baseline models.")
    parser.add_argument("--lgbm-run-id", help="default: the latest lightgbm_classifier run")
    parser.add_argument("--cox-run-id", help="default: the latest cox_ph run")
    args = parser.parse_args()
    run(args.lgbm_run_id, args.cox_run_id)
