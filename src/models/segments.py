"""Player segments with k-means: an optional input feature for the churn models, never the score.

    .venv/bin/python src/models/segments.py      # fit on the latest dataset, write docs/player_segments_brand{brand}.md

k-means comes from the catalog (`kmeans`). The number of segments is the one with the highest
silhouette coefficient among the catalog's param_grid.n_clusters. It is fitted on the training
months only (never the test months), on a 30-day behaviour profile standardised to mean 0 and
standard deviation 1.

train.py fits the same segmentation, adds it as one-hot columns (segment_1 ... segment_{k-1},
segment 0 is the reference) and lets Stage 2 selection decide whether a model keeps them. The
fitted segmentation is logged with each run, so scoring assigns new players the same way.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
from build_features import _unscale, brand_label

# The 30-day behaviour profile: how often, how much, how they pay, bonus use and recency.
# Deposit features are left out: they are empty before March 2026 and k-means needs every value.
SEGMENT_FEATURES = [
    "active_days_l30d", "wagered_eur_l30d", "ggr_eur_l30d", "games_breadth_30d",
    "bonus_stake_share_30d", "engagement_score", "days_since_last_bet",
]
LABELS = {
    "active_days_l30d": "active days (30d)", "wagered_eur_l30d": "stake (30d, EUR)",
    "ggr_eur_l30d": "net loss (30d, EUR)", "games_breadth_30d": "distinct games (30d)",
    "bonus_stake_share_30d": "share of stake from bonus (30d)", "engagement_score": "engagement score",
    "days_since_last_bet": "days since last bet",
}
SEGMENT_PREFIX = "segment_"
SILHOUETTE_SAMPLE = 10_000  # silhouette is quadratic in rows, so it is measured on a sample
SEED = 42
ARTIFACT = "segments/kmeans.joblib"
# One report per brand ("basel" when the dataset has several brands), so runs never overwrite each other.
DOC_PATH = PROJECT_ROOT / "docs/player_segments_brand{brand}.md"
FIGURE_DIR = PROJECT_ROOT / "data/03_output/segments/brand{brand}"


@dataclass
class SegmentModel:
    scaler: StandardScaler
    kmeans: object
    features: list[str]
    k: int
    silhouette: dict[int, float]
    order: np.ndarray  # raw k-means label -> segment id, segment 0 = the largest

    def assign(self, rows: pd.DataFrame) -> np.ndarray:
        return self.order[self.kmeans.predict(self.scaler.transform(rows[self.features].fillna(0)))]

    def centroids(self) -> pd.DataFrame:
        """Standardised centroid per segment (0 = average player, +1 = one std above)."""
        centres = pd.DataFrame(self.kmeans.cluster_centers_, columns=self.features)
        centres.index = self.order[np.arange(self.k)]
        return centres.sort_index()


def fit_segments(rows: pd.DataFrame) -> SegmentModel:
    """k-means on `rows` (the training months), k chosen by the silhouette coefficient."""
    from train import instantiate, load_catalog_entry  # train.py imports this module

    entry, _, _ = load_catalog_entry("kmeans")
    # An empty value (e.g. bonus share with no stake) counts as 0: k-means needs every value.
    scaler = StandardScaler().fit(rows[SEGMENT_FEATURES].fillna(0))
    X = scaler.transform(rows[SEGMENT_FEATURES].fillna(0))
    sample = min(SILHOUETTE_SAMPLE, len(X))
    # n_init=10 instead of the catalog's "auto" (a single start): k-means depends on the start.
    fitted, silhouette = {}, {}
    for k in entry["param_grid"]["n_clusters"]:
        km = instantiate(entry, {"n_clusters": k, "n_init": 10, "random_state": SEED})
        labels = km.fit_predict(X)
        fitted[k] = km
        silhouette[k] = float(silhouette_score(X, labels, sample_size=sample, random_state=SEED))
    best = max(silhouette, key=silhouette.get)
    sizes = np.bincount(fitted[best].labels_, minlength=best)
    order = np.empty(best, dtype=int)
    order[np.argsort(-sizes)] = np.arange(best)
    return SegmentModel(scaler, fitted[best], list(SEGMENT_FEATURES), best, silhouette, order)


def add_segment_features(data: pd.DataFrame, model: SegmentModel) -> pd.DataFrame:
    """One-hot segment_1 ... segment_{k-1}; segment 0 (the largest) is the reference."""
    segment = model.assign(data)
    out = data.copy()
    for s in range(1, model.k):
        out[f"{SEGMENT_PREFIX}{s}"] = (segment == s).astype("int64")
    return out


def load_segment_model(run_id: str) -> SegmentModel | None:
    """The segmentation logged with a training run, or None for a run without one."""
    import mlflow

    try:
        return joblib.load(mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{ARTIFACT}"))
    except Exception:
        return None


def _name(z: pd.Series) -> str:
    """A short, rule-based name from the standardised centroid."""
    activity = ("Frequent" if z["active_days_l30d"] >= 0.5
                else "Occasional" if z["active_days_l30d"] <= -0.5 else "Regular")
    value = ("high-value" if z["wagered_eur_l30d"] >= 0.5
             else "low-value" if z["wagered_eur_l30d"] <= -0.5 else "mid-value")
    name = f"{activity} {value}"
    if z["days_since_last_bet"] >= 0.75:
        name = f"Fading, {name.lower()}"
    if z["bonus_stake_share_30d"] >= 0.75:
        name += ", bonus-heavy"
    return name


def _describe(z: pd.Series, n: int = 3) -> str:
    """The features where the segment differs most from the average player."""
    top = z.reindex(z.abs().sort_values(ascending=False).index).head(n)
    return ", ".join(f"{'high' if v > 0 else 'low'} {LABELS[f]} ({v:+.1f} sd)" for f, v in top.items() if abs(v) >= 0.3) \
        or "close to the average player"


def _use_in_models() -> str:
    """What Stage 2 decided about the segment features in the latest run of each baseline model."""
    import mlflow

    from train import latest_run

    lines = []
    for model_id in ("lightgbm_classifier", "cox_ph"):
        try:
            run = latest_run(model_id)
            table = pd.read_csv(mlflow.artifacts.download_artifacts(f"runs:/{run.run_id}/feature_selection.csv"))
        except Exception:
            lines.append(f"- **{model_id}**: no training run with a feature selection report yet.")
            continue
        rows = table[table["feature"].str.startswith(SEGMENT_PREFIX)]
        if rows.empty:
            lines.append(f"- **{model_id}** (`{run['tags.mlflow.runName']}`): trained without segments.")
            continue
        for r in rows.itertuples():
            verdict = "kept" if r.selected else f"dropped at step {int(r.step)} ({r.reason})"
            lines.append(f"- **{model_id}** (`{run['tags.mlflow.runName']}`): `{r.feature}` {verdict}.")
    return "\n".join(lines)


def write_report(data: pd.DataFrame, split: pd.Series, model: SegmentModel, dataset_path: Path) -> Path:
    training, test = data[split != "test"], data[split == "test"]
    label = brand_label(data["brandId"].unique())
    doc_path = Path(str(DOC_PATH).format(brand=label))
    fig_dir = Path(str(FIGURE_DIR).format(brand=label))
    seg_all = pd.Series(model.assign(data), index=data.index)
    real = _unscale(data[model.features])
    z = model.centroids()
    names = {s: _name(z.loc[s]) for s in z.index}

    profile = []
    for s in range(model.k):
        rows = (seg_all == s) & (split != "test")
        med = real[rows].median()
        profile.append({
            "segment": s, "name": names[s], "share": rows.sum() / len(training),
            **{LABELS[f]: med[f] for f in model.features},
            "churn 60d (training months)": data.loc[rows, "event_60d"].mean(),
            "churn 60d (test months)": data.loc[(seg_all == s) & (split == "test"), "event_60d"].mean(),
        })
    profile = pd.DataFrame(profile)
    by_cutoff = pd.crosstab(data["cutoff_date"], seg_all, normalize="index")
    by_cutoff.columns = [f"segment {c}" for c in by_cutoff.columns]

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    ks = list(model.silhouette)
    ax.plot(ks, [model.silhouette[k] for k in ks], marker="o")
    ax.axvline(model.k, color="grey", linestyle=":", label=f"chosen k = {model.k}")
    ax.set_xlabel("number of segments (k)"); ax.set_ylabel("silhouette coefficient"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "silhouette.png", dpi=120)

    fig = Figure(figsize=(10, 0.6 * model.k + 2.8))
    ax = fig.subplots()
    im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(model.features)), [LABELS[f] for f in model.features], rotation=35, ha="right")
    ax.set_yticks(range(model.k), [f"{s}: {names[s]}" for s in z.index])
    for i in range(model.k):
        for j in range(len(model.features)):
            ax.text(j, i, f"{z.iloc[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="std. deviations from the average player")
    ax.set_title("Segment profiles (standardised centroids)")
    fig.tight_layout(); fig.savefig(fig_dir / "segment_profiles.png", dpi=120)
    profile.to_csv(fig_dir / "segment_profiles.csv", index=False)

    rel = lambda p: os.path.relpath(p, doc_path.parent)
    fmt = lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}" if isinstance(v, float) else str(v)
    sil_table = "\n".join(f"| {k} | {v:.4f} | {'yes' if k == model.k else ''} |" for k, v in model.silhouette.items())
    runner_up = sorted(model.silhouette, key=model.silhouette.get, reverse=True)[1]
    dummies = ("`segment_1`" if model.k == 2 else f"`segment_1` ... `segment_{model.k - 1}`")
    k2_note = (' With k = 2 the split is mostly "engaged vs not engaged". A finer set of types scores a lower '
               "silhouette, so its groups would overlap more." if model.k == 2 else "")
    cols = ["segment", "name", "share"] + [LABELS[f] for f in model.features]
    prof_table = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
        "| " + " | ".join([str(r["segment"]), r["name"], f"{r['share']:.1%}"] + [fmt(r[LABELS[f]]) for f in model.features]) + " |"
        for _, r in profile.iterrows())
    churn_table = "| segment | name | what stands out | churn 60d, training months | churn 60d, test months |\n|---|---|---|---|---|\n" + "\n".join(
        f"| {s} | {names[s]} | {_describe(z.loc[s])} | {profile.loc[s, 'churn 60d (training months)']:.1%} | "
        f"{profile.loc[s, 'churn 60d (test months)']:.1%} |" for s in range(model.k))
    stab_table = "| cutoff | " + " | ".join(by_cutoff.columns) + " |\n|" + "---|" * (len(by_cutoff.columns) + 1) + "\n" + "\n".join(
        f"| {c} | " + " | ".join(f"{v:.1%}" for v in row) + " |" for c, row in by_cutoff.iterrows())
    overall = data.loc[split != "test", "event_60d"].mean()

    brands = ", ".join(str(b) for b in sorted(data["brandId"].unique()))
    text = f"""# Player Segments v0 (brandId {brands})

I group players by how they behaved in the 30 days before the cutoff, with k-means. The segments are an **input feature** for the churn models, never the churn score: the score comes only from LightGBM (probability) and Cox PH (survival). Stage 2 selection decides whether a model keeps them.

## 1. Method

- **Brands**: one segmentation for every brand in the dataset ({brands}). The features were already computed and winsorised per brand upstream; here the segments are common to all brands, so "high value" means high in EUR, not high relative to the player's own brand.
- **Rows**: the train and validation months ({', '.join(sorted(str(c) for c in training['cutoff_date'].unique()))}), {len(training):,} player snapshots. The test months ({', '.join(sorted(str(c) for c in test['cutoff_date'].unique()))}) are only assigned to a segment, never used to fit them.
- **Features** (the 30-day behaviour profile): {', '.join(f'`{f}`' for f in model.features)}. They are on the sign-log scale from `build_features.py` and then standardised (mean 0, std 1), so no feature dominates because of its units.
- **Model**: `kmeans` from `catalog/model_library_catalog.json` (`sklearn.cluster.KMeans`), 10 starts, seed {SEED}.
- **Number of segments**: the k with the highest silhouette coefficient among {ks}. Silhouette goes from -1 to 1: high means players are close to their own segment and far from the others. I measure it on a random sample of {min(SILHOUETTE_SAMPLE, len(training)):,} rows, because it is slow on all of them.
- **Order**: segment 0 is the largest. In the models it is the reference, so they get {dummies} (1 if the player is in that segment).

## 2. Choosing k

| k | silhouette | chosen |
|---|---|---|
{sil_table}

![Silhouette by k]({rel(fig_dir / 'silhouette.png')})

**Chosen: k = {model.k}** (silhouette {model.silhouette[model.k]:.4f}). The next best is k = {runner_up} ({model.silhouette[runner_up]:.4f}).{k2_note}

## 3. Player Types

Median values per segment, in real units (training months):

{prof_table}

How each segment differs from the average player (standardised centroid, top 3 features) and its churn rate. The average churn in the training months is {overall:.1%}. The names come from simple rules on the centroid (activity, stake, recency, bonus use), so they stay the same when I rerun this.

{churn_table}

![Segment profiles]({rel(fig_dir / 'segment_profiles.png')})

## 4. Stability Over Time

Share of players in each segment, per cutoff (the last two are the test months):

{stab_table}

## 5. Use in the Models

Stage 2 decision for the segment features in the latest training run of each model (read from MLflow when this file is generated, so run `make train-baseline` before `make segments`):

{_use_in_models()}

When the segment is dropped at step 4, the model already gets the same information from the raw features it is built from (active days, stake, deposits). The segment is still useful to describe players: `make score` adds it to every player's score.

## 6. Known Limits

- **k-means looks for round groups of similar size.** Real player types can be long or uneven shapes; a low silhouette means the groups overlap a lot and the borders between them are soft.
- **The churn rates are descriptive.** They show that segments differ, not that the segment causes the churn.
- **A small segment can be dropped by Stage 2** as near-zero variance, and the models can ignore the segments when the raw features already carry the same information. `feature_selection.csv` in each MLflow run says what happened.

Generated by `src/models/segments.py` (`make segments`) from `{os.path.relpath(dataset_path, PROJECT_ROOT)}`.
"""
    doc_path.write_text(text, encoding="utf-8")
    return doc_path


def run_report(dataset_path: str | Path | None = None, run_id: str | None = None) -> Path:
    """Write docs/player_segments_brand{brand}.md for a training dataset (default: the latest), with the
    segmentation logged in training run `run_id` (trained on that dataset), or a new fit without one."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import PROCESSED_DIR, split_rows

    if dataset_path is None:
        dataset_path = max(PROCESSED_DIR.glob("train_dataset_*.parquet"), key=lambda p: p.stat().st_mtime)
    dataset_path = Path(dataset_path)
    data = pd.read_parquet(dataset_path)
    split = split_rows(data)
    model = (run_id and load_segment_model(run_id)) or fit_segments(data[split != "test"])
    doc = write_report(data, split, model, dataset_path)
    print(f"k = {model.k} | silhouette {', '.join(f'{k}: {v:.3f}' for k, v in model.silhouette.items())}")
    print(f"saved {os.path.relpath(doc, PROJECT_ROOT)}")
    return doc


if __name__ == "__main__":
    run_report()
