"""Ablation of every feature family: what each family adds, on all the features. docs/feature_ablation.md.

    .venv/bin/python src/models/ablation_full.py           (make ablation)

The ablation in feature_importance.py only tests the families that the selected model uses. Here, for each
brand, the reference LightGBM is trained on all the features of its registered dataset (same split,
same parameters, early stopping on the validation month), and then again without each family. On the
test months it reports the AUC, the log-loss and the ECE of the probabilities calibrated on the
validation month (isotonic), and the GGR at risk reached by 1,000 contacts per month ranked by value at
risk. Local files only; aggregates only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / p) for p in ("src/evaluation", "src/models", "src/features", "src")]
import signal_snapshots  # noqa: E402
from build_features import FEATURE_COLUMNS  # noqa: E402
from evaluate import run_data  # noqa: E402
from train import (  # noqa: E402
    BRAND,
    TARGET,
    load_catalog_entry,
    make_adapter,
    model_columns,
    model_input,
    run_info,
)

from evaluation.metrics import expected_calibration_error  # noqa: E402

BRANDS = [14, 23, 64, 72, 73]
CONTACTS = 1000
KEY = ["tenant_id", "player_id"]
REPORT = PROJECT_ROOT / "docs/feature_ablation.md"
OUT_DIR = PROJECT_ROOT / "data/03_output/feature_ablation"


def value_30d(ggr_l90d: pd.Series) -> pd.Series:
    """Expected 30-day GGR, as scoring computes it."""
    return (ggr_l90d.clip(lower=0) / 90 * 30).fillna(0)


def capture(test: pd.DataFrame, p: np.ndarray, contacts: int = CONTACTS) -> float:
    """Mean over the test months of the share of GGR at risk reached by the top `contacts` by value at risk."""
    shares = []
    for _, rows in test.assign(_p=p).groupby("cutoff_date"):
        value = value_30d(rows["ggr_eur_l90d"])
        realised = value * rows[TARGET]
        top = (value * rows["_p"]).to_numpy().argsort(kind="stable")[::-1][:min(contacts, len(rows))]
        shares.append(realised.iloc[top].sum() / realised.sum())
    return float(np.mean(shares))


def scores(model, entry, features, valid, test) -> dict:
    cols = model_columns(entry, features)
    pv = model.predict_proba(model_input(valid, cols))[:, 1]
    pt = model.predict_proba(model_input(test, cols))[:, 1]
    cal = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(pv, valid[TARGET])
    pc = cal.predict(pt)
    y = test[TARGET].to_numpy()
    return {"auc": roc_auc_score(y, pt), "log_loss": log_loss(y, pc), "ece": expected_calibration_error(y, pc),
            "ggr_captured": capture(test, pc)}


def brand_ablation(brand: int, client) -> pd.DataFrame:
    version = max(client.search_model_versions(f"name='churn_lightgbm_classifier_brand{brand}'"), key=lambda v: int(v.version))
    info = run_info(version.run_id)
    data, split = run_data(info)
    entry = load_catalog_entry(info["model_id"])[0]
    train, valid, test = (data[split == s] for s in ("train", "valid", "test"))
    snaps = pd.concat([signal_snapshots.snapshots([pd.Timestamp(c).date()], brand, build_missing=False)[KEY + ["snapshot_date", "ggr_eur_l90d"]]
                       for c in test["cutoff_date"].unique()])
    snaps["cutoff_date"] = pd.to_datetime(snaps.pop("snapshot_date"))
    test = test.assign(cutoff_date=pd.to_datetime(test["cutoff_date"])).drop(columns="ggr_eur_l90d", errors="ignore") \
        .merge(snaps, on=KEY + ["cutoff_date"], how="left")
    features = [f for cols in FEATURE_COLUMNS.values() for f in cols if f in data and data[f].notna().any()]
    features += [BRAND] if BRAND in data else []
    adapter = make_adapter(entry, info["params"], train, valid)
    rows = []
    for family in ["none", *FEATURE_COLUMNS]:
        members = [] if family == "none" else [f for f in FEATURE_COLUMNS[family] if f in features]
        if family != "none" and not members:
            continue
        kept = [f for f in features if f not in members]
        model = adapter.fit(kept)
        rows.append({"brand": brand, "removed": family, "n_removed": len(members), **scores(model, entry, kept, valid, test)})
        print(f"brand {brand} without {family}: {rows[-1]['auc']:.4f}", flush=True)
    out = pd.DataFrame(rows)
    full = out.iloc[0]
    for m in ("auc", "log_loss", "ece", "ggr_captured"):
        out[f"delta_{m}"] = out[m] - full[m]
    return out


def plot(table: pd.DataFrame, path: Path) -> None:
    t = table[table["removed"] != "none"]
    fams = [f for f in FEATURE_COLUMNS if f in set(t["removed"])]
    colours = {14: "#2f6f8f", 23: "#8fb3c7", 64: "#b5452f", 72: "#e3b5a8", 73: "#b9b2a6"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    y = np.arange(len(fams)); h = 0.16
    for j, b in enumerate(BRANDS):
        g = t[t["brand"] == b].set_index("removed").reindex(fams)
        axes[0].barh(y + (j - 2) * h, -g["delta_auc"], h, color=colours[b], label=f"Brand {b}")
        axes[1].barh(y + (j - 2) * h, -100 * g["delta_ggr_captured"], h, color=colours[b])
    axes[0].set_yticks(y, fams); axes[0].invert_yaxis()
    axes[0].set(xlabel="AUC lost without the family", title="Ranking"); axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set(xlabel="points of GGR at risk lost without the family", title="Value reached by 1,000 contacts")
    for ax in axes:
        ax.axvline(0, color="grey", lw=1); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def write(table: pd.DataFrame, figure: Path) -> Path:
    lines = ["# Feature Family Ablation", "",
             "For each brand, the reference LightGBM is trained on all the features of its registered dataset and "
             "then again without each family (same split, parameters and early stopping). Test months of the "
             "registered model; probabilities calibrated on the validation month (isotonic); GGR at risk reached by "
             f"{CONTACTS:,} contacts per month ranked by value at risk. A negative delta for AUC or GGR (positive for "
             "log-loss or ECE) means the family helps. Generated by `src/models/ablation_full.py` (`make ablation`).", "",
             f"![Ablation](../{figure.relative_to(PROJECT_ROOT).as_posix()})", ""]
    for b, g in table.groupby("brand"):
        full = g.iloc[0]
        lines += [f"## Brand {b}", "",
                  f"All features: AUC {full['auc']:.4f}, log-loss {full['log_loss']:.4f}, ECE {full['ece']:.4f}, "
                  f"GGR at risk reached {100 * full['ggr_captured']:.1f}%.", "",
                  "| removed family | features | AUC delta | log-loss delta | ECE delta | GGR reached delta (points) |",
                  "|---|---|---|---|---|---|"]
        for r in g.iloc[1:].itertuples(index=False):
            lines.append(f"| {r.removed} | {r.n_removed} | {r.delta_auc:+.4f} | {r.delta_log_loss:+.4f} | "
                         f"{r.delta_ece:+.4f} | {100 * r.delta_ggr_captured:+.2f} |")
        lines.append("")
    REPORT.write_text("\n".join(lines) + "\n")
    return REPORT


def run() -> Path:
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    client = mlflow.MlflowClient()
    table = pd.concat([brand_ablation(b, client) for b in BRANDS], ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "ablation.csv", index=False)
    figure = OUT_DIR / "ablation.png"
    plot(table, figure)
    return write(table, figure)


if __name__ == "__main__":
    print(f"saved {run()}")
