"""60 against 90 days of churn: does the threshold change who the model picks? docs/churn_threshold_comparison.md.

    .venv/bin/python src/models/threshold_comparison.py       (make threshold-comparison)

Needs, for each brand, the official registered LightGBM and the one of a sensitivity run
(`make pipeline AS_OF=... BRAND=b CHURN_DAYS=90`, registered as ..._churn90d). On the test cutoffs that
are out of sample for both models, the same players are scored by each model (its own features, caps
and calibrator) and judged against both labels: AUC, top-decile precision, overlap of the lists, and
the GGR at risk reached by 1,000 contacts ranked by value at risk. Local files only; aggregates only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / p) for p in ("src/evaluation", "src/models", "src/features", "src")]
import signal_snapshots  # noqa: E402
from evaluate import run_data  # noqa: E402
from segments import add_segment_features, load_segment_model  # noqa: E402
from train import churn_probability, load_calibrator, run_info  # noqa: E402

from evaluation.metrics import top_decile_precision  # noqa: E402

BRANDS = [14, 23, 64, 72, 73]
OTHER_DAYS = 90
CONTACTS = 1000
KEY = ["tenant_id", "player_id"]
REPORT = PROJECT_ROOT / "docs/churn_threshold_comparison.md"
OUT = PROJECT_ROOT / "data/03_output/churn_threshold_comparison.csv"


def latest_version(client, name: str) -> str:
    return max(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)).run_id


def scored(run_id: str) -> tuple[pd.DataFrame, set, set]:
    """Test rows scored by the model of the run, its test cutoffs and its train/validation cutoffs."""
    info = run_info(run_id)
    data, split = run_data(info)
    data = data.assign(cutoff_date=pd.to_datetime(data["cutoff_date"]))
    segments = load_segment_model(run_id)
    test = data[split == "test"]
    X = add_segment_features(test, segments) if segments is not None else test
    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    p = churn_probability(model, load_calibrator(run_id), X, info["features"])
    rows = test[KEY + ["cutoff_date", "event_60d", "days_since_last_bet"]].assign(p=np.asarray(p))
    return rows, set(test["cutoff_date"]), set(data.loc[split != "test", "cutoff_date"])


def capture(value: pd.Series, y: pd.Series, score: pd.Series) -> float:
    realised = value * y
    top = (value * score).nlargest(min(CONTACTS, len(value))).index
    return float(realised[top].sum() / realised.sum())


def cutoff_row(brand: int, cutoff: pd.Timestamp, d: pd.DataFrame) -> dict:
    snap = signal_snapshots.snapshots([cutoff.date()], brand, build_missing=False)[KEY + ["ggr_eur_l90d"]]
    d = d.merge(snap, on=KEY, how="left")
    value = (d["ggr_eur_l90d"].clip(lower=0) / 90 * 30).fillna(0)
    y60, y90 = d["y60"].astype(int), d["y90"].astype(int)
    n_top = int(round(len(d) * 0.1))
    row = {"brand": brand, "cutoff": cutoff.date(), "players": len(d), "churn_60d": y60.mean(), "churn_90d": y90.mean(),
           "back_61_90": 1 - y90[y60 == 1].mean(), "spearman": d[["p60", "p90"]].corr("spearman").iloc[0, 1],
           "top_decile_overlap": len(set(d.nlargest(n_top, "p60").index) & set(d.nlargest(n_top, "p90").index)) / n_top,
           "top_contacts_overlap": len(set((value * d["p60"]).nlargest(CONTACTS).index)
                                       & set((value * d["p90"]).nlargest(CONTACTS).index)) / min(CONTACTS, len(d))}
    for s, name in (("p60", "model60"), ("p90", "model90"), ("days_since_last_bet", "recency")):
        for lab, y in (("y60", y60), ("y90", y90)):
            row[f"auc_{name}_{lab}"] = roc_auc_score(y, d[s])
            row[f"precision_{name}_{lab}"] = top_decile_precision(y, d[s])
            if name != "recency":
                row[f"ggr_{name}_{lab}"] = capture(value, y, d[s])
    row |= {"mean_p60": d["p60"].mean(), "mean_p90": d["p90"].mean()}
    return row


def brand_rows(brand: int, client) -> list[dict]:
    a, test_a, fit_a = scored(latest_version(client, f"churn_lightgbm_classifier_brand{brand}"))
    b, test_b, fit_b = scored(latest_version(client, f"churn_lightgbm_classifier_brand{brand}_churn{OTHER_DAYS}d"))
    common = sorted((test_a & test_b) - fit_a - fit_b)
    if not common:
        raise ValueError(f"brand {brand}: no test cutoff out of sample for both models")
    rows = []
    for c in common:
        d = (a[a["cutoff_date"] == c].rename(columns={"p": "p60", "event_60d": "y60"})
             .merge(b[b["cutoff_date"] == c][KEY + ["p", "event_60d"]].rename(columns={"p": "p90", "event_60d": "y90"}),
                    on=KEY))
        rows.append(cutoff_row(brand, c, d))
        print(f"brand {brand} {c.date()}: AUC on the 90-day label {rows[-1]['auc_model60_y90']:.4f} (60-day model) "
              f"vs {rows[-1]['auc_model90_y90']:.4f} (90-day model)", flush=True)
    return rows


def write(t: pd.DataFrame) -> Path:
    lines = ["# Churn Threshold: 60 against 90 Days", "",
             f"For each brand, the official LightGBM (60 days) and the LightGBM of a {OTHER_DAYS}-day sensitivity run "
             "score the same players on the test cutoffs that are out of sample for both, each with its own features, "
             "caps and calibrator. Both are judged against both labels. Generated by "
             "`src/models/threshold_comparison.py` (`make threshold-comparison`).", "",
             "## Ranking: AUC", "",
             "| Brand | Cutoff | Players | Churn 60d | Churn 90d | 60d churners back in days 61 to 90 | "
             "60d model on 60d label | 90d model on 60d label | 60d model on 90d label | 90d model on 90d label | "
             "Recency on 90d label |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples(index=False):
        lines.append(f"| {r.brand} | {r.cutoff} | {r.players:,} | {100 * r.churn_60d:.1f}% | {100 * r.churn_90d:.1f}% | "
                     f"{100 * r.back_61_90:.1f}% | {r.auc_model60_y60:.3f} | {r.auc_model90_y60:.3f} | "
                     f"{r.auc_model60_y90:.3f} | {r.auc_model90_y90:.3f} | {r.auc_recency_y90:.3f} |")
    lines += ["", "## Who is picked, and the value reached", "",
              f"Overlap of the riskiest 10% and of the {CONTACTS:,} contacts ranked by value at risk; GGR at risk reached "
              f"by those {CONTACTS:,} contacts, under each label.", "",
              "| Brand | Cutoff | Rank correlation | Top-10% overlap | Contacts overlap | GGR 60d label: 60d / 90d model | "
              "GGR 90d label: 60d / 90d model | Mean p 60d model (rate) | Mean p 90d model (rate) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples(index=False):
        lines.append(f"| {r.brand} | {r.cutoff} | {r.spearman:.3f} | {100 * r.top_decile_overlap:.0f}% | "
                     f"{100 * r.top_contacts_overlap:.0f}% | {100 * r.ggr_model60_y60:.1f}% / {100 * r.ggr_model90_y60:.1f}% | "
                     f"{100 * r.ggr_model60_y90:.1f}% / {100 * r.ggr_model90_y90:.1f}% | "
                     f"{r.mean_p60:.3f} ({r.churn_60d:.3f}) | {r.mean_p90:.3f} ({r.churn_90d:.3f}) |")
    REPORT.write_text("\n".join(lines) + "\n")
    return REPORT


def run(brands: list[int] = BRANDS) -> Path:
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    client = mlflow.MlflowClient()
    table = pd.DataFrame([r for b in brands for r in brand_rows(b, client)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    return write(table)


if __name__ == "__main__":
    print(f"saved {run([int(b) for b in sys.argv[1:]] or BRANDS)}")
