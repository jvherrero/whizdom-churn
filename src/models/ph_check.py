"""Proportional hazards check of the registered Cox PH models (review item P1 7): docs/cox_ph_check.md.

    .venv/bin/python src/models/ph_check.py
    .venv/bin/python src/models/ph_check.py --run-ids 64=<cox run id> 14=<cox run id>

Cox PH assumes that each covariate multiplies the hazard by the same factor at every time. For each
brand, on the training rows of its latest Cox run:

    test         Schoenfeld residual test (lifelines, rank of time). With tens of thousands of events
                 almost any deviation is significant, so the p-value alone says little.
    beta(t)      the effect of each covariate in time windows after the cutoff: beta plus the mean
                 scaled Schoenfeld residual of the events in the window (Grambsch and Therneau, 1994).
    verdict      stable / changes size / sign changes, from beta(t).

Aggregates only. Local files and the local MLflow store, no S3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from lifelines.statistics import proportional_hazard_test
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / p) for p in ("src/evaluation", "src/models", "src/features", "src")]
from evaluate import run_data  # noqa: E402
from train import DURATION, EVENT, cox_input, run_info  # noqa: E402

BRANDS = [14, 23, 64, 72, 73]
REPORT = PROJECT_ROOT / "docs/cox_ph_check.md"
OUT_DIR = PROJECT_ROOT / "data/03_output/cox_ph_check"
# Windows of the event day after the cutoff (day 0 = churn starts on the cutoff itself).
WINDOWS = [(-1, 0, "day 0"), (0, 7, "1 to 7"), (7, 30, "8 to 30"), (30, 90, "31 to 90"), (90, 10_000, "over 90")]
NEGLIGIBLE = 0.05      # |beta(t)| below this in every window: no real effect to keep proportional
SIZE_RATIO = 1.5       # largest over smallest |beta(t)| above this: the effect changes size


def latest_cox_run(client, brand: int) -> str:
    """The latest Cox run of a brand that is not a backtest or robustness run."""
    experiments = [e.experiment_id for e in client.search_experiments()]
    for r in client.search_runs(experiments, f"attributes.run_name LIKE 'cox_ph_brand{brand}_%'",
                                order_by=["attributes.start_time DESC"], max_results=50):
        if "backtest_id" not in r.data.tags and "scenario" not in r.data.tags:
            return r.info.run_id
    raise LookupError(f"no Cox run for brand {brand}")


def beta_by_window(model, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """beta(t) per covariate and window, events per window, and the Spearman correlation of the
    scaled Schoenfeld residuals with the rank of the event time."""
    resid = model.compute_residuals(frame, "scaled_schoenfeld")
    t = frame.loc[resid.index, DURATION]
    window = pd.cut(t, [w[0] for w in WINDOWS] + [WINDOWS[-1][1]], labels=[w[2] for w in WINDOWS])
    beta_t = (resid.groupby(window, observed=False).mean() + model.params_).T
    beta_t.columns = [str(c) for c in beta_t.columns]
    events = window.value_counts().reindex([w[2] for w in WINDOWS]).fillna(0).astype(int)
    rho = pd.Series({c: spearmanr(resid[c], t.rank()).statistic for c in resid.columns})
    return beta_t, events, rho


def verdict(row: pd.Series) -> str:
    b = row.dropna()
    if (b.abs() < NEGLIGIBLE).all():
        return "negligible"
    if (b > NEGLIGIBLE).any() and (b < -NEGLIGIBLE).any():
        return "sign changes"
    big = b[b.abs() >= NEGLIGIBLE].abs()
    return "changes size" if big.max() / big.min() > SIZE_RATIO else "stable"


def check(brand: int, run_id: str, client) -> dict:
    model = joblib.load(client.download_artifacts(run_id, "model/model.joblib"))
    data, split = run_data(run_info(run_id))
    train = data[split == "train"]
    frame = cox_input(model, train).assign(**{DURATION: train[DURATION].to_numpy(), EVENT: train[EVENT].to_numpy()})
    test = proportional_hazard_test(model, frame, time_transform="rank").summary
    beta_t, events, rho = beta_by_window(model, frame)
    table = pd.DataFrame({"beta": model.params_, "rho": rho, "chi2": test["test_statistic"], "p": test["p"]})
    table = table.join(beta_t).assign(verdict=beta_t.apply(verdict, axis=1))
    return {"brand": brand, "run_name": client.get_run(run_id).info.run_name, "rows": len(frame),
            "events": int(frame[EVENT].sum()), "events_by_window": events, "table": table}


def plot(results: list[dict], path: Path) -> None:
    labels = [w[2] for w in WINDOWS]
    covariates = sorted({c for r in results for c in r["table"].index})
    colour = dict(zip(covariates, plt.get_cmap("tab20").colors))   # the same colour in every panel
    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.4), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), results):
        for cov, row in r["table"].iterrows():
            ax.plot(labels, np.exp(row[labels].astype(float)), marker="o", ms=3, lw=1.4, color=colour[cov], label=cov)
        ax.axhline(1, color="grey", lw=0.8, ls="--")
        ax.set_yscale("log"); ax.set_title(f"Brand {r['brand']}", fontsize=10)
        ax.set_yticks([0.4, 0.6, 0.8, 1, 1.25]); ax.set_yticklabels(["0.4", "0.6", "0.8", "1", "1.25"])
        ax.minorticks_off(); ax.tick_params(axis="x", rotation=45, labelsize=8); ax.grid(axis="y", alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("hazard ratio per unit, exp(beta(t))")
    fig.supxlabel("days from the cutoff to the start of the churn", fontsize=9)
    handles = [plt.Line2D([], [], color=colour[c], marker="o", ms=3, lw=1.4) for c in covariates]
    fig.legend(handles, covariates, loc="lower center", ncol=min(len(covariates), 6), fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(path, dpi=130); plt.close(fig)


def write(results: list[dict], figure: Path) -> Path:
    labels = [w[2] for w in WINDOWS]
    lines = ["# Cox PH: Proportional Hazards Check (review item P1 7)", "",
             "Cox PH assumes that each covariate multiplies the hazard of churn by the same factor at every "
             "time after the cutoff. I check it on the training rows of the registered Cox model of each brand. "
             "Generated by `src/models/ph_check.py` (`make ph-check`).", "",
             "- **Test**: Schoenfeld residual test (lifelines `proportional_hazard_test`, rank of time). With tens of "
             "thousands of events almost any deviation is significant, so I read the size of the change, not the p-value.",
             "- **beta(t)**: the effect of the covariate on the events of each time window, beta plus the mean scaled "
             "Schoenfeld residual of those events (Grambsch and Therneau, 1994). Day 0 means that the churn starts on "
             "the cutoff itself (the `event_60d` = 1 players).",
             f"- **Verdict**: *negligible* if |beta(t)| < {NEGLIGIBLE} in every window; *sign changes* if beta(t) is "
             f"above {NEGLIGIBLE} in one window and below -{NEGLIGIBLE} in another; *changes size* if the largest "
             f"|beta(t)| is more than {SIZE_RATIO} times the smallest; *stable* otherwise.", "",
             f"![beta(t) by window](../{figure.relative_to(PROJECT_ROOT).as_posix()})", ""]
    for r in results:
        ev = r["events_by_window"]
        lines += [f"## Brand {r['brand']}", "",
                  f"Run `{r['run_name']}`: {r['rows']:,} training rows, {r['events']:,} events "
                  "(" + ", ".join(f"{k}: {v:,}" for k, v in ev.items()) + ").", "",
                  "| covariate | beta | " + " | ".join(f"beta(t) {w}" for w in labels) + " | rho | chi2 | p | verdict |",
                  "|---|---|" + "---|" * len(labels) + "---|---|---|---|"]
        for cov, row in r["table"].iterrows():
            lines.append(f"| `{cov}` | {row['beta']:.3f} | " + " | ".join(f"{row[w]:.3f}" for w in labels)
                         + f" | {row['rho']:.3f} | {row['chi2']:.0f} | {row['p']:.1e} | {row['verdict']} |")
        lines.append("")
    REPORT.write_text("\n".join(lines) + "\n")
    return REPORT


def run(run_ids: dict[int, str] | None = None) -> Path:
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    client = mlflow.MlflowClient()
    results = []
    for b in BRANDS:
        run_id = (run_ids or {}).get(b) or latest_cox_run(client, b)
        results.append(check(b, run_id, client))
        print(f"brand {b}: {results[-1]['table']['verdict'].value_counts().to_dict()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = OUT_DIR / "beta_by_window.png"
    plot(results, figure)
    pd.concat([r["table"].assign(brand=r["brand"]) for r in results]).to_csv(OUT_DIR / "beta_by_window.csv")
    return write(results, figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proportional hazards check of the registered Cox PH models.")
    parser.add_argument("--run-ids", nargs="*", default=[], help="brand=cox_run_id, default: the latest Cox run")
    args = parser.parse_args()
    print(f"saved {run({int(k): v for k, v in (a.split('=') for a in args.run_ids)})}")
