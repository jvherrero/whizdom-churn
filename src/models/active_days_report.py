"""Review item P0 2: are the active_days_* features right, and what changes when they are rebuilt?

    .venv/bin/python src/models/active_days_report.py                       # part 1 only
    .venv/bin/python src/models/active_days_report.py --after 64=backtest_brand64_...   # adds part 2

Part 1 compares, for every brand and cutoff of the last backtest, the active days the models read
from the signals table (kept in the backtest's feature files) with the canonical count from the
activity cache (src/features/build_features.active_days). Part 2 compares the backtest metrics
before and after the rebuild: AUC, ECE, top-decile precision and the campaign replay.
Everything is read from local files; no S3. Output: docs/active_days.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
import daily_cache  # noqa: E402
from build_features import ACTIVE_DAY_WINDOWS, KEY, active_days  # noqa: E402

BRANDS = [14, 23, 64, 72, 73]
BACKTEST_DIR = PROJECT_ROOT / "data/02_intermediate/backtest"
OUTPUT_DIR = PROJECT_ROOT / "data/03_output/backtest"
REPORT = PROJECT_ROOT / "docs/active_days.md"


# The backtests before the rebuild (each new backtest rewrites docs/backtest_v0_brand{id}.md).
BEFORE = {14: 'backtest_brand14_2026-10-06_1791539629', 23: 'backtest_brand23_2026-10-06_1791554664', 64: 'backtest_brand64_2026-10-06_1791536375', 72: 'backtest_brand72_2026-10-06_1791890715', 73: 'backtest_brand73_2026-10-06_1791868851'}


def backtest_id(brand: int) -> str:
    """The backtest before the rebuild of the active days."""
    return BEFORE[brand]


def source_values(bt_id: str) -> pd.DataFrame:
    """Active days as the models read them (signals table), one row per cutoff and player."""
    cols = ["cutoff_date", *KEY, *ACTIVE_DAY_WINDOWS]
    frames = [pd.read_parquet(f, columns=cols) for f in sorted((BACKTEST_DIR / bt_id).glob("test*.parquet"))]
    rows = pd.concat(frames).drop_duplicates(["cutoff_date", *KEY])
    rows[list(ACTIVE_DAY_WINDOWS)] = np.expm1(rows[list(ACTIVE_DAY_WINDOWS)]).round()   # undo the signed log1p
    return rows


def agreement(brand: int) -> pd.DataFrame:
    src = source_values(backtest_id(brand))
    act = daily_cache.read("activity", brand, columns="tenant_id, player_id, day", active_only=True)
    act["day"] = pd.to_datetime(act["day"])
    out = []
    for c, part in src.groupby("cutoff_date"):
        canon = active_days(act, pd.Timestamp(c).date()).reindex(pd.MultiIndex.from_frame(part[KEY])).fillna(0)
        for name in ACTIVE_DAY_WINDOWS:
            s, k = part[name].to_numpy(), canon[name].to_numpy()
            out.append({"brand": brand, "cutoff": str(c)[:10], "feature": name, "players": len(part),
                        "equal": float(np.mean(s == k)), "source_higher": float(np.mean(s > k)),
                        "source_lower": float(np.mean(s < k)), "mean_diff": float(np.mean(s - k))})
    return pd.DataFrame(out)


def backtest_summary(bt_id: str, model: str = "lgbm_optimised") -> dict:
    """Mean over the cutoffs of one backtest, for the model and against the recency rule."""
    m = pd.read_csv(OUTPUT_DIR / f"{bt_id}_metrics.csv")
    r = pd.read_csv(OUTPUT_DIR / f"{bt_id}_replay.csv")
    lgbm = m[m["comparator"] == model].set_index("cutoff")
    rec = m[m["comparator"] == "recency"].set_index("cutoff")
    return {"auc": lgbm["auc"].mean(), "auc_min": lgbm["auc"].min(), "ece_max": lgbm["ece"].max(),
            "months_ece_over_0.08": int((lgbm["ece"] > 0.08).sum()),
            "top_decile_precision": lgbm["top_decile_precision"].mean(),
            "top_decile_ratio": (lgbm["top_decile_precision"] / rec["top_decile_precision"]).mean(),
            "ggr_at_risk_captured": r.loc[r["ranking"] == model, "ggr_at_risk_captured"].mean()}


def write(agree: pd.DataFrame, after: dict[int, str]) -> Path:
    lines = ["# Active days: definition, check against the source and rebuild (review item P0 2)", "",
             "## 1. Definition", "",
             "An **active day** is a calendar day (the `event_date` of `gld_player_gaming_daily`) on which the "
             "player staked a non-zero amount from real money or released bonus funds; stakes only from playable "
             "bonus funds do not count (measure M053 of `gld-schemas`). It is "
             "the same event as the churn label: churn is 60 days without an active day. The canonical source is "
             "the daily activity cache (`daily_cache` \"activity\" read with `active_only=True`: one row per player "
             "and day with `wagered_real + wagered_released_bonus > 0` in `gld_player_gaming_daily`), whose grain was validated with Great Expectations and whose bet counts "
             "match the raw Kafka transactions (`docs/data_quality/data_quality.pdf`).", "",
             "| Feature | Window (days before the cutoff) |", "|---|---|"]
    for name, (lo, hi) in ACTIVE_DAY_WINDOWS.items():
        lines.append(f"| `{name}` | {hi} to {lo - 1} (the cutoff day is day 0) |")
    lines += ["", "## 2. The signals table against the canonical count", "",
              "Share of players with the same value, per brand and feature, over every cutoff of the last backtest. "
              "When the source differs, it is almost always higher: it counts the days with stakes only from playable bonus, "
              "which are not active days, and some days without a bet. It is lower in at most 0.01% of the players.", "",
              "| Brand | Feature | Cutoffs | Player-cutoffs | Equal | Source higher | Source lower | Worst cutoff (equal) |",
              "|---|---|---|---|---|---|---|---|"]
    for (brand, feature), g in agree.groupby(["brand", "feature"]):
        w = (g["equal"] * g["players"]).sum() / g["players"].sum()
        hi = (g["source_higher"] * g["players"]).sum() / g["players"].sum()
        lo = (g["source_lower"] * g["players"]).sum() / g["players"].sum()
        worst = g.loc[g["equal"].idxmin()]
        lines.append(f"| {brand} | `{feature}` | {len(g)} | {g['players'].sum():,} | {100 * w:.1f}% | {100 * hi:.1f}% | "
                     f"{100 * lo:.2f}% | {worst['cutoff']} ({100 * worst['equal']:.1f}%) |")
    lines += ["", "**Decision.** The source cannot be validated independently: its definition of an active day is "
              "not documented and in some months it counts days with no bet. The features are rebuilt from the "
              "canonical activity (`src/features/build_features.py`, `active_days`), and the models and backtests "
              "are run again.", ""]
    lines += ["## 3. Backtest before and after the rebuild", "",
              "Before: the source features and a bet from any wallet as activity. After: the features rebuilt from the "
              "official active days, which also change the population and the label, so the two rows of a brand are "
              "not scored on exactly the same players.", ""]
    if not after:
        lines.append("Pending: run `make backtest` for each brand and then this script with `--after`.")
    else:
        lines += ["| Brand | Version | Mean AUC | Min AUC | Max ECE | Months ECE > 0.08 | Top-decile precision | "
                  "Ratio vs recency | GGR at risk captured |", "|---|---|---|---|---|---|---|---|---|"]
        for brand, new_id in after.items():
            for label, bt in (("before", backtest_id(brand)), ("after", new_id)):
                s = backtest_summary(bt)
                lines.append(f"| {brand} | {label} | {s['auc']:.3f} | {s['auc_min']:.3f} | {s['ece_max']:.3f} | "
                             f"{s['months_ece_over_0.08']} | {s['top_decile_precision']:.3f} | "
                             f"{s['top_decile_ratio']:.2f} | {100 * s['ggr_at_risk_captured']:.1f}% |")
    REPORT.write_text("\n".join(lines) + "\n")
    return REPORT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active days: check against the source, before and after.")
    parser.add_argument("--after", nargs="*", default=[], help="brand=backtest_id of the backtests after the rebuild")
    args = parser.parse_args()
    agree = pd.concat([agreement(b) for b in BRANDS], ignore_index=True)
    after = {int(k): v for k, v in (a.split("=") for a in args.after)}
    print(f"saved {write(agree, after)}")
