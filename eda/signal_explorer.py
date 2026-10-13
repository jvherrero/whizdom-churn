"""Churn signal explorer (T4): tests every behaviour hypothesis in the same way.

    frame = build_frame(TRAINING_CUTOFFS, brand_id=64, data_end=...)
    table = evaluate_all(frame)

For each signal, on the training cutoffs only:

    lift            churn rate with / without the signal (flags), or top / bottom quartile
    hazard ratio    univariate Cox PH with a 95% interval
    AUC             AUC of the signal alone for the 60-day label
    stability       lift at each cutoff; a signal whose effect changes direction is dropped

A signal is promoted when the hazard-ratio interval excludes 1, |AUC - 0.5| >= MIN_AUC_GAP, the
effect never changes direction and the three measures agree. To add a hypothesis, add a column to
the signals and a Signal to SIGNALS.
"""

from __future__ import annotations

import datetime as dt
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
import build_features  # noqa: E402
import churn_labels  # noqa: E402
from build_features import DEPOSIT_WINDOWS, DEPOSITS_VALID_FROM  # noqa: E402,F401  (documented here)

KEY = ["tenant_id", "player_id"]
LABEL = "event_60d"
MIN_AUC_GAP = 0.05
FLIP_TOLERANCE = 0.05           # a lift within 5% of 1 is not a change of direction
MIN_GROUP = 200                 # smallest group size for a lift


@dataclass(frozen=True)
class Signal:
    name: str           # column in the frame
    hypothesis: str     # hypothesis group of the plan
    kind: str           # "continuous" or "flag"
    expected: str       # "+" means more churn, "-" less churn
    definition: str
    reference: bool = False  # only for comparison


SIGNALS = [
    # Deposit frequency
    Signal("n_deposit_days_7d", "Deposit frequency", "continuous", "-", "days with a deposit in the last 7 days"),
    Signal("n_deposit_days_30d", "Deposit frequency", "continuous", "-", "days with a deposit in the last 30 days"),
    Signal("deposit_days_ratio_30_vs_prior30", "Deposit frequency", "continuous", "-",
           "(deposit days, last 30 + 1) / (deposit days, previous 30 + 1)"),
    Signal("deposit_frequency_score", "Deposit frequency", "continuous", "-", "signals table: deposits last 7 vs last 30 days, 0-100"),
    # Deposit recency
    Signal("days_since_last_deposit", "Deposit recency", "continuous", "+", "days since the last deposit (31 = none in 30 days)"),
    Signal("deposited_within_3d", "Deposit recency", "flag", "-", "a deposit in the last 3 days"),
    Signal("deposited_within_7d", "Deposit recency", "flag", "-", "a deposit in the last 7 days"),
    Signal("deposited_within_14d", "Deposit recency", "flag", "-", "a deposit in the last 14 days"),
    Signal("deposit_recency_score", "Deposit recency", "continuous", "-", "signals table: deposit recency, 0-100"),
    # Deposit size trend
    Signal("deposit_size_trend", "Deposit size trend", "continuous", "-",
           "mean deposit per deposit day, last 30 days / previous 90 days"),
    Signal("deposits_30_vs_prior30", "Deposit size trend", "continuous", "-", "deposits EUR last 30 days / previous 30 days"),
    # Heavy loss
    Signal("net_loss_7d", "Heavy loss", "continuous", "+", "player net loss (GGR) in the last 7 days, EUR"),
    Signal("heavy_loss_multiple", "Heavy loss", "continuous", "+", "net loss last 7 days / weekly average of the last 90"),
    Signal("heavy_loss_flag", "Heavy loss", "flag", "+", "net loss last 7 days > 2x the weekly average of the last 90"),
    Signal("rg_loss_chasing_30d", "Heavy loss", "continuous", "+", "signals table: loss chasing, share of stake lost in 30 days"),
    # Big win then withdrawal
    Signal("big_win_then_withdrawal", "Big win then withdrawal", "flag", "+",
           "in the last 30 days, a day with net win > 5x the median daily stake, then a withdrawal within 3 days"),
    Signal("withdrawals_30d_share", "Big win then withdrawal", "continuous", "+", "withdrawals / deposits, last 30 days"),
    # Withdrawal without redeposit
    Signal("withdrawal_no_redeposit", "Withdrawal without redeposit", "flag", "+",
           "a withdrawal in the last 14 days and no deposit after it"),
    # Losing streak
    Signal("losing_streak", "Losing streak", "continuous", "+", "consecutive latest playing days with a net loss"),
    # Session decay
    Signal("session_ratio_7_vs_prior7", "Session decay", "continuous", "-", "(sessions last 7 days + 1) / (previous 7 + 1)"),
    Signal("session_length_ratio", "Session decay", "continuous", "-", "mean session length last 7 days / previous 7"),
    Signal("active_days_7_vs_prior7", "Session decay", "continuous", "-", "(active days last 7 + 1) / (previous 7 + 1)"),
    # Bonus dependence
    Signal("bonus_stake_share_30d", "Bonus dependence", "continuous", "+", "share of the stake from bonus money, last 30 days"),
    Signal("bonus_granted_30d", "Bonus dependence", "continuous", "-", "bonus granted EUR, last 30 days"),
    # Product narrowing
    Signal("games_breadth_30d", "Product narrowing", "continuous", "-", "distinct games, last 30 days"),
    Signal("games_breadth_ratio", "Product narrowing", "continuous", "-", "(distinct games last 30 + 1) / (previous 30 + 1)"),
    # Failed deposits
    Signal("failed_deposits_14d", "Failed deposits", "continuous", "+", "failed deposits in the last 14 days"),
    # Prior dormancy
    Signal("prior_dormancy_spells_14d", "Prior dormancy", "continuous", "+", "returns after 14+ silent days, last 180 days"),
    Signal("prior_dormancy_spells_30d", "Prior dormancy", "continuous", "+", "returns after 30+ silent days, last 180 days"),
    Signal("days_since_last_return", "Prior dormancy", "continuous", "-",
           "days since the last return after 14+ silent days (181 = none in 180 days)"),
    # Tenure
    Signal("tenure_days", "Tenure", "continuous", "-", "signals table: days since registration (not consistent across days)"),
    Signal("days_since_first_bet", "Tenure", "continuous", "-", "days since the first bet seen (activity), capped at 150"),
    # Engagement (signals table composite)
    Signal("engagement_score", "Engagement", "continuous", "-", "signals table: active days, sessions and play time over 30 days"),
    # Bet recency
    Signal("days_since_last_bet", "Bet recency", "continuous", "+", "days since the last bet (activity cache)"),
    # Reference: the platform's current score

    Signal("churn_score", "Reference", "continuous", "+", "platform churn score (the incumbent rule)", reference=True),
]


# Frame

def build_frame(cutoffs: list[dt.date], brand_id: int, data_end: dt.date, verbose: bool = True) -> pd.DataFrame:
    """One row per player and cutoff with all the signals and the churn labels."""
    parts = []
    for c in cutoffs:
        raw = build_features.raw_features(c, brand_id)
        labels = churn_labels.churn_targets(raw[KEY], c, brand_id, data_end).drop(columns="cutoff_date")
        part = raw.merge(labels, on=KEY, how="left")
        parts.append(part)
        if verbose:
            print(f"  {c}: {len(part):,} players, churn {part[LABEL].mean():.1%}, "
                  f"in the signals table {part['tenure_days'].notna().mean():.1%}")
    frame = pd.concat(parts, ignore_index=True)
    frame["player_key"] = frame["tenant_id"].astype(str) + ":" + frame["player_id"].astype(str)
    return frame


# Evaluation

def _signed_log(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


def groups(values: pd.Series, kind: str) -> pd.Series:
    """Group: present/absent for a flag, quartiles Q1 to Q4 otherwise."""
    if kind == "flag":
        return values.map({1: "present", 0: "absent"})
    # Break ties at random (fixed seed), not by row order, which is sorted by cutoff.
    tie = np.random.default_rng(0).random(len(values))
    order = np.lexsort((tie, values.fillna(np.inf).to_numpy()))
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    q = pd.qcut(pd.Series(ranks, index=values.index).where(values.notna()), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return q.astype("object").where(values.notna())


def _lift(y: pd.Series, g: pd.Series, kind: str) -> tuple[float, float, float, float, float]:
    """Lift with its CI, and the churn rates of the high and low groups."""
    hi, lo = ("present", "absent") if kind == "flag" else ("Q4", "Q1")
    a, n1 = y[g == hi].sum(), (g == hi).sum()
    b, n0 = y[g == lo].sum(), (g == lo).sum()
    if min(n1, n0) < MIN_GROUP or a == 0 or b == 0:
        return (np.nan,) * 5
    rr = (a / n1) / (b / n0)
    se = np.sqrt(1 / a - 1 / n1 + 1 / b - 1 / n0)
    return rr, rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se), a / n1, b / n0


def _hazard_ratio(df: pd.DataFrame, s: Signal) -> tuple[float, float, float, float]:
    x = df[s.name].astype(float)
    if s.kind == "continuous":
        x = _signed_log(x)
        x = (x - x.mean()) / x.std() if x.std() > 0 else x * 0
    data = pd.DataFrame({"T": df["duration_days"] + 1, "E": df["event_observed"], "x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter().fit(data, "T", "E")
    row = cph.summary.loc["x"]
    return row["exp(coef)"], row["exp(coef) lower 95%"], row["exp(coef) upper 95%"], row["p"]


def evaluate(frame: pd.DataFrame, s: Signal) -> tuple[dict, pd.DataFrame]:
    """The four measures of one signal and its lift per cutoff."""
    df = frame[frame[LABEL].notna() & frame[s.name].notna()]
    y = df[LABEL].astype(int)
    g = groups(df[s.name], s.kind)
    lift, lo, hi, rate_hi, rate_lo = _lift(y, g, s.kind)
    auc = roc_auc_score(y, df[s.name]) if y.nunique() == 2 and df[s.name].nunique() > 1 else np.nan
    hr = _hazard_ratio(df, s) if len(df) > MIN_GROUP and df[s.name].nunique() > 1 else (np.nan,) * 4

    per_cut = []
    for c, part in df.groupby("cutoff_date"):
        l = _lift(part[LABEL].astype(int), groups(part[s.name], s.kind), s.kind)
        per_cut.append({"signal": s.name, "cutoff_date": c, "lift": l[0]})
    per_cut = pd.DataFrame(per_cut)
    side = np.sign(np.log(lift)) if pd.notna(lift) else 0
    clear = per_cut["lift"].notna() & ((per_cut["lift"] - 1).abs() > FLIP_TOLERANCE)
    flips = int((np.sign(np.log(per_cut.loc[clear, "lift"])) != side).sum()) if side else 0

    observed = "+" if pd.notna(auc) and auc > 0.5 else "-"
    reasons = []
    if not (pd.notna(hr[1]) and (hr[1] > 1 or hr[2] < 1)):
        reasons.append("hazard ratio interval includes 1")
    if not (pd.notna(auc) and abs(auc - 0.5) >= MIN_AUC_GAP):
        reasons.append(f"AUC within {MIN_AUC_GAP} of 0.5")
    if flips:
        reasons.append(f"effect flips side at {flips} cutoff(s)")
    if pd.isna(lift):
        reasons.append(f"a lift group has fewer than {MIN_GROUP} players")
    directions = {np.sign(np.log(v)) for v in (lift, hr[0]) if pd.notna(v)} | ({np.sign(auc - 0.5)} if pd.notna(auc) else set())
    if len(directions - {0}) > 1:
        reasons.append("lift, hazard ratio and AUC disagree on the direction")
    decision = "REFERENCE" if s.reference else ("PROMOTED" if not reasons else "REJECTED")
    return {
        "signal": s.name, "hypothesis": s.hypothesis, "kind": s.kind, "definition": s.definition,
        "rows": len(df), "coverage": len(df) / frame[LABEL].notna().sum(),
        "churn_rate_high": rate_hi, "churn_rate_low": rate_lo, "lift": lift, "lift_low95": lo, "lift_high95": hi,
        "hazard_ratio": hr[0], "hr_low95": hr[1], "hr_high95": hr[2], "hr_p": hr[3], "auc": auc,
        "auc_gap": abs(auc - 0.5) if pd.notna(auc) else np.nan,
        "cutoffs_checked": int(per_cut["lift"].notna().sum()), "flips": flips,
        "expected": s.expected, "observed": observed, "as_expected": observed == s.expected,
        "decision": decision, "reason": "; ".join(reasons) if decision == "REJECTED" else "",
    }, per_cut


def evaluate_all(frame: pd.DataFrame, signals: list[Signal] = SIGNALS) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, cuts = [], []
    for s in signals:
        if s.name not in frame or frame[s.name].notna().sum() == 0:
            continue
        row, per_cut = evaluate(frame, s)
        rows.append(row)
        cuts.append(per_cut)
    table = pd.DataFrame(rows).sort_values("auc_gap", ascending=False).reset_index(drop=True)
    return table, pd.concat(cuts, ignore_index=True)
