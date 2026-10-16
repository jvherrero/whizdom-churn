# Decision: Top-Decile Backtest Condition

Date: 2026-10-09, updated on 2026-10-16. Applies to the backtest pass conditions of the delivery plan (T10), in `configs/backtest.yaml`. Evidence: the backtests of the five brands with the official active day (14, 23 and 64: `backtest_brand{b}_2026-10-06_1792074259`; 72: `..._1792078504`; 73: `..._1792081564`; 11 monthly test cutoffs each, files `data/03_output/backtest/*_metrics.csv`). The first version of this note used brands 14, 64 and 73 with the old active day (a bet from any wallet), in `eda/09_top_decile_condition.ipynb`; the update gives the same decision.

**Decision.** The KPI stays the same (the mean top-decile precision of the model divided by the recency rule's, over the backtest cutoffs), and I propose that its threshold goes from 1.5x to **1.10x**.

**Status: proposed, not frozen yet** (section 5).

## 1. Problem

- **1.5x cannot be met.** Precision cannot be above 1, so the ratio is at most 1 / recency precision. With a 60-day churn rate of 17% to 66% by month, the players silent for weeks are mostly churners, and recency alone is 57% to 78% precise. The most any model can reach on the mean is 1.27x to 1.75x by brand, and in the month with the most churn it falls to 1.15x to 1.40x. In brands 72 and 73 not even a perfect model passes 1.5x, and in the other three the model is far from it (1.28x to 1.39x).
- **The ceiling moves with the season.** When churn rises, recency becomes more precise and the ceiling falls. So the threshold must leave room under the ceiling of the worst month.
- **Option set aside.** Measuring on the odds scale (p / (1 - p), churners found per wasted contact) removes the ceiling, but it changes the KPI. I keep the KPI and lower the threshold.

## 2. Which Value

For each threshold: the brands whose model passes on the mean, the brands where it passes with 95% confidence (the lower end of the 95% interval is at or above the threshold; circular block bootstrap of 3 consecutive cutoffs, 2,000 resamples, as in the backtest reports), and the mean share of the possible gain over recency it asks for (from 1x to the ceiling of each brand).

| Threshold | Brands passing | Passing with 95% confidence | Share of the possible gain asked |
|---|---|---|---|
| 1.05 | 5 of 5 | 5 of 5 | 11% |
| **1.10** | **5 of 5** | **5 of 5** | **23%** |
| 1.15 | 5 of 5 | 4 of 5 | 34% |
| 1.20 | 4 of 5 | 4 of 5 | 45% |
| 1.25 | 4 of 5 | 2 of 5 | 56% |

- **1.10 is the strictest round value the model meets with 95% confidence in every brand.** At 1.15 brand 72 is not robust (lower end just under 1.15), and at 1.20 it fails (1.19).
- **It holds month by month.** The model is at or above 1.10x in all 55 brand-months; the lowest month is brand 72 in December 2025 (1.11x).
- **It separates what adds value from what does not.** Cox passes (1.19x to 1.38x); the platform's `churn_score` fails (0.31x to 0.60x).
- **Limit.** The simple model (a logistic regression on recency, frequency and tenure) reaches about the same ratio as LightGBM (1.18x to 1.40x). The KPI shows that the model beats recency, not that it beats simple models; no value of it can.

## 3. Result

| Brand | Recency | Model | Ratio | 95% interval | Ceiling (mean) | Cutoffs at or above 1.10 | Lowest month | Verdict |
|---|---|---|---|---|---|---|---|---|
| 14 | 0.642 | 0.869 | 1.35 | 1.30 to 1.41 | 1.56 | 100% | 1.22 (Nov 2025) | pass |
| 23 | 0.572 | 0.796 | 1.39 | 1.32 to 1.46 | 1.75 | 100% | 1.25 (Dec 2025) | pass |
| 64 | 0.645 | 0.829 | 1.28 | 1.23 to 1.35 | 1.55 | 100% | 1.16 (Oct 2025) | pass |
| 72 | 0.784 | 0.935 | 1.19 | 1.15 to 1.23 | 1.27 | 100% | 1.11 (Dec 2025) | pass |
| 73 | 0.730 | 0.914 | 1.25 | 1.20 to 1.29 | 1.37 | 100% | 1.15 (Dec 2025) | pass |

As in the plan, the condition is on the mean over the cutoffs. With the official active day the ratios are higher than in the first version (brand 14: 1.35 against 1.19), because the players who only play with playable bonus are no longer in the population and recency is less precise without them.

## 4. What Changes

- `configs/backtest.yaml`: `top_decile_precision_vs_recency: 1.10` (was 1.5).
- The backtest report shows the ratio with its 95% interval; the model card shows both precisions, the ratio and why the plan's 1.5x was lowered.
- `make backtest-report ID=...` rewrites the report of a finished backtest with the current conditions, without training again.

## 5. Freezing the Threshold

I chose 1.10x after seeing the backtests above, so these backtests show that it is reachable, but they cannot confirm it. To be a fair test, the threshold must be fixed before any result of a month the models have not seen is read.

- **First unseen month.** The backtests run on 6 October 2026 cover the test months October 2025 to August 2026. The first month outside them is September 2026, whose 60-day label closes on 31 October 2026.
- **Rule.** 1.10x is approved before 31 October 2026 and before any backtest with a run date on or after that day is read. From then on it does not change for that evaluation. If a later evaluation shows a reason to change it, the new value applies only to the evaluation after it.
- **What does not count as unseen.** The backtests run again after the active days rebuild use the same test months as before (October 2025 to August 2026). They check the rebuild; they do not test the threshold.
- **Where it is fixed.** `configs/backtest.yaml`, `pass_conditions.top_decile_precision_vs_recency`. The git commit that freezes it is the record.

| Field | Value |
|---|---|
| Proposed threshold | 1.10x |
| Proposed on | 2026-10-09 |
| Approved by | (pending) |
| Approved on | (pending, before 2026-10-31) |
| First evaluation it applies to | test month 2026-09-01 and later |

