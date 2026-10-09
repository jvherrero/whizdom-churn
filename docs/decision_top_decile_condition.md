# Decision: Top-Decile Backtest Condition

Date: 2026-10-09. Applies to the backtest pass conditions of the delivery plan (T10), in `configs/backtest.yaml`. Evidence: `eda/09_top_decile_condition.ipynb`, on the backtests of brands 14, 64 and 73 (11 monthly cutoffs each).

**Decision.** The KPI stays the same (the mean top-decile precision of the model divided by the recency rule's, over the backtest cutoffs), and its threshold goes from 1.5x to **1.10x**.

## 1. Problem

- **1.5x cannot be met.** Precision cannot be above 1, so the ratio is at most 1 / recency precision. With a 60-day churn rate of 30% to 70%, the players silent for weeks are almost all churners, and recency alone is 71% to 74% precise. The most any model can reach is 1.35x to 1.40x: not even a perfect model passes.
- **The ceiling moves with the season.** When churn rises, recency becomes more precise and the ceiling falls (down to 1.18x in the month with the most churn). So the threshold must leave room under the ceiling.
- **Option set aside.** Measuring on the odds scale (p / (1 - p), churners found per wasted contact) removes the ceiling, but it changes the KPI. I keep the KPI and lower the threshold.

## 2. Which Value

For each threshold: the brands whose model passes on the mean, the brands where it passes with 95% confidence (the lower end of the bootstrap interval, cutoffs resampled, is above the threshold), and the share of the possible gain over recency it asks for (from 1x to the ceiling).

| Threshold | Brands passing | Passing with 95% confidence | Share of the possible gain asked |
|---|---|---|---|
| 1.05 | 3 of 3 | 3 of 3 | 14% |
| **1.10** | **3 of 3** | **3 of 3** | **27%** |
| 1.15 | 3 of 3 | 2 of 3 | 41% |
| 1.20 | 2 of 3 | 0 of 3 | 54% |
| 1.25 | 0 of 3 | 0 of 3 | 68% |

- **1.10 is the strictest round value the model meets with 95% confidence in every brand.** At 1.15 brand 14 is not robust, and at 1.20 it fails (1.19).
- **It keeps the plan's intent.** At today's precisions, 1.10x on precision equals 1.5 times the churners found per wasted contact (the plan's 1.5 on the odds scale).
- **It separates what adds value from what does not.** Cox passes (1.17x to 1.24x); the platform's `churn_score` fails (0.53x to 0.72x).
- **Limit.** Simple two-variable rules (recency and active days) also reach about 1.13x to 1.22x in the top decile. The KPI shows the model beats recency, not that it beats simple rules; no value of it can.

## 3. Result

| Brand | Recency | Model | Ratio | 95% interval | Cutoffs at or above 1.10 | Verdict |
|---|---|---|---|---|---|---|
| 14 | 0.737 | 0.875 | 1.19 | 1.13 to 1.25 | 82% | pass |
| 64 | 0.713 | 0.861 | 1.21 | 1.17 to 1.26 | 100% | pass |
| 73 | 0.740 | 0.919 | 1.24 | 1.19 to 1.29 | 100% | pass |

Intervals from the notebook (2,000 resamples); the backtest report uses 200, so its intervals differ a little. The two cutoffs under 1.10 are brand 14 in November and December 2025 (1.04 and 1.06), the months with the most churn. As in the plan, the condition is on the mean over the cutoffs.

## 4. What Changes

- `configs/backtest.yaml`: `top_decile_precision_vs_recency: 1.10` (was 1.5).
- The backtest report shows the ratio with its 95% interval; the model card shows both precisions, the ratio and why the plan's 1.5x was lowered.
- `make backtest-report ID=...` rewrites the report of a finished backtest with the current conditions, without training again.
