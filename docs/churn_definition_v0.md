
# Baseline Churn Definition v0

## 1. Executive Summary and Core Definition

This document defines **Player Churn** for the baseline scoring pipeline.

Churn is an **inactivity threshold** on `bet` events: a player has churned after `N` days with no `bet`. We use `bet` as the activity signal, not `player` (profile changes, not play) and not `transaction`/payments (deposits/withdrawals, much rarer than bets, and a player can keep playing for days on existing balance without a new deposit).

We test three thresholds, `N = 14, 30, 60` days, side by side:

- **14 days**: an early signal. Easy to confirm, but noisy: many "churned" players under this rule still come back.
- **30 days**: The data shows about 15% of players still return after 30 days of silence.
- **60 days**: the safest definition. Under 4% return after this. But with only ~104 days of history, this threshold has the most missing information (64.4% of players cannot be confirmed yet either way).


## 2. Session Reconstruction Methodology

There is no `session` table in `10-landing` (checked for both operators). We rebuild player activity from `bet.dateTime` in two ways:

**a) First/last bet per player**: one row per player: `first_bet`, `last_bet`, `n_bets`. A simple `GROUP BY partyId` in DuckDB, run directly on the S3 files. Used for the Kaplan-Meier labels.

**b) Active days per player**: one row per (player, day with at least one bet). Used for the return-after-dormancy analysis.

Both cover the full history (`2026-06-18` to `2026-09-29`, 104 days, both operators, 385,328 players). Both are slow to compute, not because of the math, but because `bet` is stored as many small hourly files (167 files per day per operator), a full scan opens about 34,000 files. Measured time: 36 minutes for (a), 14 minutes for (b). Both results are saved to local files after the first run, so this cost is paid only once.


## 3. Empirical Justification & Survival Analysis

### Return-after-Dormancy Analysis

For each threshold: of all the times a player went quiet for at least `N` days, how often did they come back later?

First result:

| Threshold | Returns | Still dormant (no return) | Return rate (raw) |
|---|---|---|---|
| 14d | 109,071 | 276,789 | 28.3% |
| 30d | 31,019 | 222,525 | 12.2% |
| 60d | 5,509 | 139,733 | 3.8% |

**This raw comparison is not fair.** A player only counts as "still dormant" at 60 days if their last bet was early enough in the window to allow 60 full days to pass. With 104 days of history, that means the 60-day number only uses players from the first 44 days. The 14-day number uses almost the whole window. So each threshold looks at a different group of players, not just a different silence length, part of the drop could come from that, not from real behavior.

**Fixed comparison**: use the same player group for all three thresholds (only players whose last bet allows a full 60-day check):

| Threshold | Returns | Still dormant (no return) | Return rate (fair) |
|---|---|---|---|
| 14d | 67,691 | 139,733 | **32.6%** |
| 30d | 24,315 | 139,733 | **14.8%** |
| 60d | 5,509 | 139,733 | 3.8% |

Fixing this does not weaken the finding, it makes it stronger: the drop from 14 to 60 days is bigger this way (32.6% $\rightarrow$ 3.8%) than in the raw version (28.3% $\rightarrow$ 3.8%). 60 days stays the same because its own rule already used this same fair group.

**In short**: 60 days (3.8% return) is the most reliable "this player is really gone" signal we have. 30 days (14.8% return, fair comparison) still lets a real share of players come back, treat a 30-day churn label as a warning sign, not a final answer.

### Survival Curves (Kaplan-Meier)

We fit one Kaplan-Meier curve per threshold with `lifelines`. `duration_days = last_bet - first_bet`. `event = 1` (churn confirmed) if `observation_end - last_bet >= N`; `event = 0` (**censored**) if not, we simply do not know yet if that player will return.

Share of players still censored (unconfirmed), out of 385,328:

| Threshold | Censored | Confirmed churns |
|---|---|---|
| 14d | 29.2% | 272,982 |
| 30d | 42.9% | 220,053 |
| 60d | 64.4% | 137,267 |

![Kaplan-Meier Survival Curves](../data/03_output/survival_curves.png)

*Figure 1: Kaplan-Meier survival function $S(t)$ showing the probability of remaining inactive across duration $t$ (days), highlighting the drop in return probability at 14, 30, and 60 days.*

Median survival: 14d $\rightarrow$ 0.8 days; 30d $\rightarrow$ 3.9 days; 60d $\rightarrow$ not reached (over half the group is still censored, so the curve never gets there).

**Important note**: these medians look very low because half of all players only ever placed bets on one day (median `duration_days` = 0.13 days across everyone). Most of them count as "churned" even at 14 days, so the curve drops fast right at the start. This is a true picture of the whole player base, but it does not answer "how long does an *active* player usually stay active", for that, a next step would look only at players with more than one bet, or activity on more than one day. Not done in this first version.

## 4. Operational Label Logic

For a player, with a chosen threshold `N` (days):

```
duration_days        = last_bet − first_bet
days_since_last_bet  = observation_end − last_bet
event_Nd              = 1  if days_since_last_bet ≥ N   (churn confirmed)
                       = 0  otherwise                    (censored — not known yet)
```

`event_Nd` is the churn label for threshold `N`. The same `duration_days` is used for all three curves, only `event_Nd` changes.

For the return-after-dormancy check, the same `N`-day rule applies to each player's active-day list, with the fair-comparison group used whenever we compare more than one threshold.

**Known limits, not fixed in this version**:

- **Left truncation**: history starts `2026-06-18`. A player active before that has an unknown true first bet, `first_bet` here means "first bet we can see", not "first bet ever".
- **Short history vs. large thresholds**: with only ~104 days of data, the 60-day threshold has little room. This improves on its own as more history builds up.
- **One-time players dominate the survival medians** do not read the median survival time as "typical active player lifetime" without that context.

## 5. Threshold Decision: Why 60 Days

We choose **60 days** as the churn threshold for the training label, not 30 days.

30 days is the common default, but it has a real cost: about 15% of players labeled "churned" at 30 days later return. Training on that means the ground truth is wrong about 1 in 7 times. 60 days brings that error down to 3.8%.


**Decision v0**: `event_60d` is the primary churn label for this baseline. `event_14d` and `event_30d` stay in the notebook as earlier warning signals and for comparison, not as the training target.
