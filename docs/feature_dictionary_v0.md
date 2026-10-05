# Feature Dictionary v0.1

This document lists the columns that `build_feature_store()` returns by default, the 30 features actually used for the first baseline model. 



## 0. How to Read This

I call the function like this:

```
df = build_feature_store(cutoff_date="2026-08-28", brand_id=64)
```

```
venv/bin/python src/features/build_features.py --as-of 2026-08-28

```


`cutoff_date` works like "today". Every feature below uses only the 30 days that end on this date (the lookback window). Nothing after it is ever read, and nothing before the window either, so a feature means the same thing at every cutoff. The earliest valid cutoff is 2026-07-18. `brand_id` can be one real brand, or the word `"basel"`, which means "every brand found under both operators".

One row is one player. The population is every player with at least one bet in the 30-day window, for that brand. A player who has not bet for 30 days or more is not in it.

## 1. Columns That Are Not Features

| Column | Meaning |
|---|---|
| `cutoff_date` | The reference date for this row. |
| `operator` | `primus` or `secundus`. |
| `brandId` | The brand this row belongs to. |
| `partyId` | The player id. |
| `label_available_60d` | True only if `cutoff_date` plus 60 days is inside the data I already have. Same value for every row at one cutoff, so this is not a behavior signal. It tells me if the real churn label (`event_60d`) can be confirmed yet or not. It does not tell me if a single new player has enough of their own history for a label to be meaningful, that is a separate problem. |

## 2. Null and Missing Values

I fill nulls the same way across every feature, by type:

- **Counts and sums** (how many times, how much money): if nothing happened, I fill with 0. This is a real value, not a missing one.
- **Recency** (`days_since_last_...`): real values go from 0 to 29. If the event did not happen in the window, I add a flag column (`..._missing`, 1 or 0) and fill the days with 30, which means "30 days or more". This value is the same at every cutoff.
- **`tenure_days`**: `player` only stores change events, so about 2 out of 3 players have no registration date in the window. A registration is itself an event, so these players registered before the window. I fill with 30 ("at least 30 days") and add `tenure_days_missing`, which is one of the 30 features, so the model can tell real and filled values apart.
- **`rtp_last_7d`**: if stake is 0, the ratio has no real value. I add `rtp_last_7d_missing` and fill with the brand's own median rtp for that cutoff.

## 3. Scaling

Before scaling, I cap the 14 EUR features (amounts and trends) with `configs/winsorisation_features_brand{id}.yaml`: p99.5 at the top, and also p0.5 at the bottom for columns that can be negative. I never delete outliers. `eda/anomalies.py --source features` writes this file.

Every count, EUR amount, and trend/drop goes through a signed log: `sign(x) * log(1 + |x|)`. It keeps the sign but shrinks big outliers, so one very large value cannot dominate a model on its own.

I do **not** scale: the `_missing` flags, `already_dormant_7`, and `rtp_last_7d`. Flags are already 0 or 1, and rtp is already a small number.

## 4. Currency

Every money feature is in EUR.

- `bet` always has a real currency value.
- `transaction` (deposits) has no currency column before **2026-08-26** (checked day by day, in both `primus` and `secundus`), a real change in the source, not a random gap. I fill it with the brand's own main bet currency.
- `bonus` has no currency column at all, ever. I always fill it with the brand's own main bet currency. This is an assumption, not a value I ever observed.
- Both fills only checked out as safe for brandId=64, which bets in one currency only (TRY). For a brand that genuinely mixes currencies, filling with "the main currency" would be a guess for the minority currency. Not checked yet for other brands, open question (query_log_2026_10_02 Query_ID: T05-0001).

## 5. The 30 Features, by Category

### Recency (3)

| Feature | Meaning |
|---|---|
| `days_since_last_active` | Days since the player's last bet, of any kind. |
| `days_since_last_deposit` | Days since the player's last completed deposit. |
| `days_since_last_bonus` | Days since the player's last bonus status change, of any status. |

### Frequency (5)

| Feature | Meaning |
|---|---|
| `n_active_days_last_7d`, `n_active_days_last_30d` | Number of distinct days with at least one bet, in the last 7 or 30 days. |
| `n_deposit_last_7d`, `n_deposit_last_30d` | Number of distinct days with at least one completed deposit, in the last 7 or 30 days. |
| `n_bonus_last_7d` | Number of distinct days with at least one bonus status change, in the last 7 days. |

### Monetary (9)

In EUR. This group breaks GGR into its own parts instead of only showing the net result.

| Feature | Meaning |
|---|---|
| `stake_last_7d`, `stake_last_30d` | Money the player bet in the last 7 or 30 days (real-money bets only, not cancelled). |
| `win_last_7d`, `win_last_30d` | Money the player won back in the last 7 or 30 days. |
| `net_loss_last_7d`, `net_loss_last_30d` | `stake - win`. Positive means the player lost money, this is the operator's gross win from this player. |
| `deposit_amount_last_7d`, `deposit_amount_last_30d` | Money the player deposited in the last 7 or 30 days. |
| `bonus_amount_last_7d` | Value of bonuses that became active for the player in the last 7 days. |

### Lag and Trend (9)

Every column here compares the last 7 days to the 7 days just before that.

| Feature | Meaning |
|---|---|
| `freq_drop_7_vs_prior7` | Active days now minus active days in the window just before. Negative means less activity now. |
| `deposit_drop_7_vs_prior7` | Same idea, for deposit days. |
| `bonus_drop_7_vs_prior7` | Same idea, for bonus days. |
| `stake_trend_7` | Stake now minus stake in the window just before, in EUR. |
| `win_trend_7` | Same idea, for win. |
| `net_loss_trend_7` | Same idea, for net loss. |
| `deposit_amount_trend_7` | Same idea, for deposit amount. |
| `bonus_amount_trend_7` | Same idea, for bonus amount. |
| `already_dormant_7` | 1 if the player had zero active days and zero stake in both windows being compared, 0 otherwise. On its own this is not a behavior signal. A drop or trend of exactly 0 can mean two very different things: a player who already stopped playing a while ago, or a player who is simply steady. This flag tells the two apart.  |

### Tenure (3)

| Feature | Meaning |
|---|---|
| `tenure_days` | Days since the player's registration date. 30 when the date is not known (see Section 2). |
| `tenure_days_missing` | 1 if the registration date is not known, 0 otherwise. |
| `max_prior_gap_days` | The longest gap, in days, between two active days inside the window, counting only gaps where both days are on or before `cutoff_date`. |

### Mix (1)

| Feature | Meaning |
|---|---|
| `rtp_last_7d` | `win / stake` in the last 7 days. Only defined when stake is more than 0. |

## 6. Known Limits

- **`bonus` has no `brandId`**, so I match bonus events to the brand by `partyId`, using the players in the snapshot. I checked that one `partyId` never bets under two brands (one day of `primus`, 2026-07-20).

- **`bonus_amount` is new and not yet checked on real data.** Every other feature here is one of the 14 signals already tested with lift, Cox, and Kaplan-Meier in `03_signal_explorer.ipynb`. `bonus_amount` is not. I added it because a separate prior study (`eda/churn_patterns.ipynb`, a different company's data) found bonus amount to be one of the strongest early-churn signals there. It is worth testing here, not yet a confirmed signal for our own brands.
- **`label_available_60d` only checks calendar time**, not whether a player has enough of their own history yet. A brand new player can have plenty of calendar time ahead and still be too new for any label to mean much. That is a different, player-level problem, and needs its own study based on days since each player's first bet, not a shared `cutoff_date`.
