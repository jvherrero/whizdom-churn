# Review script: errors of the signals table


---

## 0. Opening (1 minute)

> "I checked the table `gld_player_signals_daily`, column by column. I use all the rows, with no sampling: two full snapshots, 15 September and 1 October 2026, and all of September for the format checks. There are five brands: 14, 23, 64, 72 and 73.

> For each column I do two things. First, I check that it follows its own rules: its formula in the Excel file, the sums of the windows, the limits. Then, for the most important ones, I compute them again myself, from each bet or from the daily tables, and I compare. A column can follow its rules and still have the wrong definition, and you only see that when you compute it again."

**How to read the figures** (say it once):
> "All the figures have the same format. At the top, in bold, is the result. Below it, what each axis shows and the formula. Red is the signals table; blue is my own calculation or the correct definition; grey is a reference. The red zone is impossible values; the orange zone is suspicious values. And under each figure there is the exact formula of what it shows."

---

## 1. Summary (section 1) and the 165 columns (section 2)

> "In one sentence: the amounts and the counts pass their checks, 99 of 165 columns. But there are serious errors in time of play, recency and active days, and one of them uses data from the future."

| Status | Columns |
|---|---|
| Wrong | 37 |
| Affected (correct calculation, but one input is wrong) | 12 |
| Unconfirmed (the Excel file has no full definition) | 11 |
| **Pass** | **99** |
| Not checkable (`_version`, `param_version`, `has_sufficient_data`, and the load times and the snapshot date, which I cannot check with the lake) | 6 |

> "The errors are in order of priority. High: it affects a score of the platform or a model today. Medium: wrong in many rows, but nobody uses it to make decisions. Low: few rows, only the history, or a small detail."

---

## 2. Evaluation criteria (section 3)

> "Before the errors, these are the criteria. There are 210 checks for the 165 columns, of ten kinds."

- **Formula and identity:** the value is equal to its formula, with a tolerance of 0.01 + 0.1%, which is the rounding of the amounts.
- **Window:** the 7 or 30 day value is the sum of its daily value.
- **Lag:** a `prior_*` column is the value of its column 7 days before.
- **Bound:** possible values (for example, 0 to 100) and the order between windows (7 days ≤ 30 days).
- **Format:** UUID, only digits, codes in upper case, never empty.
- **Distribution:** the categories are not uniform; the value and risk levels go down from the first to the last. I check it in its figure, with no test.

**If they ask about the distributions of the tiers and bands (figure 42):**\
*What it shows:* the % of rows in each tier or band (value, risk, churn, lifecycle), in the order of the levels; in grey, Not Scored or empty.
> "They are fine. They are not uniform: the value and risk levels go down from the first to the last. For value, from Plankton, 23.9 percent, to Whale, 0.3 percent: many players of low value and few of high value, as expected. The grey bars, Not Scored, are players without enough data to compute the score."
- **Share:** the tax is a part of the GGR.
- **Validated:** compared with the daily tables.

> "A check passes if it is true in at least 99% of the rows; it is almost, from 95% to 99%; and it fails below that. Annex B gives, for each column, its status, its point and each check, with the exact rule and its percentage."

**If they ask why I do not compare player by player:**
> "The `accountId` of the transactions is not the same id as the `player_id` of the table: only 2 of about 18 000 match. It is not an error of the table, it is a different numbering. So I compare my own numbers as a distribution."

**If they ask "what does 100% mean?":**
> "That all the players, or all the rows of the group, meet the condition. The formula is under each figure."

---

# High priority

## Point 1. `days_since_bet`: it uses data from the future

**What it is:** the days since the last bet of the player, looking only up to the snapshot day (D). If the player bet 3 days ago, it should be 3.

**What I found:** it does not look up to day D, but up to 7 days later.

**Figure 15, what I say:**\
*What it shows:* x axis, the shift s, from 0 to 10 days; y axis, the % of players whose `days_since_bet` on day D is equal to their real recency on day D+s.
> "For each player I compare `days_since_bet` on day D with the real recency on another day: D, D+1, D+2, up to D+10. That is the shift on the x axis. If the column were correct, it would match at shift 0, the blue bar, at almost 100%. But at 0 only 6% match. Where it matches is shift 7: 92% of the players. That whole red zone is days that have not happened yet on day D. The grey lines are the brands: all the same."

**Figure 16, what I say:**\
*What it shows:* the AUC of each column alone to predict if the player bets between D+1 and D+7 (0.5 = chance, 1 = perfect), by brand; red, `days_since_bet`; blue, the real recency up to D; yellow, `churn_score`.
> "This is the effect. With this column alone, predicting if the player bets in the next 7 days gives an AUC of 0.99, almost perfect. With the real recency, the one you can know on day D, it gives 0.79 to 0.87. That 0.99 is not real. Any model with this column will look very good in the tests and will be bad in real use."

**Figure 14 (the history):**\
*What it shows:* in each monthly snapshot, the % of players who bet in the 30 days before with `days_since_bet` = 0, by brand.
> "Also, the history is not useful: until May 2026 it is 0 for almost every active player. It only has values from the daily process, from August."

**Why it matters:** it goes into `churn_score`, `deposit_recency_score` and `lifecycle_stage`.

**What I propose:**
> "Compute it with the daily activity up to day D, like the churn pipeline already does with `days_since_last_bet`. The churn model does not use this column: it uses its own."

---

## Point 2. `active_days_*`: it counts the days with playable bonus

**What it is:** how many days the player played in the last 7, 30 or 90 days. By the official definition (M053), a day counts only if the player bet with **real money or released bonus**. A day with only **playable bonus** does not count.

**What I found:** the table also counts the days with only playable bonus. It counts too many days.

**Figure 17, what I say:**\
*What it shows:* the % of players whose `active_days` in the table is equal to my active days: blue, with the official definition; red, with any wallet.
> "For each player I counted the days in two ways: with the official definition, and with any wallet, playable bonus included. Then I check which one the table matches. With any wallet it matches almost always, 96 to 100 percent of the players: the red bars. With the official one, much less, 48 to 86 percent: the blue bars."

**Figure 18, what I say:**\
*What it shows:* the % of players who bet in the 30 days up to 1 September, but only with playable bonus; on top, how many they are and the active days the table gives them.
> "The clearest case: players who bet only with playable bonus in 30 days. For the official definition they have 0 active days, they are inactive. They are 17 to 34 percent of the recent players, and the table gives all of them 1 to 2 active days."

**Why it matters:**
> "The active day is the base of churn: it decides who is active and who has left. It also affects `frequency_decline` and `engagement_score`."

**What I propose:**
> "Count only the days with a bet of real money or released bonus. The churn model already does this."

**If they ask why the red bars do not reach 100%:**
> "When it does not match, the table has one extra day, almost never more. I think it counts days with a row but no bet, like the day a sports bet is settled. I still need to confirm it, but it does not change the result: it counts too many, never too few."

---

## Point 3. `churn_score`: correct calculation, wrong inputs

> "`churn_score` follows its formula: 0.5 for recency pressure, 0.3 for frequency decline and 0.2 for value decline. The problem is not the calculation, it is the inputs. All three come from columns we just saw are wrong: recency with future data (point 1), frequency with the playable bonus days (point 2) and value decline with no limit (point 4). So the calculation is correct, but on bad data, and it cannot be used as it is."

**Figures:** 16 (yellow bar) and 20.

**If they ask why its AUC is not 0.99:**
> "Because it mixes other inputs, and the recency goes in with a limit of 100, so the leak gets smaller. Its AUC is 0.78 to 0.83, but it still uses data from the future."

**What I propose:** compute it again with recency up to the date, the official active days and `value_decline` limited to [0, 100], or do not use it.

---

## Point 4. `value_decline`: no upper limit

**What it is:** how much the value of the player has gone down. It compares the mean GGR per day of the last week with the one of 90 days. 0 = no drop; 100 = full drop. By the Excel file, it goes from 0 to 100. It goes into `churn_score`.

**What I found:** it reaches **57 million** in September (802 million in the history). 0.84% of the rows are above 100.

**Why it happens:**
> "The problem is the division. If in 90 days the player left almost no GGR, for example 1 euro, the bottom of the division is almost zero. If the player also won this week, the GGR of the week is negative. A negative number divided by almost zero gives a huge number, and the formula has no limit."

**Figure 20, top left panel, what I say:**\
*What it shows:* four histograms of September (x axis, the value; y axis, % of rows on a log scale); the red bar is the value that shows the problem. Top left, `value_decline`.
> "The blue bars are the possible values, from 0 to 100. Most are at 0, no drop, and 16 percent are exactly at 100, which is normal: the player who did not play this week. The red bar, apart, outside the scale, is the values above 100: 0.84 percent of the rows, with values up to 57 million. That is impossible for a score from 0 to 100."


---

## Point 5. All the `tp_*` columns: coverage and freshness

**What they are:** the "time of play" columns: session length, gap between sessions, night, weekend, hourly entropy, and so on.

**Figure 13, top panel, what I say:**\
*What it shows:* top, the rows with a time of play value each day; bottom, the % of those rows with an impossible value; the red lines are the frozen days.
> "These are the rows with a time of play value each day. They start on 6 July 2026 with about 5 000 rows a day; on 1 August they jump to about 300 000, sixty times more; and on 19 August the way to compute the session changes. These are three different stages, and before 6 July there is nothing."

**Figure 13, red lines, what I say:**
> "The red lines are 9 frozen days: 21 to 24 and 26 August, and 4, 11, 14 and 15 October. On those days the columns have exactly the same rows and the same impossible values as the day before, to the exact number, in every brand, while the rest of the table does change. The process ran, but the `tp_*` columns were not computed again."

**Why it matters:**
> "A model needs months of history. Here there are three months, with three different definitions and repeated days."

**What I propose:** do not use the history of the `tp_*` columns (use the recomputed ones) and add a freshness check to the daily process: if today's `tp_*` values are the same as yesterday's, send an alert.

---

# Medium priority

## Points 6 to 8. Sessions, gaps and `play_secs`

**What I found:**
- **Point 6, session length:** 0.6% of the sessions last more than 1 day, up to 28 days. 1.8% last exactly 1 second (sessions with only one bet). 8.4% are outside 10 s to 4 h, and 14.1% in brand 73.
- **Point 7, gap between sessions:** it is negative in 0.91% of the rows, down to −262 h. This means a session starts before the one before it ends.
- **Point 8, `play_secs`:** 3.7% of the days with play are above 24 h, with a maximum of 26 million seconds (305 days). Most of them are sports bets (77% of the cases); in casino, it adds up the games played at the same time.

**Figure 1, what I say:**\
*What it shows:* x axis, the mean session length of each player in September (log scale); y axis, the % of players in each bar; red, the table; blue, recomputed from each bet.
> "The red line is the table; the blue one is the sessions I built again from each bet: a session is a run of bets with no gap longer than 30 minutes, and it lasts from the first to the last bet. The centre of both curves is the same: the table is right for the normal cases. But only the table has that tail of sessions longer than 4 hours and longer than one day, in the orange and red zones. When I build them again, it goes away: only 0.3 percent is out of range, against 8.4."

**Figure 3:**\
*What it shows:* figure 1 with one panel per brand; the box is the % of players outside 10 s to 4 h.
> "The same, brand by brand: the tail is in all five, and the worst is 73."

**Figure 4, what I say:**\
*What it shows:* a table of brands and rules; each cell is the % of rows of that brand that break the rule.
> "This table sums up the rules that any real session follows: it does not last more than one day, the gap is not negative, the time of one day is not more than 24 hours. Each cell is the percentage of rows that break the rule. It should all be 0, and every brand breaks every rule. The problem is in how the sessions are built, not in one brand."

**Figure 5, what I say (sports or casino):**\
*What it shows:* three panels (time above 24 h, session longer than a day, negative gap); each bar is the % of rows of the group (sports, casino or both) with that impossible value.
> "Before I say that all this is a calculation error, I checked if they are sports bets, because a sports bet stays open until the match ends. I join with the gaming table, where providers 58 and 50 are sports.
>
> On the left, `play_secs` above 24 hours, by what the player played that day: only sports, sports and casino on the same day, or only casino. The days with some sports bet have it in 14 percent of the rows; the days with only casino, in 1 percent. The sports days are 77 percent of the cases.
>
> In the centre, sessions longer than one day: sports players have 7 times more, and they are 64 percent of the cases.
>
> On the right, negative gaps: here it is not sports, 60 percent are players with only casino.
>
> So there are two problems. One: the table mixes sports and casino, and in sports the time shows how long the bet is open. Two: even in casino only, there are still calculation errors, the negative gaps and the time of games played at the same time."

**If they ask why the bars do not add up to 100%:**
> "Each bar is the percentage inside its group: of the 58 927 days with only sports, 8 196 are above 24 hours, 13.9 percent. It answers 'if I take a day of this kind, what is the chance it is above 24 hours'. What adds up to 100 is the number below: of all the cases, which part comes from each group, 42, 35 and 23 percent."

**What I propose:**
- Separate the sports bets (providers 58 and 50) before computing sessions and time of play.
- In casino, build the sessions again from each bet and compute `play_secs` as the union of the time intervals of the day, without adding up games played at the same time.

---

## Points 9 to 12. Night, `rg_night_play_ratio`, weekend and entropy

**What they are:** shares from 0 to 1: what part of the stake is at night, what part at the weekend, and how spread the play is across the hours.

**Figure 8, what I say:**\
*What it shows:* the histogram of night, weekend and entropy in September, without the zeros (y axis on a log scale); in red, what is above 1; the dotted lines, the exact fractions.
> "All three go above 1, and that is impossible for a share: weekend in 21.7% of the rows, night in 7.8%, entropy in 25.6%. And the values fall on exact fractions. For weekend, 3.5/1, 3.5/2, and so on, because the table divides the share by 2/7, which is what an even spread would give. So it is an index, not a share. Night is the same, divided by 6/24, with 4, 2, 4/3, and so on. And the entropy is in bits, not scaled to 0 to 1, with exact peaks at log₂ 2, log₂ 3 and log₂ 4."

**Figure 7:**\
*What it shows:* by brand, the % of rows in September with night, weekend or entropy above 1.
> "It happens in every brand: it is a calculation error, not a problem of one brand."

**Figures 9, 10 and 11** (if they want to see them one by one):\
*What it shows:* the histogram of each index on its own (x axis, the value; y axis, % of rows); the red zone, above 1.
> "Each index on its own. Everything in the red zone is above 1, and that is impossible for a share: night up to 4, weekend up to 3.5 and entropy up to 4.6."

**Point 10, `rg_night_play_ratio`:**
> "It should be equal to `tp_night_index` and it only matches in 91.5% of the rows. It looks like a version cut at 1, and it goes into `risk_score`."

**What I propose:** compute them as shares from 0 to 1, as the Excel file says. In the meantime, the scale can be undone: × 2/7 for weekend, × 6/24 for night and ÷ log₂ 24 for entropy.

---

## Point 13. `deposit_recency_score`

> "It is a deposit score, but it is computed with the days since the last **bet**, not since the last **deposit**. A player who bets every day but has not deposited for two months looks like they deposited yesterday. The error is in the definition in the Excel file: the table follows that formula. It also gets the leak of `days_since_bet`."

**What I propose:** compute the days since the last deposit with the payments table.

---

## Point 14. `tenure_days`

**What it is:** the days since the first active day of the player. Between two snapshots it must grow by exactly the number of days between them.

**Figure 19, what I say:**\
*What it shows:* each point is, for one brand and one monthly snapshot, the % of players whose tenure grew by the right amount since the snapshot before.
> "Each point is the percentage of players of a brand whose tenure grew by the right amount since the snapshot before; it must be 100 percent. Until July 2026 it is only 11 to 33 percent. On 1 August, none: that day it jumps by 10 to 16 extra days at once. From September it grows correctly, 100 percent."

**If they ask what the vertical line is (figures 14 and 19):**
> "It is 1 August 2026, when the daily process that writes the table every day starts. The snapshots before that were filled all at once, with a backfill. The tenure error, and the 0 of `days_since_bet`, are in that backfilled history, not in today's daily calculation."

**What I propose:** do not use the old snapshots, and compute it with the daily activity. The churn model computes its own.

---

## Point 15. `recency_pressure`

> "It is 100 in 91.8% of the rows, so it almost does not tell players apart. It is not a calculation error, it is a design problem: it reaches 100 after 3 days without a bet."

**Figure 20, top right:**\
*What it shows:* the histogram of `recency_pressure`; the red bar is the value 100.

**What I propose:** use the ratio without the limit (days without a bet divided by the usual gap) or the recency in days.

---

## Point 16. Empty `brand_id`

**Figure 40, what I say:**\
*What it shows:* each point is the % of rows with an empty `brand_id` in the file of one day.
> "`brand_id` is empty in 100% of the rows until 5 July 2026, and from August it is still empty in a third of them."

**What I propose:** take the brand from the gaming table (each player has only one).

---

## Point 17. Empty `country`, `language` and `currency`

**Figure 41, what I say:**\
*What it shows:* the % of rows with empty `country`, `language` and `currency`, by brand, in September.
> "`country` and `language` are empty in 100 percent of the rows, in every brand, and `currency` in 85 to 89 percent. The values that are filled have the right format: upper case and three letters, with no mix like TR and tr. But there is no data to check."

**What I propose:** fill them from the players table, or leave NULL when the value is not known, not empty text.

---

# Low priority

## Point 18. `top_peak_hour` and `top_peak_day`

> "They are 0 in 75 and 77 percent of the rows. The problem is that 0 is also midnight and a real day, so you cannot tell 'no data' from 'plays at midnight'. They should be NULL when there is no data."

**Figure 20, bottom:**\
*What it shows:* the histograms of `top_peak_hour` and `top_peak_day`; the red bars are the value 0.

---

## Point 19. `hold_pct_l30d`

**What it is:** the percentage of the stake of the last 30 days that the house keeps:

$$h = 100 \cdot \frac{S - P}{S}$$

with **S** the stake and **P** what the player won in those 30 days.


> "The hold is the part of the stake that the house keeps. To be above 100%, the player would have to win less than zero euros. Below, it can be very negative, if the player wins a lot."

**What I found:**

**Figure 21, what I say:**\
*What it shows:* on the left, the highest hold of each daily file; on the right, the lowest 30 day win of each daily file.
> "In the history, the hold is above 100% on 479 days, up to 6551%. That can only happen if the amount won is negative, and you can see it on the right: there are negative wins."

**Figure 24, what I say (where and when):**\
*What it shows:* a map of brands (rows) by days (columns); the colour is the % of rows with a hold above 100%.
> "This map shows where the problem is. Each row is a brand and each column is a day; the colour is the percentage of rows with a hold above 100 percent. It only appears between 17 June and 17 August 2026, almost all in brand 72, with a little in brand 64 at the start of July. In the other brands and days, nothing. It is very few rows: at most 0.06 percent of a day, 232 in total. And it lasts a few weeks because the column is a 30 day window: a negative win stays inside the window for a whole month."

> "So it is not a general calculation error, but some specific data of one brand on those dates. That is why it is low priority."

---

## Point 20. `deposit_frequency_score`

> "It is empty before 2026, because the payments table has almost no deposits before then. Do not use it in the history."

## Point 21. Units of the scores

> "The Excel file says that all the scores go from 0 to 1, and the table uses 0 to 100. It is an error in the Excel file."

## Point 22. `tax_eur_*` always 0

> "The 27.1 million rows of September have the tax at 0, not empty: at 0, in every window. I cannot check that it is a logical part of the GGR, between 10 and 20 percent. We need to confirm if these brands have tax; if they do, the source does not send it."

# Unconfirmed (points 23 to 25)

> "I cannot say these three are right or wrong, because the Excel file does not give the full definition."

- **Point 23, NGR:** "It matches GGR minus bonus minus tax in 79% to 99% of the rows, but the Excel file does not define the bonus cost."
- **Point 24, `prior_*_score`:** "They are not the value of 7 days before (they match in 49% to 93%), and the Excel file does not say which date they are from."
- **Point 25, z scores:** "The Excel file says 'the mean and the standard deviation of the tenant', but it does not say which players they are computed on."

> "To close them I need those definitions."

---

## Impossible and suspicious values (section 8), if there is time

> "This table shows, for each metric, how many rows have an impossible or suspicious value, in each brand and on how many days. If it appears in every brand and every day, it is a calculation error; if it is in one brand or a few days, it is a specific problem. The maps in the annex show it day by day."

---

## Closing (1 minute)

> "To sum up: the amounts and the counts are fine. What is wrong is the player behaviour: recency, active days, time of play and some scores. And the most serious one is `days_since_bet`, which uses data from the future: a model with that column looks perfect, and it is not.
>
> For the churn model there is no risk: it does not use those columns, it computes its own from the daily tables, with the official definition in the Excel file.
>
> What I propose to the platform, in order: fix recency and active days, limit `value_decline` and review `churn_score`; add a freshness check to the daily process; keep every daily version in the lake; and complete the definitions that are missing in the Excel file."

---

## Likely questions

**"How do you know it is not your own error?"**
> "The code has tests with synthetic data, with errors put in on purpose that it must find. And when I say something is wrong, I do not stop at 'it breaks a rule': I compute it again from the source (each bet or the daily tables) and I show the difference, like in recency or active days."

**"Why only September and two snapshots?"**
> "September is the first full month with the daily process, with the brand in almost every row. The two snapshots are complete, with no sampling: 1.8 million rows. The definition errors look the same on any date."

**"Does this affect the churn model?"**
> "No. It uses its own features, computed from the daily tables. What it does is not use these columns of the signals table."

**"What is the first thing you would fix?"**
> "`days_since_bet`, because it puts data from the future into any model that uses it. Then the active days, because they are the base of churn."
