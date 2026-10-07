<!-- gold-register:start -->
## Source-table register: gold layer (the model's source)

| Table name | Grain | Date range | Row count | Top null columns | Gap count |
| --- | --- | --- | --- | --- | --- |
| gld_player_signals_daily | One row per player ever seen and as-of day (snapshot_date, UTC); the precomputed signals: l7d/l30d/l90d/l180d windows, prior windows, tenure, recency, platform scores. | 2025-04-01 to 2026-10-06 | 689211331 | brand_id (86.7789%) | 0 |
| gld_player_engagement_daily | One row per player ever seen and day: sessions, bets and active day that day. | 2025-04-01 to 2026-10-06 | 688181493 | none | 0 |
| gld_player_financial_daily | One row per player ever seen and day: EUR wagered, GGR, NGR, deposits, withdrawals that day. | 2025-04-01 to 2026-10-06 | 688181082 | none | 0 |
| gld_player_gaming_daily | One row per player, game, provider, day and currency with play that day (sparse). | 2025-04-01 to 2026-10-06 | 59644836 | none | 0 |
| gld_player_payments_daily | One row per player, processed day and currency with payments that day (sparse). | 2025-09-18 to 2026-10-06 | 5583579 | none | 40 (see JSON report for the full list) |
| gld_player_bonus_daily | One row per player, day, award type and currency with bonus activity that day (sparse). | 2025-04-01 to 2026-10-06 | 176426517 | none | 0 |
<!-- gold-register:end -->

## Source-table register: landing layer (exploration only)

| Table name | Grain | Date range | Row count | Top null columns | Gap count |
| --- | --- | --- | --- | --- | --- |
| player | One row per player change event; partyId repeats, uuid identifies the event and is unique. | 2026-06-18 to 2026-09-29 | 194110 | lastKycRequestedDate (99.9799%), profession (99.9732%), lockedUntil (99.9681%) | 0 |
| transaction | One row per payment status-change event, not one row per payment; paymentId repeats. | 2026-06-18 to 2026-09-29 | 823934 | requestedAmount (98.28%), processDate (59.0669%) | 0 |
| bonus | One row per bonus status-change event; changeId is unique, id (the bonus) repeats. | 2026-06-18 to 2026-09-29 | 707640 | releaseDate (61.1596%) | 0 |
| bet | One row per bet/transaction event; transactionId is unique. | 2026-06-18 to 2026-09-29 | 22381408 | rollbackTranId (99.951%), address (8.994%), firstname (0.2783%) | 0 |
| session | One row per (player, active day) (see notes); partyId repeats, (operator, partyId, activity_date) is unique. | 2026-06-18 to 2026-09-30 | 2634934 | none | 0 |
