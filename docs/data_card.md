<!-- source-register:start -->
## Source-table register

| Table name | Grain | Date range | Row count | Top null columns | Gap count |
| --- | --- | --- | --- | --- | --- |
| gld_player_signals_daily | One row per player ever seen and as-of day (snapshot_date, UTC); the precomputed signals: l7d/l30d/l90d/l180d windows, prior windows, tenure, recency, platform scores. | 2025-04-01 to 2026-10-06 | 689211331 | brand_id (86.7789%) | 0 |
| gld_player_engagement_daily | One row per player ever seen and day: sessions, bets and active day that day. | 2025-04-01 to 2026-10-06 | 688181493 | none | 0 |
| gld_player_financial_daily | One row per player ever seen and day: EUR wagered, GGR, NGR, deposits, withdrawals that day. | 2025-04-01 to 2026-10-06 | 688181082 | none | 0 |
| gld_player_gaming_daily | One row per player, game, provider, day and currency with play that day (sparse). | 2025-04-01 to 2026-10-06 | 59644836 | none | 0 |
| gld_player_payments_daily | One row per player, processed day and currency with payments that day (sparse). | 2025-09-18 to 2026-10-06 | 5583579 | none | 40 (see JSON report for the full list) |
| gld_player_bonus_daily | One row per player, day, award type and currency with bonus activity that day (sparse). | 2025-04-01 to 2026-10-06 | 176426517 | none | 0 |
<!-- source-register:end -->
