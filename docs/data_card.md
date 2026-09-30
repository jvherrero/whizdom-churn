## Source-table register

| Table name | Grain | Date range | Row count | Top null columns | Gap count |
| --- | --- | --- | --- | --- | --- |
| player | One row per player change event; partyId repeats, uuid identifies the event and is unique. | 2026-06-18 to 2026-09-29 | 194110 | lastKycRequestedDate (99.9799%), profession (99.9732%), lockedUntil (99.9681%) | 0 |
| transaction | One row per payment status-change event, not one row per payment; paymentId repeats. | 2026-06-18 to 2026-09-29 | 823934 | requestedAmount (98.28%), processDate (59.0669%) | 0 |
| bonus | One row per bonus status-change event; changeId is unique, id (the bonus) repeats. | 2026-06-18 to 2026-09-29 | 707640 | releaseDate (61.1596%) | 0 |
| bet | One row per bet/transaction event; transactionId is unique. | 2026-06-18 to 2026-09-29 | 22381408 | rollbackTranId (99.951%), address (8.994%), firstname (0.2783%) | 0 |
| session | One row per derived player session (see notes); partyId repeats, (operator, partyId, session_id) is unique. | 2026-09-23 to 2026-09-30 | 373675 | none | 0 |

`session` has no source table under `org/10-landing/kafka-sink`; it is derived from `bet` (see `docs/dq_reports/session.json` for the full method and notes). Its date range runs one day past the other tables' window (`2026-09-30` vs `2026-09-29`) because it is dated by the real event timestamp (local time, +02:00), while the other 4 tables are dated by their S3 partition folder (UTC day) -- the last day's partition holds some local timestamps that fall on the next calendar date.
