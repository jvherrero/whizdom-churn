## Source-table register

| Table name | Grain | Date range | Row count | Top null columns | Gap count |
| --- | --- | --- | --- | --- | --- |
| player | One row per player change event; partyId repeats, uuid identifies the event and is unique. | 2026-06-18 to 2026-09-29 | 174860 | lastKycRequestedDate (99.9806%), profession (99.9788%), lockedUntil (99.9708%) | 0 |
| transaction | One row per payment status-change event, not one row per payment; paymentId repeats. | 2026-06-18 to 2026-09-29 | 742284 | requestedAmount (98.2752%), processDate (59.1485%) | 0 |
| bonus | One row per bonus status-change event; changeId is unique, id (the bonus) repeats. | 2026-06-18 to 2026-09-29 | 648283 | releaseDate (61.2921%) | 0 |
| bet | One row per bet/transaction event; transactionId is unique. | 2026-06-18 to 2026-09-29 | 22381408 | rollbackTranId (99.951%), address (8.994%), firstname (0.2783%) | 0 |

| session | Not available | N/A | N/A | N/A | N/A |

`session` has no source under `org/10-landing/kafka-sink` for either operator (`primus`, `secundus`) both prefixes were listed in full, this is not a listing truncation. No DQ report is generated for it. Pending: confirm with the data team whether session/login telemetry is ingested through a different path.
