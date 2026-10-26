-- The rounds of the window that reach neither ref_games nor the resolution table, by platform_id.
WITH act AS (
    SELECT tenant_id, game_id, platform_id, sum(rounds) AS n_rounds
    FROM whizdomai_org_gmntc_medallion.slv_game_daily FINAL
    WHERE event_date >= addMonths(today(), -{months:Int32}) AND event_date < today()
    GROUP BY ALL
),
flagged AS (
    SELECT platform_id, game_id, n_rounds,
           game_id NOT IN (SELECT game_id FROM whizdomai_org_gmntc_medallion.ref_games FINAL)
           AND (tenant_id, game_id, platform_id) NOT IN (SELECT tenant_id, game_id, provider_id FROM whizdomai_org_gmntc_medallion.slv_games_discovered_resolved) AS unmatched
    FROM act
)
SELECT
    platform_id,
    sum(n_rounds) AS total_rounds,
    sumIf(n_rounds, unmatched) AS rounds_unmatched,
    rounds_unmatched / total_rounds AS share_unmatched,
    uniqExactIf(game_id, unmatched) AS game_ids_unmatched,
    uniqExact(game_id) AS game_ids
FROM flagged
GROUP BY platform_id
ORDER BY rounds_unmatched DESC
LIMIT 15
