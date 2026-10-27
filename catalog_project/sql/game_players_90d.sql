-- By game_uid, the unique players of the last 30 and 90 days in ONE tenant (approximate count, uniq). The script runs
-- it once per tenant, to stay under the time limit; a player belongs to one tenant, so the sum is the total.
-- (The 30 day figure of gld_game_signals_daily is the sum of the daily players, not distinct players.)
WITH
rounds AS (
    SELECT player_id, platform_id, event_date, lower(replaceRegexpAll(game_id, '[^a-zA-Z0-9]', '')) AS game_key
    FROM whizdomai_org_gmntc_medallion.slv_game_rounds
    WHERE tenant_id = {tenant:String} AND event_date >= today() - 90 AND event_date < today()
),
resolution AS (
    SELECT game_id, provider_id, any(game_uid) AS uid
    FROM whizdomai_org_gmntc_medallion.slv_games_discovered_resolved
    GROUP BY game_id, provider_id
)
SELECT
    resolution.uid AS game_uid,
    uniq(rounds.player_id) AS unique_players_90d,
    uniqIf(rounds.player_id, rounds.event_date >= today() - 30) AS unique_players_30d
FROM rounds
LEFT JOIN resolution ON rounds.game_key = resolution.game_id AND rounds.platform_id = resolution.provider_id
GROUP BY game_uid
