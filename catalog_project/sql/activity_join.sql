-- Check of the join: the share of the rounds of the window that reach a game of ref_games through the resolution
-- table. The unresolved share is 1 minus share_resolved. Same normalization and join as active_games.sql.
WITH
activity AS (
    SELECT platform_id, lower(replaceRegexpAll(game_id, '[^a-zA-Z0-9]', '')) AS game_key, sum(rounds) AS n_rounds
    FROM whizdomai_org_gmntc_medallion.slv_game_daily FINAL
    WHERE event_date >= addMonths(today(), -{months:Int32}) AND event_date < today()
    GROUP BY platform_id, game_key
),
resolution AS (
    SELECT game_id, provider_id, any(game_uid) AS uid, uniqExact(game_uid) AS n_uid
    FROM whizdomai_org_gmntc_medallion.slv_games_discovered_resolved
    GROUP BY game_id, provider_id
)
SELECT
    sum(n_rounds) AS total_rounds,
    sumIf(n_rounds, n_uid > 0) / total_rounds AS share_resolved,
    sumIf(n_rounds, n_uid > 1) / total_rounds AS share_ambiguous,
    sumIf(n_rounds, uid IN (SELECT game_uid FROM whizdomai_org_gmntc_medallion.ref_games FINAL)) / total_rounds AS share_in_ref_games
FROM activity
LEFT JOIN resolution ON activity.game_key = resolution.game_id AND activity.platform_id = resolution.provider_id
