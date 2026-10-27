-- By game_uid, over the window: GGR (wagered minus won, real money and bonus, in EUR), rounds, wagered EUR, and the
-- sum of the daily unique players of the last 30 and 90 days (for stickiness). game_uid '' is the activity that
-- does not resolve to a game.
WITH
-- the activity by platform and game; game_id is normalized like ref_games (lower case, letters and digits only)
activity AS (
    SELECT
        platform_id,
        lower(replaceRegexpAll(game_id, '[^a-zA-Z0-9]', '')) AS game_key,
        sum(wagered_real_eur + wagered_released_bonus_eur + wagered_playable_bonus_eur) AS wagered,
        sum(won_real_eur + won_released_bonus_eur + won_playable_bonus_eur) AS won,
        wagered - won AS ggr,
        sum(rounds) AS n_rounds,
        sumIf(unique_players, event_date >= today() - 30) AS daily_players_30d,
        sumIf(unique_players, event_date >= today() - 90) AS daily_players_90d
    FROM whizdomai_org_gmntc_medallion.slv_game_daily FINAL
    WHERE event_date >= addMonths(today(), -{months:Int32}) AND event_date < today()
    GROUP BY platform_id, game_key
),
-- the game_uid of each (game_id, platform_id = provider_id)
resolution AS (
    SELECT game_id, provider_id, any(game_uid) AS uid
    FROM whizdomai_org_gmntc_medallion.slv_games_discovered_resolved
    GROUP BY game_id, provider_id
)
SELECT
    resolution.uid AS game_uid,
    toFloat64(sum(activity.ggr)) AS ggr_eur_6m,
    sum(activity.n_rounds) AS rounds_6m,
    toFloat64(sum(activity.wagered)) AS wagered_eur_6m,
    sum(activity.daily_players_30d) AS daily_players_30d,
    sum(activity.daily_players_90d) AS daily_players_90d
FROM activity
LEFT JOIN resolution ON activity.game_key = resolution.game_id AND activity.platform_id = resolution.provider_id
GROUP BY game_uid
