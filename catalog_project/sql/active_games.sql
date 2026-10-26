-- Rounds and wagered EUR of the window by game_uid. The game_id of the activity is normalized like the catalog
-- (lower case, letters and digits only) and joined to the resolution table by (game_id, platform_id = provider_id).
-- game_uid '' is the activity that does not resolve.
WITH
act AS (
    SELECT platform_id, lower(replaceRegexpAll(game_id, '[^a-zA-Z0-9]', '')) AS game_key,
           sum(rounds) AS n_rounds,
           toFloat64(sum(wagered_real_eur + wagered_released_bonus_eur + wagered_playable_bonus_eur)) AS wagered
    FROM whizdomai_org_gmntc_medallion.slv_game_daily FINAL
    WHERE event_date >= addMonths(today(), -{months:Int32}) AND event_date < today()
    GROUP BY ALL
),
res AS (
    SELECT game_id, provider_id, any(game_uid) AS uid
    FROM whizdomai_org_gmntc_medallion.slv_games_discovered_resolved
    GROUP BY game_id, provider_id
)
SELECT res.uid AS game_uid, sum(act.n_rounds) AS rounds_6m, sum(act.wagered) AS wagered_eur_6m
FROM act LEFT JOIN res ON act.game_key = res.game_id AND act.platform_id = res.provider_id
GROUP BY game_uid
