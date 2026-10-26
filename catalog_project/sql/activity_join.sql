-- The rounds of the window, with the game_id of the activity normalized like the catalog (lower case, letters and
-- digits only) and joined to the resolution table by (game_id, platform_id = provider_id). All, then the biggest
-- platforms by rounds that stay unmatched.
WITH
act AS (
    SELECT platform_id, lower(replaceRegexpAll(game_id, '[^a-zA-Z0-9]', '')) AS game_key, sum(rounds) AS n_rounds
    FROM whizdomai_org_gmntc_medallion.slv_game_daily FINAL
    WHERE event_date >= addMonths(today(), -{months:Int32}) AND event_date < today()
    GROUP BY ALL
),
res AS (
    SELECT game_id, provider_id, any(game_uid) AS uid, uniqExact(game_uid) AS n_uid
    FROM whizdomai_org_gmntc_medallion.slv_games_discovered_resolved
    GROUP BY game_id, provider_id
),
flagged AS (
    SELECT act.platform_id AS pid, act.n_rounds AS r, res.n_uid AS n_uid,
           act.game_key IN (SELECT game_id FROM whizdomai_org_gmntc_medallion.ref_games FINAL) AS key_in_ref,
           res.uid IN (SELECT game_uid FROM whizdomai_org_gmntc_medallion.ref_games FINAL) AS uid_in_ref
    FROM act LEFT JOIN res ON act.game_key = res.game_id AND act.platform_id = res.provider_id
)
SELECT * FROM (
    SELECT 'all' AS scope, sum(r) AS total_rounds,
           sumIf(r, n_uid > 0) / total_rounds AS share_resolved,
           sumIf(r, n_uid > 1) / total_rounds AS share_ambiguous,
           sumIf(r, uid_in_ref) / total_rounds AS share_resolved_in_ref,
           sumIf(r, n_uid = 0 AND key_in_ref) / total_rounds AS share_by_id_only,
           sumIf(r, n_uid = 0 AND NOT key_in_ref) / total_rounds AS share_unmatched
    FROM flagged
)
UNION ALL
SELECT * FROM (
    SELECT concat('platform ', toString(pid)), sum(r) AS total_rounds,
           sumIf(r, n_uid > 0) / total_rounds, sumIf(r, n_uid > 1) / total_rounds, sumIf(r, uid_in_ref) / total_rounds,
           sumIf(r, n_uid = 0 AND key_in_ref) / total_rounds, sumIf(r, n_uid = 0 AND NOT key_in_ref) / total_rounds AS share_unmatched
    FROM flagged
    GROUP BY pid
    ORDER BY sumIf(r, n_uid = 0 AND NOT key_in_ref) DESC
    LIMIT 12
)
