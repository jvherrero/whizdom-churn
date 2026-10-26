-- Games of the last client area snapshot that are not in ref_games.
SELECT
    uniqExact(game_uid) AS client_game_uids,
    uniqExactIf(game_uid, game_uid NOT IN (SELECT game_uid FROM whizdomai_org_gmntc_medallion.ref_games FINAL)) AS not_in_ref_games
FROM whizdomai_org_gmntc_medallion.slv_client_games FINAL
WHERE snapshot_date = (SELECT max(snapshot_date) FROM whizdomai_org_gmntc_medallion.slv_client_games)
