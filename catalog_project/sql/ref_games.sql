-- The catalog as it is now: one row per game_uid.
SELECT game_uid, game_id, provider_code, game_name, provider_name, vertical, game_type, theme, status, enrichment_status,
       release_date, first_seen
FROM whizdomai_org_gmntc_medallion.ref_games FINAL
