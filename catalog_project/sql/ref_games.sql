-- The catalog as it is now.
SELECT game_uid, game_id, provider_code, game_name, provider_name, vertical, game_type, theme, status, enrichment_status
FROM whizdomai_org_gmntc_medallion.ref_games FINAL
