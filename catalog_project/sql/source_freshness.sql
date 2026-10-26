-- Last load of the tables that feed ref_games and of the activity table.
SELECT 'brz_client_games' AS source, toString(max(snapshot_date)) AS last_date, max(_ingested_at) AS last_ingested_at
FROM whizdomai_org_gmntc_medallion.brz_client_games
UNION ALL
SELECT 'slv_client_games', toString(max(snapshot_date)), max(_ingested_at)
FROM whizdomai_org_gmntc_medallion.slv_client_games
UNION ALL
SELECT 'slv_game_daily', toString(max(event_date)), max(_ingested_at)
FROM whizdomai_org_gmntc_medallion.slv_game_daily
