-- The tenants with activity in the last 90 days.
SELECT DISTINCT tenant_id
FROM whizdomai_org_gmntc_medallion.slv_game_daily
WHERE event_date >= today() - 90
