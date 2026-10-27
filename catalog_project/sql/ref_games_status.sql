-- ref_games by status, enrichment_status and the source flags.
SELECT status, enrichment_status, in_cms, in_transactions, in_client_area, count() AS games
FROM whizdomai_org_gmntc_medallion.ref_games FINAL
GROUP BY ALL
ORDER BY games DESC
