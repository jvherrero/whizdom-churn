-- ref_games after FINAL: size, duplicates, empty keys and the last load.
SELECT
    count() AS rows,
    uniqExact(game_uid) AS game_uids,
    uniqExact(game_id, provider_code) AS game_id_provider_pairs,
    countIf(game_id = '') AS empty_game_id,
    countIf(provider_code = '') AS empty_provider_code,
    max(_ingested_at) AS last_ingested_at,
    dateDiff('hour', max(_ingested_at), now()) AS hours_since_ingest,
    max(last_seen) AS max_last_seen
FROM whizdomai_org_gmntc_medallion.ref_games FINAL
