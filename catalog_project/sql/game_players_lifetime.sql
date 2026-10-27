-- By game_uid, the unique players over the whole life of the game. The table is per tenant, game and platform, and a
-- player belongs to one tenant, so the sum over tenants is exact.
WITH
players AS (
    SELECT platform_id, lower(replaceRegexpAll(game_id, '[^a-zA-Z0-9]', '')) AS game_key, unique_players_lifetime AS n
    FROM whizdomai_org_gmntc_medallion.gld_game_unique_players FINAL
),
resolution AS (
    SELECT game_id, provider_id, any(game_uid) AS uid
    FROM whizdomai_org_gmntc_medallion.slv_games_discovered_resolved
    GROUP BY game_id, provider_id
)
SELECT resolution.uid AS game_uid, sum(players.n) AS unique_players_lifetime
FROM players
LEFT JOIN resolution ON players.game_key = resolution.game_id AND players.platform_id = resolution.provider_id
GROUP BY game_uid
