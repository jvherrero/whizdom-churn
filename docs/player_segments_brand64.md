# Player Segments v0 (brandId 64)

I group players by how they behaved in the 30 days before the cutoff, with k-means. The segments are an **input feature** for the churn models, never the churn score: the score comes only from LightGBM (probability) and Cox PH (survival). Stage 2 selection decides whether a model keeps them.

## 1. Method

- **Brands**: one segmentation for every brand in the dataset (64). The features were already computed and winsorised per brand upstream; here the segments are common to all brands, so "high value" means high in EUR, not high relative to the player's own brand.
- **Rows**: the train and validation months (2025-09-01, 2025-10-01, 2025-11-01, 2025-12-01, 2026-01-01, 2026-02-01, 2026-03-01, 2026-04-01, 2026-05-01, 2026-06-01), 120,892 player snapshots. The test months (2026-07-01, 2026-08-01) are only assigned to a segment, never used to fit them.
- **Features** (the 30-day behaviour profile): `active_days_l30d`, `wagered_eur_l30d`, `ggr_eur_l30d`, `games_breadth_30d`, `bonus_stake_share_30d`, `engagement_score`, `days_since_last_bet`. They are on the sign-log scale from `build_features.py` and then standardised (mean 0, std 1), so no feature dominates because of its units.
- **Model**: `kmeans` from `catalog/model_library_catalog.json` (`sklearn.cluster.KMeans`), 10 starts, seed 42.
- **Number of segments**: the k with the highest silhouette coefficient among [2, 3, 4, 5, 6, 7, 8]. Silhouette goes from -1 to 1: high means players are close to their own segment and far from the others. I measure it on a random sample of 10,000 rows, because it is slow on all of them.
- **Order**: segment 0 is the largest. In the models it is the reference, so they get `segment_1` ... `segment_3` (1 if the player is in that segment).

## 2. Choosing k

| k | silhouette | chosen |
|---|---|---|
| 2 | 0.3245 |  |
| 3 | 0.3054 |  |
| 4 | 0.3269 | yes |
| 5 | 0.2856 |  |
| 6 | 0.2448 |  |
| 7 | 0.2586 |  |
| 8 | 0.2525 |  |

![Silhouette by k](../data/03_output/segments/brand64/silhouette.png)

**Chosen: k = 4** (silhouette 0.3269). The next best is k = 2 (0.3245).

## 3. Player Types

Median values per segment, in real units (training months):

| segment | name | share | active days (30d) | stake (30d, EUR) | net loss (30d, EUR) | distinct games (30d) | share of stake from bonus (30d) | engagement score | days since last bet |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Occasional mid-value | 48.4% | 2.00 | 151 | 29.31 | 3.00 | 0.02 | 4.31 | 12.00 |
| 1 | Frequent high-value | 28.0% | 13.00 | 3,777 | 348 | 7.00 | 0.07 | 37.77 | 1.00 |
| 2 | Occasional low-value, bonus-heavy | 12.5% | 1.00 | 45.26 | 6.21 | 2.00 | 1.00 | 2.28 | 14.00 |
| 3 | Frequent high-value | 11.1% | 7.00 | 2,022 | -176 | 6.00 | 0.02 | 19.66 | 2.00 |

How each segment differs from the average player (standardised centroid, top 3 features) and its churn rate. The average churn in the training months is 35.0%. The names come from simple rules on the centroid (activity, stake, recency, bonus use), so they stay the same when I rerun this.

| segment | name | what stands out | churn 60d, training months | churn 60d, test months |
|---|---|---|---|---|
| 0 | Occasional mid-value | low active days (30d) (-0.6 sd), low engagement score (-0.5 sd), low stake (30d, EUR) (-0.5 sd) | 48.1% | 41.1% |
| 1 | Frequent high-value | high active days (30d) (+1.2 sd), high engagement score (+1.1 sd), high stake (30d, EUR) (+0.9 sd) | 6.2% | 4.4% |
| 2 | Occasional low-value, bonus-heavy | high share of stake from bonus (30d) (+2.4 sd), low stake (30d, EUR) (-0.9 sd), low active days (30d) (-0.9 sd) | 66.2% | 57.4% |
| 3 | Frequent high-value | low net loss (30d, EUR) (-2.1 sd), high stake (30d, EUR) (+0.7 sd), high distinct games (30d) (+0.5 sd) | 15.1% | 10.9% |

![Segment profiles](../data/03_output/segments/brand64/segment_profiles.png)

## 4. Stability Over Time

Share of players in each segment, per cutoff (the last two are the test months):

| cutoff | segment 0 | segment 1 | segment 2 | segment 3 |
|---|---|---|---|---|
| 2025-09-01 | 54.1% | 24.7% | 9.6% | 11.6% |
| 2025-10-01 | 51.7% | 26.9% | 10.6% | 10.8% |
| 2025-11-01 | 49.3% | 26.2% | 14.1% | 10.3% |
| 2025-12-01 | 56.1% | 23.7% | 10.1% | 10.1% |
| 2026-01-01 | 39.8% | 24.9% | 25.1% | 10.2% |
| 2026-02-01 | 43.9% | 30.8% | 13.9% | 11.4% |
| 2026-03-01 | 45.4% | 30.8% | 11.1% | 12.7% |
| 2026-04-01 | 43.4% | 33.8% | 10.9% | 11.9% |
| 2026-05-01 | 46.0% | 32.3% | 10.6% | 11.2% |
| 2026-06-01 | 40.9% | 33.8% | 13.8% | 11.6% |
| 2026-07-01 | 41.6% | 33.7% | 12.7% | 12.0% |
| 2026-08-01 | 44.8% | 28.2% | 16.2% | 10.7% |

## 5. Use in the Models

Stage 2 decision for the segment features in the latest training run of each model (read from MLflow when this file is generated, so run `make train-baseline` before `make segments`):

- **lightgbm_classifier** (`lightgbm_classifier_brand64_1792127339`): `segment_2` kept.
- **lightgbm_classifier** (`lightgbm_classifier_brand64_1792127339`): `segment_3` dropped at step 4 (permutation importance 0.0000, outside the top 7).
- **lightgbm_classifier** (`lightgbm_classifier_brand64_1792127339`): `segment_1` dropped at step 4 (permutation importance 0.0000, outside the top 7).
- **cox_ph**: no training run with a feature selection report yet.

When the segment is dropped at step 4, the model already gets the same information from the raw features it is built from (active days, stake, deposits). The segment is still useful to describe players: `make score` adds it to every player's score.

## 6. Known Limits

- **k-means looks for round groups of similar size.** Real player types can be long or uneven shapes; a low silhouette means the groups overlap a lot and the borders between them are soft.
- **The churn rates are descriptive.** They show that segments differ, not that the segment causes the churn.
- **A small segment can be dropped by Stage 2** as near-zero variance, and the models can ignore the segments when the raw features already carry the same information. `feature_selection.csv` in each MLflow run says what happened.

Generated by `src/models/segments.py` (`make segments`) from `data/processed/train_dataset_64_2025-09-01_2025-10-01_2025-11-01_2025-12-01_2026-01-01_2026-02-01_2026-03-01_2026-04-01_2026-05-01_2026-06-01_2026-07-01_2026-08-01_1792127230.parquet`.
