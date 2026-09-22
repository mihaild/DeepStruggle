# P21 + P22 — every rung in one field

Elo is field-relative, so ratings from the per-rung reports cannot be put side by side. This
tournament rates every rung's current snapshots together: the ladder's seed pair (s3 at 80M and
160M, s5 at 80M) for every rung from M2a on, M2d s3 to 240M and s5 to 160M, the early rungs
(M0, M1, M2 at 80M and 160M; M2e at 80M), both anchor legs, the default architecture, the
recovered M2.5b s3 at 240M, and `HeuristicBot` pinned at 1500.

41 models, 100 games per seat per pair, temperature 0, 1,062 s. Engine checked fresh
(`acf49f66`). Report: `data/reports/P21_full_field.{md,json}`.

Win rates are the row's **as USSR / as US**. The M0, M1 and M2 arms predate the seed-pair
protocol and ran seeds 1 (80M) and 2 (160M), so their 80M→160M difference is *not* within-seed.
The anchor ran unseeded.

| # | model | rung | description | Elo | vs M2d s3@160M | vs anchor@80M |
|---:|:---|:---|:---|---:|---:|---:|
| 1 | **E4-08-03@240M** | M2d s3 | country head only | 2248.4 | 61 / 53 | 62 / 81 |
| 2 | **E4-08-03@160M** | M2d s3 | country head only | 2230.9 | — | 70 / 69 |
| 3 | E4-17-03@240M | M2.5b s3 | identity replaces per-type constants — **recovered** | 2212.9 | 51 / 46 | 59 / 76 |
| 4 | E4-14-03@160M | M2b s3 | M2d head, no per-type constants | 2181.3 | 47 / 28 | 68 / 61 |
| 5 | E4-24-03@160M | P22-a s3 | M2d + identity-keyed card lookup | 2169.7 | 41 / 30 | 60 / 68 |
| 6 | **E4-08-05@160M** | M2d s5 | country head only | 2150.8 | 49 / 27 | 61 / 63 |
| 7 | E4-07-02@160M | M2 s2 | country + card per-entity heads | 2101.8 | 44 / 27 | 49 / 46 |
| 8 | **E4-08-03@80M** | M2d s3 | country head only | 2101.0 | 34 / 32 | 41 / 58 |
| 9 | E4-03-01@80M | anchor unseeded | late-E3 bundle | 2097.4 | 31 / 30 | — |
| 10 | E4-13-03@160M | M2a s3 | M2d head, no trunk context | 2074.4 | 28 / 36 | 36 / 67 |
| 11 | E4-07-01@80M | M2 s1 | country + card per-entity heads | 2054.7 | 34 / 25 | 40 / 50 |
| 12 | E4-18-03@160M | M2.5c s3 | M2.5 + card head w/ identity | 2039.3 | 44 / 15 | 38 / 32 |
| 13 | E4-15-03@160M | M2c s3 | M2d head, dynamic slots only | 2037.9 | 29 / 22 | 43 / 41 |
| 14 | E4-14-03@80M | M2b s3 | M2d head, no per-type constants | 2020.4 | 36 / 12 | 42 / 31 |
| 15 | **E4-08-05@80M** | M2d s5 | country head only | 2018.6 | 31 / 23 | 35 / 48 |
| 16 | E4-15-03@80M | M2c s3 | M2d head, dynamic slots only | 2017.7 | 24 / 14 | 33 / 40 |
| 17 | E4-16-03@80M | M2.5 s3 | M2d + country identity | 1977.4 | 9 / 18 | 33 / 47 |
| 18 | E4-14-05@80M | M2b s5 | M2d head, no per-type constants | 1972.7 | 31 / 15 | 24 / 46 |
| 19 | E4-16-05@80M | M2.5 s5 | M2d + country identity | 1970.7 | 18 / 14 | 29 / 45 |
| 20 | E4-13-03@80M | M2a s3 | M2d head, no trunk context | 1968.6 | 24 / 15 | 25 / 44 |
| 21 | E4-23-03@160M | W s3 | M2d, entity_proj_dim 512 | 1953.8 | 26 / 6 | 35 / 25 |
| 22 | E4-24-05@80M | P22-a s5 | M2d + identity-keyed card lookup | 1952.1 | 31 / 4 | 45 / 17 |
| 23 | E4-17-05@80M | M2.5b s5 | identity replaces per-type constants | 1947.6 | 20 / 12 | 26 / 30 |
| 24 | E4-13-05@80M | M2a s5 | M2d head, no trunk context | 1939.8 | 32 / 8 | 16 / 33 |
| 25 | E4-23-03@80M | W s3 | M2d, entity_proj_dim 512 | 1932.7 | 21 / 4 | 30 / 29 |
| 26 | E4-17-03@80M | M2.5b s3 | identity replaces per-type constants | 1929.0 | 21 / 6 | 21 / 27 |
| 27 | E4-18-05@80M | M2.5c s5 | M2.5 + card head w/ identity | 1926.7 | 22 / 12 | 24 / 33 |
| 28 | E4-18-03@80M | M2.5c s3 | M2.5 + card head w/ identity | 1920.5 | 16 / 10 | 18 / 43 |
| 29 | E4-23-05@80M | W s5 | M2d, entity_proj_dim 512 | 1915.6 | 22 / 1 | 23 / 18 |
| 30 | E4-24-03@80M | P22-a s3 | M2d + identity-keyed card lookup | 1913.5 | 22 / 12 | 23 / 27 |
| 31 | E4-15-05@80M | M2c s5 | M2d head, dynamic slots only | 1895.4 | 13 / 9 | 9 / 42 |
| 32 | E4-06-02@160M | M1 s2 | grouped positional projections | 1808.0 | 5 / 6 | 17 / 10 |
| 33 | E4-05-02@160M | M0 s2 | flat MLP | 1782.3 | 7 / 3 | 9 / 9 |
| 34 | E4-06-01@80M | M1 s1 | grouped positional projections | 1746.8 | 10 / 2 | 9 / 15 |
| 35 | E4-09-01@80M | M2e s1 | card head only | 1738.2 | 7 / 2 | 9 / 10 |
| 36 | E4-03-01@160M | anchor unseeded | late-E3 bundle | 1731.5 | 5 / 5 | 6 / 16 |
| 37 | E4-04-01@80M | default s1 | E4 default arch | 1664.5 | 6 / 0 | 3 / 9 |
| 38 | E4-17-03@160M | M2.5b s3 | identity replaces per-type constants — **collapsed** | 1653.9 | 8 / 0 | 11 / 1 |
| 39 | E4-05-01@80M | M0 s1 | flat MLP | 1619.9 | 4 / 0 | 8 / 5 |
| 40 | E4-16-03@160M | M2.5 s3 | M2d + country identity | 1558.6 | 4 / 2 | 4 / 4 |
| 41 | HeuristicBot | —  | rule-based baseline | 1500.0 | 6 / 0 | 1 / 1 |

## Reading

* **M2d is the top of the ladder at every budget.** At 160M s3 no other rung wins on both seats
  against it; the closest are recovered M2.5b@240M (51 / 46, with 80M more steps) and M2b
  (47 / 28). The only snapshot above it is its own 240M continuation.
* **Every per-rung rejection survives re-rating in one field.** Every rung sits below seed-matched
  M2d at 80M on both seeds: M2d s3@80M is 2101.0 against a best of 2020.4 (M2b), and M2d s5@80M
  is 2018.6 against a best of 1972.7 (M2b).
* **The failures are the outliers at the bottom.** `E4-17-03@160M` (collapsed, 0% as US against
  M2d), the anchor's 160M leg (−366 against its own 80M, entropy inflation) and
  `E4-16-03@160M` (M2.5 s3, −419 against its own 80M), which rates below every other network.
