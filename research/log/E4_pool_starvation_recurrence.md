# E4-01-01 collapsed for want of an opponent pool — and it was a launch error

**2026-09-19.** The first run on the post-P17 engine (Grain Sales flattened, Missile Envy starred
card fixed) collapsed one seat and was aborted at 184M of 240M. **The cause is configuration, not
the engine.** Superseded by `E4-02-01`.

## What happened

`E4-01-01_20260919_003959` trained well for 170M steps and then fell over:

| | early / peak | at abort (184M) |
|:---|---:|---:|
| vs HeuristicBot | 95% @105M | 89% (US **80**, USSR **98**) |
| `us_win_rate` (self-play) | 0.82 | **0.006** |
| `ussr_win_rate` | 0.18 | **0.994** |
| `adv_std_raw` | 0.28 | **0.074** |
| `critic_auc` | 0.861 | **0.659** |
| `critic_brier_skill` | 0.382 | **0.014** |
| `critic_base_rate` | 0.52 | **0.980** |
| `mean_turn` | 7.2 | **4.3** |

The critic ended with essentially no predictive skill. `us_win_rate` eroded gradually from about
120M rather than falling off a cliff, so there is a long shallow onset before the fast part.

## The cause

`opponent_frac = 0.0`, `opponent_self_pool = False`. The configured `opponent_pool_size = 12` was
never used, so the policy trained only against its own current self with no diversity.

Every prior run in the lineage used a pool — checked, not assumed:

| runs | `opponent_frac` | `opponent_self_pool` | pool |
|:---|---:|---:|---:|
| E3-30-28 … E3-37-31 (11 runs) | 0.3 | True | 12 |
| **E4-01-01** | **0.0** | **False** | 12, unused |

The omission is mine. The training command in `CLAUDE.md` §"Running training" does not carry the
pool flags, and I launched from it directly without diffing against a prior run's `metadata.json`.

## It is the X4b failure again, by a different route

[P15_X4b_collapse_is_pool_starvation.md](P15_X4b_collapse_is_pool_starvation.md) diagnosed that
collapse as an opponent pool shrunk to one by a `dirname` path bug, and recorded a fingerprint.
Side by side:

| | X4b (`E3-29-28`) | E4-01-01 |
|:---|---:|---:|
| `mean_turn` | 3.82 | 4.29 |
| DEFCON-1 share of wins | 60.8% | 58.9% |
| win rate vs its opponent(s) | 99.7% | 99.4% |

Same picture. X4b lost its pool; E4-01-01 never had one.

**This exonerates the P17 engine changes for this collapse**, which matters because the run existed
to exercise them. Not by argument but by having an independent, documented cause with a matching
fingerprint — and the first 105M steps were healthy, climbing 35% → 95% against HeuristicBot.

## Fix

`E4-02-01_20260919_040456`, identical but with `--opponent-frac 0.3 --opponent-self-pool
--opponent-pool-size 12`. Startup confirms it: `[opponent pool] self (seeded from the initial
policy), frac=0.3, capacity=12, self-growing=True`.

**`CLAUDE.md`'s training command should carry the pool flags**, or the next person launches the same
broken run. Fixed in the same change as this note.

## Monitoring lessons, independent of the collapse

Three alarms fired during this run and **all three were false**:

1. `critic_base_rate` > 0.75 — it does not track side balance at all. When it read 0.752 the actual
   self-play split was 57/43.
2. self-play side split > 65/35 on a single reading — the split oscillates in a 0.45–0.73 band.
3. a 50-iteration windowed mean crossing a bound — this run oscillates over *hundreds* of
   iterations, so a 50-iteration window manufactures trends.

**`us_win_rate` / `ussr_win_rate` are logged directly** in `training_metrics.jsonl` and are the
measure to watch; `critic_base_rate` is not a proxy for them. And the one signal that was real,
`adv_std_raw` declining 0.28 → 0.19, was visible as a clean monotone *series* while no single
sample of it had crossed any threshold. Read series, not samples.

The per-side frozen-opponent split is what finally distinguished a real collapse from oscillation:
US 80% against HeuristicBot while USSR held 98%. That is the check worth running first, and
`opp_pool_size` / `opp_win_rate_mean` are absent from this run's metrics entirely — their absence
is itself the tell, and would have been a faster diagnosis than any of the above.
