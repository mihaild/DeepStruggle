# The arm registry — every arm, and where its result is written

One row per arm: what it varied, at which seeds and budgets, which directory under
`data/checkpoints/` holds it, and **where the result is written up**. Where there is no writeup the
row says *no writeup* rather than leaving the cell blank — an arm that was run and never reported
is a fact about this project, and hiding it behind an empty cell is how an arm gets re-run.

The naming scheme — what `E4-02-01` means, when the engine letter bumps — is
[`method/run_nomenclature.md`](method/run_nomenclature.md). For *what we concluded* about a
question rather than a particular arm, start from [`questions.md`](questions.md).

**Ladder reset 2026-09-19.** E3 and everything before it ran on the pre-P17 engine, whose action
space, Grain Sales decision stream and Missile Envy card handling all differ. No number from it is
comparable, so the registry restarts here. The old one, with its findings distilled, is
[`archive/E3_ladder/`](archive/E3_ladder/README.md).

## E4 — the post-P17 engine

Engine baseline: Grain Sales flattened to one decision (`aa5ec64`), Missile Envy starred-card
removal fixed (`14745cf`), action space repacked 212 → 220, observation v2.3 unchanged at 3,824.

**The attempt number is the intervention, the suffix is the seed.** Fixed for E4 so the pair
stays legible:

| attempt | what it is |
|:---|:---|
| **E4-01** | **unpooled**, default architecture — `opponent_frac 0.0`, no self-pool |
| **E4-02** | **pooled**, default architecture — `opponent_frac 0.3`, self-pool, capacity 12 |
| **E4-03** | **pooled, late-E3 architecture, cold** — `identity_dim 16`, `per_entity_heads 64`, `graph_layers 0`, `self_transform` |
| **E4-04** | **pooled, default architecture, cold** — the control for E4-03; only the network differs |
| **E4-05** | **P21 rung M0** — flat MLP, the ladder's floor |
| **E4-06** | **P21 rung M1** — grouped positional projections |
| **E4-07** | **P21 rung M2** — both per-entity heads |
| **E4-08** | **P21 rung M2d** — country head only. Also the **35-arm seed census** that measured the side collapse |
| **E4-09** | **P21 rung M2e** — card head only |
| **E4-10** | **collapse attribution** — `--seed` split into initialisation / sampling / deals / opponent-draw, one moved at a time |
| **E4-13** | **P21 rung M2a** — head without trunk context |
| **E4-14** | **P21 rung M2b** — head without the 11 per-type constants |
| **E4-15** | **P21 rung M2c** — head with dynamic slots only |
| **E4-16** | **P21 rung M2.5** — M2d plus a learned per-country identity vector |
| **E4-17** | **P21 rung M2.5b** — identity replacing the 11 per-type constants; the ladder's reproducible collapse |
| **E4-18** | **P21 rung M2.5c** — M2.5 plus the card head, with identity |
| **E4-23** | **P22 width probe** — M2d at `entity_proj_dim` 512 |
| **E4-24** | **P22-a** — M2d plus the identity-keyed card lookup |

**A continuation is not an attempt.** Taking an arm further on the same seed keeps its name —
`E4-08-03` covers 0–80M, 80–160M and 160–240M in three directories — so its snapshots read
`E4-08-03@160M`, `E4-08-03@240M`. Attempt numbers 11, 12, 19, 20, 21, 22 and 25 were spent on
continuations and branches before that rule was applied, and were **renamed on 2026-09-22** into
the lineages they continue ([`method/run_nomenclature.md`](method/run_nomenclature.md)). They are
retired rather than reused, because reports written before the rename still carry them.

**E4-01 and E4-02 are the new baseline**, owner's decision 2026-09-19, and their architecture is
*not* E3's. Both were launched with the bare defaults by mistake
([`findings/training/e4_architecture_discontinuity.md`](findings/training/e4_architecture_discontinuity.md));
rather than discard 560M steps the discontinuity is accepted and recorded, so no E3 number is
comparable to an E4 one.

**E4-03 exists because the collapse may belong to the architecture.** E3's collapse appeared only
on its late architecture, which was also the strongest network E3 produced. E4-02 ran 320M pooled
on the *default* architecture and did not collapse. So "the pool prevents collapse" and "this
architecture does not collapse" both fit everything measured, and they are different claims.
E4-03 holds the pool fixed and puts the network back, which separates them.

**E4-03 and E4-04 are an A/B pair at 80M**, both cold-started and both pooled, differing only in
the network. E4-02-01 cannot serve as E4-03's control: it is warm-started, which biases exactly
the first tens of millions of steps the comparison covers. 80M first because the precondition for
hunting a collapse in the late-E3 architecture is that it is actually stronger on this engine --
which is what E3 reported for it, and what this pair tests.

A third unintended difference surfaced while setting this up: **every late E3 arm was a cold
start, and both E4-01 and E4-02 were warm-started.** The cause was the same template, whose
numbered "Phase 0 / Phase 1-3" framing reads as a mandatory pipeline; `CLAUDE.md` now says
plainly that a warm start is optional and is not what the ladder does.

So `E4-01-01` beside `E4-02-01` is visibly the pooled/unpooled comparison at one seed, and
`E4-02-01` beside `E4-02-02` would be visibly a seed pair. E4-01 began as an accident — the pool
flags were omitted at launch — and is kept as the unpooled arm because the comparison is worth
having deliberately.

| arm | varies | budget | directory | writeup |
|:---|:---|---:|:---|:---|
| **E4-01-01** | unpooled: `frac 0.0`, `self_pool False` | 240M, aborted at 184M | `E4-01-01_20260919_003959` | [`log/E4_pool_starvation_recurrence.md`](log/E4_pool_starvation_recurrence.md) |
| **E4-02-01** | pooled: `frac 0.3`, self-pool, capacity 12 | 240M, **complete** | `E4-02-01_20260919_040456` | [`log/E4_round_robin_240M.md`](log/E4_round_robin_240M.md) |
| **E4-03-01@80M** | late-E3 architecture, cold, pooled | 80M | `E4-03-01_20260919_140716` | [`log/E4_architecture_ab_result.md`](log/E4_architecture_ab_result.md) |
| **E4-04-01** | default architecture — E4-03's matched control | 80M | `E4-04-01_20260919_122721` | [`log/E4_architecture_ab_result.md`](log/E4_architecture_ab_result.md) |
| **E4-05-01** | P21 M0, flat MLP | 80M | `E4-05-01_20260919_164141` | [`log/P21_M0_flat_mlp.md`](log/P21_M0_flat_mlp.md) |
| **E4-06-01** | P21 M1, grouped projections | 80M | `E4-06-01_20260919_175806` | [`log/P21_M1_grouped.md`](log/P21_M1_grouped.md) |
| **E4-07-01** | P21 M2, both per-entity heads | 80M | `E4-07-01_20260919_191647` | [`log/P21_M2_lookup.md`](log/P21_M2_lookup.md) |
| **E4-08-01@80M** | P21 M2d seed 1 — **collapsed**; produced a since-withdrawn conclusion | 80M | `E4-08-01_20260919_211756` | [`log/P21_M2d_country_head_collapse.md`](log/P21_M2d_country_head_collapse.md) |
| **E4-08-\*** | **sweep, 34 arms / 32 seeds** — the M2d seed census | 80M each | `E4-08-{01..35}_*` | [`findings/training/side_collapse.md`](findings/training/side_collapse.md) |
| **E4-08-13@80M** | the M2d **rung representative** — median of the rung at both budgets | 80M | `E4-08-13_20260920_080342` | [`log/P21_M2d_160M_slope.md`](log/P21_M2d_160M_slope.md) |
| **E4-09-01** | P21 M2e, card head only | 80M | `E4-09-01_20260919_215818` | [`log/P21_M2d_country_head_collapse.md`](log/P21_M2d_country_head_collapse.md) |
| **E4-10-\*** | **sweep, 8 arms** — one seed stream moved at a time, both directions | 80M each | `E4-10-{01..08}_*` | [`log/E4_collapse_attribution.md`](log/E4_collapse_attribution.md) |
| **E4-08-\*@160M** | **sweep, 28 arms** — every non-collapsed M2d arm continued to 160M on its own seed | 80M → 160M | `E4-08-NN_20260921_*` | [`log/P21_M2d_160M_slope.md`](log/P21_M2d_160M_slope.md) |
| **E4-08-{01,12,22,26}@160M** | collapse censoring test — four arms scored COLLAPSED at 80M, continued to see whether they recover | 80M → 160M | `E4-08-{01,12,22,26}_20260922_*` | [`log/E4_collapse_is_recoverable.md`](log/E4_collapse_is_recoverable.md) |
| **E4-08-01-2**, **E4-08-01-3** | replicates of the collapsing seed 1 — byte-identical flags; `-3` also resumes every 5M | 80M | `E4-08-01-{2,3}_20260920_*` | [`log/P21_M2d_country_head_collapse.md`](log/P21_M2d_country_head_collapse.md) |
| **E4-08-03@240M** | M2d seed 3 continued 160M → 240M — does plain training still add strength? | 160M → 240M | `E4-08-03_20260922_202204` | [`log/P21_M2d_240M.md`](log/P21_M2d_240M.md) |
| **E4-08-14@240M** | the late collapse continued — censoring test; **never recovered** | 160M → 240M | `E4-08-14_20260922_123720` | [`log/E4_collapse_is_recoverable.md`](log/E4_collapse_is_recoverable.md) |
| **E4-08-14@160M** | the **late collapse** — clean at 80M, collapsed at 106M, −249 Elo against its own 80M self | 160M | `E4-08-14_20260921_060659` | [`findings/training/side_collapse.md`](findings/training/side_collapse.md) |
| **E4-03-01@160M** | the anchor to 160M — **lost 356 Elo**, no detector fired | 160M | `E4-03-01_20260921_091750` | [`findings/training/entropy_inflation.md`](findings/training/entropy_inflation.md) |
| **E4-13-03/05** | P21 M2a — head without trunk context | 160M / 80M | `E4-13-0{3,5}_*` | [`log/P21_M2abc_head_inputs.md`](log/P21_M2abc_head_inputs.md) |
| **E4-14-03/05** | P21 M2b — head without the per-type constants | 160M / 80M | `E4-14-0{3,5}_*` | [`log/P21_M2abc_head_inputs.md`](log/P21_M2abc_head_inputs.md) |
| **E4-15-03/05** | P21 M2c — head with dynamic slots only | 160M / 80M | `E4-15-0{3,5}_*` | [`log/P21_M2abc_head_inputs.md`](log/P21_M2abc_head_inputs.md) |
| **E4-16-03/05** | P21 M2.5 — M2d plus country identity | 160M / 80M | `E4-16-0{3,5}_*` | [`log/P21_identity_rungs.md`](log/P21_identity_rungs.md) |
| **E4-18-03/05** | P21 M2.5c — identity plus the card head | 160M / 80M | `E4-18-0{3,5}_*` | [`log/P21_identity_rungs.md`](log/P21_identity_rungs.md) |
| **E4-17-03/04/05/06** | P21 M2.5b — identity replacing the per-type constants; **entered a collapse on seeds 3, 4 and 6** (6 recovered by 73M) | 160M / 160M / 80M / 160M | `E4-17-0{3,4,5,6}_*` | [`log/P21_identity_rungs.md`](log/P21_identity_rungs.md) |
| **E4-17-0{3,4}@240M** | censoring test — seed 3 **recovered**, seed 4 did not | 160M → 240M | `E4-17-0{3,4}_20260922_1*` | [`log/E4_collapse_is_recoverable.md`](log/E4_collapse_is_recoverable.md) |
| **E4-17-06-5M.11**, **-5M.12** | branches of seed 6 from its **healthy** 5M state under new seeds 11 and 12 — does it still enter the collapse? **Neither did** | 5M → 60M | `E4-17-06-5M.1{1,2}_*` | [`log/E4_collapse_is_recoverable.md`](log/E4_collapse_is_recoverable.md) |
| **E4-17-06-50M.11**, **-50M.12** | branches of seed 6 from **inside** its pin at 50M under new seeds 11 and 12 — does it escape? **Both did** | 50M → 110M | `E4-17-06-50M.1{1,2}_*` | [`log/E4_collapse_is_recoverable.md`](log/E4_collapse_is_recoverable.md) |
| **E4-23-03/05** | P22 width probe — `entity_proj_dim` 256 → 512 | 160M / 80M | `E4-23-0{3,5}_*` | [`log/P22_width_probe_and_card_lookup.md`](log/P22_width_probe_and_card_lookup.md) |
| **E4-24-03/05** | P22-a — identity-keyed card lookup | 160M / 80M | `E4-24-0{3,5}_*` | [`log/P22_width_probe_and_card_lookup.md`](log/P22_width_probe_and_card_lookup.md) |

**Sweeps are one row each** (maintenance rule 7 in [`plans/README.md`](plans/README.md)). 111
E4 run directories exist, under 75 short names — a continuation adds a directory, not a name — and
listing each would bury every other arm. Individual rows go to rung representatives and to anomalies — the
collapsed seed, the late collapse, the anchor that fell.

**Seeds 3 and 5 are the ladder's working pair** from E4-13 onward, not seed 1. Both are clean for
M2d at both budgets and already rated there, so a rung-vs-M2d comparison is seed-matched; seed 1
is the seed on which M2d collapses.

E4-01-01 is kept deliberately. It is a second independent instance of pool starvation, reached by a
different route than the X4b `dirname` bug, and its first 105M steps are a healthy 35% → 95% climb
that stands as evidence the post-P17 engine trains normally.

## What the 240M round robin showed

[`log/E4_round_robin_240M.md`](log/E4_round_robin_240M.md). 42,000 games, 100 per seat per pair.
The pooled arm has one sharp, isolated **US-seat dip at 200M** -- 39% to 11% against an
independent reference while the USSR seat does not move -- that fully recovers, with `@240M`
topping the field. The unpooled arm peaks at 160M and falls 8pp by 180M, which is where its
self-play slide began, but its last snapshot is 180M so the steep section is not in the field.
The USSR seat is stronger in **both** arms at every budget, which is the game's asymmetry rather
than a pathology.

The sharpest result is negative: at 180M the unpooled arm's self-play `us_win_rate` read 0.02
while the same checkpoint scored 23% as US against fixed external references. Side balance in
self-play is not a measure of strength.

## What to establish first on this ladder

The ladder has no anchors yet. Until it does, every E4 number is relative to `HeuristicBot` and
`RandomBot`, which are rule-based and therefore the only things comparable across the engine change.

1. A frozen anchor from E4-02-01's final snapshot, and a round robin over its snapshots — the E3
   lesson was that no live metric rates an arm past ~120M.
2. Re-baseline the blunder probes. Every rate logged by a run before `ai/eval/blunders.py` was
   fixed on 2026-09-19 is wrong; measure from snapshots.
3. Re-ask the open questions carried over in
   [`archive/E3_ladder/README.md`](archive/E3_ladder/README.md), rather than re-reading their old
   answers.
