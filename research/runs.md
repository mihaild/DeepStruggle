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
| **E4-26** | **slow π_ref** — M2d with `--ref-update-freq 5000000` |
| **E4-27** | **longer credit horizon** — M2d with `--gae-lambda 0.99` |
| **E4-28** | **online search distillation** — M2d with E3's X4b search CE (coef 0.5, 64 sims, all nodes, 1 in 8) |
| **E4-29** | **λ 0.99 from 160M** — E4-27's change applied only after `E4-08-03@160M` |
| **E4-30** | **λ 0.97 from 160M** — the other direction |
| **E4-31** | **slow π_ref (5M) from 160M** — E4-26's change applied only after `E4-08-03@160M` |
| **E4-32** | **fast π_ref (100k) from 160M** — the other direction |
| **E4-33** | **half the KL coefficient from 160M** — `--eta 0.05` (default 0.1) |
| **E4-34** | **slow π_ref from 80M** — an earlier switch point for E4-31's change |
| **E4-35** | **P25 bench, per-seat advantage norm** — E4-27 (λ 0.99) plus `--per-seat-adv-norm` |
| **E4-36** | **P25 bench, seat balancing** — E4-27 (λ 0.99) plus `--seat-balance` |
| **E4-37** | **P25 bench, slow π_ref** — E4-27 (λ 0.99) plus `--ref-update-freq 5000000` |
| **E4-38** | **P25 bench, WoLF seat weights** — E4-27 (λ 0.99) plus `--wolf-seat-weight` (power 1) |
| **E4-39** | **P25 bench, WoLF on the whole policy objective** — E4-38 with `--wolf-scope policy` |
| **E4-40** | **P25 bench, gentler WoLF** — E4-39 with `--wolf-power 0.5` |
| **E4-41** | **P25 bench, WoLF with a dead zone** — E4-39 with `--wolf-dead-zone 0.15` |
| **E4-42** | **WoLF on the normal recipe** — E4-08 (λ 0.98) plus `--wolf-seat-weight --wolf-scope policy` |
| **E4-43** | **P25 bench, WoLF dead zone with a jump** — E4-41 with `--wolf-dead-zone-mode jump` |
| **E4-44** | **Late-collapse test, WoLF dead zone (shift)** — `E4-08-05@160M` to 240M with E4-41's WoLF flags |
| **E4-45** | **Late-collapse test, WoLF dead zone (jump)** — `E4-08-05@160M` to 240M with E4-43's WoLF flags |
| **E4-46** | **P25 bench, WoLF dead zone (shift) + seat balancing** — E4-41 with `--seat-balance` |
| **E4-47** | **P25 bench, narrower WoLF dead zone** — E4-41 with `--wolf-dead-zone 0.10` |
| **E4-48** | **Late-collapse test, fixed pool** — `E4-08-05@160M` to 240M with no change but the opponent-pool resume fix (`3803d5d`), from a repaired 160M state |
| **E4-49** | **P25 3j–3l bench control** — E4-27's flags (λ 0.99 from scratch), seeds 10–13, 80M |
| **E4-50** | **P25 3j, advantage-normaliser floor c 0.8** — E4-49 plus `--adv-norm-floor 0.8` |
| **E4-51** | **P25 3j, advantage-normaliser floor c 0.5** — E4-49 plus `--adv-norm-floor 0.5` |
| **E4-52** | **P25 3k, one-sided per-seat entropy ceiling** — E4-49 plus `--entropy-ceiling 1.8` |
| **E4-53** | **P25 3l, per-seat KL early stop** — E4-49 plus `--target-kl k`, k from E4-49's own distribution |
| **E4-54** | **Lower entropy coefficient on the recipe** — E4-08 with `--entropy-coef 0.0076` (control for E4.1-02) |

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
| **E4-26-03/05** | `--ref-update-freq 5000000` | 80M / 160M (s5 from scratch) | `E4-26-0{3,5}_*` | [`log/E4_dynamics_ref_lambda.md`](log/E4_dynamics_ref_lambda.md) |
| **E4-27-03/05** | `--gae-lambda 0.99` | 80M / 160M (s5 from scratch) | `E4-27-0{3,5}_*` | [`log/E4_dynamics_ref_lambda.md`](log/E4_dynamics_ref_lambda.md) |
| **E4-28-03** | online search distillation, resumed from `E4-08-03@160M` on seed 3 — its control is E4-08-03's own 160→240M leg | 160M → 200M | `E4-28-03_20260923_001303` | [`log/E4_search_distillation.md`](log/E4_search_distillation.md) |
| **E4-29-03, E4-30-03, E4-31-03, E4-32-03, E4-33-03** | one change each (λ 0.99, λ 0.97, π_ref 5M, π_ref 100k, η 0.05), all resumed from `E4-08-03@160M` on seed 3 — E4-08-03's own 160→240M leg is the step-matched control for all five | 160M → 240M | `E4-{29..33}-03_20260923_*` | [`log/E4_late_dynamics.md`](log/E4_late_dynamics.md) |
| **E4-31-05** | seed-5 replicate of E4-31-03: `E4-08-05@160M` run to 240M with `--ref-update-freq 5000000`. Baseline: E4-08-05's own 160→240M continuation (`E4-08-05_20260923_095214`) | 160M → 240M | `E4-31-05_20260923_122401` | [`log/E4_late_dynamics.md`](log/E4_late_dynamics.md) |
| **E4-34-03** | `E4-08-03@80M` run to 240M with `--ref-update-freq 5000000`: is 80M a better or worse switch point than 160M? | 80M → 240M | `E4-34-03_20260923_095214` | [`log/E4_late_dynamics.md`](log/E4_late_dynamics.md) |
| **E4-36-03/05, E4-37-03/05, E4-35-03** | P25 collapse stress bench: E4-27's flags (λ 0.99 from scratch, collapsed on both seeds) plus one lever each — `--seat-balance` (E4-36), slow π_ref `--ref-update-freq 5000000` (E4-37), `--per-seat-adv-norm` (E4-35, seed 3 only, predicted null). Control: E4-27-0{3,5} at matched steps | 0 → 60M | `E4-36-03_20260923_144424`, `E4-3{5,6,7}-0*_20260923_151*` (launched 15:12–15:14; each `launch_flags --diff` against its E4-27 shows only its lever and `--train-steps`) | [`log/P25_stress_bench.md`](log/P25_stress_bench.md) |
| **E4-38-03/05** | P25 bench, WoLF seat weights: E4-27's flags (λ 0.99 from scratch) plus `--wolf-seat-weight` at power 1 (w_us = 2x, w_ussr = 2(1−x), x the USSR's smoothed pure-self-play share). Control: E4-27-0{3,5} and the other bench levers at 60M | 0 → 60M | `E4-38-03_20260923_175411`, `E4-38-05_20260923_175441` | [`log/P25_stress_bench.md`](log/P25_stress_bench.md) |
| **E4-39-03/05** | P25 bench, WoLF as a per-seat learning rate: E4-38's flags plus `--wolf-scope policy`, so the surrogate, entropy bonus and KL to π_ref are all scaled by the seat's weight. E4-38's surrogate-only weights held entropy at ~1.75 and lost −105 on seed 5. Control: E4-27-0{3,5}; comparison: E4-38-0{3,5} | 0 → 60M | `E4-39-03_20260923_201105`, `E4-39-05_20260923_201135` | [`log/P25_stress_bench.md`](log/P25_stress_bench.md) |
| **E4-40-03/05, E4-41-03/05** | P25 bench, gentler WoLF on λ 0.99 from scratch: E4-39's flags with `--wolf-power 0.5` (E4-40) or `--wolf-dead-zone 0.15` (E4-41). Controls: E4-27-0{3,5}; comparison: E4-39-0{3,5} | 0 → 60M | `E4-40-03_20260923_233501`, `E4-40-05_*`, `E4-41-0{3,5}_*` | [`log/P25_stress_bench.md`](log/P25_stress_bench.md) |
| **E4-42-03/05** | WoLF (`--wolf-scope policy`, power 1) on the normal λ 0.98 recipe: does it cost strength where there is no fast collapse? Control: E4-08-0{3,5} at 60M and 80M | 0 → 80M | `E4-42-03_20260924_003341`, `E4-42-05_20260924_003411` (stalled at ~76–77.6M, see [`log/training_throughput_cpu.md`](log/training_throughput_cpu.md)); continued 75 → 80M in `E4-42-0{3,5}_20260924_02*` | [`log/P25_stress_bench.md`](log/P25_stress_bench.md) |
| **E4-43-03/05** | P25 bench: E4-41's flags (dead zone 0.15) with `--wolf-dead-zone-mode jump`, the full brake outside the zone. Controls: E4-27-0{3,5}; comparison: E4-41-0{3,5} | 0 → 60M | `E4-43-03_20260924_020548`, `E4-43-05_20260924_020618` | [`log/P25_stress_bench.md`](log/P25_stress_bench.md) |
| **E4-44-05, E4-45-05** | The late-collapse test: `E4-08-05@160M` (resume state of `E4-08-05_20260921_033333`, 12-member pool) to 240M with WoLF dead zone 0.15, in shift (E4-44) or jump (E4-45) mode. Every earlier continuation from this state collapsed (plain 195M, seed-11 branch 215M, slow π_ref 235M) | 160M → 240M | `E4-44-05_20260924_023337`, `E4-45-05_20260924_023408` (to 190M; aborted by the stall watchdog, the concurrent-graph deadlock fixed in `7e260ab`), a 190 → 240M continuation was stopped at launch (moved to `checkpoints/_aborted/`) when it came back with 6 of 12 pool members: the pool resume bug, fixed in `3803d5d`. Superseded by E4-48 | [`log/P25_stress_bench.md`](log/P25_stress_bench.md) |
| **E4-46-03/05, E4-47-03/05** | P25 bench, λ 0.99 from scratch: E4-41's flags (WoLF dead zone 0.15, shift) plus `--seat-balance` (E4-46), or with `--wolf-dead-zone 0.10` (E4-47). The shift sharpened but let seed 5 collapse; the jump (E4-43) held the split but did not sharpen. Controls: E4-27-0{3,5}; comparisons: E4-41, E4-43 | 0 → 60M | `E4-46-0{3,5}_*`, `E4-47-0{3,5}_*` | [`log/P25_stress_bench.md`](log/P25_stress_bench.md) |
| **E4-48-05, E4-48-05-160M.11** | The late-collapse test on a fixed pool. Before `3803d5d` every resume recorded its restored pool at step 0; spacing eviction then drained it to 2 pre-resume members, so every continuation from `E4-08-05@160M` (all of which collapsed) trained against its recent self. Plain continuations (no other change) from a repaired copy of the 160M state (`poolfix_resume_160038912.pt`, 12 members at 5, 80, 85…160M, written by `data/logs/p25/repair_pool_160M.py`), seed 5 and a seed-11 branch. Baselines: `E4-08-05_20260923_095214` and `E4-08-05-160M.11_20260923_122421`, the same continuations on the drained pool | 160M → 240M | `E4-48-05_20260924_043225`, `E4-48-05-160M.11_20260924_043255` (died at ~215M on a CUDA `unspecified launch failure`; continued 215 → 240M in `E4-48-05-160M.11_20260924_05*`) | [`log/P25_pool_resume_bug.md`](log/P25_pool_resume_bug.md) |
| **E4-48-03** | The same fixed-pool continuation on seed 3's lineage, from a repaired `E4-08-03@160M` state. Seed 3's drained continuations did not collapse, so this isolates the fix's effect on strength. Baseline: `E4-08-03_20260922_202204` | 160M → 240M | `E4-48-03_20260924_053454` | [`log/P25_pool_resume_bug.md`](log/P25_pool_resume_bug.md) |
| **E4-48-05-160M.12** | Third fixed-pool replicate of E4-48 on seed 5's lineage (seed-12 branch), collapse readout only | 160M → 220M | `E4-48-05-160M.12_20260924_055334` | [`log/P25_pool_resume_bug.md`](log/P25_pool_resume_bug.md) |
| **E4-27-06/07, E4-41-06/07** | New-seed replicates of the bench control (λ 0.99 from scratch) and of E4-41 (WoLF dead zone 0.15, shift), seeds 6 and 7: does the bench collapse on other seeds, and is E4-41's gain a seed-3/5 property? | 0 → 60M | `E4-27-06_20260924_040324`, `E4-41-06_20260924_040354`, `E4-{27,41}-07_20260924_05*` | [`log/P25_stress_bench.md`](log/P25_stress_bench.md) |
| **E4-49..53-10..13** | P25 3j–3l bench: λ 0.99 from scratch, 80M, seeds 10–13, two at a time: control (E4-49), advantage-normaliser floor 0.8 / 0.5 (E4-50 / E4-51), one-sided entropy ceiling 1.8 (E4-52), per-seat KL early stop (E4-53; k = p90 of the controls' per-seat approximate KL over 5–40M). Every earlier λ 0.99 run directory (E4-27, E4-29, E4-35..41, E4-43, E4-46, E4-47; 26 in all) was moved to `checkpoints/_archive/lambda_0.99/` before it | 0 → 80M | `E4-49-10_20260924_075031`, `E4-49-11_20260924_0751*`, … | [`log/P25_stress_bench.md`](log/P25_stress_bench.md#closing-p25-2026-09-24): controls done (0 of 6 pinned); E4-50 floor 0.8 on 4 seeds (one pin, seed 11); E4-51-10 to 40M; E4-52/53 never ran. Stopped when P25 closed |
| **E4-49-03, E4-49-05** | Bench control rerun on the two seeds whose first λ 0.99 controls (E4-27-03/05) pinned, on the current code: neither pinned (peaks 0.94, 0.85). A commit-by-commit check shows no change of behaviour, only 1e-9 rounding in 4124562 and 7766e3d, so the pin rate (2 of 10 plain controls) is the recipe's | 0 → 80M | `E4-49-03_20260924_090531`, `E4-49-05_20260924_090601` | [`log/P25_stress_bench.md`](log/P25_stress_bench.md) |
| **E4-08-36, E4-08-37** | Clean E4-08 replicates (M2d, λ 0.98, pool 0.3/12), from scratch straight to 240M in one run each: no resume, so no drained-pool confound; fixed CUDA-graph code (`9bfceb1`), clang engine. Compared with the E4-08 lineage (seeds 3, 5) and the fixed-pool continuations E4-48-03/05 | 0 → 240M | `E4-08-36_20260924_131544`, `E4-08-37_20260924_131613` | [`log/E4_clean_replicates.md`](log/E4_clean_replicates.md) |
| **E4.1-01-36** | E4.1 (merged-influence view) on the clean recipe: E4-08-36's flags plus `--merged-influence`, seed 36, 160M in one run. The earlier E4.1 A/B collapsed early on 2 of 2 seeds; same-seed baseline E4-08-36 (2199 at 160M). Collapsed again on the US seat, pinned most of 35–130M; −300 at 160M | 0 → 160M | `E4.1-01-36_20260924_151641` | [`log/P23_E4_1_ab.md`](log/P23_E4_1_ab.md) |
| **E4.1-02-36, E4-54-36** | Entropy coefficient 0.01 → 0.0076 (the starting-entropy ratio 2.08 / 2.75), symmetric across seats, on both views, seed 36, 160M: E4.1-02-36 (E4.1) and E4-54-36 (E4 control). A 2×2 grid with E4-08-36 and E4.1-01-36 at 0.01 | 0 → 160M | `E4.1-02-36_20260924_160527`, `E4-54-36_20260924_160557` | [`log/P23_E4_1_ab.md`](log/P23_E4_1_ab.md): neutral on E4 (+56..−45); E4.1 still collapses on the US seat, never recovers by 160M |
| **E4.1-03-36** | E4.1 with the entropy bonus normalised per decision: E4.1-01-36's flags plus `--entropy-normalize --entropy-coef 0.021` (the default average bonus on E4's decision mix). Tests whether the ~50-option op-mode decisions give the losing seat's bonus the room that drives E4.1's collapse | 0 → 160M | `E4.1-03-36_20260924_175915` | stopped at ~35M: pinned by 25–30M, no better than the raw bonus |
| **E4.1-04-36** | E4.1 with no entropy bonus: E4.1-01-36's flags plus `--entropy-coef 0`, seed 36, 160M. Tests whether the bonus has any part in E4.1's US-seat collapse (5 of 5 under raw 0.01, raw 0.0076, normalised) | 0 → 160M | `E4.1-04-36_20260924_181050` | stopped at ~85M: pinned by 80M with no entropy bonus (E4.1 collapses 6 of 6) |
| **E4-56-40..42, E4-57-40..42** | P26 TF32 ablation: the E4-08 recipe (M2d, λ 0.98, pool 0.3/12) from scratch to 80M on `bea6311` (snapshots every 10M, pool every 5M). E4-56 fp32 control, E4-57 `--tf32`; the two arms of a seed run as a pair | 0 → 80M | `E4-5{6,7}-4{0,1,2}_20260924_*` | [`log/P26_quick_screen.md`](log/P26_quick_screen.md): TF32 level or better on every seed (+3 / +156 / +101 head to head, late snapshots), +15% steps/s paired; passes the adoption gates. fp32 E4-56-41 stalled without a pin (−139 over 50–80M) |
| **E4.1-05-36** | E4.1 warm-started from a distillation of E4-08-36@80M with exact action translation (`tools/lib/merged_targets.py`; KL 0.044 at op-choice, top-1 91.9%), then the E4-08 recipe with `--merged-influence`, seed 36, 80M. A warm start: fresh optimizer, π_ref and pool. Compared with E4-08-36@80–160M | 0 → 80M (from E4-08-36@80M) | `E4.1-05-36_20260924_190850`; student `E4.1-05-36-distilled6_20260924_190531` | [`log/P23_E4_1_ab.md`](log/P23_E4_1_ab.md): stopped at ~20M, pinned by 12M; the US policy went near-uniform on every decision type |
| **E4-55-36** | Control for E4.1-05-36: the same warm start in the E4 view. E4-08-36@80M's weights with a fresh optimizer, π_ref and pool, the E4-08 recipe, seed 36, 20M. Compared with E4-08-36's own 80→100M | 0 → 20M (from E4-08-36@80M) | `E4-55-36_20260924_191813` | [`log/P23_E4_1_ab.md`](log/P23_E4_1_ab.md): the US seat also dissolves (entropy fraction 0.06–0.47 → 0.71–0.88), and share drifts to 0.84 by 20M. The reset causes it in both views; E4.1 makes it worse |
| **E4.1-01-03/05** | P23 A/B: M2d in the E4.1 merged-influence view, from scratch. Both collapsed (US) early; level with E4 before, −105 / −360 inside. Paused | 80M / 30M (stopped) | `E4.1-01-0{3,5}_20260923_*` | [`log/P23_E4_1_ab.md`](log/P23_E4_1_ab.md) |
| **E4-08-05-160M.11** | seed branch of seed 5: `E4-08-05@160M` continued to 240M under seed 11, a second seed-5 baseline because the plain seed-5 continuation (`E4-08-05_20260923_095214`) collapsed (US 1% of self-play at 209M) | 160M → 240M | `E4-08-05-160M.11_20260923_122421` | [`log/E4_late_dynamics.md`](log/E4_late_dynamics.md) |
| **E4-08-03-160M.11** | seed branch: `E4-08-03@160M` continued to 240M under seed 11, otherwise identical to E4-08-03's 160→240M leg. Is that leg's 200M dip seed 3's or M2d's? Also a second baseline for E4-29..33 | 160M → 240M | `E4-08-03-160M.11_20260923_072049` | [`log/E4_late_dynamics.md`](log/E4_late_dynamics.md) |
| **E4-28-05** | the same, seed-5 replicate, resumed from `E4-08-05@80M` — its control is E4-08-05's own 80→160M leg | 80M → ~108M (stopped at 07:00) | `E4-28-05_20260923_031034` | [`log/E4_search_distillation.md`](log/E4_search_distillation.md) |

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
