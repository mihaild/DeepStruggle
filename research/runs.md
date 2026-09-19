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
| **E4-01** | **unpooled** — `opponent_frac 0.0`, no self-pool |
| **E4-02** | **pooled** — `opponent_frac 0.3`, self-pool, capacity 12 |

So `E4-01-01` beside `E4-02-01` is visibly the pooled/unpooled comparison at one seed, and
`E4-02-01` beside `E4-02-02` would be visibly a seed pair. E4-01 began as an accident — the pool
flags were omitted at launch — and is kept as the unpooled arm because the comparison is worth
having deliberately.

| arm | varies | budget | directory | writeup |
|:---|:---|---:|:---|:---|
| **E4-01-01** | unpooled: `frac 0.0`, `self_pool False` | 240M, aborted at 184M | `E4-01-01_20260919_003959` | [`log/E4_pool_starvation_recurrence.md`](log/E4_pool_starvation_recurrence.md) |
| **E4-02-01** | pooled: `frac 0.3`, self-pool, capacity 12 | 240M | `E4-02-01_20260919_040456` | *in flight* |

E4-01-01 is kept deliberately. It is a second independent instance of pool starvation, reached by a
different route than the X4b `dirname` bug, and its first 105M steps are a healthy 35% → 95% climb
that stands as evidence the post-P17 engine trains normally.

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
