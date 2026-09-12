# Plans — what is queued, in what order, and why

This directory is the **queue**. Each file is one step that has not been run yet. `experiments.md`
is the **record** of what was run. A step lives in exactly one of the two places at a time.

> **Stale baseline throughout this directory.** Every plan here was written when observation
> layout **v2.2** was the adopted baseline and arm G's 320M snapshot the strongest checkpoint.
> Both are gone, and the situation has since gone further than "v2.2 is retired": **there is now
> exactly one observation layout.** legacy, v2.1 and v2.2 are deleted, and so is every argument
> that could name one — `extract_observation` and `VectorizedBatchRunner` take no layout,
> `TsVectorizedEnv` takes no layout or `obs_flags`, and `tools/train.py` has no `--obs-layout` or
> `--engine-flag`. A checkpoint from a retired layout is refused by `check_checkpoint_layout`
> rather than silently misread, so **arms F, F2 and G cannot be run at all**. The current
> baseline is that one layout on the corrected engine (the starred-card fix), arms H, H2 and I.
> Read "v2.2 / arm F recipe / arm G's 320M snapshot" in the plans below as "the current baseline
> recipe", and re-derive any control number: everything measured on a v2.2 checkpoint is void.
> `research/metrics.md` §1.5.3 also retires the claim that the corrected engine lengthens games,
> which two of these plans lean on. See `CLAUDE.md` for the layout table.
>
> **P0 has since been rewritten against v2.3 and arms H/H2/I and is current. The other six have
> not.**

## The current goal

A no-search player at the level of a **mediocre human**: no simple mistakes, basic strategy. In
this game that means: a sane setup (Poland ≥ 3 as USSR, West Germany ≥ 4 as US), contesting
battlegrounds instead of leaving ~7.5 of them empty from turn 8, not losing to its own DEFCON,
and disposing of a card it must not play through the exit that card has — spacing it, running it
through UN Intervention, or holding it — rather than spending the scarce exit on the card that
had its own. (Not leaving 1–2-influence footholds exposed to Voice of America is on the same
list, but it is behind contesting battlegrounds at all, so its probe sits in
[`reserve.md`](reserve.md).) Elo against the heuristic bot is the secondary number; the probes
are the acceptance test. Search is acceptable as a diagnostic and as a distillation source, not as the
product. The literature behind the ordering is in [`../references.md`](../references.md).

## The queue

| step | file | what it tests | budget | gate |
|:---|:---|:---|:---|:---|
| P0 | [instruments](P0_instruments.md) | setup probe, card-disposal probe, chance-variance decomposition, pre-deal calibration | CPU/eval only, ~1 day | none — do first |
| P1 | [categorical value + advantage filtering](P1_categorical_value_advantage_filtering.md) | replace the two scalar heads by one categorical VP head; filter low-advantage samples | 2 arms × 2 seeds × 80M, then confirm | P0 probes exist |
| P2 | [chance-aware targets](P2_chance_aware_targets.md) | exact dice expectation in value targets; bootstrap at the pre-deal `TURN_CLEANUP` node | 1 arm × 2 seeds × 80M, then confirm | P0 decomposition says dice/deal matter; **owner approval for the bindings helper** |
| P3 | [determinized search → expert iteration](P3_determinized_search_expert_iteration.md) | how much strength is in the value function but not the policy; distil it if the gap is large | eval ~1 day; 1 arm if triggered | P1/P2 winner exists |
| P4 | [setup: macro-action credit](P4_setup_macro_action_credit.md) | λ=1 inside a placement block, bootstrap at its boundary | 1 arm × 2 seeds × 80M | P0 setup probe |
| P5 | [oracle critic](P5_oracle_critic.md) | the implemented-but-never-measured privileged critic as a deal-side variance reducer | 1 arm × 2 seeds × 80M | after P2, so it is not confounded |
| P6 | [attention backbone](P6_attention_backbone.md) | global attention over country + card tokens | 1 arm × 2 seeds × 80M | perturbation probe still flat after P1–P2 |
| P7 | [human data](P7_human_data.md) | ~3,000 strong one-perspective games: conversion + strength-split instruments; positions as a small start pool; critic-only targets; a §22 replication as the discriminator | conversion now; 3 arms × 2 seeds × 80M | conversion first; the arms slot in once positions exist |
| P8 | [teach the DEFCON conjunction](P8_teach_the_defcon_conjunction.md) | auxiliary risk head on a label that sees provoked endings; a narrower blunder window | 2 arms × 2 seeds × 80M | none — instruments exist |
| — | [reserve](reserve.md) | annealed shaping, league, AIVAT evaluation, human-policy anchors, the cloud consolidation run | — | triggered by specific measurements |

Order is by expected information per GPU-hour, and P4 may move ahead of P2 if P0's setup probe
is as bad as the anecdotes say — it is the cheapest change aimed at the most visible failure.
P7's conversion is engineering time, not GPU time, and runs alongside P0–P2; its arms take the
next free slot once positions exist.

## The baseline these arms run against

Observation layout **v2.2** — the decision context in, everything unread out — the default for
`--arch v2` and worth **~+92 Elo** over v2.1 at matched budget, confirmed on a second seed
(`experiments.md` §24): the first observation change in the project to clear noise, and it keeps
gaining at budgets where earlier layouts stopped (arm G: 1957.2 at 320M, beating `dec_turns40`
84.8%). The strongest checkpoint is arm G's 320M snapshot; the clean baseline *recipe* is arm
F's (v2.2, `staged_cards` off — the flag is not demonstrated, §24.1). Consequences for this
queue:

- **The observation is not to be changed without asking** (`CLAUDE.md`). Every step here leaves
  it alone: P1/P5/P6 change heads, losses and the encoder; P2/P4 change the trace.
- v2.2's gain came from *information*, not capacity — which supports the ordering here: fix what
  the network is told (value targets included) before making it bigger (P6 last).
- Controls predating v2.2 are stale; every control is rerun on the current recipe. A
  checkpoint's `engine_config` travels in its `metadata.json` and evaluation matches on
  `(layout, flags)` (§24.2), so old checkpoints stay evaluable but are not controls.

## Budget rule

An 80M-step arm is ~1.5 h on the 4090, and v2.2 still gains at 320M (`experiments.md` §24). So:
**screen** every factor at 2 seeds × 80M (~3 h); **confirm** only the winner of a screen at
2 seeds × 240M (~9 h); never compare across budgets. And read `metrics.md` §20.6–20.7 before
quoting anything:

- rate **four late snapshots** and compare arms by the **pooled head-to-head over all sixteen
  snapshot pairings**; a single-cell comparison cannot resolve below ~50 Elo, which is most
  effects worth arguing about (§20.7, learned again on `staged_cards`);
- a continuation seed is worth ~15 Elo and a leg's *gain* carries **±16 Elo** (§20.6) — quote a
  within-lineage gain with that bar, or not as a trend;
- never compare Elo across tournaments (~±15 between pools of the same files); within one
  tournament a direct head-to-head carries ~±12 Elo before any seed effect;
- every snapshot is now a branch point and `--seed` works on a resume, so paired continuations
  from one state are cheap — but they still differ by the seed floor, so branching does not
  substitute for two seeds.

An arm inside these bars is *neutral*, not *better*. Read `metrics.md` §1 before adding a new
instrument.

## How to maintain this directory

1. **Before running a step**, read its file and fill in anything marked *decide before running*.
   If the design changed since it was written, edit the file first so the log entry can quote it.
2. **While running**, keep the file — add the run directory names under *Runs* so a second
   session can find them.
3. **When a step is done** (adopted, rejected, or abandoned with a reason):
   - write the entry in `experiments.md` — setup, numbers, verdict, caveats, and the
     *Follow-ups* the step file listed, with which of them are being queued;
   - `git rm` the step file and delete its row above; the log entry replaces it;
   - queue the follow-ups that survived as new step files, each with a row above;
   - if a settled result changes a gate or the ordering of the remaining steps, update the table
     and say why in the same commit.
4. **New ideas** go in `reserve.md` with a trigger — the measurement that would promote them to a
   step — not as a step file. A step file means someone intends to run it next.
5. **Do not keep results here.** Partial numbers from a run in progress belong in the run
   directory, and final ones belong in `experiments.md`. A plan file that has accumulated
   results is a log entry that has not been written yet.
6. **One step, one file.** If a step grows a second hypothesis, split it; arms that test two
   things at once produce results that cannot be attributed (`metrics.md` §1.3).

## Step file template

```
# Pn — <name>

**Status:** queued | running | blocked on <what>
**Gate:** <what must be true before this runs>
**Needs approval:** <engine / bindings changes, or "none">

## Goal
## Why (which measured failure, which finding, which paper)
## Change (files, flags, what stays fixed)
## Procedure (arms, seeds, budget, control)
## Measure (probes first, Elo last)
## Decision rule (adopt / kill / extend — decided before running)
## Follow-ups (what to queue depending on the outcome)
## Runs (filled in while running)
```
