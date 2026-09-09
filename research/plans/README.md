# Plans — what is queued, in what order, and why

This directory is the **queue**. Each file is one step that has not been run yet. `experiments.md`
is the **record** of what was run. A step lives in exactly one of the two places at a time.

## The current goal

A no-search player at the level of a **mediocre human**: no simple mistakes, basic strategy. In
this game that means: a sane setup (Poland ≥ 3 as USSR, West Germany ≥ 4 as US), contesting
battlegrounds instead of leaving ~7.5 of them empty from turn 8, not losing to its own DEFCON,
and — as USSR — not leaving 1–2-influence footholds exposed to Voice of America while VOA is
unaccounted for. Elo against the heuristic bot is the secondary number; the probes are the
acceptance test. Search is acceptable as a diagnostic and as a distillation source, not as the
product. The literature behind the ordering is in [`../references.md`](../references.md).

## The queue

| step | file | what it tests | budget | gate |
|:---|:---|:---|:---|:---|
| P0 | [instruments](P0_instruments.md) | setup probe, VOA-exposure probe, chance-variance decomposition, pre-deal calibration | CPU/eval only, ~1 day | none — do first |
| P1 | [categorical value + advantage filtering](P1_categorical_value_advantage_filtering.md) | replace the two scalar heads by one categorical VP head; filter low-advantage samples | 2 arms × 2 seeds × 80M, then confirm | P0 probes exist |
| P2 | [chance-aware targets](P2_chance_aware_targets.md) | exact dice expectation in value targets; bootstrap at the pre-deal `TURN_CLEANUP` node | 1 arm × 2 seeds × 80M, then confirm | P0 decomposition says dice/deal matter; **owner approval for the bindings helper** |
| P3 | [determinized search → expert iteration](P3_determinized_search_expert_iteration.md) | how much strength is in the value function but not the policy; distil it if the gap is large | eval ~1 day; 1 arm if triggered | P1/P2 winner exists |
| P4 | [setup: macro-action credit](P4_setup_macro_action_credit.md) | λ=1 inside a placement block, bootstrap at its boundary | 1 arm × 2 seeds × 80M | P0 setup probe |
| P5 | [oracle critic](P5_oracle_critic.md) | the implemented-but-never-measured privileged critic as a deal-side variance reducer | 1 arm × 2 seeds × 80M | after P2, so it is not confounded |
| P6 | [attention backbone](P6_attention_backbone.md) | global attention over country + card tokens | 1 arm × 2 seeds × 80M | perturbation probe still flat after P1–P2 |
| P7 | [human data](P7_human_data.md) | ~3,000 strong one-perspective games: conversion + strength-split instruments; positions as a small start pool; critic-only targets; a §22 replication as the discriminator | conversion now; 3 arms × 2 seeds × 80M | conversion first; the arms slot in once positions exist |
| — | [reserve](reserve.md) | annealed shaping, league, AIVAT evaluation, human-policy anchors, the cloud consolidation run | — | triggered by specific measurements |

Order is by expected information per GPU-hour, and P4 may move ahead of P2 if P0's setup probe
is as bad as the anecdotes say — it is the cheapest change aimed at the most visible failure.
P7's conversion is engineering time, not GPU time, and runs alongside P0–P2; its arms take the
next free slot once positions exist.

## Budget rule

An 80M-step arm is ~1.5 h on the 4090; gains are still measurable at 240M (`experiments.md`
§23). So: **screen** every factor at 2 seeds × 80M (~3 h); **confirm** only the winner of a
screen at 2 seeds × 240M (~9 h); rate the last four snapshots of a run, not the final one
(`metrics.md` §20); never compare across budgets. A tournament number is reproducible to
~1.5 points at 1,000 games, and a rating to ~20 Elo between runs — an arm that lands inside that
is *neutral*, not *better*. Read `metrics.md` §1 before adding a new instrument.

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
