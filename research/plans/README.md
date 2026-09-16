# Plans — what is queued, in what order, and why

This directory is the **queue**. Each file is one step that has not been run yet.
[`../log/`](../log/README.md) is the **record** of what was run. A step lives in exactly one of the two
places at a time.

> `experiments.md`, named throughout this file, was split by programme into `../log/` on
> 2026-09-13 and no longer exists; its section numbers were kept, so a `§N` reference still
> resolves via [`../log/README.md`](../log/README.md). Read "write the entry in `experiments.md`"
> as "write the entry in the programme's file under `../log/`", and see
> [`../method/bookkeeping.md`](../method/bookkeeping.md) for the rest of what a finished step
> owes: a row in [`../runs.md`](../runs.md) and a verdict in [`../questions.md`](../questions.md).

> **Reading old plan files.** Plans below the fold were written across three baselines
> (v2.1/v2.2 layouts, then the single-layout corrected engine, now the E3 series). There is
> exactly one observation layout; a checkpoint from a retired layout is refused by
> `check_checkpoint_layout`, so arms F/F2/G cannot be run and every control number measured on
> them is void. Read any "arm F recipe / arm G snapshot / v2.x" in a plan file as "the current
> baseline recipe" below, and re-derive the control. **P0 is current; the other pre-P15 files
> have not been rebased** — before running one, update its file first (maintenance rule 1).

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
product. The literature behind the ordering is in [`../method/references.md`](../method/references.md).

## The binding constraint, and the priority that follows

**Every configuration stalls after ~160M**: side balance oscillates (bidirectionally — the
runaway side varies by run), the critic degenerates toward the base rate, `adv_std_raw`
collapses, and strategy keeps changing while strength does not. The opponent pool settles the
*balance* half decisively and is suggestive on strength, but does not cancel the stall
([`../findings/training/pooling.md`](../findings/training/pooling.md)). Meanwhile the honest
searcher beats the raw policy ~75% *with the same network*
([`../log/search_cost_and_coverage.md`](../log/search_cost_and_coverage.md)), and there is no
in-run instrument that can rate an arm past ~120M.

Until training converts compute into strength again, no probe-level step (setup, DEFCON,
value-target refinements) can pay off — an arm that trains into the stall measures the stall.
So the queue is tiered: **[P15](P15_breaking_the_cycle.md) and its instruments first**,
no-GPU work in parallel, everything else behind them, re-ranked once the stall's mechanism is
known.

## The queue

### Tier 0 — unblock measurement, confirm the diagnosis (this week; gates everything)

| item | what | cost | gate |
|:---|:---|:---|:---|
| **P15-X0** | frozen-anchor in-run evals (per-side WR vs fixed strong checkpoints) — restores rating above 120M, makes the oscillation quantifiable | eval config only | none |
| **P15-X1** | frozen-exploiter diagnostic (P10 exp 2, never run): does a response to the runaway strategy exist? | ~2h GPU | side-lock in trainer |
| bookkeeping | write up **E3-18-22** (P10 exp 1, the do-nothing control): its one rating, +11 over its parent after +80M, *is* the "no self-recovery" result | none | none |

### Tier 1 — the stall attack (gated by X1 climbing; details in [P15](P15_breaking_the_cycle.md))

| item | what | cost |
|:---|:---|:---|
| **P15-X4** | expert-iteration distillation: CE toward the 32-sim honest search policy, card/play-mode nodes, 1-in-8 — the one gradient source that does not die with the outcome signal. Build starts now, in parallel with X1 | build + ~8h/arm |
| **P15-X2** | anchor timescale: `--ref-update-freq` 200k → {5M, 20M} — the KL anchor currently refreshes every ~16s of wall clock and tracks the cycle it should damp | flags; ~2h/cell |
| **P15-X3** | pool memory: span-the-run pool with δ-mix (a); ~~PFSP (b)~~ **postponed — ran as E3-23-28 and is a null: +11.6 Elo at 160M against an 83–221 Elo seed spread** ([pooling.md §3c](../findings/training/pooling.md)); one-opponent-per-episode (c, only if a moves) | small trainer change; ~2h/arm |

### Tier 2 — parallel, no GPU

| item | what |
|:---|:---|
| [P7](P7_human_data.md) **7a only** | human-corpus conversion + strength-split instruments (CPU/engineering); 7b–7d wait for Tier 1's outcome |
| [P13](P13_one_game_driver.md) | one game driver — removes the five-loop divergence that has already produced search bugs; enabling work for X4-class arms |
| [P0](P0_instruments.md) | the remaining probes (current file; rebased) |
| [P16](P16_replay_policy_and_critic_trace.md) | policy probabilities and critic values recorded on every replay step, and shown in the workbench — turns "it blundered" into "it blundered at p=0.02" and makes the critic's degeneracy visible on a played game |

### Tier 3 — after training converts compute again (re-rank on the X-outcome)

| item | what it tests | note |
|:---|:---|:---|
| [P8](P8_teach_the_defcon_conjunction.md) | auxiliary risk head for provoked DEFCON-1 | instruments exist |
| [P4](P4_setup_macro_action_credit.md) | macro-action credit for placement blocks | rebase file first |
| [P2](P2_chance_aware_targets.md) | dice expectation + pre-deal bootstrap in value targets | **read the E3-22 lesson first**: a better offline return estimate lost by 520 Elo; P2's changes must be judged on training outcome, never on offline target quality |
| [P5](P5_oracle_critic.md) | oracle critic as deal-side variance reducer | rebase file first |
| aux per-country control head | KataGo-style ownership target ([restoring_advantage_signal](restoring_advantage_signal.md) option 3) | complement, not a fix |

### Done, absorbed, or superseded — kept for their text, not for running

| file | status |
|:---|:---|
| [P1](P1_categorical_value_advantage_filtering.md) | **done** — categorical head settled negative (E3-06, −28); advantage filtering settled positive and adopted (E3-07, +25/+24). Verdicts in [`../questions.md`](../questions.md) |
| [P3](P3_determinized_search_expert_iteration.md) | **absorbed into P15-X4** — its diagnostic half is answered (the search gap is ~75%, large), so its trigger fired |
| [P6](P6_attention_backbone.md) | **superseded by the P9 programme** — identity embeddings, self-transform and per-entity heads landed as E3-15 and are the baseline ([`../findings/training/architecture.md`](../findings/training/architecture.md)) |
| [P9](P9_graph_architecture.md) | its programme ran; record in [`../log/P9_architecture.md`](../log/P9_architecture.md); the open map-layer question sits in [`../questions.md`](../questions.md) |
| [P10](P10_opponent_sampling.md) | **absorbed** — exp 1 ran (needs the Tier-0 writeup), exp 4 became the pooled arms, exp 2 is P15-X1; exp 3 (seed-resume) folds into X2's screens |
| [P14](P14_one_definition_of_legality.md) | **done** — landed with zero decision-stream divergence ([`../findings/engine/engine_change_decision_stream.md`](../findings/engine/engine_change_decision_stream.md)) |
| [restoring_advantage_signal](restoring_advantage_signal.md) | **absorbed** — its final proposal (historical opponent sampling) became the pooled arms; the VP-margin idea is dead by measurement; the aux control head moved to Tier 3 |
| [reserve](reserve.md) | ideas with triggers — now including PSRO-lite meta-Nash sampling, optimism/extragradient, per-side capacity |

## The baseline these arms run against

- **Engine:** E3 (post-P14; the mask/step collapse and Missile Envy fix are measured as
  decision-stream-neutral). One observation layout; old-layout checkpoints are refused, not
  misread. The observation is not to be changed without asking (`CLAUDE.md`).
- **Architecture:** the E3-15 recipe — identity embeddings, graph self-transform, per-entity
  residual heads ([`../findings/training/architecture.md`](../findings/training/architecture.md)).
- **Training recipe / matched control:** **E3-20-28** — pooled opponents (frac 0.30, capacity 12,
  snapshots every 5M), `--snapshot-every-steps 5000000`, adv-filter on. It is the strongest
  measured arm (~2150 in `arena_p12` @160M, flat to 320M) and the registered baseline for new
  arms ([`../runs.md`](../runs.md)). Its 80M resume state is the standard healthy resume point.
- **Anchors:** `E2-02-21-480M` is the standing cross-era anchor with a known caveat
  ([`../findings/engine/engine_revisions.md`](../findings/engine/engine_revisions.md)); in-run
  anchors above 120M do not exist until P15-X0.
- **Before reading any A/B**, diff the configs and pool growth with `tools/compare_runs.py` —
  the E3-22 void (eval cadence silently coupled to the pool) is the standing example of why.

## Budget rule

An 80M-step arm is **~2h** on the 4090 (14.2k steps/s unpooled, 10.3k pooled — wall-clock
figures from metadata are configured budgets, not elapsed time). **Screen** every factor at
1–2 seeds × 80M; **confirm** only a screen's winner at 2 seeds at the larger budget; never
compare across budgets. Seed variance is **~95 Elo between seeds** and effects under **~40 Elo
are not measurable at affordable seed counts** — do not run an arm whose expected effect is
smaller ([`../findings/training/seed_variance.md`](../findings/training/seed_variance.md)).
And read `../log/variance_and_noise.md` before quoting anything:

- rate **four late snapshots** and compare arms by the **pooled head-to-head over all sixteen
  snapshot pairings**; a single-cell comparison cannot resolve below ~50 Elo, which is most
  effects worth arguing about (`../log/variance_and_noise.md`, learned again on `staged_cards`);
- a continuation seed is worth ~15 Elo and a leg's *gain* carries **±16 Elo** (same file) — quote a
  within-lineage gain with that bar, or not as a trend;
- never compare Elo across tournaments (~±15 between pools of the same files); within one
  tournament a direct head-to-head carries ~±12 Elo before any seed effect;
- every snapshot is now a branch point and `--seed` works on a resume, so paired continuations
  from one state are cheap — but they still differ by the seed floor, so branching does not
  substitute for two seeds.

An arm inside these bars is *neutral*, not *better*. Read `../method/measurement_pitfalls.md` before adding a new
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
   things at once produce results that cannot be attributed (`../log/measurement_bugs.md`, *evaluation consumed most of a training run*).

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
- [P15 — Breaking the oscillate-then-stall cycle](P15_breaking_the_cycle.md) — **the active programme**: the standard remedies for self-play cycling mapped to this record — frozen-anchor instrument, the exploiter diagnostic, anchor timescale, pool memory, and search distillation
- [P11 — The experiment programme, and running it in parallel](P11_scaling_out.md) — seeds, phases, and cheap GPU options
- [Restoring the advantage signal](restoring_advantage_signal.md) / [P10](P10_opponent_sampling.md) — absorbed into P15; kept for the reasoning and the pooled-arm record
