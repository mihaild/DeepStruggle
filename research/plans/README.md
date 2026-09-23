# Plans — what is queued, in what order, and why

This directory is the **queue**. Each file is one step that has not been run yet.
[`../log/`](../log/README.md) is the **record** of what was run. A step lives in exactly one of the
two places at a time, and a step that finishes moves to `../archive/` rather than being deleted —
see maintenance rule 3.

> **Reorganised 2026-09-21.** Twelve closed plans moved out of this directory into
> `../archive/E3_ladder/plans/` and `../archive/E4_ladder/plans/`. The queue below is E4-era; the
> E3 tiering that used to stand here is archived with the programme it belonged to
> ([`../archive/E3_ladder/plans/P15_breaking_the_cycle.md`](../archive/E3_ladder/plans/P15_breaking_the_cycle.md)).

## The current goal

A no-search player at the level of a **mediocre human**: no simple mistakes, basic strategy. In
this game that means a sane setup (Poland ≥ 3 as USSR, West Germany ≥ 4 as US), contesting
battlegrounds instead of leaving ~7.5 of them empty from turn 8, not losing to its own DEFCON,
and disposing of a card it must not play through the exit that card has — spacing it, running it
through UN Intervention, or holding it — rather than spending the scarce exit on the card that had
its own. Elo against the heuristic bot is the secondary number; the probes are the acceptance
test. Search is acceptable as a diagnostic and as a distillation source, not as the product. The
literature behind the ordering is in [`../method/references.md`](../method/references.md).

## Where things stand

The E3-era framing that used to open this file — *"every configuration stalls after ~160M, so P15
and its instruments first"* — belonged to a programme that is now archived. E4 replaced the
question rather than answering it: **the architecture turned out to dominate**, and most of the
strength the late-E3 bundle had is one mechanism.

What E4 has established, collected in
[`../log/P21_ladder_status.md`](../log/P21_ladder_status.md):

* Against the ~440 Elo from a flat MLP to the anchor, **grouped positional projections are worth
  +109 and the country per-entity head +282**. The card head is worth **−7**.
* **M2d has not plateaued at 80M**: +159.8 Elo from 80M to 160M, within-seed over six seeds.
* M2d runs at **~50,000 steps/s against the anchor architecture's ~11,600** — 4.3×. At matched
  wallclock it beats the anchor decisively; at matched steps and 80M the anchor is still ~50 Elo
  ahead.
* **No part of the country head's input can be removed**
  ([`../log/P21_M2abc_head_inputs.md`](../log/P21_M2abc_head_inputs.md)).

And two failure modes, which together are why nothing here should be read off a single arm:

* **Side collapse** ([`../findings/training/side_collapse.md`](../findings/training/side_collapse.md),
  revised by [`../log/E4_collapse_is_recoverable.md`](../log/E4_collapse_is_recoverable.md)) —
  **entry** is common (12.5% of seeds are inside an episode at 80M) and bitwise reproducible, but
  it is **not an outcome**: 5 of 7 arms scored COLLAPSED recover when continued, and a collapse
  that recovers costs **−1.4 Elo**, i.e. nothing. A collapse that does *not* recover costs
  **−394**. Nothing measured at onset predicts which follows except `adv_std_raw` returning.
* **Entropy inflation** ([`../findings/training/entropy_inflation.md`](../findings/training/entropy_inflation.md))
  — the anchor lost **356 Elo** over its second 80M with *every* collapse indicator healthy. One
  arm, no replicate.

**The ~160M stall is untested on the new architecture, not refuted.** M2d climbs through the
80M→160M window; nothing has run it past 160M.

**The anchor is not a stable reference above 80M.** Rungs compared at 80M are unaffected — that is
where it was measured and where it is healthy.

## The queue

### Active

| item | what | cost | status |
|:---|:---|:---|:---|
| **[P23](P23_merged_influence_E4_1.md)** | **E4.1**: "influence, first point in X" as one decision, defined as the composition of the two E4 steps it replaces, so it is opt-in per agent and bit-identical to E4 when off. ~11% fewer decisions per game; then the ops budget in the observation (separate approval) before any long run | ~1 day of engineering + ~2 GPU-h of A/B | **running** -- implemented (`d6c89ad`), A/B from scratch in progress |
| **[P24](P24_league.md)** | an AlphaStar-style league (main agent, main exploiter reset to the main's early snapshot, league exploiter) with no supervised policy to reset to. A small trainer change (a pool that grows from other runs' directories) plus a driver over `tools/train.py` | 3 concurrent learners; judged at matched **wallclock** | **proposed**, gated on the P23 verdict and the slow-π_ref replicate |
| **[P22](P22_card_lookup_attention.md)** | identity-keyed card lookup — P22-a | ~84k params | **done, rejected**: −154 / −72 at 80M ([`../log/P22_width_probe_and_card_lookup.md`](../log/P22_width_probe_and_card_lookup.md)); archive with the next reorganisation |
| width probe | `E4-23-03/05`: `entity_proj_dim` 256 → 512 | 2 arms | **done, rejected**: −129 to −291 ([`../log/P22_width_probe_and_card_lookup.md`](../log/P22_width_probe_and_card_lookup.md)) |
| **[P21](P21_architecture_ladder.md)** | the architecture ladder. **11 of 14 rungs done** — M0, M1, M2, M2d, M2e, M2a, M2b, M2c, M2.5, M2.5c, M2.5b. **Nothing has beaten M2d**: all three removals lose and all three identity additions lose. M3/M4/M5 remain and are **blocked** on the `board_mode`/`card_mode` split that [P22](P22_card_lookup_attention.md) carries | ~1.3 GPU-h per rung | **M2 family closed** |

### Queued, unblocked

| item | what | why now |
|:---|:---|:---|
| **anchor replicate** | a second anchor arm to 160M | the 356 Elo fall is one arm with `seed = None`. ~3.9 GPU-h, and it gates whether "the late-E3 architecture degrades" can be said at all |
| **`ref_update_freq` ablation** | run: 5M from scratch **loses** (E4-26, −168 / −68 at 80M); 5M from 160M **wins** (E4-31-03, +83 over the seed-11 baseline at 240M, one seed); 100k from 160M rated in `late_round2_240M`; 5M from 80M (E4-34-03) and the seed-5 replicate (E4-31-05) running | [`../log/E4_dynamics_ref_lambda.md`](../log/E4_dynamics_ref_lambda.md), [`../log/E4_late_dynamics.md`](../log/E4_late_dynamics.md) |
| **[P18](P18_training_throughput.md)** | why one training arm cannot fill the GPU | its stated gate (E4-01-01's continuation) passed long ago, and throughput is now a *first-class* ladder axis — the 4.3× gap between architectures is most of why M2d wins |
| **[P0](P0_instruments.md)** | the remaining probes | needs a status pass first: several instruments it proposes now exist (`watch_run.py` health alarms, the collapse detector, `ladder_report.py`) |

### Queued, no GPU

| item | what |
|:---|:---|
| [P7](P7_human_data.md) 7b–7d | 7a is done — conversion, BC warmup and injection all have log entries. The arms remain |
| [P13](P13_one_game_driver.md) | one game driver — removes the five-loop divergence that has already produced search bugs |
| [P11](P11_scaling_out.md) | running the programme in parallel on rented GPUs. Nothing rented, no money spent |

### Behind the ladder

Tiered behind the E3 stall programme and **not re-ranked against what E4 knows**. Re-rank before
running any of them.

| item | what it tests | note |
|:---|:---|:---|
| [P8](P8_teach_the_defcon_conjunction.md) | auxiliary risk head for provoked DEFCON-1 | instruments exist |
| [P4](P4_setup_macro_action_credit.md) | macro-action credit for placement blocks | rebase the file first |
| [P2](P2_chance_aware_targets.md) | dice expectation + pre-deal bootstrap in value targets | **read the E3-22 lesson first**: a better offline return estimate lost by 520 Elo. Judge on training outcome, never on offline target quality |
| [P5](P5_oracle_critic.md) | oracle critic as a deal-side variance reducer | rebase the file first |
| [reserve](reserve.md) | ideas with triggers — PSRO-lite meta-Nash sampling, optimism/extragradient, per-side capacity | |

## Archived programmes

| where | what |
|:---|:---|
| [`../archive/E4_ladder/plans/`](../archive/E4_ladder/plans/P19_architecture_ab.md) | **P19** — is the late-E3 bundle stronger on this engine? Answered: +447 Elo. **[P20](../archive/E4_ladder/plans/P20_positional_board_encoder.md)** — the pooled board path is unmotivated without a graph. Answered *by P21*, which built the positional path as a side effect of a different question: M1 alone is +109 |
| [`../archive/E3_ladder/plans/`](../archive/E3_ladder/plans/P15_breaking_the_cycle.md) | P1, P3, P6, P9, P10, P14, P15, P16, P17 (×3), restoring_advantage_signal — the E3 stall programme and the engine work under it |

## The baseline these arms run against

- **Engine:** post-P17. One observation layout; old-layout checkpoints are refused, not misread.
  The observation is not to be changed without asking (`CLAUDE.md`).
- **Architecture / matched control:** **M2d** — `--arch ladder --ladder-input-mode grouped
  --ladder-aggregation flatten --drop-static --per-entity-heads 64 --ladder-head-entities
  country`, with `--ladder-head-context` and `--ladder-head-static` both on. Its representative is
  **seed 13** (`E4-08-13@80M` at 80M, `E4-08-13@160M` at 160M), chosen as the rung's median on *both*
  budgets so it biases later comparisons in neither direction.
- **Do not use seed 1 for a ladder arm.** It is the seed on which M2d collapses, and it already
  produced one published-then-withdrawn conclusion. Seeds 3 and 5 are clean for M2d at both
  budgets and already rated there, which makes a rung-vs-M2d comparison **seed-matched**.
- **Every arm gets the collapse detector**, but a `COLLAPSED` verdict means *"inside an episode at
  the budget's end"*, not *"failed"*. Do not re-run on it and do not read the arm's Elo as the
  rung's strength — extend it instead, or record it as censored. Re-run only when `adv_std_raw`
  fails to return, which is the case that costs 394 Elo. Record the per-rung **entry** count: it is
  the free observable, and M2.5b entering on 3 of 3 seeds is the one architecture-linked result it
  has produced.
- **Before reading any A/B**, diff the configs with `tools/scripts/launch_flags.py --diff`.

## Budget rule

Throughput is now architecture-dependent to a degree that changes planning, so quote both:

| architecture | steps/s | 80M | 160M |
|:---|---:|---:|---:|
| M2d-class (ladder, positional) | ~50,000 | ~27 min | ~53 min |
| late-E3 bundle (the anchor) | ~11,600 | ~1.9 h | ~3.9 h |

**Screen** every factor at 1–2 seeds × 80M; **confirm** only a screen's winner at 2 seeds at the
larger budget; never compare across budgets. Seed variance is **~95–100 Elo between seeds** and
effects under **~40 Elo are not measurable at affordable seed counts**
([`../archive/E3_ladder/findings/seed_variance.md`](../archive/E3_ladder/findings/seed_variance.md)).

**The collapse tax is much smaller than it looked.** Entry happens to ~12.5% of seeds by 80M, but
an arm that enters and recovers is **worth as much as one that never entered** (−1.4 Elo, 0.05
pooled sd), so entry is not a reason to re-run. Only a *non-recovering* arm costs anything, and
that is 2 of 7 observed. Budget for roughly **one re-run in twenty arms**, not one in eight — and
detect it by `adv_std_raw` failing to return above ~0.05, not by `us_episode_frac` pinning, which
flagged four arms that were fine
([`../log/E4_collapse_is_recoverable.md`](../log/E4_collapse_is_recoverable.md)).

And read [`../log/variance_and_noise.md`](../log/variance_and_noise.md) before quoting anything:

- rate **four late snapshots** and compare arms by the **pooled head-to-head over all sixteen
  snapshot pairings**; a single-cell comparison cannot resolve below ~50 Elo;
- a continuation seed is worth ~15 Elo and a leg's *gain* carries **±16 Elo**;
- **never compare Elo across tournaments.** Bradley-Terry is field-relative: one unchanged anchor
  checkpoint has rated 2074.2, 2089.7, 2093.3 and 2149.9 across four fields;
- every snapshot is a branch point and `--seed` works on a resume, so paired continuations from
  one state are cheap — but they still differ by the seed floor, so branching does not substitute
  for two seeds.

An arm inside these bars is *neutral*, not *better*. Read
[`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md) before adding an
instrument.

## How to maintain this directory

1. **Before running a step**, read its file and fill in anything marked *decide before running*.
   If the design changed since it was written, edit the file first so the log entry can quote it.
2. **While running**, keep the file — add the run directory names under *Runs* so a second session
   can find them.
3. **When a step is done** (adopted, rejected, or abandoned with a reason):
   - write the entry under [`../log/`](../log/README.md) — setup, numbers, verdict, caveats, and
     the *Follow-ups* the step file listed, with which of them are being queued;
   - **`git mv` the step file into `../archive/<programme>/plans/`** and delete its row above. Do
     not `git rm` it: a closed plan's diagnosis often outlives its proposal. P20 is the standing
     example — its question was answered by a different programme, and its analysis of *why*
     pooling was the bottleneck remains the clearest in the record;
   - **correct the file's own status header in the same commit.** The header is what a reader
     trusts first, and P21 carried "proposed, not launched" while being the most-executed
     programme in the repo;
   - queue the follow-ups that survived as new step files, each with a row above.
4. **New ideas** go in `reserve.md` with a trigger — the measurement that would promote them to a
   step — not as a step file. A step file means someone intends to run it next.
5. **Do not keep results here.** Partial numbers from a run in progress belong in the run
   directory, final ones under `../log/`. A plan file that has accumulated results is a log entry
   that has not been written yet.
6. **One step, one file.** If a step grows a second hypothesis, split it; arms that test two things
   at once produce results that cannot be attributed
   ([`../log/measurement_bugs.md`](../log/measurement_bugs.md), *evaluation consumed most of a
   training run*).
7. **A sweep is one row, not N.** The M2d seed census produced 62 run directories; recording each
   in [`../runs.md`](../runs.md) would bury every other arm. Record the sweep once with its census
   linked, and give individual rows only to rung representatives and anomalies.

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
