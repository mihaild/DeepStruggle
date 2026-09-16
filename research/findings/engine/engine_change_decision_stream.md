# An engine change that moves no decision

Invariant 13 says never measure against a stale engine, and the reflex that follows is that any
engine change invalidates the comparisons that straddle it. That reflex is right about *absolute*
numbers and wrong as a blanket rule: a correctness change that does not alter the decision stream
cannot alter what a training run experiences, so results either side of it stay comparable.

P14 is the worked example, and it is worth recording because the instinct was to bump the engine
letter and re-run a baseline at roughly double the compute.

## What changed

[`../../plans/P14_one_definition_of_legality.md`](../../plans/P14_one_definition_of_legality.md),
in two parts:

* **The Missile Envy forced-play rule.** `forced_card_id` only ever holds `MISSILE_ENVY`, so the
  disjunct `forced_card_id == card || forced_card_id == MISSILE_ENVY` was always true while a force
  was live and the effective rule became "every card must go to Ops". Both the mask and
  `StateMachine::step` now test phase, in-hand and card identity.
* **One definition of legality.** `StateMachine::step` validates against
  `ActionMask::generate_flat_mask_212` instead of re-deriving each rule, and the mask stopped
  emitting actions nobody can play (`DecisionType::NONE`, a `CHOOSE_BRANCH` outside an event frame,
  a node with no decision player). The blanket "ensure at least one action is legal" fallback went
  with them.

## What it moved: nothing

Both engine builds were compiled side by side — `a09e15a` (before P14) and the fixed build, whose
committed part is `5938e52` — and
the same games were played through each, recording at every step the decision type, the **legal
mask's popcount and hash**, and the action chosen. The mask hash is the sensitive part: an engine
change can alter what is *offered* without altering what a deterministic chooser *picks*, and a
policy sees the offer.

| policy | games | steps | outcome | action | mask |
|---|---:|---:|---:|---:|---:|
| policy-free deterministic walk | 300 | 55,786 | 0 | 0 | 0 |
| E3-20-28 @160M, argmax (pooled) | 256 | 118,472 | 0 | 0 | 0 |
| E3-20-28 @320M, argmax (pooled) | 256 | 116,001 | 0 | 0 | 0 |
| E3-17-25 @160M, argmax (unpooled) | 256 | 95,553 | 0 | 0 | 0 |
| **total** | **1,068** | **385,812** | **0** | **0** | **0** |

Zero divergences of any kind, at every step of every game. Not one mask differed.

## Why that is the expected result, in hindsight

The conditions P14 touches do not arise in play. A 1.54M-position sweep of sampled self-play found
**zero** naturally occurring mask/step disagreements; every class needed a seeded conjunction. The
three C++ tests the change broke were all hand-built states, and two of them were asserting a coup
that the rules forbid — US couping Egypt at DEFCON 2, illegal because Egypt is Middle East — which
passed only because `step` never validated coup targets while the mask did.

So the change fixes what a caller could *construct* and what a crash could *expose*, not what
self-play actually does.

## What this licenses, and what it does not

**Licensed.** The engine letter stays E3. `E3-20-28` remains a matched baseline for `E3-21-28`,
the value-bootstrap arm, which is the difference between running one arm and two. Checkpoints,
datasets and Elo anchors taken before P14 stay comparable to those taken after.

**Not licensed.** "The engine changed but the numbers are probably fine" as a general argument. The
licence here comes from a measurement — 385,812 steps, four policies, mask included — not from the
change looking small. An engine change that *does* move the stream still invalidates everything
that straddles it, and the measurement is cheap enough that there is no reason to assume either
way. Build both engines, replay fixed seeds, diff the masks.

## How to repeat it

`git worktree add --detach <tmp> <old-commit>`, build it into its own directory, then run the same
dump script under each build and diff. Two `ts_engine` modules cannot share an interpreter, so it
has to be two processes writing to files. The scripts used are throwaway; the method is the part
worth keeping.
