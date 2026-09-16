# P14 — one definition of legality

`ActionMask::generate_flat_mask_212` and `StateMachine::step` each decide what is legal, from
separate code. Where they differ, the mask offers an action the engine refuses: the caller either
crashes (since `59d2331`) or, before that, silently failed to advance while the trainer credited
the transition anyway.

This plan collapses them to one definition and fixes the rule defects the collapse would otherwise
freeze in place.

## Why now

`E3-21-28` died twice at 4.0M of 160M steps on `engine refused flat action 112 in env 39
(SELECT_PLAY_MODE); the action WAS legal in the mask it was sampled from`. That is the first time
this class has been *visible*. It has been present for the life of the project.

## What is actually wrong

Six of twelve hand-constructed forced-play conditions disagree, and a sweep over 1.54M sampled
positions found four further classes, two of which deadlock — the mask's entire legal set is
refused, so the position cannot advance at all.

| # | class | mask | step | correct side |
|---|---|---|---|---|
| 1 | forced play during `HEADLINE` | EVENT/OPS/SPACE | OPS only | **mask** — the card says "next action round" |
| 2 | forced play, China Card pending | OPS/SPACE | OPS only | **mask** |
| 3 | forced play, **scoring card** pending (via Grain Sales) | EVENT only | OPS only → **deadlock** | **mask** — a scoring card has no Ops value |
| 4 | `CHOOSE_BRANCH` with `resolving_card == 0` | two branches | no case → **deadlock** | neither; the mask branch is dead code |
| 5 | `DecisionType::NONE` | flat 211 | no case → refuses | neither; NONE should not be a live decision |
| 6 | `decision_player == NONE` at an interactive node | substitutes `phasing_player` | refuses | **step** — the mask invents legality |

Class 6 is the one that matters for the design: it is the single case where the mask is the
permissive, wrong side. A blanket "trust the mask" would legalise it.

### The underlying rule defect

`forced_card_id` is written in exactly two places (`events/mid_war.cpp:164`,
`card_dispatcher.cpp:1028`) and **both assign `card_ids::MISSILE_ENVY`**. There is no other forced
card. So in

```cpp
forced_card_id == card || forced_card_id == MISSILE_ENVY
```

the second disjunct is true whenever a force is live, the first is dead, and the effective rule is
*"while forced, every card must go to Ops"* — not the printed rule, which constrains **which card
you must play**.

It is also never cleared on a timer. Writers: the two above. Clearers: `init_new_game`, the
trap-discard branch (`state_machine.cpp:888`, only when the discarded card *is* the forced card),
and an OPS play (`:980`). Every other route by which Missile Envy can leave the hand leaves the
flags set. Measured: **21 positions in 1,280,000 had `forced_card_id` naming a card not in that
player's hand**. The mask survives this because its `SELECT_CARD` branch tests
`in_hand_of(forced_card_id, p)` before restricting; `step` has no such test.

## Design

**The mask becomes the definition of legality, and `step` validates against it.** Three guards
stay outside it, because the flat space cannot express them:

1. **`decision_type` must match the context** (`b6874af`). Subsumed by mask membership, but kept as
   a cheap early-out and a clearer error.
2. **A forced die must be 0..6** on both `primary_id` and `secondary_id`. `ROLL_DIE`'s value is
   *data*, not an index — flat 211 is its only mask entry — so the mask cannot constrain it.
3. **Sub-index fields.** Several `MicroAction`s map to one flat index, so mask membership is
   necessary and not sufficient for hand-built actions (replayer, tests, web).

### Steps

1. **Fix the mask's forced-play rule**, since it becomes the definition:
   - add `in_hand_of(MISSILE_ENVY, p)` to the `SELECT_PLAY_MODE` branch, matching what
     `SELECT_CARD` already does — this makes staleness unreachable rather than merely unlikely;
   - drop the always-true wildcard so the restriction applies to Missile Envy itself;
   - move the forced branch **above** the China Card and scoring branches so ordering stops
     mattering, or leave it below and let the in-hand test carry it (see question 4).
2. **Clear the forced flags when Missile Envy leaves the hand.** Belt and braces with (1): (1)
   makes a stale flag harmless, this makes it not exist. A single helper called wherever card 49's
   location changes, rather than hooks in each event.
3. **Delete the dead mask branches** for classes 4 and 5, and **remove the `phasing_player`
   fallback** at `action_mask.cpp:22` for class 6, so the mask stops inventing legality.
4. **Add mask validation to `StateMachine::step`** and delete the per-rule re-derivations it
   replaces — the forced-play check, and any other guard now expressible as mask membership.
5. **Regression tests**, C++ (`engine/tests/test_bugs_regression.cpp`) per class: headline, stale
   force, China Card, scoring card via Grain Sales, `CHOOSE_BRANCH`/`NONE`/`decision_player`
   deadlocks, plus the forced-die range, which must still be refused.
6. **Differential re-sweep**: re-run the 1.54M-position sweep; it must report zero by construction.
7. **Corpus**: `tests/replayer` including `-m corpus_full`. This is the strongest check that the
   new legality matches real human play — the mask's guards already carry ts-replayer citations,
   and a rule change that contradicts the corpus will show up as a conversion failure.

## Cost

Free on the path that matters. `VectorizedBatchRunner` already regenerates the mask after every
step — measured at 579 ns of a 684 ns batched step per env — so validation there is a lookup into a
buffer that already exists. A single `Engine::step` pays one mask generation, roughly doubling a
394 ns step, which is irrelevant at those call rates.

## Does this invalidate the baseline?

**Probably not, and it is cheap to establish.** The sweep found **zero naturally occurring
disagreements** in 1.54M sampled positions across every decision type; every class needed a seeded
conjunction. If the decision stream is genuinely unchanged for self-play, the engine letter does
not bump and `E3-20-28` stays a matched baseline for the bootstrap arm — which is the difference
between one arm and two.

Verify before assuming: replay a fixed set of self-play games through both engine builds from the
same seeds and compare the decision sequences step for step. Unchanged ⇒ keep E3; any change ⇒ bump
the letter and re-run the baseline.

## Open rules questions

Blocking, because they change what the mask should emit.

1. **Quagmire/Bear Trap + Missile Envy.** Today the mask's forced branch returns before the trap
   branch, so only Missile Envy is offered, for Ops. The owner's reading is that the player
   discards Missile Envy to the trap and that consumes the action round. Is the discard the whole
   action round, and does it satisfy the Missile Envy obligation?
2. **Held scoring + Missile Envy.** If the player holds as many scoring cards as they have action
   rounds left, the held-scoring obligation presumably outranks the force. Which wins, and does
   playing the scoring card discharge the force or defer it?
3. **Grain Sales handing over a scoring card.** `card_dispatcher.cpp:1068` routes it through
   `SELECT_PLAY_MODE`, the only way a scoring card reaches that node. Should it auto-resolve as an
   event instead, the way `SELECT_CARD` does at `state_machine.cpp:894`?
4. **Missile Envy and the China Card.** The China Card is not in a hand — it is tracked by
   `china_card_holder` — so an `in_hand_of` test on the *forced* card leaves it selectable while a
   force is live. Should a forced player be able to play the China Card at all?

## Non-blocking

- Whether `forced_card_id` collapses to a boolean. It only ever holds 0 or 49.
- Whether `decision_type` stays as a separate early-out once mask membership subsumes it.
