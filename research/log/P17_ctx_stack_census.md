# What `ctx_stack` actually holds — a census

Measured because the Grain Sales restructure (P17 §5) broke on an assumption about the stack that
nobody had checked. Engine `bccd6ad6…`, verified fresh. 1,500 random games, **251,126 decisions**;
a 300-game run agreed on every ratio.

Probe: at every decision, `to_save_dict()["ctx_stack"]` walked frame by frame.
`/home/claude/.claude/jobs/aed99808/tmp/stack_census2.py` (scratch; reproduce by re-running, it is
not committed).

## The numbers

```
decisions                        251,126
depth > 0                          5,173   (2.06%)
max depth observed                     2   (array is sized 6)
depth histogram        {0: 245,953, 1: 5,124, 2: 49}

frames owing ops (SELECT_OP_MODE), by depth   {0: 4,894, 1: 133, 2: 1}
MIDDLE frame owes ops                               0
bottom AND a pushed frame owe ops at once         117

event_granted_ops set                        114  (depth 0: 111, depth 1: 3)
  cards: Junta 101, Tear Down this Wall 8, Soviets Shoot Down KAL-007 5
```

Depth distribution by phase, from the 300-game run:

```
HEADLINE       {0: 3,726, 1: 18}      0.48% nested
ACTION_ROUND   {0: 48,932, 1: 1,066, 2: 9}   2.15% nested
```

## What it means

**Three findings, in order of how much they change the design.**

### 1. The stack is doing two unrelated jobs

Compare the two lists the census produced — cards resolving on a *pushed* frame, and cards
suspended at the *bottom* while something is pushed:

```
pushed frame          bottom frame
Marshall Plan   735   De-Stalinization  712
De-Stalinization712   Marshall Plan     595
Decolonization  524   Decolonization    524
Warsaw Pact     424   Warsaw Pact       424
Comecon         360   Comecon           360
Suez Crisis     332   Suez Crisis       332
```

They are **the same cards**. That is not a coincidence, it is the dominant case: on an EVENT_FIRST
play the engine stages the card's own deferred Ops at depth 0 and pushes *the same card's* event at
depth 1. Two frames, one card. The stack is being used to sequence two halves of a single play.

The genuinely nested cases are the residual: CIA Created (270) and Five Year Plan (224) appear at
the bottom but not in the pushed top-15, because their events cause a *different* card's event to
run. Missile Envy and Star Wars are the same shape.

So the stack has two uses:

* **(A) the timing split of one card** — event now, this card's own Ops later. Dominant.
* **(B) real event nesting** — card A's event fires card B's event. Rare.

**The headline rate confirms the split.** Nesting is 2.15% in an action round but 0.48% in a
headline — a 4.5x difference with an exact cause: you cannot play an opponent's card as a headline,
so EVENT_FIRST does not exist there, so use (A) is absent and only (B) remains.

### 2. Ops are owed at the bottom or the top — never in the middle

`middle_owes == 0` across 251,126 decisions. And 133 + 1 frames at depth ≥ 1 *do* owe Ops, so
"only the bottom frame owes Ops" is false — Junta, Tear Down this Wall and KAL-007 grant free Ops
inside an event. 117 decisions have the bottom and a pushed frame owing Ops simultaneously.

This is currently an emergent property of the unwind loop, not an asserted invariant. It should be
one.

### 3. The headline's own structure is not on the stack

`headline_stage`, `headline_first_card`, `headline_second_card`, `headline_us_card`,
`headline_ussr_card` are plain scalars. The stack is used *inside* a headline card's resolution for
exactly the same reasons as in an action round, and for nothing else.

### 4. Depth 2 is what play reaches; depth 5 is what the rules permit

Max observed is 2 over 251,126 random decisions, and that is the *sampling* ceiling, not the rules
ceiling. The owner's worked case — Five Year Plan discards Star Wars, Star Wars replays Grain
Sales, Grain Sales draws a card, UN Intervention is played on it, the USSR event resolves — reaches
depth 4–5 and is legal. Rarity is not a licence to cap it; the array's 6 slots must stay.

**This breaks a normalizer today.** `observation.cpp:309`:

```cpp
out_buf->global_features[58] = static_cast<float>(state.ctx_stack_depth) / 3.0f;
```

The divisor assumes a maximum depth of 3. `ctx_stack` is sized 6, so depth runs 0..5: depth 4 feeds
the network 1.33 and depth 5 feeds it 1.67, outside the [0,1] range every other normalized global
respects. Nothing has caught this because random play never exceeds depth 2. The fix is `/5.0f`, and
it is a one-constant observation change — but it *is* an observation change, so it rides with the
P17 break rather than landing on its own.

Worth adding to `ts_fuzz`: an assertion that depth never exceeds 5, and a scripted deep chain in the
engine tests, since 251k decisions of random play exercise none of this.

### 5. The top frame is already at a fixed offset — keep it that way

The owner's constraint: which card is resolving *now* is the most relevant fact, and the model must
read it at a constant position. That holds today, by two separate mechanisms:

* the global context block (`ctx_slots::BASE = 72`) is filled from `state.ctx()`, i.e. the **top**
  frame, at fixed offsets regardless of depth;
* the card block's `ACTIVE_CARD` slot is keyed by **card id**, so it is positionally fixed by
  construction, and the chain walk grades the top `ACTIVE_NOW` against `ACTIVE_SUSPENDED` below.

So no work is needed now. It becomes a live constraint the moment anything exposes more than one
frame to the model: frames must then be indexed **from the top** (top = slot 0), never from the
array base, or the currently-resolving card would slide between offsets as the chain deepens —
which is exactly the kind of representation change a network cannot report and silently misreads.

## Two smaller facts

* **`event_granted_ops` is rare but live** — 114 occurrences, three cards. An earlier 300-game run
  saw zero and I nearly recorded it as dead; random play ends games before the late war often
  enough to hide it. Rare is not dead, and a run that samples it zero times is not evidence.
* **Half of `GameState` is this stack.** `sizeof(GameState)` is 1,536 of a 4,096 budget;
  `ctx_stack` is 6 x 128 = 768 of it. With max depth 2 observed, three frames (384 bytes) are never
  touched. Not worth shrinking against 2,560 bytes of headroom, but an assert at depth 3 would
  catch a runaway recursion far earlier than the existing `invariant_failed` at depth 5.

## Consequence for P17 §5

`DecisionContext` has 128 bytes of which 79 are declared fields — 49 bytes of trailing padding — so
naming the frame's role costs nothing. See
[P17_grain_sales_child_frame.md](../plans/P17_grain_sales_child_frame.md).
