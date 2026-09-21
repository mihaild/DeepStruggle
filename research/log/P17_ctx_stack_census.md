# What `ctx_stack` holds — derivation, with a census as a cross-check

Written because the Grain Sales restructure (P17 §5) broke on an unchecked assumption about the
stack. **The depth bound below is derived from the card rules. The census is only a cross-check on
what ordinary play exercises, and it is not evidence about the bound** — random play never reaches
the cases that matter.

Engine `bccd6ad6…`, verified fresh. Census: 1,500 random games, 251,126 decisions, via
`to_save_dict()["ctx_stack"]` at every decision.

## 1. What consumes a frame

A frame exists to give something its own decision scratch — `visited_nodes`, `node_count_bits`,
`max_per_country`, `remaining_steps`. Two constructs need one:

* **(E) an event that requires decisions**, pushed by whoever fires it;
* **(P) a card play whose Ops are deferred** — the EVENT_FIRST staging, which keeps the playing
  frame below while the event runs above.

## 2. The cards that can extend a chain

Exactly five cards bring another card into play. Only **four** can extend the *event* chain:

| card | brings in | fires that card's event? | frames added |
|:--|:--|:--|:--|
| Five Year Plan | a random card from the opponent's hand | yes, if it is the discarder's opponent's card | +1 E |
| Missile Envy | the opponent's highest-Ops card | yes if same-side or neutral; otherwise Ops only | +1 E, else +1 P |
| Star Wars | a non-scoring card from the discard pile | yes | +1 E |
| Grain Sales | a random card from the USSR hand | yes, at the US player's choice | +1 P (after §5), +1 E if event-first |
| **UN Intervention** | a companion card from the player's own hand | **no — explicitly suppressed** | **+0 E** |

UN Intervention is a chain **terminator**, not an extender. `card_dispatcher.cpp:770` sets
`suppress_op_card_event = 1`, sets `SELECT_OP_MODE` on the *current* frame, and does not push: the
companion card's Ops are used and its Event never fires. Worth stating because the natural reading
of "cards that play other cards" puts it in the same group as the other four.

## 3. The worst case, derived

Five Year Plan, Star Wars and Grain Sales are all **US-sided**; Missile Envy and UN Intervention are
neutral. The EVENT_FIRST split needs an *opponent's* card, so the chain must be started by the
**USSR** playing the US's Five Year Plan event-first:

```
d0  CARD_PLAY   USSR's deferred Ops for Five Year Plan    (P: the EVENT_FIRST split)
d1  EVENT       Five Year Plan -> USSR discards Star Wars (US-sided) -> fires
d2  EVENT       Star Wars      -> US replays Missile Envy from the discard
d3  EVENT       Missile Envy   -> USSR hands over Grain Sales (US-sided) -> fires
d4  EVENT       Grain Sales    -> draws a card from the USSR hand
d5  CARD_PLAY   the drawn card                            (§5's child frame)
d6  EVENT       the drawn card's own event, if played event-first
```

**Depth 6 — seven frames.** `ctx_stack` is `std::array<DecisionContext, 6>` and `push_context()`
refuses at `depth + 1 >= 6`, so the maximum is depth 5, six frames. Overflow calls
`invariant_failed`, which is `[[noreturn]]` and ends in **`std::abort()`** — it kills the process,
so in training it kills the run, not the game.

**Today the engine fits with exactly zero margin.** Without §5, Grain Sales plays the drawn card on
its own frame (d4), so an event-first drawn card lands at d5 — the last available slot. **§5's child
frame adds one and pushes the worst case over the edge.**

Every link is contrived but none is illegal: Star Wars needs a US space-track lead, Missile Envy
must find Grain Sales as the USSR's highest-Ops card, Five Year Plan's discard is random. The point
is not the probability. It is that the bound is currently derived from nothing, the margin is zero
or one, and the failure mode is `abort()`.

**Action for §5: raise `ctx_stack` to 8.** Two extra frames cost 256 bytes; `sizeof(GameState)`
goes 1,536 → 1,792 against a 4,096 budget. Size it from this derivation plus margin rather than
from observed play, and add a scripted deep chain to the engine tests plus a fuzz assertion on
depth — 251k random decisions exercise none of it.

## 3b. Only one extender has work left to do — the chain can be flattened

An ordinary play does **not** push. `state_machine.cpp:1126` runs a friendly or neutral card's
Event on the *current* frame — "that is the whole play". A frame is pushed in exactly two
situations: the EVENT_FIRST timing split, and an event firing *another card's* event. Pushes are
for chains, not for plays.

That raises the owner's question: does a middle card need its frame while the next card resolves?
Checked card by card, and the answer is no for three of the four:

| card | work remaining after it hands off | frame needed? |
|:--|:--|:--|
| Five Year Plan | none — returns the nested event's `done` directly | **no, tail position** |
| Star Wars | none — same shape | **no, tail position** |
| Missile Envy | none; and the exchange is already written to `card_locations` **before** the push (`mid_war.cpp:160-161`) | **no, tail position** |
| **Grain Sales** | **2 Ops, owed if the drawn card is returned** | **yes** |

So a chain could **reuse one frame** by tail-replacement rather than nesting, and only Grain Sales
would have to stay. Since there is exactly one Grain Sales card, at most one extender per chain is
non-tail. The bound collapses:

```
d0  CARD_PLAY   base play's deferred Ops (EVENT_FIRST split)
d1  EVENT       the whole chain, one frame reused: 5YP -> Star Wars -> Missile Envy -> Grain Sales
d2  CARD_PLAY   Grain Sales' drawn card
d3  EVENT       the drawn card's own event, if event-first
```

**Depth 3, four frames** — against seven today. The owner's formulation, "simultaneously only top
card → Grain Sales → Grain Sales target", is right, with the base play's deferred Ops as a fourth.

**But it is not free, and it should not ride with §5.** The enabling move is relocating a card
before the next card's event runs, and that changes what the nested event can *see*. One concrete
hazard already has a comment in the tree: `highest_takeable_ops` deliberately skips
`ctx().resolving_card`, because "an opponent's card played for Operations fires its own event, so
the event can otherwise find the very card in front of it" (`mid_war.cpp:141-144`). Pop the parent
frame and `resolving_card` changes, so that guard stops protecting the card it was written for —
Missile Envy could take the very card whose event is running. Every zone-timing change of this kind
in this engine has cost a corpus regression.

**Recommendation: resize to 8 now** — cheap, no behaviour change, unblocks §5 — and do
tail-replacement as its own change with its own whole-corpus run.

## 4. The census, and what it does *not* show

```
decisions                        251,126
depth > 0                          5,173   (2.06%)
max depth observed                     2
depth histogram        {0: 245,953, 1: 5,124, 2: 49}

frames owing ops (SELECT_OP_MODE), by depth   {0: 4,894, 1: 133, 2: 1}
MIDDLE frame owes ops                               0
bottom AND a pushed frame owe ops at once         117

event_granted_ops set                        114  (depth 0: 111, depth 1: 3)
  cards: Junta 101, Tear Down this Wall 8, Soviets Shoot Down KAL-007 5
```

**A correction to an earlier version of this note.** It listed the cards most often seen on a pushed
frame — Marshall Plan, De-Stalinization, Decolonization, Warsaw Pact Formed — and read their
overlap with the bottom-frame list as evidence of structure. That inference was wrong. The census
counts **decisions, not occurrences**, so an event that places influence across seven decisions
contributes seven counts. The histogram therefore ranks events by *how many decisions they take*,
which is close to unrelated to what it was being used to argue. Which cards can nest is fixed by
§2 above and by nothing in a log.

What the census does support:

* **Ops are owed at the bottom or the top, never in the middle** — `middle_owes == 0` across
  251,126 decisions. Currently emergent from the unwind loop; should be an assertion.
* **Ordinary play is shallow** — max depth 2, and depth 2 is 0.02% of decisions. So the deep cases
  carry no test coverage from random play or from self-play, which is exactly why §3 must be a
  derivation and must get its own scripted test.
* **`event_granted_ops` is rare but live** — 114 occurrences, three cards. A 300-game run saw zero
  and I nearly recorded it as dead. Rare is not dead, and a sample of zero is not evidence.

## 5. The headline does not use the stack for its own structure

`headline_stage`, `headline_first_card`, `headline_second_card`, `headline_us_card`,
`headline_ussr_card` are plain scalars. The stack is used *inside* a headline card's resolution for
the same reasons as in an action round, and for nothing else. Nesting is rarer there — 0.48% of
headline decisions against 2.15% in an action round — because an opponent's card cannot be
headlined, so the EVENT_FIRST split (P) does not arise.

## 6. Depth normalization is wrong today

`observation.cpp:309`:

```cpp
out_buf->global_features[58] = static_cast<float>(state.ctx_stack_depth) / 3.0f;
```

The divisor assumes a maximum depth of 3 while the array permits 5 — and §3 argues for 7. Depth 4
feeds the network 1.33 and depth 5 feeds it 1.67, outside the [0,1] range every other normalized
global respects. It should be derived from `ctx_stack.size() - 1`, not written as a literal.
Nothing caught it because random play never exceeds depth 2. It is an observation change, so it
rides with the P17 break.

## 7. The top frame must stay at a fixed offset

Which card is resolving *now* is the most relevant fact and must sit at a constant position. That
holds today by two independent mechanisms: the global context block (`ctx_slots::BASE = 72`) is
filled from `state.ctx()`, the **top** frame, at fixed offsets regardless of depth; and the card
block's `ACTIVE_CARD` slot is keyed by card id, so it is positionally fixed by construction.

No work now. It becomes a live constraint the moment anything exposes more than one frame: frames
must then be indexed **from the top** (top = slot 0), never from the array base, or the resolving
card slides between offsets as the chain deepens — the kind of representation change a network
cannot report and silently misreads.

## Consequence for P17 §5

See [P17_grain_sales_child_frame.md](../archive/E3_ladder/plans/P17_grain_sales_child_frame.md). The stack must be
resized before §5 lands, not after.
