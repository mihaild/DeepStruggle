# P17 — one decision per choice

The action space spends 35.6% of its decisions on three nodes that average 2.1, 2.0 and 2.9 legal
actions, encodes "nothing happens" five different ways, and aliases one index to three meanings.
This plan collapses the chain, gives every "nothing happens" a single index, and promotes two
choices that name real game concepts into their own heads.

Owner-approved 2026-09-18. Supersedes the analysis in
[`../findings/engine/flattening_card_play.md`](../findings/engine/flattening_card_play.md), which
proposed the merge and measured its size.

**Status of the corpus check:** restored to 300/300, 0 guessed, 0 board mismatches, 122,056
decisions. The two conversion regressions the merge introduced, the consumers it had not reached
(`action_encoder.py`, all six bots, `behavioral_cloning.py` indexing the refused slots), and the
`advance_root` finding behind search returning illegal actions are written up in
[`../log/P17_corpus_restoration.md`](../log/P17_corpus_restoration.md).

## Why now

The ladder is being reset anyway. Every checkpoint and `(seed, actions)` dataset is invalidated by
an action-space change, which is why this was postponed twice; the search-CE programme has just
ended negative ([`../log/P15_X4b_collapse_is_pool_starvation.md`](../log/P15_X4b_collapse_is_pool_starvation.md)),
so there is no arm whose ladder this would void.

## 1. The resolution node — nine slots become five

Today a card play is `SELECT_PLAY_MODE → [CHOOSE_TIMING_BRANCH] → SELECT_OP_MODE`, occupying flat
slots `[110..118]`. It becomes one node of five:

| slot | own / neutral card | opponent card |
|---:|:---|:---|
| 0 | event | event-first *(op mode deferred until the event resolves)* |
| 1 | space | space |
| 2 | ops → placement | ops-first → placement, then the event |
| 3 | ops → coup | ops-first → coup, then the event |
| 4 | ops → realignment | ops-first → realignment, then the event |

**"My event" and "event first" share slot 0.** Both mean "the event resolves now"; ownership says
what follows. This is not the aliasing being removed elsewhere, and the distinction is worth
stating as a rule, because it decides several later questions:

> Combine when the disambiguator is a salient, always-present state feature the network sees on
> every instance. Split when it is a rare conjunction.

Ownership passes — it is a static card property, present in the card block, salient on every card
play. `event_granted_ops && card ∈ {Junta, Tear Down}` fails: two cards in 110, a conjunction of a
flag and an id.

The same argument forces the ops options to share too. "Ops → placement" on an opponent card
already means something different from the same choice on one's own (the event fires afterwards).
Splitting event by ownership while leaving ops shared would encode ownership into the action space
for one option of five and leave it to the state for the other four.

**The deferred node after an event-first event reuses slots 2/3/4.** The choice there is exactly
"how do I spend the ops", so the head sees placement/coup/realign as one concept whether reached
directly or after an event.

### Legality: exactly one lookahead

The merged mask is a function of card properties and flags already available at play-mode time —
scoring card, China Card, Missile Envy/Defectors forcing, ownership, `can_trigger_event`,
`can_attempt_space` — with one exception:

* **coup must be gated** on `get_coup_target_mask` having any legal country, because a coup is a
  single action with no stop: once selected it must happen.
* **realignment and placement are offered unconditionally**, because both are repeatable actions
  with an early stop. Choosing realignment with no legal target resolves to "stop immediately".

That asymmetry is the engine's own mechanics rather than a special case, and it reduces the
"mask needs lookahead it does not do today" risk — the part of this refactor that fails silently —
from three predicates to **one**, which §7's differential test covers directly.

Note the cost of the permissive direction: a policy may select realignment with nothing to hit and
waste a card. That is a legal, strictly dominated action, and the position is one the policy should
learn to avoid rather than one the mask hides.

### The deferred ops node must stay distinguishable

Verified before starting. `EVENT_FIRST` stages the deferred ops decision on the *current* frame --
`timing_branch = EVENT_FIRST`, `decision_type = SELECT_OP_MODE`, `pending_ops_value` set -- then
pushes a new frame for the event; `pop_context()` returns to that staged frame with `timing_branch`
intact (`state_machine.cpp:1044-1075`). The observation already exposes it as
`TIMING_OPS_FIRST` / `TIMING_EVENT_FIRST` (`observation.cpp:358-360`), so the model gets a
three-way distinction at an ops node: `EVENT_FIRST` = the event has already resolved,
`OPS_FIRST` = it is still pending, 255 = own or neutral card with no event coming.

**So the merged resolution MUST keep setting `ctx.timing_branch`** even though
`CHOOSE_TIMING_BRANCH` disappears as a decision. Dropping it is easy to do by accident while
deleting the node, and it would silently blank two observation features and make the deferred node
ambiguous -- the model would be choosing how to spend Ops without knowing whether the opponent's
event has fired.

### Ordinary ops placement is mandatory

`allow_early_stop = 1` at `state_machine.cpp:1109` comes off: spending Ops on influence places all
of them.

**An earlier draft of this section said event placements were "genuinely stoppable" and should keep
their flag. That was wrong.** `rules/cards.json` gives exact counts, not maxima:

| card | text |
|:---|:---|
| Socialist Governments | "Remove a **total of 3** US Influence" |
| Comecon | "Add 1 USSR Influence to **each of 4**" |
| Marshall Plan | "Add 1 US Influence to **each of any 7**" |
| Decolonization, Colonial Rear Guards | "**each of any 4**" |
| OAS Founded | "a **total of 2**" |
| The Voice of America | "**Remove 4** USSR Influence" |
| Liberation Theology | "a **total of 3**" |

None says "up to". So `allow_early_stop = 1` on roughly twenty cards lets a player place or remove
fewer than the card requires, and §3's group A is **not** correct for them -- it is a rules bug of
its own, the same size as the three inherited-flag sites in §6.

The one caveat is the empty mask. `can_place_influence` admits superpower-adjacent countries
unconditionally, so a target nearly always exists, but the mask also requires
`ops_available >= cost` and cost is 2 for an opponent-controlled country. A position with influence
nowhere, every superpower-adjacent country opponent-controlled and 1 Op left has **no legal
placement**. This is rare to the point of being hard to construct -- and the failure mode of
removing early stop without handling it is a deadlock on an empty mask, which is the worst
available outcome.

§3's rule covers it without a special case: the decline index is in the mask when nothing else is,
which is a position offering nothing rather than an early stop being offered as a choice.

## 2. The resolution always acts on the top of the stack

`GameState` already carries `std::array<DecisionContext, 6> ctx_stack` with `ctx_stack_depth`, and
the observation already walks it — each card's slot is graded `ACTIVE_NOW` / `ACTIVE_SUSPENDED` /
`ACTIVE_NEXT` across the whole chain (`observation.cpp:221-240`). **So this costs no input width.**

What is missing is the discipline. `resolving_card` (whose event is executing) and
`pending_op_card` (whose Ops are being spent) are two fields that are usually but not always the
same, and where they disagree the wrong handler reads the decision. That is a live bug class, not a
hypothetical: with an empty USSR hand, Grain Sales left `resolving_card` set, so the chosen op mode
reached Grain Sales' `CHOOSE_BRANCH` reader and `INFLUENCE == 0` was taken for "play the drawn
card" when there was none — two Influence lost at turn 8 AR7 of ts-replayer game 219
(`mid_war.cpp:391-408`).

**Rule: the resolution node acts on `ctx_stack[ctx_stack_depth]`'s card, always.** One field
answers "which card is this decision about".

This is what makes §3 and §5 fall out rather than needing special cases.

## 3. One index for "nothing happens"

Five encodings today, catalogued over all 110 cards:

| | how it is encoded | used by |
|---|:---|:---|
| A | `ctx.allow_early_stop` → shared slot 211 | ~25 cards, realignment's stop, ordinary influence ops, Space Walk |
| B | a dedicated `CHOOSE_BRANCH` index | Wargames (branch 1), Summit (branch 2) |
| C | **aliased `INFLUENCE`, slot 116** | Junta, Tear Down, KAL-007, Glasnost — *and* the generic "nothing is legal" skip |
| D | automatic fizzle, no node opens | Camp David/Arab-Israeli, Independent Reds, Cambridge Five, Special Relationship, UN Intervention, Star Wars, plus a hardcoded `may_fizzle` whitelist of four |
| E | no decline at all | Chernobyl, How I Learned, Olympic Games, Grain Sales, Warsaw Pact |

Slot 116 carries **three** meanings — a real influence placement, "nothing at all is legal", and
"decline this event's free bonus" — and a consumer of the flat stream cannot separate them without
reading `event_granted_ops` and the card id. Three engine bugs are cited against it in the source
comments (ts-replayer games 141, 146, 105), each a human making the same confusion.

**All of A, B, C collapse into one decline index.** `INFLUENCE` goes back to meaning influence
placement and nothing else. Wargames' "don't end the game" and Grain Sales' "return the card"
become that index, not bespoke branches.

**D becomes a card property.** `may_fizzle::allowed` (`constants.hpp:300-307`) hardcodes four cards
that may fizzle quietly while every other card in the same position raises an anomaly report — two
mechanisms for one outcome, selected by a list. Instead each card declares whether declining is
permitted, and the "no legal target" case has one answer: the decline index is in the mask, or it
is not and the position is genuinely broken and says so. That keeps the anomaly report meaningful
instead of whitelisted around.

**`ROLL_DIE` comes off slot 211.** A chance node with one legal action is not a decline; sharing
the index is the same aliasing in a cheaper place. It gets its own index.

## 4. Two heads that name real concepts

| choice | representation | why a head rather than a branch |
|:---|:---|:---|
| **DEFCON value** — How I Learned, **Summit** | shared 5-slot value head, masked per card | DEFCON is a first-class quantity |
| **Region** — Chernobyl | 6-slot region head | regions are first-class: scoring cards, DEFCON restrictions |

**Summit becomes "set DEFCON to V"** with mask `{current−1, current, current+1} ∩ [1,5]`, rather
than ±1 with a third branch for "unchanged". How I Learned is the same head with a wider mask. This
is better than routing Summit's "unchanged" to the decline index: "unchanged" becomes a positive
choice, the decline index keeps exactly one meaning, and two cards share one head with the mask
doing the work.

The test a dedicated head must pass is that it names a concept the game already has. Region and
DEFCON pass. "Participate vs boycott" does not.

## 5. Two restructures that remove bespoke branches

* **Grain Sales** — `CHOOSE_BRANCH{play, return}` becomes: the drawn card is pushed on the stack;
  **decline** returns it and continues with Grain Sales' own 2 Ops; anything else is the ordinary
  5-way resolution acting on the stack top. No card-specific branch remains. This only works given
  §2, which is the argument for §2.
* **South African Unrest** — `CHOOSE_BRANCH{+2 SA, +1 SA then split}` becomes `POINT_NODE` over
  {South Africa, adjacent} with ops counted after.

Both follow one principle worth stating: **prefer reusing a high-traffic head over adding a
low-traffic branch.** `POINT_NODE` is 39.3% of all decisions and exercised constantly; a
`CHOOSE_BRANCH` index seen a few times per thousand games is close to untrainable whatever its
parameters.

What remains genuinely branch-shaped afterwards is **two cards**: Warsaw Pact (add/remove) and
Olympic Games (participate/boycott). They get **separate plain slots**, not shared ones — see §9.

## 6. Bugs to fix in the same change

Found while cataloguing; all are the class the codebase has already fixed twice with explicit
comments (`late_war.cpp:63`, `late_war.cpp:271`).

1. **`allow_early_stop` left unset**, so it inherits the previous decision's value and a mandatory
   choice silently becomes optional:
   * Missile Envy's tied-card selection — `mid_war.cpp:190-199`
   * UN Intervention — `early_war.cpp:462-479`
   * NORAD's post-DEFCON-drop placement — `state_machine.cpp:446-450`, which also skips the full
     `ctx() = DecisionContext{}` reset the normal path does at line 496, leaving `max_per_country`,
     `visited_nodes` and `node_count_bits` stale as well
2. **Likely-dead branch** — `action_mask.cpp:66-77` handles `pending_op_card == UN_INTERVENTION`,
   but nothing writes that value and the `resolving_card` check at line 53 fires first. Confirm
   before deleting.

## 6a. The rules fixes land AFTER the refactor, not with it

§6's three sites were implemented and then **reverted** (`57df782`, reverted by `75da57f`). They
are correct fixes and they were landed in the wrong order.

**Every one of them changes the decision stream.** Setting `allow_early_stop = 0` where it
previously inherited a 1 removes flat 211 from that node's mask; NORAD's context reset also changes
which countries are legal. The same is true, on a much larger scale, of the "place exactly N" class
above -- roughly twenty cards.

That is incompatible with how §7 verifies this refactor. The adapter's whole value is that a
divergence between old and new means **one** thing: the representation changed. If rules fixes land
first, every divergence is ambiguous -- a real representation bug and a deliberate rules correction
look identical in the diff, and the differential test stops being able to fail usefully.

**So: the refactor preserves the decision stream exactly, except where the representation itself
requires a change** (§1's coup gating, which §7 counts explicitly). Once the adapter is validated
and the merged engine is trusted, the rules fixes land as their own change, with their own
before/after, where a decision-stream diff is the *expected* output rather than a warning.

The batch, when it comes:

* the three inherited-`allow_early_stop` sites — Missile Envy, UN Intervention, NORAD (§6);
* the "place exactly N" class — every event placement whose card text gives a count rather than a
  maximum, currently `allow_early_stop = 1`;
* the `may_fizzle` whitelist, which §3 replaces with a per-card property.

All three are the same bug in different clothes: *the engine lets a player do less than the card
says*. Fixing them together, after the representation is settled, also gives one clean measurement
of what that class was worth.

## 7. The adapter, and how this is verified

**A Python adapter drives an old-representation checkpoint on the new engine.** The observation is
unchanged by this refactor, so an old model's input stays valid; only its action semantics differ.
The merged mask contains everything needed to reconstruct the old two-step masks, so **no legacy
C++ is required**:

```
own / neutral:                          opponent:
  old EVENT ⟸ bit0                        old EVENT  never legal
  old SPACE ⟸ bit1                        old SPACE  ⟸ bit1
  old OPS   ⟸ bit2|bit3|bit4              old OPS    ⟸ bit0|bit2|bit3|bit4
                                          timing: EVENT_FIRST ⟸ bit0
                                                  OPS_FIRST   ⟸ bit2|bit3|bit4
  if OPS → op mode = {placement: bit2, coup: bit3, realign: bit4}
```

Query the old model once or twice, combine, emit the merged index.

**The adapter must count divergences, not paper over them.** The merge tightens legality in one
corner — today `OPS` is legal unconditionally for any non-scoring card (`action_mask.cpp`, "Ops
play: always legal"), and under §1 coup-only positions with no coup target lose that option. Where
the reconstructed old mask and the real old engine would disagree, the adapter logs the position
rather than choosing. If the count is zero in practice the tightening is vacuous; if not, those are
exactly the positions worth reading.

### Two constraints found while building the harness

**`GameState` cannot be serialised through the bindings** — no `to_bytes`, no pickle support. So a
position cannot be captured on one build and reloaded on another, which rules out the obvious
"save a corpus of positions, run both mask implementations over it" design. The differential test
must instead **replay games by semantic action** (card, resolution mode, target country), which
both representations can interpret, and compare the state fingerprint after every step. That is
the same technique the ts-replayer already uses to drive human logs through the engine.

**The position corpus must come from policy play, not random play.** Random walks rarely build
control, rarely reach a contested late war, and rarely produce the board shapes a trained policy
spends its time in — so a mask defect that only appears in real positions can hide from them
entirely. `decision_stream_baseline.py` is deliberately random because it is pinning the engine
action-for-action and must not depend on a policy; the *mask* differential tests take their
positions from `outcome_distribution.py`-style policy play instead.

**Differential test**, before trusting any of it:

1. over many self-play positions, assert the reconstructed old-style masks match what the current
   engine's `generate_flat_mask_212` produces, and log every divergence with its position;
2. assert every action the merged mask offers is accepted by `StateMachine::step` — the P14
   invariant, which this change could otherwise reintroduce;
3. play old-checkpoint against old-checkpoint through the adapter on the new engine, and check the
   result distribution matches the same pair on the old engine. That is the sanity check the whole
   adapter exists for, and `tools/scripts/outcome_distribution.py` is it: the exact stream cannot
   survive the change (the RNG draws differently once the decision count moves) but the **outcome
   mix, game length, DEFCON, tracks and ending reasons must not move at all**. Baseline frozen at
   2,000 games, `/workspace/data/p17/outcome_baseline_f0e5840.json`.

## 8. Order of work

1. Catalogue frozen (done — §3 table and §6 are its output).
2. Decline index + `ROLL_DIE` off 211, **no merge yet**. Small, isolated, and (2) of §7 applies.
3. `allow_early_stop` fixes and the dead-branch confirmation.
4. The 5-way merge and the stack rule together — they are one change; §5's Grain Sales depends on
   both.
5. DEFCON and region heads; South African Unrest.
6. Adapter and the three differential tests.
7. Regenerate stubs, rebuild, re-anchor the ladder from scratch.

Steps 2 and 3 are worth landing and testing before 4 begins, because they are independently
correct and shrink the diff that the merge has to be reviewed against.

## 9. Deferred, deliberately

* **Card-conditioned branch head.** The right long-term answer for Warsaw Pact and Olympic Games is
  to compute the branch logit from the resolving card's embedding rather than a bare slot — the
  mechanism `--per-entity-heads` already applies to cards and countries, and the branch head is the
  one place that does not get it. Postponed because it is measurable once there is a ladder to
  measure against, and because aliasing the two cards now would bake in the interference we would
  then be trying to measure. Until then they get separate slots, which is inert rather than wrong.
* **Slot 211's remaining occupants.** After `ROLL_DIE` moves out and the decline unifies, what is
  left on 211 is one meaning; no further work.
* **`POINT_NODE` reachability.** The measured pathology — a trained policy drives *legal* country
  nodes to e⁻¹¹⁷ (median worst) and e⁻⁴⁰⁷ (extreme), specific to `POINT_NODE` while every other
  decision type stays within −1 to −7 — is not addressed by anything here. It is the larger
  problem and it wants its own plan.

## What this costs

Every checkpoint, every `(seed, actions)` dataset and the Elo ladder. Both the action indices and
the decision stream change. Accepted deliberately: the ladder is being reset anyway, and no arm is
currently running against it.

Expected gain, measured rather than assumed: **−12.6% decisions** for the play-mode/op-mode merge,
**−17.5%** with the timing branch folded in, on 440.3 decisions per game.
