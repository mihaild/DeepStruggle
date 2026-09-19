# P17 — what the decision stack actually needs to hold

Companion to [P17_grain_sales_child_frame.md](P17_grain_sales_child_frame.md). Evidence in
[P17_ctx_stack_census.md](../log/P17_ctx_stack_census.md).

Today every frame is a full 128-byte `DecisionContext`, whether it is the decision being made or a
card waiting three levels down. The owner's observation is that those are not the same thing:
*"it is impossible to use 2 Ops from Five Year Plan, then trigger the event, then use the last Op."*
A suspended frame's Ops are never **partially** spent — so it does not need the machinery that
tracks a partially spent sequence.

That is true, and it is the fact the whole design rests on.

## 1. What a suspended frame carries today — measured

1,500 games, **5,152 suspended (non-top) frames** inspected field by field.

| field | non-default on a suspended frame |
|:--|:--|
| `decision_player` | 5,152 |
| `decision_type` | 5,088 — `SELECT_OP_MODE` 4,845, then `SELECT_CARD` 80, `ROLL_DIE` 59, `SELECT_PLAY_MODE` 55, `POINT_NODE` 49 |
| `pending_op_card` | 5,010 |
| `pending_ops_value` | 4,955 |
| `timing_branch` | 4,955 |
| `resolving_card` | 252 |
| `allow_early_stop` | 84 |
| `op_mode` | 77 |
| `pending_roll` / `roll_actor` | 72 |
| `roll_target` | 68 |
| `remaining_steps` | 45 |
| `start_influence_nodes` | 36 |
| `max_per_country` | 29 |
| **`visited_nodes`** | **0** |
| **`node_count_bits`** | **0** |
| **`suppress_op_card_event`** | **0** |
| **`event_granted_ops`** | **0** |
| **`event_stage`** | **0** |

### The tail of that table is stale, not live

`remaining_steps > 0` on a suspended frame looks like a counterexample — a sequence caught
mid-flight. It is not. Every such frame has `visited_nodes == 0` **and** `node_count_bits == 0`,
which a genuinely mid-sequence frame cannot: those are what record the steps already taken. Dumped
in full, all of them are the same shape — frame 0, `resolving_card` Five Year Plan, `ROLL_DIE` or
`POINT_NODE` left over, `remaining_steps` 1, nothing visited.

The cause is structural: `pop_context()` does **not** clear the frame it leaves, and
`push_context()` zeroes only the *new* top. The base frame therefore accumulates debris from
earlier action rounds and keeps it indefinitely.

**This is harmless today only by luck.** The unwind loops read exactly one field of a suspended
frame — `decision_type == SELECT_OP_MODE` — and the observation reads two. Nothing else looks, so
nothing else sees the garbage. Any design that reads *more* fields off a suspended frame, including
the `frame_kind`/`owes` proposal, is reading stale bytes unless suspension writes them explicitly.
That is a trap worth closing rather than stepping around.

### And the four always-zero fields are derivable, not lucky

* `visited_nodes`, `node_count_bits` — the owner's point. A suspension happens *between* sequences.
* `suppress_op_card_event` — UN Intervention's flag. UN Intervention terminates a chain and never
  pushes, so its frame is never suspended.
* `event_granted_ops` — event-granted Ops are spent on the top frame, immediately.
* `event_stage` — multi-stage events (CHE, De-Stalinization) fire no other card's event, so they
  never suspend.

## 1b. The simpler design — flatten, don't shrink

§2 below shrinks the suspended frame. The owner's proposal goes further: **remove suspension
almost entirely**, by representing each card's outcome as a modified card play rather than as a
frame to return to. Analysed card by card, it holds.

### Missile Envy — no frame, and the presentation is a live bug fix

The exchange is written to `card_locations` **before** any push (`mid_war.cpp:160-161`), so Missile
Envy has genuinely nothing to return to. Either the taken card's Event fires — which is that card's
resolution, not Missile Envy's — or the card is used for Ops with its Event prevented.

The presentation must be "Event **will not** fire", not "Event **already** fired". The owner flagged
this: an already-fired presentation tells the model a lasting effect is active when it is not. The
right mechanism exists — `suppress_op_card_event`, which UN Intervention sets.

**Missile Envy does not set it today, and that is a bug**: a starred opponent card taken and used
for Ops is removed from the game though its Event never occurred. Demonstrated with NATO; see
[missile_envy_starred_removal.md](../findings/engine/missile_envy_starred_removal.md). The proposed
presentation fixes it.

### Five Year Plan and Star Wars — no frame; at most an owed-Ops record

Both are US-sided, so:

| how it reaches its Event | owed afterwards | frame? |
|:--|:--|:--|
| US plays it as its Event | nothing — the Event is the whole play | no |
| USSR plays it Ops-first | nothing — the Ops are already spent | no |
| USSR plays it Event-first | **the USSR's Ops, after the chain resolves** | no — an owed-Ops record |

Only the third case persists anything, and what it persists is one sentence: *player P owes N Ops
from card C*. That is not a decision context.

### UN Intervention — already correct

`card_dispatcher.cpp:770` sets `suppress_op_card_event = 1`, puts `SELECT_OP_MODE` on the current
frame and pushes nothing. It is already the model the other cards should follow.

### Grain Sales — LANDED in `aa5ec64`

Grain Sales looked like the hard case because P17 §5's first attempt pushed the child frame
*before* the US answered, so the parent had to survive a possible decline. **The choice comes
first.** Once it is made, Grain Sales owes nothing either way, so it is tail-position like the rest:

* the merged resolution node sits on **Grain Sales' own frame** — no push;
* **`CONFIRM_DONE`** (decline) → return the card to the USSR hand marked known, and convert the same
  frame to Grain Sales' own 2 Ops;
* **any `Resolution`** → discard Grain Sales and **replace** the same frame with the drawn card's
  play at that resolution;
* **UN Intervention's card slot** → replace with the drawn card played for Ops,
  `suppress_op_card_event = 1`;
* **headline** → only `CONFIRM_DONE` is legal when the drawn card is UN Intervention; auto-advance
  settles it.

So §5's goal — `CHOOSE_BRANCH{play, return}` disappears — is reached with **zero nesting**, and the
depth-7 overflow that made work item 3.0 necessary never arises. If the drawn card is played
Event-first, the US owes its Ops afterwards: one more owed-Ops record, the same shape as Five Year
Plan's.

**Two things the implementation taught that this plan did not predict.**

1. **The drawn card must move into the US hand when it is DRAWN, not when it is kept.** Staging it
   at `PEEKED_TEMP` splits the engine: the mask is generated against a position and `step`
   validates against it, so every predicate asking where the card is must give both the same
   answer. With the card staged, ActionMask's Missile Envy forced-play test read `in_hand_of` as
   false and offered the Event while `step`'s identical test read it as true and refused. The old
   two-decision shape hid this, because the card moved *between* the two nodes. The fuzzer found
   it. Fidelity ("shown, not taken") lost to consistency, and nothing is leaked: the USSR is never
   the decision player anywhere in Grain Sales, so no observation is taken for them in that window.
2. **A rule written only in the flat mask is enforced but invisible.** `step` validates against the
   flat mask, so the headline/UN-Intervention restriction appeared to work — while every caller
   building an action from the *per-decision* mask still offered a play the engine then refused.
   The fuzzer hit that too. Such rules belong in `generate_mask`, which the flat builder maps.

`step`'s type-match guard also needed one narrow, named exception, since UN Intervention on the
drawn card is the only card-shaped action legal at a resolution node.

### Three refinements from the owner

**(a) Owed Ops must survive the *whole chain*, not one card.** If the USSR plays the US's Five Year
Plan, Star Wars or Grain Sales Event-first, they still get their Ops once everything the chain
started has resolved — however deep it went.

Today this works through the unwind loop finding frame 0's `SELECT_OP_MODE`; those are the 4,845
suspended `SELECT_OP_MODE` frames in the census, so the current engine does handle it. The
flattened design handles it *better*: `owed[0]` is not the frame below, it is a separate record, so
the chain may replace `ctx` any number of times without touching it. Chain depth stops mattering at
all.

This is the argument for keeping `owed` outside `ctx` rather than modelling it as "the frame
underneath". A frame underneath has to survive every replacement — which is exactly what broke in
§5's first attempt.

**(b) The US must see which card it is being offered.** It already does, and the reason should be
preserved on purpose rather than by accident: the drawn card sits at `PEEKED_TEMP`, which maps to
`card_slots::PEEKED` with no perspective check (`observation.cpp:188`), so it is named in the card
block at the decision node. That is the argument for moving the card to hand only when the frame is
**replaced**, never at setup.

One wrinkle worth knowing. The mask needs `pending_op_card = <drawn>` to offer that card's
resolutions, while `resolving_card = GRAIN_SALES`. The observation's chain walk treats the two
identically, so **both cards grade `ACTIVE_NOW`** at this node — `ACTIVE_NOW` stops being unique
there. Nothing is lost, because the drawn card is still singled out by its `PEEKED` location slot;
but if the two should be distinguishable in the `ACTIVE_CARD` slot itself, that is an observation
content change and needs its own decision.

**(c) UN Intervention is inverted here, and this is its only such use.** Normally UN Intervention is
the card played and it *names* a companion: `SELECT_CARD`(UN Intervention) → resolution `EVENT` →
`SELECT_CARD`(companion), with `resolving_card = UN_INTERVENTION`. Under Grain Sales the companion
already exists — it is the drawn card — and UN Intervention is named **as that card's resolution**.
The direction reverses.

In the action space this is the only node where a **card slot is legal at a `SELECT_PLAY_MODE`
decision**. Checked, and the encoding already supports it: `decode_flat_action` routes slots
0..109 to `MicroAction{SELECT_CARD, id}` **unconditionally**, without consulting
`ctx.decision_type` (`action_mask.cpp:693-694`). So no change to the flat layout — only the state
machine needs to route a `SELECT_CARD` arriving at a resolution node, before `primary_id` is read
as a `Resolution`.

What must be checked rather than assumed: anything that assumes a resolution node's legal set is a
subset of `RESOLUTION`/`OP_MODE`/`CONFIRM_DONE`. The candidates are `p17_adapter.py`, the
converter's branch search, and `bindings/safety.py`.

### What is left

```cpp
DecisionContext ctx;          // the one live decision, a fixed member
OwedOps         owed[2];      // {player, card, ops, flags} -- 4 bytes each
uint8_t         owed_count;
```

**At most two owed-Ops records**, derived rather than measured: owed Ops arise only from an
EVENT_FIRST play, which requires playing a card from hand. The base play is one. Grain Sales' drawn
card is the only other card in a chain that receives a full resolution, and there is one Grain Sales
card, so there is no third.

768 bytes of stack become about 140, and there is no depth limit left to overflow.

### Three things to watch

1. **`owed` is still LIFO.** Two records must unwind innermost-first, so it is a stack — just two
   deep and four bytes wide. Worth naming as one so nobody later assumes a single slot.
2. **`relocate_played_card` must move before the replacement.** Today the caller relocates after
   `trigger_event` returns; with the frame replaced there is no caller to return to. This is the one
   systematic edit and it carries the known hazard: `highest_takeable_ops` skips
   `ctx().resolving_card` precisely to stop an Event finding the card in front of it
   (`mid_war.cpp:141-144`), and replacing the frame changes what `resolving_card` holds at that
   moment.
3. **`ACTIVE_SUSPENDED` changes meaning.** The observation computes it by walking `ctx_stack`
   (`observation.cpp:234`). With no stack it would grade the cards named in `owed` instead — which
   is arguably more accurate, since that is exactly "this card is waiting", but it is an observation
   content change and needs the owner's sign-off.

**Discarding the chain's history is safe.** Only three sites read a non-top frame at all —
`observation.cpp:234` and the two save/load sites — and they read `resolving_card` and
`pending_op_card`, both of which survive in `owed`. No rule consults the chain.

## 2. Alternative: keep the stack, shrink the frame

```cpp
struct SuspendedFrame {          // 8 bytes
    Player  decision_player;
    uint8_t resolving_card;      // whose handler resumes; 0 = a plain card play
    uint8_t pending_op_card;     // the card whose Ops are owed
    uint8_t pending_ops_value;   // Ops still owed; 0 = nothing owed
    uint8_t timing_branch;
    uint8_t flags;               // owes_event; frame kind
    uint8_t pad[2];
};

DecisionContext               ctx;               // THE decision being made — one fixed member
std::array<SuspendedFrame, 7> suspended;         // everything below it
uint8_t                       suspended_depth;
```

Four things this buys.

**The top frame is at a fixed address by construction.** Not `ctx_stack[ctx_stack_depth]` — a
single member. The owner's requirement that the currently-resolving card always be read from the
same place stops being a convention the observation has to honour and becomes a property of the
type.

**Stale debris becomes unrepresentable.** Suspending marshals six named bytes. There is no room
left for anything to be stale in.

**Memory falls.** 6 x 128 = 768 bytes today; 128 + 7 x 8 = 184. That is 584 bytes back, and
`sizeof(GameState)` drops from 1,536 toward ~950 — while raising the depth limit from 5 to 8, past
the derived worst case of 6.

**The resume rule becomes explicit.** `pending_ops_value > 0 || owes_event`, instead of
`decision_type == SELECT_OP_MODE` — a decision type currently doing double duty as a flag, and the
thing Grain Sales' parent frame could not express.

## 3. Why rehydration is sound

Resuming means rebuilding a full `DecisionContext` from eight bytes. That works precisely because
of §1: the Ops were never partially spent, so every field not carried across is *already* at its
default when the frame resumes. Concretely:

* **deferred Ops** (the EVENT_FIRST split) → `decision_type = SELECT_OP_MODE` plus the carried
  card, value, player and timing. `op_mode`, `remaining_steps`, `visited_nodes`, `node_count_bits`
  are all correctly zero: the sequence has not begun.
* **a chain extender resuming** — only Grain Sales, which resumes to its 2 Ops on a decline.
* **a frame that is merely unwound through** needs nothing at all.

If this invariant were ever false, rehydration would silently lose state. So it is asserted, not
assumed: **on suspend, `visited_nodes` and `node_count_bits` must be zero**, and violating it is an
`invariant_failed`. The owner is explicit that abort is the right failure here — it would mean a
rules implementation is wrong, and a wrong game reaching a training set is worse than a crash.

## 4. Call sites that change

Small and enumerable — only three places read a non-top frame:

* `observation.cpp:234` — the chain walk grading `ACTIVE_SUSPENDED`. Reads `resolving_card` and
  `pending_op_card`; both are carried.
* `ts_bindings.cpp:168` / `:276` — `to_save_dict` / `state_from_save_dict`. The per-frame dict
  shrinks to the carried fields. Named-field, so old saves still load.
* the three unwind loops (`state_machine.cpp:375`, `:414`, `:989`) — `pop_context()` becomes
  "rehydrate `ctx` from the top suspended record", and the resume test becomes the explicit one.

Everything else already goes through `ctx()`, which becomes `ctx`.

## 5. Sequencing

This is a bigger change than §5 and should not be bundled with it.

1. **Resize `ctx_stack` 6 → 8** (§5 work item 3.0). Unblocks Grain Sales, no behaviour change.
2. **Land §5** on the existing layout.
3. **Then this**, with its own whole-corpus run and its own fuzz sweep.

Doing it in the other order means debugging a Grain Sales restructure and a stack rewrite in the
same failure.
