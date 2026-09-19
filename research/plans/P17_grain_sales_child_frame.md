# P17 §5 — Grain Sales on a child frame

> **SUPERSEDED — kept for the diagnosis, not the design.** The child frame was abandoned. Grain
> Sales landed in `aa5ec64` with **no nesting at all**: the offer is the drawn card's own
> resolution node on Grain Sales' frame, converted on a decline and replaced on a play. The
> owner's card-by-card analysis is what made that possible — once the choice precedes the frame,
> Grain Sales owes nothing either way and has nothing to come back to. See
> [P17_stack_layout.md](P17_stack_layout.md) §1b for the design that shipped.
>
> Two things here remain true and are worth keeping. §1's failure analysis is what ruled the child
> frame out, and work item **3.0 (resize `ctx_stack` 6 → 8) is no longer needed**: the depth-7
> worst case was caused by the child frame, and without it the bound never moves.
>
> One prediction below was wrong and the fix is recorded in the shipped design: §3.4 argued for
> keeping the drawn card at `PEEKED_TEMP` until first use. That splits the mask from `step` —
> ActionMask and `step` run identical predicates against different positions — and the fuzzer
> caught it. The card now moves into the US hand when drawn.


Detail plan for the half of §5 that was attempted, failed and reverted (commit `47f6043` is the
clean baseline it must be re-applied to). §5's other half, South African Unrest, is landed.

The goal is unchanged: **`CHOOSE_BRANCH{play, return}` disappears.** The drawn card is pushed as a
child frame and resolved by the ordinary 5-way resolution; *decline* returns it and continues with
Grain Sales' own 2 Ops. No card-specific branch remains.

## 1. Why the first attempt failed

Not "nothing pops the frame" — that was the symptom I named at the time and it was too vague to
act on. The engine already unwinds pushed frames, in three places that are textually near-identical
(`state_machine.cpp:375`, `:414`, `:989`):

```cpp
while (state.ctx_stack_depth > 0) {
    state.pop_context();
    if (state.ctx().decision_type == DecisionType::SELECT_OP_MODE) { resumed = true; break; }
    state.ctx().resolving_card = 0;
}
```

So there is exactly one resume rule in the engine: **a frame resumes iff it holds
`SELECT_OP_MODE`** — read as "Ops are still owed here". Everything below follows from taking that
rule seriously.

### Failure A — the parent frame is ambiguous under that rule

The decline path wants the parent to hold `SELECT_OP_MODE` with Grain Sales' 2 Ops. The play path
must *not*: when the drawn card's Ops finish, the unwind pops into the parent, sees
`SELECT_OP_MODE`, and resumes it — handing the US the drawn card's Ops **and** Grain Sales' own 2.
The card grants those only when the drawn card is returned.

The parent is created before the US answers, so it cannot hold the decline's state speculatively.

### Failure B — the drawn card is never relocated (the real board-mismatch source)

`advance_after_ops`'s headline tail relocates the played card *after* unwinding to depth 0:

```cpp
uint8_t op_card = state.ctx().pending_op_card;      // state_machine.cpp:390
```

In the baseline the whole Grain Sales play happens on the base frame, so at that point
`pending_op_card` **is** the drawn card and it gets relocated. With a child frame, the drawn card
lives on the child; by line 390 the child is popped and `pending_op_card` at depth 0 is Grain Sales.
The drawn card is never relocated and stays in the US hand.

That is precisely the bug the comment eight lines above documents — turn 4's headline of replay 14,
where Marshall Plan was drawn out of the USSR hand, couped Brazil, and stayed in the US hand
afterwards, in the running for Missile Envy and playable a second time. **The restructure
reintroduces it**, and it explains the shape of what I saw: board *and* score mismatches spread
across replays 100, 105, 106, 111, 112, 113 rather than a clean single-cause failure.

### Failure C — the owed-Event guards ask the wrong question

Both guards read `state.ctx_stack_depth == 0`, meaning "a pushed frame is Ops an event granted, not
Ops bought with a card". A Grain Sales child frame breaks that equivalence: it **is** a card the
player played, and it owes its Event. Replay 137, turn 4 headline — the US takes Willy Brandt from
the USSR hand, plays Ops first, and Willy Brandt's Event must still follow.

This one was already diagnosed and fixed in the attempt; the fix is carried forward below.

## 2. The frame contract

State this once and check every site against it.

| | parent (Grain Sales) | child (drawn card) |
|:--|:--|:--|
| `resolving_card` | `GRAIN_SALES` | `0` |
| `decision_type` | **`NONE`** while the child lives | `SELECT_PLAY_MODE` |
| `pending_op_card` | the drawn card id | the drawn card id |
| `pending_ops_value` | 0 | set by the resolution |

Three consequences:

1. **`decision_type = NONE` on the parent** makes it inert to the resume rule (Failure A). The
   decline pops first and installs `SELECT_OP_MODE` *after*, which is the only moment the 2 Ops
   are actually owed.
2. **`pending_op_card` is written on both frames.** The parent's copy is what the decline reads to
   find the card to hand back (`PEEKED_TEMP` is empty by then) and what line 390 reads to relocate
   it (Failure B). Setting it on the parent costs nothing and closes both.
3. The parent keeps `resolving_card = GRAIN_SALES`, so a decline routes to Grain Sales' handler.

## 3. Work items, in order

Each step ends with the check that would catch it going wrong. Do not batch them.

**3.0 — Resize `ctx_stack` from 6 to 8, FIRST.** Derived in
[P17_ctx_stack_census.md](../log/P17_ctx_stack_census.md) §3: the worst legal chain is
Five Year Plan → Star Wars → Missile Envy → Grain Sales → drawn card → its event, which with an
event-first base reaches depth 6 (seven frames). The array holds six, and `push_context()` failing
calls `invariant_failed` → `std::abort()`. **Today the engine fits with exactly zero margin and
§5's child frame is the one that pushes it over.** 256 bytes; `sizeof(GameState)` 1,536 → 1,792
against a 4,096 budget. Check: a scripted deep chain in the engine tests and a fuzz assertion on
depth, since 251k decisions of random play never exceed depth 2.

**3.1 — Parent contract + relocation.** Trigger sets `pending_op_card = chosen_card` and
`decision_type = NONE` on the parent before `push_context()`. Verify against Failure B directly:
a scripted headline Grain Sales that draws a card, coups with it, and asserts the drawn card is in
the discard pile, not the US hand. This is replay 14; add it as an engine test, since a comment is
what guarded it before.

**3.2 — Decline path.** Grain Sales' handler pops, then installs `SELECT_OP_MODE` with 2 Ops,
reading the card to return from the parent's `pending_op_card` rather than scanning `PEEKED_TEMP`.
The returned card goes to `hand_of(USSR, known=true)`. Check: replay 119 turn 4 headline
(Brezhnev Doctrine returned), and replay 279 turn 7 (returned, then Angola couped with Grain
Sales' own Ops).

**3.3 — Owed-Event guard.** Replace both `ctx_stack_depth == 0` tests with a helper that asks
*which* frame: depth 0, or a parent whose `resolving_card` is `GRAIN_SALES`. Check: replay 137
turn 4 headline.

**3.4 — Card staging.** Leave the drawn card at `PEEKED_TEMP` when the frame is pushed; move it to
`hand_of(US, known=true)` on the **first resolution action**, not at push time. Two reasons: the
observation should say "shown to me", not "mine", at the choice node; and the card must leave
`PEEKED_TEMP` before any Event fires, or Our Man in Tehran — which peeks five cards of its own —
would be inside its own peek. Check: the existing Our Man in Tehran test, plus a new one asserting
the card reads `card_slots::PEEKED` at the choice node.

**3.5 — UN Intervention, headline.** UN Intervention cannot be played in a headline. If Grain Sales
draws it there, mask everything but the decline; auto-advance then settles the node, since decline
is the only legal action. The returned card must be marked known. Check: a scripted headline that
forces the draw.

**3.6 — UN Intervention on the drawn card** (your case 3). Offer UN Intervention's own card slot at
the resolution node when the US holds it and the drawn card is a USSR event. The action arrives
card-shaped at a `SELECT_PLAY_MODE` node, so the state machine must route it before reading
`primary_id` as a `Resolution`. Effect: the drawn card's Event is cancelled, the US takes its Ops,
both cards are spent, and Grain Sales' own 2 Ops are **forfeited** — the card grants those only on
a return. Check: the probe I already had working, asserting slot 31 legal.

**3.7 — Your case 2** (the US draws UN Intervention and pairs it with a USSR card from its own
hand). Expected to need no change — "play the card drawn through Grain Sales as its event" already
covers it. **Unverified**; write the test before assuming.

## 4. Blast radius

* **Engine tests** — `test_reentrancy.cpp`, `test_cards_mid.cpp`, `test_card_edge_cases.cpp`,
  `test_state_lifecycle.cpp` all drive Grain Sales through `CHOOSE_BRANCH`. Convert only the sites
  that follow a `GRAIN_SALES` trigger: Star Wars and Five Year Plan have branches of their own and
  a blanket replace rewrites those too. That mistake was already made once.
* **Python tests** — `test_grain_sales_through_an_event.py`, `test_card_in_play_not_in_hand.py`,
  `test_headline_ops_card_discarded.py`, `test_hand_knowledge_triggers.py`,
  `test_event_first_single_resolution.py`, `test_card_fixes.py`.
* **Converter** — `ts_replayer_convert.py`'s branch search probes each legal action and scores by
  `probe.ctx().pending_op_card` / `resolving_card`, which still discriminates correctly under the
  new shape (a decline pops to a parent whose `pending_op_card` is Grain Sales). Expected to need
  no change, but the whole-corpus run is the proof, not this paragraph.
* **Adapter** — `p17_adapter.py` must map the old two-node chain onto the new one-node form for
  the play path.

## 5. What it costs in decisions and observation

* **Play path: 2 decisions → 1.** Return path stays at 2 but re-encodes the first as
  `CONFIRM_DONE` (flat 208) instead of `CHOOSE_BRANCH` branch 1 (flat 201).
* **Observation, four slots**, at unchanged width: `ctx_stack_depth` (flat 3782) `0.0 → 0.333`;
  the `DECISION_TYPE` one-hot moves from bit 6 to bit 2; the drawn card's `ACTIVE_CARD` goes
  `0 → ACTIVE_NOW`; Grain Sales' goes `ACTIVE_NOW → ACTIVE_SUSPENDED`. The last two fall out of the
  chain walk at `observation.cpp:233` with no new code.
* **The action mask never branches on `ctx_stack_depth`** — zero references in `action_mask.cpp` —
  so the resolution's own legal set is identical on a pushed frame. `EVENT_GRANTED_OPS` reads an
  explicit ctx field rather than depth, so it does not spuriously flip; it is the correctly-written
  mirror of Failure C.

## 6. Verification ladder

In this order, stopping at the first failure:

1. `./build/release/engine/ts_tests`
2. `./build/release/engine/ts_fuzz --games 10000`
3. Corpus: 300/300, 0 guessed, 0 board mismatches — **and the decision count is expected to DROP**
   from 122,058 by one per Grain Sales play-the-card. Count those plays first and predict the new
   total; a number that matches by luck is not a check.
4. `pytest -q -n auto tests/bindings tests/engine_logic tests/replayer tests/training`
5. `pytest -q -m corpus_full tests/replayer`
6. `pyrefly check ai tools tests web bindings`
7. Adapter equivalence on the play and decline paths specifically, not just whole-game tournaments
   — whole-game comparison said "differs" three times during the adapter work without ever
   localising the cause.

## 7. Parked — the reveal is NOT part of this refactor

**Out of scope. Recorded here so the finding is not lost, not to be built now.** This plan is the
stack/child-frame restructure and nothing else.

The finding, confirmed rather than assumed: **the USSR never learns the US saw their card.**
`observation.cpp:171` tests `in_hand_of(loc, my_player) → MY_HAND` *before* any known-ness check,
so the known flag is consulted only for the opponent's hand — a returned card reads identically to
one never touched. There is no history channel to put it in either: `state.action_history` is dead,
with no writer in `engine/src`, absent from the observation, and deliberately excluded from
`to_save_dict`.

When it is eventually taken up, the cheap form is not a new slot: **grade `MY_HAND` the way
`ACTIVE_CARD` is already graded** — 1.0 for a card the opponent has not seen, lower for one they
have. Zero additional floats, since `known_to_opponent(loc)` is already maintained in state, and it
serves CIA Created, Five Year Plan, The Cambridge Five, Missile Envy, Terrorism and Lone Gunman
alike rather than Grain Sales alone. It is an observation content change at constant width, so it
needs the owner's approval and its own ablation whenever it happens.
