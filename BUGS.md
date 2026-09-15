# Known bugs

Defects that are known, reproducible, and not yet fixed. A bug leaves this file when it is fixed
and has a regression test, not when it is understood.

Anything here is a thing the code gets *wrong*, not a thing it does not do yet. Missing features
and unfinished work belong in the documentation for the area they concern.

---

## ENG-2 — a forced die outside 1..6 is accepted

**Area:** engine · **Severity:** medium · **Status:** open, needs owner decision (invariant 11)

`Engine::step` accepts `MicroAction(ROLL_DIE, primary_id, ...)` for **any** `primary_id`. A forced
roll of 26, or 200, is applied as though it were a legal die.

**Reproduce.** Advance any game to a chance node and step it directly:

```python
ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 200, 0, 0))   # -> True
```

**Where it came from.** The `decision_type` guard added to `StateMachine::step` this cycle checks
that the action answers the decision being asked. It does not check the *value*, and `ROLL_DIE` is
the one decision whose value rides in `primary_id` rather than being an index into the mask — so
the mask cannot constrain it. Forcing a die is a deliberate affordance for the replayer and for
tests, which is why the field is read at all; the gap is that the range is not checked.

**Effect.** This is the mechanism behind the "magic roll of 26" seen in a generated match. The old
`tools/play_match.py` had no chance-node handling, so at a `ROLL_DIE` node owned by nobody
`acting_player` fell through to `phasing_player` and a **bot** was asked to choose the roll,
returning whatever `primary_id` its action carried. Both writers now drain chance nodes through
`GameLoop`, so no current caller can reach it — but the engine still accepts it from any caller,
and a replay recording such a roll would re-drive it faithfully.

**Fix, once approved.** Reject `ROLL_DIE` whose `primary_id` is outside 0..6 (0 meaning "roll
normally"), and the same for `secondary_id`, in `StateMachine::step` beside the existing
`decision_type` guard. Not done here: invariant 11 reserves engine changes for the owner.

---
## ENG-1 — UN Intervention offers companion cards the rules forbid

**Area:** engine · **Severity:** medium · **Status:** open

UN Intervention (#32) reads "play this card simultaneously with a card containing your opponent's
associated Event". The companion must therefore be an opponent-associated, non-scoring card. The
action mask offers **every card in the player's hand**, including scoring cards and the player's
own cards.

**Where.** Three pieces, and the middle one is the defect:

* `engine/src/events/early_war.cpp` — `trigger_un_intervention` gets the *gate* right: it only
  raises the decision when the player actually holds an opponent, non-scoring card. It then sets
  `ctx().resolving_card`.
* `engine/src/action_mask.cpp` — the `resolving_card != 0` branch runs first and delegates to
  `CardHandlers::get_event_action_mask`, returning before the dedicated UN Intervention branch
  below it. That branch has the rule right and is unreachable.
* `engine/src/card_dispatcher.cpp` — `get_event_action_mask`'s `SELECT_CARD` switch has no case
  for UN Intervention, so it falls to `default:`, which offers the whole hand.

So the rule exists twice and the reachable copy is the wrong one.

**Effect.** Nothing is corrupted. The resolution handler re-validates the choice and, on an
illegal one, falls through without discarding the named card or granting its Operations — so a
scoring card named this way stays in hand. But UN Intervention is spent, no event fires, no
Operations are granted, and the action round ends. **The player silently loses a whole action
round.**

That silent absorption is the second half of the bug, and the reason it went unnoticed: an illegal
action is swallowed rather than rejected.

**Scale.** About half of every offered companion list is illegal by rule. Trained policies pick an
illegal one in the low single digits of percent of the times they play the card.

**Fix.** Add a `case card_ids::UN_INTERVENTION:` to the `SELECT_CARD` switch in
`get_event_action_mask`, mirroring the test the resolution handler already applies
(`in_hand_of && side == opponent && !is_scoring_card`), and **delete** the now-provably-dead branch
in `action_mask.cpp` rather than leaving a second copy of a rule nothing reaches. A regression test
must pin the companion mask to opponent non-scoring cards only — the bug is precisely that a
correct filter existed and was never consulted.

**Why it is still open.** The fix changes the decision stream, so every trained checkpoint is then
playing a game it was not trained on, and results measured before and after are not comparable.
It is deferred to a point where that cost is acceptable, not because the fix is unclear.

---

## TEST-1 — the differential fuzzing suite does not run

**Area:** tests · **Severity:** low · **Status:** open

`tests/differential/` cross-checks this engine against an independent implementation. Its modules
fail at *import*, which would abort a whole pytest run, so `tests/conftest.py` skips them at
collection and they sit behind the `differential_fuzz` marker and a `--run-fuzz` flag.

The consequence is that the repository's cross-engine check is not currently a check. Nothing else
depends on it, and no other suite is affected.

**Fix.** Repair the imports and re-enable collection, or remove the suite. Leaving it collectable
but broken is the one option to avoid, since a suite that aborts the run is worse than one that is
honestly gated.

---

## TODO — no forced-deal affordance, so replays cannot survive a shuffle change

**Status:** deferred by the owner. Not urgent while the engine is not changing between a replay's
generation and its playback.

Die rolls have a designed override: a `ROLL_DIE` action carries the acting player's value in
`primary_id` and the opponent's in `secondary_id`, and `state_machine.cpp` documents it as "the
only source of a forced die". Card deals have no equivalent -- `StateMachine::deal_cards_to_hands`
draws from the deck through `state.rng_state` with no way for a caller to supply the result.

**Consequence.** A replay cannot be made fully self-contained. Even with the starting position
saved (`to_save_dict`/`state_from_save_dict`, added) and every die recorded, re-driving a recorded
game on an engine whose shuffle or draw order has changed diverges at the first deal -- silently,
producing a different game rather than an error.

**Fix when it matters.** A forced-deal affordance mirroring the forced die: the replay source
supplies the cards a deal produces, inert in normal play. Record *what was dealt* rather than deck
order, since that also survives a change to the deal algorithm itself and not merely to the
shuffle. See `research/plans/P13_one_game_driver.md` § "a replay must carry its own randomness".
