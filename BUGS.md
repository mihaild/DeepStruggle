# Known bugs

Defects that are known, reproducible, and not yet fixed. A bug leaves this file when it is fixed
and has a regression test, not when it is understood.

Anything here is a thing the code gets *wrong*, not a thing it does not do yet. Missing features
and unfinished work belong in the documentation for the area they concern.

---

## ENG-3 — the forced-play rule is enforced differently by the mask and the state machine

**Area:** engine · **Severity:** high (it kills a training run) · **Status:** open, needs owner
decision (invariant 11)

Missile Envy's forced play is implemented twice, with different guards, so the action mask offers
play modes that `StateMachine::step` then refuses.

| | check |
|---|---|
| `engine/src/action_mask.cpp:200` | `current_phase == ACTION_ROUND && forced_card_player == p && (forced_card_id == card \|\| forced_card_id == MISSILE_ENVY)` → OPS only |
| `engine/src/state_machine.cpp:934` | `forced_card_player == p && (forced_card_id == card \|\| forced_card_id == MISSILE_ENVY)` → refuse any mode but OPS |

Two differences, and both produce a disagreement:

1. **The phase guard.** The mask restricts to OPS only during an action round; the state machine
   restricts in *every* phase, including `HEADLINE`. The mask's own comment gives the reason for
   its guard — "a headlined card is played as its Event as usual" — and the printed card text
   agrees: the opponent must play Missile Envy *on their next action round*.
2. **Ordering against the China Card.** The mask handles the China Card and returns `OPS|SPACE`
   *before* it reaches the forced-play branch; the state machine checks forced play *first*. So a
   forced player holding the China Card is offered SPACE and refused it.

**Reproduce.** Reach any `SELECT_PLAY_MODE` node, set `forced_card_player` to the mover and
`forced_card_id = 49`, then compare the mask against the engine:

```
B. forced during HEADLINE   mask offers EVENT, OPS, SPACE   engine refuses EVENT, SPACE
C. forced, pending card = 6 mask offers OPS, SPACE          engine refuses SPACE
```

**How it surfaced.** It killed `E3-21-28` twice at 3.9M of 160M steps, deterministically, with
`engine refused flat action 112 in env 39 of 512 (SELECT_PLAY_MODE) ... The action WAS legal in the
mask it was sampled from`. Before the batched-step guard (`59d2331`) this was silent: the env
simply did not advance and the trainer credited the transition anyway. It is rare — one occurrence
in ~4M env-steps, and 0 in a 22,747-action fuzz of random play — which is why it went unnoticed.

**Fix, once decided.** The rules text points at the mask being right and the state machine needing
the `Phase::ACTION_ROUND` guard; the China Card ordering is a separate question, since the
wildcard `forced_card_id == MISSILE_ENVY` means "while forced, any card must go to Ops", which the
mask does not implement for the China Card. Both halves are the owner's call: they change what is
legal, so they change the decision stream and invalidate comparisons across the change.

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
