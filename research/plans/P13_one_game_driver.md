# P13 — One game driver

**Status:** proposed
**Needs approval:** the migration touches training, tournaments, replays and the web server.

## The loop we want

```
state = init(seed)
while not terminal(state):
    mask   = legal_mask(state)
    action = source.choose(state, mask)     # policy | search | bot | human | replay
    state  = step(state, action)            # checked, recorded
```

That is the whole thing. Everything below exists to make the five current loops *be* that loop.

## Why it is not that today

Five loops drive a game forward -- `TsVectorizedEnv`, `play_match`, `BatchMatchRunner`, the
searcher, and `GameSession` -- each written for a different purpose at a different time, with
nothing forcing convergence. They diverge in four accidental ways:

* **Three encodings for one action.** Flat 212-index, `MicroAction`, and a JSON action dict.
  Conversions between them are where a searcher's action for the wrong decision slipped through.
* **Two state representations.** `GameState` for the engine, `GameStateDict` for the browser and
  bot protocol. `play_match` drives bots with dicts, so a searcher needing the real state had to be
  special-cased with `wants_game_state`.
* **Three ways to settle.** `drain_chance_nodes`, `Engine::auto_advance_step`, and
  `step(auto_advance=True)`. Nothing says which to use; each loop picked one, and a mismatch
  between two picks produced 1,355 refused actions recorded as though they had happened.
* **Recording bolted into the loop** rather than wrapping it, which is how refused actions reached
  a replay.

Two things genuinely complicate the loop and must be designed for, not designed away:

1. **Someone must answer decisions with no choice** -- die rolls, single-option nodes, forced
   targets. The engine surfaces them as real decisions.
2. **Training needs batching.** 13,000 steps/s requires many games in flight.

## Decision 1: record every action, including the forced ones

**A replay must be re-drivable by applying its actions in order, with no knowledge of who produced
it.** Today it is not. Re-driving the same ten replays through the engine:

| replay type | replayed without settling | replayed with auto-advance |
|:---|---:|---:|
| search (settles nothing, records every node) | **0 rejected** | 110–489 rejected |
| plain policy (relies on unrecorded auto-advance) | 122–648 rejected | 9–223 rejected |

The search replays are exactly faithful because every decision the engine asked about was answered
explicitly and written down. The plain ones are clean under neither policy, because the engine
resolved chance nodes itself and nothing recorded that it had.

So a **recording** loop settles by *playing the forced actions itself* -- take the single legal
action, record it, step -- rather than calling `auto_advance_step`. No engine change is needed:
`auto_advance_step` returns a count, not the actions it took, so it cannot be recorded, and asking
it to report them would be the larger change.

Cost: a recorded game grows from ~182 decisions to ~336 steps. Replays are already 5–20MB and this
does not change their order of magnitude.

**Training keeps the fast path.** It records no replay, so it keeps `auto_advance` in C++ -- which
is 0.76x the wall time of the Python drain loop, i.e. faster as well as simpler. This is the one
justified fork, and it is justified by measurement rather than by convenience.

## Decision 2: one conversion boundary

Three functions, one module (`tools/lib/game_json.py`), wrapping what already exists in scattered
form:

| function | wraps | used by |
|:---|:---|:---|
| `state_to_json(state, for_role)` | `ts.state_to_dict` + session extras | viewer rendering, replay snapshots |
| `action_to_json(state, flat)` | `decode_flat_action` + description | replay writing, action log |
| `json_to_action(state, obj) -> flat` | `MicroAction` + `encode_micro_action` | viewer input, replay re-drive |

`json_to_action` is the one that does not exist today and is the reason the viewer cannot re-drive
a replay through the engine. With it, **the replay viewer and the live game become the same code
path**: a replay is just an action source that reads from a file.

Inside the loop there is exactly one encoding: the flat 212-index. JSON exists only at the edge.

## Decision 3: the viewer's three modes are three action sources

`GameSession` already owns a `GameState`, validates against the `DecisionContext`, steps, and
broadcasts. Under the driver it keeps all of that and differs only in where actions come from:

* **replay viewing** -- source reads recorded actions in order; stepping the engine reproduces the
  game rather than trusting the stored snapshots. This is what makes the viewer a *check* on the
  engine instead of a picture of it.
* **hot seat** -- source is the browser, via `json_to_action`. Both seats human, which is the mode
  that tests engine rules directly.
* **against a bot** -- one seat is the browser, the other a `PlayerAgent`. Identical loop.

## Decision 4: a replay must carry its own randomness

A replay today stores `initial_state = {"seed": 4001}` and nothing else. Re-driving means
`init_game(seed)` and replaying actions, so **any** engine change to setup, shuffling or the order
in which random numbers are drawn makes every recorded game diverge -- silently, mid-replay,
producing a different game rather than an error. That is broader than the observation-layout case:
an observation change does not touch `GameState` at all, and the viewer renders from
`state_snapshot`, so those replays already survive.

Three gaps, of which one is already solved:

**Die rolls -- already designed, and it works.** `state_machine.cpp` is explicit: *"The only source
of a forced die is this action. `primary_id` is the acting player's die and `secondary_id` the
opponent's."* A `ROLL_DIE` action carrying the value is the supported override, and the
decision-type guard preserves it -- verified across all six values at box 4, where 1-3 advance and
4-6 do not. What the guard closed was the *mis-encoded* route: flat 203-208 decode to
`CHOOSE_BRANCH`, and the value reached `primary_id` only because the engine was not checking the
type. Same laxity that let a `SELECT_CARD` action consume a coup's roll.

**Card deals -- no equivalent exists.** `deal_cards_to_hands` draws from the deck through
`state.rng_state` with no forced-deal path. So even with every die recorded, a replay diverges at
the first deal if shuffling changes. This needs a matching affordance: a replay source supplies the
cards a deal produces, ignored in normal play. Recording *what was dealt* is more robust than
recording deck order, because it survives changes to the deal algorithm itself and not merely to
the shuffle.

**The starting position -- `to_dict` is lossy.** It omits `rng_state` entirely and carries only the
top `decision_context`, not the `ctx_stack_depth` nesting, so no round-trip is possible today.

### What to build

* extend `to_dict` to be semantically complete (`rng_state`, full ctx stack) and add `from_dict`
  that **defaults anything missing**. Named fields, not a raw struct blob: a blob is perfectly
  lossless and layout-fragile, so adding one field would unload every old replay -- exactly the
  failure this is meant to prevent. With defaults, a replay recorded before a starting-bonus
  feature loads with the bonus at its default and plays.
* a forced-deal affordance mirroring the forced die.
* record, per replay: the full starting state, every action *including the forced ones*, every die
  result, and every deal.

A replay is then self-contained. It re-drives on any engine version, and where the engine has
genuinely changed it **fails at that point** rather than quietly producing a different game --
which turns the replay corpus into a differential test of engine changes, the same role
`ts_replayer` plays for human games.

## Migration order

Each step is independently useful and revertable.

1. **`step_checked`** in `tools/lib/game_step.py` -- **done**. The nucleus: a refused action raises
   instead of being ignored or recorded.
2. **`game_json.py`** with the three conversions, and `json_to_action` tested round-trip against
   `action_to_json` over a full game -- **done**.
3. **`GameLoop`** with an injected source and an explicit settle policy -- **done**.
4. **Recording as a wrapper** -- **done for both replay writers**: `tools/play_match.py` and
   `tools/lib/self_play.py` (the writer invariant 6 names) now build sources and delegate. Both
   carried the same defect, and it is the one this step existed to remove: each called
   `log_step` and only *then* checked the engine's return value, so a refused action was written
   into the replay as though it had happened. Under `GameLoop` a refusal raises out of
   `step_checked` and nothing is recorded.
5. **`GameSession`** onto the shared primitive -- **done**, though not in the shape this plan
   assumed. The web session is *driven* by the network rather than driving itself: it applies one
   action per message and returns. So it shares what sits underneath `GameLoop` -- `step_checked`
   plus `drain_chance` -- rather than wrapping `GameLoop.run()`. Wrapping the loop would have meant
   inverting control of a websocket handler for no gain.

   This closed a live hang. The session's private drain looped `while` the game sat on a chance
   node and exited only when the engine *accepted* the roll, discarding the return value. That was
   survivable only because the engine accepted every forced die; once a die outside 0..6 is
   refused, a client sending `secondary_id: 99` spins that loop forever. Measured: the old drain
   body iterated 1,000 times without leaving the node for dice 7, 99 and 255, while a valid die
   exits after one. It now raises, the handler rolls back to the snapshot it already took, and the
   session stays playable.

   The manual-roll affordance is preserved: the UI's `selectedDieRoll` (0 = auto, 1..6) rides in
   `secondary_id` and is passed as `drain_chance(forced_die=...)`, which is deliberate workbench
   behaviour for testing the engine, not a leaked field.
6. **Training and `BatchMatchRunner`** -- **done, and deliberately not by wrapping `GameLoop`.**

   Both drive N games at once through the C++ `VectorizedBatchRunner`, which already drains chance
   nodes (`ts_bindings.cpp:1127`) and already refuses illegal actions. Putting them inside a
   one-game Python loop would destroy exactly the batching this step's regression budget exists to
   protect, so they share the *contract* rather than the implementation.

   What was actually missing was the second half of that contract. `step_flat_all` returns a
   per-game verdict -- 0 refused, 1 accepted, 2 terminal -- and **both callers discarded the
   vector**: the same unchecked-return defect as the replay writers, in the hottest paths. A
   refusal there means a game silently did not advance while the trainer credits the transition and
   the tournament counts the result.

   Measured before adding the check: **0 refusals in 5,052 batched steps** with actions sampled from
   the legal mask. So these are guards, not fixes -- and a guard nothing exercises is decoration,
   so `tests/training/test_batched_step_is_checked.py` forces a refusal in one game of a batch and
   requires it to be raised and to name the game.

   **Cost.** `0 in results` is a C-level scan with no allocation and lands inside run-to-run noise
   on a batch of 256 (the raw call itself varied 130-146 us between runs). The obvious
   `np.asarray(results)` version cost **+29%** of the step call, which is why it is not used; every
   expensive part of the diagnostic runs only after a refusal has been found.

## The replay format's contract, as its readers implement it

This was read off the readers rather than assumed, after an earlier verification harness applied a
replay's actions with no draining and reported 85-95% refusals on files that are in fact correct.
The harness was wrong, not the replays.

* **One drain, in `tools/lib/game_step.py`.** `drain_chance` is the single implementation; it was
  written four times before, and the web copy was the one that could not fail safely.
* **A chance node is not a step.** A `ROLL_DIE` with `decision_player == NONE` is reproducible from
  the RNG, so no writer records it and every reader regenerates it
  (`tests/replayer/test_replay_reproduces.py::_drain`, `web/ui`). `GameLoop` drains them inside the
  step and never offers one to an `ActionSource` -- which also closes the hole that let a *bot*
  choose its own dice (see ENG-2 in `BUGS.md`).
* **`state_snapshot` is the state AFTER the action and after that drain.** Confirmed by
  `web/server/session.py:475` (`state_snapshot=state_after`), `web/ui/src/main.ts:217` (reads
  `state_snapshot.die_roll`, which only exists once the roll resolved) and `main.ts:178` (diffs VP
  between consecutive snapshots).
* **Rolling is a real action, and it used to roll 255.** `generate_flat_mask_212` had no
  `ROLL_DIE` case, so its fallback supplied flat 211, which decodes to `primary_id = 255` — and for
  `ROLL_DIE`, `primary_id` is the *forced die* rather than an index. Any caller that read the mask
  and called `step_flat` therefore forced a maximum roll: on seed 777 a space race gave VP +3
  against +1 with a real roll, and on seed 99 a coup rolled 255 against 5. The old `play_match` was
  such a caller. Fixed in the engine with an explicit mask case, a `ROLL_DIE`-first decode, and a
  range guard on forced dice; two regression tests in `engine/tests/test_bugs_regression.cpp`.
* **`flat_action_idx` is mandatory.** `_assert_reproduces` *skips* a replay that lacks it rather
  than failing, so omitting the field retires the check instead of breaking it.

A single-option *player* decision is a real step and is recorded (`SettlePolicy.RECORD_FORCED`);
only chance nodes are silent.

## Constraints

* **No decision-stream change** (invariant 13): any behavioural difference invalidates checkpoints
  and Elo. Verify as the search rewrite was verified -- same seeds, same actions, same visit
  counts, which held at 288/288 positions.
* **Training throughput must not regress** beyond ~2%; measure before and after, and keep a raw
  path behind a flag if it does.
* **`ts::VectorizedBatchRunner` stays a batching primitive** underneath the loop, not a loop beside
  it. Note that it *resolves chance nodes on `set_state`*, which is why it must never be used to
  featurise a state the caller still owns.
