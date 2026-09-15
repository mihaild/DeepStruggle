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

## Migration order

Each step is independently useful and revertable.

1. **`step_checked`** in `tools/lib/game_step.py` -- done. The nucleus: a refused action raises
   instead of being ignored or recorded.
2. **`game_json.py`** with the three conversions, and `json_to_action` tested round-trip against
   `action_to_json` over a full game.
3. **`GameLoop`** with an injected source and an explicit settle policy. `play_match` and
   `BatchMatchRunner` migrate first: least coupled, already the same shape.
4. **Recording as a wrapper**, writing every action including forced ones. Verified by re-driving
   every replay the suite produces and requiring **zero** refusals, with no settle policy applied.
5. **`GameSession`** onto the loop, with the three sources above.
6. **Training last** -- hottest path, costliest regression, and the only caller keeping the
   batched fast path.

## Constraints

* **No decision-stream change** (invariant 13): any behavioural difference invalidates checkpoints
  and Elo. Verify as the search rewrite was verified -- same seeds, same actions, same visit
  counts, which held at 288/288 positions.
* **Training throughput must not regress** beyond ~2%; measure before and after, and keep a raw
  path behind a flag if it does.
* **`ts::VectorizedBatchRunner` stays a batching primitive** underneath the loop, not a loop beside
  it. Note that it *resolves chance nodes on `set_state`*, which is why it must never be used to
  featurise a state the caller still owns.
