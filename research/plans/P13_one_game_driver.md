# P13 — One game driver

**Status:** proposed
**Needs approval:** the migration touches training, tournaments, replays and the web server.
**Motivation:** a real corruption, not tidiness.

## The problem, as it actually appeared

Five independent loops drive a game forward:

| loop | used by |
|:---|:---|
| `bindings/ts_env.py` `TsVectorizedEnv` | training rollouts |
| `ts::VectorizedBatchRunner` (C++) | the above, and batched featurisation |
| `tools/play_match.py` | single matches, replay generation |
| `tools/lib/batch_tournament.py` `BatchMatchRunner` | tournaments |
| `ai/search/batched_mcts.py` | search |

Each answers two questions its own way, and nothing enforces either:

1. **Who settles the state** — auto-advance, drain chance nodes, or neither.
2. **Who checks `step_flat`'s return value** — in practice, nobody did.

Both gaps fired together in one game. The searcher settled its clone and returned an action for a
later decision; `play_match`, which does not settle, submitted it; the engine refused it and
changed nothing; the loop discarded the `False` and re-offered the same node. 1,355 identical
rejected actions until the step cap, recorded in the replay as if they had happened — which is why
it read as "CIA Created played twice" and "a coup result with no coup".

The engine now rejects mismatched actions (it previously *accepted* some, silently consuming a die
roll). That converts silent corruption into a visible refusal — but only for callers that look.

## The contract

One function owns stepping:

```python
def step_checked(state, action) -> None:
    """Advance the game, or raise. A rejected action is a CALLER bug, never a game event."""
```

A masked action is always accepted, so `False` means the caller chose outside the mask or handed
the engine a state it was not asking about. Raising is correct: silently ignoring it produced the
1,355-step loop, and recording it produced a replay of moves that never occurred.

Around that nucleus, `GameDriver` adds the rest of the shared contract:

* **settle policy**, declared once and explicitly, rather than assumed per loop;
* **who chooses the action** — policy, search, scripted bot, human — as the only thing callers vary;
* **what gets recorded**, so the replay writer never sees a rejected action.

## Migration order

Each step is independently useful and independently revertable.

1. **`step_checked` alone**, in `tools/lib/`, adopted by all five loops. Small, and it closes the
   class of bug that motivated this. *This is GameDriver's nucleus, not a stopgap to be removed.*
2. **Settle policy as an explicit parameter** on the searcher (`advance_root`, already done) and on
   `play_match`. Makes the mismatch impossible to reintroduce silently.
3. **`GameDriver`** wrapping both, with the action source injected. `play_match` and
   `BatchMatchRunner` migrate first: they are the least coupled and already share a shape.
4. **Replay writing** moves behind the driver, so `generate_self_play_replay` records only
   accepted actions. Fixes replay fidelity for good.
5. **Training last.** `TsVectorizedEnv` is the hottest path and the one where a regression is most
   expensive; it should migrate only once the driver is proven by the others.

## What this must not do

* **Not slow the training rollout.** It is the one loop where per-step overhead matters: 13,000
  steps/s today, and a Python-side check per step is affordable only because the engine call
  already dominates. Measure before and after; if it costs more than ~2%, training keeps the raw
  path with the check behind a debug flag.
* **Not unify the C++ `VectorizedBatchRunner`.** It is a batching primitive, not a game loop, and
  it belongs underneath the driver rather than beside it.
* **Not change the decision stream.** Any behavioural difference invalidates checkpoints and Elo
  (invariant 13). The migration must be verifiable as identical: same seeds, same actions, same
  visit counts -- the check already used for the search rewrite, which held at 288/288 positions.
