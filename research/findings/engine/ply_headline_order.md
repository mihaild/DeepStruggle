# `ply` numbers headlines USSR-then-US; the engine resolves them in ops order

**Status:** open, unsized beyond the bound below. Found 2026-09-16 while making
`StateMachine::step` `[[nodiscard]]` (see [[engine_change_decision_stream]]).

## What

`ai/game_length.ply(turn, action_round, is_us)` assigns the two headline plies of a turn as
USSR first, US second:

```
ply(3, 0, is_us=False) == 29
ply(3, 0, is_us=True)  == 30
```

The engine does not resolve headlines in that order. Both sides reveal simultaneously and the
higher-ops card resolves first, so when the US holds the higher-ops headline the walk observes
ply 30 and then ply 29 — the index runs backwards inside `Phase::HEADLINE`.

Reproduced deterministically at seed 2, step 121:

```
turn=3 AR=0 phase=Phase.HEADLINE phasing=Player.USSR
decision=DecisionType.POINT_NODE decision_player=Player.USSR
previous ply 30 -> current ply 29
```

## Why it went unnoticed

`tests/engine_logic/test_game_length_plies.py::test_ply_never_runs_backwards_across_a_real_game`
is exactly the test that should have caught it, and it was vacuous. It took its action from
`Engine.get_legal_action_indices`, which returns indices into the **128-wide per-decision** mask,
and fed them to `Engine.step_flat`, which reads the **flat 212-dim** space. Every step was
refused; the refusal was discarded because `step` returned a bool nobody read; the loop ran its
full 4000 iterations against a state that never moved. The monotonicity assertion passed because
it never saw a second ply.

Making `step` raise is what surfaced it.

## Impact

`ply` has one consumer that reaches a metric: `bindings/ts_env.py` computes `terminal_plies` from
it, which feeds the decisiveness metric (`--decisiveness-turns`). The error is bounded at **one
ply out of 154**, and only for a game that terminates inside a headline. Every other consumer
(`tournament_evaluator`, `batch_tournament`) uses it the same way, as a length.

It does **not** reach the observation, the reward, or the action space.

## Not fixed here

Two defensible resolutions, and the choice is the owner's because it moves a logged metric:

1. Make `ply` order the headline by resolving ops rather than by side. Correct, but `ply` would
   then need the card ops, which it does not currently take.
2. Give both headline plies of a turn the same index, since "which side resolved first" is not
   a length distinction. Cheapest, and adequate for every current consumer.

The test now pins what is actually true — `(turn, action_round)` never runs backwards, and `ply`
is monotonic across action rounds — and documents the headline exception rather than asserting
it away.
