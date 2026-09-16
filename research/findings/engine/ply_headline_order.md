# `ply` numbered headlines by side; the engine resolves them in Ops order

**Status:** fixed 2026-09-16 in `ai/game_length.py`. Previously logged `terminal_plies` and
decisiveness numbers **stand** — the correction is 0.00026 SD, sized below.

## What was wrong

`ai.game_length.ply(turn, action_round, is_us)` numbered a turn's two headline plies USSR first,
US second. The engine does not resolve them in that order: both players reveal simultaneously and
the higher-Ops card resolves first, so when the US holds the bigger headline the walk observed
ply 30 and then ply 29 — the index ran backwards inside `Phase::HEADLINE`.

Reproduced deterministically at seed 2, step 121:

```
turn=3 AR=0 phase=Phase.HEADLINE phasing=Player.USSR
decision=DecisionType.POINT_NODE decision_player=Player.USSR
previous ply 30 -> current ply 29
```

## The fix

`ply` now takes `headline_stage` and orders the two headline plies by **resolution**, not by side.
`state.headline_stage` is the engine's own marker: 0 while both players are still choosing, 1
while the first card resolves, 2 while the second does. Stages 0 and 1 are the turn's first
headline ply; stage 2 is its second. Action rounds are unchanged — `advance_after_action_round`
really does alternate USSR then US.

`headline_stage` is **required** when `action_round == 0`, not defaulted. A wrong ply returns a
plausible number rather than failing, so the omission has to raise; this repo has been bitten
before by a defaulted parameter that silently returned the wrong thing.

`FULL_GAME_PLIES` stays 154 and `plies_in_turn` is unchanged: a headline is still two plies, one
per side. Only their order within the turn changed, so the ply scale is the same scale and no
previous number is on a different axis.

## Size of the correction

Over 2,000 random-policy games:

| quantity | value |
|:---|---:|
| games terminating inside a headline | 24 (1.20%) |
| games whose terminal ply changed | 14 (0.70%) |
| shift per affected game | exactly 1 ply |
| mean shift over all games | 0.007 plies |
| mean terminal ply / SD | 40.44 / 26.54 |
| **shift as a fraction of one SD** | **0.00026** |

Caveat on the sample: random play ends early (mean terminal ply 40 of 154), so the *rate* at which
real games end in a headline may differ. The per-game bound does not — it is one ply, always, and
only for a game that ends between the two headline cards.

`ply` reaches one metric, `terminal_plies` in `bindings/ts_env.py`, which feeds the decisiveness
metric. It reaches neither the observation, the reward, nor the action space.

## How it stayed hidden

`tests/engine_logic/test_game_length_plies.py::test_ply_never_runs_backwards_across_a_real_game`
is exactly the test that should have caught it, and it was vacuous. It took its action from
`Engine.get_legal_action_indices` — the 128-wide **per-decision** space — and fed it to
`Engine.step_flat`, which reads the **flat 212-dim** space. Every step was refused, the refusal
discarded because `step` returned a bool nobody read, and the loop ran its full 4000 iterations
against a state that never moved. The monotonicity assertion passed having never seen a second ply.

Making `step` raise ([[engine_change_decision_stream]], commit `941b45b`) is what surfaced it. The
test now asserts monotonicity across the whole walk, headlines included, over five seeds.
