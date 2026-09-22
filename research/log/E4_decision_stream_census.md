# E4 decision stream — how cards are played, and what merging the op mode would save

Asked 2026-09-22: most cards look like they are played for ops. If the op-mode choice were folded
into the first placement (choose the card, then "coup / realign / place first influence in
country X"), how many decisions would a game lose, and what would that buy?

## Data

95 traced self-play games of the M2d s3 lineage, 160M–240M: `data/replays/wg_scan/` (5 per
snapshot) and `m2d_{160M,240M}_selfplay_*`. Temperature 0.1. Auto-advance is on, so recorded
decisions are the ones the policy made: 36,140 of 36,637 had more than one legal action.

## The stream today

A card played in an action round is `SelectCard → Resolution → PointNode × n`. The engine
already merges play mode and op mode into the one `Resolution` choice (event / ops-influence /
ops-coup / ops-realign / space). When the opponent's event fires first, the player's op choice
comes afterwards as a separate `OP_MODE` decision.

| decision type | per game |
|:---|---:|
| action-round `POINT` (placements, coup and realign targets, event placements) | 155.8 |
| action-round `CARD` | 91.2 |
| action-round `RESOLUTION` | 77.9 |
| action-round `OP_MODE` after an opponent's event | 15.7 |
| headline `CARD` | 15.3 |
| setup `POINT` | 15.0 |
| headline, other | 13.7 |
| action-round `BRANCH` | 0.9 |
| **total** | **386** |

How action-round cards are resolved (`RESOLUTION` plus post-event `OP_MODE`):

| resolution | per game | share | US | USSR |
|:---|---:|---:|---:|---:|
| ops — influence | 43.0 | 45.9% | 21.8 | 21.2 |
| event | 24.7 | 26.4% | 10.9 | 13.9 |
| ops — coup | 14.3 | 15.3% | 6.5 | 7.9 |
| space race | 9.1 | 9.8% | 5.7 | 3.4 |
| ops — realignment | 2.4 | 2.6% | 1.1 | 1.3 |

**Ops are 64% of card resolutions**: influence 46%, coup 15%, realignment 3%.

## What the merge would save

* **(a) Influence only** — fold `Resolution: OPS_INFLUENCE` into the first placement:
  **43.0 decisions per game, 11.1%** of the stream.
* **(b) Plus coup and realignment target heads** — the op type and its first target in one
  decision: **59.7 per game, 15.5%**.

### In credit-assignment terms this equals a small λ change

Training runs γ = 1 and λ = 0.98 on the joint stream of both players' decisions
(`per_player_gae` off), so credit decays by λ per decision. Removing a fraction f of the decisions
is the same, for credit assignment, as raising λ to 1 − (1 − λ)(1 − f):

| | decisions / game | equivalent λ | effective horizon 1/(1−λ) | credit from game end to setup (≈ λ^370) |
|:---|---:|---:|---:|---:|
| today | 386 | 0.980 | 50 decisions ≈ 1.3 turns | 5.7e-4 |
| (a) | 343 | 0.9822 | 56 decisions | 1.3e-3 |
| (b) | 326 | 0.9831 | 59 decisions | 1.8e-3 |

The attribution gain is therefore available with a single flag, `gae_lambda`, and needs no engine
or action-space change. Merging the op mode is still worth considering for what λ cannot give:
about 15% fewer forward passes per game, and choosing an op together with its target. It should be
judged on those. It is an engine change (a new engine letter) and an action-space change: every
checkpoint and dataset is invalidated, and the ladder resets.
