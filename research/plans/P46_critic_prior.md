# P46 — the critic predicts the result relative to a prior for this opponent and side (2026-11-14)

**Status:** proposed (owner's idea, 2026-11-14). P46a, a check that trains nothing, first; P46b is not built.
**Gate:** P46b waits for P46a's pass, for the owner's approval (it changes how the value is computed), and for
one opponent per game in the own pool (P45's L6, the owner's call).
**Needs approval:** P46a none. P46b: trainer-side only, no engine, bindings or observation change.

## Goal

A critic that does not have to guess who it is playing, without a head per opponent group: the trainer keeps,
for each opponent and seat, a running average of the learner's recent results, and the critic predicts the
result **relative to that prior**. P46a asks whether a prior is enough -- whether an opponent's strength acts
on the self-play value as one shift -- before anything is built.

## Why

* **The pool taught the critic to read style.** 30% of games are against older snapshots that lose 55-73%;
  the critic has no input saying who plays, so "the side that plays unlike the current self loses" is its
  best predictor: 8-13 points against a side whose turn 1 an old checkpoint played, where the real effect
  is 0-2.6. Training the value on self-play rows only (R124, `--value-selfplay-only`) removed it in 390M
  (-8.4 / -11.4 -> +0.1 / -1.9), and by 12,400M the branch plays the human USSR setup in 84% of deals and
  beats the control 54.1% head to head ([report](../reports/p44_strategy_drift.html)).
* **R124 is an interim.** Its pool games' advantages come from a critic that never trained on them, which
  lambda < 1 turns into a biased policy gradient there.
* **A head per group (self-play, own pool, league) leaves the same shortcut inside a group:** the own pool
  spans opponents the learner beats 55% and 73% of the time, and a league adds members of both kinds. A
  prior per opponent has no groups.
* **Precedents:** KataGo's 2020 run tells the network which side is stronger as a scalar, not as an identity
  (lightvector's run notes); PopArt has the critic predict returns relative to a running per-task mean
  (Hessel et al. 2019). Neither is a self-play pool.

## The forms to tell apart

`p_f(s)` is the self-play critic's win probability for the learner at position `s` (R124's win head);
`r(o, side)` the learner's running win rate against opponent `o` in that seat; `r_self(side)` its self-play
rate in that seat.

| | prediction against opponent `o` | expectation |
|:---|:---|:---|
| M0 | `p_f(s)` | off by the opponent's edge at every turn |
| D | `p_f(s) + r(o, side) - r_self(side)` (the plain difference) | right at the first move, wrong late: a won position is won whoever the opponent is, so the same board gets different targets and style pays again, late |
| L | `sigmoid(logit p_f(s) + beta(o, side))`, `beta = logit r(o, side) - logit r_self(side)` | decided positions saturate, so the shift fades by itself |
| W | `sigmoid(logit p_f(s) + w(turn) * beta(o, side))`, `w` fitted per turn | L, if strength matters less with fewer decisions left |

In self-play games `beta = 0` in every form: the prior is about the opponent, not the seat. The 2-3 points
the critic holds against the US at setup in both runs are therefore not this plan's to remove (a Monte
Carlo value target at setup states is; P44's report, step 5).

## P46a — is an opponent's strength a shift on the self-play value? (CPU, no training)

**Critic and player:** R124's 12,320-12,400M SWA (`E8.1-A8-R119-S45@11230M+R124`), whose value was trained on
self-play rows only. The control's SWA at the same step is run through the same games for contrast (its
critic is the mixture critic).

**Games:** the R124 network against each of the control's own snapshots at 400M, 2,000M, 4,000M, 6,000M,
8,000M, 10,000M and 12,000M, and against itself; **one opponent for the whole game**; the learner in each
seat; 2,000 deals per opponent and seat with the same deal seeds for every opponent; both sides sampled at
T = 1, as training games are. 28,000 pool-style games and 4,000 self-play games. At every decision of the
R124 side: `p_f`, the turn and action round, and at the end the result. Odd deal seeds fit, even ones test.

**Read, on the test half:**

1. **Calibration** of M0, D, L and W, per opponent and seat, in three bands of turns (1-3, 4-7, 8-10) and in
   three bands of `|logit p_f|`: mean result minus mean prediction, in points, with its standard error.
2. **Log loss** of each form, and of L with `beta` fitted freely per opponent and seat.
3. **Is the prior enough?** The fitted `beta` against `logit r - logit r_self` computed from the fit half's
   win rates alone: the trainer would only have the latter.
4. **W's profile:** the fitted `w` by turn (1 everywhere means L is the form).
5. **What is left for style to exploit:** the largest absolute mean residual of the best form over (opponent,
   seat, turn band) cells, beside the same for self-play games (the noise floor); and the handover positions
   (an old checkpoint's turn 1 for one side, then the old checkpoint **continues** that side) scored by the
   best form.
6. **The control's critic on the same games**, raw: how much of the pool it already encodes, by opponent.

**Pass:** L or W within 2.5 points in every cell of read 1 (cells of 2,000 games resolve about 1.5), `beta`
from the rates within 0.1 of the fitted one, and read 5 no larger than twice the self-play floor.
**If only W passes,** P46b carries a weight on the prior that depends on the position (a small critic-only
output), not on the opponent. **If neither passes** (a band off by 5 points or more), strength is not a
shift, the prior is dropped, and the opponent enters the critic as an input or as group heads (P44's report,
step 2).

## P46b — the arm, if P46a passes

The control branched at 11,230M, where R124 branched: the win head trained on **every** row with the shift
(`sigmoid(z(s) + beta)`, or W's form), `beta` from exponential averages of the learner's results per opponent
and seat (the pool already counts them; a new member starts at the pool's mean), the same value used in
that game's advantages, `beta = 0` in self-play. 1,200M, read at R124's steps.

* Against R124 and the control at the same step: the handover error, the whole-setup preference, the
  critic's calibration in pool games by opponent (R124's is expected to be off by the opponent's edge; this
  arm's within 2-3 points), setups by step, the old reference, head to head.
* **Adopt over R124 and over group heads** when the handover error and the setup preference are at R124's
  level, pool games are calibrated, and strength is not below R124's on the old reference or head to head.
* The VP head stays as in R124 (self-play rows) in this arm; a prior for it is a follow-up.

## Runs (filled in while running)
