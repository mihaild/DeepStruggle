# The afterstate critic inside the advantage: what one rollout says (2026-11-21)

**Conclusion.**

* **What a side does not control is 90% of the variance of a step that has an afterstate:** its own move changes
  the critic's value by 0.042 (SD), its dice by 0.040, the opponent's turn by 0.097.
* **Only 30% of decisions have an afterstate;** the rest lead straight to the side's own next decision, a step of
  0.034.
* **A separate decay at the afterstate is mostly a much shorter reach:** at lambda 0.96, `lambda_opp` 0.8 / 0.5 /
  0 takes the advantage's SD from 0.231 to 0.130 / 0.078 / 0.052, as the plain chain does at lambda 0.88 / 0.69 /
  below 0.6.
* **It is a little better than a shorter lambda:** at equal return error its spread is 6% lower at
  `lambda_opp` 0.8, 11% at 0.65, 15% at 0.5 -- the first estimator measured here that leaves the common
  frontier of [`P50_lambda_clock_and_own_view.md`](P50_lambda_clock_and_own_view.md).
* **The afterstate critic is as calibrated as the critic at decisions** (result against value, RMSE 0.849
  against 0.851), and its own error is a small part of what a move changes: cov(d1, d2) is -0.00024 against
  var(d1) 0.00172.
* **Between the moves the policy considers, the afterstate critic separates little:** 0.013 (SD, weighted by the
  policy) over 18 legal moves, against 0.029 by which the moves' mean differs from the decision's own value. Its
  best move is the policy's top move in 52% of decisions and holds 0.42 of the policy's probability.

**Question** (owner, 2026-11-21, through the running session). Split each side's own step at the afterstate and
give what follows it its own decay: is it sound, and is it worth an arm. Plan:
[`../plans/P56_afterstate_advantage.md`](../plans/P56_afterstate_advantage.md).

**Instrument.** Self-play of `E8.1-A8-R169-S45` (SWA 1,120-1,200M: B6's recipe with the afterstate and pre-deal
critics trained from scratch), sampled at temperature 1, 300 games, 127,853 decisions, CPU, the running
worktree's engine build. A session probe (not in the repository);
`data/reports/p56_afterstate_adv_r169_1200M.json` (untracked). The afterstate is the first moment
`ai.training.afterstate_critic.walk_afterstates` returns after the action taken (right after the action, before
any die), read from the mover's view with the same network; "after its dice" is the walk's last moment. Each
side's own chain, gamma 1:

```
d1 = V_after - V(s)                 d2 = V(the side's next decision) - V_after      (the result where the game ends)
A  = d1 + lambda_opp * (d2 + lambda * A_next)          no afterstate:  A = step + lambda * A_next
```

`lambda_opp` 1 is per-player GAE.

## 1. The parts of a step

| | SD |
|:---|---:|
| d1, the move's own effect (decisions with an afterstate, 30.0%) | 0.0415 |
| the side's dice (0.0636 where a die is rolled; a die follows 11.7% of decisions) | 0.0398 |
| the opponent's turn, to the side's next decision | 0.0967 |
| d2, the two together | 0.1042 |
| the whole step, d1 + d2 | 0.1100 |
| a step with no afterstate (70.0%) | 0.0344 |

cov(d1, d2) -0.000241 (correlation -0.056); var(d1) 0.001721. An error of the afterstate critic that the
decision's value does not share enters d1 and d2 with opposite signs, so minus the covariance estimates its
variance: about a seventh of var(d1).

| decision type | decisions | with an afterstate | SD d1 | SD d2 | corr(d1, d2) | result - V_after, mean |
|:---|---:|---:|---:|---:|---:|---:|
| card | 35,128 | 18.7% | 0.0498 | 0.1028 | -0.033 | -0.021 ± 0.011 |
| play mode | 26,306 | 20.1% | 0.0420 | 0.1011 | -0.088 | -0.003 ± 0.012 |
| point | 62,289 | 42.2% | 0.0390 | 0.1052 | -0.057 | -0.003 ± 0.005 |

Result against value, RMSE, at the afterstate / at the decision: US 0.849 / 0.851, USSR 0.847 / 0.847; turns 1-3
0.932 / 0.932, 4-7 0.822 / 0.822, 8-10 0.740 / 0.744; a die follows 0.832 / 0.833, none 0.858 / 0.859. (The
errors quoted are per row; the rows of one game share its result, so the per-seat means of this sample, -0.078
and +0.074, are 300 games' luck and the same at both nodes.)

## 2. The advantage over lambda_opp, lambda 0.96

| lambda_opp | advantage SD | return RMSE | correlation with lambda_opp 1 | the plain chain's SD at the same return RMSE (its lambda) |
|---:|---:|---:|---:|:---|
| 1.0 | 0.2313 | 0.7638 | 1.000 | 0.2313 (0.96) |
| 0.9 | 0.1653 | 0.8074 | 0.957 | |
| 0.8 | 0.1299 | 0.8258 | 0.885 | 0.139 (0.88) |
| 0.65 | 0.0979 | 0.8395 | 0.771 | 0.110 (0.81) |
| 0.5 | 0.0777 | 0.8468 | 0.659 | 0.092 (0.69) |
| 0.3 | 0.0609 | 0.8527 | 0.499 | |
| 0.0 | 0.0523 | 0.8583 | 0.214 | |

The plain chain (lambda_opp 1) at lambda 0.9 / 0.8 / 0.6: 0.1482 / 0.8215, 0.1074 / 0.8411, 0.0811 / 0.8510; the
last column interpolates between them. Return RMSE measures how far the target leans on the result, not quality.

## 3. Every legal move's afterstate

On 3,188 decisions (2.5% sampled; 18.2 legal moves on average), each legal move's afterstate valued (the side's
next decision where the move leads straight to it, the result where it ends the game).

| | all | card | play mode | op mode | point |
|:---|---:|---:|---:|---:|---:|
| decisions | 3,188 | 887 | 683 | 102 | 1,506 |
| sum over the policy of V_after, minus V(s): mean | +0.0006 | +0.0014 | -0.0001 | +0.0065 | 0.0000 |
| the same, SD | 0.0288 | 0.0303 | 0.0243 | 0.0286 | 0.0291 |
| SD of V_after over the moves, weighted by the policy | 0.0128 | 0.0124 | 0.0176 | 0.0166 | 0.0105 |
| the critic's best move is the policy's top move | 51.7% | 47.0% | 62.7% | 84.3% | 47.1% |
| the critic's best move over the policy's mix, by the critic | +0.0114 | +0.0125 | +0.0118 | +0.0047 | +0.0111 |

Unweighted, the SD of V_after over the legal moves is 0.047; the critic's best move holds 0.42 of the policy's
probability; the correlation of log-probability with V_after over a decision's moves is 0.65.

**What these do not show.** Whether the moves the afterstate critic prefers are better: that is a match between
the player that picks the critic's best move and the policy, not run here. Whether a shorter reach trains a
stronger network: nothing shorter than lambda 0.97 has been trained
([`E4_late_dynamics.md`](E4_late_dynamics.md)). One rollout of one network at 1,200M, far from its plateau.
