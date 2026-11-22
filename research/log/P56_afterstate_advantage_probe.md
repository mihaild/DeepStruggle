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

## 4. Is the critic before a die roll the average of the critic after it? (2026-11-21, the owner's question)

**No.** Unbiased on average, but position by position it differs from the rolls' average by 0.052 (SD), against
0.064 of luck in the roll itself; and it sees about a sixth of what the move changed.

The same network, 200 games; 4,993 moments right before a roll that follows the mover's action (half sampled),
each rolled 30 times from the same position with the random state reseeded, the value after the roll read from
the mover's view (the result where the roll ends the game, 0.9% of rolls).
`data/reports/p56_dice_consistency_r169_1200M.json` (untracked).

| | all | card | play mode | point |
|:---|---:|---:|---:|---:|
| rolls | 4,993 | 268 | 925 | 3,784 |
| V_before - mean V_after: mean | -0.0002 | -0.0027 | +0.0034 | -0.0009 |
| the same, SD net of the 30-roll sampling error | 0.0517 | 0.0369 | 0.0380 | 0.0555 |
| luck: RMS over positions of the SD of V_after over the rolls | 0.0638 | 0.0718 | 0.0485 | 0.0665 |

The move's effect, measured from the decision's own value: by the rolls' average SD 0.0562, by the critic before
the roll SD 0.0271; the regression slope of the second on the first is 0.17 (correlation 0.36). The first also
carries the after-roll critic's own error, which lowers the slope, so 0.17 is a floor; the critic before a roll
stays close to the value at the decision whatever the move.

**What follows.** For a move followed by a die (a coup, a realignment, the Space Race, a war: 39% of the
afterstates), the afterstate's value should be the rolls' average of the value after the roll, in the advantage
and as the before-roll row's training target. The roll's own step, the value after the actual roll minus that
average, then has mean zero by construction whatever the critic's quality, and can be left out of the advantage
with no bias.

## 5. The approved plan's checks 1 and 2, on the build (2026-11-21, the running session)

**Verdict: both pass.** T1's flags cost 11% of throughput, and leaving the roll's luck out of the advantage
biases no kind of move.

**Check 1, throughput.** R169's configuration, 60M steps from scratch, an idle GPU, one engine build
(`1d2cae4`).

| | steps a second | rollout | update |
|:---|---:|---:|---:|
| the baseline | 100.4k | 335 s | 251 s |
| T1's flags (`--per-player-gae --gae-lambda 0.96 --afterstate-advantage`) | 89.4k (-11%) | 409 s | 252 s |

The first build read 77k (-23%); the difference is one batched value read a rollout, a pinned fetch of the
outcomes, no re-extraction of a position already seen, and the per-player pass on the host (it was 12% of the
rollout as small GPU operations, so most of P50's 14% slowdown was probably that loop, not measured alone). On
trained play at 512 envs: 201 afterstate rows and 212 distinct outcomes an env step, from 54 rolls.

> **Update, the same day (`fc65976`): the cost is 4.5%, not 11%.** The same probe on the later build: the baseline
> 98.6k steps a second (rollout 343 s, update 255 s), T1's flags 94.1k (rollout 376 s, update 251 s). What
> changed: a realignment, the Summit and the Olympics enumerate the 11 differences of the dice, weighted
> (6 - |d|) / 36, instead of 36 pairs (the owner's point: they depend on the dice only through the difference;
> equal to the 36-pair enumeration on 2,246 rolls); a dynamic parallel schedule; one gather for the afterstate
> rows and the outcomes; per-player GAE in C++ (equal to the numpy loop within 5e-7). No result changes.

**Check 2, the new advantage on one rollout.** `E8.1-A8-R169-S45` at 2,010M (`snapshot_final`), the trainer's
own code path on the CPU: 1,024 games warmed through 4 x 256 steps, then one recorded rollout of 262,144 rows;
lambda 0.96, `lambda_opp` 1. The kind of move is the walk's first moment and its roll.

| kind of move | rows | advantage SD, luck kept | dropped | mean, kept | dropped | paired difference |
|:---|---:|---:|---:|---:|---:|---:|
| all | 262,144 | 0.2275 | 0.2185 (-4%) | +0.0016 | +0.0017 | +0.0001 ± 0.0001 |
| no afterstate (70% of rows) | | 0.2206 | 0.2119 | | | |
| an afterstate, no die | 48,563 | 0.2398 | 0.2323 | +0.0001 | +0.0002 | +0.0001 ± 0.0003 |
| coup | 12,443 | 0.2416 | 0.2275 (-6%) | +0.0080 | +0.0080 | +0.0000 ± 0.0008 |
| realignment | 5,824 | 0.2564 | 0.2470 (-4%) | +0.0005 | +0.0018 | +0.0013 ± 0.0011 |
| the Space Race | 4,732 | 0.2393 | 0.2275 (-5%) | +0.0041 | +0.0053 | +0.0012 ± 0.0011 |
| a war | 2,370 | 0.2514 | 0.2129 (-15%) | +0.0028 | +0.0031 | +0.0003 ± 0.0028 |
| other dice (the Olympics, the Summit, a trap) | 3,793 | 0.2669 | 0.2589 | +0.0001 | -0.0000 | -0.0001 ± 0.0012 |

* Every paired difference is within 1.2 standard errors of zero.
* Rows with no die of their own shrink too: the luck of later rolls sits in their chain.
* Outcomes read after merging, on trained play: 110,219 for 26,464 rolls, 4.16 a roll (2.9 on random play).
* With the luck kept, T1's advantage equals the plain per-player chain at 0.96 exactly (0.2275 both).

**The baseline for read 3.** The critic before a roll against its outcomes' exact average, R169 at 2,010M: 0.0293
mean absolute (signed mean -0.0048). Section 4's 0.052 is an SD at 1,200M by 30 sampled rolls; its mean absolute
was 0.0285, so the two agree and the gap has not closed by itself between 1,200M and 2,010M.

**The plain per-player chain's spread over lambda** (the same rollout): 0.50: 0.0805, 0.62: 0.0881, 0.70: 0.0961,
0.78: 0.1088, 0.82: 0.1185, 0.86: 0.1321, 0.90: 0.1532, 0.94: 0.1916, 0.96: 0.2275, 0.98: 0.2969. L8
(`lambda_opp` 0.8, luck dropped) has SD 0.1318, the plain chain's at 0.86; L5 0.0839, the plain chain's at 0.56;
T1 0.2185, between 0.95 and 0.96. Section 2's 0.88 and 0.69 were matched on return error on a network at 1,200M;
these are matched on spread on the baseline itself and are the ones the arms use.

**What the dice removal is worth.** 4% of the advantage's spread overall, 15% on wars: the dice are a small part
of what a side does not control (section 1: 0.040 against the opponent's 0.097). Step 1's case is that it is
exact and that the before-roll value gets a correct target, not the size of this number.

## 6. Pair 1: step 1 (T1) against the baseline continued (C), +800M (2026-11-22)

**Verdict: T1 meets its rule with room to spare (53.7% ± 0.8 against C, the references above C's); whether the
margin is more than one recipe's own noise waits for C'.** The one read that failed its prediction, the critic
before a roll, failed for a reason found the same day: the observation does not show which roll is pending.

Both from `E8.1-A8-R169-S45` at 2,010M to 2,810M in the two slots. C: R169 continued in its own directory. T1:
`E8.1-A8-R169-S45@2010M+R170` (`--per-player-gae --gae-lambda 0.96 --afterstate-advantage`, the only lines
`launch_flags.py --diff` shows). SWAs 2,720-2,800M, greedy, `tools/tournament.py`, per seat from the JSON. Runs,
matches and reads 3-4 by the running session.

| read | T1 | C |
|:---|:---|:---|
| 1. T1 against C, 2,000 deals a side | **53.7% ± 0.8** (as US 50.3, as USSR 56.0) | |
| 2. against J1's best, 1,000 deals a side | 43.1 ± 1.1 (42.0 / 43.0) | 41.4 ± 1.1 (40.3 / 41.0) |
| 2. against A13 | 52.9 ± 1.1 (50.4 / 54.3) | 50.3 ± 1.1 (45.7 / 53.5) |
| 3. the critic before a roll against its outcomes' exact average, mean absolute (signed) | 0.0282 (-0.0021) | 0.0277 (-0.0016) |
| 3. the before-roll row's training loss | 0.0088 (target: the average) | 0.0560 (target: the lambda-return) |
| 3. critic AUC, T1 - C over 161 windows of 5M | +0.0008, 95% CI [-0.0003, +0.0019] | |
| 3. critic Brier skill, T1 - C | +0.0013, CI [-0.0005, +0.0032] | |
| 4. entropy / clip fraction / KL / explained variance | 1.126 / 0.0375 / 0.0029 / 0.849 | 1.129 / 0.0369 / 0.0029 / 0.845 |
| 4. value loss / self-play US win rate / advantage SD | 0.0438 / 0.466 / 0.201 | 0.0440 / 0.469 / 0.211 |

Health was level at +20M, +100M and +400M as well. The commonest opening setup's share was not measured.
T1 differs from C by per-player GAE (P50: 51.5% ± 0.8 on the earlier base) and by step 1 together; the run does
not say how the 3.7 points divide.

### Why the critic before a roll did not move: the observation cannot tell which roll is pending

The plan predicted that training the before-roll row on its outcomes' exact average would close its 0.029 gap.
It did not. On R169's SWA at 2,000M (40 self-play games, a quarter of the decisions sampled, CPU; a session
probe), at 447 decisions with two or more legal moves each followed at once by a roll, the mover's observation of
the before-roll moment was compared between those moves:

| | |
|:---|---:|
| decisions where the before-roll observation is identical for every such move | 74.9% |
| the same, target choices (`point`, 408 decisions) | 82.1% |
| pairs of such moves with identical observations | 84.8% |
| SD of the network's before-roll value over a decision's roll moves | 0.0037 |

A coup or a realignment in one country and in another are the same observation before the die: the pending
target is not in it. So the critic there can learn only the average over the targets the policy picks, which is
close to the decision's own value -- the slope of 0.17 in section 4, and the low training loss beside an unmoved
gap in the table above.

**What follows.**

* **For the advantage, nothing is lost:** T1 does not read the network before a roll; the node's value is the
  outcomes' average of the values *after* the roll, where the board shows what happened.
* **Read 3's first line is withdrawn as a criterion.** It cannot move without the observation showing the pending
  target, which is a change of the observation and the owner's alone; nothing here asks for it.
* **Anything that reads the critic at a before-roll moment is reading a blind value:** the critic's-best-move
  player and a search leaf that stops there should take the outcomes' average too.

> **Walk fix, 2026-11-22 (fix verification V-4).** A die rolled right after another -- an Ops-first coup or
> realignment on Korean War or Arab-Israeli War followed by the war's own roll, two immediate-roll headlines back to
> back, a coup then the Summit -- had no before-roll row and no outcomes of its own: the walk kept a before-roll
> moment only after a forced step or at the decision itself, so the second roll's luck stayed in the advantage (about
> 1% of die rolls, despite audit T-1's later-roll term). Both walks (the runner's and `walk_afterstates`) now keep
> the first roll's after-state again as the second roll's before-roll row, with its outcomes, and `node_tensors`
> pairs it as any later roll. The numbers above and pair 1's runs were taken before it, with that 1% left in.
