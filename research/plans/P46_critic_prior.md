# P46 — the critic predicts the result relative to a prior for this opponent and side (2026-11-14)

**Status:** P46a run (see *Runs*); P46b and L6 built (owner, 2026-11-14): `--opponent-per-game`, `--critic-prior-window`, `--critic-prior-turn-weight` (recipe codes R126, R127). The ablation and the J1 openings test are under *Runs*.
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

## P46c — the weight on the prior is learned (owner, 2026-11-14)

P46b as built (R127) weights the prior by a fixed line in the turn, 0.98 -> 0.46, fitted in P46a on one
critic at 12,400M. P46c lets the critic learn the weight. **The only quantity from outside the network is the
prior**; how much of it applies is learned with the correction, by the same value loss.

**The form.** Two critic outputs, both functions of the position only: a correction `c(s)` (today's win
logit) and a gain `g(s)`. Win value = `sigmoid(c(s) +- g(s) * beta)`, `beta` the pool's prior for that member
and seat as built (+ on the learner's rows, - on the member's, 0 in self-play), applied in the stored values,
the bootstrap and the value loss as now.

**Why.** A correction that sees only the position cannot be right for every opponent: late in a game the
board already shows most of a weak opponent's mistakes, so only part of `beta` should still count, and how
much depends on `beta` itself -- which a position-only output can know only by reading style. P46a measured
it: the plain log-odds shift is off by up to 7 points late, the turn line by 3.0 with 4.5-5.8 left in
near-even positions, so the turn alone is not the right variable. From scratch the profile early in training
is not known at all.

| arm | from | start | question |
|:---|:---|:---|:---|
| A | the control's 11,230M state (R124's and R127's branch point) | `g = 0` everywhere, `c` the current output: the value is unchanged at the branch point | does a critic that has the style shortcut unlearn it when given the prior? |
| B | R124's 13,200M state, whose critic is already clean (P46a's W form was measured on it) | `g` at P46a's line, trained from there | what does the prior add over the value on self-play rows only? |

800M each, after R127. If one slot is free, A first: it also says whether the prior can be applied to a
checkpoint such as J1. A higher rate on the gain output is allowed if it moves slowly, and is recorded.

**Reads**, at +400M and +800M, beside R127 and R124 at the same step:

1. the handover error (the control -8.4 / -11.4, R124 +0.1 / -1.9);
2. the learned `g` by turn against 0.98 -> 0.46, and in the near-even band;
3. the critic's calibration in pool games by opponent (R124's is off by the opponent's edge by construction);
4. setups by step, the old reference, head to head.

**Prediction.** A: handover error under 3 points by +400M, with `g` within 0.15 of the line in turns 1-5.
B: pool games calibrated within 3 points from the first read, strength not below R124's.
**Rule.** In A, `g` under 0.3 with the handover error still above 5 at +400M -> the critic kept the shortcut
and a zero start is not enough; start `g` at the line, as R127 does. Otherwise the learned gain replaces the
fixed line in every later arm, from scratch included.

**The prior table, for every arm:** a new own-pool member starts at 50%, not at the members' mean for its
seat. It is the learner's snapshot from moments ago; early in a from-scratch run the mean is far from even,
and the running average needs about 700 games to settle. The window and this starting value are the table's
two settings.

## Runs (filled in while running)

### P46a (2026-11-14, CPU): W nearly passes, L and D fail

R124's 12,320-12,400M SWA against the control's snapshots at 400 / 2,000 / 4,000 / 6,000 / 8,000 / 10,000 / 12,000M
and against itself, one opponent a game, each seat, deal seeds 1-2,000 (the same for every opponent), T = 1 both sides:
32,000 games, 6.9M decisions of the R124 side (`p46a_games.py`, `p46a_read.py` in the session's tmp; data
`data/reports/p46a_games.npz`, full read `data/reports/p46a_read.txt`). Fit on odd seeds, read on even ones.
R124's score falls from 92-95% against 400M to 54-62% against 12,000M; in self-play 49.2% as US, 53.5% as USSR.

| read | M0 | D | L | W |
|:---|:---|:---|:---|:---|
| 2. log loss, pool games (L with beta fitted freely: 0.4623) | 0.5163 | 0.5371 | 0.4644 | **0.4612** |
| 1. largest \|mean residual\| over (opponent, seat, turn band) cells, points (SE ~1.2-1.4) | 31.8 | 9.6 | 7.1 | **3.0** |
| 1. the same over bands of \|logit p_f\| | 41.5 | 9.5 | 4.9 | 5.8 |

* **D** is right in turns 1-3 (within ~4) and too low late (-5 to -10 in turns 8-10), as expected: the plain
  difference over-predicts once the position has absorbed the opponent's weakness. **L** is right early and off by
  up to 7 late in the same direction.
* **3. beta from the rates is larger than the freely fitted constant beta:** gap -0.48 / -0.37 against 400M, -0.30 to
  -0.42 against 2,000M, -0.02 to -0.15 from 4,000M on. The gap is the turn profile below, not noise: a constant beta
  averages over turns.
* **4. W's weight falls with the turn:** 0.99, 0.92, 0.87, 0.80, 0.75, 0.69, 0.61, 0.56, 0.51, 0.39 (turns 1-10).
  The weak side's mistakes reach the board as the game goes on, so the self-play critic already sees part of the
  edge late; with this profile, beta from the rates alone gives the best log loss of all forms.
* **5.** W's largest cell, 3.0, against a self-play floor of 4.1 (the critic's US offset: -2.9 / -4.1 as US in
  self-play): ratio 0.72. At the handover points (the R124 side's first decision of turn 2, the old checkpoint
  continuing its side) W is within -2.0 to +3.1 against every opponent; the raw critic is off by +5 to +35.
* **W's misses:** +4.5 (control 4,000M) and +5.8 (control 10,000M) points in the near-even band (|logit p_f| < 0.5)
  with R124 as USSR (SE 1.3-1.4): near even positions carry more of the opponent's edge than the shift gives them.
* **6. The control's critic, raw, on the same games:** +10 to +30 points under the result in turns 1-3 against old
  snapshots, and still +10.4 / +8.7 against its own 12,000M checkpoint (R124 scores 54-62% there): it marks down
  R124's side, whose play is now unlike its own, which is the style bias seen from the other side.

**Against the pass rule:** W within 2.5 in every turn-band cell -- no, by 0.5 (3.0, within noise of 42 cells);
within 2.5 in every |logit| cell -- no (5.8, 4.5); beta from the rates within 0.1 of the fitted constant -- no for
L (0.48), and irrelevant to W, which uses the rates with the turn weight; read 5 under twice the floor -- yes (0.72).
So no form passes cleanly. W is close: if P46b is built, it should carry W's position-dependent weight (fitted
per turn here, a critic-only output there) and be read on the near-even band in particular. Whether that is
enough, or the opponent should enter the critic as an input (P44's report, step 2), is the owner's call.

**Does W's turn weight depend on the opponent? (owner, 2026-11-14) -- no, not measurably.** A line
w(t) = w1 + (w10 - w1)(t - 1)/9 fitted per opponent (both seats, all deals, beta from the fit half's rates), standard
errors from a 30-sample game bootstrap (`p46a_w_by_opp.py`; `data/reports/p46a_w_by_opp.txt`): 400M 0.96 +- 0.02 ->
0.46 +- 0.04; 2,000M 0.96 +- 0.02 -> 0.40 +- 0.04; 4,000M 1.03 +- 0.04 -> 0.56 +- 0.07; 6,000M 1.06 +- 0.06 -> 0.51 +-
0.09; 8,000M 0.94 +- 0.06 -> 0.41 +- 0.11; 10,000M 1.16 +- 0.10 -> 0.56 +- 0.15; 12,000M 0.88 +- 0.14 -> 0.12 +- 0.21;
pooled 0.98 -> 0.46. The two best-determined opponents (largest beta) give the same line, and the others sit
within ~2 SE of the pooled one (their small beta leaves w loosely determined). One shared turn profile is enough
for P46b; the 10,000M opponent's high early weight (1.8 SE) coincides with W's largest near-even miss (+5.8), a hint
only.

### Built (2026-11-14, owner: "build it")

* **L6, `--opponent-per-game`** (`OpponentPool(per_game_own=True)`): an own-pool member is drawn at each game's start
  from its own stream (`game_rng`, the pool's draw distribution restricted to the own members) into `env_member`, the
  field P45 uses for league members, so the rollout's grouped opponent pass and the per-game crediting already serve
  it; a member evicted mid-game hands its games to the iteration's opponent (`opp_own_orphaned_games`).
* **P46b, `--critic-prior-window N`** (needs `--opponent-per-game`; excludes `--value-selfplay-only`): the pool keeps
  the learner's running score per (member, seat) and self-play's running US score (exponential averages over ~N
  games; a new member starts at 50%, P46c's table setting) and returns per env
  `prior_shift = +- w(turn) * beta` (+ where the learner decides, - where the member does, 0 in self-play);
  `w(turn)` is linear from `--critic-prior-turn-weight W1 W10` (default P46a's pooled 0.98 -> 0.46). The trainer
  shifts the win value in log-odds (`prior_apply`) in the rollout's stored values (GAE), the bootstrap and the value
  loss, recording each row's shift, so the network fits the self-play value on every row. Logged:
  `opp_prior_m<id>_score_<seat>`, `..._beta_<seat>`, `opp_prior_sp_us`, `opp_per_game_frac`.
* **Checks:** `tests/training/test_p46_prior.py` (12); CPU smoke of the real trainer (64 envs, half pool games, a
  resume): every pool game had a whole-game member, priors tracked per member and seat. The running-average estimate
  of beta over 1,000 games was checked offline on P46a's games: median error 0.07, within 0.1 after ~700 games
  (`p46a_games_needed.py`).

### Built: P46c (2026-11-14)

* **`--critic-prior-gain learned`** (R128 = arm A with `--critic-prior-gain-init zero`, R129 = arm B, init `line`):
  the win value on a row with a prior is `sigmoid(c(s) +- g(s) * beta)`, `c` the win head's logit as before and
  `g(s) = base(turn) + h(s)`, base 0 (A) or the turn line (B). **`h` is a critic head on the shared trunk**
  (LadderNet's `prior_gain_head`: trunk -> 128 -> 1, GELU, last layer zero, so `g` starts exactly at its base and the
  value is unchanged at the branch point; owner, 2026-11-14, over a first build as a separate network on the
  observation). forward() never reads it, so acting and the value are untouched; it is trained with the network by
  the value loss on the rows with a prior, its gradient reaching the trunk as the value's does, in the main AdamW
  (`--critic-prior-gain-lr-mult`, default 1: the main rate). A state written without it resumes into it
  (`ADDABLE_HEADS`, fresh head, every other weight and moment restored); checkpoints carry it and it is read back
  from the weights. The head reads the trunk in a second pass, on the rows with a prior only (the pool share,
  ~30%), in the rollout and in the update. The stored rollout values, the bootstrap and the value loss use the
  same `g` (computed with gradient in the update, on the rows' stored beta and base).
* **The gradient at `g = 0`:** the shift is applied on every row with a prior (beta != 0), not only where the shift
  is nonzero -- otherwise arm A's zero start would never move. Found by the smoke (`g` stuck at 0, no shifted rows).
* **Logged:** `prior_gain_t1_3`, `_t4_7`, `_t8_10`, `_near_even` (mean `g` over the update's prior rows; near even =
  |c| < 0.5), `prior_shift_rows_frac`, `prior_shift_abs_mean`.
* **Checks:** `tests/training/test_p46_prior.py` (16: a new member at 50%, the two factors of the shift, the head
  zero at the start and read from the weights, a branch adding the head plays and values as saved, a learned gain
  refused without the head); CPU smoke from scratch with a resume (the separate-network build: `g` from 0 to
  1.25 / 0.96 / 0.89 in turns 1-3 / 4-7 / 8-10 in 600k steps, half the rows shifted; the head build: `g` from 0 to 1.37 / 1.22 / 1.18 in 600k steps, continuous across the resume).

### B2 (R126, one opponent per game, standard critic): done (2026-11-14)

`E8.1-A8-R119-S45@11230M+R126`, the control branched at 11,230M with `--opponent-per-game`, 1,200M (continued in
place at 11,320M once the per-game members ran as CUDA graphs: 28k -> 42k steps/s; the control ran at 61k). SWA reads,
the control's own read at the same step in brackets:

| step | vs control, same step | vs R124, same step | old reference | J1's 13,600M SWA | US setup (top) | USSR setup |
|---:|:---|:---|:---|:---|:---|:---|
| 11,600M | 51.2 (USSR 55.8 / US 46.6) | 49.0 | 61.3 (58.9) | 50.8 (50.1) | WG 4 Ita 3 Fra 1 Iran 1 66% | Fin 3 Pol 3 100% |
| 12,000M | 50.8 (54.7 / 46.9) | 48.9 | 61.7 (60.6) | 50.4 (50.6) | WG 4 Ita 3 Fra 1 Iran 1 82% | Fin 3 Pol 3 100% |
| 12,400M | 51.8 (55.5 / 48.2) | 48.6 | 60.1 (59.2) | 50.0 (48.9) | WG 4 Ita 3 Fra 1 Iran 1 84% | Fin 3 Pol 3 100% |

Critic at 12,400M: handover error **-9.4 / +10.8** (the control -8.3 / +11.4, R124 +0.2 / +2.0); whole-setup preference
Fin 3 Pol 3 over Pol 4 EG 1 Yug 1 **+9.7 points** (the control +5.6, R124 -0.2).

* **Strength: at most a small gain** -- 1-2 points over the control at every read, each under 2 SE (~0.8), and 1-1.4
  behind R124.
* **The style bias is untouched (larger, if anything):** it comes from the value fitting pool games, however their
  opponents are assigned -- the prior's job (arm A) or R124's.
* **The US setup moved anyway**, to WG 4 Ita 3 Fra 1 Iran 1, the nearest to the human US setup (WG 4 Ita 3 Iran 2) of
  any run; the USSR, which the bias steers, stays on Finland 3 Poland 3.

### P46c arm A (R128, learned gain head from zero): both predictions met (2026-11-15)

`E8.1-A8-R119-S45@11230M+R128`, the control branched at 11,230M with one opponent per game, the prior over ~1,000 games
(every member starting at 50%) and the gain head added at zero, 800M; extended in place to 12,830M to read it at
R124's 12,400 / 12,800M steps.

| step | vs control, same step | vs R124, same step | old reference | J1's 13,600M SWA | handover error (US old / USSR old) | g t1-3 / t4-7 / t8-10 |
|---:|:---|:---|:---|:---|:---|:---|
| 11,600M | 51.8 | 49.7 | 62.9 (control 58.9, R124 61.5) | 49.8 (50.1, 50.8) | **-0.2 / +1.2** | 0.90 / 0.75 / 0.55 |
| 12,000M | 51.8 (USSR 52.9 / US 50.6) | 50.2 | 60.9 (60.6, 60.5) | 52.4 (50.6, 50.5) | **-0.1 / +1.7** | 0.84 / 0.77 / 0.49 |

The control's handover error is -8.3 / +11.4, R124's +0.2 / +2.0, B2's -9.4 / +10.8; P46a's turn line is
0.92 / 0.72 / 0.52.

* **Prediction met:** the handover error is under 3 points by the first read, and g is within 0.15 of the line -- within
  0.08 in every band at both reads. Started at zero, the gain overshot to ~2.5 in the first 5M steps while every
  member's beta was still near 0 (the table starts empty at a branch), and settled onto the line as the table filled.
  **By the rule, the learned gain replaces the fixed line in every later arm.**
* **The critic gives up the shortcut while still training on pool games:** as clean as R124's, and the whole-setup
  preference (Fin 3 Pol 3 over Pol 4 EG 1 Yug 1) is -1.7 points at 12,000M (control +5.6, B2 +9.7, R124 -0.2). A uniform
  offset (the critic +3.8 on the US in self-play positions at 11,600M) fell to +1.0 by 12,000M.
* **Strength:** level with R124, 1-2 points ahead of the control (under 2 SE); best of the arms against J1's 13,600M SWA
  at 12,000M (52.4).
* **Setups:** USSR Pol 4 Aus 1 Fin 1 94% at 11,600M, then **Pol 4 EG 2 83% / Pol 4 EG 1 Yug 1 17%** at 12,000M -- off
  Finland, but most deals now without the Yugoslavia/Austria point; US Fra 3 WG 3 Ita 2 Iran 1.
* Throughput ~37-44k steps/s (one opponent per game; beside another trainer).

### The plan from here (owner, 2026-11-14)

* **J1 openings test, done (2026-11-14):** `E8-A8-R102-S44@12010M+R125` (J1's recipe with the value on self-play rows
  only), from 12,010M where J1 held US Fra 3 Ita 3 Can 2 Iran 1 and USSR Fin 3 Pol 3, 1,600M. A first launch picked up
  the trainer's newer default reward (terminal, no blunder window) instead of J1's and was voided (`..._VOID`).
  **R124's result repeats on a second lineage: the play moves towards human setups and gets stronger.** SWA reads
  (J1's own read at the same step in brackets):

  | step | head to head vs J1, same step | old reference | J1's 13,600M SWA | US setup (top) | USSR setup (top) |
  |---:|:---|:---|:---|:---|:---|
  | 12,400M | **53.0** (USSR 53.1 / US 52.9) | 62.2 (62.5) | 51.2 (47.1) | Fra 3 Ita 3 Can 2 Iran 1 66% | Fin 3 Pol 3 100% |
  | 12,800M | **53.0** (51.7 / 54.3) | 62.7 (60.0) | 52.3 (49.7) | Fra 3 WG 3 Ita 2 Iran 1 75% | Pol 4 Aus 1 Fin 1 74%; Pol 4 Aus 1 EG 1 24% |
  | 13,200M | **54.8** (55.3 / 54.2) | 64.4 (59.7) | 54.6 (47.6) | Fra 3 WG 3 Ita 2 Iran 1 47%; WG 4 Fra 2 Ita 2 Iran 1 38% | **Pol 4 Aus 1 EG 1 91%** |
  | 13,600M | **51.5** (52.5 / 50.4) | 63.4 (63.5) | 51.4 (50.0) | Fra 3 WG 3 Ita 2 Iran 1 82% | Pol 4 Aus 1 EG 1 67%; Pol 4 Aus 2 23% |

  By 12,800M the USSR holds Austria (access to Italy) in every game, by 13,200M 91% play the human setup; the US drops
  Canada for West Germany but keeps France 3 (humans: WG 4 Ita 3 Iran 2). Ahead of J1 at every step (53.0-54.8, SE
  ~0.8) until 13,600M, J1's best step, where it is level (51.5).
* **Order (owner, 2026-11-14, approved with B2 and arm A swapped):** B2 = R126 (one opponent per game, standard
  critic, from the control at 11,230M, 1,200M) when the control reaches 14,000M; P46c arm A = R128 (learned gain
  head from zero, from the control at 11,230M, 800M) when the J1 test ends; arm B = R129 (gain from the turn line,
  from R124's ~13,200M state, 800M) in the next free slot. B3 = R127 (the fixed line on the control's critic) is
  dropped: arm A answers its question with a learned gain. B0 = the control and B1 = R124 (stopped at 13,200M) are
  the references. B4 (one opponent per game + value on self-play rows) is decided later.
* **Next after P46 (owner, 2026-11-14): the pool checks,
  [P47](P47_pool_composition.md)** -- who is in the pool and how often each plays (PFSP, a recent window, the
  oldest member unpinned, no pool), each carrying whatever P46 adopts.
