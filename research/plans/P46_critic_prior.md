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
  games; a new member starts at the members' mean for that seat) and returns per env
  `prior_shift = +- w(turn) * beta` (+ where the learner decides, - where the member does, 0 in self-play);
  `w(turn)` is linear from `--critic-prior-turn-weight W1 W10` (default P46a's pooled 0.98 -> 0.46). The trainer
  shifts the win value in log-odds (`prior_apply`) in the rollout's stored values (GAE), the bootstrap and the value
  loss, recording each row's shift, so the network fits the self-play value on every row. Logged:
  `opp_prior_m<id>_score_<seat>`, `..._beta_<seat>`, `opp_prior_sp_us`, `opp_per_game_frac`.
* **Checks:** `tests/training/test_p46_prior.py` (12); CPU smoke of the real trainer (64 envs, half pool games, a
  resume): every pool game had a whole-game member, priors tracked per member and seat. The running-average estimate
  of beta over 1,000 games was checked offline on P46a's games: median error 0.07, within 0.1 after ~700 games
  (`p46a_games_needed.py`).

### The plan from here (owner, 2026-11-14)

* **J1 openings test, running:** `E8-A8-R102-S44@12010M+R125` (J1's recipe with the value on self-play rows only),
  from 12,010M where J1 held US Fra 3 Ita 3 Can 2 Iran 1 and USSR Fin 3 Pol 3, 1,600M; reads every 400M (setups,
  both references, head to head against J1 at the same step). A first launch picked up the trainer's newer default
  reward (terminal, no blunder window) instead of J1's and was voided (`..._VOID`).
* **Ablation, from the control at 11,230M** where R124 branched, 1,200M each, one slot each as slots free:
  B3 = R127 (one opponent per game + prior) when the control reaches 14,000M; B2 = R126 (one opponent per game,
  standard critic) after the J1 test. B0 = the control and B1 = R124 (stopped at 13,200M) are the references.
  B4 (one opponent per game + value on self-play rows) is decided later.
