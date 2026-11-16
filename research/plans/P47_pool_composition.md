# P47 — who is in the opponent pool, and how often each plays (2026-11-14)

**Status (2026-11-15):** arm 1 (PFSP, R130) running since 05:24 UTC; `--opponent-pool-eviction {spread, recent, unpinned}` built for the window and unpinned arms (tests `tests/training/test_pool_eviction.py`, CPU smoke). Was: queued (owner, 2026-11-14): **next after P46**, once its arms have answered whether one opponent per game
works (B2, R126) and whether the critic prior works (P46c arm A, R128; arm B, R129). Nothing built or launched.
**Needs approval:** each arm's launch; trainer-side only, no engine, bindings or observation change.

## Why

The pool is an E3 design (runs of 160M steps, where the oldest member was 20M old and a fair sparring partner) run
unchanged at 11-14B steps:

* **12 members, one added every 5M steps** (`pool_every_steps`), evicted by spacing: the oldest and the newest are
  always kept and the interior member with the closest neighbours goes, so the pool spreads over the whole run.
  The control's pool at 11,230M: 475, 1,420, 2,485, 3,270, 4,340, 5,725, 6,410, 7,350, 8,415, 9,480 and 10,860M,
  plus the newest one or two -- about half of it from the first 40% of training.
* **The oldest member is never evicted.** The endpoints are never eviction candidates, so 475M (the oldest left once
  the step-0 seed, which had no snapshot file, was dropped on a resume) plays 1/12 of all pool games for the rest
  of the run.
* **Uniform draw**: about half of all pool games (~15% of all training games at frac 0.3) go to members from the first
  5B steps, which the learner beats ~90% of the time (P46a: R124 scores 92-95% against the control's 400M snapshot).
  Those games are also where the critic learned to mark down play unlike the current self (P44, P46).
* **It is not what long runs do.** AlphaStar keeps every past player but weights the draw towards the opponents the
  agent struggles against (PFSP); OpenAI Five plays 80% against itself and 20% against past versions whose quality
  score falls each time they are beaten; PSRO-style meta-solvers give dominated policies zero weight; simple PPO
  self-play keeps a window of recent snapshots. All of them stop spending games on opponents that no longer
  challenge the agent; ours does not.
* **Never tested at this scale.** Fraction and capacity were never varied (`questions.md`: "not run", all arms 0.3 / 12);
  `--opponent-pfsp` (var weighting, x(1-x)) was a null once, in E3 at 160M (E3-23-28, +11.6 Elo). Whether a pool is
  needed at all at this scale is open too (`questions.md`; E3's strength result was suggestive, p = 0.057 paired).

## Arms (each a branch from the control at 11,230M, like P46's, beside B0 = the control)

| arm | change | build |
|:---|:---|:---|
| PFSP | `--opponent-pfsp` (var: opponents the learner beats 90% of the time get ~1/3 the weight of even ones) | none |
| recent window | keep the newest 12 members (~60M of history) | an eviction-mode flag |
| unpinned | keep the spread, let the oldest member be evicted | an eviction-mode flag |
| no pool (B5) | `--opponent-frac 0` | none; overrides invariant 14, so the owner's explicit go. A clean version also slows pi_ref (it refreshes every 200k steps, ~3 iterations, and is no memory of past play) and needs both one-change arms beside it |

Reads as P46's: both references, head to head against the control at the same step, setups, the handover check,
and the fixed-reference per-seat check every 400M so a seat runaway shows early. Which arms, and in what order, is
the owner's call when P46 lands; whatever P46 adopts (one opponent per game, the prior) is carried into every arm.

## Runs (filled in while running)

### Arm 1, PFSP (R130): stopped at 11,400M -- the draw stayed uniform (2026-11-15)

`E8.1-A8-R119-S45@11230M+R130`, arm A's recipe (R128) + `--opponent-pfsp`, launched 05:24 UTC. After 170M the draw was
uniform in all but name: `opp_pfsp_entropy` 0.999, `opp_pfsp_max_prob` 0.088 against 1/12 = 0.083. **PFSP weights each
member by its lifetime win rate** (`OpponentPool.win_rate`: all wins and games since the member joined, Beta-smoothed),
which lags far behind the learner: the 475M member reads 0.74 lifetime against ~0.92 measured now (P46a), and the rates
span only 0.49-0.74, where x(1-x) is 0.19-0.25 -- nearly flat. The arm was a copy of arm A and was stopped by PID.
**PFSP needs a windowed win rate** (e.g. the critic prior's running score per member, `prior_rate`, over ~1,000 games)
before it can test anything; the same lag sits under `opp_win_rate_*`, which are lifetime figures too.

### Arm 2, recent window (R131): level with the spread pool, no style bias without the prior (2026-11-15)

`E8.1-A8-R119-S45@11230M+R131`, arm A's recipe + `--opponent-pool-eviction recent`, from the control at 11,230M, 800M:
the oldest member goes at each admission (every 5M), so after ~60M the pool is the newest 12 (~55M of history). Alone on
the GPU after 06:45 it ran at ~72k steps/s.

| step | vs arm A (spread pool), same step | vs control | vs R124 | old reference | J1's 13,600M SWA | handover error (US old / USSR old) | win rate vs pool | USSR setup (top) |
|---:|:---|:---|:---|:---|:---|:---|:---|:---|
| 11,600M | 49.5 | 52.4 | 50.4 | 62.8 | 50.8 | -0.3 / +2.9 | 0.50 | Fin 3 Pol 3 100% |
| 12,000M | 49.5 (USSR 48.8 / US 50.2) | 51.7 | 50.6 | 62.5 | 52.0 | +0.2 / +1.0 | 0.50 | Pol 4 Aus 1 EG 1 36%, the rest with Finland 1 |

Whole-setup preference at 12,000M: -0.3 points.

* **Strength: level with arm A and R124** (49.5 / 50.4-50.6) and 1.7-2.4 ahead of the control: dropping the old members
  costs nothing measurable in 800M.
* **The prior has nothing to do:** against a pool of recent selves the learner scores 0.50 (min 0.46, max 0.53), so every
  beta is ~0 and the shift is ~0.02-0.07 (arm A: ~0.45); the learned g wanders (-0.2 / -0.6 / -1.0 at 11,600M, 0.24 /
  0.13 / -0.05 at 12,000M) because nothing pins it. The critic is clean anyway: the style bias came from the old
  members, and a recent window removes its source.
* The USSR setup is moving off Finland 3 Poland 3 by 12,000M (the Austria point in 55%), more slowly than arm A's.

**Extended to 12,830M (2026-11-15; resumed in place at 12,050M after two host restarts):**

| step | vs arm A, same step | vs control | vs R124 | old reference | J1's 13,600M SWA | handover error | USSR setup (top) |
|---:|:---|:---|:---|:---|:---|:---|:---|
| 12,400M | 51.5 | 53.5 | 49.9 | 62.9 | 51.1 | -- | Pol 4 Aus 1 Fin 1 73%; Pol 4 Aus 1 Yug 1 27% |
| 12,800M | 49.7 (USSR 48.2 / US 51.3) | 54.6 | 50.9 | **65.4** | 51.7 | -0.1 / +0.3 | **Pol 4 Aus 1 Yug 1 59%; Pol 4 EG 1 Yug 1 37%** |

Level with arm A and R124 at every read (49.5-51.5), and as far ahead of the control (53.5 / 54.6) as they were; by
12,800M the USSR holds the Yugoslavia/Austria point in every deal and Finland is gone. Whole-setup preference -1.4
points. **The recent window does what the prior does, without the prior** -- the critic is clean because there are no
old opponents for it to learn style from.

### B5, pure self-play (R132): level with every bias-fixed arm at 400M, no runaway (2026-11-15)

`E8.1-A8-R119-S45@11230M+R132` (owner, 2026-11-15, the explicit go invariant 14 needs): the control at 11,230M with
`--opponent-frac 0` -- every game the learner against its current self; pi_ref unchanged (200k), no prior; 400M.

| 11,600M | B5 | (control at 11,600M) |
|:---|:---|:---|
| vs control, same step | **52.2** (USSR 53.4 / US 51.0) | -- |
| vs R124 / vs arm A, same step | **50.3 / 50.5** | -- |
| old reference / J1's 13,600M SWA | 62.0 / 51.0 | 58.9 / 50.1 |
| handover error (US old / USSR old) | -0.7 / +1.6 | -8.3 / +11.4 |
| whole-setup preference (Fin 3 Pol 3 over Pol 4 EG 1 Yug 1) | +0.2 points | +5.6 |
| USSR setup | Pol 4 Fin 2 40%; Pol 4 Aus 1 Fin 1 28%; Fin 3 Pol 3 18%; Pol 3 Fin 2 Aus 1 14% | Fin 3 Pol 3 100% |
| US setup | Fra 3 WG 3 Ita 2 Iran 1 56%; WG 4 Ita 3 Fra 1 Iran 1 14% | Fra 3 Ita 3 Can 2 Iran 1 |

Self-play US win rate by 50M window: 0.452, 0.468, 0.468, 0.465, 0.463, 0.468, 0.473, 0.470 -- flat, no seat running away.

* **At 400M the pool earns nothing:** pure self-play is level with R124 and arm A and as far ahead of the control as
  every arm that removed the critic's style bias -- which it does by having no old opponents to learn it from.
* **No sign of the E3/E4 failure** in 400M; that collapse built over 120-184M on a far weaker setup, but a longer run is
  what would show it.
* The USSR leaves Finland 3 Poland 3 but keeps Finland points; the Yugoslavia/Austria point in 42% of deals, slower than
  R124's or arm A's move to the human setup.

### B6, no pool from scratch (E8.1-A8-R132-S45): no side loss through 400M, ahead of the pooled control (2026-11-15)

The step 1 control's recipe and seed 45 (R119) with `--opponent-frac 0`, from scratch (owner, 2026-11-15: does the
current architecture still lose a seat without a pool?). Earlier: E4-01-01's self-play US share eroded from ~120M to 0.006
at 184M; E3's unpooled seeds' side gaps grew 12.5 -> 33.2 pp over 80 -> 160M. A first launch with the launcher's seed
streams at 44 was stopped within a minute (`..._135414_VOID`).

| step | head to head vs control, same step | old reference (control) | J1's 13,600M SWA (control) |
|---:|:---|:---|:---|
| 200M | **66.1** (USSR 68.3 / US 63.9) | 14.2: US 12.8 / USSR 14.9 (11.3) | 11.2: US 11.9 / USSR 9.9 (5.6) |
| 400M | **56.7** (USSR 56.7 / US 56.6) | 24.5: US 20.6 / USSR 27.3 (18.6) | 16.4: US 15.5 / USSR 16.7 (12.5) |

Self-play US share by 25M window, 0-400M: B6 0.44-0.49 throughout (0.46 0.47 0.49 0.47 0.48 0.45 0.44 0.46 0.45 0.44
0.46 0.46 0.45 0.48 0.46 0.47); the pooled control 0.42-0.56.

* **No side loss through 400M**, more than twice E4-01-01's collapse step; the split is flatter than the pooled
  control's and B6 wins by the same margin in either seat.
* **Ahead of the pooled control at the same step**, by less at 400M than at 200M: the pool looks like an early cost in
  sample efficiency (30% of games against older selves), not a cap.
* What changed since E3/E4 that plausibly prevents the old arrest (not attributed): a much tighter, less noisy update
  (minibatch 4,096 -> 16,384, PPO clip 0.2 -> 0.1, muP) where loosening the update was what brought collapses on; the
  zero-avoiding term (KL(uniform || pi) over legal moves, 0.01), which keeps the trailing seat exploring; the A8
  architecture; and four engine epochs of rules fixes (the collapse always hit the US). One seed only.
* Healthy by the owner's rule (the side split inside [0.30, 0.70] in every window to 410M), so continued in place to
  810M.

**To 810M (2026-11-15):**

| step | head to head vs control, same step | old reference: B6 (control, J1) | J1's 13,600M SWA: B6 (control, J1) |
|---:|:---|:---|:---|
| 600M | **60.7** (USSR 59.3 / US 62.1) | 30.1: US 28.2 / USSR 31.4 (23.2, --) | 22.3: US 23.1 / USSR 20.2 (17.2, --) |
| 800M | **61.3** (USSR 56.6 / US 65.9) | **34.9**: US 33.8 / USSR 34.8 (26.6, 30.4) | **27.5**: US 27.9 / USSR 25.6 (21.6, 21.2) |

Self-play US share by 25M window, 400-800M: B6 0.46-0.49 (rising slightly to 0.49 by 800M); the control 0.44-0.47.
Setups at 800M: US with West Germany 4 in ~75% (WG 4 Fra 2 Ita 2 Iran 1 33%, WG 4 Fra 3 Ita 2 33%, WG 4 Ita 3 Fra 1
Iran 1 9%); USSR Pol 5 Yug 1 41%, **Pol 4 EG 1 Yug 1 32%**, Pol 3 Yug 3 20% -- the Yugoslavia point in ~96%.

* **Healthy to 800M**, both seats level against both references; the split never left 0.44-0.49.
* **Ahead of the pooled control at every read** (66.1 / 56.7 / 60.7 / 61.3) and **ahead of J1 itself at 800M** on both
  yardsticks (34.9 vs 30.4, 27.5 vs 21.2) -- the leaderboard's lineage, with its blunder-window reward.
* Its openings are the nearest to the human setups of any run at this stage: West Germany 4 for the US, the Yugoslavia
  point for the USSR.

**Continued toward saturation (owner, 2026-11-15; resumed in place at 810M, 20,000M budget, ~105k steps/s alone).**
SWA reads, head to head at the same step and the two references (J1's and the control's own reads in brackets):

| step | vs control | vs J1 | old reference: B6 (J1 / control) | J1's 13,600M SWA: B6 (J1 / control) | self-play US share |
|---:|:---|:---|:---|:---|:---|
| 1,200M | 59.3 | 58.3 | 42.5 (33.5 / 28.8) | 32.1 (23.7 / 21.8) | 0.46-0.48 |
| 1,600M | 59.0 | 54.2 | 36.4 (35.5 / 28.0) | 28.7 (23.6 / 21.9) | 0.47-0.51 |
| 2,000M | 60.3 | 57.8 | 45.3 (38.6 / 35.0) | 35.7 (26.6 / 25.2) | 0.46-0.48 |
| 2,400M | 62.6 | 58.1 | 49.8 (41.0 / 38.1) | 40.6 (30.7 / 27.2) | 0.46-0.47 |
| 2,800M | 62.9 | 59.5 | 51.8 (43.4 / 39.9) | 41.0 (30.0 / 30.1) | 0.45-0.47 |
| 3,200M | 59.2 | 57.9 | 50.2 (45.0 / 39.8) | 41.8 (32.9 / 32.2) | 0.45-0.47 |
| 3,600M | 58.3 | 58.0 | 53.3 (47.4 / 42.8) | 43.0 (34.7 / 35.8) | 0.46-0.47 |

Both seats win in every pairing; no side loss. Setups: US West Germany 4 in ~97% from 1,200M (Iran in 40-55%), USSR the
Yugoslavia point in every deal (Pol 3 Yug 3 54-77% at 1,600-2,000M, East Germany rare).

**Behaviour at 1,600M against the control and J1 at the same step** (16,384 greedy self-play games each): the US wins
51.5% (control 45.8, J1 48.9); 20 VP 55.4% of games (49.7, 47.9); final scoring 20.1% (23.5, 24.3); USSR Europe Control
wins 197 (517, 519). US event shares: Nuclear Subs 34% (31, 18), Chernobyl 3% (2, 5), Our Man in Tehran 2% (1, 2). At the
late checkpoints of the control, J1 and every P46/P47 arm (8,192 games each) the same three read 32-46%, 3-9% and 1-4%:
no critic or pool change moves them.
