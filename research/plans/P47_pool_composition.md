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
