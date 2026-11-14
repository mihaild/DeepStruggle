# P47 — who is in the opponent pool, and how often each plays (2026-11-14)

**Status:** queued (owner, 2026-11-14): **next after P46**, once its arms have answered whether one opponent per game
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
