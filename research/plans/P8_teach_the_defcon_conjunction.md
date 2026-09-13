# P8 — Teach the DEFCON conjunction, rather than avoid the region

**Status:** queued. Written from the windowing result; nothing here has been run.
**Gate:** none. The instruments exist (`../log/P9_architecture.md`) and the flags are implemented.
**Needs approval:** none for the first two arms (trainer and reward only). The third proposes an
observation change and must not be run without asking.

## Goal

Make the network *represent* the one mistake the project goal names first — handing the opponent
a DEFCON-dropping card at DEFCON 2 — instead of learning to avoid the region of play it lives in.

## Why

Three measurements, in order:

- **The mistake is frequent and never learned.** ~10.6 opportunities per game, error rate ~7%,
  flat from 5M to 80M and still present at 480M (`../log/P9_architecture.md`). It is not data scarcity.
- **The critic is blind to it.** Selecting the fatal card moves `v_win` by at most 0.008 and
  committing it to Operations by at most 0.014; two of eight deltas point the wrong way (`research/log/P9_architecture.md`).
- **Windowing fixes the behaviour and not the understanding.** `--window-provoked-defcon` cut
  provoked DEFCON-1 endings 36.2% → 13.7%, against a human 11.7%, while the self-inflicted share
  moved +0.1 — as clean an attribution as this project has produced. It cost **59 Elo**, and the
  critic came out *no more reactive than the control's* on the same stored positions (`research/log/P9_architecture.md`).

That last pair is the finding this step is built on. A blunder window sets the advantage to
`-1 - v_t`, which routes around the value function by construction: it can teach a policy to stay
away without ever teaching the network what the danger is. Combined with a window that
over-reaches **5.6×** — 124 decisions pinned per provoked ending, 22 of them at or after the
fatal choice — the arm learned a blunt avoidance across a whole turn. The Elo is the bill for
that.

## Change

Three arms, in increasing cost, each attributable on its own.

**8a — the DEFCON-risk head, on a label that can see the phenomenon.** `--defcon-coef` with
`--window-provoked-defcon`. The head is implemented and has never been measured; its label rides
on the same `defcon_blunder` flag, so without the window it is trained only on unprovoked
suicides and is blind to two thirds of the cases. Auxiliary supervision is the standard way to
force a trunk to represent something the main objective under-weights, and it is the only lever
on the table that acts on the *critic* rather than around it.

*Smoke-tested:* the loss is wired and non-zero (0.40–0.49 over 14 iterations) and the label is
non-degenerate. Positives are **0.3–1% of steps**, which puts `pos_weight` at or above the clamp
of 100 in `nash_pg._risk_loss` — check whether the clamp binds before reading a null result as
the head not helping.

**8b — narrow the window.** Scope the blunder window to the action round instead of the turn.
The buffer tracks `turns` and not action rounds, so this needs `action_rounds` threaded from
`ts_env` into `RolloutBuffer.add`. Run it *with* 8a rather than instead: 8a addresses why the
policy learned the wrong lesson, 8b addresses how much ordinary play was condemned alongside.

**8c — representation (ask first).** The conjunction spans three encoder branches: the card's
side, the global DEFCON, and board influence in a coupable battleground. Nothing crosses them
before the fusion trunk. This is an observation change and the layout is the owner's call. The
standing objection — do not spend a large vector on a rare mechanism — does not apply here
(10.6 opportunities per game), but the change still invalidates every checkpoint and resets the
ladder, so propose and wait.

## Procedure

2 seeds × 80M per arm, against `p1_scalar_nofilter` at 80M — same control as the windowing arm,
so all four cells sit on one scale. Rate four late snapshots and pool all sixteen pairings.

## Measure

In this order, because the point is the mechanism and not the ladder:

1. **The paired critic trace** (the paired-critic method in `../log/P9_architecture.md`): both checkpoints over the same
   stored positions, reading the delta at the deciding choice. This is the acceptance test — if
   the delta stays under 0.02, the arm did not teach the conjunction whatever else it did.
2. Provoked and self-inflicted DEFCON-1 shares, and `defcon_suicide_with_alternative`.
3. Ending mix and mean ply against the ITS reference.
4. Pooled head-to-head Elo, last.

## Decision rule

- **Adopt** if the critic's delta at the deciding choice exceeds 0.05 *and* the pooled Elo is
  within 20 of the control. Behaviour alone is not enough — windowing already bought that and
  the bill was 59 Elo.
- **Kill 8a** if the critic is unmoved with the clamp verified not to bind: the head cannot be
  the answer, and the next move is 8c.
- **Read 8b alone as a control on width**: if it recovers most of the 59 Elo without moving the
  critic, the Elo cost was over-reach and the behavioural win survives at a narrower scope.

## Runs

(none yet)
