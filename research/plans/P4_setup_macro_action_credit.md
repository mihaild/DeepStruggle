# P4 — Setup, and macro-action credit for placement blocks

**Status:** queued
**Gate:** P0 setup probe exists. May run before P2 — it is the cheapest change aimed at the most
visible failure.
**Needs approval:** none (`ai/training/rollout_buffer.py` only). No engine change: the block
boundaries are already visible from the decision context.

## Goal

Make the setup sane — Poland ≥ 3 as USSR, West Germany ≥ 4 as US — by fixing the credit path
through the micro-action decomposition, not by telling the network where to place.

## Why

Setup is placed after the first deal (`rules/rules.md` §3; `engine/src/state_machine.cpp`
`init_new_game`), so it is conditioned on the mover's own hand, which the policy sees; the
opponent's hand and later deals are unknown. None of that is the obstacle: the expectation over
own hands is what SGD over ~230k games per 80M steps computes, and the right placement is nearly
hand-independent (humans do it unconditionally). `references.md` §6 has the argument.

What breaks is credit. USSR setup is six consecutive `PLACE_INFLUENCE` micro-actions and the
value of the resulting board is only realised at the last one. Under a critic that prices
control as a step (`experiments.md` §12), the first two influence into Poland have zero measured
advantage and the third gets all of it; the policy never reaches the third. The intermediate
states inside a placement block are not real positions — nobody ever has to play from "USSR has
placed 2 of 6" — and the critic is at its worst there, yet with λ = 0.98 it supplies the targets
for the earlier micro-actions.

The same holds for every multi-step ops play in the game: placing 4 influence one at a time,
choosing coup target then rolling, etc. Ataraxos trains its setup network on Monte Carlo returns
for the same reason: the value net is least reliable at the start.

## Change

`ai/training/rollout_buffer.py`, the GAE trace: inside a *block* — a maximal run of consecutive
micro-actions by the same mover that belong to one card play or to the setup — use λ = 1 (no
bootstrapping between micro-actions); at the block's end, bootstrap once from the critic at the
first real position after it (for setup: the turn-1 headline state; for an ops play: the next
decision node). The critic is still trained at every state, but only real positions supply
targets. Block boundaries come from the decision context already recorded in the buffer
(`DecisionType`, phase, mover) — *decide before running:* the exact boundary predicate, and
whether a `ROLL_DIE` inside a block ends it (it should not; the coup's value is realised after
the roll — and with P2 adopted the roll is averaged anyway).

Setup-specific: the setup block's target is the full λ-return from the turn-1 headline state,
i.e. the same rule, no special case. A pure MC target for setup alone (Ataraxos) is the fallback
if the block rule is not enough, because the critic at turn 1 is itself being learned.

## Procedure

1 arm × 2 seeds × 80M against the current baseline (P1's if adopted, else arm E). Confirm at
240M. Cheap enough to run twice: once as "setup block only", once as "all blocks", if the first
moves the setup probe — that separates the setup effect from the general effect.

## Measure

Setup probe first (the placement histogram, and P(Poland ≥ 3), P(WG ≥ 4) against the human
rate). Then empty battlegrounds at turn 8 (the all-blocks version should help ops plays too),
agreement with human placements (`metrics.md` §9.11), then Elo.

## Decision rule

- Adopt if the setup probe moves toward the human rate by a margin the 2,000-opening sample can
  see, at Elo not worse than −20. It is allowed to be Elo-neutral: a bad setup costs a few
  points of win rate at this level, not tens.
- If the probe does not move at all, the block rule is not the missing piece: the fallback is
  the MC-target-for-setup variant, and if *that* does not move it either, setup is a value
  problem (the turn-1 critic does not see Poland's worth) and P3's search over setup — sample
  opponent hands, roll out to turn 2, evaluate — is the diagnostic to run.

## Follow-ups

- Adopted: the block predicate is also what P3's searcher should use to decide which nodes are
  worth searching, and what an eventual "macro-action" policy factorisation would be built on.
- Whichever way it goes, log the placement histogram: the failure mode ("spreads 1 each",
  "stacks East Germany", "random") says more than the rate.

## Runs

(none yet)
