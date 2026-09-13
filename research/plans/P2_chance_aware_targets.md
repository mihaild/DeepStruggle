# P2 — Chance-aware value targets

**Status:** queued
**Gate:** P0's chance-variance decomposition puts dice + deal at a substantial share of return
variance; P1's target form is settled (the expectation is taken over whatever P1 adopted).
**Needs approval:** **yes** — a batched forced-roll enumeration helper in `bindings/`. It is not
a rules change: `ROLL_DIE` already accepts a forced die in `primary_id`/`secondary_id`
(`engine/src/state_machine.cpp`, `case DecisionType::ROLL_DIE`), the helper only steps clones
with each value. Put the exact signature in front of the owner before writing it.

## Goal

Take the expectation over chance in the value target where the engine makes it exact, instead
of bootstrapping through the one outcome that happened:

1. **Dice.** For a decision whose step lands on a `ROLL_DIE` node, the bootstrap value is the
   average of the critic over all outcomes (6 for one die, 36 for two) stepped on clones, not
   the value after the sampled roll.
2. **Deal.** The λ-return stops at the `TURN_CLEANUP` node (the pre-deal afterstate) and
   bootstraps from the critic there; nothing before the deal is credited with the hand that came
   after it.

## Why

- "GAE falls short" (`references.md` §3): even with a perfect state-value critic, GAE keeps the
  variance of *sampling* the future; the fix is expected backups (Q-boosting / Expected SARSA).
  Our dominant sampled-future noise is the die, and unlike a poker opponent's action, it is
  enumerable and its distribution is known exactly.
- Stochastic MuZero: value on afterstates, chance averaged explicitly. Suphx GRP: a boundary
  value so per-round credit is not swamped by luck across rounds. The engine hands us both
  objects for free — the forced-roll node and the explicit pre-deal node.
- The failures this targets are the dice-adjacent ones: 40–70% of games ending at DEFCON 1
  (`experiments.md` §18.2), coup decisions, and forced-win misses where the critic's optimism
  is partly one lucky roll's worth of bootstrapped value.

## Change

`bindings/ts_bindings.cpp`: `enumerate_roll_outcomes(states, ...)` → for each state whose
context is a pending `ROLL_DIE` of a dice type, the clones after each forced value, plus the
outcome count (so `TURN_CLEANUP` is excluded — it is a deal, handled by 2). Batched over the
512 envs; `GameState` is 4 KB, so this is ≤ 36 × 512 × 4 KB per step.

`ai/training/rollout_buffer.py`: at steps that led to a dice node, replace `next_val` by the mean
critic value over the clones (one extra batched forward pass over ≤ 36 × 512 observations per
such step; dice nodes are a small fraction of steps, so a few percent of rollout time — measure
it). At `TURN_CLEANUP` steps, treat the pre-deal state as a bootstrap point: the trace from
earlier steps ends there with `V(pre-deal)`, and the critic is trained at that state like any
other. The `--slice-turn-boundaries` flag already exists in `ai/training/train.py` — *decide
before running:* whether it already does part 2, and if so, whether it was ever measured.

Sign discipline: every value in the clone average is from the mover's perspective at the
decision step; the mover can change between the decision and the roll. This is the bug class in
`../log/measurement_bugs.md` (*a policy was choosing its own dice*) and `ai/search/pimcts.py`'s docstring — write the test first.

## Procedure

Three arms if the budget allows, two if not: {dice only, deal only, both} vs the P1 baseline,
2 seeds × 80M each. If only one arm: *both*. Confirm the winner at 2 × 240M.

## Measure

Pre-deal calibration (P0 item 4) first; DEFCON-1 ending share and final-turn distribution;
forced-win take rate and its cost-of-declining (`experiments.md` §4.4 table); empty battlegrounds
at turn 8; then Elo.

## Decision rule

- Adopt if DEFCON-1 share drops or pre-deal calibration improves, at Elo not worse than −20.
- Kill if rollout throughput drops by more than ~10% without a probe moving — then the
  expectation is not where the variance was, and P0's decomposition was misread.
- If *deal only* wins and *dice only* is neutral, note it: it says the deal, not the coup, is the
  chance that matters, and P5 (oracle critic) is the next variance reducer on that side.

## Follow-ups

- Adopted: the same batched forward pass over dice clones is the building block for the
  shallow expectimax at play time that TD-Gammon used (`references.md` §3) — queue as a reserve
  item, since the owner prefers a no-search player.
- Adopted: AIVAT-style evaluation baselines (reserve) become straightforward, since the die-node
  control variate is exactly this average.

## Runs

(none yet)
