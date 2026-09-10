# P0 — Instruments

**Status:** queued
**Gate:** none. Everything after this is read through these, so they come first.
**Needs approval:** none (evaluation code only; no engine, bindings, or trainer changes).

## Goal

Four measurements that the later steps are judged by, taken once on the current best checkpoint
(`dec_turns40`, arm E lineage — `experiments.md` §21–22) so every later arm has a before/after.
Two are new probes, two are existing instruments pointed at new states.

## Why

The goal is "no simple mistakes", and the simple mistakes named so far — setup that ignores
Poland / West Germany, footholds left exposed to Voice of America — are not measured by anything
in `ai/eval/`. Elo does not move for them (`experiments.md` §4.4 shows a behaviour can be worth
two points of win rate and still be the thing a human notices first). And P2 spends an arm on
chance-aware targets; whether that is worth an arm depends on how much of the return variance is
chance in the first place, which nobody has measured.

## Change

New, in `ai/eval/`, wired into `tools/behavioral_test.py` where the shape fits (constructed
position with a known answer) and into a standalone CLI otherwise:

1. **Setup probe.** From `init_new_game` over ~2,000 seeds (so hands vary), run the policy through
   the setup block and record the placement histogram. Report: P(Poland ≥ 3 | USSR),
   P(West Germany ≥ 4 | US), and the full distribution over the 6/7 placements. Yardstick: the
   same statistic from the 266 human games (`ai/eval/human_corpus.py` gives the positions).
   *Decide before running:* whether to condition on "Europe Scoring in hand", which is the one
   hand feature that plausibly changes the answer.
2. **VOA-exposure probe.** Over ~500 self-play games, at every USSR end-of-turn: is Voice of
   America unplayed (in deck, or in an unknown slot of the US hand — the v2.1 observation's
   `CardLocation` tracking says which), and does USSR hold a non-European country at ≤ 2
   influence with no control? Report the rate, and the fraction of those exposures VOA actually
   punished when it was later played. Same statistic on the human corpus.
3. **Chance-variance decomposition.** From ~2,000 saved decision states spread over turns, replay
   to the end N times with the same frozen policy under (a) the same seed, (b) re-rolled dice
   only (forced-roll `ROLL_DIE` micro-actions replaying the original deals), (c) re-dealt hands
   only, (d) both. Variance of the outcome under each condition splits the return variance into
   policy-sampling, dice, and deal shares. This is the number that decides whether P2 runs.
   *Decide before running:* N (16 is probably enough for a share, not for a small difference).
4. **Pre-deal calibration.** `ai/eval/critic_calibration.py` run on `TURN_CLEANUP` states only
   (the pre-deal afterstate) in addition to its usual decision states, so P2's "critic prices the
   board" claim has a curve to move.

Existing instruments to run unchanged on the same checkpoint so the baseline row is complete:
`battleground_value.py` (§12.1 perturbation probe), `position_diagnostics.py` (empty
battlegrounds at turn 8, final turn distribution, DEFCON-1 share), forced-win take rate as a floor.

## Procedure

One checkpoint, CPU or a few GPU-minutes. Commit the probes with a baseline row in
`experiments.md` in the same change.

## Measure

The four numbers above plus the existing row. No Elo.

## Decision rule

- Setup probe well below the human rate (expected): P4 stays in the queue and may move ahead of P2.
- Dice + deal share of return variance ≥ ~40%: P2 runs. Below ~20%: P2 is demoted to reserve and
  its arm goes to P4/P5.
- VOA exposure: baseline only; it is the acceptance number for the belief-head weighting
  decision in P5's follow-ups.

## Follow-ups

- Every later step reports these four numbers. Add them to the standard eval row in `metrics.md`.
- If the human corpus is too small for a stable VOA yardstick (it may be — VOA is one card in
  266 games), say so in the log and use the exposure rate alone.

## Runs

(none yet)
