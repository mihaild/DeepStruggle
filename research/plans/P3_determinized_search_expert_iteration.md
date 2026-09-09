# P3 — Determinized search as a diagnostic, then expert iteration if it pays

**Status:** queued
**Gate:** a P1/P2 winner exists to search over. Running this on the old scalar critic answers a
question about a critic we are about to replace.
**Needs approval:** none for the diagnostic; the expert-iteration arm adds a loss term to
`nash_pg.py` only.

## Goal

Answer one question: **how much strength is in the value function that the policy does not
express?** Then act on the answer:

- gap large → the value function knows things the policy does not; distil the search policy into
  the network (expert iteration) so the *no-search* player gets it;
- gap small → the value function is the bottleneck; more search buys nothing and the next arm is
  a target-quality change, not a search change.

## Why

- The owner wants a player that does not need search at play time. Search is therefore a
  measurement and a teacher here, not the product.
- The game sits in the corner where determinization is legitimate — high disambiguation, high
  leaf correlation (PIMC conditions, Maven; `references.md` §2) — so sampling consistent
  opponent hands from the v2.1 known-card tracking is the right search, and the belief network
  Ataraxos needed for Stratego is not required.
- `ai/search/pimcts.py` exists but is *perfect-information*: it sees the opponent's hand, so it
  measures an upper bound with privileged information. It was run once (`experiments.md` §4.4
  mentions +85% win rate without moving the forced-win rate) and never logged as an entry. The
  determinized version measures what a deployable searcher would actually get.

## Change

`ai/search/`: a determinized rollout searcher next to `pimcts.py`, using the same US-perspective
sign convention:

1. at a `SELECT_CARD` / `SELECT_PLAY_MODE` node (the decisions that matter; skip placement
   micro-actions), sample N opponent hands uniformly over those consistent with the observation's
   `CardLocation` tracking (hand size known; known cards fixed; unknown slots filled from the
   unaccounted-for set);
2. for each legal action a and each sampled hand, roll out D plies with the policy net and
   evaluate the leaf with the critic (P1's categorical head gives `v_win`); Q̂(a) = mean;
3. π′(a) ∝ π(a) · exp(Q̂(a)/τ) — the one-step KL-regularised improvement (`paper_kl_regularized_search.md`,
   `paper_magnetic_mirror_descent.md`), τ chosen so π′ is not a hard argmax.

Cost for N = 8, |A| ≈ 7, D = 24 is ~1,300 forward passes per searched decision — well under a
second on the 4090 in batched form; the owner accepts a few seconds per move.

Expert-iteration arm (only if triggered): add `ce_coef · CE(π_θ(·|s), π′(·|s))` on a subsample of
buffer positions at card/play-mode nodes, search run in the rollout worker on that subsample
only, so data generation stays direct policy sampling (Ataraxos) with a small distillation tax.
*Decide before running:* the subsample rate (1 in 8 searched decisions is the first guess) and
whether the CE term is applied against π′ or against π′ mixed with the raw policy.

## Procedure

Diagnostic: 1,000-game paired-seed tournaments (`metrics.md` §7) of {policy, policy + search
at card/play-mode nodes} against the heuristic bot and against each other; vary N and D once
each to see where the gain saturates. About a day of evaluation.

Arm (if triggered): 1 arm × 2 seeds × 80M against the P1/P2 baseline; confirm at 240M.

## Measure

Search-vs-no-search Elo gap, and on *which probes* the gap appears: forced wins (tactical, within
the horizon) versus empty battlegrounds and setup (beyond it). The `pimcts.py` docstring states
the prediction: if forced wins move and battlegrounds do not, the horizon argument holds and the
critic is what needs fixing. For the arm: the same probes on the no-search network, then Elo.

## Decision rule

- Gap ≥ 40 Elo → run the expert-iteration arm. Adopt if the *no-search* network gains ≥ 20 Elo
  or a probe moves, at matched steps.
- Gap < 20 Elo → do not run the arm; log it as "value function is the bottleneck" and move to
  P4–P6.
- In between → run the arm only after P4 and P5, so it is measured on the best available critic.

## Follow-ups

- If search helps on battlegrounds/setup (beyond-horizon decisions), that contradicts the
  horizon argument and says the *policy head* is the weak part — queue an entropy / KL-to-π_ref
  schedule arm (Ataraxos's damping) in reserve.
- Play-time search stays a reserve option for a stronger product later; the owner's preference
  is a no-search player first.

## Runs

(none yet)
