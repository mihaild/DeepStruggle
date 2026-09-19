# P15-X4a addendum — where the search signal actually is

**Measured 2026-09-17.** The question X4a left queued: it distilled *card/play-mode* nodes and got
+47.7 Elo, but those nodes average 4.1 legal actions and the policy already agreed with the
searcher 93.1% of the time. `POINT_NODE` placements average 17.5 and were never looked at. X0 had
separately rated a searcher over **all** nodes at +129.2 Elo on the same weights, so most of the
edge was unaccounted for.

Answered offline, before spending an RL arm on the wrong filter.

## Setup

`p28_200M` plays every move; the honest searcher answers **every** decision at 96 simulations,
`determinize=True`, `advance_root=False`. 200 games, **85,113 searched positions, 100% of
decisions** — dataset `/workspace/data/datasets/x4a_targets_allnodes.jsonl.gz`, engine verified
fresh against the sources at generation time.

Mean legal actions across all searched decisions: **12.6**, against 4.1 under the card/play-mode
filter.

## The result: agreement is high everywhere; the *signal* is not

| type | n | top-1 agree | KL(search ‖ policy) | mean legal |
|:---|---:|---:|---:|---:|
| **POINT_NODE** | 33,420 | 92.9% | **0.1239** | **26.4** |
| SELECT_CARD | 21,058 | 90.5% | 0.0431 | 5.6 |
| SELECT_PLAY_MODE | 15,362 | 96.6% | 0.0192 | 2.1 |
| SELECT_OP_MODE | 10,682 | 94.8% | 0.0319 | 2.9 |
| CHOOSE_TIMING_BRANCH | 4,179 | 95.7% | 0.0175 | 2.0 |
| CHOOSE_BRANCH | 412 | 97.8% | 0.0189 | 2.6 |

Overall: 93.4% top-1 agreement, 99.4% within the policy's top 3, median rank 0, KL 0.0677.

**Top-1 agreement does not discriminate.** It sits between 90.5% and 97.8% for every decision type,
including the one with 26 legal actions. Reading agreement alone, there is nothing to learn
anywhere — which is what the original X4a diagnostic appeared to say.

**KL does discriminate, by 7×.** It is what the CE term actually gradients on, and it is
concentrated. Weighting each type's KL by its count:

| | share of total CE signal |
|:---|---:|
| **POINT_NODE** | **71.8%** |
| SELECT_CARD + SELECT_PLAY_MODE (what X4a distilled) | 20.9% |
| everything else | 7.3% |

So the policy picks the searcher's top placement almost as often as it picks its top card — but it
distributes the remaining mass across 26 legal placements very differently from the searcher, and
that disagreement is where nearly three quarters of the available gradient lives.

## This reconciles X4a with X0

| | |
|:---|---:|
| searcher over all nodes, X0 | **+129.2 Elo** |
| distilling card/play-mode only, X4a | **+47.7 Elo** (37% of it) |
| share of CE signal in card/play-mode | **20.9%** |

X4a extracted roughly a third of the search edge from the fifth of the signal it looked at. The
two numbers are not equal and there is no reason they should be — Elo is not linear in nats — but
they are the same order, and they point the same way: **the unextracted majority of the edge is at
`POINT_NODE`.**

## The scoping this overturns

P3 scoped search to card and play-mode decisions as "the decisions that matter", and X4a and X4b
inherited it. That is wrong on the merits as well as on this census — where influence goes decides
which country flips, which battleground is contested and what the opponent can do next, and it is
not a detail of executing a card. Written up in
[`findings/training/which_decisions_to_search.md`](../findings/training/which_decisions_to_search.md),
along with the offline test of the fix.

## What this changes

X4b's specification in [`plans/P15_breaking_the_cycle.md`](../plans/P15_breaking_the_cycle.md)
says card/play-mode nodes, following P3's argument that those are "the decisions that matter".
**That argument is wrong twice over.** It is wrong as a claim about where a searcher disagrees
with this policy, which is what the table above measures. It is also wrong on its own terms:
choosing to play a card for operations is not a plan until the operations land somewhere, and
where they land decides which country flips, which battleground is contested, and what the
opponent can do next. `SELECT_PLAY_MODE`, which the filter *does* search, averages 2.1 legal
actions — a node with two options cannot carry an edge.

**X4b is therefore run with `--search-node-filter all`**, a deliberate deviation from the plan
spec, recorded here and in the arm's own `metadata.json`. The cost is throughput — `all` makes
placements eligible, so more decisions pass the filter and more searches run per game at the same
subsample — which is measured rather than assumed before the arm is sized.

## Caveats

* One checkpoint (`p28_200M`), one searcher configuration, 200 games.
* KL share is a statement about the *gradient available*, not about Elo. That distillation
  converts nats into rating at all is X4a step 3's result; that it does so at the same rate at
  `POINT_NODE` is an assumption this measurement does not test.
* `subsample: 1.0` here — every eligible decision is searched, because the point was to census the
  decision types, not to imitate the training-time sampling.
