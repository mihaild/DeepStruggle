# MCTS on E3-20-28 @200M against the frozen anchors — P15-X0 follow-up

**Measured 2026-09-17.** How much of the gap to the anchors is skill the policy has but does not
express? Same anchors and the same 200 games a side as
[`P15_X0_frozen_anchors.md`](P15_X0_frozen_anchors.md), so the anchor columns are comparable.

The searcher is **honest**: 64 simulations, determinized, node filter `all`. It samples the hidden
state rather than reading it. A privileged searcher sees the opponent's hand and wins ~100%, which
measures nothing about deployable strength.

## The table

Win rate of the **row** player against each opponent, `as USSR / as US`.

| player | vs anchor @80M | vs anchor @160M | vs E3-20-28 @200M (raw) | Elo |
|:---|:---:|:---:|:---:|---:|
| E3-20-28 @200M (raw) | 68.0% / 68.0% | 61.5% / 55.0% | — | 1641.7 |
| **MCTS-64 on @200M** | **83.5% / 85.0%** | **78.0% / 75.0%** | **69.5% / 60.0%** | **1770.9** |

Anchors for scale: `@80M` = 1500.0 (the anchor of this field), `@160M` = 1573.4.

## What it says

**Search is worth +129.2 Elo over the same weights**, and beats its own base policy **64.8%** on
average — 69.5% as USSR, 60.0% as US. This is the quantity P15 leans on: the network already
contains strength the greedy policy does not express, so distilling the searcher back into the
policy (X4) has a gradient even where the outcome signal is dead.

**It is smaller than the figure P15 quotes.** `search_cost_and_coverage.md` records ~75% against
the raw policy (76.7% at 384 sims, ~72.5% at 96). At 64 sims this reads 64.8%, which fits the
downward trend in simulation count — but see the caveat below, because those earlier numbers were
taken through a broken configuration and this is the first one taken through a fixed one.

**Search does not fix the side tilt, it inherits it.** Against the raw 200M policy the searcher is
9.5 pp better as USSR than as US (69.5 vs 60.0). Against `@160M` the raw policy is 6.5 pp better as
USSR (61.5 vs 55.0) — the same direction. Searching harder over a network that evaluates one seat
better does not make the seats equal; it amplifies whichever seat the value head reads well. That
is worth knowing before X4 is treated as a remedy for the oscillation rather than for the dead
gradient.

**The raw policy is even against the older anchor and tilted against the nearer one** — 68.0/68.0
against `@80M`, 61.5/55.0 against `@160M`. A weaker opponent flatters both seats equally; a closer
one exposes the difference.

## Caveats

* **200 games a side**, so each cell is ±3.5 pp.
* **The searcher is cheap.** 64 sims, every node. The measured relationship between simulation
  count and advantage is the subject of `search_cost_and_coverage.md`, not of this table.
* **Every prior search number in this project was taken through a broken configuration.**
  `load_agent` did not pin `advance_root=False`, so the searcher settled its own root and could
  return an action for a decision the caller was not at; and a determinized search can propose a
  move that is illegal in the real world, which nothing filtered
  ([`../findings/engine/hidden_information_legality.md`](../findings/engine/hidden_information_legality.md)).
  Before `step` validated, both produced a quietly wrong game rather than an error. The 76.7% /
  72.5% figures predate the fix and **have not been re-measured**; this 64.8% is the first search
  number taken on a build where an illegal search action cannot be played.
