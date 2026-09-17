# Flattening card play: merge play-mode with op-mode

**Analysis only, 2026-09-17.** Proposed by the owner; nothing implemented. Engine changes need
approval (invariant 11) and this one is larger than it looks.

## The proposal

Today a card play is a chain of shallow decisions: choose **play mode** (event / ops / space /
pass), then for an opponent event choose a **timing branch** (ops-first / event-first), then for
ops choose an **op mode** (influence / coup / realign), then place.

The proposal collapses that into one choice:

> space / event (self or neutral, or the opponent's if event-first) / placement / coup /
> realignment, with the opponent event resolving afterwards where needed.

## How much shorter, measured

Every decision of 400 self-play games, from the all-nodes search-target set (`node_filter: all`,
`subsample: 1.0`, so nothing is sampled out). **176,122 decisions, 440.3 per game.**

| type | count | share | per game | mean legal actions |
|:---|---:|---:|---:|---:|
| `POINT_NODE` | 69,223 | 39.3% | 173.1 | 27.0 |
| `SELECT_CARD` | 43,399 | 24.6% | 108.5 | 5.5 |
| `SELECT_PLAY_MODE` | 31,814 | 18.1% | 79.5 | **2.1** |
| `SELECT_OP_MODE` | 22,154 | 12.6% | 55.4 | **2.9** |
| `CHOOSE_TIMING_BRANCH` | 8,662 | 4.9% | 21.7 | **2.0** |
| `CHOOSE_BRANCH` | 870 | 0.5% | 2.2 | 2.6 |

Every `SELECT_OP_MODE` is preceded by a `SELECT_PLAY_MODE` that would fold into it, so the saving
is the op-mode count itself:

| merge | decisions removed | per game | shorter by |
|:---|---:|---:|---:|
| play-mode + op-mode | 22,154 | 440.3 → 384.9 | **12.6%** |
| + timing branch | 30,816 | 440.3 → 363.3 | **17.5%** |

## The action space barely moves

The 212-dim layout currently spends **9 slots** on the three: `[110..113]` play mode,
`[114..115]` timing, `[116..118]` op mode. A merged node needs about **5** (space, event,
placement, coup, realign), or ~8 if the event-first and ops-first variants get their own entries.
So the width does not have to grow and may shrink — this is one wider head replacing three narrow
ones, not extra heads.

## Why it is attractive beyond the step count

The three nodes being merged average **2.1, 2.0 and 2.9** legal actions and together consume
**35.6% of all decisions**. They are the decisions with almost nothing to choose between.

The search census says the same thing from another direction
([`../../log/P15_X4a_where_the_search_signal_is.md`](../../log/P15_X4a_where_the_search_signal_is.md)):
`SELECT_PLAY_MODE` shows 96.6% policy/searcher agreement at KL 0.0192 and
`CHOOSE_TIMING_BRANCH` 95.7% at 0.0175. A 64-simulation search spends its budget there to confirm
what the mask already implied. A ~4-way merged node carries strictly more information per decision
than two sequential 2-way ones, and it shortens every credit-assignment path through a card play.

## What it costs, which is the part to weigh

1. **It invalidates every checkpoint and every `(seed, actions)` dataset, and resets the Elo
   ladder.** Both the action indices and the decision stream change. This is the same class of
   cost as changing the observation, and it lands on `p28_200M`, the frozen anchors, and every arm
   run to date.
2. **It is a state-machine restructure**, and card event handlers that assume the two-step flow
   need auditing rather than trusting.
3. **The mask gets harder, in the direction that fails silently.** Whether "ops-coup" is legal is
   currently settled *after* committing to ops. Merged, the mask must know up front whether any
   coup target exists, whether DEFCON permits it, and whether the card's ops suffice. That is
   lookahead the mask does not do today, and an error there produces an illegal action accepted as
   legal — the failure class that has already cost this project time twice.

## Reading

Right on the merits, and 17.5% is a real gain on every arm's throughput and credit assignment. But
it is a ladder-resetting change, so the moment for it is a deliberate break rather than mid-flight
— `E3-31-28` is currently answering the collapse question against a ladder this would void.

If a smaller first step is wanted, `CHOOSE_TIMING_BRANCH` is the cheapest: 2.0 mean legal actions,
4.9% of decisions, and almost pure overhead by both the branching and the agreement measures.
