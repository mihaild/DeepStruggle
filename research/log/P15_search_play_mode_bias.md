# Does the searcher dodge operations because it cannot place influence?

**Measured 2026-09-17.** A specific failure mode worth ruling out before reading anything else
into the X4 results: a searcher that places influence badly should *underrate the entire ops
branch*, because the value it backs up from that branch comes from the placements it chose inside
it. It would then drift toward EVENT — or toward SPACE, the other way to avoid placing — and
distilling it would install a systematically biased play-mode preference. No amount of that builds
a strong player.

This matters because [`findings/training/which_decisions_to_search.md`](../findings/training/which_decisions_to_search.md)
found that distilling placements offline buys nothing measurable, despite `POINT_NODE` carrying
71.8% of the CE signal. "The searcher is bad at placements" is a live explanation for that.

## Method

`p28_200M`, the checkpoint every X4 result is built on. 400 self-play games, searcher answering
**every** decision at 96 simulations, `determinize=True` — dataset
`x4a_targets_allnodes_400.jsonl.gz`, 176,122 targets.

At each `SELECT_PLAY_MODE` node, the raw policy's masked distribution against the searcher's visit
distribution over the same legal set. Flat actions are fixed: 110 EVENT, 111 OPS, 112 SPACE,
113 PASS. Restricted to the **18,097** nodes where *both* EVENT and OPS were legal, since a node
offering only one says nothing about preference.

Tool: `tools/scripts/play_mode_preference.py`.

## Result: the shift is toward ops, not away from it

| | policy | search | shift |
|:---|---:|---:|---:|
| EVENT | 41.26% | 40.18% | **−1.08 pp** |
| **OPS** | 50.98% | **53.13%** | **+2.16 pp** |
| SPACE | 7.77% | 6.69% | −1.08 pp |
| PASS | 0.00% | 0.00% | — |

Share of the EVENT/OPS pair going to EVENT: policy **44.73%**, search **43.12%** — a shift of
**−1.61 pp**, toward ops.

Argmax disagreements, over the same 18,097 decisions:

| | share |
|:---|---:|
| both say EVENT | 39.93% |
| both say OPS | 56.29% |
| policy OPS → search EVENT | 1.44% |
| policy EVENT → search OPS | **2.34%** |

Net flow **−0.91 pp, toward ops**. The asymmetry is small but not noise: about 261 against 424
discordant decisions, so **61.9% of the disagreements favour ops**, z ≈ 6.2 on a McNemar-style
test of the discordant pairs.

`CHOOSE_TIMING_BRANCH`, the same question about ordering, agrees: OPS_FIRST 44.01% → 44.80%
(+0.79 pp).

## Reading

**The hypothesis is refuted, and refuted in the direction that matters.** A searcher pessimistic
about the ops branch would shift *toward* EVENT; this one shifts away from it. It also *reduces*
space-race usage by 1.08 pp, and the space race is the other way to spend a card without placing
influence. On both of the available escape routes from a placement decision, search takes them
less often than the raw policy does.

So "the searcher cannot place, therefore it avoids placing" is not why offline placement
distillation fails to pay. Something else explains that.

**What this does not establish.** That search shifts toward ops says it *prefers* the ops branch;
it does not say the placements it then chooses are good. A searcher could pick ops more often and
still place poorly inside it — the preference and the competence are separate questions, and only
the first is measured here. The direct test of the second is whether a search agent beats its own
raw policy by more when placements are searched than when they are not, which is what the
search-headroom sweep over the X4b snapshots is measuring.

The shifts are also **small** — one to two points on every statistic. The searcher and the policy
largely agree about play mode, which is consistent with the census: `SELECT_PLAY_MODE` averages
2.1 legal actions and carries 20.9% of the CE signal alongside `SELECT_CARD`. There is not much
room to disagree about a decision with two options.

## See also

* [`findings/training/which_decisions_to_search.md`](../findings/training/which_decisions_to_search.md)
* [`P15_X4a_where_the_search_signal_is.md`](P15_X4a_where_the_search_signal_is.md)
* [`P15_X4b_search_during_rl.md`](P15_X4b_search_during_rl.md)
