# A searching baseline that needs no checkpoint

**Measured 2026-09-19** on the post-P17 engine. `HeuristicMCTSBot` is MCTS over the true
`GameState` with a rules-derived leaf value (`ai/search/heuristic_eval.py`) and a one-ply
lookahead prior. It exists because the ladder's only engine-independent reference points were
`HeuristicBot` — a fixed rule list with no lookahead — and `RandomBot`.

Invoke it as `heuristic_mcts` or `heuristic_mcts:<sims>` anywhere a bot spec is taken
(`tools/play_match.py`).

## Strength

20 games per configuration, 10 with the bot as US and 10 as USSR, against `HeuristicBot`.

| simulations | overall | as US | as USSR |
|---:|---:|---:|---:|
| 1 | 9/20 (45%) | 1/10 | 8/10 |
| **16** (first evaluation) | 18/20 (90%) | 9/10 | 9/10 |
| **16** (owner's evaluation) | **20/20 (100%)** | 10/10 | 10/10 |
| 96 | 17/20 (85%) | 8/10 | 9/10 |
| 256 | 18/20 (90%) | 9/10 | 9/10 |

At 16 simulations it is clearly stronger than `HeuristicBot` and roughly symmetric across seats.
A full game costs about 1.6 s at 96 simulations, so it is cheap enough to use as a tournament
opponent.

## `simulations` is a coarse dial, and here is why

The jump from 1 to 16 is the whole effect; 96 and 256 buy nothing. That is not search failing —
it is the **one-ply prior already being right**. Each node ranks its children by stepping every
legal action once and evaluating the result, then softmaxes at `prior_temperature = 0.25`, which
is sharp. PUCT therefore concentrates on the top-prior action within a few simulations, and the
rest of the budget deepens a branch that was already chosen.

The 1-simulation row is the control that makes this readable: with the tree effectively disabled
the bot drops to 45% and becomes wildly asymmetric (1/10 as US). So the tree *is* contributing —
it is just saturated by 16.

**If a finer difficulty dial is wanted, vary `prior_temperature`, not `simulations`.** A flatter
prior would hand more of the decision back to the tree and should re-open the gap between 16 and
256. Not done here; stated so the next person does not repeat the `simulations` sweep.

## What the evaluation measures

From the US perspective throughout, matching `Engine.get_terminal_utility` and `pimcts`.

**The first version weighted regions by a fixed table** — Europe 1.35 down to Africa 0.50 — plus a
DEFCON risk term and a space term. The owner replaced it with something better, and the
measurement agrees: 18/20 to 20/20.

* **victory points**, the only term that is not a proxy;
* **region standing weighted by whether its scoring card is live**: `net_delta` from
  `Scoring.evaluate_region` (the engine's own rule, asked rather than reimplemented) times **1.0**
  when the scoring card is in a hand or the deck and **0.5** when it is in the discard. This is
  the substantive improvement. A fixed table says Africa matters less than Europe even when Africa
  Scoring is in hand and Europe Scoring has already been played; conditioning on liveness says
  what is actually still winnable;
* **controlled battlegrounds, 0.2 each** — what scoring pays for, banked;
* **battlegrounds with access, 0.1 each** — reachable by the next Operation, so potential rather
  than banked, and worth half;
* **held scoring cards, penalised on a countdown** — holding one at turn end loses outright, so
  the penalty rises as the action rounds run out rather than staying flat.

DEFCON risk and the space term are gone. DEFCON moved to the action layer, where it belongs: the
question is never "is DEFCON 2 bad" but "is *this* card the one to space at DEFCON 2".

Deliberately excluded: hand contents. The search sees hands only because it is a
perfect-information opponent, and a term unreadable in the deployable case would make the
evaluation strong in analysis and wrong in play.

Weights are a first cut chosen from the rules, not fitted. `HeuristicWeights` collects them so an
arm can vary them without touching the logic.

## Play rules live in the prior, not the value

A position score cannot see them. Spacing a card removes it without firing its Event, so every
space play looks identical to the evaluation -- which card to space is decided entirely by what
that card was worth, and the board after is the same either way.

So `spaced_own_or_neutral` is applied as a penalty on the prior, reusing
`dominance.space_dominance_outcome` -- the same predicate `ai/eval/blunders.py` measures against.
A second copy would be a second definition, and the one that drifted would be the copy.

It biases, never forbids: the tree can still play a "blunder" if the position after it is good.
These rules are position-independent generalisations and the search is not.

## Two caveats that limit how this may be quoted

**It is a perfect-information opponent.** It searches the true state and therefore sees the
opponent's hand. That is legitimate for a benchmark and disqualifying for deployment: a win rate
*against* this bot is not a claim about play under the real information set. The same caveat
applies to `pimcts` and is the reason that file calls itself a diagnostic.

**20 games per row.** At n=20 the standard error is about 7pp, so 85% and 90% are not
distinguishable; the 45% row is. Read the table as "1 sim is much worse than 16" and not as a
ranking among 16, 96 and 256.

## Implementation note

The search is `PIMCTSAgent`, unchanged. It gained one optional parameter, `leaf_fn`, which
replaces the model's prior-and-value pair; PUCT, backup and the sign convention are shared rather
than copied, so a fix to one is a fix to both. `PIMCTSAgent(model=None)` without a `leaf_fn`
raises rather than failing later.

One bug surfaced while testing and is worth knowing about for any other bot built on this search:
`PIMCTSAgent.search` drains chance nodes on its root clone, so a caller sitting on an unresolved
die roll gets an action legal at the *next* decision and illegal at the current one. The match
loop settles first and never sees it; anything stepping raw does. `HeuristicMCTSBot` answers a
chance node directly instead of searching it, which is correct on its own terms — a die roll is
the engine's to settle, never a policy's.
