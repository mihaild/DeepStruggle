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
| **16** | **18/20 (90%)** | 9/10 | 9/10 |
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

From the US perspective throughout, matching `Engine.get_terminal_utility` and `pimcts`:

* **victory points**, the only term that is not a proxy;
* **region standing**, as the weighted sum of `Scoring.evaluate_region(...).net_delta` — the
  engine's own scoring rule, asked rather than reimplemented. `net_delta` is what a scoring card
  for that region would pay right now, which is the quantity being contested. Europe is weighted
  highest because Europe Control ends the game;
* **DEFCON proximity** as an asymmetric risk: DEFCON 1 loses for whoever *causes* it, so at
  DEFCON 2 the danger belongs to the phasing player;
* **space race**, small.

Deliberately excluded: hand contents. The search sees hands only because it is a
perfect-information opponent, and a term unreadable in the deployable case would make the
evaluation strong in analysis and wrong in play.

Weights are a first cut chosen from the rules, not fitted. `HeuristicWeights` collects them so an
arm can vary them without touching the logic.

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
