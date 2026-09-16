# The legal action *set* can depend on hidden information

**Status:** measured 2026-09-16. A rules property, not a defect — but it invalidates an assumption
every determinized search makes, and it is why `BatchedMCTSAgent` filters its own search result.

## The case

**The Cambridge Five** (card 104, USSR, 2 ops):

> The US reveals all scoring cards in their hand of cards. The USSR player may add 1 USSR
> Influence to a single Region named on one of the revealed scoring cards.

The placement targets are therefore a function of **which scoring cards the US player is holding**
— information the USSR player does not have, and which a determinized searcher *samples*.

## What it does to a search

Determinization reshuffles the opponent's unknown hand and the draw deck, preserving counts. Both
the real world and the sampled world are internally consistent; they simply have **different legal
moves** at this node. Measured over 7,380 decisions of random play:

| | |
|:---|---:|
| decisions where `determinize` changed the legal mask | **1** |
| all of them at | `POINT_NODE` |
| worked example: real legal actions | 21 |
| the same node in a sampled world | **36** |
| newly legal | flat 150–164 = countries 31–45, all of **Asia** |

The sampled US hand held Asia Scoring; the real one did not.

That is one decision in ~7,400 — rare enough to survive a 2,928-decision smoke test unnoticed,
common enough that a 2,400-game tournament hit it and died.

## Why it matters beyond this card

Determinized search (DMCTS, and the `search:` agent behind `tools/tournament.py`) assumes every
sampled world offers **the same action set**, so that visit counts from different worlds can be
summed over a common set of root children. This card breaks that assumption outright: an action
can accumulate visits in the worlds where it exists and then be **illegal in the world actually
being played**.

Before `step` validated against the mask this produced a silently wrong game — the engine refused
the action and the caller carried on or stalled. It now raises, which is how it was found.

## What the code does about it

`BatchedMCTSAgent.select_actions_batch` treats the search as a *proposal* and the caller's own mask
as the authority: a pick that is illegal in the real state is replaced by the agent's greedy policy
over the real mask and counted in `world_mismatch_count`. A pick still illegal after that cannot be
a world mismatch — the policy is masked by the same mask — so it raises as a misrooted tree.

What is **not** done, and would be better: sum visits only over actions legal in the real state,
inside the search, so the substitution is never needed. That requires the root children to be
built from the true mask rather than each world's, and is left until a search arm actually needs
it. `world_mismatch_count` is the instrument that would say whether it does.

## Whether other cards do this

Only Cambridge Five was found in this sweep, and the sweep was random play over 7,380 decisions,
so a rarer instance would not have shown. Any card whose *targets* — rather than its effect — are
named by the contents of a hidden zone belongs on this list. The search for them has not been done
systematically; `rules/cards.json` is the place to do it.
