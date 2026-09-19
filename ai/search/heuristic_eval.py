"""A hand-written position evaluation, for search that has no network.

`pimcts` leans on the critic for its leaf value. That makes it a diagnostic rather than a
baseline: it cannot exist before a network does, and its strength is the network's. This module
supplies the other option -- a value function written from the rules, so a searching opponent can
be built at any time and its strength attributed entirely to the search.

**Sign convention: every value here is from the US perspective**, matching
`Engine.get_terminal_utility` and `pimcts`. Positive favours the US.

## What it measures, and why each term

The win condition is victory points, so VP is the spine and everything else is a proxy for VP the
position has not yet banked:

* **Victory points**, scaled by the 20-VP automatic win. The only term that is not a proxy.
* **Region standing**, from `Scoring.evaluate_region` -- the engine's own scoring rule, asked
  rather than reimplemented. `net_delta` is what a scoring card for that region would pay right
  now, which is exactly the quantity a player is bidding for. Weighted by how often each region
  actually scores and how much it pays: Europe dominates because Europe Control ends the game.
* **DEFCON proximity**, as a risk term rather than a value. At DEFCON 2 the phasing player is one
  coup away from losing outright, so the danger is asymmetric and belongs to whoever is about to
  act.
* **Space race**, small. It is worth VP and its abilities matter, but a track position is a weak
  predictor next to board control.

Deliberately NOT included: card counts and hand quality. The searcher sees hands only in a
perfect-information setting, and a term that is unreadable in the deployable case is worse than
absent -- it would make the evaluation strong in analysis and wrong in play.

The weights are a first cut, chosen from the rules rather than fitted. `tune_weights` exists to
make that explicit and revisable; nothing here claims they are optimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

import ts_engine as ts

#: How much a region's `net_delta` is worth relative to a victory point.
#:
#: Europe is first by a distance: Europe Control is an instant win, so standing there is worth
#: more than the scoring card pays. Asia and the Middle East score often and pay well. The
#: Americas and Africa pay less and are contested later, which is also why they are the
#: battlegrounds a policy learns to neglect (`position_diagnostics` reports exactly that).
REGION_WEIGHT: Dict[int, float] = {
    int(ts.Region.EUROPE): 1.35,
    int(ts.Region.ASIA): 1.00,
    int(ts.Region.MIDDLE_EAST): 0.85,
    int(ts.Region.SOUTH_AMERICA): 0.70,
    int(ts.Region.CENTRAL_AMERICA): 0.55,
    int(ts.Region.AFRICA): 0.50,
}


@dataclass
class HeuristicWeights:
    """Every coefficient in one place, so an arm can vary them without touching the logic."""

    vp: float = 1.0
    region: float = 0.45
    defcon_risk: float = 0.35
    space: float = 0.10
    #: Divides the summed score into roughly [-1, 1] before the squash. Set from the scale the
    #: terms actually reach rather than guessed: VP alone spans +/-20.
    scale: float = 22.0


DEFAULT_WEIGHTS = HeuristicWeights()

_REGIONS: List[int] = list(REGION_WEIGHT)


def region_advantage(state: ts.GameState) -> float:
    """Summed, weighted `net_delta` across the six regions, US perspective.

    `net_delta` is signed the way this module is -- positive favours the US -- because
    `Scoring.evaluate_region` reports `us_score - ussr_score`.
    """
    total = 0.0
    for region in _REGIONS:
        summary = ts.Scoring.evaluate_region(state, ts.Region(region))
        total += REGION_WEIGHT[region] * float(summary.net_delta)
    return total


def defcon_risk(state: ts.GameState) -> float:
    """Who is endangered by the current DEFCON, US perspective.

    Dropping DEFCON to 1 loses the game for **the player who caused it**, so at DEFCON 2 the risk
    sits with whoever is about to act, not with the board. Returns 0 above DEFCON 3, where the
    mechanism cannot be reached in one action.
    """
    defcon = int(state.defcon)
    if defcon > 3:
        return 0.0
    # Steeper at 2 than at 3: at 2 a single battleground coup ends it.
    severity = 1.0 if defcon <= 2 else 0.35
    phasing = state.phasing_player
    if phasing == ts.Player.US:
        return -severity          # the US is the one who can blunder into it
    if phasing == ts.Player.USSR:
        return severity
    return 0.0


def space_advantage(state: ts.GameState) -> float:
    return float(state.us_space_track) - float(state.ussr_space_track)


def evaluate(state: ts.GameState, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> float:
    """Position value in (-1, 1), US perspective. Terminal states return the exact utility."""
    if ts.Engine.is_terminal(state):
        return float(ts.Engine.get_terminal_utility(state))

    raw = (weights.vp * float(state.victory_points)
           + weights.region * region_advantage(state)
           + weights.defcon_risk * defcon_risk(state)
           + weights.space * space_advantage(state))

    # tanh rather than a clamp: a clamp makes every winning position identical once it saturates,
    # which flattens exactly the gradient the search needs to choose between two good moves.
    return float(np.tanh(raw / max(weights.scale, 1e-6)))


def explain(state: ts.GameState, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> Dict[str, float]:
    """The same computation, term by term. For debugging a move the bot would not play."""
    terms = {
        "vp": weights.vp * float(state.victory_points),
        "region": weights.region * region_advantage(state),
        "defcon_risk": weights.defcon_risk * defcon_risk(state),
        "space": weights.space * space_advantage(state),
    }
    terms["raw_total"] = sum(terms.values())
    terms["value_us"] = float(np.tanh(terms["raw_total"] / max(weights.scale, 1e-6)))
    return terms
