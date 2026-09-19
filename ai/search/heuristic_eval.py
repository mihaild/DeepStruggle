"""A hand-written position evaluation, for search that has no network.

`pimcts` leans on the critic for its leaf value, which makes it a diagnostic rather than a
baseline: it cannot exist before a network does, and its strength is the network's. This supplies
the other option -- a value function written from the rules, so a searching opponent can be built
at any time and its strength attributed entirely to the search.

**Sign convention: every value here is from the US perspective**, matching
`Engine.get_terminal_utility` and `pimcts`. Positive favours the US.

## The evaluation, and why each term

The spine is **victory points that are still reachable**, not board presence in the abstract. A
region is only worth contesting to the extent its scoring card can still be played:

* **Region standing weighted by whether its scoring card is live.** `Scoring.evaluate_region`
  gives `net_delta`, which is what that region's scoring card would pay *right now* -- the engine's
  own rule, asked rather than reimplemented. It is then multiplied by:
  - **1.0** when the scoring card is in a hand or the draw deck: it is coming.
  - **0.5** when it is in the discard pile: only a reshuffle brings it back, so the standing is
    worth keeping but not worth paying for.

  An earlier version of this file weighted regions by a fixed table (Europe 1.35, Africa 0.50).
  That is strictly worse: it says Africa matters less than Europe even in a position where Africa
  Scoring is in hand and Europe Scoring has already been played.

* **Controlled battlegrounds, 0.2 each.** Control is what scoring pays for, and a battleground
  held is a battleground that counts in every future scoring of that region.
* **Battlegrounds with access, 0.1 each.** A battleground where influence can still be placed --
  one already touched, or adjacent to something touched -- is a cheap future option. Worth half a
  controlled one because it is potential rather than banked.
* **Held scoring cards, penalised.** Holding one at the end of a turn loses the game outright, so
  this is not a preference but a countdown: the penalty scales with how few action rounds remain.

Deliberately NOT included: general hand quality. The searcher sees hands only because it is a
perfect-information opponent, and a term unreadable in the deployable case would make the
evaluation strong in analysis and wrong in play. The held-scoring-card term is the exception and
earns it by being a rule about losing, not about advantage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

import ts_engine as ts

#: Scoring card -> the region it pays for. Southeast Asia Scoring (38) is deliberately absent: it
#: scores a set of countries rather than a `Region`, so it has no `evaluate_region` to weight.
SCORING_CARD_FOR_REGION: Dict[int, int] = {
    int(ts.Region.ASIA): 1,
    int(ts.Region.EUROPE): 2,
    int(ts.Region.MIDDLE_EAST): 3,
    int(ts.Region.CENTRAL_AMERICA): 37,
    int(ts.Region.AFRICA): 79,
    int(ts.Region.SOUTH_AMERICA): 81,
}

_REGIONS: List[int] = list(SCORING_CARD_FOR_REGION)


@dataclass
class HeuristicWeights:
    """Every coefficient in one place, so an arm can vary them without touching the logic."""

    #: Multiplier on a region's net_delta when its scoring card is still in a hand or the deck.
    region_live: float = 1.0
    #: ...and when it is in the discard pile, reachable only through a reshuffle.
    region_discarded: float = 0.5
    controlled_battleground: float = 0.2
    accessible_battleground: float = 0.1
    #: Penalty for holding a scoring card, multiplied by how far through the turn we are.
    #: Holding one at turn end is an immediate loss, so this rises rather than staying flat.
    held_scoring_card: float = 4.0
    vp: float = 1.0
    #: Divides the summed score into roughly [-1, 1] before the squash, from the range the terms
    #: actually reach rather than guessed: VP alone spans +/-20.
    scale: float = 22.0


DEFAULT_WEIGHTS = HeuristicWeights()

#: Action rounds in a turn, by turn number. Turns 1-3 have 6, later turns have 7 (Deluxe).
def _action_rounds(turn: int) -> int:
    return 6 if turn <= 3 else 7


def _scoring_liveness(state: ts.GameState, region: int, w: HeuristicWeights) -> float:
    """How much a region's standing is worth, given where its scoring card is."""
    card = SCORING_CARD_FOR_REGION.get(region)
    if card is None:
        return w.region_live
    loc = state.get_card_location(card)
    if loc == ts.CardLocation.DISCARD_PILE:
        return w.region_discarded
    if loc == ts.CardLocation.REMOVED_FROM_GAME:
        return 0.0
    # In a hand, in the draw deck, or staged: it is coming.
    return w.region_live


def region_advantage(state: ts.GameState, w: HeuristicWeights = DEFAULT_WEIGHTS) -> float:
    """Summed `net_delta` across regions, each weighted by its scoring card's liveness.

    `net_delta` is `us_score - ussr_score`, so the sum is already US-positive.
    """
    total = 0.0
    for region in _REGIONS:
        summary = ts.Scoring.evaluate_region(state, ts.Region(region))
        total += _scoring_liveness(state, region, w) * float(summary.net_delta)
    return total


def battleground_counts(state: ts.GameState) -> tuple:
    """(net controlled, net accessible) battlegrounds, US minus USSR.

    Accessible means influence is already there or in a neighbour -- the places the next Operation
    can reach. Control is read from the country itself rather than recomputed, so the stability
    rule lives in one place.
    """
    us_ctrl = ussr_ctrl = us_acc = ussr_acc = 0
    for cid in range(84):
        info = ts.MapData.get_country_info(cid)
        if not info["battleground"]:
            continue
        c = state.get_country(cid)
        us_inf, ussr_inf = int(c.us_influence), int(c.ussr_influence)
        stability = int(info["stability"])
        if us_inf - ussr_inf >= stability:
            us_ctrl += 1
        elif ussr_inf - us_inf >= stability:
            ussr_ctrl += 1
        us_here = us_inf > 0
        ussr_here = ussr_inf > 0
        if not (us_here and ussr_here):
            for nb in info["neighbors"]:
                if nb >= 84:
                    continue
                n = state.get_country(nb)
                us_here = us_here or int(n.us_influence) > 0
                ussr_here = ussr_here or int(n.ussr_influence) > 0
        us_acc += 1 if us_here else 0
        ussr_acc += 1 if ussr_here else 0
    return (us_ctrl - ussr_ctrl), (us_acc - ussr_acc)


def held_scoring_penalty(state: ts.GameState, w: HeuristicWeights = DEFAULT_WEIGHTS) -> float:
    """US-perspective penalty for scoring cards still in hand, rising as the turn runs out.

    Holding one when the turn ends is an immediate loss, so the pressure is real and grows. Early
    in a turn it is merely a card you must spend; in the last action round it is the game.
    """
    total_ar = _action_rounds(int(state.turn))
    remaining = max(0, total_ar - int(state.action_round) + 1)
    urgency = 1.0 / max(remaining, 1)
    us_held = ussr_held = 0
    for card in SCORING_CARD_FOR_REGION.values():
        loc = state.get_card_location(card)
        if ts.in_hand_of(loc, ts.Player.US):
            us_held += 1
        elif ts.in_hand_of(loc, ts.Player.USSR):
            ussr_held += 1
    # Southeast Asia Scoring counts too -- it is just as fatal to hold.
    loc = state.get_card_location(38)
    if ts.in_hand_of(loc, ts.Player.US):
        us_held += 1
    elif ts.in_hand_of(loc, ts.Player.USSR):
        ussr_held += 1
    return -w.held_scoring_card * urgency * (us_held - ussr_held)


def evaluate(state: ts.GameState, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> float:
    """Position value in (-1, 1), US perspective. Terminal states return the exact utility."""
    if ts.Engine.is_terminal(state):
        return float(ts.Engine.get_terminal_utility(state))

    ctrl, acc = battleground_counts(state)
    raw = (weights.vp * float(state.victory_points)
           + region_advantage(state, weights)
           + weights.controlled_battleground * ctrl
           + weights.accessible_battleground * acc
           + held_scoring_penalty(state, weights))

    # tanh rather than a clamp: a clamp makes every winning position identical once it saturates,
    # flattening exactly the gradient the search needs to choose between two good moves.
    return float(np.tanh(raw / max(weights.scale, 1e-6)))


def explain(state: ts.GameState, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> Dict[str, float]:
    """The same computation, term by term. For debugging a move the bot would not play."""
    ctrl, acc = battleground_counts(state)
    terms = {
        "vp": weights.vp * float(state.victory_points),
        "region": region_advantage(state, weights),
        "controlled_bg": weights.controlled_battleground * ctrl,
        "accessible_bg": weights.accessible_battleground * acc,
        "held_scoring": held_scoring_penalty(state, weights),
    }
    terms["raw_total"] = sum(terms.values())
    terms["value_us"] = float(np.tanh(terms["raw_total"] / max(weights.scale, 1e-6)))
    return terms
