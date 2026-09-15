"""Determinized MCTS: search that does NOT see the opponent's hand.

`pimcts.py` searches the true `GameState`, so it reads the opponent's hand. That makes it a
teacher with privileged information and a diagnostic only -- it measures how much strength sits in
the value function, but it cannot be deployed and its numbers are an upper bound on what a real
searcher could reach.

This searches an *information set* instead. Before each search it resamples the hidden state:
the cards it cannot see are shuffled and redealt, preserving every count and every card it *has*
seen. It then runs an independent tree on each sample and sums the root visit counts. That is
determinized MCTS (also called Perfect Information Monte Carlo): a standard, honest baseline for
imperfect-information games, and the same family as the approach named in
`research/plans/P3_determinized_search_expert_iteration.md`.

**What the engine makes this possible without any C++ change:** card locations already
distinguish `HAND_US_KNOWN` from `HAND_US_UNKNOWN`, so the engine records which of the opponent's
cards we are entitled to know. The hidden pool is exactly the opponent's `*_UNKNOWN` cards plus
the draw deck; everything else stays where it is.

**Known limitation, stated because it is a real one.** Determinization is not a solution to
imperfect information: averaging over independently-solved perfect-information worlds suffers
*strategy fusion* (it assumes it may act differently in each world, when in truth one policy must
cover them all) and *non-locality*. It cannot represent a belief-dependent plan, so it will not
bluff, will not hedge, and will over-trust plans that only work in the worlds it happened to
sample. It is a strong baseline and a fair yardstick, not a principled solution -- the principled
versions are ISMCTS with shared statistics across worlds, or a belief-conditioned network.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import ts_engine as ts

from ai.search.pimcts import PIMCTSAgent, PIMCTSConfig, acting_player, drain_chance_nodes

#: Card ids are 1..110 inclusive; `get_card_location` raises outside that range.
CARD_IDS = range(1, 111)

_UINT64 = 1 << 64


def hidden_pool(state: ts.GameState, me: ts.Player) -> Tuple[List[int], List[int], int]:
    """Cards `me` cannot see, split into (opponent-hand-unknown, draw-deck, opp_hand_count).

    A card in the opponent's hand that we have *seen* -- `HAND_*_KNOWN`, set when an event or a
    peek reveals it -- is not hidden and must not be reshuffled, or the determinization would
    contradict something the player already observed.
    """
    if int(me) == int(ts.Player.US):
        opp_unknown = ts.CardLocation.HAND_USSR_UNKNOWN
    else:
        opp_unknown = ts.CardLocation.HAND_US_UNKNOWN

    opp_hand: List[int] = []
    deck: List[int] = []
    for cid in CARD_IDS:
        loc = state.get_card_location(cid)
        if loc == opp_unknown:
            opp_hand.append(cid)
        elif loc == ts.CardLocation.DRAW_DECK:
            deck.append(cid)
    return opp_hand, deck, len(opp_hand)


def determinize(state: ts.GameState, me: ts.Player, rng: random.Random) -> ts.GameState:
    """Return a clone with the unseen cards reshuffled between opponent hand and draw deck.

    Counts are preserved exactly, so the sampled world is consistent with everything `me` can
    observe: the opponent holds the same number of cards, the deck holds the same number, and
    every card `me` has seen stays where it was.
    """
    out = state.clone()
    opp_hand, deck, n_hand = hidden_pool(state, me)
    pool = opp_hand + deck
    if not pool:
        return out
    rng.shuffle(pool)

    if int(me) == int(ts.Player.US):
        hand_loc = ts.CardLocation.HAND_USSR_UNKNOWN
    else:
        hand_loc = ts.CardLocation.HAND_US_UNKNOWN

    for cid in pool[:n_hand]:
        out.set_card_location(cid, hand_loc)
    for cid in pool[n_hand:]:
        out.set_card_location(cid, ts.CardLocation.DRAW_DECK)
    return out


@dataclass
class DMCTSConfig(PIMCTSConfig):
    #: Independent sampled worlds per move. The simulation budget is split across them, so this
    #: trades tree depth for belief coverage at a fixed cost -- the trade determinization exists
    #: to make, and worth sweeping rather than assuming.
    determinizations: int = 8


class DeterminizedMCTSAgent(PIMCTSAgent):
    """PIMCTS run over sampled worlds instead of the true one. Deployable: it never cheats."""

    def __init__(self, model, name: str = "DMCTS", device=None,
                 config: Optional[DMCTSConfig] = None) -> None:
        super().__init__(model, name=name, device=device, config=config or DMCTSConfig())

    def search(self, state: ts.GameState) -> Tuple[List[int], np.ndarray]:
        """Sum root visit counts over independently determinized trees."""
        root_state = state.clone()
        drain_chance_nodes(root_state)
        if ts.Engine.is_terminal(root_state):
            return [], np.zeros(0)

        me = acting_player(root_state)
        cfg = self.cfg
        n_worlds = max(1, int(getattr(cfg, "determinizations", 1)))
        per_world = max(1, cfg.simulations // n_worlds)

        totals: Dict[int, float] = {}
        for _ in range(n_worlds):
            world = determinize(root_state, me, self._rng)
            # Re-seed so the worlds differ in their die rolls as well as their deals.
            world.rng_state = self._rng.getrandbits(64) % _UINT64
            root = self._evaluate(world)
            if root.terminal or not root.actions:
                continue
            if cfg.dirichlet_frac > 0.0 and len(root.actions) > 1:
                noise = self._np_rng.dirichlet([cfg.dirichlet_alpha] * len(root.actions))
                f = cfg.dirichlet_frac
                root.priors = (1.0 - f) * root.priors + f * noise
            for _ in range(per_world):
                self._simulate(root)
            for a, n in zip(root.actions, root.n):
                totals[int(a)] = totals.get(int(a), 0.0) + float(n)

        if not totals:
            return [], np.zeros(0)
        actions = sorted(totals)
        return actions, np.array([totals[a] for a in actions], dtype=float)
