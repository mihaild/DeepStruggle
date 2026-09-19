"""HeuristicMCTSBot: a searching baseline that needs no checkpoint.

`HeuristicBot` plays a fixed rule list with no lookahead. This plays the same game with MCTS over
the real engine, using `ai.search.heuristic_eval` at the leaf, so its strength is attributable to
search and a hand-written evaluation rather than to a network.

**Measured: 18/20 against HeuristicBot at 16 simulations**, balanced across seats. `simulations`
is a dial only up to a point -- 1 sim gives 9/20, 16 gives 18/20, and 96 and 256 give the same
18/20. The strength comes from the one-ply lookahead prior, which the tree then confirms rather
than improves. See `research/findings/training/heuristic_mcts_baseline.md`.

**It searches the true state, so it sees the opponent's hand.** That is deliberate for a
benchmark and disqualifying for deployment; a win rate against this bot is not a claim about play
under the real information set. `ai/search/heuristic_mcts.py` says the same thing at more length.

Only `select_flat_action` is a real implementation. `select_action` -- the dict protocol the web
server speaks -- is not, because the search needs the `GameState` the dict is a projection of, and
reconstructing one from the dict would be a second, lossier copy of the engine's state.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

import ts_engine as ts
from ai.search.heuristic_mcts import HeuristicMCTSConfig, make_heuristic_mcts_agent
from bindings.action_encoder import ActionEncoder
from bot.base_bot import BaseBot


class HeuristicMCTSBot(BaseBot):
    """MCTS over the true state with a rules-derived evaluation."""

    def __init__(self, role: str, name: Optional[str] = None,
                 simulations: int = 64,
                 config: Optional[HeuristicMCTSConfig] = None) -> None:
        super().__init__(role, name or f"HeuristicMCTS{simulations}")
        cfg = config or HeuristicMCTSConfig(simulations=simulations)
        self.cfg = cfg
        self.agent = make_heuristic_mcts_agent(name=self.name, config=cfg)

    #: Tells the match loop to hand over the engine state rather than its JSON view. Same
    #: protocol the `search:<checkpoint>` bot uses in tools/play_match.py.
    wants_game_state = True

    def select_flat_action(self, state: ts.GameState, player: ts.Player) -> int:
        """Search and return a flat action legal **at this node**.

        The chance-node guard is not a nicety. `PIMCTSAgent.search` drains chance nodes on its
        root clone, so if the caller is sitting on an unresolved die roll the search answers
        about the decision *after* it and hands back an action the engine refuses at the current
        one -- the loop then re-offers the same node forever. Callers that settle first (the
        match loop does) never reach this; callers that step raw (tests, the vectorised path) do.
        A die roll is the engine's to settle anyway, never a policy's, so answering it directly
        is the correct move rather than a workaround.
        """
        ctx = state.ctx()
        if (ctx.decision_type == ts.DecisionType.ROLL_DIE
                and ctx.decision_player == ts.Player.NONE):
            return int(ActionEncoder.ROLL_DIE_INDEX)
        return int(self.agent.select_action(state, player, temperature=self.cfg.temperature))

    def select_from_state(self, state: ts.GameState) -> Dict[str, Any]:
        """The match-loop entry point: search the real state, return the action as a dict."""
        player = state.ctx().decision_player
        if player == ts.Player.NONE:
            player = state.phasing_player
        flat = self.select_flat_action(state, player)
        legal = np.asarray(ts.Engine.get_flat_action_mask(state))
        if not legal[flat]:
            # Refused rather than fixed up. An illegal action here means the search and the mask
            # disagree, and the loop would re-offer the same node forever while the bot kept
            # answering it the same way.
            raise RuntimeError(
                f"heuristic MCTS returned flat action {flat}, illegal at decision_type="
                f"{int(state.ctx().decision_type)}")
        ma = ts.decode_flat_action(state, flat)
        return {
            "decision_type": int(ma.decision_type),
            "primary_id": int(ma.primary_id),
            "secondary_id": int(ma.secondary_id),
            "flags": int(ma.flags),
        }

    def select_action(self, state: Dict[str, Any],
                      legal_actions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Not supported: the search needs the engine state, not its serialized projection.

        Raising rather than falling back to a rule list. A bot that silently played something
        other than what its name says would make every tournament row that quotes it wrong, and
        this bot exists to be a reference point.
        """
        raise NotImplementedError(
            "HeuristicMCTSBot searches the engine GameState; use select_flat_action. "
            "The dict protocol cannot carry the state the search needs.")

    def reset(self) -> None:
        self.agent.reset()
