"""A searcher must return a move that is legal in the state the caller actually holds.

Two distinct ways it did not, both found on 2026-09-16 when a tournament with a `search:` model
died on "engine refused flat action 104".

**Root settling.** `load_agent` never passed `advance_root`, so it defaulted to True and the
searcher settled its own root. At a node whose mask holds one legal action `auto_advance_step`
consumes it, the root moves to the successor, and the search returns an action legal there and
illegal in the caller's state. The CLIs do not settle.

**Hidden information changing the legal SET.** The Cambridge Five (card 104) lets the USSR place
in a region named on the *US player's hidden scoring cards*. Determinization reshuffles that hand,
so a sampled world where the US holds Asia Scoring makes all of Asia placeable when the real state
does not — measured at 1 decision in ~7,400, widening a POINT_NODE mask from 21 legal actions to
36. Determinization's usual assumption, that every sampled world shares one action set, is false
in this game. The search may therefore propose an illegal move, and the true mask has to dispose.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np
import pytest
import ts_engine as ts

from ai.search.batched_mcts import BatchedMCTSAgent, BatchedMCTSConfig, _legal_here
from bindings.action_encoder import ActionEncoder


def _agent(sims: int = 8) -> BatchedMCTSAgent:
    """A searcher on freshly initialised weights, configured exactly as `load_agent` does.

    Not a checkpoint: `data/` is git-ignored, so no test may assume it holds anything. These
    tests are about which action comes back, which random weights exercise just as well. The
    config is kept in step with tools/lib/player_agent.load_agent by
    `test_load_agent_does_not_settle_the_root`, which reads the real thing.
    """
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    net = create_coldwar_net_v2(device="cpu", graph_layers=0).eval()
    cfg = BatchedMCTSConfig(simulations=sims, temperature=0.0, auto_advance=True,
                            advance_root=False, determinize=True)
    return BatchedMCTSAgent(net, name="search", device="cpu", config=cfg)


def _opening(seed: int = 4242) -> ts.GameState:
    s = ts.GameState()
    ts.Engine.init_game(s, seed)
    return s


def test_load_agent_does_not_settle_the_root() -> None:
    """The CLIs hand over an unsettled state, so the searcher must not advance past it.

    Read from the source rather than by building an agent, because building one needs a
    checkpoint and `data/` is git-ignored.
    """
    import inspect

    from tools.lib import player_agent

    src = inspect.getsource(player_agent.load_agent)
    assert "advance_root=False" in src, (
        "load_agent no longer pins advance_root=False: the searcher will settle its own root and "
        "return actions for a decision the caller is not at")


def test_legal_here_is_the_only_authority() -> None:
    s = _opening()
    mask = np.asarray(ActionEncoder.get_legal_mask(s))
    legal = np.flatnonzero(mask)
    assert _legal_here(s, int(legal[0]))
    illegal = next(i for i in range(len(mask)) if not mask[i])
    assert not _legal_here(s, illegal)
    assert not _legal_here(s, -1)
    assert not _legal_here(s, 10_000)


class _RogueMCTS:
    """Stands in for a search whose sampled world had a different legal set."""

    def __init__(self, real: Any, bogus: int) -> None:
        self._real = real
        self._bogus = bogus
        self.cfg = real.cfg
        self.device = real.device

    def should_search(self, st: ts.GameState) -> bool:
        return True

    def best_actions(self, states: Sequence[ts.GameState]) -> List[int]:
        return [self._bogus] * len(states)


def test_an_action_from_another_world_falls_back_to_the_policy_and_is_counted() -> None:
    """The Cambridge Five case: the search's pick is legal in its world, not in this one."""
    agent = _agent()
    s = _opening()
    mask = np.asarray(ActionEncoder.get_legal_mask(s))
    bogus = next(i for i in range(len(mask)) if not mask[i])

    agent.mcts = _RogueMCTS(agent.mcts, bogus)          # type: ignore[assignment]
    picks = agent.select_actions_batch([s])

    assert _legal_here(s, picks[0]), (
        "an action illegal in the caller's state survived the fallback")
    assert getattr(agent, "world_mismatch_count", 0) == 1, (
        "the substitution must be counted, or a search that is constantly proposing illegal "
        "moves looks identical to one that is not")


def test_a_still_illegal_action_raises_rather_than_being_played() -> None:
    """If the policy fallback cannot fix it the tree is misrooted, which is a fault, not noise."""
    agent = _agent()
    s = _opening()
    mask = np.asarray(ActionEncoder.get_legal_mask(s))
    bogus = next(i for i in range(len(mask)) if not mask[i])

    agent.mcts = _RogueMCTS(agent.mcts, bogus)          # type: ignore[assignment]
    agent._policy_actions = lambda states: [bogus] * len(states)   # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="not legal in the caller's state"):
        agent.select_actions_batch([s])


def test_the_searcher_plays_a_legal_move_at_every_node_of_a_real_game() -> None:
    """End to end, which is how both bugs actually surfaced."""
    agent = _agent()
    states = [_opening(7000 + i) for i in range(2)]
    for _ in range(60):
        live = [s for s in states if not ts.Engine.is_terminal(s)]
        if not live:
            break
        picks = agent.select_actions_batch(live)
        for st, a in zip(live, picks):
            assert _legal_here(st, a), f"illegal action {a} at decision {int(st.ctx().decision_type)}"
            ts.Engine.step_flat(st, int(a))
            while (not ts.Engine.is_terminal(st)
                   and st.ctx().decision_player == ts.Player.NONE
                   and st.ctx().decision_type == ts.DecisionType.ROLL_DIE):
                m = np.asarray(ActionEncoder.get_legal_mask(st))
                lg = np.flatnonzero(m)
                if not lg.size:
                    break
                ts.Engine.step_flat(st, int(lg[0]))
