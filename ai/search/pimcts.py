"""Perfect-information MCTS over the true game state, using the policy as prior and v_win as leaf.

This is a *diagnostic*, not a training component. It answers one question: how much does search
buy, and is what it buys tactical or positional?

The search runs on the real `GameState`, so it sees the opponent's hand -- it is a teacher with
privileged information, not a deployable policy. The leaf evaluator is the network's own
`v_win`, which is deliberately the cheapest choice: it needs no training, and it makes the
prediction sharp. Search reaches terminal states near the end of a game and backs up exact
values there, so forced wins should improve a lot. Everything beyond the search horizon still
comes from `v_win`, and a game here is ~350 micro-action plies, so a punishment two turns away
sits 60-80 plies down. If empty battlegrounds do not move while forced wins do, the horizon
argument holds and the critic, not the searcher, is what needs fixing.

Sign convention: **every value in this file is from the US perspective**, matching
`Engine.get_terminal_utility`. The network's `v_win` is from the perspective of the player to
move, so it is negated when USSR is to move. Keeping one global convention avoids the
alternating-perspective sign errors that this action space invites, since the mover can change
partway through a turn.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

import ts_engine as ts
from bindings.settle import SettleMode, settle
from bindings.action_encoder import ActionEncoder

_UINT64 = 1 << 64


def drain_chance_nodes(state: ts.GameState) -> None:
    """Resolve pending die rolls. A chance node is the engine's to settle, never a policy's.

    Stays CHANCE, not FORCED, because the name is the contract: callers asking to drain chance
    nodes get chance nodes drained. A caller that wants everything forced settled says so with
    `settle(state, SettleMode.FORCED)`.
    """
    settle(state, SettleMode.CHANCE)


def acting_player(state: ts.GameState) -> ts.Player:
    ctx = state.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player


@dataclass
class PIMCTSConfig:
    simulations: int = 64
    c_puct: float = 1.5
    temperature: float = 0.0        # 0 => pick the most-visited action
    seed: int = 12345
    # Root exploration noise. A low-prior action is otherwise liable never to be expanded:
    # at 64 simulations over ~9.7 actions its PUCT bonus stays under the Q gap, so search
    # cannot back up a terminal value from a branch it never visits. That is the leading
    # explanation for search failing to improve the forced-win rate, and the misses are
    # exactly the low-prior cases. alpha ~ 10/branching, as in AlphaZero.
    dirichlet_alpha: float = 1.0
    dirichlet_frac: float = 0.0


@dataclass
class _Node:
    state: ts.GameState
    mover: int                       # +1 US, -1 USSR
    terminal: bool
    value_us: float                  # leaf estimate, US perspective
    actions: List[int] = field(default_factory=list)
    priors: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))   # summed value, US perspective
    children: Dict[int, "_Node"] = field(default_factory=dict)


class PIMCTSAgent:
    """PlayerAgent-compatible bot that searches the true state before moving."""

    def __init__(self, model, name: str = "PIMCTS", device=None,
                 config: Optional[PIMCTSConfig] = None) -> None:
        self.model = model
        self.name = name
        self.device = device or next(model.parameters()).device
        self.cfg = config or PIMCTSConfig()
        self._rng = random.Random(self.cfg.seed)
        self._np_rng = np.random.RandomState(self.cfg.seed)
        self.model.eval()

    # -- evaluation ---------------------------------------------------------------------

    def _evaluate(self, state: ts.GameState) -> "_Node":
        """Build a node: terminal value if the game is over, else policy prior + v_win."""
        if ts.Engine.is_terminal(state):
            return _Node(state=state, mover=0, terminal=True,
                         value_us=float(ts.Engine.get_terminal_utility(state)))

        mover = int(acting_player(state))
        mask = np.asarray(ActionEncoder.get_legal_mask(state))
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return _Node(state=state, mover=mover, terminal=True, value_us=0.0)

        obs = ts.extract_observation(state, acting_player(state))
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, v_win, _ = self.model.forward(obs_t, mask_t)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            v_mover = float(v_win.squeeze().item())

        priors = probs[legal]
        total = priors.sum()
        priors = priors / total if total > 1e-12 else np.full(len(legal), 1.0 / len(legal))

        # v_win is from the mover's perspective; convert to the US perspective used throughout.
        value_us = v_mover if mover == int(ts.Player.US) else -v_mover

        return _Node(state=state, mover=mover, terminal=False, value_us=value_us,
                     actions=[int(a) for a in legal], priors=priors,
                     n=np.zeros(len(legal)), w=np.zeros(len(legal)))

    # -- search -------------------------------------------------------------------------

    def _select(self, node: _Node) -> int:
        """PUCT, read from the mover's side of a US-perspective value."""
        total_n = float(node.n.sum())
        sqrt_total = math.sqrt(max(total_n, 1.0))
        q = np.where(node.n > 0, node.w / np.maximum(node.n, 1.0), node.value_us)
        # The mover maximises its own outcome: US maximises q, USSR minimises it.
        oriented_q = q if node.mover == int(ts.Player.US) else -q
        u = self.cfg.c_puct * node.priors * sqrt_total / (1.0 + node.n)
        return int(np.argmax(oriented_q + u))

    def _simulate(self, root: _Node) -> None:
        path: List[tuple] = []
        node = root

        while True:
            if node.terminal:
                value_us = node.value_us
                break
            idx = self._select(node)
            path.append((node, idx))
            action = node.actions[idx]

            child = node.children.get(action)
            if child is None:
                nxt = node.state.clone()
                # Re-seed before stepping so repeated visits sample different die outcomes;
                # without this every simulation replays one fixed roll.
                nxt.rng_state = self._rng.getrandbits(64) % _UINT64
                ts.Engine.step_flat(nxt, action)
                drain_chance_nodes(nxt)
                child = self._evaluate(nxt)
                node.children[action] = child
                value_us = child.value_us
                break
            node = child

        for parent, idx in path:
            parent.n[idx] += 1.0
            parent.w[idx] += value_us

    def search(self, state: ts.GameState) -> tuple:
        """Returns (actions, visit_counts) at the root."""
        root_state = state.clone()
        drain_chance_nodes(root_state)
        root = self._evaluate(root_state)
        if root.terminal or not root.actions:
            return [], np.zeros(0)
        if self.cfg.dirichlet_frac > 0.0 and len(root.actions) > 1:
            noise = self._np_rng.dirichlet([self.cfg.dirichlet_alpha] * len(root.actions))
            f = self.cfg.dirichlet_frac
            root.priors = (1.0 - f) * root.priors + f * noise
        for _ in range(self.cfg.simulations):
            self._simulate(root)
        return root.actions, root.n

    # -- PlayerAgent interface ------------------------------------------------------------

    def select_action(self, state: ts.GameState, player: ts.Player,
                      temperature: float = 0.1) -> int:
        actions, visits = self.search(state)
        if not actions:
            mask = np.asarray(ActionEncoder.get_legal_mask(state))
            legal = np.flatnonzero(mask)
            return int(legal[0]) if len(legal) else 0
        temp = self.cfg.temperature if self.cfg.temperature > 0 else temperature
        if temp <= 0.05 or visits.sum() <= 0:
            return int(actions[int(np.argmax(visits))])
        weights = visits ** (1.0 / max(temp, 1e-3))
        weights = weights / weights.sum()
        return int(self._rng.choices(actions, weights=list(weights), k=1)[0])

    def reset(self) -> None:
        self._rng = random.Random(self.cfg.seed)
        self._np_rng = np.random.RandomState(self.cfg.seed)
