"""MCTS over many positions at once, so the network is called on batches instead of single states.

`pimcts.py` and `dmcts.py` evaluate one leaf per simulation. That is fine for a diagnostic and
hopeless as a training component: measured on this network, a forward pass costs 0.825 ms at batch
1 and 2.19 ms at batch 512, so a batch of 512 costs 2.7x the wall time of a batch of 1 and gives
**193x more states per second**. The trunk is 3.15M parameters -- at batch 1 the GPU is idle and the
cost is launch latency.

The consequence for training is not marginal. A 160M-step arm with search on both sides at 96
simulations per move is about **137 days** with batch-1 search and about **18 hours** batched.

The structure here is *tree parallelism across positions*, not within a position: N independent
searches advance in lockstep, each contributing exactly one leaf per iteration, and all leaves are
evaluated in a single forward pass. That is the shape expert iteration needs -- it already runs
many games at once -- and it avoids virtual loss entirely, since no tree ever has two unresolved
leaves at the same time. Searching one position hard would need virtual loss and is deliberately
not attempted here.

Sign convention follows `pimcts.py`: every value is from the US perspective.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.dmcts import determinize
from ai.search.pimcts import PIMCTSConfig, acting_player, drain_chance_nodes
from bindings.action_encoder import ActionEncoder

_UINT64 = 1 << 64


@dataclass
class BatchedMCTSConfig(PIMCTSConfig):
    #: Resample the hidden state before each search, so the tree never reads the opponent's hand.
    #: False reproduces PIMCTS (a privileged teacher); True reproduces DMCTS (deployable).
    determinize: bool = False


@dataclass
class _BNode:
    state: ts.GameState
    mover: int
    terminal: bool
    value_us: float = 0.0
    actions: List[int] = field(default_factory=list)
    priors: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    children: Dict[int, "_BNode"] = field(default_factory=dict)
    #: An expanded node has had its priors filled in from a network evaluation. A node created
    #: during descent starts unexpanded and is completed by the batch it belongs to.
    expanded: bool = False


class BatchedMCTS:
    """Run one MCTS per input position, stepping them together to batch the network calls."""

    def __init__(self, model, device=None, config: Optional[BatchedMCTSConfig] = None) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.cfg = config or BatchedMCTSConfig()
        self._rng = random.Random(self.cfg.seed)
        self._np_rng = np.random.RandomState(self.cfg.seed)
        self.model.eval()

    # -- evaluation -----------------------------------------------------------------------

    def _evaluate_batch(self, nodes: Sequence[_BNode]) -> None:
        """Fill in priors and value for a batch of unexpanded, non-terminal nodes."""
        if not nodes:
            return
        obs = np.stack([np.asarray(ts.extract_observation(nd.state, acting_player(nd.state)),
                                   dtype=np.float32) for nd in nodes])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(nd.state)) for nd in nodes])
        obs_t = torch.from_numpy(obs).to(self.device)
        mask_t = torch.from_numpy(masks).to(self.device)
        with torch.no_grad():
            logits, v_win, _ = self.model.forward(obs_t, mask_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            values = v_win.squeeze(-1).cpu().numpy()

        for i, nd in enumerate(nodes):
            legal = np.flatnonzero(masks[i])
            if len(legal) == 0:
                nd.terminal = True
                nd.value_us = 0.0
                nd.expanded = True
                continue
            pri = probs[i][legal]
            total = pri.sum()
            nd.priors = pri / total if total > 1e-12 else np.full(len(legal), 1.0 / len(legal))
            nd.actions = [int(a) for a in legal]
            nd.n = np.zeros(len(legal))
            nd.w = np.zeros(len(legal))
            # v_win is from the mover's perspective; store it from the US perspective.
            v = float(values[i])
            nd.value_us = v if nd.mover == int(ts.Player.US) else -v
            nd.expanded = True

    @staticmethod
    def _make_node(state: ts.GameState) -> _BNode:
        if ts.Engine.is_terminal(state):
            return _BNode(state=state, mover=0, terminal=True, expanded=True,
                          value_us=float(ts.Engine.get_terminal_utility(state)))
        return _BNode(state=state, mover=int(acting_player(state)), terminal=False)

    # -- search ---------------------------------------------------------------------------

    def _select(self, node: _BNode) -> int:
        total_n = float(node.n.sum())
        sqrt_total = math.sqrt(max(total_n, 1.0))
        q = np.where(node.n > 0, node.w / np.maximum(node.n, 1.0), node.value_us)
        oriented = q if node.mover == int(ts.Player.US) else -q
        u = self.cfg.c_puct * node.priors * sqrt_total / (1.0 + node.n)
        return int(np.argmax(oriented + u))

    def _descend(self, root: _BNode) -> Tuple[List[Tuple[_BNode, int]], _BNode]:
        """Walk to a leaf. Returns the path taken and the leaf reached (possibly unexpanded)."""
        path: List[Tuple[_BNode, int]] = []
        node = root
        while True:
            if node.terminal or not node.expanded:
                return path, node
            idx = self._select(node)
            path.append((node, idx))
            action = node.actions[idx]
            child = node.children.get(action)
            if child is None:
                nxt = node.state.clone()
                nxt.rng_state = self._rng.getrandbits(64) % _UINT64
                ts.Engine.step_flat(nxt, action)
                drain_chance_nodes(nxt)
                child = self._make_node(nxt)
                node.children[action] = child
                return path, child
            node = child

    @staticmethod
    def _backup(path: Sequence[Tuple[_BNode, int]], value_us: float) -> None:
        for parent, idx in path:
            parent.n[idx] += 1.0
            parent.w[idx] += value_us

    def run(self, states: Sequence[ts.GameState]) -> List[Tuple[List[int], np.ndarray]]:
        """Search every position. Returns (actions, visit_counts) per input, in order."""
        cfg = self.cfg
        roots: List[Optional[_BNode]] = []
        for st in states:
            s = st.clone()
            drain_chance_nodes(s)
            if cfg.determinize and not ts.Engine.is_terminal(s):
                s = determinize(s, acting_player(s), self._rng)
                s.rng_state = self._rng.getrandbits(64) % _UINT64
            roots.append(self._make_node(s))

        # One batch for every root, then one batch per simulation round.
        self._evaluate_batch([r for r in roots if r is not None and not r.expanded])

        for r in roots:
            if r is None or r.terminal or not r.actions:
                continue
            if cfg.dirichlet_frac > 0.0 and len(r.actions) > 1:
                noise = self._np_rng.dirichlet([cfg.dirichlet_alpha] * len(r.actions))
                f = cfg.dirichlet_frac
                r.priors = (1.0 - f) * r.priors + f * noise

        for _ in range(cfg.simulations):
            pending: List[Tuple[List[Tuple[_BNode, int]], _BNode]] = []
            for r in roots:
                if r is None or r.terminal or not r.actions:
                    continue
                pending.append(self._descend(r))
            # Every tree contributed at most one leaf, so a single batch completes the round.
            self._evaluate_batch([leaf for _, leaf in pending
                                  if not leaf.terminal and not leaf.expanded])
            for path, leaf in pending:
                self._backup(path, leaf.value_us)

        out: List[Tuple[List[int], np.ndarray]] = []
        for r in roots:
            if r is None or r.terminal or not r.actions:
                out.append(([], np.zeros(0)))
            else:
                out.append((r.actions, r.n.copy()))
        return out

    def best_actions(self, states: Sequence[ts.GameState]) -> List[int]:
        """Most-visited action per position; falls back to the first legal action."""
        res = self.run(states)
        picks: List[int] = []
        for (actions, visits), st in zip(res, states):
            if not actions:
                mask = np.asarray(ActionEncoder.get_legal_mask(st))
                legal = np.flatnonzero(mask)
                picks.append(int(legal[0]) if len(legal) else 0)
            else:
                picks.append(int(actions[int(np.argmax(visits))]))
        return picks

    def reset(self) -> None:
        self._rng = random.Random(self.cfg.seed)
        self._np_rng = np.random.RandomState(self.cfg.seed)
