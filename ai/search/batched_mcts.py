"""MCTS over many positions at once, so the network is called on batches instead of single states.

`pimcts.py` and `dmcts.py` evaluate one leaf per simulation. That is fine for a diagnostic and
hopeless as a training component: measured on this network, a forward pass costs 0.825 ms at batch
1 and 2.19 ms at batch 512, so a batch of 512 costs 2.7x the wall time of a batch of 1 and gives
**193x more states per second**. The trunk is 3.15M parameters -- at batch 1 the GPU is idle and the
cost is launch latency.

The consequence for training is not marginal. A 160M-step arm with search on both sides is about
**137 days** with batch-1 search and about **2 days** batched, measured rather than projected:
476,190 games x 182 decisions (a game is ~336 micro-actions, but the rest are chance nodes the
engine drains itself) at 64 simulations, which is where the strength curve flattens -- 64 scores
75.0% against the raw policy and 96 scores 73.4%, inside noise of each other.

An earlier version of this note said 18 hours. That came from a forward-pass microbenchmark
predicting 193x; the real searcher gets 45.7x once tree bookkeeping, engine clones and Python
overhead are counted.

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


def settle(state: ts.GameState, auto_advance: bool) -> None:
    """Advance past everything the player has no say in, by whichever rule is configured."""
    if auto_advance:
        ts.Engine.auto_advance_step(state)
    else:
        drain_chance_nodes(state)


@dataclass
class BatchedMCTSConfig(PIMCTSConfig):
    #: Resample the hidden state before each search, so the tree never reads the opponent's hand.
    #: False reproduces PIMCTS (a privileged teacher); True reproduces DMCTS (deployable).
    determinize: bool = False
    #: Carry the subtree under the action just played into the next search instead of starting
    #: empty. Measured on this game, the chosen action holds ~37% of root visits, so a
    #: 64-simulation search needs only ~40 new simulations -- a 0.63x cost multiplier with no
    #: approximation, because every decision is still searched to the full budget.
    #:
    #: Ignored when `determinize` is set: the honest searcher resamples the hidden state for each
    #: search, so a tree built under one sampled world says nothing about the next one. Reusing
    #: across determinizations would silently mix worlds.
    reuse_subtree: bool = False
    #: Let the engine skip decisions with no discretion -- chance rolls, single legal actions and
    #: deterministic event targets -- instead of draining chance nodes alone. Measured: 31.2%
    #: fewer decisions per game, forced decisions from 6.1% to 0.0%, and 0.76x the engine time,
    #: since the Python drain loop is replaced by one C++ call. Changes the sequence of positions
    #: the agent is asked about, so it is a flag: a run with it is not comparable to one without.
    auto_advance: bool = True


@dataclass
class _BNode:
    state: ts.GameState
    mover: int
    terminal: bool
    value_us: float = 0.0
    actions: List[int] = field(default_factory=list)
    # Plain lists, not numpy arrays. Branching is ~10, where numpy's dispatch overhead costs more
    # than the arithmetic it performs -- 4.7x more, measured. Selection agrees exactly either way.
    priors: List[float] = field(default_factory=list)
    n: List[float] = field(default_factory=list)
    w: List[float] = field(default_factory=list)
    children: Dict[int, "_BNode"] = field(default_factory=dict)
    #: An expanded node has had its priors filled in from a network evaluation. A node created
    #: during descent starts unexpanded and is completed by the batch it belongs to.
    expanded: bool = False


class BatchedMCTS:
    """Run one MCTS per input position, stepping them together to batch the network calls."""

    def __init__(self, model, device=None, config: Optional[BatchedMCTSConfig] = None,
                 featurise_capacity: int = 0) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.cfg = config or BatchedMCTSConfig()
        #: Optional C++ featuriser. `extract_observation` per leaf is 22.7% of a search; the
        #: runner does the whole batch at once for ~10x less. Sized once; batches beyond its
        #: capacity use the per-node path rather than being silently truncated.
        self._featuriser = (ts.VectorizedBatchRunner(featurise_capacity, int(self.cfg.seed))
                            if featurise_capacity > 0 else None)
        self._featurise_capacity = featurise_capacity
        self._rng = random.Random(self.cfg.seed)
        self._np_rng = np.random.RandomState(self.cfg.seed)
        self.model.eval()
        #: Persistent roots, keyed by whatever the caller uses to identify a position stream
        #: (a game index, typically). Only populated when `reuse_subtree` is on.
        self._trees: Dict[object, _BNode] = {}

    # -- evaluation -----------------------------------------------------------------------

    def _evaluate_batch(self, nodes: Sequence[_BNode]) -> None:
        """Fill in priors and value for a batch of unexpanded, non-terminal nodes."""
        if not nodes:
            return
        obs, masks = self._featurise(nodes)
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
            total = float(pri.sum())
            k = len(legal)
            nd.priors = (pri / total).tolist() if total > 1e-12 else [1.0 / k] * k
            nd.actions = [int(a) for a in legal]
            nd.n = [0.0] * k
            nd.w = [0.0] * k
            # v_win is from the mover's perspective; store it from the US perspective.
            v = float(values[i])
            nd.value_us = v if nd.mover == int(ts.Player.US) else -v
            nd.expanded = True

    def _featurise(self, nodes: Sequence[_BNode]) -> Tuple[np.ndarray, np.ndarray]:
        """Observations and legal masks for a batch of leaves."""
        n = len(nodes)
        if self._featuriser is not None and n <= self._featurise_capacity:
            for i, nd in enumerate(nodes):
                self._featuriser.set_state(i, nd.state)
            # REQUIRED: set_state leaves the cached observation buffer stale, and the stale
            # features match no perspective -- a silent corruption of every leaf evaluation.
            self._featuriser.refresh_all()
            obs = np.asarray(self._featuriser.get_observations(), dtype=np.float32)[:n]
            masks = np.asarray(self._featuriser.get_action_masks())[:n]
            return obs, masks
        obs = np.stack([np.asarray(ts.extract_observation(nd.state, acting_player(nd.state)),
                                   dtype=np.float32) for nd in nodes])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(nd.state)) for nd in nodes])
        return obs, masks

    @staticmethod
    def _make_node(state: ts.GameState) -> _BNode:
        if ts.Engine.is_terminal(state):
            return _BNode(state=state, mover=0, terminal=True, expanded=True,
                          value_us=float(ts.Engine.get_terminal_utility(state)))
        return _BNode(state=state, mover=int(acting_player(state)), terminal=False)

    # -- search ---------------------------------------------------------------------------

    def _select(self, node: _BNode) -> int:
        """PUCT, read from the mover's side of a US-perspective value.

        Written as scalar arithmetic over lists rather than numpy: on ~10 elements the six numpy
        calls cost 4.17us against 0.89us here, and both pick the same index -- verified over 3,000
        random draws per branching level, near-ties included.
        """
        n, w, priors = node.n, node.w, node.priors
        total = 0.0
        for x in n:
            total += x
        sqrt_total = math.sqrt(total if total > 1.0 else 1.0)
        c = self.cfg.c_puct
        value_us = node.value_us
        us_moves = node.mover == int(ts.Player.US)
        best_i, best_v = 0, -1e30
        for i in range(len(n)):
            ni = n[i]
            q = (w[i] / ni) if ni > 0 else value_us
            if not us_moves:
                q = -q
            v = q + c * priors[i] * sqrt_total / (1.0 + ni)
            if v > best_v:
                best_v, best_i = v, i
        return best_i

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
                settle(nxt, self.cfg.auto_advance)
                child = self._make_node(nxt)
                node.children[action] = child
                return path, child
            node = child

    @staticmethod
    def _backup(path: Sequence[Tuple[_BNode, int]], value_us: float) -> None:
        for parent, idx in path:
            parent.n[idx] += 1.0
            parent.w[idx] += value_us

    def run(self, states: Sequence[ts.GameState],
            keys: Optional[Sequence[object]] = None) -> List[Tuple[List[int], np.ndarray]]:
        """Search every position. Returns (actions, visit_counts) per input, in order.

        `keys` identifies each position's stream so its subtree can be carried forward by
        `advance()`. Required only when `reuse_subtree` is set.
        """
        cfg = self.cfg
        reuse = cfg.reuse_subtree and not cfg.determinize and keys is not None
        # A concrete list, so the type checker can see the indexing below is guarded. `reuse`
        # already encodes `keys is not None`, but that narrowing does not survive the variable.
        key_list: List[object] = list(keys) if keys is not None else []
        roots: List[Optional[_BNode]] = []
        inherited: List[bool] = []
        for i, st in enumerate(states):
            carried = self._trees.pop(key_list[i], None) if reuse else None
            if carried is not None and not carried.terminal and carried.expanded:
                roots.append(carried)
                inherited.append(True)
                continue
            s = st.clone()
            settle(s, cfg.auto_advance)
            if cfg.determinize and not ts.Engine.is_terminal(s):
                s = determinize(s, acting_player(s), self._rng)
                s.rng_state = self._rng.getrandbits(64) % _UINT64
            roots.append(self._make_node(s))
            inherited.append(False)

        # One batch for every root, then one batch per simulation round.
        self._evaluate_batch([r for r in roots if r is not None and not r.expanded])

        for r, was_inherited in zip(roots, inherited):
            if r is None or r.terminal or not r.actions:
                continue
            # Root noise belongs to a fresh search. Re-applying it to an inherited tree would
            # perturb priors that its existing visit counts were already collected under.
            if not was_inherited and cfg.dirichlet_frac > 0.0 and len(r.actions) > 1:
                noise = self._np_rng.dirichlet([cfg.dirichlet_alpha] * len(r.actions))
                f = cfg.dirichlet_frac
                r.priors = [(1.0 - f) * pr + f * float(nz) for pr, nz in zip(r.priors, noise)]

        # Each root runs until IT holds `simulations` visits. A single shared count would be
        # decided by the least-inherited tree in the batch -- one fresh tree would force the full
        # budget on every other root, which is how the first version of this saved nothing.
        remaining = [max(0, cfg.simulations - int(sum(r.n)))
                     if (r is not None and not r.terminal and r.actions) else 0
                     for r in roots]

        while any(remaining):
            pending: List[Tuple[List[Tuple[_BNode, int]], _BNode]] = []
            for i, r in enumerate(roots):
                if r is None or r.terminal or not r.actions or remaining[i] <= 0:
                    continue
                remaining[i] -= 1
                pending.append(self._descend(r))
            # Every tree contributed at most one leaf, so a single batch completes the round.
            self._evaluate_batch([leaf for _, leaf in pending
                                  if not leaf.terminal and not leaf.expanded])
            for path, leaf in pending:
                self._backup(path, leaf.value_us)

        out: List[Tuple[List[int], np.ndarray]] = []
        for i, r in enumerate(roots):
            if r is None or r.terminal or not r.actions:
                out.append(([], np.zeros(0)))
            else:
                out.append((r.actions, np.array(r.n, dtype=float)))
            if reuse and r is not None:
                self._trees[key_list[i]] = r
        return out

    def advance(self, key: object, action: int, chance_intervened: bool) -> None:
        """Carry this stream's tree down to the child under `action`, or drop it.

        `chance_intervened` MUST be true when a die roll resolved between the search and the
        resulting position. Each child caches a single sampled die outcome, taken when the child
        was first expanded; if the real roll differed, that subtree describes a position that did
        not occur and reusing it would search the wrong state. Measured on this game a chance node
        follows 4.7% of decisions, so this is a small but not negligible carve-out -- and the
        caller is the only one that can observe it.
        """
        if not self.cfg.reuse_subtree or self.cfg.determinize:
            return
        root = self._trees.pop(key, None)
        if root is None or chance_intervened:
            return
        child = root.children.get(int(action))
        if child is not None and not child.terminal and child.expanded:
            self._trees[key] = child

    def forget(self, key: object) -> None:
        """Drop a stream's tree, e.g. when its game ends."""
        self._trees.pop(key, None)

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
                picks.append(int(actions[max(range(len(visits)),
                                                key=visits.__getitem__)]))
        return picks

    def reset(self) -> None:
        self._rng = random.Random(self.cfg.seed)
        self._np_rng = np.random.RandomState(self.cfg.seed)
