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
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch

import ts_engine as ts
from bindings.settle import SettleMode
from bindings.settle import settle as _settle
from ai.search.dmcts import determinize
from ai.search.gumbel_root import GumbelRoot, GumbelStats
from ai.search.pimcts import PIMCTSConfig, acting_player, drain_chance_nodes
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import model_obs_features

_UINT64 = 1 << 64
_US = int(ts.Player.US)
#: Smallest featuriser the batched path builds. Batches are sized up to the next power of two from
#: here, so `refresh_all` -- which rebuilds every slot, filled or not -- does at most 2x the work.
_MIN_FEATURISE_BUCKET = 64
#: Trees per evaluation group: a batch of at least twice this many is split in two, and each half's
#: leaves go to the network on their own, so the C++ tree can work on one half while the GPU
#: evaluates the other. Both trees group the same way, because it changes the numbers, not only
#: the speed: the network's output for a row depends on the batch it is evaluated in (measured on
#: the GPU, no row of a split batch matched the whole batch bit for bit).
#:
#: 512, not less: below that a half's forward pass is mostly fixed cost, and splitting only doubles
#: the calls -- measured, 200 positions x 256 simulations took 0.79 s in halves of 100 against 0.27 s
#: as one batch, while 3,200 x 32 (chunks of 1,024, halves of 512) went from 0.61 s to 0.30 s.
_MIN_GROUP = 512


def search_groups(n: int) -> List[Tuple[int, int]]:
    """[begin, end) of each evaluation group of `n` trees: two halves from 2 x _MIN_GROUP on, else one."""
    if n <= 0:
        return []
    g = 2 if n >= 2 * _MIN_GROUP else 1
    per = -(-n // g)
    return [(b, min(n, b + per)) for b in range(0, n, per)]


def _legal_here(state: ts.GameState, action: int) -> bool:
    """Is `action` legal in this exact state? The only authority on what may be played."""
    mask = np.asarray(ActionEncoder.get_legal_mask(state))
    return 0 <= action < len(mask) and bool(mask[action])


def settle(state: ts.GameState, auto_advance: bool) -> None:
    """Advance past everything the player has no say in, by whichever rule is configured.

    The boolean is kept because `BatchedMCTSConfig.auto_advance` is a boolean and a run recorded
    under one value is not comparable with a run under the other; the depths themselves now come
    from bindings/settle.py, so this cannot drift from what the rest of the stack does.
    """
    _settle(state, SettleMode.FORCED if auto_advance else SettleMode.CHANCE)


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
    #: Settle the ROOT before searching. False searches exactly the state handed over, which is
    #: the only setting that cannot be wrong: when the caller has already settled, settling again
    #: is a no-op, and when it has not, True roots the tree at a later decision than the caller
    #: holds and returns an action illegal there.
    #:
    #: It defaulted to True and every one of the five production searchers overrode it -- a
    #: default that every caller disables is not a default. play_match.py's comment records what
    #: it cost before anyone noticed: the engine refusing an action "1,355 times in one game".
    #: Children are settled regardless; they are the searcher's own hypotheticals, not the
    #: caller's position.
    advance_root: bool = False
    #: Which decisions to search. "all" searches every node handed over -- the setting the ~+27pp
    #: measurement used. "card_playmode" searches only SELECT_CARD and SELECT_PLAY_MODE, which
    #: P3 argues are "the decisions that matter"; measured over 8 self-play games they are 42.0%
    #: of all decisions, against 39.5% POINT_NODE placements.
    node_filter: str = "all"
    #: Fraction of the decisions passing `node_filter` that are actually searched, chosen per
    #: decision from the searcher's own RNG. P3's first guess is 1 in 8.
    subsample: float = 1.0
    #: Where the trees live. "cpp" is `ts_engine.BatchedSearch` (bindings/batched_search.hpp):
    #: selection, expansion, settling and featurisation in C++, the network here, two crossings of
    #: the binding per simulation round. It draws its chance seeds from this object's own
    #: random.Random, in the Python tree's order, so the two are bit-identical: same visit counts,
    #: same generator state afterwards, same games. "python" is the tree in this file, kept as the
    #: reference the C++ one is tested against, and still the only one that carries a tree
    #: between calls (`reuse_subtree`, which falls back to it).
    backend: str = "cpp"
    #: First-play urgency: an unvisited move is valued at its node's value less this, from the
    #: mover's side. 0, the default, is the original rule (the node's value). Both trees.
    fpu_reduction: float = 0.0
    #: How the side not searching plays inside the tree. "search" (the default): both sides are
    #: searched, PUCT at every node. "greedy": a best-response search -- at every node where the
    #: other side moves it takes that side's most probable move by the network's prior, so the
    #: searcher plans against the greedy policy rather than against a searching opponent (owner,
    #: 2026-10-09: can search find what the greedy network walks into?). Python tree only.
    opponent: str = "search"
    #: Choose the move by a Gumbel root (ai/search/gumbel_root.py): k candidates by Gumbel-top-k,
    #: the budget split by sequential halving. `simulations` is then a budget of network
    #: evaluations, which PUCT at the same `simulations` never undercuts. 0, the default, is the
    #: most-visited PUCT move.
    gumbel_k: int = 0
    #: Scale of the Gumbel noise; 0 takes the k most probable moves, deterministically.
    gumbel_scale: float = 1.0
    #: The Gumbel root's logits are divided by this before candidates are taken and ranked. 1, the
    #: default, is the paper's rule. Above 1 the network's preferences count for less against the
    #: searched values: with logit gaps of 15-30 nats, sigma(completed Q) -- at most ~15 at 256
    #: evaluations -- cannot overturn the prior, so a move the network has written off is never
    #: chosen however it searches (research/log/E7_cuban_missile_crisis_probe.md).
    gumbel_prior_temperature: float = 1.0
    #: Independent worlds a Gumbel root searches each candidate in, per halving phase. A world is a
    #: fresh draw of the hidden cards and of the dice: the search resolves a move's dice once,
    #: when it first expands it, and every later simulation through that move reuses that roll, so
    #: more simulations in one world never average over rolls -- only more worlds do. A phase's
    #: per-candidate share of evaluations is split evenly over min(worlds, share) worlds, the
    #: candidates of a world sharing its draw. 1, the default, is one world per phase.
    gumbel_worlds: int = 1
    #: Run the network through CUDA graphs (the C++ tree on CUDA only): one graph per batch size,
    #: padded up to a multiple of GRAPH_BUCKET rows. A graph replays the same kernels eager torch
    #: launches, so at one batch size the outputs are bit-identical -- but the padding changes the
    #: batch size, and the value head's output for a row depends on it (up to 1.5e-7 on about half
    #: the rows, measured), so a search with graphs is not bit-identical to one without. Off by
    #: default for that reason; what it buys is the CPU time to issue each forward (0.74 ms eager,
    #: 0.06 ms replayed, at 1,900 rows), on the thread every round waits for.
    cuda_graphs: bool = False


#: CUDA graphs are captured at batch sizes that are multiples of this, and a batch is padded up.
GRAPH_BUCKET = 64


class _GraphedNetwork:
    """The network (and the softmax of its policy) as CUDA graphs, one per padded batch size, all
    in one memory pool. Rows are independent in this network, so the padding rows -- whatever was
    last copied there -- do not touch the real ones; only the batch size does (see
    BatchedMCTSConfig.cuda_graphs). A graph's outputs live in the shared pool: they are copied out
    on the same stream before the next replay, which is what makes sharing it safe."""

    def __init__(self, model, obs_width: int, device: torch.device) -> None:
        self.model = model
        self.obs_width = obs_width
        self.device = device
        self.pool: Optional[Any] = None       # torch's memory-pool handle (a private type)
        #: Batch size -> (graph, obs input, mask input, probs output, values output).
        self.graphs: Dict[int, Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor,
                                     torch.Tensor, torch.Tensor]] = {}

    def _capture(self, n: int) -> Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor,
                                        torch.Tensor, torch.Tensor]:
        obs = torch.zeros((n, self.obs_width), dtype=torch.float32, device=self.device)
        masks = torch.zeros((n, ActionEncoder.FLAT_ACTION_SIZE), dtype=torch.uint8, device=self.device)
        masks[:, 0] = 1                       # padding rows never have an empty mask
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(2):                # warm-up outside the capture, as CUDA graphs require
                self.model.forward(obs, masks)
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool), torch.no_grad():
            logits, v_win, _ = self.model.forward(obs, masks)
            probs = torch.softmax(logits.float(), dim=-1)
            values = v_win.float().reshape(-1)
        if self.pool is None:
            self.pool = graph.pool()
        entry = (graph, obs, masks, probs, values)
        self.graphs[n] = entry
        return entry

    def run(self, obs_src: torch.Tensor, masks_src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Probabilities and values for the k rows of `obs_src` / `masks_src` (page-locked host
        tensors), as device tensors valid until the next replay on this stream."""
        k = int(obs_src.shape[0])
        n = -(-k // GRAPH_BUCKET) * GRAPH_BUCKET
        graph, obs, masks, probs, values = self.graphs.get(n) or self._capture(n)
        obs[:k].copy_(obs_src, non_blocking=True)
        masks[:k].copy_(masks_src, non_blocking=True)
        graph.replay()
        return probs[:k], values[:k]


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
    #: sum(n), kept by `_backup` so that `_select` does not re-add every child's count on every
    #: visit (7.6% of an eval search, profiled). Exact: the counts are small integers in floats.
    total: float = 0.0
    #: An expanded node has had its priors filled in from a network evaluation. A node created
    #: during descent starts unexpanded and is completed by the batch it belongs to.
    expanded: bool = False
    #: The side the search is for (the root's mover unless the caller says otherwise); read only by
    #: `opponent="greedy"`, which plays the network's top move wherever the other side moves.
    searcher: int = 0


class _SearchBuffers:
    """The leaf and result buffers of one C++ searcher: page-locked on CUDA, so the copies to and
    from the GPU are asynchronous. Rows are the searcher's leaf rows."""

    def __init__(self, capacity: int, obs_width: int, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        pin = self.cuda
        self.obs = torch.zeros((capacity, obs_width), dtype=torch.float32, pin_memory=pin)
        self.masks = torch.zeros((capacity, ActionEncoder.FLAT_ACTION_SIZE), dtype=torch.uint8,
                                 pin_memory=pin)
        self.probs = torch.zeros((capacity, ActionEncoder.FLAT_ACTION_SIZE), dtype=torch.float32,
                                 pin_memory=pin)
        self.values = torch.zeros(capacity, dtype=torch.float32, pin_memory=pin)
        self.probs_np = self.probs.numpy()
        self.values_np = self.values.numpy()


class BatchedMCTS:
    """Run one MCTS per input position, stepping them together to batch the network calls."""

    def __init__(self, model, device=None, config: Optional[BatchedMCTSConfig] = None,
                 featurise_capacity: int = 0) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.cfg = config or BatchedMCTSConfig()
        #: Optional C++ featurisers. `extract_observation` per leaf is 22.7% of a search; a runner
        #: does the whole batch at once for ~10x less. One runner per power-of-two size up to
        #: `featurise_capacity`, built on first use: `refresh_all` rebuilds EVERY slot, so a single
        #: 4,096-slot runner spent most of its time on empty slots when a batch held ~200 leaves
        #: (8.25 ms per call against 2.09 ms at 256; 20% of an eval search). Batches beyond the
        #: capacity use the per-node path rather than being silently truncated.
        self._featurisers: Dict[int, ts.VectorizedBatchRunner] = {}
        self._featurise_capacity = featurise_capacity
        #: The C++ searcher (backend "cpp") with the buffers it writes leaves into and reads
        #: results from: one, sized to the largest batch so far and used for every smaller one.
        #: A searcher per batch size kept every size it had ever met -- each with its trees and
        #: page-locked buffers -- for the life of the process.
        self._cpp: Optional[Tuple[ts.BatchedSearch, _SearchBuffers]] = None
        #: The network as CUDA graphs (`cuda_graphs`), built on the first CUDA search.
        self._graphed: Optional[_GraphedNetwork] = None
        if self.cfg.backend not in ("python", "cpp"):
            raise ValueError(f"unknown search backend {self.cfg.backend!r}")
        #: Roots of the current call, so _evaluate_batch can tell a caller-owned state from one
        #: the searcher created itself.
        self._root_nodes: List[_BNode] = []
        self._rng = random.Random(self.cfg.seed)
        self._np_rng = np.random.RandomState(self.cfg.seed)
        self.model.eval()
        #: The Gumbel root (`gumbel_k`), with the plain PUCT searcher it searches candidates with:
        #: the same network and settings, one world per phase chosen by the root, so no
        #: determinization, filtering, subsampling, tree reuse or root noise of its own.
        self._gumbel: Optional[GumbelRoot] = None
        if self.cfg.gumbel_k > 0:
            sub_cfg = replace(self.cfg, determinize=False, node_filter="all", subsample=1.0,
                              gumbel_k=0, reuse_subtree=False, dirichlet_frac=0.0)
            sub = BatchedMCTS(model, device=self.device, config=sub_cfg,
                              featurise_capacity=featurise_capacity)
            self._gumbel = GumbelRoot(self, sub)
        #: Persistent roots, keyed by whatever the caller uses to identify a position stream
        #: (a game index, typically). Only populated when `reuse_subtree` is on.
        self._trees: Dict[object, _BNode] = {}

    # -- evaluation -----------------------------------------------------------------------

    def _evaluate_batch(self, nodes: Sequence[_BNode]) -> None:
        """Fill in priors and value for a batch of unexpanded, non-terminal nodes."""
        if not nodes:
            return
        # Roots are only safe on the fast path when the searcher settled them itself.
        allow_fast = self.cfg.advance_root or not any(nd is r for nd in nodes
                                                      for r in self._root_nodes)
        obs, masks = self._featurise(nodes, allow_fast=allow_fast)
        obs_t = torch.from_numpy(obs).to(self.device)
        mask_t = torch.from_numpy(masks).to(self.device)
        with torch.no_grad():
            logits, v_win, _ = self.model.forward(obs_t, mask_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            values = v_win.squeeze(-1).cpu().numpy().tolist()

        # The legal entries of every row at once: one nonzero over the batch instead of four
        # numpy calls per node. Row-major order keeps each row's actions ascending, as before.
        rows, cols = np.nonzero(masks)
        counts = np.bincount(rows, minlength=len(nodes)).tolist()
        pri_all = probs[rows, cols]
        cols_l = cols.tolist()
        start = 0
        for i, nd in enumerate(nodes):
            k = counts[i]
            if k == 0:
                nd.terminal = True
                nd.value_us = 0.0
                nd.expanded = True
                continue
            # In double precision, as the C++ tree does, so the two agree to the last bit.
            pri = pri_all[start:start + k].astype(np.float64)
            total = float(pri.sum())
            nd.priors = (pri / total).tolist() if total > 1e-12 else [1.0 / k] * k
            nd.actions = cols_l[start:start + k]
            start += k
            nd.n = [0.0] * k
            nd.w = [0.0] * k
            nd.total = 0.0
            # v_win is from the mover's perspective; store it from the US perspective.
            v = float(values[i])
            nd.value_us = v if nd.mover == _US else -v
            nd.expanded = True

    def _featurise(self, nodes: Sequence[_BNode],
                   allow_fast: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Observations and legal masks for a batch of leaves.

        `allow_fast=False` forces the per-node path. The C++ runner resolves chance nodes when a
        state is set into it, so for a state the CALLER still owns it reports the mask of a later
        decision -- at a ROLL_DIE node it offered 6 actions where the caller had 1. Interior nodes
        are the searcher's own settled hypotheticals and are unaffected.
        """
        n = len(nodes)
        if allow_fast and 0 < n <= self._featurise_capacity:
            runner = self._featuriser_for(n)
            for i, nd in enumerate(nodes):
                runner.set_state(i, nd.state)
            # REQUIRED: set_state leaves the cached observation buffer stale, and the stale
            # features match no perspective -- a silent corruption of every leaf evaluation.
            runner.refresh_all()
            obs = np.asarray(runner.get_observations(), dtype=np.float32)[:n]
            masks = np.asarray(runner.get_action_masks())[:n]
            return obs, masks
        obs = np.stack([np.asarray(ts.extract_observation(nd.state, acting_player(nd.state)),
                                   dtype=np.float32) for nd in nodes])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(nd.state)) for nd in nodes])
        return obs, masks

    def _featuriser_for(self, n: int) -> ts.VectorizedBatchRunner:
        """The smallest cached runner holding `n` states (a power of two, capped at capacity)."""
        cap = _MIN_FEATURISE_BUCKET
        while cap < n:
            cap *= 2
        cap = min(cap, self._featurise_capacity)
        runner = self._featurisers.get(cap)
        if runner is None:
            runner = ts.VectorizedBatchRunner(cap, int(self.cfg.seed))
            self._featurisers[cap] = runner
        return runner

    @staticmethod
    def _make_node(state: ts.GameState, searcher: int = 0) -> _BNode:
        if ts.Engine.is_terminal(state):
            return _BNode(state=state, mover=0, terminal=True, expanded=True,
                          value_us=float(ts.Engine.get_terminal_utility(state)), searcher=searcher)
        mover = int(acting_player(state))
        return _BNode(state=state, mover=mover, terminal=False, searcher=searcher or mover)

    # -- search ---------------------------------------------------------------------------

    def _select(self, node: _BNode) -> int:
        """PUCT, read from the mover's side of a US-perspective value.

        Written as scalar arithmetic over lists rather than numpy: on ~10 elements the six numpy
        calls cost 4.17us against 0.89us here, and both pick the same index -- verified over 3,000
        random draws per branching level, near-ties included.
        """
        if self.cfg.opponent == "greedy" and node.mover != node.searcher:
            return max(range(len(node.priors)), key=node.priors.__getitem__)
        total = node.total
        sqrt_total = math.sqrt(total if total > 1.0 else 1.0)
        c = self.cfg.c_puct
        fpu = self.cfg.fpu_reduction
        value_us = node.value_us
        us_moves = node.mover == _US
        best_i, best_v = 0, -1e30
        # Same arithmetic in the same order as the reference loop, so the choice is bit-identical.
        for i, (ni, wi, pi) in enumerate(zip(node.n, node.w, node.priors)):
            q = (wi / ni) if ni > 0 else value_us
            if not us_moves:
                q = -q
            if ni == 0:
                q -= fpu
            v = q + c * pi * sqrt_total / (1.0 + ni)
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
                child = self._make_node(nxt, node.searcher)
                node.children[action] = child
                return path, child
            node = child

    @staticmethod
    def _backup(path: Sequence[Tuple[_BNode, int]], value_us: float) -> None:
        for parent, idx in path:
            parent.n[idx] += 1.0
            parent.w[idx] += value_us
            parent.total += 1.0

    def run(self, states: Sequence[ts.GameState],
            keys: Optional[Sequence[object]] = None) -> List[Tuple[List[int], np.ndarray]]:
        """Search every position. Returns (actions, visit_counts) per input, in order.

        `keys` identifies each position's stream so its subtree can be carried forward by
        `advance()`. Required only when `reuse_subtree` is set.
        """
        out: List[Tuple[List[int], np.ndarray]] = []
        for r in self._search(states, keys):
            if r is None or r.terminal or not r.actions:
                out.append(([], np.zeros(0)))
            else:
                out.append((r.actions, np.array(r.n, dtype=float)))
        return out

    def _search(self, states: Sequence[ts.GameState],
                keys: Optional[Sequence[object]] = None,
                simulations: Optional[Sequence[int]] = None,
                searchers: Optional[Sequence[int]] = None) -> List[Optional[_BNode]]:
        """The search behind `run`, returning each position's root node rather than its visit
        counts, so a caller choosing a move can see the values and priors behind them.

        `simulations` gives each position its own budget (default `cfg.simulations` for all), so
        searches of different sizes share one set of rounds -- the Gumbel root's candidates."""
        cfg = self.cfg
        budgets = ([int(b) for b in simulations] if simulations is not None
                   else [int(cfg.simulations)] * len(states))
        if len(budgets) != len(states):
            raise ValueError(f"{len(budgets)} budgets for {len(states)} positions")
        reuse = cfg.reuse_subtree and not cfg.determinize and keys is not None
        if cfg.opponent not in ("search", "greedy"):
            raise ValueError(f"unknown opponent model {cfg.opponent!r}")
        if cfg.backend == "cpp" and not reuse:
            if cfg.opponent != "search":
                raise ValueError("opponent='greedy' is implemented in the Python tree only "
                                 "(backend='python')")
            return self._search_cpp(states, budgets)
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
            if cfg.advance_root:
                settle(s, cfg.auto_advance)
            if cfg.determinize and not ts.Engine.is_terminal(s):
                s = determinize(s, acting_player(s), self._rng)
                s.rng_state = self._rng.getrandbits(64) % _UINT64
            roots.append(self._make_node(s, int(searchers[i]) if searchers is not None else 0))
            inherited.append(False)

        # One batch per group for the roots, then one per group per simulation round.
        groups = search_groups(len(roots))
        self._root_nodes = [r for r in roots if r is not None]
        for b, e in groups:
            self._evaluate_batch([r for r in roots[b:e] if r is not None and not r.expanded])
        self._root_nodes = []

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
        remaining = [max(0, budget - int(sum(r.n)))
                     if (r is not None and not r.terminal and r.actions) else 0
                     for r, budget in zip(roots, budgets)]

        while any(remaining):
            for b, e in groups:
                pending: List[Tuple[List[Tuple[_BNode, int]], _BNode]] = []
                for i in range(b, e):
                    r = roots[i]
                    if r is None or r.terminal or not r.actions or remaining[i] <= 0:
                        continue
                    remaining[i] -= 1
                    pending.append(self._descend(r))
                # Every tree contributed at most one leaf, so one batch completes the group's round.
                self._evaluate_batch([leaf for _, leaf in pending
                                      if not leaf.terminal and not leaf.expanded])
                for path, leaf in pending:
                    self._backup(path, leaf.value_us)

        if reuse:
            for i, r in enumerate(roots):
                if r is not None:
                    self._trees[key_list[i]] = r
        return roots

    def _root_states(self, states: Sequence[ts.GameState]) -> List[ts.GameState]:
        """Each position as it is searched: settled if `advance_root`, resampled if determinized."""
        out: List[ts.GameState] = []
        for st in states:
            s = st.clone()
            if self.cfg.advance_root:
                settle(s, self.cfg.auto_advance)
            if self.cfg.determinize and not ts.Engine.is_terminal(s):
                s = determinize(s, acting_player(s), self._rng)
                s.rng_state = self._rng.getrandbits(64) % _UINT64
            out.append(s)
        return out

    def _cpp_for(self, n: int) -> Tuple[ts.BatchedSearch, "_SearchBuffers"]:
        """The C++ searcher, rebuilt at the next power of two when `n` trees exceed its capacity.
        A searcher's capacity only bounds how many trees it holds: a batch searched by a larger one
        is searched identically."""
        if self._cpp is not None and self._cpp[0].capacity >= n:
            return self._cpp
        cap = _MIN_FEATURISE_BUCKET
        while cap < n:
            cap *= 2
        self._cpp = None                      # release the old one before allocating the new
        cs = ts.BatchedSearch(cap, float(self.cfg.c_puct), bool(self.cfg.auto_advance),
                              int(model_obs_features(self.model)), False,
                              float(self.cfg.fpu_reduction))
        bufs = _SearchBuffers(cap, int(cs.obs_width), torch.device(self.device))
        cs.set_buffers(bufs.obs.numpy(), bufs.masks.numpy())
        self._cpp = (cs, bufs)
        return self._cpp

    def _launch(self, cs: ts.BatchedSearch, bufs: "_SearchBuffers", g: int) -> Optional[Tuple[int, int, object]]:
        """Group g's next leaves from the C++ tree, sent to the network without waiting for it.

        On CUDA the copies in and out are asynchronous (page-locked buffers) and an event marks
        the results ready, so the caller can run the tree for another group meanwhile.
        """
        k = cs.select_leaves(g)
        if k == 0:
            return None
        off = cs.group_offset(g)
        with torch.no_grad():
            if self.cfg.cuda_graphs and bufs.cuda:
                if self._graphed is None:
                    self._graphed = _GraphedNetwork(self.model, int(cs.obs_width), torch.device(self.device))
                probs, values = self._graphed.run(bufs.obs[off:off + k], bufs.masks[off:off + k])
            else:
                obs = bufs.obs[off:off + k].to(self.device, non_blocking=True)
                masks = bufs.masks[off:off + k].to(self.device, non_blocking=True)
                logits, v_win, _ = self.model.forward(obs, masks)
                probs = torch.softmax(logits.float(), dim=-1)
                values = v_win.float().reshape(-1)
            bufs.probs[off:off + k].copy_(probs, non_blocking=True)
            bufs.values[off:off + k].copy_(values, non_blocking=True)
        event = None
        if bufs.cuda:
            event = torch.cuda.Event()
            event.record()
        return k, off, event

    @staticmethod
    def _finish(cs: ts.BatchedSearch, bufs: "_SearchBuffers", g: int,
                job: Optional[Tuple[int, int, object]]) -> None:
        """Wait for group g's results and hand them to the C++ tree."""
        if job is None:
            return
        k, off, event = job
        if event is not None:
            cast(torch.cuda.Event, event).synchronize()
        cs.expand_and_backup(g, bufs.probs_np[off:off + k], bufs.values_np[off:off + k])

    def _search_cpp(self, states: Sequence[ts.GameState],
                    budgets: Sequence[int]) -> List[Optional[_BNode]]:
        """`_search` on the C++ tree. The roots are prepared here exactly as the Python path
        prepares them, and come back as `_BNode`s carrying the root statistics, so every caller
        (`run`, `best_actions`, the agent) reads them unchanged."""
        cfg = self.cfg
        roots = self._root_states(states)
        if not roots:
            return []
        cs, bufs = self._cpp_for(len(roots))
        groups = search_groups(len(roots))
        # The children's chance seeds come from this object's generator, drawn in C++ exactly as
        # the Python tree would draw them, and the generator is handed back advanced.
        rng_version, rng_words, rng_gauss = self._rng.getstate()
        if all(b == budgets[0] for b in budgets):
            cs.reset(roots, int(budgets[0]), list(rng_words), len(groups))
        else:
            cs.reset(roots, [int(b) for b in budgets], list(rng_words), len(groups))
        # The roots, every group's sent before any is waited for.
        jobs = [self._launch(cs, bufs, g) for g in range(len(groups))]
        for g, job in enumerate(jobs):
            self._finish(cs, bufs, g, job)
        # Root noise, after the roots' own evaluation, from this object's stream -- as the Python
        # path draws it.
        if cfg.dirichlet_frac > 0.0:
            f = cfg.dirichlet_frac
            for i in range(len(roots)):
                pri = cs.root(i)[4]
                if len(pri) > 1:
                    noise = self._np_rng.dirichlet([cfg.dirichlet_alpha] * len(pri))
                    cs.set_root_priors(i, [(1.0 - f) * p + f * float(z) for p, z in zip(pri, noise)])
        # The rounds, pipelined: while the network evaluates one group's leaves, the tree selects
        # the next group's. Rounds are still taken group 0, 1, 0, 1, ... -- the Python tree's
        # order -- so the chance seeds come out the same.
        queue = deque((g, self._launch(cs, bufs, g)) for g in range(len(groups))
                      if cs.group_has_budget(g))
        while queue:
            g, job = queue.popleft()
            self._finish(cs, bufs, g, job)
            if cs.group_has_budget(g):
                queue.append((g, self._launch(cs, bufs, g)))
        self._rng.setstate((rng_version, tuple(cs.mt_state()), rng_gauss))
        out: List[Optional[_BNode]] = []
        for i, st in enumerate(roots):
            terminal, mover, value_us, actions, priors, n, w = cs.root(i)
            nd = _BNode(state=st, mover=int(mover), terminal=bool(terminal), value_us=float(value_us),
                        actions=list(actions), priors=list(priors), n=list(n), w=list(w),
                        expanded=True, total=float(sum(n)))
            out.append(nd)
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

    def should_search(self, state: ts.GameState) -> bool:
        """Is this a decision this configuration searches?

        Kept on the searcher rather than in each caller so that a coverage setting means the same
        thing in an evaluation tournament and in a training rollout -- the two numbers are only
        comparable if the rule is one implementation.
        """
        if self.cfg.node_filter == "card_playmode":
            dt = int(state.ctx().decision_type)
            if dt not in (int(ts.DecisionType.SELECT_CARD),
                          int(ts.DecisionType.SELECT_PLAY_MODE)):
                return False
        elif self.cfg.node_filter != "all":
            raise ValueError(f"unknown node_filter {self.cfg.node_filter!r}")
        if self.cfg.subsample >= 1.0:
            return True
        return self._rng.random() < self.cfg.subsample

    def improved_policies(self, states: Sequence[ts.GameState]) -> List[Tuple[List[int], np.ndarray]]:
        """The Gumbel root's improved policy per position (`gumbel_k` > 0), as (actions, probs)
        in the shape `run` returns visit counts, so a trainer can use either as its target."""
        if self._gumbel is None:
            raise ValueError("improved_policies needs the Gumbel root (gumbel_k > 0)")
        return self._gumbel.improved_policies(states)

    @property
    def gumbel_stats(self) -> List[GumbelStats]:
        """The Gumbel root's record of the last `best_actions` / `improved_policies` call, one per
        position (`GumbelRoot.last_stats`); empty without `gumbel_k`."""
        return self._gumbel.last_stats if self._gumbel is not None else []

    def best_actions(self, states: Sequence[ts.GameState]) -> List[int]:
        """Most-visited action per position; falls back to the first legal action. With
        `gumbel_k` set, the Gumbel root's choice instead (ai/search/gumbel_root.py)."""
        if self._gumbel is not None:
            return self._gumbel.choose(states)
        picks: List[int] = []
        for r, st in zip(self._search(states), states):
            if r is None or r.terminal or not r.actions:
                mask = np.asarray(ActionEncoder.get_legal_mask(st))
                legal = np.flatnonzero(mask)
                picks.append(int(legal[0]) if len(legal) else 0)
            else:
                picks.append(int(r.actions[self._most_visited(r)]))
        return picks

    @staticmethod
    def _most_visited(root: _BNode) -> int:
        """Index of the move to play: most visits, then the better mean value for the mover,
        then the higher prior.

        Visit counts tie often at small budgets -- at two simulations whenever the favourite's
        first visit sends the second to the runner-up -- and resolving a tie by list position
        played whichever move the engine happened to enumerate first. A 2-simulation search
        scored 47.7% against its own network at 512 games a side, where it should have matched
        it. With the value, then the prior, breaking ties, one simulation plays exactly the
        network's argmax.
        """
        us_moves = root.mover == int(ts.Player.US)

        def key(i: int) -> Tuple[float, float, float]:
            n = root.n[i]
            if n > 0:
                q = root.w[i] / n
                q = q if us_moves else -q
            else:
                q = -math.inf
            return (n, q, root.priors[i])

        return max(range(len(root.actions)), key=key)

    def reseed(self, seed: int) -> None:
        """Restart every stream this search draws from -- the worlds it determinizes, the chance
        nodes it expands, the decisions it subsamples, the root noise -- at `seed`.

        The streams are this object's own, so the global seeds do not reach them, and a search
        that plays several games in one process carries them from game to game. A tournament
        shard calls this so that its games do not depend on what the process searched before
        (tools/lib/parallel_tournament.py).
        """
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

    def reset(self) -> None:
        self.reseed(self.cfg.seed)


class BatchedMCTSAgent:
    """PlayerAgent-compatible wrapper, for single-position play through the standard CLIs.

    The batching this class is built on pays off for many concurrent games; one position at a
    time is the worst case for it. It is still the right searcher to expose, because it is the one
    that will run in training -- a replay generated by a different implementation would not show
    what training actually does.
    """

    def __init__(self, model, name: str = "search", device=None,
                 config: Optional[BatchedMCTSConfig] = None,
                 featurise_capacity: int = 4096) -> None:
        self.model = model
        self.name = name
        # Sized for the batched path by default. At capacity 1 every leaf is featurised one at a
        # time, which `_featurise` notes costs ~10x more than letting the runner do the batch.
        self.mcts = BatchedMCTS(model, device=device, config=config,
                                featurise_capacity=featurise_capacity)

    def reseed(self, seed: int) -> None:
        """Restart the search's own random streams at `seed` (`BatchedMCTS.reseed`)."""
        self.mcts.reseed(seed)

    def _policy_actions(self, states: Sequence[ts.GameState]) -> List[int]:
        """The unsearched fallback: this agent's own greedy policy, in one batched pass."""
        obs = np.stack([np.asarray(ts.extract_observation(
            st, st.ctx().decision_player if st.ctx().decision_player != ts.Player.NONE
            else st.phasing_player), dtype=np.float32) for st in states])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
                          for st in states])
        dev = self.mcts.device
        with torch.no_grad():
            logits, _, _ = self.model(torch.from_numpy(obs).to(dev),
                                      torch.from_numpy(masks).to(dev))
            return [int(a) for a in torch.argmax(logits, dim=-1).cpu().numpy()]

    def select_actions_batch(self, states: Sequence[ts.GameState]) -> List[int]:
        """Actions for a whole batch, searching only the positions this coverage setting covers.

        One search call for every covered position in the batch, and one policy forward for the
        rest. Calling `select_action` per game instead is what makes a search tournament
        unaffordable.
        """
        states = list(states)
        want = [self.mcts.should_search(st) for st in states]
        out: List[int] = [0] * len(states)

        searched_idx = [i for i, w in enumerate(want) if w]
        if searched_idx:
            picks = self.mcts.best_actions([states[i] for i in searched_idx])
            for i, a in zip(searched_idx, picks):
                out[i] = int(a)

        plain_idx = [i for i, w in enumerate(want) if not w]
        if plain_idx:
            picks = self._policy_actions([states[i] for i in plain_idx])
            for i, a in zip(plain_idx, picks):
                out[i] = int(a)

        # A determinized search can legitimately return an action that is illegal in the real
        # state, because in this game the legal SET itself can depend on hidden information.
        # The Cambridge Five (card 104) is the worked example: it names the regions on the
        # opponent's hidden scoring cards, so a sampled world where the US holds Asia Scoring
        # makes all of Asia placeable when the true state does not. Determinization's usual
        # assumption -- same action set in every world -- does not hold here.
        #
        # So the search proposes and the true mask disposes: an illegal pick is replaced by this
        # agent's own greedy policy over the real mask, and counted. A pick that is illegal after
        # that is a genuine fault (a root rooted at the wrong decision) and raises.
        bad = [i for i, a in enumerate(out)
               if not _legal_here(states[i], a)]
        if bad:
            repl = self._policy_actions([states[i] for i in bad])
            for i, a in zip(bad, repl):
                out[i] = int(a)
            self.world_mismatch_count = getattr(self, "world_mismatch_count", 0) + len(bad)

        for i, a in enumerate(out):
            if not _legal_here(states[i], a):
                mask = np.asarray(ActionEncoder.get_legal_mask(states[i]))
                legal = np.flatnonzero(mask)
                raise RuntimeError(
                    f"action {a} for batch position {i} is not legal in the caller's state even "
                    f"after the policy fallback (decision_type="
                    f"{int(states[i].ctx().decision_type)}, {legal.size} legal action(s): "
                    f"{legal[:8].tolist()}). The policy is masked by this same mask, so this is "
                    f"not a determinization mismatch -- the tree is rooted at a different "
                    f"decision than the caller holds. Set advance_root=False when the caller "
                    f"does not settle the state itself.")

        self.searched_count = getattr(self, "searched_count", 0) + len(searched_idx)
        self.decision_count = getattr(self, "decision_count", 0) + len(states)
        return out

    def select_action(self, state: ts.GameState, player, temperature: float = 0.1) -> int:
        action = int(self.mcts.best_actions([state])[0])
        # Fail loudly rather than hand back something the caller will have rejected. The engine
        # already returns False for a mismatched action and leaves the state untouched, so a
        # caller that ignores the return value loops on the same node forever -- which is exactly
        # how this surfaced: 1,355 rejected actions at one POINT_NODE.
        mask = np.asarray(ActionEncoder.get_legal_mask(state))
        if not (0 <= action < len(mask)) or not mask[action]:
            raise RuntimeError(
                f"search returned action {action}, which is not legal in the caller's state "
                f"(decision_type={int(state.ctx().decision_type)}). This means the tree was "
                f"rooted at a different decision -- set advance_root=False when the caller does "
                f"not settle the state itself.")
        return action

    def reset(self) -> None:
        self.mcts.reset()
