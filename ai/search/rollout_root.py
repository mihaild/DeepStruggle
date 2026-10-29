"""A rollout root: the network's top moves, each played out in many sampled worlds, then a cautious choice.

The Ataraxos recipe for Stratego (Sokota et al., 2025) at test time: sample possible worlds, run
depth-limited rollouts by the network from each candidate move, average, and move the policy only as
far as the evidence carries it. Here, at one decision:

1. **Candidates.** The network's `k` most probable legal moves (the search bank found the better
   move inside the top 4 for ~92% of the network's errors).
2. **Worlds.** `worlds` independent draws of the mover's unseen cards and of the dice. Every
   candidate is played in every world, so two candidates' values in one world share the cards and
   the dice: differences are paired.
3. **Rollouts.** After the candidate, the network plays both sides greedily through `horizon`
   action-round boundaries (0: none -- the critic reads the position the move leads to), then the
   critic reads the leaf from the mover's side; a finished game counts its result. A rollout move
   that ends the game against its own player on the spot is replaced by that player's next-best
   move, so a candidate is not credited with an opponent's blind suicide.
4. **Choice** (`rule`): `argmax` -- the best mean value; `z<x>` -- the best mean value only if its
   paired lead over the network's own move is at least x standard errors, else the network's move;
   `kl<t>` -- argmax of log pi(a) + Q(a) / t over the candidates, one mirror-descent step from the
   network's policy (Q the mover's value in [-1, 1]).

The defaults are the measured player, `rollout:<ckpt>:4:16:4:z2`
(research/log/E7_search_depth_and_value.md section 4). The budget is network rows: k x worlds x
(1 + rollout steps) plus the root. Honest: candidates come from the real position, values only from
sampled worlds. A world where a candidate is illegal (The Cambridge Five can change the legal set)
is dropped for every candidate of that position.

Racing (`race`, off by default): every candidate is played on the first `race` worlds; a challenger
whose paired lead over the network's move is then below 0 leaves the race, and the other
`worlds - race` worlds are played only by the network's move and the challengers left. The choice reads
those over all the worlds. Offline, racing 16 of 32 gained as much over the network's move as every
candidate on all 32 (-0.002 +- 0.002 x100 a decision on held-out worlds, departing as often) at ~83%
of the playouts; in games, `4:64:...:race16` plays as 4:64 for ~1.5x the time of 4:32
(research/log/E7_rollout_race.md). A survivor's statistics include the worlds that kept it in the race,
so its lead reads a little high; a rollout teacher, whose targets read them, is never raced.

Everything but the network calls is C++ shared with the browser (bindings/rollout_batch.hpp):
`ts.rollout_candidates` picks step 1 from the root's logits, `ts.RolloutBatch` is steps 2 and 3,
and `ts.rollout_decide` is step 4 with the statistics it reports. This file is the loop that feeds
the network's logits and values in -- the same loop web/ui/src/analysis/rollout.ts runs with
onnxruntime -- so the two searchers cannot drift apart in anything but the network's arithmetic.
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.belief import BeliefTracker
from ai.search.gumbel_root import sigma_completed
from ai.search.pimcts import acting_player
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import model_obs_features, observe

#: A playout that has not met its horizon in this many plies is an error, not a leaf: reading an
#: unfinished rollout where it stands would value it silently at a depth nobody asked for.
_MAX_ROLLOUT_STEPS = 400
#: Rows per network call.
_ROWS = 4096


#: Where a playout stops (the engine's `rollout::Stop`): after `horizon` action-round boundaries (the
#: measured root), at the end of the decision's turn, or at the later or the earlier of the two.
STOPS = ("boundaries", "turn", "later", "earlier", "predeal", "earlier_predeal")

#: Whose side a leaf's critic is read from (the engine's `rollout::Leaf`): the root mover's at every leaf
#: (the measured root), or the leaf's decider's -- the side the critic is trained on -- negated into the
#: mover's frame where that is the opponent (P33, research/log/P33_diagnostics.md), or -- `own` -- played on
#: past the stop to the root mover's next decision and read there from its side.
LEAF_SIDES = ("mover", "decider", "own")


@dataclass
class RolloutConfig:
    k: int = 4
    worlds: int = 16
    horizon: int = 4            # action-round boundaries (read by every stop but "turn")
    rule: str = "z2"
    seed: int = 0
    stop: str = "boundaries"    # one of STOPS
    leaf_side: str = "mover"    # one of LEAF_SIDES
    #: Play the playouts' forwards (every ply, and the leaf reads) through CUDA graphs. Opt-in: a
    #: graph's batch is padded, and a row's value can move by ~1e-7 with the batch size (the Gumbel
    #: searcher's measurement), so the search is not bit-identical to the eager one. What it buys is
    #: the time to launch ~20 small kernels per ply on the one thread every ply waits for.
    cuda_graphs: bool = False
    #: Keyed chance (bindings/keyed_chance.hpp): each world's dice and deals tied to the situation they
    #: are drawn in, so its candidates draw alike wherever they reach the same one -- a tighter paired
    #: lead for the same worlds. The measured root shares one dice stream per world instead.
    keyed: bool = False
    #: Keyed chance with the situation's player fields narrowed to a byte (bindings/keyed_chance.hpp's
    #: keyed2); implies keyed. `keyed` alone keeps the original packing, in which a USSR side hides
    #: the turn from the key, so every keyed player measured before keyed2 stays what it was.
    keyed2: bool = False
    #: The oracle's worlds: the opponent's true hand kept, only the draw deck reshuffled. Not a
    #: deployable player -- the ceiling on what a belief over the opponent's hand could give the root.
    oracle: bool = False
    #: P35 3b: the worlds dealt from the policy-likelihood belief over the opponent's hand
    #: (ai/search/belief.py) with this many importance-sampled hands; 0 = uniform redeals. Needs the
    #: tournament's game hooks (`RolloutRootAgent.observe`), since the belief reads the opponent's moves.
    belief: int = 0
    #: Count every candidate's horizon from the end of the mover's own action round
    #: (rollout_batch.hpp's Config::aligned). Off: a candidate that leaves choices inside the round
    #: counts the round's end and is read an action round before one that ends it (2026-10-19); the
    #: headline is one boundary however its events resolve.
    aligned: bool = False
    #: Space Race box 4 (rollout_batch.hpp's Config::box4): at the headline choice of the side holding Man
    #: in Space alone, the opponent's headline -- chosen first and revealed -- stays in every world (and
    #: in every belief hand). Off: the world forgets it and asks it again.
    box4: bool = False
    #: Race the worlds: every candidate on the first `race`, then the rest for the network's move and the
    #: challengers whose paired lead over it is not below 0 (RolloutBatch.race). 0: every candidate on
    #: every world, as every rollout player measured before it.
    race: int = 0


@dataclass(frozen=True)
class _CapturedGraph:
    """One captured batch size: the graph, its input buffers and its output tensors."""
    graph: Any                        # torch.cuda.CUDAGraph
    obs: torch.Tensor
    masks: torch.Tensor
    logits: torch.Tensor
    values: torch.Tensor


class _GraphedForward:
    """A network's logits and values as CUDA graphs, one per padded batch size, in one memory pool --
    the rollout root's playout plies and leaf reads, which are many small forwards on one thread.
    Batch sizes are padded to a power of two from 64 (to multiples of 1,024 above 4,096), so a game's
    worth of plies captures a handful of graphs. Rows are independent in these networks: the padding
    rows do not touch the real ones."""

    def __init__(self, model: Any, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.pool: Optional[Any] = None
        self.graphs: Dict[int, _CapturedGraph] = {}

    @staticmethod
    def bucket(k: int) -> int:
        if k <= 4096:
            n = 64
            while n < k:
                n *= 2
            return n
        return -(-k // 1024) * 1024

    def _capture(self, n: int, width: int) -> _CapturedGraph:
        obs = torch.zeros((n, width), dtype=torch.float32, device=self.device)
        masks = torch.zeros((n, ActionEncoder.FLAT_ACTION_SIZE), dtype=torch.uint8, device=self.device)
        masks[:, 0] = 1                       # a padding row never has an empty mask
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(2):                # warm-up outside the capture, as CUDA graphs require
                self.model(obs, masks)
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool), torch.no_grad():
            lg, v, _ = self.model(obs, masks)
            logits = lg.float()
            values = v.float().reshape(-1)
        if self.pool is None:
            self.pool = graph.pool()
        entry = _CapturedGraph(graph, obs, masks, logits, values)
        self.graphs[n] = entry
        return entry

    def __call__(self, obs: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = int(obs.shape[0])
        n = self.bucket(k)
        g = self.graphs.get(n) or self._capture(n, int(obs.shape[1]))
        g.obs[:k].copy_(torch.from_numpy(obs))
        g.masks[:k].copy_(torch.from_numpy(masks))
        g.graph.replay()
        return g.logits[:k].cpu().numpy(), g.values[:k].cpu().numpy()


def _stop_kw(cfg: RolloutConfig) -> Dict[str, Any]:
    """The batch's `stop` and `leaf` arguments, each passed only when it is not the default -- so an
    engine build from before they existed still runs the measured root."""
    kw: Dict[str, Any] = {} if cfg.stop == "boundaries" else {"stop": cfg.stop}
    if cfg.leaf_side != "mover":
        kw["leaf"] = cfg.leaf_side
    if cfg.keyed:
        kw["keyed"] = True
    if cfg.keyed2:
        kw["keyed2"] = True
    if cfg.oracle:
        kw["oracle"] = True
    if cfg.aligned:
        kw["aligned"] = True
    if cfg.box4:
        kw["box4"] = True
    return kw


def parse_rule(rule: str) -> Tuple[str, float]:
    """(kind, parameter) of a rule -- argmax, z<x> or kl<t>; ValueError otherwise. The grammar is
    the engine's (`ts.rollout_parse_rule`), so a spec means the same thing here and in the page."""
    return ts.rollout_parse_rule(rule)


class RolloutRoot:
    def __init__(self, model: Any, config: RolloutConfig, device: Any = "cpu", leaf_model: Any = None) -> None:
        """`leaf_model` (P33a): another network's critic reads the leaves, while `model` still chooses
        the candidates and plays the playouts -- the two channels a network reaches the search by,
        separated. It must read the same observation (the same feature blocks)."""
        self.model = model
        self.leaf_model = leaf_model
        self.cfg = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.rule, self.param = parse_rule(config.rule)
        if config.stop not in STOPS:
            raise ValueError(f"stop must be one of {STOPS}, got {config.stop!r}")
        if config.leaf_side not in LEAF_SIDES:
            raise ValueError(f"leaf_side must be one of {LEAF_SIDES}, got {config.leaf_side!r}")
        if config.race and not 0 < config.race < config.worlds:
            raise ValueError(f"race must be above 0 and below the worlds ({config.worlds}), got {config.race}")
        self.features = model_obs_features(model)
        use_graphs = bool(config.cuda_graphs) and self.device.type == "cuda"
        self._graphed: Dict[int, _GraphedForward] = {}
        if use_graphs:
            for net in (model, leaf_model):
                if net is not None and id(net) not in self._graphed:
                    self._graphed[id(net)] = _GraphedForward(net, self.device)
        if leaf_model is not None and model_obs_features(leaf_model) != self.features:
            raise ValueError(f"the leaf network reads observation features {model_obs_features(leaf_model)!r}, "
                             f"the root's network {self.features!r}: their leaves would not be the same rows")
        self._rng = random.Random(config.seed)
        #: Per position of the last `choose`: the candidates, each one's mean value and paired lead
        #: over the network's move with its standard error, and the worlds used.
        self.last_stats: List[Dict[str, Any]] = []
        #: Network rows of the last `choose`: the root plus every playout ply and leaf.
        self.rows = 0
        #: The root forward of the last `choose` -- logits, masks, each position's value for its
        #: mover -- so a caller that needs the network's own reading (RolloutTeacher's targets) reuses
        #: it rather than running the same batch through the network twice.
        self.last_forward: Tuple[np.ndarray, np.ndarray, np.ndarray] = (
            np.zeros((0, ActionEncoder.FLAT_ACTION_SIZE), dtype=np.float32),
            np.zeros((0, ActionEncoder.FLAT_ACTION_SIZE), dtype=np.uint8), np.zeros(0, dtype=np.float32))

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def _net(self, obs: np.ndarray, masks: np.ndarray, model: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        """Logits and values for observation rows already built, in chunks of _ROWS (`model`, else the
        root's own network)."""
        net = self.model if model is None else model
        graphed = self._graphed.get(id(net))
        if graphed is not None and obs.shape[0]:
            lg_np, v_np = graphed(np.ascontiguousarray(obs, dtype=np.float32), np.ascontiguousarray(masks, dtype=np.uint8))
            self.rows += int(obs.shape[0])
            return lg_np, v_np
        lgs, vs = [], []
        for lo in range(0, obs.shape[0], _ROWS):
            with torch.no_grad():
                lg, v, _ = net(torch.from_numpy(obs[lo:lo + _ROWS]).to(self.device),
                               torch.from_numpy(masks[lo:lo + _ROWS]).to(self.device))
            lgs.append(lg.float().cpu().numpy())
            vs.append(v.float().reshape(-1).cpu().numpy())
        self.rows += int(obs.shape[0])
        if not lgs:
            return np.zeros((0, ActionEncoder.FLAT_ACTION_SIZE), dtype=np.float32), np.zeros(0, dtype=np.float32)
        return np.concatenate(lgs), np.concatenate(vs)

    def _forward(self, states: Sequence[ts.GameState]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Logits, masks and each position's value for its own acting player."""
        obs = np.stack([observe(s, acting_player(s), self.features) for s in states])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
        lg, v = self._net(obs, masks)
        return lg, masks, v

    def _playouts(self, batch: Any) -> None:
        """Drive the batch to its leaves and read them with the critic."""
        for _ in range(_MAX_ROLLOUT_STEPS):
            if batch.all_done():
                break
            obs, masks = batch.fill_live()
            lg, _v = self._net(np.ascontiguousarray(obs), np.ascontiguousarray(masks))
            batch.step(np.ascontiguousarray(lg, dtype=np.float32))
        else:
            raise RuntimeError(f"{batch.live()} playouts still running after {_MAX_ROLLOUT_STEPS} plies")
        if batch.refused():
            raise RuntimeError(f"the engine refused a playout action in {batch.refused()} rows")
        obs, masks = batch.fill_leaves()
        if obs.shape[0]:
            _lg, v = self._net(np.ascontiguousarray(obs), np.ascontiguousarray(masks), self.leaf_model)
            batch.set_leaf_values(np.ascontiguousarray(v, dtype=np.float32))

    def choose(self, states: Sequence[ts.GameState], hands: Optional[Sequence[Optional[np.ndarray]]] = None) -> List[int]:
        """The root's move at each position; `hands[i]`, when given and not None, is position i's
        worlds' opposing hands (worlds x 110 uint8, a belief's) instead of uniform redeals."""
        states = list(states)
        self.rows = 0
        lg, masks, v = self._forward(states)
        lg = np.ascontiguousarray(lg, dtype=np.float32)
        self.last_forward = (lg, masks, v)
        cands: List[List[int]] = [ts.rollout_candidates(lg[i], masks[i], self.cfg.k) for i in range(len(states))]
        for i, c in enumerate(cands):
            if not c:
                raise RuntimeError(
                    f"rollout root: position {i} has no legal action (terminal="
                    f"{ts.Engine.is_terminal(states[i])}, decision_type="
                    f"{int(states[i].ctx().decision_type)}); a searcher is never handed one")
        batch = ts.RolloutBatch(horizon=self.cfg.horizon, shared_dice=True, guard=True,
                                features=self.features, seed=self._rng.getrandbits(64), **_stop_kw(self.cfg))
        position_of: Dict[int, int] = {}
        first = self.cfg.race or self.cfg.worlds
        for i, st in enumerate(states):
            if len(cands[i]) >= 2 and not ts.Engine.is_terminal(st):
                position_of[i] = batch.positions()
                h = None if hands is None else hands[i]
                if h is not None:
                    batch.add(st, cands[i], first, hands=np.ascontiguousarray(h[:first]))
                else:
                    batch.add(st, cands[i], first)
        if batch.rows():
            self._playouts(batch)
        if self.cfg.race and position_of:
            for i, p in position_of.items():
                h = None if hands is None else hands[i]
                if h is not None:
                    batch.race(p, self.cfg.worlds - first, hands=np.ascontiguousarray(h[first:]))
                else:
                    batch.race(p, self.cfg.worlds - first)
            self._playouts(batch)
        picks: List[int] = []
        self.last_stats = []
        for i in range(len(states)):
            raw = cands[i][0]
            p = position_of.get(i)
            if p is None:
                picks.append(raw)
                self.last_stats.append({"candidates": cands[i], "worlds": 0})
                continue
            d = ts.rollout_decide(batch, p, cands[i], lg[i], masks[i], self.cfg.rule)
            if d["worlds"] == 0:
                picks.append(raw)
                self.last_stats.append({"candidates": cands[i], "worlds": 0})
                continue
            picks.append(int(d["pick"]))
            stats: Dict[str, Any] = {
                "candidates": cands[i], "worlds": int(d["worlds"]),
                "q": {a: float(q) for a, q in zip(cands[i], d["q"])},
                "lead": {a: [float(m), float(se)] for a, m, se in zip(cands[i], d["lead"], d["lead_se"])},
            }
            if self.cfg.race:
                stats["dropped"] = [a for c, a in enumerate(cands[i]) if not batch.in_race(p, c)]
            self.last_stats.append(stats)
        return picks


class ContinuationValues:
    """R49 (P33): value targets from the rollout root's own leaves. At a searched decision the position
    is played on greedily -- from the network's own move, both sides the network -- through the root's
    horizon in its sampled worlds, and the critic read at the stop (a finished game is its result); the
    target is the mean, from the mover's side. One candidate, so a quarter of the root's k = 4 cost.

    The interface of the trainer's searchers (`should_search`, `improved_policies`), with no policy
    target: `improved_policies` returns none and leaves each position's value in `last_values` (NaN
    where nothing was played out), which the trainer blends into v_win's target (`--search-value-beta`).
    The network is meant to be the learner's own frozen snapshot -- the trainer passes pi_ref, refreshed
    in place -- since the target is "greedy play from here", which any snapshot of the line gives."""

    def __init__(self, model: Any, config: RolloutConfig, device: Any = "cpu", subsample: float = 1.0,
                 node_filter: str = "all", seed: int = 97531) -> None:
        if node_filter not in ("all", "card_playmode"):
            raise ValueError(f"unknown node_filter {node_filter!r}")
        self.root = RolloutRoot(model, config, device=device)
        self.root.reseed(seed + 1)
        self.subsample = float(subsample)
        self.node_filter = node_filter
        self._rng = random.Random(seed)
        self.model = model
        self.last_values: List[float] = []

    def should_search(self, state: ts.GameState) -> bool:
        if self.node_filter == "card_playmode" and int(state.ctx().decision_type) not in (
                int(ts.DecisionType.SELECT_CARD), int(ts.DecisionType.SELECT_PLAY_MODE)):
            return False
        return self.subsample >= 1.0 or self._rng.random() < self.subsample

    def values(self, states: Sequence[ts.GameState]) -> List[float]:
        """Each position's greedy-continuation value from its mover's side, in [-1, 1]; NaN where no
        world was kept (or the position has no legal move)."""
        states = list(states)
        root = self.root
        lg, masks, _ = root._forward(states)
        lg = np.ascontiguousarray(lg, dtype=np.float32)
        batch = ts.RolloutBatch(horizon=root.cfg.horizon, shared_dice=True, guard=True,
                                features=root.features, seed=root._rng.getrandbits(64), **_stop_kw(root.cfg))
        cand: Dict[int, List[int]] = {}
        position_of: Dict[int, int] = {}
        for i, st in enumerate(states):
            if ts.Engine.is_terminal(st) or not masks[i].any():
                continue
            cand[i] = [int(np.argmax(np.where(masks[i].astype(bool), lg[i], -np.inf)))]
            position_of[i] = batch.positions()
            batch.add(st, cand[i], root.cfg.worlds)
        if batch.rows():
            root._playouts(batch)
        out = [float("nan")] * len(states)
        for i, p in position_of.items():
            d = ts.rollout_decide(batch, p, cand[i], lg[i], masks[i], "argmax")
            if int(d["worlds"]) > 0:
                out[i] = float(d["q"][0])
        return out

    def improved_policies(self, states: Sequence[ts.GameState]) -> List[Tuple[List[int], np.ndarray]]:
        self.last_values = self.values(states)
        return [([], np.zeros(0)) for _ in self.last_values]


class RolloutTeacher:
    """Training targets from a rollout root: R32's online improved-policy targets (P32 T1) with the
    rollout root in place of the Gumbel root. At a searched decision the target over every legal move
    is softmax(logit + sigma(completed Q)) -- the formula R32's Gumbel targets used
    (`ai.search.gumbel_root.sigma_completed`) -- with Q each candidate's mean rollout value, n its
    worlds, and the unsearched moves at the mixed value. The same interface as `BatchedMCTS`
    (`should_search`, `improved_policies`), so the trainer's search-CE path takes either.

    Not raced (RolloutConfig.race): a race leaves a dropped candidate's Q on the first worlds only and
    a survivor's on the worlds that selected it, and the targets read every candidate's Q over every
    world."""

    def __init__(self, model: Any, config: RolloutConfig, device: Any = "cpu", subsample: float = 1.0,
                 node_filter: str = "all", seed: int = 86420) -> None:
        if node_filter not in ("all", "card_playmode"):
            raise ValueError(f"unknown node_filter {node_filter!r}")
        if config.race:
            raise ValueError(f"a rollout teacher is not raced (race={config.race}): its targets read every "
                             "candidate over every world")
        self.root = RolloutRoot(model, config, device=device)
        self.subsample = float(subsample)
        self.node_filter = node_filter
        self._rng = random.Random(seed)
        self.model = model
        self.root.reseed(seed + 1)

    def should_search(self, state: ts.GameState) -> bool:
        if self.node_filter == "card_playmode" and int(state.ctx().decision_type) not in (
                int(ts.DecisionType.SELECT_CARD), int(ts.DecisionType.SELECT_PLAY_MODE)):
            return False
        return self.subsample >= 1.0 or self._rng.random() < self.subsample

    def improved_policies(self, states: Sequence[ts.GameState]) -> List[Tuple[List[int], np.ndarray]]:
        return self.decide(states)[1]

    def decide(self, states: Sequence[ts.GameState]) -> Tuple[List[int], List[Tuple[List[int], np.ndarray]]]:
        """The root's move at each position and its improved policy over the legal moves (P36's active
        search plays the first and trains toward the second)."""
        states = list(states)
        picks = self.root.choose(states)          # fills last_stats (candidates, worlds, Q) and last_forward
        lg, masks, v = self.root.last_forward
        out: List[Tuple[List[int], np.ndarray]] = []
        for i, st in enumerate(self.root.last_stats):
            acts = [int(a) for a in np.flatnonzero(masks[i])]
            if not acts:
                out.append(([], np.zeros(0)))
                continue
            logits = {a: float(lg[i][a]) for a in acts}
            if st["worlds"] > 0:
                q = {int(a): float(x) for a, x in st["q"].items()}
                n = {a: float(st["worlds"]) for a in q}
                sig = sigma_completed(logits, float(v[i]), n, q)
            else:                                   # nothing searched: the teacher's own policy
                sig = {a: 0.0 for a in acts}
            z = np.array([logits[a] + sig[a] for a in acts], dtype=np.float64)
            z = np.exp(z - z.max())
            out.append((acts, z / z.sum()))
        return picks, out


class RolloutRootAgent:
    """A tournament entrant (`select_actions_batch`, the state-based batch hook)."""

    def __init__(self, model: Any, name: str, config: RolloutConfig, device: Any = "cpu", leaf_model: Any = None) -> None:
        self.model = model
        self.name = name
        self.root = RolloutRoot(model, config, device=device, leaf_model=leaf_model)
        self.belief: Optional[BeliefTracker] = (
            BeliefTracker(model, self.root.device, particles=config.belief, seed=config.seed, reveal=config.box4)
            if config.belief else None)

    def reseed(self, seed: int) -> None:
        self.root.reseed(seed)
        if self.belief is not None:
            self.belief.reseed(seed)

    # The tournament's game hooks (tools/lib/batch_tournament.py), for the belief's record.
    def new_games(self) -> None:
        if self.belief is not None:
            self.belief.new_games()

    def observe(self, games: Sequence[int], states: Sequence[ts.GameState], actions: Sequence[int],
                players: Sequence[int]) -> None:
        if self.belief is not None:
            self.belief.observe(games, states, actions, players)

    def select_actions_games(self, games: Sequence[int], states: Sequence[ts.GameState]) -> List[int]:
        """`select_actions_batch` with each position's game, which the belief keys its record by."""
        states = list(states)
        hands = None if self.belief is None else self.belief.worlds(list(games), states, self.root.cfg.worlds)
        return self._checked(states, self.root.choose(states, hands))

    def select_actions_batch(self, states: Sequence[ts.GameState]) -> List[int]:
        if self.belief is not None:
            raise RuntimeError("a belief root needs its games: call select_actions_games (the tournament's hook)")
        states = list(states)
        return self._checked(states, self.root.choose(states))

    def _checked(self, states: List[ts.GameState], picks: List[int]) -> List[int]:
        for i, (st, a) in enumerate(zip(states, picks)):
            if not np.asarray(ActionEncoder.get_legal_mask(st))[a]:
                raise RuntimeError(f"rollout root returned an illegal action {a}")
        return picks

    def select_action(self, state: ts.GameState, *_: Any, **__: Any) -> int:
        return self.select_actions_batch([state])[0]


class TeacherPolicyTargets:
    """The control for the rollout teacher: at a searched decision the target is the frozen teacher's
    own policy over the legal moves, with no search -- R39's mechanics with the rollout root taken out.
    The same interface as `RolloutTeacher` (`should_search`, `improved_policies`)."""

    def __init__(self, model: Any, device: Any = "cpu", subsample: float = 1.0, node_filter: str = "all",
                 seed: int = 97531) -> None:
        if node_filter not in ("all", "card_playmode"):
            raise ValueError(f"unknown node_filter {node_filter!r}")
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.features = model_obs_features(model)
        self.subsample = float(subsample)
        self.node_filter = node_filter
        self._rng = random.Random(seed)

    def should_search(self, state: ts.GameState) -> bool:
        if self.node_filter == "card_playmode" and int(state.ctx().decision_type) not in (
                int(ts.DecisionType.SELECT_CARD), int(ts.DecisionType.SELECT_PLAY_MODE)):
            return False
        return self.subsample >= 1.0 or self._rng.random() < self.subsample

    def improved_policies(self, states: Sequence[ts.GameState]) -> List[Tuple[List[int], np.ndarray]]:
        states = list(states)
        if not states:
            return []
        obs = np.stack([observe(s, acting_player(s), self.features) for s in states])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
        with torch.no_grad():
            lg, _v, _ = self.model(torch.from_numpy(obs).to(self.device), torch.from_numpy(masks).to(self.device))
        lg_np = lg.float().cpu().numpy()
        out: List[Tuple[List[int], np.ndarray]] = []
        for i in range(len(states)):
            acts = [int(a) for a in np.flatnonzero(masks[i])]
            if not acts:
                out.append(([], np.zeros(0)))
                continue
            z = np.array([lg_np[i][a] for a in acts], dtype=np.float64)
            z = np.exp(z - z.max())
            out.append((acts, z / z.sum()))
        return out


def is_entry_decision(st: ts.GameState) -> bool:
    """A headline card choice, or an action round's own card play (not a card chosen inside an event)."""
    ctx = st.ctx()
    if int(ctx.decision_type) != int(ts.DecisionType.SELECT_CARD) or int(ctx.resolving_card) != 0:
        return False
    if st.current_phase == ts.Phase.HEADLINE:
        return int(st.headline_stage) == 0
    return st.current_phase == ts.Phase.ACTION_ROUND


class EntryRolloutAgent:
    """A rollout root applied only at entry decisions (`is_entry_decision`: a headline card choice or an
    action round's card play), the network's greedy move everywhere else -- what P36's active search and
    P40 Arm S's searched opponent search, where `rollout:` alone searches every decision with two or more
    legal moves. `entry:rollout:...` in tournament specs."""

    def __init__(self, inner: RolloutRootAgent) -> None:
        self.inner = inner
        self.name = f"{inner.name}@entry"
        self.searched = 0
        self.decisions = 0

    def reseed(self, seed: int) -> None:
        self.inner.reseed(seed)

    def select_actions_batch(self, states: Sequence[ts.GameState]) -> List[int]:
        states = list(states)
        lg, masks, _ = self.inner.root._forward(states)
        picks = [int(np.argmax(np.where(masks[i].astype(bool), lg[i], -np.inf))) for i in range(len(states))]
        chosen = [i for i in range(len(states)) if is_entry_decision(states[i])]
        if chosen:
            for i, a in zip(chosen, self.inner.select_actions_batch([states[i] for i in chosen])):
                picks[i] = a
        self.decisions += len(states)
        self.searched += len(chosen)
        return picks

    def select_action(self, state: ts.GameState, *_: Any, **__: Any) -> int:
        return self.select_actions_batch([state])[0]


class SparseRolloutAgent:
    """A rollout root applied at a random `frac` of decisions, the network's greedy move at the rest -- to
    measure whether its corrections help one at a time (the network following up) or only in sequence (the
    search following up). `sparse:<frac>:rollout:...` in tournament specs."""

    def __init__(self, inner: RolloutRootAgent, frac: float, seed: int = 24601) -> None:
        if not 0.0 < frac <= 1.0:
            raise ValueError("sparse: the searched fraction must be in (0, 1]")
        self.inner = inner
        self.frac = float(frac)
        self.name = f"{inner.name}@{frac:g}"
        self._rng = random.Random(seed)
        self.searched = 0
        self.decisions = 0

    def reseed(self, seed: int) -> None:
        self.inner.reseed(seed)
        self._rng = random.Random(seed + 1)

    def select_actions_batch(self, states: Sequence[ts.GameState]) -> List[int]:
        states = list(states)
        lg, masks, _ = self.inner.root._forward(states)
        if not masks.any(axis=1).all():
            raise RuntimeError(f"{self.name}: asked to move at a position with no legal action")
        picks = [int(np.argmax(np.where(masks[i].astype(bool), lg[i], -np.inf))) for i in range(len(states))]
        chosen = [i for i in range(len(states)) if self._rng.random() < self.frac]
        if chosen:
            for i, a in zip(chosen, self.inner.select_actions_batch([states[i] for i in chosen])):
                picks[i] = a
        self.decisions += len(states)
        self.searched += len(chosen)
        return picks

    def select_action(self, state: ts.GameState, *_: Any, **__: Any) -> int:
        return self.select_actions_batch([state])[0]


def _spec_tail(spec: str) -> Tuple[str, Optional[str], bool]:
    """(stop, leaf checkpoint, CUDA graphs) of rollout:<ckpt>:k:worlds:horizon:rule[:stop][:graphs][:leaf].
    The trailing fields are told apart by value: a stop word, the word `graphs`, a flag
    `rollout_spec_config` reads, or the leaf -- a checkpoint path (ending in .pt or naming a directory;
    a sixth field that is a path is the leaf, the form before the others existed). Anything else is a
    misspelt flag and is refused: read as a leaf path it would only fail later, as a missing file, or
    -- worse -- not at all if the word happened to name one."""
    parts = spec.split(":")
    stop, leaf, graphs = "boundaries", None, False
    for p in parts[6:]:
        if not p:
            continue
        if p in STOPS:
            stop = p
        elif p in LEAF_SIDES:
            continue                                   # rollout_spec_config reads the leaf side
        elif p == "graphs":
            graphs = True
        elif p in ("keyed", "keyed2", "oracle", "aligned", "box4") or re.fullmatch(r"belief\d*|race\d+", p):
            continue                                   # rollout_spec_config reads them
        elif p.endswith(".pt") or os.sep in p:
            if leaf is not None:
                raise ValueError(f"two leaf checkpoints ({leaf!r}, {p!r}) in rollout spec {spec!r}")
            leaf = p
        else:
            raise ValueError(f"unknown field {p!r} in rollout spec {spec!r}: expected a stop "
                             f"({', '.join(STOPS)}), a leaf side ({', '.join(LEAF_SIDES)}), graphs, keyed, "
                             f"keyed2, oracle, aligned, box4, belief<n>, race<n>, or a leaf checkpoint path")
    return stop, leaf, graphs


def _belief_particles(tail: Sequence[str]) -> int:
    """`belief` (2,048 hands) or `belief<n>` in a rollout spec's tail -> the particle count, else 0."""
    for p in tail:
        m = re.fullmatch(r"belief(\d*)", p)
        if m:
            return int(m.group(1) or 2048)
    return 0


def _race_worlds(tail: Sequence[str]) -> int:
    """`race<n>` in a rollout spec's tail -> n (RolloutConfig.race), else 0."""
    for p in tail:
        m = re.fullmatch(r"race(\d+)", p)
        if m:
            return int(m.group(1))
    return 0


def rollout_spec_leaf(spec: str) -> Optional[str]:
    """The leaf checkpoint of rollout:<checkpoint>:k:worlds:horizon:rule[:stop]:<leaf checkpoint> (P33a),
    or None: the root's own network reads its leaves."""
    return _spec_tail(spec)[1]


def rollout_spec_config(spec: str) -> Tuple[str, RolloutConfig, str]:
    """rollout:<checkpoint>[:k[:worlds[:horizon[:rule[:stop][:decider|own][:keyed|keyed2][:aligned][:box4][:oracle][:race<n>][:graphs][:leaf checkpoint]]]]] ->
    (checkpoint, config, label); an omitted field is the measured player's (4, 16, 4, z2, boundaries,
    the mover's side), the stop one of STOPS, `decider` reads each leaf from its decider's side, `own`
    plays on to the mover's next decision, `keyed` ties each draw to its situation (`keyed2` with the
    player fields narrowed, RolloutConfig.keyed2), `aligned` counts every candidate's horizon from the end of the mover's
    own action round (the headline one boundary), `box4` keeps the opponent's headline revealed by Space Race box 4,
    `oracle` keeps the opponent's true hand in every world, `race<n>` races the worlds from the first n
    (RolloutConfig.race), and the
    leaf checkpoint is `rollout_spec_leaf`'s."""
    parts = spec.split(":")
    if parts[0].lower() != "rollout" or len(parts) < 2:
        raise ValueError(f"not a rollout spec: {spec!r}")

    def field(i: int) -> Optional[str]:
        return parts[i] if len(parts) > i and parts[i] else None

    d = RolloutConfig()
    stop, _leaf, graphs = _spec_tail(spec)
    side = next((p for p in parts[6:] if p in LEAF_SIDES), "mover")
    cfg = RolloutConfig(k=int(field(2) or d.k), worlds=int(field(3) or d.worlds),
                        horizon=int(field(4) or d.horizon), rule=field(5) or d.rule, stop=stop, cuda_graphs=graphs,
                        leaf_side=side, keyed="keyed" in parts[6:], keyed2="keyed2" in parts[6:],
                        oracle="oracle" in parts[6:],
                        aligned="aligned" in parts[6:], box4="box4" in parts[6:],
                        belief=_belief_particles(parts[6:]), race=_race_worlds(parts[6:]))
    if cfg.horizon < 0:
        raise ValueError(f"horizon must be >= 0 in {spec!r}")
    if cfg.race and not 0 < cfg.race < cfg.worlds:
        raise ValueError(f"race must be above 0 and below the worlds ({cfg.worlds}) in {spec!r}")
    parse_rule(cfg.rule)
    stop_label = {"boundaries": "", "turn": "-turnend", "later": "-orturnend", "earlier": "-untilturnend",
                  "predeal": "-predeal", "earlier_predeal": "-untilpredeal"}[cfg.stop]
    horizon = "" if cfg.stop in ("turn", "predeal") else f"-h{cfg.horizon}"
    side_label = {"mover": "", "decider": "-dleaf", "own": "-ownleaf"}[cfg.leaf_side]
    keyed = ("-keyed2" if cfg.keyed2 else "-keyed" if cfg.keyed else "") + ("-aligned" if cfg.aligned else "") + ("-box4" if cfg.box4 else "") + ("-oracle" if cfg.oracle else "") + (
        f"-belief{cfg.belief}" if cfg.belief else "") + (f"-race{cfg.race}" if cfg.race else "")
    return parts[1], cfg, f"rollout-k{cfg.k}-w{cfg.worlds}{horizon}{stop_label}{side_label}{keyed}-{cfg.rule}"
