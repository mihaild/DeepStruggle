"""A Gumbel root for play: choose the move by sequential halving over sampled candidates.

Danihelka et al., "Policy improvement by planning with Gumbel" (ICLR 2022), the root half. PUCT
first visits a move of prior p after about 1/p simulations, so at 16-32 simulations it rarely tries
anything the network is unsure of. Here, at a searched decision (`BatchedMCTSConfig.gumbel_k = k`):

1. **Candidates.** min(k, simulations) moves by Gumbel-top-k over the network's logits:
   g(a) + logit(a), g ~ Gumbel(0, gumbel_scale). gumbel_scale 0 takes the most probable moves,
   deterministically.
2. **Sequential halving.** The `simulations` budget is split over ceil(log2 m) phases for m
   candidates; in each, every surviving candidate's position is searched by the ordinary batched
   search with an equal share, and the better half by g + logit + sigma(completed Q) survives.
   sigma and the completed Q follow mctx (c_visit 50, c_scale 0.1, Q rescaled to [0, 1] over the
   legal moves), as the paper does for the root's choice. The last survivor is played.

**The budget is network evaluations, as PUCT's is.** A candidate's share of `per` is one search of
its position with `per - 1` simulations: the evaluation of the position itself, then `per - 1`
more. A phase that cannot give every survivor at least one evaluation is not run, and the best of
the survivors so far is played. So a Gumbel root at `simulations` never evaluates more positions
than PUCT at `simulations` -- both spend at most `simulations` evaluations plus one for the root.

Honest search samples one world from the decider's side for each phase and searches the candidates'
positions in it undeterminized -- after the move the opponent usually decides, and resampling from
there would show the decider the opponent's real hand.

**Worlds** (`BatchedMCTSConfig.gumbel_worlds`). A world fixes the hidden cards and the dice, and the
search below a candidate resolves each move's dice once, when it expands it: more simulations in
one world deepen the tree under one roll and never average over rolls. With `gumbel_worlds = m`,
each phase draws min(m, share) worlds and splits each candidate's share of evaluations over them,
so a candidate's value averages that many draws of the cards and the dice; 1 is one world per phase
(research/log/E7_search_depth_and_value.md: more worlds make the pick reproducible and, at a fixed
budget, shallower and worse).

Unlike Gumbel MuZero, each phase searches a candidate's position afresh rather than growing one
tree (visits and values pooled across phases), so the batched C++ search does the work unchanged.
`improved_policies` gives the paper's improved-policy training target (P32 T1) and, with it, each
position's search value (T2).

This module takes the searcher it drives from its caller rather than importing it: BatchedMCTS
builds the root, and importing it back would make the two modules a cycle.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
import torch

import ts_engine as ts
from ai.search.dmcts import determinize
from ai.search.pimcts import acting_player
from bindings.action_encoder import ActionEncoder
from bindings.settle import SettleMode, settle
from bindings.ts_env import model_obs_features

if TYPE_CHECKING:
    from ai.search.batched_mcts import BatchedMCTS, _BNode

_UINT64 = 1 << 64
_US = int(ts.Player.US)
C_VISIT, C_SCALE = 50.0, 0.1


class GumbelStats(TypedDict):
    """One position's record of the last `choose` / `improved_policies` call (`last_stats`)."""
    candidates: List[int]                  # the root's candidates, most probable first
    n: Dict[int, float]                    # evaluations spent on each
    q: Dict[int, Optional[float]]          # mean value of its position for the mover (None: unvisited)
    value: float                           # the network's value of the position for the mover


def world_shares(per: int, worlds: int) -> List[int]:
    """A candidate's `per` evaluations in one phase, split as evenly as they go over
    min(worlds, per) independent worlds (`BatchedMCTSConfig.gumbel_worlds`)."""
    m = max(1, min(int(worlds), int(per)))
    base, extra = divmod(int(per), m)
    return [base + (1 if j < extra else 0) for j in range(m)]


def completed_q(logits: Dict[int, float], value_mover: float, n: Dict[int, float],
                q: Dict[int, float]) -> Dict[int, float]:
    """The completed Q per legal move, from the mover's side: a visited move's mean value, an
    unvisited one the mixed value -- the network's value blended with the prior-weighted mean Q of
    the visited moves, weighted by the visit total (mctx)."""
    acts = list(logits)
    top = max(logits.values())
    z = {a: math.exp(logits[a] - top) for a in acts}
    zs = sum(z.values())
    prior = {a: z[a] / zs for a in acts}
    total = sum(n.get(a, 0.0) for a in acts)
    visited = [a for a in acts if n.get(a, 0.0) > 0]
    if visited:
        pz = sum(prior[a] for a in visited)
        mean_q = sum(prior[a] * q[a] for a in visited) / pz if pz > 0 else \
            sum(q[a] for a in visited) / len(visited)
        v_mix = (value_mover + total * mean_q) / (1.0 + total)
    else:
        v_mix = value_mover
    return {a: (q[a] if n.get(a, 0.0) > 0 else v_mix) for a in acts}


def sigma_completed(logits: Dict[int, float], value_mover: float, n: Dict[int, float],
                    q: Dict[int, float]) -> Dict[int, float]:
    """sigma(completed Q) per legal move: unvisited moves take the mixed value, Q is rescaled to
    [0, 1] over the legal moves, then scaled by (c_visit + max visits) * c_scale (mctx)."""
    acts = list(logits)
    cq = completed_q(logits, value_mover, n, q)
    lo, hi = min(cq.values()), max(cq.values())
    scale = (hi - lo) if hi - lo > 1e-8 else 1.0
    max_n = max((n.get(a, 0.0) for a in acts), default=0.0)
    return {a: (C_VISIT + max_n) * C_SCALE * (cq[a] - lo) / scale for a in acts}


def halving_phases(candidates: int) -> int:
    """Phases of sequential halving for this many candidates: ceil(log2 m), none for one."""
    return math.ceil(math.log2(candidates)) if candidates > 1 else 0


def phase_share(simulations: int, phases: int, alive: int, remaining: int) -> int:
    """Evaluations per surviving candidate in one phase: an equal share of the whole budget
    (mctx's floor(n / (log2 m * alive)), at least one), cut to what is left. 0 ends the halving."""
    per = max(1, simulations // (phases * alive))
    return min(per, remaining // alive)


class GumbelRoot:
    """`BatchedMCTS.best_actions` when `gumbel_k > 0`.

    `mcts` is the deciding searcher (its configuration, network and random streams); `sub` is a
    plain PUCT searcher over the same network that searches the candidates' positions, each with
    its own budget.
    """

    def __init__(self, mcts: "BatchedMCTS", sub: "BatchedMCTS") -> None:
        self.mcts = mcts
        self.sub = sub
        self.last_saturated_frac = 0.0
        self.last_values = np.zeros(0, dtype=np.float64)
        #: Per position of the last call: the candidates, each one's evaluations and mean value for
        #: the mover, and the network's value. Read by offline tools (search targets that keep the
        #: root's values beside its choice, tools/search_reliability.py); nothing in play reads it.
        self.last_stats: List[GumbelStats] = []

    def _network(self, states: Sequence[ts.GameState]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Logits, legal masks and the mover's value at the real positions."""
        feats = model_obs_features(self.mcts.model)
        obs = np.stack([np.asarray(ts.extract_observation_features(s, acting_player(s), feats),
                                   dtype=np.float32) for s in states])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
        dev = self.mcts.device
        with torch.no_grad():
            lg, v, _ = self.mcts.model(torch.from_numpy(obs).to(dev), torch.from_numpy(masks).to(dev))
        return lg.float().cpu().numpy(), masks, v.float().reshape(-1).cpu().numpy()

    def choose(self, states: Sequence[ts.GameState]) -> List[int]:
        """The move to play at each position: the last survivor of the halving."""
        alive = self._halve(states)[2]
        return [int(c[0]) if c else 0 for c in alive]

    def improved_policies(self, states: Sequence[ts.GameState]) -> List[Tuple[List[int], np.ndarray]]:
        """The improved policy at each position, as a training target (P32 T1): softmax over
        every legal move of logit / gumbel_prior_temperature + sigma(completed Q), with the visits
        and values the halving gathered -- an unsearched move takes the mixed value (Danihelka et
        al. 2022, mctx). Unlike PUCT visit counts it is sharp: where the search agrees with the
        network it stays the network's policy, and it moves only where the searched values differ.
        It also cannot move a policy whose logits are saturated beyond sigma's reach (at most
        ~15 at 256 evaluations), hence the logit cap it is meant to run with."""
        logits, _g, _alive, n, w, v = self._halve(states)
        out: List[Tuple[List[int], np.ndarray]] = []
        #: P32 T2: each position's search value from the mover's side, the improved policy's
        #: expectation of the completed Q (NaN where nothing was legal), read by the trainer.
        self.last_values = np.full(len(logits), np.nan, dtype=np.float64)
        for i in range(len(logits)):
            acts = list(logits[i])
            if not acts:
                out.append(([], np.zeros(0)))
                continue
            q = {a: w[i][a] / n[i][a] for a in n[i] if n[i][a] > 0}
            sig = sigma_completed(logits[i], float(v[i]), n[i], q)
            z = np.array([logits[i][a] + sig[a] for a in acts], dtype=np.float64)
            z = np.exp(z - z.max())
            pi = z / z.sum()
            cq = completed_q(logits[i], float(v[i]), n[i], q)
            self.last_values[i] = float(sum(p * cq[a] for p, a in zip(pi, acts)))
            out.append((acts, pi))
        return out

    def _halve(self, states: Sequence[ts.GameState]) -> Tuple[
            List[Dict[int, float]], List[Dict[int, float]], List[List[int]],
            List[Dict[int, float]], List[Dict[int, float]], np.ndarray]:
        """Sequential halving at each position: (logits, Gumbel noise, survivors best first,
        visits, value sums -- from the mover's side -- and the network's value)."""
        mcts, cfg = self.mcts, self.mcts.cfg
        states = list(states)
        lg, masks, v = self._network(states)
        # The share of these positions whose network policy is saturated (max p > 1 - 1e-6), where
        # no sigma(completed Q) at this budget can move it (P32 T1 reads it before and after).
        if len(states):
            z = np.where(masks > 0, lg, -np.inf)
            z = z - z.max(axis=1, keepdims=True)
            pmax = 1.0 / np.exp(z).sum(axis=1)
            self.last_saturated_frac = float((pmax > 1.0 - 1e-6).mean())
        movers = [int(acting_player(s)) for s in states]
        logits: List[Dict[int, float]] = []
        g: List[Dict[int, float]] = []
        alive: List[List[int]] = []
        m_max = max(1, min(cfg.gumbel_k, cfg.simulations))
        for i in range(len(states)):
            legal = [int(a) for a in np.flatnonzero(masks[i])]
            logits.append({a: float(lg[i, a]) / cfg.gumbel_prior_temperature for a in legal})
            if cfg.gumbel_scale > 0.0:
                noise = mcts._np_rng.gumbel(size=len(legal)) * cfg.gumbel_scale
                g.append({a: float(x) for a, x in zip(legal, noise)})
            else:
                g.append({a: 0.0 for a in legal})
            alive.append(sorted(legal, key=lambda a: -(g[i][a] + logits[i][a]))[:m_max])
        initial = [list(c) for c in alive]
        n = [{a: 0.0 for a in lo} for lo in logits]
        w = [{a: 0.0 for a in lo} for lo in logits]
        phases = [halving_phases(len(c)) for c in alive]
        remaining = [cfg.simulations] * len(states)
        for ph in range(max(phases, default=0)):
            jobs: List[Tuple[int, int, ts.GameState, int]] = []
            searched: List[int] = []
            for i, st in enumerate(states):
                if ph >= phases[i]:
                    continue
                per = phase_share(cfg.simulations, phases[i], len(alive[i]), remaining[i])
                if per == 0:
                    phases[i] = ph                    # out of budget: play the best so far
                    continue
                remaining[i] -= per * len(alive[i])
                searched.append(i)
                for share in world_shares(per, cfg.gumbel_worlds):
                    world = st.clone()
                    if cfg.determinize and not ts.Engine.is_terminal(world):
                        world = determinize(world, ts.Player(movers[i]), mcts._rng)
                    world.rng_state = mcts._rng.getrandbits(64) % _UINT64
                    world_mask = np.asarray(ActionEncoder.get_legal_mask(world))
                    for a in alive[i]:
                        if not world_mask[a]:
                            continue                  # illegal in this world (Cambridge Five)
                        s = world.clone()
                        ts.Engine.step_flat(s, a)
                        settle(s, SettleMode.FORCED if cfg.auto_advance else SettleMode.CHANCE)
                        jobs.append((i, a, s, share))
            # Every candidate of the phase in one search, each with its own share, so the phase
            # takes as many network rounds as its largest share rather than the sum of them.
            # The root's mover is the side searching, also in candidate positions where the
            # opponent moves next (read only by a best-response search, opponent="greedy").
            roots = self._search([j[2] for j in jobs], [j[3] for j in jobs],
                                 [movers[j[0]] for j in jobs]) if jobs else []
            for (i, a, _s, k), r in zip(jobs, roots):
                if r is None:
                    continue
                if r.terminal or not r.actions:
                    v_us = float(r.value_us)
                else:
                    v_us = (float(r.value_us) + float(sum(r.w))) / (1.0 + float(sum(r.n)))
                n[i][a] += k
                w[i][a] += (v_us if movers[i] == _US else -v_us) * k
            for i in searched:
                q = {a: w[i][a] / n[i][a] for a in n[i] if n[i][a] > 0}
                sig = sigma_completed(logits[i], float(v[i]), n[i], q)
                ranked = sorted(alive[i], key=lambda a: -(g[i][a] + logits[i][a] + sig[a]))
                alive[i] = ranked[:1] if ph == phases[i] - 1 else ranked[:max(1, (len(ranked) + 1) // 2)]
        # Survivors are ranked best first, whether the halving finished or the budget ran out.
        self.last_stats = [
            GumbelStats(candidates=initial[i], n={a: n[i][a] for a in initial[i]},
                        q={a: (w[i][a] / n[i][a] if n[i][a] > 0 else None) for a in initial[i]},
                        value=float(v[i]))
            for i in range(len(states))]
        return logits, g, alive, n, w, v

    def _search(self, positions: Sequence[ts.GameState], evaluations: Sequence[int],
                searchers: Optional[Sequence[int]] = None) -> List[Optional["_BNode"]]:
        """Search each position with its `evaluations` network evaluations: its own, then
        `evaluations - 1` simulations, all in one batched search. The sub-searcher's streams are
        reseeded from the deciding searcher's, so a game depends only on the deciding searcher's
        seed."""
        self.sub._rng.seed(self.mcts._rng.getrandbits(64))
        return self.sub._search(positions, simulations=[e - 1 for e in evaluations],
                                searchers=searchers)
