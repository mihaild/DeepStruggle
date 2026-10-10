"""A rollout root: the network's top moves, each played out in many sampled worlds, then a cautious choice.

The Ataraxos recipe for Stratego (Sokota et al., 2025) at test time: sample possible worlds, run
depth-limited rollouts by the network from each candidate move, average, and move the policy only as
far as the evidence carries it. Here, at one decision:

1. **Candidates.** The network's `k` most probable legal moves (the search bank found the better
   move inside the top 4 for ~92% of the network's errors).
2. **Worlds.** `worlds` independent draws of the mover's unseen cards and of the dice
   (`determinize` plus a fresh random stream). Every candidate is played in every world, so two
   candidates' values in one world share the cards and the dice: differences are paired.
3. **Rollouts.** After the candidate, the network plays both sides greedily through `horizon`
   action-round boundaries (0: none -- the critic reads the position the move leads to), then the
   critic reads the leaf for the mover; a finished game counts its result. A rollout move that ends
   the game against its own player on the spot (`loses_now`) is replaced by that player's next-best
   move, so a candidate is not credited with an opponent's blind suicide.
4. **Choice** (`rule`): `argmax` -- the best mean value; `z<x>` -- the best mean value only if its
   paired lead over the network's own move is at least x standard errors, else the network's move;
   `kl<t>` -- argmax of log pi(a) + Q(a) / t over the candidates, one mirror-descent step from the
   network's policy (Q the mover's value in [-1, 1]).

The budget is network rows: k x worlds x (1 + rollout steps) plus the root. Honest: candidates come
from the real position, values only from sampled worlds. A world where a candidate is illegal (The
Cambridge Five can change the legal set) is dropped for every candidate of that position.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.dmcts import determinize
from ai.search.pimcts import acting_player
from bindings.action_encoder import ActionEncoder
from bindings.settle import SettleMode, settle

_UINT64 = 1 << 64
_US = ts.Player.US
#: A rollout that has not met its horizon in this many steps is read where it stands.
_MAX_ROLLOUT_STEPS = 400
#: Rows per network call.
_ROWS = 4096


@dataclass
class RolloutConfig:
    k: int = 4
    worlds: int = 64
    horizon: int = 0
    rule: str = "z2"
    seed: int = 0


def parse_rule(rule: str) -> Tuple[str, float]:
    if rule == "argmax":
        return "argmax", 0.0
    if rule.startswith("z"):
        return "z", float(rule[1:] or 2.0)
    if rule.startswith("kl"):
        return "kl", float(rule[2:] or 0.05)
    raise ValueError(f"unknown rule {rule!r}: argmax, z<x> or kl<t>")


def _ar_key(st: ts.GameState) -> Tuple[int, int, int, int]:
    return (int(st.turn), int(st.action_round), int(st.phasing_player), int(st.current_phase))


def _loses_now(st: ts.GameState, a: int, who: ts.Player) -> bool:
    probe = st.clone()
    ts.Engine.step_flat(probe, int(a))
    settle(probe, SettleMode.CHANCE)
    if not ts.Engine.is_terminal(probe):
        return False
    u = float(ts.Engine.get_terminal_utility(probe))
    return (u < 0) if who == _US else (u > 0)


#: Decisions at which one move can end the game against its own player at DEFCON 2.
_GUARDED = {int(ts.DecisionType.SELECT_PLAY_MODE), int(ts.DecisionType.CHOOSE_BRANCH),
            int(ts.DecisionType.POINT_NODE)}


class RolloutRoot:
    def __init__(self, model: Any, config: RolloutConfig, device: Any = "cpu") -> None:
        self.model = model
        self.cfg = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.rule, self.param = parse_rule(config.rule)
        self._rng = random.Random(config.seed)
        #: Per position of the last `choose`: the candidates, each one's mean value and paired lead
        #: over the network's move with its standard error, the worlds used and the rows evaluated.
        self.last_stats: List[Dict[str, Any]] = []
        self.rows = 0

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def _forward(self, states: Sequence[ts.GameState]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Logits, masks and each position's value for its own acting player."""
        lgs, vs, ms = [], [], []
        for lo in range(0, len(states), _ROWS):
            chunk = states[lo:lo + _ROWS]
            obs = np.stack([np.asarray(ts.extract_observation(s, acting_player(s)), dtype=np.float32)
                            for s in chunk])
            masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in chunk])
            with torch.no_grad():
                lg, v, _ = self.model(torch.from_numpy(obs).to(self.device),
                                      torch.from_numpy(masks).to(self.device))
            lgs.append(lg.float().cpu().numpy())
            vs.append(v.float().reshape(-1).cpu().numpy())
            ms.append(masks)
            self.rows += len(chunk)
        return np.concatenate(lgs), np.concatenate(ms), np.concatenate(vs)

    def _rollout(self, states: List[ts.GameState]) -> None:
        """The network plays every state on, greedily, through `horizon` action-round boundaries."""
        if self.cfg.horizon <= 0:
            return
        key = [_ar_key(s) for s in states]
        left = [self.cfg.horizon] * len(states)
        for _ in range(_MAX_ROLLOUT_STEPS):
            live = [j for j, s in enumerate(states) if left[j] > 0 and not ts.Engine.is_terminal(s)]
            if not live:
                return
            lg, masks, _v = self._forward([states[j] for j in live])
            for row, j in enumerate(live):
                s = states[j]
                z = np.where(masks[row].astype(bool), lg[row], -np.inf)
                a = int(np.argmax(z))
                if int(s.defcon) <= 2 and int(s.ctx().decision_type) in _GUARDED:
                    who = acting_player(s)
                    order = [int(x) for x in np.argsort(-z) if np.isfinite(z[x])]
                    for alt in order[:4]:
                        if not _loses_now(s, alt, who):
                            a = alt
                            break
                ts.Engine.step_flat(s, a)
                settle(s, SettleMode.FORCED)
                now = _ar_key(s)
                if now != key[j]:
                    key[j] = now
                    left[j] -= 1

    def choose(self, states: Sequence[ts.GameState]) -> List[int]:
        states = list(states)
        self.rows = 0
        lg, masks, _ = self._forward(states)
        movers = [acting_player(s) for s in states]
        cands: List[List[int]] = []
        for i in range(len(states)):
            z = np.where(masks[i].astype(bool), lg[i], -np.inf)
            cands.append([int(a) for a in np.argsort(-z)[:self.cfg.k] if np.isfinite(z[a])])
        jobs: List[Tuple[int, int, int]] = []          # (position, world, candidate)
        kids: List[ts.GameState] = []
        used_worlds = [0] * len(states)
        for i, st in enumerate(states):
            if len(cands[i]) < 2:
                continue
            for w in range(self.cfg.worlds):
                world = st.clone()
                if not ts.Engine.is_terminal(world):
                    world = determinize(world, movers[i], self._rng)
                world.rng_state = self._rng.getrandbits(64) % _UINT64
                wm = np.asarray(ActionEncoder.get_legal_mask(world))
                if not all(wm[a] for a in cands[i]):
                    continue
                used_worlds[i] += 1
                for a in cands[i]:
                    s = world.clone()
                    ts.Engine.step_flat(s, a)
                    settle(s, SettleMode.FORCED)
                    jobs.append((i, w, a))
                    kids.append(s)
        self._rollout(kids)
        value = np.zeros(len(kids))
        open_ = [j for j, s in enumerate(kids) if not ts.Engine.is_terminal(s)]
        if open_:
            _lg, _m, v = self._forward([kids[j] for j in open_])
            for row, j in enumerate(open_):
                value[j] = v[row] if acting_player(kids[j]) == movers[jobs[j][0]] else -v[row]
        for j, s in enumerate(kids):
            if ts.Engine.is_terminal(s):
                u = float(ts.Engine.get_terminal_utility(s))
                value[j] = u if movers[jobs[j][0]] == _US else -u
        per: Dict[Tuple[int, int], Dict[int, float]] = {}
        for (i, w, a), v_ in zip(jobs, value):
            per.setdefault((i, w), {})[a] = float(v_)
        picks: List[int] = []
        self.last_stats = []
        for i in range(len(states)):
            raw = cands[i][0] if cands[i] else int(np.argmax(np.where(masks[i].astype(bool), lg[i], -np.inf)))
            worlds = [per[(i, w)] for w in range(self.cfg.worlds) if (i, w) in per]
            if len(cands[i]) < 2 or not worlds:
                picks.append(raw)
                self.last_stats.append({"candidates": cands[i], "worlds": 0})
                continue
            q = {a: float(np.mean([wv[a] for wv in worlds])) for a in cands[i]}
            lead: Dict[int, Tuple[float, float]] = {}
            for a in cands[i]:
                d = [wv[a] - wv[raw] for wv in worlds]
                m = float(np.mean(d))
                se = float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0
                lead[a] = (m, se)
            if self.rule == "argmax":
                pick = max(cands[i], key=lambda a: q[a])
            elif self.rule == "z":
                best = max(cands[i], key=lambda a: q[a])
                m, se = lead[best]
                pick = best if best != raw and m > 0 and (se == 0.0 or m / se >= self.param) else raw
            else:
                z = lg[i] - np.max(lg[i][masks[i].astype(bool)])
                logp = {a: float(z[a] - math.log(np.exp(z[masks[i].astype(bool)]).sum())) for a in cands[i]}
                pick = max(cands[i], key=lambda a: logp[a] + q[a] / self.param)
            picks.append(int(pick))
            self.last_stats.append({"candidates": cands[i], "worlds": len(worlds), "q": q,
                                    "lead": {a: list(v) for a, v in lead.items()}})
        return picks


class RolloutRootAgent:
    """A tournament entrant (`select_actions_batch`, the state-based batch hook)."""

    def __init__(self, model: Any, name: str, config: RolloutConfig, device: Any = "cpu") -> None:
        self.model = model
        self.name = name
        self.root = RolloutRoot(model, config, device=device)

    def reseed(self, seed: int) -> None:
        self.root.reseed(seed)

    def select_actions_batch(self, states: Sequence[ts.GameState]) -> List[int]:
        states = list(states)
        picks = self.root.choose(states)
        for i, (st, a) in enumerate(zip(states, picks)):
            if not np.asarray(ActionEncoder.get_legal_mask(st))[a]:
                raise RuntimeError(f"rollout root returned an illegal action {a}")
        return picks

    def select_action(self, state: ts.GameState, *_: Any, **__: Any) -> int:
        return self.select_actions_batch([state])[0]


def rollout_spec_config(spec: str) -> Tuple[str, RolloutConfig, str]:
    """rollout:<checkpoint>[:k[:worlds[:horizon[:rule]]]] -> (checkpoint, config, label)."""
    parts = spec.split(":")
    if parts[0].lower() != "rollout" or len(parts) < 2:
        raise ValueError(f"not a rollout spec: {spec!r}")

    def field(i: int) -> Optional[str]:
        return parts[i] if len(parts) > i and parts[i] else None

    d = RolloutConfig()
    cfg = RolloutConfig(k=int(field(2) or d.k), worlds=int(field(3) or d.worlds),
                        horizon=int(field(4) or d.horizon), rule=field(5) or d.rule)
    parse_rule(cfg.rule)
    return parts[1], cfg, f"rollout-k{cfg.k}-w{cfg.worlds}-h{cfg.horizon}-{cfg.rule}"
