"""A player that asks only the critic: at each decision it plays every legal move on a copy of the
position and picks the one whose resulting position the critic, read from its own side, likes best
(owner, 2026-11-18). No policy, no rollout -- one network call per legal move.

It reads the critic right after the move, before anything else happens, so it is the measure of a
network trained with --afterstate-value-coef (ai/training/afterstate_critic.py), whose critic has
seen those positions; on one trained without it the same reads are an extrapolation.

The move is applied in a determinized copy -- the cards the mover cannot see (the opponent's hand and
the draw deck) reshuffled, a fresh RNG -- so whatever a move reveals or decides at random inside the
engine is a sample, not the real game's future. `worlds` averages that many such copies; every legal
move is read in the same copies, so their values are compared on the same draws. A move that ends the
game is valued at its result (+1 / 0 / -1 for the mover), not by the critic.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
import ts_engine as ts

from bindings.ts_env import model_obs_features, observe

class CriticArgmaxAgent:
    """A tournament entrant (`select_actions_batch`, the state-based batch hook): the legal move whose
    afterstate the critic values highest, from the mover's side."""

    def __init__(self, model: Any, name: str, device: Any = "cpu", worlds: int = 1, seed: int = 0) -> None:
        if worlds < 1:
            raise ValueError(f"worlds must be >= 1, got {worlds}")
        self.model = model
        self.name = name
        self.device = torch.device(device)
        self.worlds = int(worlds)
        self.features = model_obs_features(model)
        self.rng = np.random.default_rng(seed)
        self.decisions = 0
        self.agreed_with_policy = 0

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _seed(self) -> int:
        return int(self.rng.integers(0, 1 << 63, dtype=np.int64))

    def select_action(self, state: Any, player: Any, temperature: float = 0.0) -> int:
        """One position (the PlayerAgent interface); the critic's move, at any temperature."""
        return self.select_actions_batch([state])[0]

    def select_actions_batch(self, states: Sequence[Any]) -> List[int]:
        obs: List[np.ndarray] = []
        owner: List[Tuple[int, int]] = []        # (state index, move index) of each critic read
        sums: List[np.ndarray] = []              # per state: per legal move, its values summed over worlds
        legal_all: List[np.ndarray] = []
        for si, st in enumerate(states):
            me = st.ctx().decision_player
            legal = np.flatnonzero(np.asarray(ts.ActionMask.generate_flat_mask(st), dtype=bool))
            legal_all.append(legal)
            sums.append(np.zeros(len(legal)))
            if len(legal) <= 1:
                continue
            sign = 1.0 if me == ts.Player.US else -1.0
            for _ in range(self.worlds):
                world = ts.determinize(st, me, self._seed(), False, True)
                for mi, a in enumerate(legal):
                    s2 = world.clone()
                    s2.rng_state = self._seed()
                    ts.Engine.step_flat(s2, int(a), False)
                    if ts.Engine.is_terminal(s2):
                        sums[si][mi] += sign * float(ts.Engine.get_terminal_utility(s2))
                    else:
                        obs.append(observe(s2, me, self.features))
                        owner.append((si, mi))
        for (si, mi), v in zip(owner, self._values(obs)):
            sums[si][mi] += float(v)
        policy = self._policy_argmax(states, legal_all)
        picks: List[int] = []
        for si, legal in enumerate(legal_all):
            if len(legal) <= 1:
                picks.append(int(legal[0]) if len(legal) else 0)
                continue
            best = int(legal[int(np.argmax(sums[si]))])
            picks.append(best)
            self.decisions += 1
            self.agreed_with_policy += int(best == policy[si])
        return picks

    def _values(self, obs: List[np.ndarray]) -> np.ndarray:
        """The critic's v_win (the observer's win value in [-1, 1]) on each observation, batched."""
        if not obs:
            return np.zeros(0, dtype=np.float32)
        out: List[np.ndarray] = []
        with torch.no_grad():
            for k in range(0, len(obs), 4096):
                o = torch.from_numpy(np.stack(obs[k:k + 4096])).to(self.device, torch.float32)
                out.append(self.model(o, None)[1].float().reshape(-1).cpu().numpy())
        return np.concatenate(out)

    def _policy_argmax(self, states: Sequence[Any], legal_all: List[np.ndarray]) -> List[int]:
        """The network's own greedy move at each state, for the agreement count."""
        rows = [i for i, legal in enumerate(legal_all) if len(legal) > 1]
        out = [-1] * len(states)
        if not rows:
            return out
        o = np.stack([observe(states[i], states[i].ctx().decision_player, self.features) for i in rows])
        m = np.stack([np.asarray(ts.ActionMask.generate_flat_mask(states[i]), dtype=np.uint8) for i in rows])
        with torch.no_grad():
            logits = self.model(torch.from_numpy(o).to(self.device, torch.float32),
                                torch.from_numpy(m).to(self.device))[0]
        for j, i in enumerate(rows):
            out[i] = int(logits[j].argmax())
        return out

    def stats(self) -> str:
        return (f"{self.name}: {self.decisions} decisions with a choice, the critic's move the policy's own in "
                f"{100 * self.agreed_with_policy / max(1, self.decisions):.1f}%")

