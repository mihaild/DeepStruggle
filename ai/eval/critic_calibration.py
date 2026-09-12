"""What is `v_win` actually predicting -- the win, or the VP margin?

`Engine::get_terminal_utility` is +/-1: a win is a win, whether it came from reaching 20 VP, from
final scoring, or from the opponent blowing up the world at DEFCON 1. But `coldwar_net` already
carries a note that the learned head drifts toward being a restatement of the scoreboard --
corr(v_win, v_vp) = 0.86 at the time it was written, pricing self-inflicted DEFCON-1 deaths at
-0.27 where ordinary losses sit at -0.72.

If that is still true, the critic is not predicting "do I win" but roughly "what will the VP margin
be, clipped", and the two come apart exactly where the game ends abruptly. A position two plays from
a DEFCON-1 suicide has a fine VP margin and a terrible outcome, and a critic reading the scoreboard
cannot see the difference.

This measures it on a given checkpoint by playing self-play games out and comparing the value it
predicted along the way against both the realised win and the realised VP margin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import ts_engine as ts

#: VP at which the game is won outright. A margin is normalised by this, so both quantities live
#: on [-1, 1] and can be compared directly against `v_win`.
VP_CAP = 20.0


@dataclass
class Calibration:
    """Predictions against outcomes, over a set of visited positions."""

    predicted: List[float] = field(default_factory=list)
    realized_win: List[float] = field(default_factory=list)
    realized_vp: List[float] = field(default_factory=list)
    ended_by: List[str] = field(default_factory=list)

    def corr(self, which: str) -> float:
        a = np.asarray(self.predicted)
        b = np.asarray(self.realized_win if which == "win" else self.realized_vp)
        if a.size < 2 or a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def brier(self) -> float:
        """Mean squared error against the actual win, with both mapped to [0, 1]."""
        p = (np.asarray(self.predicted) + 1.0) / 2.0
        y = (np.asarray(self.realized_win) + 1.0) / 2.0
        return float(np.mean((p - y) ** 2)) if p.size else float("nan")

    def by_ending(self) -> Dict[str, Tuple[int, float, float, float]]:
        """Per ending: how many, mean prediction, mean realised win, mean realised VP."""
        out: Dict[str, Tuple[int, float, float, float]] = {}
        endings = np.asarray(self.ended_by)
        for kind in sorted(set(self.ended_by)):
            sel = endings == kind
            out[kind] = (int(sel.sum()),
                         float(np.mean(np.asarray(self.predicted)[sel])),
                         float(np.mean(np.asarray(self.realized_win)[sel])),
                         float(np.mean(np.asarray(self.realized_vp)[sel])))
        return out


def classify_ending(state: ts.GameState) -> str:
    """How the game finished, from the terminal position.

    The distinction that matters is whether the scoreboard explains the result. A DEFCON-1 loss
    can end a game the loser was winning on VP, and that is precisely the case a scoreboard-shaped
    value function cannot represent.
    """
    vp = int(state.victory_points)
    if int(state.defcon) <= 1:
        return "DEFCON 1"
    if vp >= 20:
        return "20 VP (US)"
    if vp <= -20:
        return "20 VP (USSR)"
    if int(state.turn) >= 10:
        return "final scoring"
    return "other"


def measure(model: Any, device: Any, num_envs: int = 256, base_seed: int = 8181,
            max_iters: int = 6000, sample_every: int = 12,
            temperature: float = 0.1) -> Calibration:
    """Play self-play games, remembering predictions, then score them against what happened."""
    import torch

    runner = ts.VectorizedBatchRunner(num_envs, base_seed)
    out = Calibration()
    # Per env: the predictions it made, from the point of view of the side that was to move.
    pending: List[List[Tuple[float, int]]] = [[] for _ in range(num_envs)]
    model.eval()

    for step in range(max_iters):
        terminals = runner.get_terminals()
        if all(terminals):
            break
        obs = np.asarray(runner.get_observations(), dtype=np.float32)
        masks = np.asarray(runner.get_action_masks())
        with torch.no_grad():
            logits, v_win, _v_vp = model(torch.from_numpy(obs).to(device),
                                         torch.from_numpy(masks).to(device))
            values = v_win.squeeze(-1).cpu().numpy()
            if temperature > 0.0:
                picks = torch.multinomial(
                    torch.softmax(logits / temperature, dim=-1), 1).squeeze(-1).cpu().numpy()
            else:
                picks = logits.argmax(dim=-1).cpu().numpy()

        if step % sample_every == 0:
            players = runner.get_decision_players()
            for i in range(num_envs):
                if terminals[i]:
                    continue
                side = int(players[i])
                if side == 0:
                    continue                  # nobody to move: nothing to take a view from
                pending[i].append((float(values[i]), side))

        runner.step_flat_all([int(a) for a in picks], auto_advance=True)

    utilities = runner.get_terminal_utilities()
    for i in range(num_envs):
        if not pending[i]:
            continue
        state = runner.get_state(i)
        ending = classify_ending(state)
        us_util = float(utilities[i])
        vp = float(state.victory_points)
        for value, side in pending[i]:
            # Both the outcome and the margin are flipped to the side that was to move, which is
            # the side `v_win` is expressed for.
            sign = 1.0 if side == int(ts.Player.US) else -1.0
            out.predicted.append(value)
            out.realized_win.append(us_util * sign)
            out.realized_vp.append(float(np.clip(vp * sign / VP_CAP, -1.0, 1.0)))
            out.ended_by.append(ending)
    return out


def calibration_buckets(cal: Calibration, edges: Sequence[float] = (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0),
                        ) -> List[Tuple[str, int, float, float]]:
    """Predicted value against realised win rate, bucketed -- the usual calibration check."""
    p = np.asarray(cal.predicted)
    y = np.asarray(cal.realized_win)
    rows: List[Tuple[str, int, float, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi) if hi < edges[-1] else (p >= lo) & (p <= hi)
        if not sel.any():
            continue
        rows.append((f"[{lo:+.1f}, {hi:+.1f})", int(sel.sum()),
                     float(p[sel].mean()), float(y[sel].mean())))
    return rows
