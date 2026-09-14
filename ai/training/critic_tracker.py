"""Live tracking of whether the critic still discriminates.

The failure this exists to catch: once one side finds a strategy the other has not answered,
outcomes become predictable, the critic has no reason to discriminate, and it degenerates into
predicting the base rate. Advantages then vanish and both policies freeze. Measured on E3-17-22
the critic went from 79.2% outcome accuracy against a 53.9% base rate at 80M to 87.5% against an
87.3% base rate at 160M -- a 0.2pp edge -- and on E3-15-22 it ended *below* the base rate.

**Why not `explained_variance`.** The GAE return is built as `G = A + V`, so `G - V = A` exactly
and `EV = 1 - Var(A)/Var(G)`. A critic whose advantages collapse scores near 1.0 by construction;
EV rose to 0.996 across precisely the window in which the critic became useless. It cannot see
this failure because it is a restatement of it.

**Why AUC and not accuracy or correlation.** The collapse is *accompanied by* a base-rate shift --
the US win rate fell from 54% to 13% over the same window. Accuracy is dominated by the base rate
(a useless critic scores 87% when one side wins 87% of the time), and point-biserial correlation
with a binary outcome is itself bounded by how balanced that outcome is. Both would move for
reasons unrelated to critic quality. **ROC AUC is invariant to the base rate** and to any monotone
rescaling of the value, so 0.5 means useless and 1.0 means perfect however lopsided the games get.

AUC only measures *ranking*, though, and GAE subtracts `V` rather than ranking it, so a
well-ordered but miscalibrated critic still produces bad advantages. The Brier skill score against
the base-rate predictor is logged beside it to cover that: **BSS <= 0 is the alarm** -- it means
the critic is no better than a constant at the base rate.

**Sampling.** One state per game per turn, taken at the turn's first decision. Averaging over
every step instead would mix turn-1 states (hard to call) with turn-9 states (nearly decided) in a
proportion that shifts as game length changes during training -- so the metric would drift for
reasons that have nothing to do with the critic. A sample is held until its game ends and the
winner is known.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank-sum identity, with ties taking their average rank.

    `labels` is +1 for the positive class and -1 otherwise. Returns NaN when only one class is
    present, because AUC is undefined there -- a run whose games are all won by one side must
    report "unknown", not a number that looks like a score.
    """
    pos = labels > 0
    n_pos = int(pos.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # Average the ranks inside each tie group, or a constant critic would score 1.0 or 0.0.
    s_sorted = scores[order]
    i = 0
    while i < s_sorted.size:
        j = i + 1
        while j < s_sorted.size and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier_skill(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier skill against the constant base-rate forecast. <= 0 means no better than constant."""
    y = (labels > 0).astype(np.float64)
    if y.size == 0:
        return float("nan")
    base = y.mean()
    bs = float(np.mean((probs - y) ** 2))
    bs_ref = float(np.mean((base - y) ** 2))
    if bs_ref < 1e-12:
        return float("nan")
    return 1.0 - bs / bs_ref


class CriticTracker:
    """Holds one sample per game per turn until that game's winner is known.

    Tier 1: the per-step work is an integer compare and an occasional append, and the AUC is one
    sort over at most `capacity` resolved samples once per iteration.
    """

    def __init__(self, num_envs: int, capacity: int = 20_000,
                 headline_turn: int = 3) -> None:
        self.num_envs = num_envs
        self.headline_turn = headline_turn
        #: (value, turn) awaiting an outcome, per env.
        self._pending: List[List[Tuple[float, int]]] = [[] for _ in range(num_envs)]
        #: Last turn already sampled per env, so each turn contributes once.
        self._last_turn = np.full(num_envs, -1, dtype=np.int32)
        self._resolved: Deque[Tuple[float, int, float]] = deque(maxlen=capacity)

    def reset_env(self, i: int) -> None:
        self._pending[i].clear()
        self._last_turn[i] = -1

    def observe(self, values_us: np.ndarray, turns: np.ndarray, alive: np.ndarray) -> None:
        """Record one sample per env on the step where its turn first changes.

        `values_us` must be the value from a *fixed* perspective (US), not the acting player's,
        or the sign would flip with whoever happens to be moving and the labels would not line up.
        """
        for i in range(self.num_envs):
            if not alive[i]:
                continue
            t = int(turns[i])
            if t == self._last_turn[i]:
                continue
            self._last_turn[i] = t
            self._pending[i].append((float(values_us[i]), t))

    def resolve(self, env_index: int, us_won: Optional[bool]) -> None:
        """Attach the outcome to every pending sample from this env, then clear it."""
        if us_won is not None:
            label = 1.0 if us_won else -1.0
            for value, turn in self._pending[env_index]:
                self._resolved.append((value, turn, label))
        self.reset_env(env_index)

    def metrics(self, min_samples: int = 200) -> Dict[str, float]:
        if len(self._resolved) < min_samples:
            return {}
        v = np.fromiter((r[0] for r in self._resolved), dtype=np.float64,
                        count=len(self._resolved))
        t = np.fromiter((r[1] for r in self._resolved), dtype=np.int32,
                        count=len(self._resolved))
        y = np.fromiter((r[2] for r in self._resolved), dtype=np.float64,
                        count=len(self._resolved))

        # v_win is in [-1, 1] from the US perspective; map to a probability for the Brier term.
        probs = np.clip((v + 1.0) / 2.0, 0.0, 1.0)
        out = {
            "critic_auc": roc_auc(v, y),
            "critic_brier_skill": brier_skill(probs, y),
            "critic_base_rate": float(max((y > 0).mean(), 1.0 - (y > 0).mean())),
            "critic_samples": float(len(self._resolved)),
        }
        m = t == self.headline_turn
        if m.sum() >= min_samples // 4:
            # A fixed turn, so the number is comparable across runs whose games differ in length.
            out[f"critic_auc_turn{self.headline_turn}"] = roc_auc(v[m], y[m])
        return out
