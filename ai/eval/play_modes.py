"""How each card gets spent, per side.

At a `SELECT_PLAY_MODE` node the actions are EVENT / OPS / SPACE / PASS, shared by every card.
There is no per-card row in the output there and, without identity embeddings, no card identity
in the input either -- so the mode distribution for two cards with the same feature vector is not
merely similar, it is *the same function*, identical to six decimal places.

That makes this the sharpest available test of what identity embeddings bought. The decision that
matters is exactly the one that was blind: event-or-ops for a card of your own or a neutral, and
ops-or-space for an opponent's.

Two readings:

* **Per (card, side)** -- the table a person reads when comparing two models. 110 x 2 x 4 cells,
  which is why this is an arm-evaluation probe and not a training metric: cheap to compute,
  unreadable as 880 TensorBoard curves.
* **Within-collision-group divergence** -- the scalar. Averaged Jensen-Shannon divergence between
  the mode distributions of cards that share a feature vector. A model that cannot see card
  identity must score ~0 here, whatever else it does, because the same input produces the same
  distribution. Anything above 0 is identity being used.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

import ts_engine as ts

MODES = ("EVENT", "OPS", "SPACE", "PASS")


@dataclass
class PlayModes:
    #: (card_id, side) -> counts per mode
    counts: Dict[Tuple[int, str], np.ndarray] = field(default_factory=dict)
    games: int = 0

    def note(self, card_id: int, side: str, mode_idx: int) -> None:
        key = (int(card_id), side)
        if key not in self.counts:
            self.counts[key] = np.zeros(len(MODES), dtype=np.int64)
        self.counts[key][mode_idx] += 1

    def distribution(self, card_id: int, side: str) -> np.ndarray | None:
        c = self.counts.get((int(card_id), side))
        if c is None or c.sum() < 1:
            return None
        return c / c.sum()

    def table(self, min_plays: int = 20) -> List[Tuple[str, str, int, np.ndarray]]:
        rows = []
        for (cid, side), c in sorted(self.counts.items(), key=lambda kv: -kv[1].sum()):
            if c.sum() < min_plays:
                continue
            rows.append((ts.CardData.get_card_name(cid), side, int(c.sum()), c / c.sum()))
        return rows


def _js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits. 0 when the two distributions are identical."""
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / np.maximum(b[nz], 1e-12))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def collision_groups() -> List[List[int]]:
    """Cards sharing one feature vector, so only identity can separate their mode choice."""
    from ai.eval.positions import PositionBuilder
    from ai.models.coldwar_net_v2 import ColdWarNetV2

    st = PositionBuilder(hand=tuple(range(1, 111)), side=ts.Player.USSR,
                         defcon=3, turn=5).build()
    o = np.asarray(ts.extract_observation(st, ts.Player.USSR), dtype=np.float32)
    off, cf = ColdWarNetV2.BOARD_FEATURES * 84, ColdWarNetV2.CARD_FEATURES
    groups: Dict[Any, List[int]] = collections.defaultdict(list)
    for cid in range(1, 111):
        groups[tuple(o[off + (cid - 1) * cf: off + cid * cf].tolist())].append(cid)
    return [g for g in groups.values() if len(g) > 1]


def within_group_divergence(pm: PlayModes, min_plays: int = 20) -> Tuple[float, int]:
    """Mean pairwise JS divergence inside collision groups, and how many pairs it averaged.

    Zero is the null: cards that look identical get the same mode distribution, which is what a
    model without card identity is *obliged* to produce.
    """
    vals: List[float] = []
    for group in collision_groups():
        for side in ("US", "USSR"):
            dists = [(c, pm.distribution(c, side)) for c in group]
            dists = [(c, d) for c, d in dists
                     if d is not None and pm.counts[(c, side)].sum() >= min_plays]
            for i in range(len(dists)):
                for j in range(i + 1, len(dists)):
                    vals.append(_js(dists[i][1], dists[j][1]))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def measure_play_modes(model: Any, num_games: int = 400, base_seed: int = 404_000,
                       temperature: float = 0.1, max_iters: int = 20_000) -> PlayModes:
    """Count how each card is spent, by the side spending it."""
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    env = TsVectorizedEnv(num_envs=num_games, base_seed=base_seed)
    obs, masks, _ = env.reset_all()
    out = PlayModes()
    pending: List[Dict[Any, int]] = [dict() for _ in range(num_games)]
    finished = [False] * num_games

    try:
        for _ in range(max_iters):
            if all(finished):
                break
            with torch.no_grad():
                acts, *_ = model.sample_action(
                    torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device),
                    torch.from_numpy(np.asarray(masks)).to(device), temperature=temperature)
            a = acts.cpu().numpy().astype(np.int64)

            for i in range(num_games):
                if finished[i]:
                    continue
                st = env.runner.get_state(i)
                if ts.Engine.is_terminal(st):
                    continue
                ctx = st.ctx()
                p = ctx.decision_player if ctx.decision_player != ts.Player.NONE \
                    else st.phasing_player
                if p == ts.Player.NONE:
                    continue
                ma = ts.ActionMask.decode_flat_action(st, int(a[i]))
                dt = int(ma.decision_type)
                if dt == 1 and int(ctx.pending_op_card) == 0:
                    pending[i][p] = int(ma.primary_id)
                elif dt == 2:
                    card = pending[i].get(p, 0)
                    if 1 <= card <= 110 and 0 <= int(ma.primary_id) < len(MODES):
                        side = "US" if p == ts.Player.US else "USSR"
                        out.note(card, side, int(ma.primary_id))

            obs, masks, _, dones, info = env.step(a)
            for i, r in enumerate(info["ending_reasons"]):
                if r and not finished[i]:
                    finished[i] = True
                    out.games += 1
    finally:
        if was_training:
            model.train()
    return out
