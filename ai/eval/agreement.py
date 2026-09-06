"""Agreement with human play, counted so that the order of an influence placement does not matter.

A card played for Operations spends its points one at a time, and the engine asks a separate
`POINT_NODE` question for each. The human's log records the order they happened to be written in,
which carries no decision: placing two Influence in Angola and one in Zaire is the same play in any
order. Scoring each point against the exact index the human's sequence happened to hold marks the
model wrong for reordering a play it agrees with, and the effect is not small -- most Operations
plays are multi-point.

The same holds for every event that distributes or removes several points across countries:
Decolonization, De-Stalinization, Colonial Rear Guards, Ussuri River Skirmish, Puppet Governments,
COMECON, Marshall Plan, The Reformer; and for removals, Socialist Governments and East European
Unrest. Nothing here is card-specific -- a run of `POINT_NODE` decisions inside one play is one
decision with several parts, whatever produced it.

**How a group is scored.** The model is teacher-forced along the human's trajectory, so its own
earlier choices cannot take it somewhere the human never went. At each step its argmax counts as
agreement if that country is still in the multiset of countries the human put points into and has
not already been matched; the match is then consumed. A group of *n* points therefore contributes
*n* comparisons exactly as before, and the total is directly comparable to the ordered figure --
what changes is only that a permutation no longer costs anything.

Placing two points in one country is two entries in the multiset, so a model that agrees on the
country but not on how many points went there still loses the difference.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import ts_engine as ts


@dataclass
class AgreementResult:
    decisions: int = 0
    ordered_hits: int = 0
    unordered_hits: int = 0
    grouped_decisions: int = 0
    groups: int = 0

    @property
    def ordered(self) -> float:
        return 100.0 * self.ordered_hits / max(1, self.decisions)

    @property
    def unordered(self) -> float:
        return 100.0 * self.unordered_hits / max(1, self.decisions)


def _play_key(state: ts.GameState, mover: ts.Player) -> Tuple[Any, ...]:
    """What makes two consecutive point decisions part of the same play."""
    ctx = state.ctx()
    return (int(state.turn), int(state.action_round), int(mover),
            int(ctx.resolving_card), int(ctx.pending_op_card))


def group_point_runs(states: Sequence[ts.GameState],
                     movers: Sequence[Any]) -> List[List[int]]:
    """Consecutive POINT_NODE decisions belonging to one play, as index runs.

    Singletons are returned too, so every decision belongs to exactly one group and the caller
    can score them uniformly.
    """
    groups: List[List[int]] = []
    current: List[int] = []
    key: Optional[Tuple[Any, ...]] = None

    for i, state in enumerate(states):
        is_point = state.ctx().decision_type == ts.DecisionType.POINT_NODE
        this_key = _play_key(state, movers[i]) if is_point else None
        if is_point and this_key == key and current:
            current.append(i)
            continue
        if current:
            groups.append(current)
        current = [i]
        key = this_key if is_point else None
    if current:
        groups.append(current)
    return groups


def score_groups(groups: Sequence[Sequence[int]],
                 human_actions: Sequence[int],
                 model_actions: Sequence[int]) -> AgreementResult:
    """Ordered and order-insensitive agreement over pre-grouped decisions.

    `model_actions[i]` is the model's argmax at the position the human actually reached, so both
    numbers are over the same decisions and differ only in how a group is credited.
    """
    out = AgreementResult()
    for group in groups:
        out.decisions += len(group)
        if len(group) > 1:
            out.groups += 1
            out.grouped_decisions += len(group)
        for i in group:
            if model_actions[i] == human_actions[i]:
                out.ordered_hits += 1
        # Order-insensitive: spend each model choice against the human's remaining multiset.
        remaining: Counter = Counter(human_actions[i] for i in group)
        for i in group:
            guess = model_actions[i]
            if remaining[guess] > 0:
                remaining[guess] -= 1
                out.unordered_hits += 1
    return out


def model_argmax(model: Any, obs: np.ndarray, mask: np.ndarray,
                 device: Any, batch: int = 2048) -> np.ndarray:
    """The model's greedy choice at each position, teacher-forced along the human's trajectory."""
    import torch

    out = np.empty(len(obs), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(obs), batch):
            o = torch.from_numpy(obs[start:start + batch]).to(device)
            m = torch.from_numpy(mask[start:start + batch]).to(device)
            out[start:start + batch] = model(o, m)[0].argmax(dim=-1).cpu().numpy()
    return out
