"""Agreement with human play, counted so that the order of an influence placement does not matter.

A card played for Operations spends its points one at a time, and the engine asks a separate
`POINT_NODE` question for each. The human's log records the order they happened to be written in,
which carries no decision: placing two Influence in Angola and one in Zaire is the same play in any
order. Scoring each point against the exact index the human's sequence happened to hold marks the
model wrong for reordering a play it agrees with. Roughly a third of human decisions sit in
multi-point plays, though the correction turns out to be worth about half a point rather than
several (research/experiments.md §9.11): teacher forcing means a disagreement is usually about
which countries, not about their order.

The same holds for every event that distributes or removes several points across countries:
Decolonization, De-Stalinization, Colonial Rear Guards, Ussuri River Skirmish, Puppet Governments,
COMECON, Marshall Plan, The Reformer; and for removals, Socialist Governments and East European
Unrest. Nothing here is card-specific -- a run of `POINT_NODE` decisions inside one play is one
decision with several parts, whatever produced it.

**Not every run of point decisions is order-free, and the exceptions are blacklisted rather than
the rest whitelisted** -- spreading Influence is the ordinary case and a new card that spreads it
should not have to be remembered here. Order matters wherever the board changes between points:

* **realignment rolls**, where each roll is made against the influence the last one left, so a
  different order is a different sequence of odds;
* **coups**, for the same reason -- and in particular **Che**, whose second coup is only offered
  if the first removed influence, so the pair is a sequence and not a set.

Those are scored strictly, exactly as before.

**How an order-free group is scored.** The model is teacher-forced along the human's trajectory, so
its own earlier choices cannot take it somewhere the human never went. At each step its argmax
counts as agreement if that country is still in the multiset of countries the human put points into
and has not already been matched; the match is then consumed. A group of *n* points therefore
contributes *n* comparisons exactly as before, and the total is directly comparable to the ordered
figure -- what changes is only that a permutation no longer costs anything.

Placing two points in one country is two entries in the multiset, so a model that agrees on the
country but not on how many points went there still loses the difference.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import ts_engine as ts

MASK_BITS_ = 212


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


CHE = 107


def order_matters(state: ts.GameState) -> bool:
    """Is the sequence of these point decisions itself a decision?

    A blacklist, not a whitelist: spreading Influence is the ordinary case, and a card that
    spreads it in some new way should be order-free without anyone having to add it here.
    """
    ctx = state.ctx()
    if ctx.op_mode in (ts.OpMode.COUP, ts.OpMode.REALIGN):
        return True
    # Che's second coup is offered only if the first removed Influence, so the two are a
    # sequence rather than a set even though both arrive as point decisions.
    return int(ctx.resolving_card) == CHE or int(ctx.pending_op_card) == CHE


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
        is_point = (state.ctx().decision_type == ts.DecisionType.POINT_NODE
                    and not order_matters(state))
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
                 human_actions: Union[Sequence[int], np.ndarray],
                 model_actions: Union[Sequence[int], np.ndarray]) -> AgreementResult:
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
            if int(model_actions[i]) == int(human_actions[i]):
                out.ordered_hits += 1
        # Order-insensitive: spend each model choice against the human's remaining multiset.
        remaining: Counter = Counter(int(human_actions[i]) for i in group)
        for i in group:
            guess = int(model_actions[i])
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


def groups_from_play_ids(play: Sequence[int]) -> List[List[int]]:
    """Turn a per-sample play id into index runs. Ids are contiguous within a play."""
    groups: List[List[int]] = []
    current: List[int] = []
    last: Optional[int] = None
    for i, pid in enumerate(play):
        if last is not None and pid == last:
            current.append(i)
        else:
            if current:
                groups.append(current)
            current = [i]
        last = pid
    if current:
        groups.append(current)
    return groups


def evaluate_dataset(model: Any, dataset_path: str, device: Any,
                     max_samples: Optional[int] = None) -> AgreementResult:
    """Agreement of `model` with the demonstrations in `dataset_path`.

    Takes either dataset: a directory is the human corpus, which stores the play grouping as a
    column, and a file is the self-play set, whose loader recovers it while replaying. The pass
    is deliberately *not* shuffled -- a play's points have to stay together to be scored as one.
    """
    import os

    if os.path.isdir(dataset_path):
        from ai.training.human_corpus_dataset import HumanCorpusDataset

        ds = HumanCorpusDataset(dataset_path)
        n = len(ds) if max_samples is None else min(len(ds), max_samples)
        obs = np.asarray(ds._column("obs")[:n], dtype=np.float32)
        mask = np.unpackbits(np.asarray(ds._column("mask")[:n]), axis=1)[:, :MASK_BITS_]
        action = np.asarray(ds._column("action")[:n], dtype=np.int64)
        groups = groups_from_play_ids([int(v) for v in ds._column("play")[:n]])
    else:
        from ai.training.warmup_dataset_loader import WarmupDataset

        rows = []
        plays: List[int] = []
        for obs_i, mask_i, act_i, _w, _v, play_i in WarmupDataset(dataset_path).stream_with_plays():
            rows.append((obs_i, mask_i, act_i))
            plays.append(play_i)
            if max_samples is not None and len(rows) >= max_samples:
                break
        if not rows:
            return AgreementResult()
        obs = np.stack([r[0] for r in rows]).astype(np.float32)
        mask = np.stack([r[1] for r in rows]).astype(np.uint8)
        action = np.asarray([r[2] for r in rows], dtype=np.int64)
        groups = groups_from_play_ids(plays)

    return score_groups(groups, action, model_argmax(model, obs, mask, device))
