"""A resumed run must keep the opponent pool it had. Losing it is a change to the experiment.

The pool is the mechanism that keeps a trailing side with winnable games, so a run that resumes
without it is not the run it claims to continue -- and the effect is invisible: nothing in the
logs says the pool came back empty, and it takes tens of millions of steps to refill. An earlier
fix made resume rebuild the pool by sampling snapshots evenly; that lands on a *different* pool
that merely looks similar, because eviction is by spacing and depends on the order members
arrived, and it discards the win/game record a PFSP draw is computed from.

These tests pin the round trip: what the pool held, and what it knew about each member.
"""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn

from ai.training.opponent_pool import OpponentPool


class _Tiny(nn.Module):
    def __init__(self, k: float) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.full((2,), k))

    def forward(self, x: Any) -> Any:  # pragma: no cover - never called
        return x


def _pool(n: int = 3, num_envs: int = 4, **kw: Any) -> OpponentPool:
    return OpponentPool([_Tiny(float(i)) for i in range(n)], num_envs=num_envs,
                        frac=0.5, seed=11, **kw)


def test_state_dict_records_members_by_step_not_index() -> None:
    """Indices shift under eviction; the step count is what identifies a snapshot on disk."""
    p = _pool()
    p.steps = [0, 5_000_000, 10_000_000]
    blob = p.state_dict()
    assert blob["steps"] == [0, 5_000_000, 10_000_000]
    assert len(blob["ids"]) == 3
    assert "wins" in blob and "games" in blob


def test_statistics_survive_the_round_trip() -> None:
    p = _pool()
    p.steps = [0, 5_000_000, 10_000_000]
    oid = p.ids[1]
    p.wins[oid] = 7.5
    p.games[oid] = 20.0
    blob = p.state_dict()

    restored = _pool()
    restored.load_state_dict(blob, [0, 5_000_000, 10_000_000])

    assert restored.ids == p.ids, "ids must be preserved, or statistics re-key onto the wrong member"
    assert restored.wins[oid] == 7.5
    assert restored.games[oid] == 20.0
    assert restored.win_rate(oid) == p.win_rate(oid)


def test_a_pruned_snapshot_drops_only_its_own_record() -> None:
    """A partially recoverable pool keeps what it can rather than being zeroed or mis-keyed."""
    p = _pool()
    p.steps = [0, 5_000_000, 10_000_000]
    keep_id, lost_id = p.ids[0], p.ids[1]
    p.wins[keep_id], p.games[keep_id] = 3.0, 10.0
    p.wins[lost_id], p.games[lost_id] = 9.0, 10.0
    blob = p.state_dict()

    restored = _pool(n=2)
    restored.load_state_dict(blob, [0, 10_000_000])   # the 5M snapshot is gone

    assert keep_id in restored.games and restored.wins[keep_id] == 3.0
    assert lost_id not in restored.games, (
        "a dropped member's record must not survive; it would otherwise attach to whichever "
        "member later takes that id")
    assert len(restored.ids) == 2


def test_an_unrecorded_member_gets_a_fresh_id() -> None:
    """Loading more snapshots than were recorded must not collide ids."""
    p = _pool()
    p.steps = [0, 5_000_000, 10_000_000]
    blob = p.state_dict()

    restored = _pool(n=4)
    restored.load_state_dict(blob, [0, 5_000_000, 10_000_000, 15_000_000])

    assert len(set(restored.ids)) == 4, "ids collided"
    assert restored._next_id > max(restored.ids)


def test_resume_state_carries_the_pool() -> None:
    """The save/load pair, not just the pool object: the blob has to reach the file."""
    import tempfile
    import os

    from ai.training.generic_trainer import save_resume_state

    class _Trainer:
        def __init__(self) -> None:
            self.optimizer = torch.optim.SGD(_Tiny(0.0).parameters(), lr=0.1)
            self.reference_net = _Tiny(0.0)
            self.total_iterations = 3
            self.device = torch.device("cpu")
            self.opponent_pool = _pool()
            self.opponent_pool.steps = [0, 5_000_000]

    tr = _Trainer()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "resume_state.pt")
        save_resume_state(path, _Tiny(0.0), tr, iteration=3, total_env_steps=5_000_000,
                          elapsed_seconds=1.0, seed=7)
        blob = torch.load(path, map_location="cpu", weights_only=False)

    assert blob.get("opponent_pool") is not None, (
        "the resume state does not carry the pool, so a resumed run cannot restore it")
    assert blob["opponent_pool"]["steps"] == [0, 5_000_000]


def test_no_pool_is_recorded_as_none_not_omitted() -> None:
    """An unpooled run must be distinguishable from a state written before pools were carried."""
    import tempfile
    import os

    from ai.training.generic_trainer import save_resume_state

    class _Trainer:
        def __init__(self) -> None:
            self.optimizer = torch.optim.SGD(_Tiny(0.0).parameters(), lr=0.1)
            self.reference_net = _Tiny(0.0)
            self.total_iterations = 0
            self.device = torch.device("cpu")
            self.opponent_pool = None

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "resume_state.pt")
        save_resume_state(path, _Tiny(0.0), _Trainer(), iteration=0, total_env_steps=0,
                          elapsed_seconds=0.0, seed=1)
        blob = torch.load(path, map_location="cpu", weights_only=False)

    assert "opponent_pool" in blob
    assert blob["opponent_pool"] is None
