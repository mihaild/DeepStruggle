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


def test_the_run_directory_a_resume_argument_names_holds_the_snapshots() -> None:
    """`--resume <dir>` must locate the snapshots, not the parent of the checkpoints tree.

    The rebuild used to compute the run directory as `dirname(abspath(resume))`, which is correct
    only when `resume` names the state *file*. Given the run *directory* -- equally valid, and
    what every recent launch passed -- it returned the PARENT, a directory holding run folders and
    no snapshot files. The glob found nothing and the pool fell back to a single frozen copy of
    the current policy.

    Nothing failed. E3-29-28_20260917_074357 trained 5M steps against that one opponent, beating
    it 99.7%, and the resulting decline was investigated as a property of search-CE training for
    two days. See research/log/P15_X4b_collapse_is_pool_starvation.md.

    So pin the property that was actually broken: for every accepted spelling of --resume, the
    directory derived from it is the one holding the snapshots.
    """
    import os
    import tempfile

    from ai.training.generic_trainer import resolve_resume

    with tempfile.TemporaryDirectory() as root:
        # mimic the real layout: a checkpoints tree whose children are run directories
        run_dir = os.path.join(root, "E9-99-01_20260101_000000")
        os.makedirs(run_dir)
        for name in ("snapshot_1000steps.pt", "snapshot_2000steps.pt", "resume_state.pt"):
            with open(os.path.join(run_dir, name), "w", encoding="utf-8") as fh:
                fh.write("x")

        for spelling in (run_dir, os.path.join(run_dir, "resume_state.pt")):
            derived = os.path.dirname(os.path.abspath(resolve_resume(spelling)))
            assert derived == os.path.abspath(run_dir), (
                f"--resume {spelling!r} derived {derived!r}, not the run directory"
            )
            snaps = [f for f in os.listdir(derived) if f.startswith("snapshot_")]
            assert len(snaps) == 2, f"expected the two snapshots, found {snaps}"

        # the parent really is snapshot-free, which is why the old expression starved the pool
        parent = os.path.dirname(os.path.abspath(run_dir))
        assert not [f for f in os.listdir(parent) if f.startswith("snapshot_")]


def test_restored_members_keep_their_steps() -> None:
    """The constructor records every seed at step 0; a restore must put the real steps back.

    It did not: after a resume the whole restored pool sat at step 0. Eviction goes by spacing,
    members at one step have zero spacing, so every new snapshot evicted a restored member until
    two were left -- a resumed pool drained to its recent end -- and the next resume, looking the
    step-0 members up on disk, found none of them and dropped them (E4-44-05's continuation at
    190M came back with 6 of 12). See research/log/P25_stress_bench.md.
    """
    p = _pool()
    p.steps = [0, 5_000_000, 10_000_000]
    blob = p.state_dict()

    restored = _pool()
    restored.load_state_dict(blob, [0, 5_000_000, 10_000_000])
    assert restored.steps == [0, 5_000_000, 10_000_000]
    assert restored.state_dict()["steps"] == [0, 5_000_000, 10_000_000], (
        "the next resume state would record the restored members at the wrong steps")


def test_a_resume_does_not_change_what_eviction_keeps() -> None:
    """Interrupting a run and resuming it must leave the same pool as running straight through."""
    def grow(pool: OpponentPool, steps: List[int]) -> None:
        for st in steps:
            pool.add(_Tiny(float(st)), st, path=f"/run/snapshot_{st}steps.pt")

    first = [s * 5_000_000 for s in range(1, 21)]
    second = [s * 5_000_000 for s in range(21, 41)]

    straight = OpponentPool([_Tiny(0.0)], num_envs=4, frac=0.5, seed=11, capacity=6)
    grow(straight, first + second)

    before = OpponentPool([_Tiny(0.0)], num_envs=4, frac=0.5, seed=11, capacity=6)
    grow(before, first)
    blob = before.state_dict()
    resumed = OpponentPool([_Tiny(0.0) for _ in blob["steps"]], num_envs=4, frac=0.5, seed=11,
                           capacity=6)
    resumed.load_state_dict(blob, blob["steps"], blob["paths"])
    grow(resumed, second)

    assert resumed.steps == straight.steps
    assert resumed.paths == straight.paths


def test_paths_are_recorded_and_restored() -> None:
    p = _pool(n=1)
    p.add(_Tiny(1.0), 5_000_000, path="/a/snapshot_5000000steps.pt")
    blob = p.state_dict()
    assert blob["paths"] == ["", "/a/snapshot_5000000steps.pt"]

    restored = _pool(n=2)
    restored.load_state_dict(blob, blob["steps"], blob["paths"])
    assert restored.paths == blob["paths"]


def test_load_state_dict_refuses_a_step_list_that_does_not_match_the_nets() -> None:
    import pytest

    p = _pool()
    with pytest.raises(ValueError):
        p.load_state_dict(p.state_dict(), [0, 5_000_000])


def test_members_of_an_earlier_leg_are_found_by_their_recorded_path() -> None:
    """A continuation lives in a new directory; members written by the leg before it do not.

    Looking members up by step in the resumed run's directory alone lost every member an earlier
    leg wrote. The recorded path finds them wherever they are.
    """
    import os
    import tempfile

    from ai.training.generic_trainer import locate_pool_members

    with tempfile.TemporaryDirectory() as root:
        leg1 = os.path.join(root, "E9-99-01_20260101_000000")
        leg2 = os.path.join(root, "E9-99-01_20260102_000000")
        for d in (leg1, leg2):
            os.makedirs(d)
        old = os.path.join(leg1, "snapshot_80000000steps.pt")
        new = os.path.join(leg2, "snapshot_120000000steps.pt")
        for f in (old, new):
            with open(f, "w", encoding="utf-8") as fh:
                fh.write("x")
        gone = os.path.join(leg1, "snapshot_40000000steps.pt")   # pruned since

        saved = {"steps": [40_000_000, 80_000_000, 120_000_000], "paths": [gone, old, new]}
        steps, paths = locate_pool_members(saved, {120_000_000: new})
        assert steps == [80_000_000, 120_000_000]
        assert paths == [old, new]

        # A state from before paths were recorded falls back to the run's own snapshots.
        steps, paths = locate_pool_members({"steps": [80_000_000, 120_000_000]},
                                           {120_000_000: new})
        assert steps == [120_000_000] and paths == [new]
