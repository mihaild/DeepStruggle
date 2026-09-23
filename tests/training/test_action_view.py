"""P23: which action view a checkpoint and a pool member decide in.

A network trained in the E4 view reads E4.1 masks wrongly and vice versa, and nothing fails when
the wrong one is used -- the model just plays worse. So the bookkeeping that decides the view is
tested directly: derived from the run directory for checkpoints, carried per member (through
eviction) in the opponent pool.
"""

from __future__ import annotations

import json
import os

import torch.nn as nn

from ai.training.opponent_pool import OpponentPool
from tools.lib.action_view import checkpoint_merged_influence, run_view


def _run(tmp_path, name: str, meta: dict) -> str:
    d = tmp_path / name
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps(meta))
    return str(d)


def test_an_unrecorded_run_is_e4(tmp_path) -> None:
    d = _run(tmp_path, "E4-08-03_20260101_000000", {"run_name": "E4-08-03"})
    assert run_view(d) == (False, 0)
    assert not checkpoint_merged_influence(os.path.join(d, "snapshot_5000000steps.pt"))


def test_an_e4_lineage_continued_as_e4_1_switches_at_its_resume_step(tmp_path) -> None:
    d = _run(tmp_path, "E4.1-01-03_20260101_000000",
             {"merged_influence": True, "merged_influence_from_step": 160_038_912})
    assert not checkpoint_merged_influence(os.path.join(d, "snapshot_160038912steps.pt"))
    assert checkpoint_merged_influence(os.path.join(d, "snapshot_165019648steps.pt"))
    assert checkpoint_merged_influence(os.path.join(d, "snapshot_final.pt"))
    # The weights the run started from were trained in E4.
    assert not checkpoint_merged_influence(os.path.join(d, "snapshot_0s.pt"))


def test_a_run_merged_from_its_first_step_is_merged_throughout(tmp_path) -> None:
    d = _run(tmp_path, "E4.1-02-03_20260101_000000",
             {"merged_influence": True, "merged_influence_from_step": 0})
    assert checkpoint_merged_influence(os.path.join(d, "snapshot_0s.pt"))
    assert checkpoint_merged_influence(os.path.join(d, "snapshot_5000000steps.pt"))


def test_the_pool_keeps_each_members_view_through_eviction() -> None:
    nets = [nn.Linear(2, 2) for _ in range(3)]
    pool = OpponentPool(nets, num_envs=4, frac=0.5, capacity=3, merged=[False, False, False])
    pool.steps = [10, 20, 30]
    pool.add(nn.Linear(2, 2), 40, merged=True)    # evicts an interior member
    assert len(pool.nets) == len(pool.merged) == len(pool.steps) == 3
    assert pool.merged[pool.steps.index(40)] is True
    assert all(not m for s, m in zip(pool.steps, pool.merged) if s != 40)


def test_the_current_opponents_view_follows_the_draw() -> None:
    nets = [nn.Linear(2, 2), nn.Linear(2, 2)]
    pool = OpponentPool(nets, num_envs=4, frac=0.5, merged=[False, True], seed=3)
    for _ in range(20):
        pool.start_iteration()
        assert pool.current_merged == pool.merged[pool.nets.index(pool.current)]
