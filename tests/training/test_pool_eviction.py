"""P47: the own pool's eviction rule (--opponent-pool-eviction): 'spread' as every run before it, 'recent' a window of
the newest members, 'unpinned' the spread rule with the oldest member a candidate too."""

from __future__ import annotations

from typing import Any, List

import pytest
import torch
import torch.nn as nn

from ai.training.opponent_pool import OpponentPool


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(2))

    def forward(self, x: Any) -> Any:  # pragma: no cover - never called
        return x


def _fill(eviction: str, steps: List[int], capacity: int = 4) -> OpponentPool:
    p = OpponentPool([_Tiny()], num_envs=4, frac=1.0, seed=0, capacity=capacity, eviction=eviction)
    p.steps[0] = steps[0]
    for s in steps[1:]:
        p.add(_Tiny(), s)
    return p


def test_spread_keeps_the_oldest_forever() -> None:
    p = _fill("spread", [0, 100, 200, 300, 400, 500, 600, 700, 800])
    assert min(p.steps) == 0 and max(p.steps) == 800 and len(p.steps) == 4


def test_recent_is_a_window_of_the_newest() -> None:
    p = _fill("recent", [0, 100, 200, 300, 400, 500, 600, 700, 800])
    assert sorted(p.steps) == [500, 600, 700, 800]


def test_unpinned_lets_the_oldest_go_when_its_gap_is_the_smallest() -> None:
    # members 0, 10, 500, 1000 + a new 1500: the oldest's doubled gap (20) is the smallest span lost
    p = _fill("unpinned", [0, 10, 500, 1000, 1500])
    assert 0 not in p.steps and 10 in p.steps and 1500 in p.steps
    # a spread pool in the same state drops the interior member 10 instead and keeps 0
    q = _fill("spread", [0, 10, 500, 1000, 1500])
    assert 0 in q.steps and 10 not in q.steps


def test_unpinned_keeps_a_wide_oldest_gap() -> None:
    # geometric spacing: the oldest covers a wide span alone, so an interior member goes
    p = _fill("unpinned", [0, 1000, 1500, 1750, 1900])
    assert 0 in p.steps and len(p.steps) == 4


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="eviction"):
        OpponentPool([_Tiny()], num_envs=4, frac=1.0, eviction="oldest")
