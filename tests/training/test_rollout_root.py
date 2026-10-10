"""ai/search/rollout_root.py: legal and reproducible picks, the network's own move with one
candidate, rollouts that spend rows only with a horizon, and the spec."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.search.rollout_root import (RolloutConfig, RolloutRoot, RolloutRootAgent, parse_rule,
                                   rollout_spec_config)
from bindings.action_encoder import ActionEncoder
from tools.lib.player_agent import load_agent


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    m.eval()
    return m


def _states(n: int = 12, advance: int = 60) -> List[ts.GameState]:
    runner = ts.VectorizedBatchRunner(n, 4242)
    for _ in range(advance):
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(np.flatnonzero(m)[0]) if m.any() else 211 for m in masks], True)
    return [runner.get_state(i) for i in range(n) if not ts.Engine.is_terminal(runner.get_state(i))]


@pytest.mark.parametrize("rule", ["argmax", "z2", "kl0.05"])
def test_picks_are_legal_and_reproducible(rule: str) -> None:
    states = _states()
    cfg = RolloutConfig(k=3, worlds=4, horizon=1, rule=rule, seed=3)
    a = RolloutRoot(_model(), cfg).choose(states)
    b = RolloutRoot(_model(), cfg).choose(states)
    assert a == b
    for st, x in zip(states, a):
        assert np.asarray(ActionEncoder.get_legal_mask(st))[x]


def test_one_candidate_is_the_networks_move() -> None:
    states = _states()
    one = RolloutRoot(_model(), RolloutConfig(k=1, worlds=2, horizon=0, rule="argmax")).choose(states)
    raw = RolloutRoot(_model(), RolloutConfig(k=1, worlds=2, horizon=0, rule="argmax"))
    lg, masks, _ = raw._forward(states)
    assert one == [int(np.argmax(np.where(m.astype(bool), l, -np.inf))) for l, m in zip(lg, masks)]


def test_a_horizon_spends_rollout_rows() -> None:
    states = _states()
    r0 = RolloutRoot(_model(), RolloutConfig(k=2, worlds=2, horizon=0))
    r0.choose(states)
    r2 = RolloutRoot(_model(), RolloutConfig(k=2, worlds=2, horizon=2))
    r2.choose(states)
    assert r2.rows > r0.rows > 0


def test_the_spec(tmp_path: Path) -> None:
    path = tmp_path / "m.pt"
    torch.save(_model().state_dict(), str(path))
    _, cfg, label = rollout_spec_config(f"rollout:{path}:4:32:2:z2")
    assert (cfg.k, cfg.worlds, cfg.horizon, cfg.rule, label) == (4, 32, 2, "z2", "rollout-k4-w32-h2-z2")
    agent = load_agent(f"rollout:{path}:2:2:0:argmax", device="cpu")
    assert isinstance(agent, RolloutRootAgent)
    st = _states(4)
    assert len(agent.select_actions_batch(st)) == len(st)
    with pytest.raises(ValueError):
        parse_rule("best")
