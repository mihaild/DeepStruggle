"""compute_gae replayed as a CUDA graph is bitwise the eager recursion.

The graph runs the same kernels on the same inputs, so every output -- advantages (after
normalisation), both returns, the DEFCON-risk label and the per-seat statistics -- must be equal,
not close. Checked over several rollouts through one captured graph (its inputs are static copies
and its outputs are rebound after normalisation, which is where a stale binding would show), and
across the scalar settings, which force a re-capture.

GPU only: CUDA graphs have no CPU counterpart.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple

import pytest
import torch

from ai.training.rollout_buffer import RolloutBuffer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

T, N = 32, 48


class _Last(NamedTuple):
    last_v_win: torch.Tensor
    last_v_vp: torch.Tensor
    last_dones: torch.Tensor
    last_players: torch.Tensor


def _fill(b: RolloutBuffer, seed: int) -> _Last:
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = b.device

    def r(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=g, device=dev)

    b.players.copy_(torch.where(r(T, N) < 0.5, 1, -1).to(torch.int8))
    b.dones.copy_(r(T, N) < 0.06)
    b.turns.copy_((torch.arange(T, device=dev).unsqueeze(1) // 5 + (r(1, N) * 3).long()).to(torch.int8))
    b.rewards.copy_(torch.where(b.dones, torch.where(r(T, N) < 0.5, 1.0, -1.0), 0.0))
    b.values_win.copy_(r(T, N) * 2 - 1)
    b.vps.copy_(r(T, N) * 40 - 20)
    b.held_scoring_us.copy_(r(T, N) < 0.05)
    b.held_scoring_ussr.copy_(r(T, N) < 0.05)
    b.defcon_blunder.copy_(torch.where(r(T, N) < 0.05, torch.where(r(T, N) < 0.5, 1, -1), 0)
                           .to(torch.int8))
    b.next_values_own.copy_(r(T, N) * 2 - 1)
    b.has_next_values_own = True
    return _Last(r(N) * 2 - 1, r(N) * 2 - 1, r(N) < 0.05,
                 torch.where(r(N) < 0.5, 1, -1).to(torch.int8))


def _outputs(b: RolloutBuffer) -> Dict[str, Any]:
    return {"advantages": b.advantages.clone(), "returns_win": b.returns_win.clone(),
            "returns_vp": b.returns_vp.clone(), "risk": b.defcon_risk_target.clone(),
            "raw_std": b.raw_advantage_std, "side": dict(b.side_advantage)}


@pytest.mark.parametrize("kw", [
    {},
    {"blunder_window": False},
    {"same_perspective_bootstrap": True},
    {"per_player_gae": True},
    {"slice_turn_boundaries": True, "gae_lambda": 0.95},
])
def test_graphed_gae_is_bitwise_eager_over_several_rollouts(kw: Dict[str, Any]) -> None:
    eager = RolloutBuffer(T, N, device="cuda")
    graphed = RolloutBuffer(T, N, device="cuda")
    graphed.graph_gae = True
    for seed in range(4):
        for b in (eager, graphed):
            last = _fill(b, seed)
            b.compute_gae(*last, **kw)
        a, g = _outputs(eager), _outputs(graphed)
        for k in ("advantages", "returns_win", "returns_vp", "risk"):
            assert torch.equal(a[k], g[k]), (seed, k)
        assert a["raw_std"] == g["raw_std"] and a["side"] == g["side"]
    assert graphed._gae_graph is not None


def test_a_changed_setting_recaptures() -> None:
    b = RolloutBuffer(T, N, device="cuda")
    b.graph_gae = True
    b.compute_gae(*_fill(b, 0))
    first = b._gae_graph
    b.compute_gae(*_fill(b, 1))
    assert b._gae_graph is first
    b.compute_gae(*_fill(b, 2), gae_lambda=0.9)
    assert b._gae_graph is not first
