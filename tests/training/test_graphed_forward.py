"""CUDA-graph replays of the rollout forward match eager.

`GraphedForward` replays the kernels eager would run, so for one network at one batch its outputs
must be bitwise identical to eager -- including after in-place optimiser steps, which the graph
must see. The rollout test runs a seeded collection twice, with and without graphs: without an
opponent pool the buffers must be identical; with one, only the opponent's rows are computed at a
different batch size (all envs rather than its own rows), so only last-bit differences in its
logits, and nothing a sampled action could see, are allowed.

GPU only: CUDA graphs have no CPU counterpart.
"""

from __future__ import annotations

import pytest
import torch

from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.training.graphed_forward import GraphCache, GraphedForward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _inputs(batch: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    import ts_engine as ts
    obs = torch.randn(batch, ts.OBS_SIZE, device="cuda", generator=gen)
    mask = (torch.rand(batch, 220, device="cuda", generator=gen) > 0.6).to(torch.uint8)
    mask[:, 0] = 1
    return obs, mask


def test_replay_is_bitwise_eager_and_follows_in_place_weight_updates() -> None:
    import ts_engine as ts
    torch.manual_seed(0)
    net = create_coldwar_net_v2().cuda().eval()
    g = GraphedForward(net, 64, ts.OBS_SIZE, 220, torch.device("cuda"))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    gen = torch.Generator(device="cuda").manual_seed(1)
    for _ in range(3):
        obs, mask = _inputs(64, gen)
        with torch.no_grad():
            eager = net(obs, mask)
        g.load(obs, mask)
        g.replay()
        assert all(torch.equal(a, b) for a, b in zip(eager, g.outputs()))
        net.train()
        _, v_win, v_vp = net(obs, mask)
        loss = v_win.float().mean() + v_vp.float().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        net.eval()


def test_a_replaced_parameter_makes_the_graph_stale_and_the_cache_recaptures() -> None:
    import ts_engine as ts
    net = create_coldwar_net_v2().cuda().eval()
    cache = GraphCache(16, ts.OBS_SIZE, 220, torch.device("cuda"))
    first = cache.get(net)
    assert cache.get(net) is first
    p = next(net.parameters())
    p.data = p.data.clone()           # a new tensor at a new address
    assert first.stale()
    assert cache.get(net) is not first


def _collect(graphs: bool, pool: bool) -> dict[str, torch.Tensor]:
    import numpy as np
    from ai.training.nash_pg import NashPGTrainer
    from ai.training.opponent_pool import OpponentPool
    from bindings.ts_env import TsVectorizedEnv
    torch.manual_seed(0)
    np.random.seed(0)
    env = TsVectorizedEnv(num_envs=32, base_seed=1234)
    tr = NashPGTrainer(active_net=create_coldwar_net_v2(), env=env, num_envs=32,
                       buffer_size=16, device="cuda", cuda_graphs=graphs)
    if pool:
        torch.manual_seed(7)
        tr.opponent_pool = OpponentPool([create_coldwar_net_v2().cuda()], num_envs=32, frac=0.5,
                                        seed=5)
    torch.manual_seed(3)
    tr.collect_rollouts()
    b = tr.buffer
    return {"actions": b.actions.clone(), "log_probs": b.log_probs.clone(),
            "values_win": b.values_win.clone(), "obs": b.obs.clone()}


def test_rollout_with_graphs_equals_eager_without_a_pool() -> None:
    a, b = _collect(False, False), _collect(True, False)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_rollout_with_graphs_matches_eager_with_a_pool() -> None:
    a, b = _collect(False, True), _collect(True, True)
    assert torch.equal(a["actions"], b["actions"])
    assert torch.equal(a["obs"], b["obs"])
    assert torch.equal(a["values_win"], b["values_win"])
    assert torch.allclose(a["log_probs"], b["log_probs"], rtol=0, atol=1e-5)


def test_two_graphs_replayed_concurrently_on_two_streams_stay_bitwise_eager() -> None:
    """The rollout replays the learner's and the opponent's graphs at once. Captured on a shared
    capture stream they shared a cuBLAS workspace and both came out wrong."""
    import ts_engine as ts
    torch.manual_seed(0)
    a = create_coldwar_net_v2().cuda().eval()
    torch.manual_seed(7)
    b = create_coldwar_net_v2().cuda().eval()
    cache = GraphCache(32, ts.OBS_SIZE, 220, torch.device("cuda"))
    gen = torch.Generator(device="cuda").manual_seed(1)
    side, main = torch.cuda.Stream(), torch.cuda.current_stream()
    for _ in range(5):
        obs, mask = _inputs(32, gen)
        with torch.no_grad():
            ea, eb = a(obs, mask)[0], b(obs, mask)[0]
        ga = cache.get(a)
        ga.load(obs, mask)
        gb = cache.get(b)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            gb.load(obs, mask)
            gb.replay()
        ga.replay()
        main.wait_stream(side)
        assert torch.equal(ga.static_out[0], ea) and torch.equal(gb.static_out[0], eb)
