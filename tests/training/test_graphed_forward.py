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
    # Checked once per rollout: within one the cache does not walk the parameters again...
    assert cache.get(net) is first
    # ...and at the next rollout it does, and recaptures.
    cache.begin_rollout()
    second = cache.get(net)
    assert second is not first and not second.stale()


def test_the_cache_checks_staleness_once_per_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    import ts_engine as ts
    net = create_coldwar_net_v2().cuda().eval()
    cache = GraphCache(16, ts.OBS_SIZE, 220, torch.device("cuda"))
    calls = []
    orig = GraphedForward.stale
    monkeypatch.setattr(GraphedForward, "stale", lambda self: calls.append(1) or orig(self))
    cache.get(net)                    # a fresh capture needs no check
    for _ in range(3):
        cache.begin_rollout()
        for _ in range(5):
            cache.get(net)
    assert len(calls) == 3


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


def test_a_cache_captures_every_graph_on_one_stream_and_replays_stay_bitwise_eager() -> None:
    """Every graph of a cache is captured on the cache's one stream, so all share one cuBLAS
    workspace, allocated eagerly by the warmup on that stream. Shared, they must still be
    replayed one after the other (E4-42, E4-44 deadlocked when they were not), and so replayed
    they stay exact."""
    import ts_engine as ts
    torch.manual_seed(0)
    a = create_coldwar_net_v2().cuda().eval()
    torch.manual_seed(7)
    b = create_coldwar_net_v2().cuda().eval()
    cache = GraphCache(32, ts.OBS_SIZE, 220, torch.device("cuda"))
    ga = cache.get(a)
    for _ in range(40):            # wrap torch's 32-stream pool: it must not matter any more
        torch.cuda.Stream()
    gb = cache.get(b)
    assert gb._capture_stream.cuda_stream == ga._capture_stream.cuda_stream == cache._stream.cuda_stream
    gen = torch.Generator(device="cuda").manual_seed(1)
    for _ in range(5):
        obs, mask = _inputs(32, gen)
        with torch.no_grad():
            ea, eb = a(obs, mask)[0], b(obs, mask)[0]
        ga.load(obs, mask)
        ga.replay()
        gb.load(obs, mask)
        gb.replay()
        assert torch.equal(ga.static_out[0], ea) and torch.equal(gb.static_out[0], eb)


def test_dropping_a_graph_leaves_the_others_exact_after_its_memory_is_reused() -> None:
    """Drop graphs, churn the allocator over the memory they released, and the survivor must still
    replay exactly and write nothing outside its own buffers. This does NOT reproduce the
    launch failure it is named after -- it passes on the pre-fix code too; that failure needed the
    full training loop (data/logs/graphstress/harness2.py). The mechanism is pinned by
    test_warmup_runs_on_the_capture_stream below."""
    import ts_engine as ts
    nets = []
    for seed in range(6):
        torch.manual_seed(seed)
        nets.append(create_coldwar_net_v2().cuda().eval())
    cache = GraphCache(64, ts.OBS_SIZE, 220, torch.device("cuda"))
    for n in nets:
        cache.get(n)
    survivor = nets[-1]
    cache.retain([survivor])
    assert len(cache._graphs) == 1
    junk = [torch.full((1 << 20,), float(i), device="cuda") for i in range(64)]   # reuse freed memory
    gen = torch.Generator(device="cuda").manual_seed(3)
    g = cache.get(survivor)
    for _ in range(5):
        obs, mask = _inputs(64, gen)
        with torch.no_grad():
            ref = survivor(obs, mask)[0]
        g.load(obs, mask)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(g.static_out[0], ref)
    assert all(bool((t == float(i)).all()) for i, t in enumerate(junk)), (
        "a replay wrote into memory the dropped graphs had released")


def test_warmup_runs_on_the_capture_stream() -> None:
    """The fix itself. cuBLAS allocates a stream's workspace on its first call there; warming up on
    the capture stream makes that call eager, so the workspace comes from the ordinary allocator,
    not from a graph's private pool that is freed when the graph is dropped. A warmup on any other
    stream would leave the first cuBLAS call inside the capture again."""
    import ts_engine as ts
    torch.manual_seed(0)
    net = create_coldwar_net_v2().cuda().eval()
    cache = GraphCache(16, ts.OBS_SIZE, 220, torch.device("cuda"))
    entered = []
    real_stream = torch.cuda.stream

    def spy(s):  # type: ignore[no-untyped-def]
        entered.append(s)
        return real_stream(s)

    torch.cuda.stream = spy  # type: ignore[assignment]
    try:
        cache.get(net)
    finally:
        torch.cuda.stream = real_stream  # type: ignore[assignment]
    assert entered, "the warmup did not run under a stream context"
    assert all(e is not None and e.cuda_stream == cache._stream.cuda_stream for e in entered), (
        "warmup ran on a stream other than the cache's capture stream")
