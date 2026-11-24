"""A side-split network (A38 --ladder-side-heads, A39 --ladder-side-nets) at one copy's cost per row in the compiled
update (ai/training/side_split.py): the batch is split by side outside the compiled region and each side's rows,
padded to a bucket, run through its own copy only.

The switch must leave running arms' dynamics alone, so the split path is held to the path it replaces -- every row
through both copies, each keeping its own side's output (`static_routing`) -- on the same weights and batch: the
outputs, the losses and every parameter's gradient, for mixed, one-sided and odd-sized batches. All on the CPU; the
compiled one goes through dynamo with the aot_eager backend, which traces exactly what inductor would be handed."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import pytest
import torch
import torch.nn as nn

from ai.models import ladder_net as ln
from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.training import NashPGTrainer
from ai.training.side_split import SplitForward, bucket_size, padded_rows
from bindings.ts_env import TsVectorizedEnv
from tests.training.test_side_split import CFG, OPTIONS, _apart, _batch

METHODS = ("forward", "forward_with_risk", "forward_with_value_logits", "forward_value")
BATCHES = ("mixed", "odd", "all_us", "all_ussr")


def _net(option: str, **over: Any) -> LadderNet:
    torch.manual_seed(0)
    return _apart(create_ladder_net("cpu", **{**CFG, option: True, **over}).eval())


def _rows(kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    obs, masks = _batch(64)
    us = obs[:, ln.ColdWarNetV2.GLOBAL_OFFSET + ln.I_AM_US_GLOBAL] > 0.5
    pick = {"mixed": torch.arange(64), "odd": torch.arange(37), "all_us": torch.nonzero(us).squeeze(1),
            "all_ussr": torch.nonzero(~us).squeeze(1)}[kind]
    assert pick.numel() > 2
    return obs[pick], masks[pick]


def _call(net: LadderNet, method: str, obs: torch.Tensor, masks: torch.Tensor) -> Any:
    fn = getattr(net, method)
    return fn(obs) if method == "forward_value" else fn(obs, masks)


def _static(net: LadderNet, method: str, obs: torch.Tensor, masks: torch.Tensor,
            monkeypatch: pytest.MonkeyPatch) -> Any:
    """The path the split replaces: every row through both copies, each keeping its own side's output."""
    with monkeypatch.context() as m:
        m.setattr(ln, "static_routing", lambda: True)
        return _call(net, method, obs, masks)


def _split(net: LadderNet, method: str, obs: torch.Tensor, masks: torch.Tensor, **kw: Any) -> Any:
    f = SplitForward(net, method, kw.pop("compile", None), **kw)
    return f(obs) if method == "forward_value" else f(obs, masks)


def _flat(out: Any) -> List[torch.Tensor]:
    return [t for t in (out if isinstance(out, tuple) else (out,)) if t is not None]


def _close(a: Any, b: Any, atol: float = 1e-5) -> None:
    fa, fb = _flat(a), _flat(b)
    assert len(fa) == len(fb)
    for x, y in zip(fa, fb):
        assert x.shape == y.shape
        assert torch.allclose(x, y, atol=atol, rtol=1e-5), float((x - y).abs().max())


def _loss(out: Any, masks: torch.Tensor) -> torch.Tensor:
    """Every output in the loss, policy and values alike, with row weights that tell rows apart."""
    w = torch.linspace(0.5, 1.5, masks.shape[0])
    if isinstance(out, torch.Tensor):
        return (out.squeeze(-1) * w).sum()
    logits = torch.where(masks.bool(), out[0], torch.zeros_like(out[0]))
    total = (logits.pow(2).sum(-1) * w).sum()
    for t in out[1:]:
        if t is not None:
            total = total + (t.reshape(t.shape[0], -1).sum(-1) * w).sum()
    return total


def _grads(net: LadderNet, run: Callable[[], Any], masks: torch.Tensor) -> Dict[str, torch.Tensor]:
    net.train()
    net.zero_grad(set_to_none=True)
    loss = _loss(run(), masks)
    loss.backward()
    return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for n, p in net.named_parameters()}


def _same_grads(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], ussr_learns: bool = True) -> None:
    """The same gradient on every parameter, to float32 summation order (relative to the tensor's largest entry: a
    shared weight's gradient sums rows in another order); the USSR copy's nonzero when the batch holds USSR rows."""
    assert a.keys() == b.keys()
    for n in a:
        err, scale = float((a[n] - b[n]).abs().max()), float(b[n].abs().max())
        assert err <= 1e-5 * max(scale, 1e-2), (n, err, scale)
    ussr = [float(g.abs().max()) for n, g in a.items() if n.startswith(("ussr_net.", "ussr_heads."))]
    assert (max(ussr) > 0) == ussr_learns


def test_buckets_and_padding() -> None:
    assert [bucket_size(n, 256) for n in (1, 256, 257, 8000)] == [256, 256, 512, 8192]
    idx = torch.tensor([4, 9, 2])
    assert padded_rows(idx, 5).tolist() == [4, 9, 2, 4, 4]
    assert padded_rows(idx, 3) is idx
    with pytest.raises(ValueError):
        padded_rows(idx, 2)
    with pytest.raises(ValueError):
        bucket_size(3, 0)


@pytest.mark.parametrize("kind", BATCHES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("option", OPTIONS)
def test_the_split_outputs_are_the_static_paths(option: str, method: str, kind: str,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    net = _net(option)
    obs, masks = _rows(kind)
    with torch.no_grad():
        want = _static(net, method, obs, masks, monkeypatch)
        _close(_split(net, method, obs, masks, bucket=8), want)
        _close(_split(net, method, obs, masks, bucket=48), want)     # every side padded, some past the batch
    assert net._fixed_side is None


@pytest.mark.parametrize("kind", ("mixed", "odd", "all_ussr"))
@pytest.mark.parametrize("method", ("forward", "forward_with_value_logits"))
@pytest.mark.parametrize("option", OPTIONS)
def test_the_split_gradients_are_the_static_paths(option: str, method: str, kind: str,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    net = _net(option)
    obs, masks = _rows(kind)
    want = _grads(net, lambda: _static(net, method, obs, masks, monkeypatch), masks)
    got = _grads(net, lambda: _split(net, method, obs, masks, bucket=16), masks)
    _same_grads(got, want)


@pytest.mark.parametrize("option", OPTIONS)
def test_compiled_split_is_the_static_path(option: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through dynamo, with the batch dimension dynamic: the side each compiled part reads (`fixed_side`, the side's
    network) is the one it was given, for both sides and every bucket, outputs and gradients alike."""
    torch._dynamo.reset()
    net = _net(option)
    f = SplitForward(net, "forward", {"backend": "aot_eager"}, bucket=8)
    for kind in ("mixed", "odd", "all_us", "all_ussr"):
        obs, masks = _rows(kind)
        with torch.no_grad():
            _close(f(obs, masks), _static(net, "forward", obs, masks, monkeypatch))
        want = _grads(net, lambda: _static(net, "forward", obs, masks, monkeypatch), masks)
        got = _grads(net, lambda: f(obs, masks), masks)
        _same_grads(got, want, ussr_learns=kind != "all_us")
    net.eval()


def test_split_forward_refuses_what_it_cannot_do() -> None:
    plain = create_ladder_net("cpu", **CFG)
    with pytest.raises(ValueError, match="side-split"):
        SplitForward(plain, "forward", None)
    with pytest.raises(ValueError, match="per-side"):
        SplitForward(_net("side_nets"), "forward_aux", None)
    with pytest.raises(ValueError, match="at least 2"):
        SplitForward(_net("side_nets"), "forward", None, bucket=1)
    with pytest.raises(ValueError):
        _net("side_heads").side_net(2)


def _trainer(net: nn.Module, **kw: Any) -> NashPGTrainer:
    env = TsVectorizedEnv(num_envs=8, base_seed=123)
    return NashPGTrainer(active_net=net, env=env, num_envs=8, buffer_size=16, batch_size=64, device="cpu", **kw)


@pytest.mark.parametrize("option", OPTIONS)
def test_the_compiled_update_splits_a_side_split_network(option: str) -> None:
    t = _trainer(_net(option), compile_update="default")
    for name in ("forward", "forward_with_risk", "forward_with_value_logits"):
        f = t._update_method(t.active_net, name)
        assert isinstance(f, SplitForward) and f.net is t.active_net and f.method == name
        assert t._update_method(t.active_net, name) is f
    assert isinstance(t._update_net(t.active_net), SplitForward)
    assert t._update_net(t.reference_net) is not t._update_net(t.active_net)
    off = _trainer(_net(option))
    assert off._update_net(off.active_net) is off.active_net
