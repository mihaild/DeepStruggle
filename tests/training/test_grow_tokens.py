"""--resume-grow-tokens: token layers grown onto a trained network at identity. The grown network must
compute exactly what the saved one did, its zero weights must take gradient (so the path can learn), and
Adam's grown state must load and step."""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

import ts_engine as ts
from ai.models.grow import grow_tokens_optimizer_state, grow_tokens_state_dict
from ai.models.ladder_net import create_ladder_net
from ai.training.generic_trainer import _load_optimizer_state
from tests.training.test_ladder_tokens import SHALLOW


def _randomise(net: torch.nn.Module) -> None:
    """Every parameter nonzero, the zero-initialised head outputs included, so a misplaced column shows."""
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)


def _pair() -> Dict[str, Any]:
    torch.manual_seed(0)
    old = create_ladder_net("cpu", **SHALLOW)
    _randomise(old)
    new = create_ladder_net("cpu", **SHALLOW, token_layers=1, token_dim=16)
    grown = grow_tokens_state_dict(old.state_dict(), new.state_dict(), 16, int(getattr(new, "identity_dim", 0)))
    new.load_state_dict(grown, strict=True)
    obs = torch.randn(32, ts.OBS_SIZE) * 0.5
    return {"old": old.eval(), "new": new.eval(), "obs": obs}


def test_the_grown_network_computes_what_the_saved_one_did() -> None:
    p = _pair()
    with torch.no_grad():
        a = p["old"](p["obs"], None)
        b = p["new"](p["obs"], None)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-5), float((x - y).abs().max())


def test_the_zero_weights_take_gradient() -> None:
    p = _pair()
    net = p["new"].train()
    logits, v, _ = net(p["obs"], None)
    (logits.square().mean() + v.square().mean()).backward()
    fusion = net.fusion_in[0].weight
    assert float(fusion.grad[:, -16:].abs().sum()) > 0


def test_adams_grown_state_loads_and_steps() -> None:
    torch.manual_seed(1)
    old = create_ladder_net("cpu", **SHALLOW)
    opt = torch.optim.AdamW(old.parameters(), lr=1e-3)
    obs = torch.randn(16, ts.OBS_SIZE) * 0.5
    old(obs, None)[0].square().mean().backward()
    opt.step()
    new = create_ladder_net("cpu", **SHALLOW, token_layers=1, token_dim=16)
    names = [n for n, _ in new.named_parameters()]
    new.load_state_dict(grow_tokens_state_dict(old.state_dict(), new.state_dict(), 16, 0), strict=True)
    opt2 = torch.optim.AdamW(new.parameters(), lr=1e-3)
    opt2.load_state_dict(grow_tokens_optimizer_state(opt.state_dict(), names, 16, 0))
    new(obs, None)[0].square().mean().backward()
    opt2.step()
    st = opt2.state_dict()["state"]
    fusion_idx = names.index("fusion_in.0.weight")
    assert st[fusion_idx]["exp_avg"].shape == new.fusion_in[0].weight.shape


def test_the_chunked_token_path_is_the_same_function_with_the_same_gradients() -> None:
    torch.manual_seed(2)
    net = create_ladder_net("cpu", **SHALLOW, token_layers=1, token_dim=16)
    _randomise(net)
    obs = torch.randn(10, ts.OBS_SIZE) * 0.5
    grads = []
    outs = []
    for chunk in (0, 4):
        net.token_chunk = chunk
        net.zero_grad()
        logits, v, vp = net(obs, None)
        (logits.square().mean() + v.square().mean() + vp.square().mean()).backward()
        outs.append(torch.cat([logits.flatten(), v.flatten(), vp.flatten()]).detach())
        grads.append(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None]))
    assert torch.allclose(outs[0], outs[1], atol=1e-5)
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def _two_group_adam(net: torch.nn.Module, lr: float = 1e-3) -> torch.optim.AdamW:
    """Contiguous groups in model order, as --mup / --policy-head-lr-mult / --no-decay-norms-biases build them."""
    params = list(net.parameters())
    half = len(params) // 2
    return torch.optim.AdamW([{"params": params[:half]}, {"params": params[half:], "lr": lr * 2}], lr=lr)


def test_a_multi_group_state_grows_into_one_model_order_group() -> None:
    """Audit T-15: only the first of several saved groups was rewritten, so the grown state indexed parameters twice
    and a --mup / per-group-rate state crashed. It comes back as one model-order group, which the trainer maps onto
    its own groups (as grow_macro_optimizer_state's), and every moment lands on its parameter."""
    torch.manual_seed(1)
    old = create_ladder_net("cpu", **SHALLOW)
    opt = _two_group_adam(old)
    obs = torch.randn(16, ts.OBS_SIZE) * 0.5
    old(obs, None)[0].square().mean().backward()
    opt.step()
    new = create_ladder_net("cpu", **SHALLOW, token_layers=1, token_dim=16)
    names = [n for n, _ in new.named_parameters()]
    new.load_state_dict(grow_tokens_state_dict(old.state_dict(), new.state_dict(), 16, 0), strict=True)
    grown = grow_tokens_optimizer_state(opt.state_dict(), names, 16, 0)
    assert len(grown["param_groups"]) == 1 and grown["param_groups"][0]["params"] == list(range(len(names)))
    opt2 = _two_group_adam(new)
    _load_optimizer_state(opt2, grown)
    old_names = [n for n, _ in old.named_parameters()]
    saved = opt.state_dict()["state"]
    loaded = opt2.state_dict()["state"]
    same = 0
    for j, name in enumerate(old_names):
        if j in saved and saved[j]["exp_avg"].shape == loaded[names.index(name)]["exp_avg"].shape:
            assert torch.equal(loaded[names.index(name)]["exp_avg"], saved[j]["exp_avg"]), name
            same += 1
    assert same > 10
    new(obs, None)[0].square().mean().backward()
    opt2.step()


def test_a_state_not_in_model_order_is_refused() -> None:
    torch.manual_seed(1)
    old = create_ladder_net("cpu", **SHALLOW)
    params = list(old.parameters())
    opt = torch.optim.AdamW([{"params": params[1:]}, {"params": params[:1]}], lr=1e-3)
    new = create_ladder_net("cpu", **SHALLOW, token_layers=1, token_dim=16)
    names = [n for n, _ in new.named_parameters()]
    sd = opt.state_dict()
    sd["param_groups"][0]["params"], sd["param_groups"][1]["params"] = (sd["param_groups"][1]["params"],
                                                                          sd["param_groups"][0]["params"])
    with pytest.raises(ValueError, match="model order"):
        grow_tokens_optimizer_state(sd, names, 16, 0)
