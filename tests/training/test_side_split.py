"""Per-side parameters (owner, 2026-11-23): --ladder-side-heads gives every head after the trunk a USSR copy, and
--ladder-side-nets a whole second network, each row routed by the side it is observed from (I_AM_US, global 61).

A batch of mixed sides computes, row for row, what each side's copy computes alone; a gradient on one side's rows never
reaches the other side's copy; the split network starts as the shared one; and the copies travel through everything a
LadderNet goes through -- the checkpoint and its detection, muP, the head L2 term and reset, the resume, the sphere
projection of a weight average."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn

import ts_engine as ts
from ai.models import ladder_net as ln
from ai.models.coldwar_net_v2 import create_like, policy_head_norms
from ai.models.head_registry import SIDE_HEAD_MODULES, copy_prefixes, head_of, is_side_split, refuse_side_split
from ai.models.hyperspherical import SPHERE_WEIGHT_KEY
from ai.models.ladder_net import I_AM_US_GLOBAL, LadderNet, create_ladder_net, ladder_config_from_state_dict
from ai.models.mup import base_shapes_for, mup_lr_mults, policy_head_lr_mults, zero_output_layers
from ai.training.generic_trainer import load_resume_state, reset_policy_head, save_resume_state
from ai.training.head_l2 import head_init_params
from ai.training.train import _ladder_config, build_parser
from bindings.ts_env import observe
from tools.export_onnx import export
from tools.lib.player_agent import NeuralAgent
from tools.scripts import launch_flags

#: A small A8: every head the options must copy -- country head with context and norm, region attention, branch and
#: play-mode heads, aux heads, the prior gain.
CFG: Dict[str, Any] = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=32,
                           card_self_attention=False, cross_attention=False, per_entity_heads=16, head_context=True,
                           head_static=True, head_entities="country", head_center=True, identity_dim=0,
                           drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False,
                           card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False,
                           branch_head=True, play_mode_head=True, head_norm=True, aux_heads=True, prior_gain=True,
                           region_attn_dim=16, region_attn_heads=4)
OPTIONS = ("side_heads", "side_nets")
COPY = {"side_heads": "ussr_heads.", "side_nets": "ussr_net."}


def _net(option: str | None, seed: int = 0, **over: Any) -> LadderNet:
    torch.manual_seed(seed)
    return create_ladder_net("cpu", **{**CFG, **({option: True} if option else {}), **over}).eval()


def _apart(net: LadderNet) -> LadderNet:
    """`net` with its USSR copy moved away from the US one, so a row read by the wrong copy shows."""
    with torch.no_grad():
        g = torch.Generator().manual_seed(5)
        for name, p in net.named_parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return net


def _batch(n: int = 48) -> tuple[torch.Tensor, torch.Tensor]:
    runner = ts.VectorizedBatchRunner(n, 4321)
    rng = np.random.default_rng(3)
    for _ in range(40):
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks], True)
    obs = torch.from_numpy(np.asarray(runner.get_observations(), dtype=np.float32))
    us = obs[:, ln.ColdWarNetV2.GLOBAL_OFFSET + I_AM_US_GLOBAL] > 0.5
    assert 0 < int(us.sum()) < n, "the batch must hold both sides"
    return obs, torch.from_numpy(np.asarray(runner.get_action_masks()))


def _alone(net: LadderNet, option: str) -> tuple[LadderNet, nn.Module]:
    """Each side's copy as a network of its own: (the US copy, the USSR copy)."""
    sd = net.state_dict()
    plain = {k: v for k, v in sd.items() if not k.startswith(COPY[option])}
    us = _net(None)
    us.load_state_dict(plain)
    if option == "side_nets":
        assert net.ussr_net is not None
        return us, net.ussr_net
    ussr = _net(None)
    ussr.load_state_dict({**plain, **{k[len(COPY[option]):]: v for k, v in sd.items() if k.startswith(COPY[option])}})
    return us, ussr


def _outputs(net: nn.Module, obs: torch.Tensor, masks: torch.Tensor) -> List[torch.Tensor]:
    with torch.no_grad():
        out = list(net(obs, masks)) + list(getattr(net, "forward_with_risk")(obs, masks))
        out += [getattr(net, "forward_value")(obs), getattr(net, "defcon_risk")(obs),
                getattr(net, "forward_prior_gain")(obs), *getattr(net, "forward_aux")(obs)]
    return out


def test_i_am_us_is_global_61_in_the_engine() -> None:
    s = ts.GameState()
    ts.Engine.init_game(s, 7)
    for player, want in ((ts.Player.US, 1.0), (ts.Player.USSR, 0.0)):
        assert float(observe(s, player, 0)[ln.ColdWarNetV2.GLOBAL_OFFSET + I_AM_US_GLOBAL]) == want


@pytest.mark.parametrize("option", OPTIONS)
def test_a_new_split_network_is_the_shared_one(option: str) -> None:
    obs, masks = _batch()
    for a, b in zip(_outputs(_net(None), obs, masks), _outputs(_net(option), obs, masks)):
        assert torch.allclose(a, b, atol=1e-5, rtol=0)       # each side's rows a smaller GEMM: equal to rounding


@pytest.mark.parametrize("option", OPTIONS)
def test_a_mixed_batch_is_each_side_through_its_own_copy(option: str) -> None:
    net = _apart(_net(option))
    obs, masks = _batch()
    us_rows = net.is_us(obs)
    us, ussr = _alone(net, option)
    got, want_us, want_ussr = (_outputs(m, obs, masks) for m in (net, us, ussr))
    for g, a, b in zip(got, want_us, want_ussr):
        assert torch.allclose(g[us_rows], a[us_rows], atol=1e-5)
        assert torch.allclose(g[~us_rows], b[~us_rows], atol=1e-5)
    # ... and the two copies do differ, so a row read by the wrong one would have shown
    assert not torch.allclose(want_us[0][~us_rows], want_ussr[0][~us_rows], atol=1e-3)


@pytest.mark.parametrize("option", OPTIONS)
def test_every_row_through_both_copies_is_the_same_function(option: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Where shapes must be static (compile, CUDA graphs, ONNX) the batch is not split: every row runs through both
    copies and keeps its own side's output."""
    net = _apart(_net(option))
    obs, masks = _batch()
    split = _outputs(net, obs, masks)
    monkeypatch.setattr(ln, "static_routing", lambda: True)
    for a, b in zip(split, _outputs(net, obs, masks)):
        assert torch.allclose(a, b, atol=1e-5)


def _grads_after(net: LadderNet, obs: torch.Tensor, masks: torch.Tensor, rows: torch.Tensor) -> Dict[str, Any]:
    net.train()
    net.zero_grad(set_to_none=True)
    logits, v_win, v_vp = net(obs, masks)
    legal = masks.bool()
    loss = (torch.where(legal, logits, torch.zeros_like(logits))[rows].pow(2).sum()
            + v_win[rows].sum() + v_vp[rows].sum() + net.forward_aux(obs)[0][rows].sum())
    loss.backward()
    return {n: p.grad for n, p in net.named_parameters()}


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("option", OPTIONS)
def test_one_sides_gradient_never_reaches_the_other_sides_copy(option: str, static: bool,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    if static:
        monkeypatch.setattr(ln, "static_routing", lambda: True)
    net = _apart(_net(option, res_dropout=0.0))
    obs, masks = _batch()
    us = net.is_us(obs)
    for rows, other, own in ((us, COPY[option], ""), (~us, "", COPY[option])):
        grads = _grads_after(net, obs, masks, rows)
        copied = [n for n in grads if n.startswith(COPY[option])]
        assert copied
        if other:          # the US rows: the USSR copy gets nothing
            for n in copied:
                assert grads[n] is None or float(grads[n].abs().max()) == 0.0, n
            assert any(grads[n] is not None and float(grads[n].abs().max()) > 0 for n in grads
                       if n.startswith("pe_country."))
        else:              # the USSR rows: the US copy of every copied module gets nothing
            originals = [n[len(own):] for n in copied]
            if option == "side_heads":
                for n in originals:
                    assert grads[n] is None or float(grads[n].abs().max()) == 0.0, n
                # ... while the shared trunk learns from them
                assert float(grads["fusion_in.0.weight"].abs().max()) > 0
            else:
                for n in originals:
                    assert grads[n] is None or float(grads[n].abs().max()) == 0.0, n
            assert any(grads[n] is not None and float(grads[n].abs().max()) > 0 for n in copied)


def test_side_heads_copy_every_head_and_nothing_else() -> None:
    net = _net("side_heads")
    assert net.ussr_heads is not None
    copied = set(net.ussr_heads.keys())
    assert copied == {n for n in SIDE_HEAD_MODULES if isinstance(getattr(net, n, None), nn.Module)}
    assert {"policy_head", "pe_trunk", "pe_country", "region_attn", "branch_head_net", "play_mode_head_net",
            "val_win_head", "val_vp_head", "head_norms", "defcon_risk_head", "aux_own_head", "aux_vp_head",
            "prior_gain_head"} <= copied
    assert not any(n.startswith(("lad_", "fusion_in", "global_proj")) for n in copied)


@pytest.mark.parametrize("option", OPTIONS)
def test_the_checkpoint_names_its_split_and_loads_everywhere(option: str, tmp_path: Any) -> None:
    net = _apart(_net(option))
    cfg = ladder_config_from_state_dict(net.state_dict())
    assert cfg is not None and cfg.get(option) is True and cfg == net.ladder_config()
    again = create_ladder_net("cpu", **cfg)
    again.load_state_dict(net.state_dict())                  # strict: the detected architecture is exact
    like = create_like(net)
    assert isinstance(like, LadderNet) and like.ladder_config() == net.ladder_config()
    path = os.path.join(str(tmp_path), "snapshot.pt")
    torch.save(net.state_dict(), path)
    agent = NeuralAgent.from_checkpoint(path, device="cpu")
    obs, masks = _batch()
    for a, b in zip(_outputs(net, obs, masks), _outputs(agent.model, obs, masks)):
        assert torch.equal(a, b)
    assert is_side_split(agent.model) and copy_prefixes(agent.model) == ("", COPY[option])
    with pytest.raises(ValueError, match="side-split"):
        refuse_side_split(agent.model, "this probe")


@pytest.mark.parametrize("option", OPTIONS)
def test_mup_gives_each_copy_its_originals_multiplier_and_zeroes_both_readouts(option: str) -> None:
    net = _net(option, entity_proj_dim=64, hidden_dim=128)
    base = base_shapes_for(net.ladder_config(), 32, 64, lambda **kw: create_ladder_net("cpu", **kw))
    mults = mup_lr_mults(net, base, 64)
    copied = [n for n, _ in net.named_parameters() if n.startswith(COPY[option])]
    assert any(n in mults for n in copied)
    for n in copied:
        assert mults.get(n) == mults.get(n[len(COPY[option]):]), n
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1)
    done = zero_output_layers(net)
    for name in ("policy_head.3", "val_win_head.3", "pe_country.2"):
        assert name in done and COPY[option] + name in done
        out = net.get_submodule(COPY[option] + name)
        assert isinstance(out, nn.Linear) and float(out.weight.detach().abs().sum()) == 0.0
    rates = policy_head_lr_mults(net, 0.5)
    for n, _ in net.named_parameters():
        assert (n in rates) == (head_of(n) in ("policy_head", "pe_trunk", "pe_country", "region_attn",
                                               "branch_head_net", "play_mode_head_net", "head_norms")), n


@pytest.mark.parametrize("option", OPTIONS)
def test_the_head_tools_cover_both_copies(option: str) -> None:
    net = _net(option)
    init = head_init_params(net)
    assert any(n.startswith(COPY[option] + "pe_country.") for n in init)
    assert any(n.startswith(COPY[option] + "val_win_head.") for n in init)
    assert not any(head_of(n).startswith(("lad_", "fusion_in")) for n in init)
    norms = policy_head_norms(net)
    assert "pe_country" in norms and COPY[option] + "pe_country" in norms
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    done = reset_policy_head(net, torch.optim.AdamW(net.parameters(), lr=1e-3))
    assert "pe_country" in done and COPY[option] + "pe_country" in done and COPY[option] + "head_norms" in done


class _FakeTrainer:
    def __init__(self, model: nn.Module) -> None:
        self.device = torch.device("cpu")
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.reference_net = create_like(model)
        self.total_env_steps = 0
        self.total_iterations = 0


def _train(net: LadderNet, trainer: _FakeTrainer, steps: int) -> None:
    obs, masks = _batch(16)
    for _ in range(steps):
        logits, v_win, _v = net(obs, masks)
        loss = torch.where(masks.bool(), logits, torch.zeros_like(logits)).pow(2).mean() + v_win.pow(2).mean()
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        trainer.total_env_steps += 1000


@pytest.mark.parametrize("option", OPTIONS)
def test_a_split_run_resumes_exactly(option: str, tmp_path: Any) -> None:
    net = _net(option).train()
    trainer = _FakeTrainer(net)
    _train(net, trainer, 2)
    path = os.path.join(str(tmp_path), "resume.pt")
    save_resume_state(path, net, trainer, iteration=2, total_env_steps=trainer.total_env_steps, elapsed_seconds=1.0)
    fresh = _net(option, seed=9).train()
    fresh_trainer = _FakeTrainer(fresh)
    load_resume_state(path, fresh, fresh_trainer)
    for k, v in net.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k]), k
    names = [n for n, _ in net.named_parameters()]
    i = next(j for j, n in enumerate(names) if n.startswith(COPY[option]))
    a = trainer.optimizer.state[list(net.parameters())[i]]["exp_avg_sq"]
    b = fresh_trainer.optimizer.state[list(fresh.parameters())[i]]["exp_avg_sq"]
    assert torch.equal(a, b)
    # A shared state is not a split one: the copies would start from nothing, so it refuses.
    shared = _net(None).train()
    shared_trainer = _FakeTrainer(shared)
    save_resume_state(path, shared, shared_trainer, iteration=0, total_env_steps=0, elapsed_seconds=0.0)
    with pytest.raises(RuntimeError, match="missing"):
        load_resume_state(path, _net(option), _FakeTrainer(_net(option)))
    with pytest.raises(ValueError, match="side-split"):
        load_resume_state(path, _net(option), _FakeTrainer(_net(option)), grow_tokens=True)


def test_the_options_are_exclusive_and_named_on_the_command_line() -> None:
    with pytest.raises(ValueError, match="alternatives"):
        _net("side_heads", side_nets=True)
    parser = build_parser()
    base = ["--arch", "ladder", "--ladder-input-mode", "grouped", "--ladder-aggregation", "flatten",
            "--ladder-entity-dim", "16", "--ladder-entity-proj-dim", "256",
            "--per-entity-heads", "64", "--ladder-head-entities", "country",
            "--ladder-head-context", "--ladder-head-static", "--ladder-hidden-dim", "480", "--ladder-res-blocks", "0",
            "--no-ladder-card-lookup", "--drop-static"]
    for flag, key in (("--ladder-side-heads", "side_heads"), ("--ladder-side-nets", "side_nets")):
        cfg = _ladder_config(parser.parse_args(base + [flag]))
        assert cfg is not None and cfg[key] is True
        assert launch_flags._LADDER[key] == "ladder_" + key
    cfg = _ladder_config(parser.parse_args(base))
    assert cfg is not None and "side_heads" not in cfg and "side_nets" not in cfg


def test_a_weight_average_reprojects_the_ussr_networks_sphere() -> None:
    assert SPHERE_WEIGHT_KEY.match("ussr_net.sphere.blocks.0.w1.weight")
    assert SPHERE_WEIGHT_KEY.match("sphere.blocks.1.w2.weight")
    assert not SPHERE_WEIGHT_KEY.match("ussr_heads.sphere.blocks.0.w1.weight")


@pytest.mark.parametrize("option", OPTIONS)
def test_it_exports_to_onnx_for_the_page(option: str, tmp_path: Any) -> None:
    """The browser workbench's file: every row through both copies (static shapes), checked by export() itself against
    torch on real positions of both sides."""
    net = _apart(_net(option, obs_features=0))
    torch.save(net.state_dict(), str(tmp_path / "net.pt"))
    export(str(tmp_path / "net.pt"), str(tmp_path / "net.onnx"), positions=16)
