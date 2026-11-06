"""P42: the macro view on the Python side. A network grown from a base state into it plays what the base did on every
base slot and starts the until-control slots at ~e^-12; its new weights take gradient; Adam's grown state loads and
steps; the configuration comes back from the weights; a base network's logits are padded to a macro mask; the pool
and the env carry the view; a training step runs in it; and the ONNX export writes each network at its own width."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ai.models.coldwar_net_v2 import pad_to_mask
from ai.models.grow import MACRO_START_BIAS, POLICY_OUT, grow_macro_optimizer_state, grow_macro_state_dict
from ai.models.ladder_net import create_ladder_net, ladder_config_from_state_dict
from ai.training import NashPGTrainer
from ai.training.opponent_pool import OpponentPool
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import TsVectorizedEnv
from tests.training.test_token_main import CFG
from tools.export_onnx import export
from tools.lib.player_agent import OnnxAgent

BASE = ActionEncoder.BASE_ACTION_SIZE
FULL = ActionEncoder.FLAT_ACTION_SIZE


def _randomise(net: torch.nn.Module) -> None:
    """Every parameter nonzero, the zero-initialised head outputs included, so a misplaced row shows."""
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)


def _grown() -> tuple[Any, Any]:
    torch.manual_seed(0)
    base = create_ladder_net("cpu", **CFG)
    _randomise(base)
    macro = create_ladder_net("cpu", **CFG, macro_influence=True)
    macro.load_state_dict(grow_macro_state_dict(base.state_dict(), macro.state_dict(), BASE), strict=True)
    return base.eval(), macro.eval()


def _macro_positions(n_envs: int = 8, steps: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Observations and macro-view masks from random play, the US setup's included."""
    env = TsVectorizedEnv(num_envs=n_envs, base_seed=5)
    env.set_action_views(2, 2)
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(0)
    all_obs, all_masks = [np.array(obs)], [np.array(masks)]
    for _ in range(steps):
        acts = [int(rng.choice(np.flatnonzero(m))) for m in masks]
        obs, masks, *_ = env.step(acts)
        all_obs.append(np.array(obs))
        all_masks.append(np.array(masks))
    return np.concatenate(all_obs), np.concatenate(all_masks)


def test_the_grown_network_plays_the_base_slots_as_the_base_did() -> None:
    base, macro = _grown()
    obs = torch.randn(32, int(base.obs_size) if hasattr(base, "obs_size") else 3824) * 0.5
    with torch.no_grad():
        a_logits, a_win, a_vp = base(obs, None)
        b_logits, b_win, b_vp = macro(obs, None)
    assert b_logits.shape[1] == FULL
    assert torch.allclose(a_logits, b_logits[:, :BASE], atol=1e-5)
    assert torch.allclose(b_logits[:, BASE:], torch.full_like(b_logits[:, BASE:], MACRO_START_BIAS), atol=1e-5)
    assert torch.allclose(a_win, b_win, atol=1e-6) and torch.allclose(a_vp, b_vp, atol=1e-6)


def test_on_real_macro_masks_the_new_slots_start_below_the_floor() -> None:
    base, macro = _grown()
    obs, masks = _macro_positions()
    offered = masks[:, BASE:].any(axis=1)
    assert offered.any(), "random play from the start never reached an until-control slot"
    obs_t, m_t = torch.from_numpy(obs).float(), torch.from_numpy(masks)
    with torch.no_grad():
        p_macro = torch.softmax(macro(obs_t, m_t)[0], dim=-1)[:, BASE:].sum(-1)
        base_only = m_t.clone()
        base_only[:, BASE:] = 0
        a = torch.log_softmax(base(obs_t, base_only)[0], dim=-1)
        b = torch.log_softmax(macro(obs_t, base_only)[0], dim=-1)
    legal = base_only[:, :BASE] > 0
    assert a.shape[1] == FULL, "a base network's logits are padded to the mask's width"
    assert float((a[:, :BASE] - b[:, :BASE])[legal].abs().max()) < 1e-4
    assert float(p_macro.max()) < 1e-3


def test_the_new_weights_take_gradient() -> None:
    _, macro = _grown()
    macro.train()
    obs = torch.randn(16, 3824) * 0.5
    logits = macro(obs, None)[0]
    logits[:, BASE:].square().mean().backward()
    assert macro.pe_macro is not None
    assert float(macro.pe_macro[-1].weight.grad.abs().sum()) > 0
    assert float(macro.policy_head[-1].weight.grad[BASE:].abs().sum()) > 0


def test_adams_grown_state_loads_and_steps() -> None:
    torch.manual_seed(1)
    base = create_ladder_net("cpu", **CFG)
    opt = torch.optim.AdamW(base.parameters(), lr=1e-3)
    obs = torch.randn(8, 3824) * 0.5
    base(obs, None)[0].square().mean().backward()
    opt.step()
    macro = create_ladder_net("cpu", **CFG, macro_influence=True)
    names = [n for n, _ in macro.named_parameters()]
    target = macro.state_dict()
    macro.load_state_dict(grow_macro_state_dict(base.state_dict(), target, BASE), strict=True)
    opt2 = torch.optim.AdamW(macro.parameters(), lr=1e-3)
    grown = grow_macro_optimizer_state(opt.state_dict(), names, {n: int(target[n].shape[0]) for n in POLICY_OUT}, BASE)
    # A base parameter keeps its moments, the dense output's gain zero rows, the macro head starts without state.
    old_names = [n for n in names if not n.startswith("pe_macro.")]
    j, k = old_names.index("pe_country.0.weight"), names.index("pe_country.0.weight")
    assert torch.equal(grown["state"][k]["exp_avg"], opt.state_dict()["state"][j]["exp_avg"])
    w = names.index(POLICY_OUT[0])
    assert not grown["state"][w]["exp_avg"][BASE:].any()
    assert not any(names[i].startswith("pe_macro.") for i in grown["state"])
    opt2.load_state_dict(grown)
    macro(obs, None)[0].square().mean().backward()
    opt2.step()
    st = opt2.state_dict()["state"]
    assert st[w]["exp_avg"].shape == macro.policy_head[-1].weight.shape


def test_the_configuration_comes_back_from_the_weights() -> None:
    _, macro = _grown()
    assert macro.ladder_config()["macro_influence"] is True
    cfg = ladder_config_from_state_dict(macro.state_dict())
    assert cfg is not None and cfg["macro_influence"] is True
    again = create_ladder_net("cpu", **cfg).eval()
    again.load_state_dict(macro.state_dict())
    obs = torch.randn(4, 3824) * 0.5
    with torch.no_grad():
        assert torch.equal(again(obs, None)[0], macro(obs, None)[0])


def test_pad_to_mask_fills_a_base_networks_missing_slots() -> None:
    logits = torch.zeros(2, BASE)
    mask = torch.ones(2, FULL, dtype=torch.uint8)
    out = pad_to_mask(logits, mask)
    assert out.shape == (2, FULL) and bool((out[:, BASE:] < -1e8).all())


def test_the_pool_carries_each_members_view() -> None:
    nets = [create_ladder_net("cpu", **CFG) for _ in range(2)]
    pool = OpponentPool(nets, num_envs=4, frac=0.5, macro=[False, True])
    assert pool.macro == [False, True]
    pool.add(nets[0], 10, macro=True)
    assert pool.macro[-1] is True
    with pytest.raises(ValueError):
        OpponentPool(nets, num_envs=4, frac=0.5, merged=[True, False], macro=[True, False])


def test_the_env_offers_the_macro_slots_only_in_the_macro_view() -> None:
    env = TsVectorizedEnv(num_envs=2, base_seed=7)
    _, masks, _ = env.reset_all()
    assert masks.shape[1] == FULL and not masks[:, BASE:].any()
    env.set_action_views(2, 2)
    _, masks, _ = env.reset_all()
    assert masks[:, BASE:].any(), "the USSR setup offers an until-control slot (Poland, East Germany, ...)"


def test_a_training_step_in_the_macro_view() -> None:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    t: Any = NashPGTrainer(active_net=create_ladder_net(dev, **CFG, macro_influence=True),
                           env=TsVectorizedEnv(num_envs=2, base_seed=11), num_envs=2, buffer_size=16, batch_size=16,
                           lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False, device=dev, macro_influence=True)
    t.collect_rollouts()
    assert int(t.env.view_us[0]) == 2 and int(t.env.view_ussr[0]) == 2
    m = t.train_step()
    assert np.isfinite(m["loss"])


def test_the_export_writes_each_network_at_its_own_width(tmp_path: Path) -> None:
    base, macro = _grown()
    torch.save(base.state_dict(), tmp_path / "base.pt")
    torch.save(macro.state_dict(), tmp_path / "macro.pt")
    mb = export(str(tmp_path / "base.pt"), str(tmp_path / "base.onnx"), positions=16)
    mm = export(str(tmp_path / "macro.pt"), str(tmp_path / "macro.onnx"), positions=16)
    assert (mb["ts.action_size"], mb["ts.macro_influence"]) == (str(BASE), "false")
    assert (mm["ts.action_size"], mm["ts.macro_influence"]) == (str(FULL), "true")
    # A base export plays a macro-width mask: it sees the base slots and its logits are padded.
    obs, masks = _macro_positions(n_envs=2, steps=4)
    logits = OnnxAgent(str(tmp_path / "base.onnx")).logits(obs.astype(np.float32), masks)
    assert logits.shape[1] == FULL and bool((logits[:, BASE:] < -1e8).all())
