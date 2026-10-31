"""P41 J5: the token transformer as the main path. No grouped path, the same 220 outputs, every token type reaching the
logits, the play-mode and branch heads reading the active card's token, chunking unchanged, the configuration recovered
from the weights, A33 untouched, a training step, and the ONNX export the workbench runs."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ai.models.ladder_net import ACTIVE_CARD_SLOT, create_ladder_net, ladder_config_from_state_dict
from ai.models.mup import base_shapes_for, mup_lr_mults
from ai.models.token_transformer import TokenBlock
from ai.training import NashPGTrainer
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import TsVectorizedEnv
from tools.export_onnx import export

A8: dict[str, Any] = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, card_self_attention=False, cross_attention=False,
          per_entity_heads=16, head_context=True, head_static=True, head_entities="country", head_center=True,
          identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False,
          card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False,
          branch_head=True, play_mode_head=True)
CFG: dict[str, Any] = dict(A8, entity_proj_dim=0, token_layers=2, token_dim=32, token_main=True, token_heads=4, token_mlp_mult=2)


def _net() -> Any:
    torch.manual_seed(0)
    return create_ladder_net("cpu", **CFG).eval()


def _obs(n: int = 3) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(n, 3824)


def test_no_grouped_path_and_the_same_outputs() -> None:
    net = _net()
    names = [k for k, _ in net.named_parameters()]
    assert not any(k.startswith(("lad_board", "lad_card", "global_proj", "tok_enc")) for k in names)
    logits, v_win, v_vp = net(_obs(), torch.ones(3, ActionEncoder.FLAT_ACTION_SIZE, dtype=torch.uint8))
    assert logits.shape == (3, ActionEncoder.FLAT_ACTION_SIZE) and v_win.shape == (3, 1) and v_vp.shape == (3, 1)


@pytest.mark.parametrize("start,stop,what", [(0, 26, "a country"), (2184, 2198, "a card"), (3724, 3824, "the globals")])
def test_every_token_type_reaches_the_value_through_the_trunk(start: int, stop: int, what: str) -> None:
    net = _net()
    o = _obs()
    other = o.clone()
    other[:, start:stop] = torch.rand(3, stop - start) + 2.0
    with torch.no_grad():
        assert not torch.allclose(net(o, None)[1], net(other, None)[1]), what


def test_the_play_mode_and_branch_heads_read_the_active_cards_token() -> None:
    net = _net()
    for head in (net.branch_head_net, net.play_mode_head_net):       # zero-initialised readouts would hide it
        torch.nn.init.normal_(head[-1].weight)
    a, b = ActionEncoder.PLAY_MODE_OFFSET, ActionEncoder.OP_MODE_OFFSET + 3
    card_row = lambda i: 2184 + i * 14 + ACTIVE_CARD_SLOT             # noqa: E731
    o = torch.zeros(1, 3824)
    first, second = o.clone(), o.clone()
    first[0, card_row(4)] = 1.0
    second[0, card_row(70)] = 1.0
    with torch.no_grad():
        assert not torch.allclose(net(first, None)[0][:, a:b], net(second, None)[0][:, a:b])
    assert net.branch_head_net[0].in_features == CFG["hidden_dim"] + 110 + CFG["token_dim"]


def test_chunked_passes_are_the_same_function() -> None:
    net = _net()
    o = _obs(5)
    with torch.no_grad():
        whole = net(o, None)[0]
        net.token_chunk = 2
        assert torch.allclose(net(o, None)[0], whole, atol=1e-5)


def test_the_configuration_comes_back_from_the_weights() -> None:
    net = _net()
    cfg = ladder_config_from_state_dict(net.state_dict())
    assert cfg is not None and cfg["token_main"] is True
    assert (cfg["token_layers"], cfg["token_dim"], cfg["token_heads"], cfg["token_mlp_mult"]) == (2, 32, 4, 2)
    again = create_ladder_net("cpu", **cfg).eval()
    again.load_state_dict(net.state_dict())
    o = _obs()
    with torch.no_grad():
        assert torch.equal(again(o, None)[0], net(o, None)[0])
    assert again.ladder_config()["token_main"] is True


def test_a33_is_untouched_and_the_grouped_axis_is_refused() -> None:
    a33: Any = create_ladder_net("cpu", **dict(A8, entity_proj_dim=64, token_layers=2, token_dim=32))
    names = {k for k, _ in a33.named_parameters()}
    assert any(k.startswith("tok_enc.") for k in names) and "lad_board.0.weight" in names
    assert not any(k.startswith("tok_main") for k in names)
    assert a33.branch_head_net[0].in_features == A8["hidden_dim"] + 110
    assert "token_main" not in a33.ladder_config()
    with pytest.raises(ValueError, match="entity-proj-dim must be 0"):
        create_ladder_net("cpu", **dict(CFG, entity_proj_dim=64))
    with pytest.raises(ValueError, match="needs token_layers"):
        create_ladder_net("cpu", **dict(CFG, token_layers=0))
    with pytest.raises(ValueError, match="not divisible"):
        TokenBlock(33, 4, 2)


def test_mup_builds_its_base_and_scales_nothing_at_the_same_width() -> None:
    net = _net()
    base = base_shapes_for(net.ladder_config(), 256, CFG["hidden_dim"], lambda **kw: create_ladder_net("cpu", **kw))
    assert mup_lr_mults(net, base, CFG["hidden_dim"]) == {}


def test_a_training_step() -> None:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    t: Any = NashPGTrainer(active_net=create_ladder_net(dev, **CFG), env=TsVectorizedEnv(num_envs=2, base_seed=11), num_envs=2,
                      buffer_size=16, batch_size=16, lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False, device=dev)
    t.collect_rollouts()
    before = t.active_net.tok_main.blocks[0].qkv.weight.detach().clone()
    m = t.train_step()
    assert np.isfinite(m["loss"])
    assert not torch.equal(before, t.active_net.tok_main.blocks[0].qkv.weight)


def test_it_exports_to_onnx(tmp_path: Path) -> None:
    torch.save(_net().state_dict(), tmp_path / "net.pt")
    export(str(tmp_path / "net.pt"), str(tmp_path / "net.onnx"), positions=16)       # checks against torch itself
    assert (tmp_path / "net.onnx").exists()
