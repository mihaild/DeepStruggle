"""P41 J5's asymmetric layout (owner, 2026-11-04): the global token, the countries and a few hand slots are the active
tokens, the 110 cards a read-only memory. The same outputs as the full layout; a card outside the hand is its own memory
entry, untouched by the board; a held card and the active card are contextualised; more held cards than slots fails
loudly; chunking and compiling change nothing; the configuration comes back from the weights; a training step; the ONNX
export the workbench runs."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ai.models.ladder_net import ACTIVE_CARD_SLOT, MY_HAND_SLOT, create_ladder_net, ladder_config_from_state_dict
from ai.training import NashPGTrainer
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import TsVectorizedEnv
from tests.training.test_token_main import CFG as FULL
from tools.export_onnx import export

CFG: dict[str, Any] = dict(FULL, token_card_memory=6)
CARD_OFFSET, CARD_FEATURES = 84 * 26, 14


def _net() -> Any:
    torch.manual_seed(0)
    return create_ladder_net("cpu", **CFG).eval()


def _obs(n: int = 3, held: tuple[int, ...] = (3, 17, 40)) -> torch.Tensor:
    """Random features with the hand slots set only for `held` (random 0..1 rows would hold half the deck)."""
    torch.manual_seed(1)
    o = torch.rand(n, 3824)
    cards = o[:, CARD_OFFSET:CARD_OFFSET + 110 * CARD_FEATURES].view(n, 110, CARD_FEATURES)
    cards[..., MY_HAND_SLOT] = 0.0
    cards[..., ACTIVE_CARD_SLOT] = 0.0
    cards[:, list(held), MY_HAND_SLOT] = 1.0
    return o


def _cards(net: Any, o: torch.Tensor) -> torch.Tensor:
    """The card tokens the heads read, (B, 110, t)."""
    b = o.shape[0]
    with torch.no_grad():
        out = net._token_path(o[:, :84 * 26], o[:, CARD_OFFSET:CARD_OFFSET + 110 * CARD_FEATURES],
                              o[:, net.GLOBAL_OFFSET:net.GLOBAL_OFFSET + net.GLOBAL_SIZE], b)
    assert out.shape == (b, 195, CFG["token_dim"])
    return out[:, 85:]


def test_the_same_outputs_and_a_memory_without_ffn() -> None:
    net = _net()
    names = {k for k, _ in net.named_parameters()}
    assert "tok_main.blocks.0.mem_kv.weight" in names and "tok_empty_slot" in names
    logits, v_win, v_vp = net(_obs(), torch.ones(3, ActionEncoder.FLAT_ACTION_SIZE, dtype=torch.uint8))
    assert logits.shape == (3, ActionEncoder.FLAT_ACTION_SIZE) and v_win.shape == (3, 1) and v_vp.shape == (3, 1)


def test_a_card_outside_the_hand_is_its_own_memory_entry_and_a_held_card_reads_the_board() -> None:
    net = _net()
    o = _obs()
    other = o.clone()
    other[:, 0:26] = torch.rand(3, 26) + 2.0                   # a country changes
    a, b = _cards(net, o), _cards(net, other)
    assert torch.equal(a[:, 5], b[:, 5])                       # not held: the board cannot reach it
    assert not torch.allclose(a[:, 17], b[:, 17])              # held: it attends the board
    again = o.clone()
    again[:, CARD_OFFSET + 5 * CARD_FEATURES + 3] += 1.0       # its own row changes
    assert not torch.allclose(_cards(net, again)[:, 5], a[:, 5])


def test_the_active_card_takes_a_slot_and_reaches_the_play_mode_head() -> None:
    net = _net()
    for head in (net.branch_head_net, net.play_mode_head_net):
        torch.nn.init.normal_(head[-1].weight)
    o = _obs(1)
    o[0, CARD_OFFSET + 70 * CARD_FEATURES + ACTIVE_CARD_SLOT] = 1.0
    other = o.clone()
    other[:, 0:26] = torch.rand(1, 26) + 2.0
    assert not torch.allclose(_cards(net, o)[:, 70], _cards(net, other)[:, 70])
    a, b = ActionEncoder.PLAY_MODE_OFFSET, ActionEncoder.OP_MODE_OFFSET + 3
    with torch.no_grad():
        assert not torch.allclose(net(o, None)[0][:, a:b], net(other, None)[0][:, a:b])


def test_more_held_cards_than_slots_fails_loudly() -> None:
    net = _net()
    with pytest.raises(RuntimeError, match="hand slots"):
        net(_obs(2, held=tuple(range(7))), None)


def test_chunked_and_compiled_passes_are_the_same_function() -> None:
    net = _net()
    o = _obs(5)
    with torch.no_grad():
        whole = net(o, None)[0]
        net.token_chunk = 2
        assert torch.allclose(net(o, None)[0], whole, atol=1e-5)
        net.token_chunk = 0
        net.tok_main.compiled = True
        assert torch.allclose(net(o, None)[0], whole, atol=1e-5)


def test_the_configuration_comes_back_from_the_weights() -> None:
    net = _net()
    cfg = ladder_config_from_state_dict(net.state_dict())
    assert cfg is not None and cfg["token_card_memory"] == 6
    again = create_ladder_net("cpu", **cfg).eval()
    again.load_state_dict(net.state_dict())
    o = _obs()
    with torch.no_grad():
        assert torch.equal(again(o, None)[0], net(o, None)[0])
    assert again.ladder_config()["token_card_memory"] == 6
    assert "token_card_memory" not in create_ladder_net("cpu", **FULL).ladder_config()
    with pytest.raises(ValueError, match="needs token_main"):
        create_ladder_net("cpu", **dict(FULL, token_main=False, entity_proj_dim=64, token_card_memory=6))


def test_a_training_step() -> None:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    t: Any = NashPGTrainer(active_net=create_ladder_net(dev, **dict(CFG, token_card_memory=16)),
                           env=TsVectorizedEnv(num_envs=2, base_seed=11), num_envs=2, buffer_size=16, batch_size=16,
                           lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False, device=dev)
    t.collect_rollouts()
    before = t.active_net.tok_main.blocks[0].mem_kv.weight.detach().clone()
    m = t.train_step()
    assert np.isfinite(m["loss"])
    assert not torch.equal(before, t.active_net.tok_main.blocks[0].mem_kv.weight)


def test_it_exports_to_onnx(tmp_path: Path) -> None:
    torch.manual_seed(0)
    torch.save(create_ladder_net("cpu", **dict(CFG, token_card_memory=16)).state_dict(), tmp_path / "net.pt")
    export(str(tmp_path / "net.pt"), str(tmp_path / "net.onnx"), positions=16)       # checks against torch itself
    assert (tmp_path / "net.onnx").exists()
