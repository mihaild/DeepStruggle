"""The two architecture variants aimed at the per-country representation failure.

`research/metrics.md` 21.12 localised the loss: a country's exact influence is recoverable from
its own raw observation slots 97% of the time, from its post-GraphConv token 63%, and from the
pooled 512-float trunk essentially never. Two changes target the two steps, and both must be
detectable from a checkpoint -- a model rebuilt without them would load the other tensors and
silently be a different network, which is how arms E and F died.
"""
from __future__ import annotations

import torch

from ai.models.coldwar_net_v2 import ColdWarNetV2, GraphConvLayer, create_coldwar_net_v2

OBS = ColdWarNetV2.TOTAL_OBS_SIZE


def _obs(n: int = 3) -> torch.Tensor:
    return torch.randn(n, OBS)


def test_self_transform_lets_a_node_keep_its_own_value() -> None:
    """Plain GCN mixes a node with its neighbours through one matrix; this adds a second."""
    plain = GraphConvLayer(4, 8)
    withs = GraphConvLayer(4, 8, self_transform=True)
    assert plain.self_linear is None
    assert withs.self_linear is not None
    adj = torch.eye(3)
    x = torch.randn(2, 3, 4)
    assert plain(x, adj).shape == withs(x, adj).shape == (2, 3, 8)


def test_self_transform_changes_the_function_and_the_checkpoint() -> None:
    a = create_coldwar_net_v2("cpu", identity_dim=16)
    b = create_coldwar_net_v2("cpu", identity_dim=16, self_transform=True)
    assert not any(k.endswith("self_linear.weight") for k in a.state_dict())
    assert any(k.endswith("self_linear.weight") for k in b.state_dict())
    assert sum(p.numel() for p in b.parameters()) > sum(p.numel() for p in a.parameters())


def test_attention_readout_keeps_the_trunk_width() -> None:
    """Every head and every probe reads a 512-float trunk; the read-out folds back into it."""
    m = create_coldwar_net_v2("cpu", identity_dim=16, attn_readout=64)
    m.eval()
    with torch.no_grad():
        h = m.extract_features(_obs())
    # extract_features returns a tuple when asked for attention weights; it is not, here.
    assert isinstance(h, torch.Tensor)
    assert h.shape == (3, 512)


def test_attention_readout_is_detectable_and_adds_parameters() -> None:
    a = create_coldwar_net_v2("cpu", identity_dim=16)
    b = create_coldwar_net_v2("cpu", identity_dim=16, attn_readout=64)
    assert "ro_query.weight" not in a.state_dict()
    assert b.state_dict()["ro_query.weight"].shape[0] == 64
    assert sum(p.numel() for p in b.parameters()) > sum(p.numel() for p in a.parameters())


def test_readout_actually_reads_the_board() -> None:
    """Changing only the board block must move the trunk, or the read-out is decorative."""
    m = create_coldwar_net_v2("cpu", identity_dim=16, attn_readout=64)
    m.eval()
    x = _obs(1)
    y = x.clone()
    y[:, : ColdWarNetV2.BOARD_SIZE] += 1.0
    with torch.no_grad():
        hx, hy = m.extract_features(x), m.extract_features(y)
    assert isinstance(hx, torch.Tensor) and isinstance(hy, torch.Tensor)
    assert not torch.allclose(hx, hy)


def test_both_variants_produce_a_usable_policy_and_value() -> None:
    m = create_coldwar_net_v2("cpu", identity_dim=16, self_transform=True, attn_readout=64)
    m.eval()
    mask = torch.zeros(2, ColdWarNetV2.ACTION_SPACE_SIZE, dtype=torch.uint8)
    mask[:, :5] = 1
    with torch.no_grad():
        acts, *_ = m.sample_action(_obs(2), mask, temperature=0.1)
    assert acts.shape == (2,)
    assert bool(((acts >= 0) & (acts < 5)).all()), "sampled outside the legal mask"


def test_a_variant_checkpoint_round_trips_through_the_loader() -> None:
    """The loader must rebuild the same network from weight names alone."""
    import tempfile

    from tools.lib.player_agent import NeuralAgent

    m = create_coldwar_net_v2("cpu", identity_dim=16, self_transform=True, attn_readout=64)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(m.state_dict(), f.name)
        agent = NeuralAgent.from_checkpoint(f.name, device="cpu")
    assert agent.model.attn_readout == 64
    assert agent.model.self_transform is True
