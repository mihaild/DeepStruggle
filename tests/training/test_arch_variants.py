"""The two architecture variants aimed at the per-country representation failure.

Probing localises the loss: a country's exact influence is recoverable from its own raw
observation slots almost always, from its post-GraphConv token much less often, and from the
pooled 512-float trunk essentially never. Two changes target the two steps, and both must be
detectable from a checkpoint -- a model rebuilt without them would load the other tensors and
silently be a different network, which has killed runs at their first snapshot.
"""
from __future__ import annotations

import pytest
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


# --- per-entity policy heads ---------------------------------------------------------------

def test_per_entity_heads_are_detectable_and_keep_the_action_space() -> None:
    a = create_coldwar_net_v2("cpu", identity_dim=16, self_transform=True)
    b = create_coldwar_net_v2("cpu", identity_dim=16, self_transform=True, per_entity_heads=64)
    assert "pe_trunk.weight" not in a.state_dict()
    assert b.state_dict()["pe_trunk.weight"].shape[0] == 64
    mask = torch.ones(2, ColdWarNetV2.ACTION_SPACE_SIZE, dtype=torch.uint8)
    b.eval()
    with torch.no_grad():
        logits, v_win, v_vp = b(_obs(2), mask)
    assert logits.shape == (2, ColdWarNetV2.ACTION_SPACE_SIZE)
    assert v_win.shape == (2, 1) and v_vp.shape == (2, 1)


def test_a_country_logit_moves_with_that_country_and_not_its_neighbour() -> None:
    """The whole point. Under dense heads every logit reads one pooled vector, so perturbing
    country i and country j move logit i by similar amounts. A per-entity head must respond far
    more to its own country's slots than to another country's.

    The correction is zero-initialised, so this is a property of the *trained* path rather than
    of a fresh model -- at initialisation the network is deliberately the dense baseline exactly
    (`test_per_entity_heads_start_as_the_dense_baseline`). The weights are given a value here to
    ask whether the path, once live, is actually local.
    """
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu", identity_dim=16, self_transform=True, per_entity_heads=64)
    for head in (m.pe_country, m.pe_card):
        out_layer = head[-1]
        assert isinstance(out_layer, torch.nn.Linear)
        torch.nn.init.normal_(out_layer.weight, std=0.5)
    m.eval()
    bf = ColdWarNetV2.BOARD_FEATURES
    own, other = 3, 40
    country_action = 119 + own

    def logit_after(country: int) -> float:
        x = torch.zeros(1, ColdWarNetV2.TOTAL_OBS_SIZE)
        y = x.clone()
        y[0, country * bf] = 1.0          # slot 0 is that country's own influence
        with torch.no_grad():
            a = m(x, None)[0][0, country_action]
            b = m(y, None)[0][0, country_action]
        return abs(float(b - a))

    assert logit_after(own) > 5.0 * logit_after(other) + 1e-6, (
        "the country's own logit is not tracking its own observation slots")


def test_shared_actions_still_come_from_the_trunk() -> None:
    """Play mode, timing, op mode, branch and confirm name no entity, so they stay dense --
    and they must still be produced, or the action space silently loses 18 of its 212."""
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu", identity_dim=16, per_entity_heads=64)
    m.eval()
    with torch.no_grad():
        logits = m(_obs(4), None)[0]
    shared = torch.cat([logits[:, 110:119], logits[:, 203:212]], dim=-1)
    assert shared.shape == (4, 18)
    assert bool(torch.isfinite(shared).all())


def test_the_mlp_backbone_refuses_per_entity_heads() -> None:
    """It reads the flat vector and forms no tokens, so the flag would be a silent no-op."""
    from ai.models.coldwar_net_v2 import ColdWarNetMLP

    with pytest.raises(ValueError, match="cannot carry per-entity policy heads"):
        ColdWarNetMLP(per_entity_heads=64)


def test_per_entity_checkpoint_round_trips_through_the_loader() -> None:
    import tempfile

    from tools.lib.player_agent import NeuralAgent

    m = create_coldwar_net_v2("cpu", identity_dim=16, self_transform=True, per_entity_heads=64)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(m.state_dict(), f.name)
        agent = NeuralAgent.from_checkpoint(f.name, device="cpu")
    assert agent.model.per_entity_heads == 64
    assert agent.model.self_transform is True


def test_per_entity_heads_start_as_the_dense_baseline() -> None:
    """The residual form's whole point, and the failure the replacing form measured.

    Computing a logit *only* from its entity plus a 64-float projection of the trunk made that
    projection the sole path from the trunk to that logit, where the dense head reads all 512.
    Each logit gained its own entity's detail and lost most of its view of the situation, and it
    cost more than the blindness it fixed. Added instead, with the output layers zeroed, the
    network begins as the dense baseline exactly and learns a refinement on top -- so it cannot
    start worse, whatever it learns later.
    """
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu", identity_dim=16, self_transform=True, per_entity_heads=64)
    m.eval()
    obs = _obs(3)
    with torch.no_grad():
        h, _attn, tokens = m._encode(obs)
        dense = m.policy_head(h)
        actual = m._policy_logits(h, tokens)
    assert torch.allclose(actual, dense, atol=1e-6), \
        "the per-entity correction is not zero at initialisation"


def test_the_correction_becomes_live_once_trained() -> None:
    """Zero at init must not mean zero forever -- the path has to carry gradient."""
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu", identity_dim=16, self_transform=True, per_entity_heads=64)
    out_layer = m.pe_country[-1]
    assert isinstance(out_layer, torch.nn.Linear)
    torch.nn.init.normal_(out_layer.weight, std=0.1)
    m.eval()
    with torch.no_grad():
        h, _attn, tokens = m._encode(_obs(2))
        assert not torch.allclose(m._policy_logits(h, tokens), m.policy_head(h), atol=1e-6)


def test_only_the_entity_actions_are_corrected() -> None:
    """The 18 actions naming no entity must come from the dense head untouched."""
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu", identity_dim=16, per_entity_heads=64)
    for head in (m.pe_country, m.pe_card):
        out_layer = head[-1]
        assert isinstance(out_layer, torch.nn.Linear)
        torch.nn.init.normal_(out_layer.weight, std=0.5)
        torch.nn.init.normal_(out_layer.bias, std=0.5)
    m.eval()
    with torch.no_grad():
        h, _attn, tokens = m._encode(_obs(2))
        dense, actual = m.policy_head(h), m._policy_logits(h, tokens)
    for lo, hi in ((110, 119), (203, 212)):
        assert torch.allclose(actual[:, lo:hi], dense[:, lo:hi], atol=1e-6), \
            f"actions {lo}..{hi - 1} name no entity and must not be corrected"
    assert not torch.allclose(actual[:, 0:110], dense[:, 0:110], atol=1e-6)
    assert not torch.allclose(actual[:, 119:203], dense[:, 119:203], atol=1e-6)


# --- graph depth ---------------------------------------------------------------------------

@pytest.mark.parametrize("layers", [2, 1, 0])
def test_graph_depth_builds_and_runs(layers: int) -> None:
    """0 keeps a per-country encoder and drops adjacency; 1 and 2 propagate that many hops.

    Adjacency's mechanical uses -- placement legality, coup legality, the realignment modifier --
    are precomputed per country in the observation, so the graph's remaining job is strategic and
    its depth is an empirical question rather than a given.
    """
    m = create_coldwar_net_v2("cpu", identity_dim=16, self_transform=True, graph_layers=layers)
    assert hasattr(m, "gconv1") is (layers >= 1)
    assert hasattr(m, "gconv2") is (layers >= 2)
    assert hasattr(m, "board_fc") is (layers == 0)
    m.eval()
    with torch.no_grad():
        h = m.extract_features(_obs(2))
    assert isinstance(h, torch.Tensor) and h.shape == (2, 512)


def test_zero_layers_makes_a_country_independent_of_its_neighbours() -> None:
    """With no adjacency, perturbing a neighbour must not move this country's token at all."""
    torch.manual_seed(0)
    bf = ColdWarNetV2.BOARD_FEATURES
    country, neighbour = 15, 14      # adjacent on the real map

    def neighbour_effect(layers: int) -> float:
        m = create_coldwar_net_v2("cpu", self_transform=True, graph_layers=layers)
        m.eval()
        x = torch.zeros(1, ColdWarNetV2.TOTAL_OBS_SIZE)
        y = x.clone()
        y[0, neighbour * bf] = 1.0
        with torch.no_grad():
            tok_a, tok_b = m._encode(x)[2], m._encode(y)[2]
        assert tok_a is not None and tok_b is not None
        return float((tok_b[0][0, country] - tok_a[0][0, country]).abs().max())

    assert neighbour_effect(0) == 0.0, "a 0-layer board branch still mixes neighbours"
    assert neighbour_effect(1) > 1e-6, "a 1-layer graph does not propagate from a neighbour"


def test_graph_depth_is_detectable_from_a_checkpoint() -> None:
    import tempfile

    from tools.lib.player_agent import NeuralAgent

    for layers in (0, 1, 2):
        m = create_coldwar_net_v2("cpu", identity_dim=16, self_transform=True,
                                  graph_layers=layers)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(m.state_dict(), f.name)
            agent = NeuralAgent.from_checkpoint(f.name, device="cpu")
        assert agent.model.graph_layers == layers, f"depth {layers} not recovered"


def test_an_impossible_depth_is_refused() -> None:
    with pytest.raises(ValueError, match="graph_layers must be 0, 1 or 2"):
        create_coldwar_net_v2("cpu", graph_layers=3)
