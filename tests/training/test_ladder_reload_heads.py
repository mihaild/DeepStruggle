"""Audit M-3.3: the card lookup's heads x dim factorisation and the attention head count are not recoverable from
the weights, and the reload pinned them (dim 32, 4 heads), so a checkpoint built otherwise silently reloaded with the
wrong head split and scaling. Each is now a buffer when off that default, read back on reload; a checkpoint at the
default has no buffer, so every earlier one loads exactly as before."""

from typing import Any, Dict

import torch

from ai.models.ladder_net import create_ladder_net, ladder_config_from_state_dict
from tests.training.test_mup import CFG


def _roundtrip(**over: Any) -> Dict[str, Any]:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **{**CFG, **over}).eval()
    with torch.no_grad():
        for p in net.parameters():            # every weight nonzero, so a wrong head split shows in the output
            p.add_(0.05 * torch.randn_like(p))
    sd = net.state_dict()
    cfg = ladder_config_from_state_dict(dict(sd))
    assert cfg is not None
    rebuilt = create_ladder_net("cpu", **cfg).eval()
    rebuilt.load_state_dict(sd, strict=True)
    obs, mask = torch.randn(4, 3824), torch.ones(4, 220, dtype=torch.bool)
    with torch.no_grad():
        a, b = net(obs, mask)[0], rebuilt(obs, mask)[0]
    assert torch.allclose(a, b, atol=1e-5), "the auto-detected rebuild computes something else"
    return cfg


def test_a_non_default_card_lookup_round_trips() -> None:
    cfg = _roundtrip(card_lookup=True, card_lookup_heads=4, card_lookup_dim=16, card_lookup_identity_dim=8)
    assert (cfg["card_lookup_heads"], cfg["card_lookup_dim"]) == (4, 16)


def test_a_conventional_card_lookup_has_no_buffer_and_reloads_as_before() -> None:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **{**CFG, "card_lookup": True, "card_lookup_heads": 2, "card_lookup_dim": 32,
                                      "card_lookup_identity_dim": 8})
    assert "cl_heads_buf" not in net.state_dict()
    cfg = _roundtrip(card_lookup=True, card_lookup_heads=2, card_lookup_dim=32, card_lookup_identity_dim=8)
    assert (cfg["card_lookup_heads"], cfg["card_lookup_dim"]) == (2, 32)


def test_a_non_default_attention_head_count_round_trips() -> None:
    entity = dict(input_mode="entity", drop_static=False, card_self_attention=True, cross_attention=True,
                  entity_proj_dim=64, hidden_dim=128, num_attn_heads=2)
    cfg = _roundtrip(**entity)
    assert cfg["num_attn_heads"] == 2


def test_the_default_networks_state_is_unchanged() -> None:
    """A8 and every default: no new buffer, so an earlier checkpoint's keys are exactly the network's."""
    torch.manual_seed(0)
    sd = create_ladder_net("cpu", **CFG).state_dict()
    assert "num_attn_heads_buf" not in sd and "cl_heads_buf" not in sd
    cfg = ladder_config_from_state_dict(dict(sd))
    assert cfg is not None and cfg["num_attn_heads"] == 4
