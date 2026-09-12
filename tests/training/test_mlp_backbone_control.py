"""The MLP backbone control: same heads, same observation, no structure.

It exists to bound what the graph convolution and the card-to-country cross-attention are worth.
For the comparison to mean anything it has to be identical everywhere else, and it has to travel
through the same loaders as any other arm -- the failure this guards is the one `create_like`'s
docstring describes twice, where a frozen evaluation copy is built as the wrong shape and the
arm dies at its first snapshot.
"""

import torch

from ai.models.coldwar_net_v2 import (ColdWarNetMLP, ColdWarNetV2, create_coldwar_net_mlp,
                                      create_coldwar_net_v2, create_like)


def test_it_presents_the_same_interface_as_v2() -> None:
    mlp = create_coldwar_net_mlp("cpu")
    obs = torch.randn(4, mlp.TOTAL_OBS_SIZE)
    mask = torch.ones(4, 212, dtype=torch.uint8)
    logits, v_win, v_vp = mlp(obs, mask)
    assert logits.shape == (4, 212) and v_win.shape == (4, 1) and v_vp.shape == (4, 1)
    assert bool((v_win >= -1).all() and (v_win <= 1).all())


def test_none_of_the_structured_encoders_survive() -> None:
    """Replaced, not bypassed -- otherwise the parameter count is a fiction."""
    mlp = create_coldwar_net_mlp("cpu")
    for name in ("gconv1", "gconv2", "card_fc", "cross_attn", "cross_card_proj", "global_proj"):
        assert getattr(mlp, name) is None, name
    keys = set(mlp.state_dict())
    assert not any(k.startswith(("gconv", "card_fc", "cross_attn")) for k in keys)


def test_create_like_preserves_the_backbone() -> None:
    """A frozen copy built as a v2 would fail to load at the first snapshot."""
    for factory, cls in ((create_coldwar_net_v2, ColdWarNetV2),
                         (create_coldwar_net_mlp, ColdWarNetMLP)):
        src = factory("cpu")
        copy = create_like(src, "cpu")
        assert type(copy) is cls
        assert set(copy.state_dict()) == set(src.state_dict())


def test_a_checkpoint_round_trips_through_the_agent_loader(tmp_path) -> None:
    from tools.lib.player_agent import NeuralAgent

    net = create_coldwar_net_mlp("cpu")
    path = tmp_path / "mlp.pt"
    torch.save(net.state_dict(), path)
    agent = NeuralAgent.from_checkpoint(str(path), device="cpu")
    assert isinstance(agent.model, ColdWarNetMLP)


def test_the_layout_check_reads_the_input_width(tmp_path) -> None:
    """No card block to infer from, but the first dense layer *is* the layout."""
    from ai.models.coldwar_net_v2 import check_checkpoint_layout
    from tools.lib.player_agent import NeuralAgent

    net = create_coldwar_net_mlp("cpu")
    sd = net.state_dict()
    check_checkpoint_layout(sd)                      # the real one passes

    sd = dict(sd)
    sd["mlp_in.0.weight"] = torch.zeros(1024, 4293)  # a retired layout's width
    try:
        check_checkpoint_layout(sd)
    except ValueError as e:
        assert "4293" in str(e)
    else:
        raise AssertionError("a retired-layout width must be refused")
