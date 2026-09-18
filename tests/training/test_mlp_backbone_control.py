"""The MLP backbone control: same heads, same observation, no structure.

It exists to bound what the graph convolution and the card-to-country cross-attention are worth.
For the comparison to mean anything it has to be identical everywhere else, and it has to travel
through the same loaders as any other arm -- the failure this guards is the one `create_like`'s
docstring describes twice, where a frozen evaluation copy is built as the wrong shape and the
arm dies at its first snapshot.
"""

import torch
from bindings.action_encoder import ActionEncoder

from ai.models.coldwar_net_v2 import (ColdWarNetMLP, ColdWarNetV2, create_coldwar_net_mlp,
                                      create_coldwar_net_v2, create_like)


def test_it_presents_the_same_interface_as_v2() -> None:
    mlp = create_coldwar_net_mlp("cpu")
    obs = torch.randn(4, mlp.TOTAL_OBS_SIZE)
    mask = torch.ones(4, ActionEncoder.FLAT_ACTION_SIZE, dtype=torch.uint8)
    logits, v_win, v_vp = mlp(obs, mask)
    assert logits.shape == (4, ActionEncoder.FLAT_ACTION_SIZE) and v_win.shape == (4, 1) and v_vp.shape == (4, 1)
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


def test_drop_static_narrows_the_input_and_survives_a_round_trip(tmp_path) -> None:
    """--drop-static removes the per-entity slots that never change within a game.

    A shared-weight encoder needs them: its tokens are permutation-equivalent, so stability and
    Ops are what tell one from another. A positional reader gets identity from the offset, so
    the same values only add a constant the bias already supplies while occupying input width.
    """
    import torch

    from ai.models.coldwar_net_v2 import (ColdWarNetV2, create_coldwar_net_mlp,
                                          static_input_mask)
    from tools.lib.player_agent import NeuralAgent

    mask = static_input_mask()
    assert int(mask.sum()) == 11 * 84 + 4 * 110 == 1364

    full = create_coldwar_net_mlp("cpu", drop_static=False)
    thin = create_coldwar_net_mlp("cpu", drop_static=True)
    assert len(full.keep_idx) == ColdWarNetV2.TOTAL_OBS_SIZE
    assert len(thin.keep_idx) == ColdWarNetV2.TOTAL_OBS_SIZE - 1364

    obs = torch.randn(2, ColdWarNetV2.TOTAL_OBS_SIZE)
    m = torch.ones(2, ActionEncoder.FLAT_ACTION_SIZE, dtype=torch.uint8)
    assert thin(obs, m)[0].shape == (2, ActionEncoder.FLAT_ACTION_SIZE)

    # The narrowed width must still read as layout v2.3, not as a retired layout.
    path = tmp_path / "thin.pt"
    torch.save(thin.state_dict(), path)
    agent = NeuralAgent.from_checkpoint(str(path), device="cpu")
    assert getattr(agent.model, "drop_static") is True


def test_dropping_static_slots_cannot_change_the_function_it_sees() -> None:
    """Only constant dimensions are removed, so varying two of them must not move the output."""
    import torch

    from ai.models.coldwar_net_v2 import (ColdWarNetV2, create_coldwar_net_mlp,
                                          static_input_mask)

    net = create_coldwar_net_mlp("cpu", drop_static=True)
    net.eval()
    mask = static_input_mask()
    obs = torch.randn(1, ColdWarNetV2.TOTAL_OBS_SIZE)
    m = torch.ones(1, ActionEncoder.FLAT_ACTION_SIZE, dtype=torch.uint8)
    with torch.no_grad():
        before = net(obs, m)[0]
        obs[0, mask] += 7.0
        after = net(obs, m)[0]
    assert torch.allclose(before, after)


def test_keep_idx_is_not_persisted_and_both_eras_load(tmp_path) -> None:
    """`keep_idx` is derived from drop_static, so it must not be in the state dict.

    It briefly was, which broke every MLP checkpoint saved before it existed -- the loader
    reported `missing ['keep_idx']` and refused. Checkpoints now exist from both eras, so the
    loader has to accept a state dict with the key and one without.
    """
    import torch

    from ai.models.coldwar_net_v2 import create_coldwar_net_mlp
    from tools.lib.player_agent import NeuralAgent

    net = create_coldwar_net_mlp("cpu", drop_static=True)
    assert "keep_idx" not in net.state_dict(), "a derived buffer must not be persisted"

    clean = tmp_path / "clean.pt"
    torch.save(net.state_dict(), clean)
    assert NeuralAgent.from_checkpoint(str(clean), device="cpu") is not None

    # A checkpoint from the era when it *was* persisted must still load.
    legacy = dict(net.state_dict())
    legacy["keep_idx"] = net.keep_idx.clone()
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    agent = NeuralAgent.from_checkpoint(str(legacy_path), device="cpu")
    assert getattr(agent.model, "drop_static") is True
