"""A checkpoint's observation layout has to be recoverable from the checkpoint.

`legacy` and `v2.1` differ in the card block width and in whether the history branch exists, and
nothing in a filename says which one a `.pt` holds. Every place that rebuilds a network to receive
saved weights therefore has to read the shape off the weights rather than take the factory
defaults.

Arm E died on its first snapshot because one such place did not: `evaluate_and_log_snapshot` built
its frozen copy at defaults and then tried to load a v2.1 policy into it. That failed loudly, which
was lucky -- the same mistake between two layouts of equal width would have loaded silently and
reinterpreted every card feature by one position.
"""

from __future__ import annotations

import torch

from ai.models.coldwar_net_v2 import ColdWarNetV2, card_features_of, create_coldwar_net_v2

LAYOUTS = [("legacy", 12, True, 4293), ("v2.1", 13, False, 3891)]


def test_each_layout_builds_at_its_own_width() -> None:
    for name, card_features, use_history, width in LAYOUTS:
        model = create_coldwar_net_v2(card_features=card_features, use_history=use_history)
        assert model.TOTAL_OBS_SIZE == width, f"{name} built at {model.TOTAL_OBS_SIZE}"
        obs = torch.zeros(2, width)
        mask = torch.ones(2, 212, dtype=torch.uint8)
        logits, v_win, v_vp = model(obs, mask)
        assert logits.shape == (2, 212)
        assert v_win.shape == (2, 1) and v_vp.shape == (2, 1)


def test_the_layout_is_recoverable_from_the_weights_alone() -> None:
    """What `load_agent` and the frozen eval copy both depend on."""
    for _name, card_features, use_history, _w in LAYOUTS:
        sd = create_coldwar_net_v2(card_features=card_features, use_history=use_history).state_dict()
        assert card_features_of(sd) == card_features
        assert any(k.startswith("hist_conv.") for k in sd) is use_history


def test_a_frozen_copy_shaped_from_the_model_can_receive_its_weights() -> None:
    """The exact operation that killed arm E, for both layouts."""
    for _name, card_features, use_history, _w in LAYOUTS:
        trained = create_coldwar_net_v2(card_features=card_features, use_history=use_history)
        frozen = create_coldwar_net_v2(
            card_features=getattr(trained, "card_features", ColdWarNetV2.CARD_FEATURES),
            use_history=getattr(trained, "use_history", True))
        frozen.load_state_dict(trained.state_dict())      # must not raise


def test_a_frozen_copy_built_at_defaults_rejects_a_v21_policy() -> None:
    """The negative control: if this stops raising, the guard above is not doing anything."""
    trained = create_coldwar_net_v2(card_features=13, use_history=False)
    default_shaped = create_coldwar_net_v2()
    try:
        default_shaped.load_state_dict(trained.state_dict())
    except RuntimeError:
        return
    raise AssertionError(
        "a default-shaped network accepted v2.1 weights, so the layouts are no longer "
        "distinguishable by shape and a mismatch would now pass silently")
