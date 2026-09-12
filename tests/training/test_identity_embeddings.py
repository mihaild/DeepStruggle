"""Identity embeddings: can the model tell two same-featured cards apart?

The card block encodes a card's properties and never which card it is -- identity exists only as
position in the 110x14 block, and v2's shared per-card MLP followed by mean+max pooling discards
position. Measured on the real observation, 110 cards collapse to 46 distinct signatures and 86%
of them collide with another card, in groups of up to six.

A learned embedding indexed by position restores the distinction. It is model-side: the
observation is not touched, which is what keeps it out of the observation rule.
"""

import numpy as np
import torch

import ts_engine as ts
from ai.eval.positions import PositionBuilder
from ai.models.coldwar_net_v2 import create_coldwar_net_v2, create_like

MARSHALL, US_JAPAN, UN_INTERVENTION = 23, 27, 32


def _trunk(net, state):
    obs = torch.from_numpy(
        np.asarray(ts.extract_observation(state, ts.Player.USSR), dtype=np.float32)
    ).reshape(1, -1)
    with torch.no_grad():
        return net.extract_features(obs).numpy().reshape(-1)


def _hands():
    a = PositionBuilder(hand=(MARSHALL, UN_INTERVENTION), side=ts.Player.USSR,
                        defcon=3, turn=5).build()
    b = PositionBuilder(hand=(US_JAPAN, UN_INTERVENTION), side=ts.Player.USSR,
                        defcon=3, turn=5).build()
    return a, b


def test_the_two_cards_really_do_have_identical_features() -> None:
    """If this ever fails the rest of the file is testing nothing."""
    from ai.models.coldwar_net_v2 import ColdWarNetV2
    a, b = _hands()
    off = ColdWarNetV2.BOARD_FEATURES * 84
    cf = ColdWarNetV2.CARD_FEATURES
    fa = np.asarray(ts.extract_observation(a, ts.Player.USSR))[
        off + (MARSHALL - 1) * cf: off + MARSHALL * cf]
    fb = np.asarray(ts.extract_observation(b, ts.Player.USSR))[
        off + (US_JAPAN - 1) * cf: off + US_JAPAN * cf]
    assert np.array_equal(fa, fb)


def test_without_embeddings_the_two_hands_are_one_vector() -> None:
    net = create_coldwar_net_v2("cpu", identity_dim=0)
    net.eval()
    a, b = _hands()
    assert np.linalg.norm(_trunk(net, a) - _trunk(net, b)) < 1e-4


def test_with_embeddings_they_are_not() -> None:
    net = create_coldwar_net_v2("cpu", identity_dim=16)
    net.eval()
    a, b = _hands()
    assert np.linalg.norm(_trunk(net, a) - _trunk(net, b)) > 1e-2


def test_create_like_carries_the_embedding_width() -> None:
    """A frozen copy of the wrong width dies at the first snapshot."""
    for dim in (0, 16):
        src = create_coldwar_net_v2("cpu", identity_dim=dim)
        copy = create_like(src, "cpu")
        assert copy.identity_dim == dim
        assert set(copy.state_dict()) == set(src.state_dict())


def test_the_layout_check_subtracts_what_the_embedding_added(tmp_path) -> None:
    """card_fc is widened by identity_dim; a naive read makes 14 look like 30."""
    from ai.models.coldwar_net_v2 import card_features_of, check_checkpoint_layout
    from tools.lib.player_agent import NeuralAgent

    net = create_coldwar_net_v2("cpu", identity_dim=16)
    sd = net.state_dict()
    assert card_features_of(sd) == 14
    check_checkpoint_layout(sd)

    path = tmp_path / "ident.pt"
    torch.save(sd, path)
    agent = NeuralAgent.from_checkpoint(str(path), device="cpu")
    assert getattr(agent.model, "identity_dim") == 16
