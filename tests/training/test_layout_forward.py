"""The layout can actually run the network, end to end from the engine.

Checking that a model's TOTAL_OBS_SIZE equals the engine's width is not the same as checking the
model runs: v2.3 once matched on width and still could not do a forward pass, because the board
reshape was hardcoded to 84x28 while its board block is 84x26. Widths agreed, 629 tests passed,
and the layout could not have trained a single step.

So this drives a real observation out of the engine, through the real model, to an action -- for
every architecture, since both read the same one layout.
"""

from typing import Callable, List

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models import create_coldwar_net, create_coldwar_net_v2
from bindings.action_encoder import ActionEncoder

#: Every architecture, by the name --arch takes. They all read observation layout v2.3.
#: Every architecture, on the CPU explicitly. V3 and V4 were removed -- neither produced a
#: logged result and both predate the corrected engine.
ARCHITECTURES: List[Callable[[], object]] = [
    lambda: create_coldwar_net("cpu"), lambda: create_coldwar_net_v2("cpu"),
]
ARCH_IDS = ["v1", "v2"]


def _state() -> ts.GameState:
    s = ts.GameState()
    ts.Engine.init_game(s, 4242)
    return s


@pytest.mark.parametrize("factory", ARCHITECTURES, ids=ARCH_IDS)
def test_the_engine_width_and_the_model_width_agree(factory) -> None:
    assert factory().TOTAL_OBS_SIZE == int(ts.OBS_SIZE)
    obs = np.asarray(ts.extract_observation(_state(), ts.Player.USSR))
    assert obs.shape[0] == int(ts.OBS_SIZE)


@pytest.mark.parametrize("factory", ARCHITECTURES, ids=ARCH_IDS)
def test_a_real_observation_reaches_an_action(factory) -> None:
    """The check the width comparison does not make."""
    state = _state()
    model = factory()
    model.eval()
    obs = np.asarray(ts.extract_observation(state, ts.Player.USSR), dtype=np.float32)
    mask = np.asarray(ActionEncoder.get_legal_mask(state))
    with torch.no_grad():
        action, logp, entropy, value, _ = model.sample_action(
            torch.from_numpy(obs)[None], torch.from_numpy(mask)[None], deterministic=True)
    idx = int(action.reshape(-1)[0].item())
    assert 0 <= idx < 212
    assert mask[idx], "the model chose an action the mask forbids"
    assert np.isfinite(float(value.reshape(-1)[0].item()))


@pytest.mark.parametrize("factory", ARCHITECTURES, ids=ARCH_IDS)
def test_a_batch_trains_one_step(factory) -> None:
    """A backward pass too: a shape that only breaks under gradient is still broken."""
    model = factory()
    width = model.TOTAL_OBS_SIZE
    obs = torch.zeros((4, width), dtype=torch.float32)
    mask = torch.ones((4, 212), dtype=torch.uint8)
    out = model.extract_features(obs)
    latent = out[0] if isinstance(out, tuple) else out
    loss = latent.square().mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_the_batch_runner_agrees_with_the_single_extractor() -> None:
    runner = ts.VectorizedBatchRunner(3, 555)
    rows = np.asarray(runner.get_observations())
    assert rows.shape[1] == int(ts.OBS_SIZE)
    for i in range(3):
        st = runner.get_state(i)
        ctx = st.ctx()
        side = ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player
        direct = np.asarray(ts.extract_observation(st, side))
        assert np.array_equal(rows[i], direct)


@pytest.mark.parametrize("factory", ARCHITECTURES, ids=ARCH_IDS)
def test_an_observation_of_the_wrong_width_raises(factory) -> None:
    """The half that used to be missing.

    Widths were checked at extraction and never at the network, so a model handed a vector from
    a retired layout read the wrong floats and returned an ordinary-looking number. Four probes
    did exactly that. A checkpoint from a retired layout now fails here instead.
    """
    model = factory()
    with pytest.raises(ValueError, match="floats wide"):
        model(torch.zeros((2, 4293), dtype=torch.float32),
              torch.ones((2, 212), dtype=torch.uint8))


def test_a_frozen_copy_matches_the_model_it_was_cloned_from() -> None:
    """The snapshot evaluator freezes a copy of the live net and loads its weights into it.

    Built by listing constructor arguments, that has broken twice -- once on card_features and
    use_history, then on board_features after the other three were fixed -- and both times only
    at the first snapshot, minutes into a run. create_like reads every dimension off the model.
    """
    from ai.models.coldwar_net_v2 import create_like

    model = create_coldwar_net_v2()
    frozen = create_like(model)
    frozen.load_state_dict(model.state_dict())          # the call that failed before create_like
    assert frozen.TOTAL_OBS_SIZE == model.TOTAL_OBS_SIZE
    assert frozen.board_features == model.board_features
    assert frozen.card_features == model.card_features
    assert frozen.GLOBAL_SIZE == model.GLOBAL_SIZE
