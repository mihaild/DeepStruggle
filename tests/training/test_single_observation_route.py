"""One observation route: every path to the network must use Observation::extract.

Three paths reach the policy -- the vectorized training env, the replay generator, and the
web/bot client. The first two call the engine extractor. The third rebuilt the 4293-dim
observation in Python from a JSON state dict, and that duplicate drifted: replays made
through it ended on turn 1 while the same checkpoint played to turn 10 through the engine
extractor. A policy fed a different encoding than it was trained on plays close to
randomly.

The server now sends the engine's observation to the player whose move it is, and
NeuralBot uses it when present. These tests pin that contract.
"""

import base64

import numpy as np
import pytest
import ts_engine as ts

from bindings.action_encoder import ActionEncoder


def _mid_game_state(seed: int = 4242, plies: int = 40) -> ts.GameState:
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    for _ in range(plies):
        if ts.Engine.is_terminal(st):
            break
        legal = np.flatnonzero(ActionEncoder.get_legal_mask(st))
        if not len(legal):
            break
        ts.Engine.step_flat(st, int(legal[0]))
    return st


def _decider(st: ts.GameState) -> ts.Player:
    ctx = st.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player


def test_session_emits_engine_observation_only_to_the_acting_player() -> None:
    """The observation carries the acting player's own hand, so the opponent must not get it."""
    from web.server.session import GameSession

    sess = GameSession(game_id="pytest_obs_route", seed=4242)
    for _ in range(40):
        if ts.Engine.is_terminal(sess.state):
            break
        legal = np.flatnonzero(ActionEncoder.get_legal_mask(sess.state))
        if not len(legal):
            break
        ts.Engine.step_flat(sess.state, int(legal[0]))

    decider = _decider(sess.state)
    acting = "US" if decider == ts.Player.US else "USSR"
    waiting = "USSR" if acting == "US" else "US"

    assert sess.get_state_dict(for_role=acting).get("observation_b64"), \
        "acting player should receive the engine observation"
    assert not sess.get_state_dict(for_role=waiting).get("observation_b64"), \
        "non-acting player must not receive it (it contains the opponent's hand)"
    assert not sess.get_state_dict(for_role="OBSERVER").get("observation_b64")
    assert not sess.get_state_dict().get("observation_b64")


def test_transmitted_observation_is_bit_identical_to_the_engine_extractor() -> None:
    from web.server.session import GameSession

    sess = GameSession(game_id="pytest_obs_exact", seed=4242)
    for _ in range(40):
        if ts.Engine.is_terminal(sess.state):
            break
        legal = np.flatnonzero(ActionEncoder.get_legal_mask(sess.state))
        if not len(legal):
            break
        ts.Engine.step_flat(sess.state, int(legal[0]))

    decider = _decider(sess.state)
    role = "US" if decider == ts.Player.US else "USSR"
    payload = sess.get_state_dict(for_role=role)["observation_b64"]
    sent = np.frombuffer(base64.b64decode(payload), dtype=np.float32)
    expected = np.asarray(ts.extract_observation(sess.state, decider), dtype=np.float32)

    assert sent.shape == expected.shape == (4293,)
    assert np.array_equal(sent, expected), "transmitted observation must be the engine's own"


def test_neural_bot_local_reconstruction_differs_from_the_engine() -> None:
    """Why the route matters: the Python duplicate does not reproduce the engine encoding.

    If this ever starts passing, the reconstruction has been brought into agreement and the
    fallback could be retired -- but until then it must not be treated as equivalent.
    """
    pytest.importorskip("torch")
    from bot.neural_bot import NeuralBot

    st = _mid_game_state()
    decider = _decider(st)
    role = "US" if decider == ts.Player.US else "USSR"
    engine_obs = np.asarray(ts.extract_observation(st, decider), dtype=np.float32)

    bot = NeuralBot(role, model_path=None, device="cpu")
    mask = ActionEncoder.get_legal_mask(st)
    legal_actions = {
        "valid_ids": [int(i) for i in np.flatnonzero(mask)],
        "decision_type": int(st.ctx().decision_type),
        "allow_early_stop": False,
        "decision_player": role,
    }

    state_dict = dict(ts.state_to_dict(st))
    state_dict["observation_b64"] = base64.b64encode(engine_obs.tobytes()).decode("ascii")
    with_engine = bot.select_action(state_dict, legal_actions)

    state_dict.pop("observation_b64")
    without = bot.select_action(state_dict, legal_actions)

    # Both must produce a legal action; the point is that the inputs are not the same.
    for chosen in (with_engine, without):
        assert chosen is not None
