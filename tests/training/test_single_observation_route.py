"""One observation route: every path to the network must use Observation::extract.

Three paths used to reach the policy -- the vectorized training env, the replay generator, and the
web/bot client. The first two call the engine extractor. The third rebuilt the observation in
Python from a JSON state dict, and that duplicate drifted: replays made through it ended on turn 1
while the same checkpoint played to turn 10 through the engine extractor.

The network client is gone with the server-side workbench -- the browser workbench runs the
WebAssembly build of the engine and reads Observation::extract itself (tests/web/
test_wasm_engine.py holds it to the native one bit for bit). What remains to pin is that the
dict-driven NeuralBot still refuses to play without the engine's observation.
"""

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



def test_the_neural_bot_refuses_to_play_without_the_engines_observation() -> None:
    """The Python reconstruction is gone, and its absence has to be loud.

    It duplicated Observation::extract across ~4300 fields, never reproduced it exactly -- games
    played through it ended on turn 1 where the same checkpoint reached turn 10 through the
    engine extractor -- and reproduced a layout no model reads any more. A duplicate encoder that
    can only be wrong is worse than no encoder, so the bot now says so instead of playing badly.
    """
    pytest.importorskip("torch")
    from bot.neural_bot import NeuralBot

    st = _mid_game_state()
    decider = _decider(st)
    role = "US" if decider == ts.Player.US else "USSR"

    bot = NeuralBot(role, model_path=None, device="cpu")
    mask = ActionEncoder.get_legal_mask(st)
    legal_actions = {
        "valid_ids": [int(i) for i in np.flatnonzero(mask)],
        "decision_type": int(st.ctx().decision_type),
        "allow_early_stop": False,
        "decision_player": role,
    }
    with pytest.raises(ValueError, match="observation_b64"):
        bot.select_action({"countries": {}}, legal_actions)
