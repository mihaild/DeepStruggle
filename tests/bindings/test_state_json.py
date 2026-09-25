"""The JSON views of a GameState (bindings/state_json.cpp), which the browser engine shares.

`to_dict()` and `to_save_dict()` are built from the same JSON trees the WebAssembly build prints,
so what these pin is that the text and the dicts are one thing: a position saved in the browser
must open in Python and the other way round, byte for byte.
"""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

import ts_engine as ts
from tools.lib.game_step import drain_chance


def _positions(seed: int, limit: int = 400):
    s = ts.GameState()
    ts.Engine.init_game(s, seed)
    drain_chance(s)
    rng = random.Random(seed)
    for _ in range(limit):
        yield s
        if ts.Engine.is_terminal(s):
            return
        legal = [int(i) for i in np.nonzero(ts.get_flat_action_mask(s, False))[0]]
        ts.Engine.step_flat(s, rng.choice(legal))
        drain_chance(s)


@pytest.mark.parametrize("seed", [3, 17])
def test_save_json_is_the_canonical_text_of_the_save_dict(seed: int) -> None:
    for s in _positions(seed):
        save = s.to_save_dict()
        assert s.to_save_json() == json.dumps(save, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("seed", [4, 21])
def test_save_json_round_trips_the_position(seed: int) -> None:
    for s in _positions(seed):
        back = ts.state_from_save_json(s.to_save_json())
        assert back.to_save_dict() == s.to_save_dict()
        assert np.array_equal(ts.get_flat_action_mask(back, False), ts.get_flat_action_mask(s, False))
        p = s.ctx().decision_player
        if p != ts.Player.NONE:
            assert np.array_equal(ts.extract_observation(back, p), ts.extract_observation(s, p))


def test_display_json_is_to_dict() -> None:
    for s in _positions(9, limit=150):
        assert json.loads(s.to_display_json()) == s.to_dict()


def test_uint64_fields_survive_exactly() -> None:
    s = next(_positions(5))
    save = s.to_save_dict()
    save["rng_state"] = 2**64 - 1
    save["persistent_effects"] = 2**63 + 12345
    back = ts.state_from_save_json(json.dumps(save))
    assert back.rng_state == 2**64 - 1
    assert back.to_save_dict()["persistent_effects"] == 2**63 + 12345


@pytest.mark.parametrize("text", ["", "{", "[1,2]x", '{"a":}', "nope"])
def test_text_that_is_not_json_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        ts.state_from_save_json(text)


def test_a_malformed_list_is_refused_not_opened_approximately() -> None:
    save = next(_positions(6)).to_save_dict()
    save["us_influence"][3] = 300
    with pytest.raises(ValueError):
        ts.state_from_save_json(json.dumps(save))
    with pytest.raises(ValueError):
        ts.state_from_save_dict(save)
