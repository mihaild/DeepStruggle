"""Guards for the Europe-attention measurement and the observation facts it rests on."""
from __future__ import annotations

import numpy as np
import ts_engine as ts

from ai.eval.europe_attention import EUROPE_BG, NAMES, NODE_OFFSET, _controls


def test_europe_battlegrounds_are_the_five() -> None:
    assert len(EUROPE_BG) == 5
    for cid in EUROPE_BG:
        info = ts.MapData.get_country_info(cid)
        assert bool(info["battleground"]), f"{NAMES[cid]} is not a battleground"
        assert str(info["region"]) == str(ts.Region.EUROPE)
    assert set(NAMES.values()) == {
        "West Germany", "France", "Italy", "East Germany", "Poland"}


def test_control_helper_matches_the_engine() -> None:
    """_controls must agree with the engine's own control rule, not re-derive it wrongly."""
    state = ts.GameState()
    ts.Engine.init_game(state, 1234)
    for cid in range(84):
        engine_ctrl = ts.Scoring.get_country_control(state, cid)
        for player in (ts.Player.US, ts.Player.USSR):
            assert _controls(state, cid, player) == (engine_ctrl == player), NAMES.get(cid, cid)


def test_node_offset_addresses_the_right_country() -> None:
    from bindings.action_encoder import ActionEncoder

    assert NODE_OFFSET == ActionEncoder.NODE_OFFSET
    state = ts.GameState()
    ts.Engine.init_game(state, 99)
    for cid in EUROPE_BG:
        name = ActionEncoder.get_action_name(state, NODE_OFFSET + cid)
        assert NAMES[cid] in name, f"{NODE_OFFSET + cid} -> {name}, expected {NAMES[cid]}"


def test_superpower_adjacency_is_perspective_relative() -> None:
    """Slots 8/9 flip with perspective; the asymmetry story depends on this being true."""
    state = ts.GameState()
    ts.Engine.init_game(state, 55)
    us = np.asarray(ts.extract_observation(state, ts.Player.US), dtype=np.float32)
    su = np.asarray(ts.extract_observation(state, ts.Player.USSR), dtype=np.float32)
    width = 26
    flipped = 0
    for cid in range(84):
        o = cid * width
        if us[o + 8] != su[o + 8]:
            flipped += 1
            # When it differs it must be a swap of the two adjacency slots, not noise.
            assert us[o + 8] == su[o + 9] and us[o + 9] == su[o + 8]
    assert flipped > 0, "superpower adjacency no longer flips with perspective"


def test_western_europe_flag_is_absolute() -> None:
    """Slots 16/17 do NOT flip -- 'Western Europe' means the same to both players."""
    state = ts.GameState()
    ts.Engine.init_game(state, 55)
    us = np.asarray(ts.extract_observation(state, ts.Player.US), dtype=np.float32)
    su = np.asarray(ts.extract_observation(state, ts.Player.USSR), dtype=np.float32)
    width = 26
    for cid in range(84):
        o = cid * width
        assert us[o + 16] == su[o + 16], f"{NAMES.get(cid, cid)} western-europe flag flipped"
        assert us[o + 17] == su[o + 17], f"{NAMES.get(cid, cid)} eastern-europe flag flipped"
    # And at least one country carries each, or the test is checking nothing.
    assert any(us[cid * width + 16] for cid in range(84))
    assert any(us[cid * width + 17] for cid in range(84))
