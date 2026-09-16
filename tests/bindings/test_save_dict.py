"""A starting position must survive a round trip, and an old save must still load.

Replays store `initial_state = {"seed": N}` and nothing else, so re-driving one calls
`init_game(seed)` -- and any engine change to setup, shuffling or draw order then produces a
different game, silently, mid-replay. Storing the position itself removes that dependency, and the
format is named fields precisely so a save written before a field existed still loads.
"""

from __future__ import annotations

import numpy as np
import pytest

import ts_engine as ts
from bindings.action_encoder import ActionEncoder


def _played(seed: int = 4242, plies: int = 0) -> ts.GameState:
    s = ts.GameState()
    ts.Engine.init_game(s, seed)
    for _ in range(plies):
        if ts.Engine.is_terminal(s):
            break
        legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(s)))
        if not len(legal):
            break
        ts.Engine.step_flat(s, int(legal[0]))
    return s


def _same_decision(a: ts.GameState, b: ts.GameState) -> None:
    """The half of the state that v1 dropped entirely.

    A restored position with the wrong pending decision offers different legal actions and a
    different observation, while every visible board field matches -- so nothing looks wrong.
    """
    assert int(a.ctx_stack_depth) == int(b.ctx_stack_depth), "decision stack depth"
    assert int(a.ctx().decision_type) == int(b.ctx().decision_type), "pending decision type"
    assert int(a.ctx().decision_player) == int(b.ctx().decision_player), "who is to decide"
    assert np.array_equal(np.asarray(ActionEncoder.get_legal_mask(a)),
                          np.asarray(ActionEncoder.get_legal_mask(b))), "legal actions differ"


def _same_position(a: ts.GameState, b: ts.GameState) -> None:
    _same_decision(a, b)
    assert int(a.victory_points) == int(b.victory_points)
    assert int(a.defcon) == int(b.defcon)
    assert int(a.turn) == int(b.turn)
    assert int(a.action_round) == int(b.action_round)
    assert int(a.us_space_track) == int(b.us_space_track)
    assert int(a.ussr_space_track) == int(b.ussr_space_track)
    assert a.china_card_holder == b.china_card_holder
    assert int(a.rng_state) == int(b.rng_state), "the RNG must survive, or dice diverge"
    for c in range(1, 111):
        assert a.get_card_location(c) == b.get_card_location(c), f"card {c} moved"
    for i in range(84):
        ca, cb = a.get_country(i), b.get_country(i)
        assert int(ca.us_influence) == int(cb.us_influence), f"country {i} US influence"
        assert int(ca.ussr_influence) == int(cb.ussr_influence), f"country {i} USSR influence"


def test_a_starting_position_round_trips() -> None:
    s = _played()
    back = ts.state_from_save_dict(s.to_save_dict())
    _same_position(s, back)


def test_the_restored_position_plays_identically() -> None:
    """A position that looks the same but plays differently would be worse than no save at all."""
    s = _played()
    back = ts.state_from_save_dict(s.to_save_dict())
    for _ in range(120):
        if ts.Engine.is_terminal(s) or ts.Engine.is_terminal(back):
            break
        ma = np.asarray(ActionEncoder.get_legal_mask(s))
        mb = np.asarray(ActionEncoder.get_legal_mask(back))
        np.testing.assert_array_equal(ma, mb, err_msg="legal masks diverged after restore")
        legal = np.flatnonzero(ma)
        if not len(legal):
            break
        pick = int(legal[0])
        assert ts.Engine.try_step_flat(s, pick) == ts.Engine.try_step_flat(back, pick)
    assert ts.Engine.is_terminal(s) == ts.Engine.is_terminal(back)


def test_a_save_missing_a_field_still_loads() -> None:
    """The whole reason this is named fields: a save predating a new field must not be unloadable."""
    s = _played()
    d = s.to_save_dict()
    for dropped in ("us_space_track", "china_card_playable", "persistent_effects", "rng_state"):
        partial = {k: v for k, v in d.items() if k != dropped}
        back = ts.state_from_save_dict(partial)   # must not raise
        assert int(back.turn) == int(s.turn), f"dropping {dropped} broke unrelated fields"


def test_an_unknown_field_is_ignored() -> None:
    """A save from a NEWER engine should still open in an older one, minus what it cannot use."""
    s = _played()
    d = dict(s.to_save_dict())
    d["starting_bonus"] = 3          # a field this build has never heard of
    back = ts.state_from_save_dict(d)
    _same_position(s, back)


def test_the_format_is_tagged() -> None:
    assert _played().to_save_dict()["format"] == "ts_save_v2"


@pytest.mark.parametrize("seed", [777, 31337, 4242])
def test_every_position_in_a_whole_game_round_trips(seed: int) -> None:
    """Round-trip at EVERY step, which is what it takes to catch a dropped decision stack.

    Saving only at a starting position tests the one case where the decision stack is whatever
    `init_game` left -- so a serializer that stores none of it passes. The failure is mid-decision:
    at a chance node a v1 save came back reading DecisionType 5 where the saved state had 7.
    """
    s = ts.GameState()
    ts.Engine.init_game(s, seed)
    steps = 0
    for _ in range(4000):
        if ts.Engine.is_terminal(s):
            break
        back = ts.state_from_save_dict(s.to_save_dict())
        _same_decision(s, back)

        p = (s.ctx().decision_player if s.ctx().decision_player != ts.Player.NONE
             else s.phasing_player)
        assert np.array_equal(np.asarray(ts.extract_observation(s, p)),
                              np.asarray(ts.extract_observation(back, p))), (
            f"seed {seed} step {steps}: the restored state yields a different observation")

        legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(s)))
        if not len(legal):
            break
        ts.Engine.step_flat(s, int(legal[steps % len(legal)]))
        steps += 1
    assert steps > 50, f"only {steps} positions exercised"


def test_a_restored_mid_game_position_continues_the_same_game() -> None:
    """The property the format exists for: reload mid-game and play on identically."""
    s = ts.GameState()
    ts.Engine.init_game(s, 31337)
    for i in range(60):
        legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(s)))
        if not len(legal) or ts.Engine.is_terminal(s):
            break
        ts.Engine.step_flat(s, int(legal[i % len(legal)]))

    back = ts.state_from_save_dict(s.to_save_dict())
    for i in range(200):
        if ts.Engine.is_terminal(s) or ts.Engine.is_terminal(back):
            break
        legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(s)))
        if not len(legal):
            break
        idx = int(legal[i % len(legal)])
        assert ts.Engine.try_step_flat(s, idx)
        assert ts.Engine.try_step_flat(back, idx), f"restored game refused action {idx} at ply {i}"
        _same_decision(s, back)
        assert int(s.victory_points) == int(back.victory_points), f"VP diverged at ply {i}"
    assert ts.Engine.is_terminal(s) == ts.Engine.is_terminal(back)
