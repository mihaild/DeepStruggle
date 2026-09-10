"""Europe Control is a distinguishable ending, not an ordinary 20 VP win.

Controlling Europe when Europe is scored ends the game at ±20 VP and GAME_OVER — which is
exactly what a 20 VP win on the track looks like from the outside. The engine records
`EUROPE_CONTROL_WIN` so the two can be told apart afterwards, the same reason
`CMC_SUICIDE_LOSS` exists. ITS reports Europe Control as 1.55% of its 44,136 finished games, so
without the flag that share was being counted as ordinary 20 VP wins.
"""

import pytest
import ts_engine

from tools.lib.tournament_evaluator import classify_game_ending_reason

EUROPE_COUNTRIES = [
    cid for cid in range(84)
    if ts_engine.MapData.get_country_info(cid)["region"] == ts_engine.Region.EUROPE
]


def _state_with_europe_controlled_by(player) -> ts_engine.GameState:
    """Pile enough influence into Europe that `player` controls the region outright.

    Written through `set_country`, which is the only way to change the board from Python:
    `get_country` hands back a *copy*, so assigning to its `us_influence` silently does
    nothing.
    """
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, 42)
    for cid in EUROPE_COUNTRIES:
        strength = int(ts_engine.MapData.get_country_info(cid)["stability"]) + 3
        if player == ts_engine.Player.US:
            state.set_country(cid, strength, 0)
        else:
            state.set_country(cid, 0, strength)
    return state


def test_the_fixture_really_puts_europe_under_control() -> None:
    """Guards the helper itself: a no-op setter would make every test below vacuous."""
    assert EUROPE_COUNTRIES, "no European countries found; the region filter is wrong"
    state = _state_with_europe_controlled_by(ts_engine.Player.US)
    summary = ts_engine.Scoring.evaluate_region(state, ts_engine.Region.EUROPE, False)
    assert summary.us_status == ts_engine.RegionalStatus.CONTROL


@pytest.mark.parametrize("player,expected_vp", [
    (ts_engine.Player.US, 20),
    (ts_engine.Player.USSR, -20),
])
def test_scoring_europe_while_controlling_it_sets_the_flag(player, expected_vp) -> None:
    state = _state_with_europe_controlled_by(player)
    assert not state.has_flag(ts_engine.EffectBits.EUROPE_CONTROL_WIN)

    ts_engine.Scoring.score_region(state, ts_engine.Region.EUROPE)

    assert state.current_phase == ts_engine.Phase.GAME_OVER
    assert int(state.victory_points) == expected_vp
    assert state.has_flag(ts_engine.EffectBits.EUROPE_CONTROL_WIN)
    assert classify_game_ending_reason(state) == "Europe Control"


def test_an_ordinary_twenty_vp_win_is_not_reported_as_europe_control() -> None:
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, 42)
    state.victory_points = 20
    state.current_phase = ts_engine.Phase.GAME_OVER
    assert not state.has_flag(ts_engine.EffectBits.EUROPE_CONTROL_WIN)
    assert classify_game_ending_reason(state) == "20 VP"


def test_scoring_a_region_nobody_controls_sets_nothing() -> None:
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, 42)
    ts_engine.Scoring.score_region(state, ts_engine.Region.EUROPE)
    assert not state.has_flag(ts_engine.EffectBits.EUROPE_CONTROL_WIN)


def test_defcon_one_still_outranks_europe_control() -> None:
    # DEFCON 1 keeps absolute precedence: a game that blows up was ended by the explosion,
    # whatever else was true of the board.
    state = _state_with_europe_controlled_by(ts_engine.Player.US)
    ts_engine.Scoring.score_region(state, ts_engine.Region.EUROPE)
    assert state.has_flag(ts_engine.EffectBits.EUROPE_CONTROL_WIN)
    state.defcon = 1
    assert classify_game_ending_reason(state).startswith("DEFCON 1")


def test_the_ending_key_is_exposed_to_the_metrics() -> None:
    from bindings.ts_env import ENDING_REASON_KEYS, _ENDING_REASON_MAP

    assert "europe_control" in ENDING_REASON_KEYS
    assert _ENDING_REASON_MAP["Europe Control"] == "europe_control"


def test_the_human_reference_splits_europe_control_out_of_twenty_vp() -> None:
    from ai.itsc_reference import ITSC_REFERENCE

    # 684 of 44,136 ITS games, previously folded into the 20 VP share.
    assert ITSC_REFERENCE["ending_frac_europe_control"] == pytest.approx(684 / 44_136, abs=1e-4)
    total = sum(v for k, v in ITSC_REFERENCE.items()
                if k.startswith("ending_frac_") and "_won_" not in k)
    assert total == pytest.approx(1.0, abs=2e-3)
