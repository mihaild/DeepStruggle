"""`--window-provoked-defcon`: is a provoked DEFCON-1 credited to the card's player?

By default it is not, so the -1 propagates back as an ordinary loss. Measured, there is nothing
for it to propagate through: the critic barely moves at the choice that decides the game. A
blunder window's advantage is `-1 - v_t` and never consults the critic, which is exactly why it
is the mechanism that works here.

The predicate is tested on constructed states rather than through self-play: a provoked ending
is rare enough that random play never produces one in a reasonable number of steps, and the only
policies that produce them reliably are checkpoints, which are git-ignored.
"""

import ts_engine as ts
from ai.eval.positions import PositionBuilder
from bindings.ts_env import TsVectorizedEnv, credits_defcon_blunder

PROVOKED = ts.EffectBits.DEFCON_SUICIDE_PROVOKED


def _terminal(defcon: int, provoked: bool) -> ts.GameState:
    st = PositionBuilder(hand=(23, 27), side=ts.Player.USSR, defcon=defcon, turn=8).build()
    if provoked:
        st.set_flag(PROVOKED)
    return st


def test_a_provoked_ending_is_uncredited_by_default() -> None:
    assert not credits_defcon_blunder(_terminal(1, provoked=True), window_provoked=False)


def test_the_flag_credits_a_provoked_ending() -> None:
    assert credits_defcon_blunder(_terminal(1, provoked=True), window_provoked=True)


def test_an_unprovoked_ending_is_credited_either_way() -> None:
    """The flag adds a case; it must not move or remove one."""
    for window in (False, True):
        assert credits_defcon_blunder(_terminal(1, provoked=False), window_provoked=window)


def test_a_game_that_did_not_reach_defcon_1_is_never_credited() -> None:
    for window in (False, True):
        for provoked in (False, True):
            assert not credits_defcon_blunder(_terminal(2, provoked), window_provoked=window)


def test_the_flag_defaults_off_on_the_env() -> None:
    assert TsVectorizedEnv(num_envs=2, base_seed=1).window_provoked_defcon is False
    assert TsVectorizedEnv(num_envs=2, base_seed=1,
                           window_provoked_defcon=True).window_provoked_defcon is True
