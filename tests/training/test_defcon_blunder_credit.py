"""The DEFCON-risk label credits an unprovoked DEFCON-1 loss to the card's player, never a provoked one.

(`--window-provoked-defcon`, which credited the provoked ones too, was retired 2026-11-22.) Tested on constructed
states: a provoked ending is too rare in random play to find quickly.
"""

import ts_engine as ts
from ai.eval.positions import PositionBuilder
from bindings.ts_env import credits_defcon_blunder


def _terminal(defcon: int, provoked: bool) -> ts.GameState:
    st = PositionBuilder(hand=(23, 27), side=ts.Player.USSR, defcon=defcon, turn=8).build()
    if provoked:
        st.set_flag(ts.EffectBits.DEFCON_SUICIDE_PROVOKED)
    return st


def test_an_unprovoked_defcon_one_is_credited() -> None:
    assert credits_defcon_blunder(_terminal(1, provoked=False))


def test_a_provoked_defcon_one_is_not() -> None:
    assert not credits_defcon_blunder(_terminal(1, provoked=True))


def test_no_credit_above_defcon_one() -> None:
    for provoked in (False, True):
        assert not credits_defcon_blunder(_terminal(2, provoked))
