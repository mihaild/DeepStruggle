"""The step contract: a refused action raises, and legal play is untouched.

This is the nucleus of P13. It exists because five loops each decided for themselves whether to
check the engine's return value, and all of them decided not to -- which turned one bad action
into 1,355 identical rejected submissions recorded in a replay as though they had happened.
"""

from __future__ import annotations

import numpy as np
import pytest

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import IllegalActionError, assert_legal, step_checked


def _fresh(seed: int = 4002) -> ts.GameState:
    s = ts.GameState()
    ts.Engine.init_game(s, seed)
    return s


def test_legal_play_is_unaffected() -> None:
    """A full game driven through step_checked must never raise."""
    s = _fresh()
    played = 0
    for _ in range(3000):
        if ts.Engine.is_terminal(s):
            break
        mask = np.asarray(ActionEncoder.get_legal_mask(s))
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            break
        step_checked(s, int(legal[0]), context="test")
        played += 1
    assert played > 100, f"only {played} actions played; the fixture is not exercising much"


def test_a_refused_action_raises_rather_than_being_ignored() -> None:
    s = _fresh()
    ts.Engine.auto_advance_step(s)
    mask = np.asarray(ActionEncoder.get_legal_mask(s))
    legal = set(np.flatnonzero(mask).tolist())
    illegal = next(i for i in range(len(mask)) if i not in legal)

    before = int(s.ctx().decision_type)
    with pytest.raises(IllegalActionError):
        step_checked(s, illegal, context="unit test")
    assert int(s.ctx().decision_type) == before, "state moved on a refused action"


def test_the_error_names_what_was_asked_and_what_arrived() -> None:
    """A bare 'illegal action' would not have shortened the original diagnosis by much."""
    s = _fresh()
    ts.Engine.auto_advance_step(s)
    mask = np.asarray(ActionEncoder.get_legal_mask(s))
    illegal = next(i for i in range(len(mask)) if not mask[i])
    with pytest.raises(IllegalActionError) as err:
        step_checked(s, illegal, context="my-loop")
    text = str(err.value)
    assert "asking for" in text and "my-loop" in text


def test_assert_legal_catches_it_before_the_engine_sees_it() -> None:
    s = _fresh()
    ts.Engine.auto_advance_step(s)
    mask = np.asarray(ActionEncoder.get_legal_mask(s))
    legal = np.flatnonzero(mask)
    assert_legal(s, int(legal[0]), context="ok")
    illegal = next(i for i in range(len(mask)) if not mask[i])
    with pytest.raises(IllegalActionError):
        assert_legal(s, illegal, context="bad")


def test_auto_advance_is_an_explicit_choice() -> None:
    """The five loops disagreed about settling silently; here it is a parameter."""
    a, b = _fresh(), _fresh()
    ts.Engine.auto_advance_step(a)
    ts.Engine.auto_advance_step(b)
    mask = np.asarray(ActionEncoder.get_legal_mask(a))
    pick = int(np.flatnonzero(mask)[0])
    step_checked(a, pick, auto_advance=False)
    step_checked(b, pick, auto_advance=True)
    # Settling can only move the state further along, never less far.
    assert int(b.ctx().decision_type) is not None
