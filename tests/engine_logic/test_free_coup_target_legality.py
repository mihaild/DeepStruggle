"""A free coup granted by an event must still be a coup the rules allow.

`Operations::can_coup_or_realign` refuses a country the opponent has no influence in
(`engine/src/ops.cpp:119`), and `get_coup_target_mask` is built on it, so an ordinary Ops coup
cannot target an empty country. Two events run their own target lists instead and never consult
it:

* **#91 Ortega Elected in Nicaragua** — `card_dispatcher.cpp:1431` accepts any country adjacent
  to Nicaragua.
* **#107 Che** — `card_dispatcher.cpp:1402` accepts any non-battleground in Central America,
  South America or Africa that has not been visited.

Neither checks opponent influence, so both offer coups the rules forbid. This was found in
ts-replayer replay 139, turn 9, action round 2: the US played Ortega (a USSR card) for Ops, the
event handed the USSR a free coup, and the engine offered **Cuba** — which the log and the engine
agree held `US 0 / USSR 3`. Couping Cuba is illegal with no US influence there, but Cuba is a
battleground, so the offered move took DEFCON from 2 to 1 and ended the game against the phasing
player. The engine's DEFCON-1 handling is correct (the phasing player loses, per the rulebook);
what is wrong is that the move was offered at all. The same shape appears at replays 16, 165 and
245.

Both handlers now filter on `can_coup`, at the target mask in `get_event_action_mask` and again
where the chosen target is applied. These tests are the regression.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

import ts_engine as ts

ORTEGA = 91
CHE = 107
NICARAGUA_NEIGHBOURS_WITH_CUBA = "Cuba is adjacent to Nicaragua and is a battleground"

# Observation board block: 84 countries x 28 features; column 21 is "can the observing player
# coup this country" (engine/src/observation.cpp:120-122).
STRIDE = 28
CAN_MY_COUP = 21


def _country_id(name: str) -> int:
    for cid in range(84):
        if ts.MapData.get_country_info(cid)["name"].lower() == name.lower():
            return cid
    raise AssertionError(f"country {name!r} not found")


def _offered_country_targets(state: ts.GameState) -> List[int]:
    """The country ids the engine currently offers at a POINT_NODE."""
    mask = np.asarray(ts.get_flat_action_mask(state))
    out: List[int] = []
    for a in np.flatnonzero(mask):
        ma = ts.decode_flat_action(state, int(a))
        if ma.decision_type == ts.DecisionType.POINT_NODE and int(ma.primary_id) < 84:
            out.append(int(ma.primary_id))
    return out


def _can_coup(state: ts.GameState, player: ts.Player, cid: int) -> bool:
    obs = np.asarray(ts.extract_observation(state, player), dtype=np.float32)
    return bool(obs[cid * STRIDE + CAN_MY_COUP] > 0.5)



def _past_setup(seed: int) -> ts.GameState:
    """A game advanced out of SETUP.

    The coup columns of the observation are hardcoded to 0 while `current_phase == SETUP`
    (`engine/src/observation.cpp:120`), so a state straight out of `init_game` would report
    "cannot coup" everywhere and make these assertions pass for the wrong reason.
    """
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    for _ in range(400):
        if state.current_phase != ts.Phase.SETUP:
            return state
        mask = np.asarray(ts.get_flat_action_mask(state))
        legal = np.flatnonzero(mask)
        assert legal.size, "setup stalled with no legal action"
        ts.Engine.step_flat(state, int(legal[0]))
    raise AssertionError("never left SETUP")


def test_ortega_free_coup_cannot_target_a_country_the_opponent_is_absent_from() -> None:
    state = _past_setup(4242)

    cuba = _country_id("Cuba")
    # The position from replay 139 T9 AR2: the USSR holds Cuba outright, the US is not there.
    state.set_country(cuba, 0, 3)
    state.defcon = 2
    state.phasing_player = ts.Player.US

    assert not _can_coup(state, ts.Player.USSR, cuba), (
        "precondition: the USSR must not be able to coup a country with no US influence"
    )

    ts.CardHandlers.trigger_event(state, ORTEGA, ts.Player.US)
    assert state.ctx().decision_type == ts.DecisionType.POINT_NODE
    assert state.ctx().decision_player == ts.Player.USSR

    offered = _offered_country_targets(state)
    assert cuba not in offered, (
        f"the engine offered Cuba as a free-coup target with US influence 0 "
        f"({NICARAGUA_NEIGHBOURS_WITH_CUBA}); offered={offered}"
    )


def test_che_free_coup_cannot_target_a_country_the_opponent_is_absent_from() -> None:
    state = _past_setup(4243)

    # Che targets non-battlegrounds in Central/South America or Africa.
    cid = _country_id("Costa Rica")
    state.set_country(cid, 0, 2)  # USSR present, US absent -> the USSR has nothing to coup
    state.phasing_player = ts.Player.US

    assert not _can_coup(state, ts.Player.USSR, cid)

    ts.CardHandlers.trigger_event(state, CHE, ts.Player.USSR)
    if state.ctx().decision_type != ts.DecisionType.POINT_NODE:
        pytest.skip("Che did not open a target node in this configuration")

    offered = _offered_country_targets(state)
    assert cid not in offered, (
        f"the engine offered a free-coup target with no US influence; offered={offered}"
    )


def test_ordinary_ops_coup_targets_are_filtered_correctly() -> None:
    """The control: `can_coup` itself is right, so the defect is in the events, not the rule."""
    state = _past_setup(4244)
    cuba = _country_id("Cuba")
    state.set_country(cuba, 0, 3)
    assert not _can_coup(state, ts.Player.USSR, cuba)
    state.set_country(cuba, 2, 3)
    assert _can_coup(state, ts.Player.USSR, cuba)
