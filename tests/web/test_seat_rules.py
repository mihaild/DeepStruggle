"""A person in the page is never offered an Event that cannot fire (ENG-9's OPP_EVENT_GATED).

The page seats a person's side with every rule bit (`TSApp.seatSides`, `WasmEngine.currentRules`),
and only the auto-play side with its model's own; an older model's rule still offers the Event.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest
import ts_engine as ts

from tests.web.js_runner import PUBLIC, run_ts

NATO = 21
EVENT = 110              # flat slot of Resolution::EVENT
SELECT_PLAY_MODE = 2
RESOLUTION_EVENT = 0


def _ussr_choosing_with_nato() -> str:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    while not (state.current_phase == ts.Phase.ACTION_ROUND and state.phasing_player == ts.Player.USSR
               and state.ctx().decision_type == ts.DecisionType.SELECT_CARD):
        legal = [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(state)) if v]
        ts.Engine.step_flat(state, legal[0])
    hand = ts.hand_of(ts.Player.USSR)
    swapped = next(c for c in range(1, 111) if state.get_card_location(c) == hand and c != NATO)
    state.set_card_location(swapped, state.get_card_location(NATO))
    state.set_card_location(NATO, hand)
    assert not ts.CardHandlers.can_trigger_event(state, NATO, ts.Player.US)
    return state.to_save_json()


@pytest.fixture(scope="module")
def offered(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    out = tmp_path_factory.mktemp("seat_rules")
    save = out / "save.json"
    save.write_text(_ussr_choosing_with_nato())
    return run_ts("seat_rules.ts", str(out), PUBLIC, str(save))


def test_the_page_knows_the_current_rules(offered: Dict[str, Any]) -> None:
    # The game rules a person plays: every rule bit but P43's canonical orders, a model's click order.
    assert offered["rules"] == ts.OBS_FEATURES_PERSON_RULES
    assert ts.OBS_FEATURES_PERSON_RULES == ts.OBS_FEATURES_RULES & ~(ts.OBS_FEATURE_CANONICAL_FIRST
                                                                       | ts.OBS_FEATURE_CANONICAL_LAST)


def test_a_person_is_not_offered_the_event(offered: Dict[str, Any]) -> None:
    person = offered["person"]
    assert person["type"] == SELECT_PLAY_MODE
    assert person["mask"][EVENT] == 0
    assert RESOLUTION_EVENT not in person["valid"]
    assert any(person["mask"]), "the Ops play is still on offer"


def test_a_pre_bit_model_still_is(offered: Dict[str, Any]) -> None:
    legacy = offered["legacy"]
    assert legacy["changed"] == [True, True]
    assert legacy["mask"][EVENT] == 1
    assert RESOLUTION_EVENT in legacy["valid"]
