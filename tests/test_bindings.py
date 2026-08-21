import pytest
import ts_engine

def test_engine_init():
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, 12345)
    assert state.turn == 1
    assert state.defcon == 5
    assert state.victory_points == 0
    assert state.current_phase == ts_engine.Phase.SETUP

def test_engine_setup_and_step():
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, 12345)

    # In setup phase, USSR places 6 influence in Eastern Europe
    # Legal action mask check
    mask = ts_engine.Engine.get_legal_action_mask(state)
    assert len(mask) == 84 # DecisionType::POINT_NODE has max 84 countries
    assert sum(mask) > 0

    # Execute valid step: Place influence in East Germany
    action = ts_engine.MicroAction(
        decision_type=ts_engine.DecisionType.POINT_NODE,
        primary_id=14, # East Germany
        secondary_id=0,
        flags=0
    )
    ok = ts_engine.Engine.step(state, action)
    assert ok
    assert state.get_country(14).ussr_influence == 4

def test_to_dict():
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, 42)
    d = state.to_dict()
    assert "countries" in d
    assert len(d["countries"]) == 84
    assert "East Germany" in d["countries"]
    assert "hands" in d
    assert len(d["hands"]["US"]) == 8
    assert len(d["hands"]["USSR"]) == 8
    assert "legal_actions" in d
    assert d["legal_actions"]["decision_type"] == int(ts_engine.DecisionType.POINT_NODE)
    assert d["legal_actions"]["decision_type_name"] == "POINT_NODE"
    assert d["current_phase_name"] == "SETUP"

    # Verify only Early War cards are dealt on Turn 1
    for cid in d["hands"]["US"] + d["hands"]["USSR"]:
        info = ts_engine.CardData.get_card_info(cid)
        assert info["era"] == int(ts_engine.WarEra.EARLY), f"Card {info['name']} should be Early War"

def test_map_and_card_data():
    eg_id = ts_engine.MapData.get_country_by_name("East Germany")
    assert eg_id == 14
    assert ts_engine.MapData.get_country_name(14) == "East Germany"

    c_info = ts_engine.MapData.get_country_info(14)
    assert c_info["stability"] == 3
    assert c_info["battleground"] is True

    card_name = ts_engine.CardData.get_card_name(6) # The China Card
    assert card_name == "The China Card"
    card_info = ts_engine.CardData.get_card_info(6)
    assert card_info["ops"] == 4
