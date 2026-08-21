import pytest
import ts_engine

def test_engine_init():
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, 12345)
    assert state.turn == 1
    assert state.defcon == 5
    assert state.victory_points == 0
    assert state.current_phase == ts_engine.Phase.SETUP
    assert state.ctx().decision_player == ts_engine.Player.USSR
    assert state.ctx().decision_type == ts_engine.DecisionType.POINT_NODE

def test_engine_setup_and_step():
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, 12345)
    
    # USSR places 6 influence in Eastern Europe
    for _ in range(6):
        legal_nodes = ts_engine.Engine.get_legal_action_indices(state)
        assert len(legal_nodes) > 0
        action = ts_engine.MicroAction(ts_engine.DecisionType.POINT_NODE, legal_nodes[0], 0, 0)
        success = ts_engine.Engine.step(state, action)
        assert success

    # Now US places 7 influence in Western Europe
    assert state.ctx().decision_player == ts_engine.Player.US
    for _ in range(7):
        legal_nodes = ts_engine.Engine.get_legal_action_indices(state)
        assert len(legal_nodes) > 0
        action = ts_engine.MicroAction(ts_engine.DecisionType.POINT_NODE, legal_nodes[0], 0, 0)
        success = ts_engine.Engine.step(state, action)
        assert success

    # Now should be HEADLINE phase
    assert state.current_phase == ts_engine.Phase.HEADLINE

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

def test_map_and_card_data():
    c_info = ts_engine.MapData.get_country_info(14)
    assert c_info["name"] == "East Germany"
    assert c_info["battleground"] is True
    
    card_info = ts_engine.CardData.get_card_info(4)
    assert card_info["name"] == "Duck and Cover"
    assert card_info["ops"] == 3
    assert card_info["side"] == "US"
