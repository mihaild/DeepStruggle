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
    ok = ts_engine.Engine.try_step(state, action)
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

def test_us_bonus_placement_full_lifecycle():
    state = ts_engine.GameState()
    ts_engine.Engine.init_game(state, 12345)

    # 1. USSR places 6 influence in Poland (15)
    for _ in range(6):
        action = ts_engine.MicroAction(
            decision_type=ts_engine.DecisionType.POINT_NODE,
            primary_id=15,
            secondary_id=0,
            flags=0
        )
        assert ts_engine.Engine.try_step(state, action)

    # 2. Transits to US Stage 0 (7 influence in Western Europe)
    assert state.current_phase == ts_engine.Phase.SETUP
    assert state.ctx().decision_player == ts_engine.Player.US
    assert state.ctx().remaining_steps == 7
    assert state.ctx().pending_ops_value == 0

    mask = ts_engine.Engine.get_legal_action_mask(state)
    assert mask[7] == 1  # West Germany legal
    assert mask[25] == 0 # Iran illegal during Western Europe stage

    # US places 4 in West Germany (7) and 3 in Italy (10)
    for _ in range(4):
        action = ts_engine.MicroAction(decision_type=ts_engine.DecisionType.POINT_NODE, primary_id=7, secondary_id=0, flags=0)
        assert ts_engine.Engine.try_step(state, action)
    for _ in range(3):
        action = ts_engine.MicroAction(decision_type=ts_engine.DecisionType.POINT_NODE, primary_id=10, secondary_id=0, flags=0)
        assert ts_engine.Engine.try_step(state, action)

    # 3. Transits to US Stage 1 (2 Bonus influence in countries with existing US presence)
    assert state.current_phase == ts_engine.Phase.SETUP
    assert state.ctx().decision_player == ts_engine.Player.US
    assert state.ctx().remaining_steps == 2
    assert state.ctx().pending_ops_value == 1

    mask = ts_engine.Engine.get_legal_action_mask(state)
    assert mask[25] == 1 # Iran (has 1 US influence) is legal!
    assert mask[7] == 1  # West Germany (has 4 US influence) is legal!
    assert mask[0] == 1  # Canada (has 2 US influence) is legal!
    assert mask[26] == 0 # Iraq (0 US influence) is ILLEGAL!

    # Place bonus 1 in Iran
    act_iran = ts_engine.MicroAction(decision_type=ts_engine.DecisionType.POINT_NODE, primary_id=25, secondary_id=0, flags=0)
    assert ts_engine.Engine.try_step(state, act_iran)
    assert state.get_country(25).us_influence == 2
    assert state.ctx().remaining_steps == 1

    # Place bonus 2 in West Germany
    act_wg = ts_engine.MicroAction(decision_type=ts_engine.DecisionType.POINT_NODE, primary_id=7, secondary_id=0, flags=0)
    assert ts_engine.Engine.try_step(state, act_wg)
    assert state.get_country(7).us_influence == 5

    # 4. Setup completes cleanly into Turn 1 HEADLINE phase
    assert state.current_phase == ts_engine.Phase.HEADLINE
    assert state.turn == 1
