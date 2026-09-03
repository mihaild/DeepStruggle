import pytest
from starlette.testclient import TestClient
from web.server.main import app
from bot import HeuristicBot, RandomBot
import ts_engine
import json
import os

client = TestClient(app)

def test_rest_api_metadata():
    map_res = client.get("/api/metadata/map")
    assert map_res.status_code == 200
    map_data = map_res.json()
    assert len(map_data["countries"]) == 84

    cards_res = client.get("/api/metadata/cards")
    assert cards_res.status_code == 200
    cards_data = cards_res.json()
    assert len(cards_data) == 110

def test_rest_api_game_creation():
    res = client.post("/api/games/new", json={"game_id": "game-api-test", "seed": 999})
    assert res.status_code == 200
    data = res.json()
    assert data["game_id"] == "game-api-test"
    assert data["seed"] == 999
    assert data["state"]["turn"] == 1

def test_action_cancellation_and_undo():
    game_id = "game-undo-test"
    res = client.post("/api/games/new", json={"game_id": game_id, "seed": 100})
    assert res.status_code == 200

    with client.websocket_connect(f"/ws/game/{game_id}?role=USSR") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "STATE_UPDATE"
        init_state = msg["state"]

        # USSR Setup: East Germany starts at 3
        assert init_state["countries"]["East Germany"]["ussr_influence"] == 3
        assert init_state["step_index"] == 0

        # Step 1: Place 1 in East Germany (ID 14)
        ws.send_json({
            "type": "PLAY_ACTION",
            "action": {"decision_type": int(ts_engine.DecisionType.POINT_NODE), "primary_id": 14, "secondary_id": 0, "flags": 0}
        })
        msg1 = ws.receive_json()
        state1 = msg1["state"]
        assert state1["countries"]["East Germany"]["ussr_influence"] == 4
        assert state1["step_index"] == 1
        assert state1["can_undo"] is True

        # Step 2: Place 1 in Poland (ID 15)
        ws.send_json({
            "type": "PLAY_ACTION",
            "action": {"decision_type": int(ts_engine.DecisionType.POINT_NODE), "primary_id": 15, "secondary_id": 0, "flags": 0}
        })
        msg2 = ws.receive_json()
        state2 = msg2["state"]
        assert state2["countries"]["Poland"]["ussr_influence"] == 1
        assert state2["step_index"] == 2

        # Send CANCEL_ACTION via WebSocket to undo Step 2
        ws.send_json({"type": "CANCEL_ACTION"})
        msg_undo = ws.receive_json()
        state_undo = msg_undo["state"]
        assert state_undo["step_index"] == 1
        assert state_undo["countries"]["Poland"]["ussr_influence"] == 0
        assert state_undo["countries"]["East Germany"]["ussr_influence"] == 4

        # Send undo via REST API to undo Step 1
        rest_undo = client.post(f"/api/games/{game_id}/undo")
        assert rest_undo.status_code == 200
        state_rest = rest_undo.json()["state"]
        assert state_rest["step_index"] == 0
        assert state_rest["countries"]["East Germany"]["ussr_influence"] == 3

def test_bot_vs_bot_simulation():
    # Simulate a fast game using TestClient WebSocket
    game_id = "bot-sim-test"
    client.post("/api/games/new", json={"game_id": game_id, "seed": 42})

    ussr_bot = HeuristicBot("USSR")
    us_bot = HeuristicBot("US")

    with client.websocket_connect(f"/ws/game/{game_id}?role=OBSERVER") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "STATE_UPDATE"
        state = msg["state"]

        steps_played = 0
        max_steps = 150 # Run up to 150 micro-decision steps

        while not state.get("is_terminal") and steps_played < max_steps:
            legal = state.get("legal_actions", {})
            decision_player = legal.get("decision_player", "")
            d_type = legal.get("decision_type", 0)

            if d_type == 0:
                break

            bot = ussr_bot if decision_player == "USSR" else us_bot
            action = bot.select_action(state, legal)
            assert action is not None, f"Bot failed to select action for player {decision_player}, type {d_type}"

            ws.send_json({
                "type": "PLAY_ACTION",
                "action": action
            })

            msg = ws.receive_json()
            assert msg["type"] == "STATE_UPDATE"
            state = msg["state"]
            steps_played += 1

        assert steps_played > 20 # Completed at least Setup + Headline + multiple AR steps
        print(f"Simulation completed {steps_played} steps successfully. Turn: {state['turn']}, Phase: {state['current_phase']}")

def test_replay_loading():
    replays_res = client.get("/api/replays")
    assert replays_res.status_code == 200
    replays = replays_res.json()
    assert isinstance(replays, list)

def test_die_roll_logged_in_action_stream():
    """Validates that structured die rolls (e.g. Korean War, Coups) are explicitly recorded in session action logs."""
    from web.server.session import describe_action_and_deltas
    
    state_before = {
        "decision_context": {
            "decision_player": "USSR",
            "pending_op_card": 11, # Korean War (#11)
            "resolving_card": 0,
            "op_mode": 0
        },
        "countries": {"South Korea": {"us_influence": 2, "ussr_influence": 0}},
        "victory_points": 0,
        "die_roll": {"type": "NONE"}
    }
    
    state_after = {
        "decision_context": {"decision_player": "USSR"},
        "countries": {"South Korea": {"us_influence": 0, "ussr_influence": 2}},
        "victory_points": -2,
        "die_roll": {
            "type": "WAR_EVENT",
            "type_id": 4,
            "card_id": 11,
            "card_name": "Korean War",
            "roller": "USSR",
            "country_id": 46,
            "country_name": "South Korea",
            "roll1": 5,
            "mod1": 0,
            "total1": 5,
            "mod2": 4,
            "success": True,
            "net_delta": 2
        }
    }
    
    action = ts_engine.MicroAction(ts_engine.DecisionType.SELECT_PLAY_MODE, 0, 0, 0)
    delta_lines = describe_action_and_deltas(state_before, state_after, action)
    
    # Assert that die roll is explicitly present in the action log details
    roll_logs = [line for line in delta_lines if "Korean War Roll" in line or "5" in line]
    assert len(roll_logs) > 0
    assert any("Korean War Roll targeting South Korea: USSR rolls 5" in line for line in delta_lines)


def test_vp_delta_logged_with_signs():
    from web.server.session import describe_action_and_deltas

    # Case 1: USSR gains 2 VP (VP: 0 -> -2)
    state_before = {
        "decision_context": {"decision_player": "USSR"},
        "victory_points": 0,
        "countries": {},
        "die_roll": {"type": "NONE"}
    }
    state_after = {
        "decision_context": {"decision_player": "USSR"},
        "victory_points": -2,
        "countries": {},
        "die_roll": {"type": "NONE"}
    }
    action = ts_engine.MicroAction(ts_engine.DecisionType.SELECT_PLAY_MODE, 0, 0, 0)
    delta_lines = describe_action_and_deltas(state_before, state_after, action)
    vp_logs = [l for l in delta_lines if "Victory Points:" in l]
    assert len(vp_logs) == 1
    assert "(-2 VP)" in vp_logs[0]

    # Case 2: US gains 3 VP (VP: -1 -> +2)
    state_before_us = {
        "decision_context": {"decision_player": "US"},
        "victory_points": -1,
        "countries": {},
        "die_roll": {"type": "NONE"}
    }
    state_after_us = {
        "decision_context": {"decision_player": "US"},
        "victory_points": 2,
        "countries": {},
        "die_roll": {"type": "NONE"}
    }
    delta_lines_us = describe_action_and_deltas(state_before_us, state_after_us, action)
    vp_logs_us = [l for l in delta_lines_us if "Victory Points:" in l]
    assert len(vp_logs_us) == 1
    assert "(+3 VP)" in vp_logs_us[0]


@pytest.mark.anyio
async def test_session_handle_action_resolves_die_roll():
    from web.server.session import GameSession
    session = GameSession("test-die-roll-game", seed=42)

    # USSR setup
    await session.handle_action({"decision_type": 5, "primary_id": 14, "secondary_id": 0, "flags": 0})
    await session.handle_action({"decision_type": 5, "primary_id": 13, "secondary_id": 0, "flags": 0})
    await session.handle_action({"decision_type": 5, "primary_id": 0, "secondary_id": 0, "flags": 128})

    # US setup
    await session.handle_action({"decision_type": 5, "primary_id": 6, "secondary_id": 0, "flags": 0})
    await session.handle_action({"decision_type": 5, "primary_id": 7, "secondary_id": 0, "flags": 0})
    await session.handle_action({"decision_type": 5, "primary_id": 0, "secondary_id": 0, "flags": 128})

    # Headline phase
    await session.handle_action({"decision_type": 1, "primary_id": 24, "secondary_id": 0, "flags": 0})
    await session.handle_action({"decision_type": 1, "primary_id": 25, "secondary_id": 0, "flags": 0})

    while session.state.to_dict()['current_phase_name'] == 'HEADLINE' or session.state.ctx().decision_type != ts_engine.DecisionType.SELECT_CARD:
        valids = session.state.to_dict()['legal_actions']['valid_ids']
        if not valids: break
        await session.handle_action({"decision_type": int(session.state.ctx().decision_type), "primary_id": valids[0], "secondary_id": 0, "flags": 0})

    # Play Ops -> Coup
    c_ussr = list(session.state.to_dict()['hands']['USSR'])[0]
    await session.handle_action({"decision_type": 1, "primary_id": c_ussr, "secondary_id": 0, "flags": 0})
    await session.handle_action({"decision_type": 2, "primary_id": 1, "secondary_id": 0, "flags": 0})
    await session.handle_action({"decision_type": 4, "primary_id": 1, "secondary_id": 0, "flags": 0})

    # Execute Coup in country 40 with manual roll 5
    await session.handle_action({"decision_type": 5, "primary_id": 40, "secondary_id": 5, "flags": 0})

    # Verify that the die roll was logged in session.action_logs
    coup_log = session.action_logs[-1]
    details = coup_log.get("details", [])
    assert any("🎲 Coup in" in d and "rolls 5" in d for d in details), f"Die roll not found in details: {details}"
