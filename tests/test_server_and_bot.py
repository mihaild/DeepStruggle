import pytest
from starlette.testclient import TestClient
from server.main import app
from bot.bot_client import HeuristicBot, RandomBot
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
