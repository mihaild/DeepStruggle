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
