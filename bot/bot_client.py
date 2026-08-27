from web.server.replay_types import WebSocketMessageDict, WebSocketActionPayloadDict
from bot.base_bot import BaseBot
from bot.random_bot import RandomBot
from bot.heuristic_bot import HeuristicBot
try:
    from bot.neural_bot import NeuralBot
except ImportError:
    NeuralBot = None
import argparse
import asyncio
import json
import random
import sys
from typing import Dict, Any, Optional
import websockets

import os
try:
    import ts_engine
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _build = os.path.join(_root, "build")
    if os.path.exists(_build) and _build not in sys.path:
        sys.path.insert(0, _build)
    import ts_engine

async def run_bot_client(server_url: str, game_id: str, role: str, bot_type: str = "heuristic", delay: float = 0.3, model_path: str = None):
    if bot_type == "neural":
        if NeuralBot is None:
            raise ImportError("NeuralBot is not available. Please ensure PyTorch and AI modules are installed.")
        bot = NeuralBot(role, model_path=model_path)
    elif bot_type == "heuristic":
        bot = HeuristicBot(role)
    else:
        bot = RandomBot(role)
    ws_uri = f"{server_url}/ws/game/{game_id}?role={role}"

    print(f"Connecting {bot_type.upper()} bot for {role} to {ws_uri}...")
    async with websockets.connect(ws_uri) as websocket:
        print(f"Connected as {role}!")
        while True:
            try:
                msg = await websocket.recv()
                data = json.loads(msg)
                if data.get("type") == "STATE_UPDATE":
                    state = data.get("state", {})
                    if state.get("is_terminal"):
                        print(f"Game over! Terminal utility: {state.get('terminal_utility')}")
                        break

                    legal_actions = state.get("legal_actions", {})
                    decision_player = legal_actions.get("decision_player", "")

                    if decision_player == role:
                        if delay > 0:
                            await asyncio.sleep(delay)
                        action = bot.select_action(state, legal_actions)
                        if action:
                            play_msg: WebSocketMessageDict = {"type": "PLAY_ACTION", "action": action}
                            await websocket.send(json.dumps(play_msg))
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twilight Struggle Bot Client")
    parser.add_argument("--server", type=str, default="ws://localhost:8000", help="WebSocket server URI")
    parser.add_argument("--game-id", type=str, default="game-1", help="Game ID to join")
    parser.add_argument("--role", type=str, default="USSR", choices=["US", "USSR"], help="Player side")
    parser.add_argument("--type", type=str, default="heuristic", choices=["random", "heuristic", "neural"], help="Bot strategy")
    parser.add_argument("--model-path", type=str, default="checkpoints/coldwar_net.pt", help="Path to trained model checkpoint")
    parser.add_argument("--delay", type=float, default=0.2, help="Artificial delay in seconds between moves")

    args = parser.parse_args()
    asyncio.run(run_bot_client(args.server, args.game_id, args.role, args.type, args.delay, args.model_path))
