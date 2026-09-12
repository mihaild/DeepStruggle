"""WebSocket Bot Client for playing against Twilight Struggle agents via the Web Workbench."""

from web.server.replay_types import WebSocketMessageDict, WebSocketActionPayloadDict
from bot import (
    BaseBot,
    RandomBot,
    HeuristicBot,
    ExploratoryBot,
    StrategicBot,
    EventHeavyBot,
)
try:
    from bot import NeuralBot
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


async def run_bot_client(
    server_url: str,
    game_id: str,
    role: str,
    bot_type: str = "heuristic",
    delay: float = 0.3,
    model_path: Optional[str] = None,
):
    bot: BaseBot
    if bot_type == "neural":
        if NeuralBot is None:
            raise ImportError("NeuralBot is not available. Please ensure PyTorch is installed.")
        bot = NeuralBot(role, model_path=model_path)
    elif bot_type == "heuristic":
        bot = HeuristicBot(role)
    elif bot_type == "strategic":
        bot = StrategicBot(role)
    elif bot_type == "event_heavy":
        bot = EventHeavyBot(role)
    elif bot_type == "exploratory":
        bot = ExploratoryBot(role)
    elif bot_type == "random":
        bot = RandomBot(role)
    else:
        raise ValueError(f"Unknown bot_type '{bot_type}'. Choose from neural, heuristic, strategic, event_heavy, exploratory, random.")

    ws_url = f"{server_url.replace('http', 'ws')}/ws/game/{game_id}?role={role}"
    print(f"[{role} Bot] Connecting to {ws_url} as {bot_type.upper()}...")

    async with websockets.connect(ws_url) as ws:
        print(f"[{role} Bot] Connected. Waiting for game updates...")
        async for message in ws:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "STATE_UPDATE":
                state = data.get("state", {})
                legal_actions = state.get("legal_actions", {})
                d_player = legal_actions.get("decision_player")

                if d_player == role:
                    if delay > 0:
                        await asyncio.sleep(delay)

                    action = bot.select_action(state, legal_actions)
                    if action is not None:
                        payload = {
                            "type": "PLAY_ACTION",
                            "action": action,
                        }
                        await ws.send(json.dumps(payload))
                    else:
                        print(f"[{role} Bot] No valid action selected.")

            elif msg_type == "ERROR":
                print(f"[{role} Bot] Received Server Error: {data.get('message')}")


def main():
    parser = argparse.ArgumentParser(description="Twilight Struggle WebSocket Bot Client")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="FastAPI Server URL")
    parser.add_argument("--game-id", type=str, required=True, help="Game session ID")
    parser.add_argument("--role", type=str, choices=["US", "USSR"], required=True, help="Player superpower")
    parser.add_argument(
        "--type",
        type=str,
        default="heuristic",
        choices=["heuristic", "random", "neural", "strategic", "event_heavy", "exploratory"],
        help="Bot strategy type",
    )
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained ColdWarNet checkpoint (for neural bot)")
    parser.add_argument("--delay", type=float, default=0.2, help="Artificial delay in seconds between bot moves")

    args = parser.parse_args()
    asyncio.run(
        run_bot_client(
            args.server_url,
            args.game_id,
            args.role,
            bot_type=args.type,
            delay=args.delay,
            model_path=args.model_path,
        )
    )


if __name__ == "__main__":
    main()
