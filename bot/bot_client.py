import argparse
import asyncio
import json
import random
import sys
from typing import Dict, Any, Optional
import websockets

import ts_engine

class BaseBot:
    def __init__(self, role: str):
        self.role = role.upper() # "US" or "USSR"

    def select_action(self, state: dict, legal_actions: dict) -> Optional[dict]:
        raise NotImplementedError

class RandomBot(BaseBot):
    def select_action(self, state: dict, legal_actions: dict) -> Optional[dict]:
        valid_ids = legal_actions.get("valid_ids", [])
        d_type = legal_actions.get("decision_type", 0)
        allow_early_stop = legal_actions.get("allow_early_stop", False)

        if not valid_ids and not allow_early_stop:
            return None

        # 10% chance to stop early if allowed
        if allow_early_stop and random.random() < 0.15:
            return {
                "decision_type": d_type,
                "primary_id": 0,
                "secondary_id": 0,
                "flags": 0x80 # CONFIRM_DONE
            }

        if not valid_ids:
            if allow_early_stop:
                return {
                    "decision_type": d_type,
                    "primary_id": 0,
                    "secondary_id": 0,
                    "flags": 0x80
                }
            return None

        choice = random.choice(valid_ids)
        return {
            "decision_type": d_type,
            "primary_id": choice,
            "secondary_id": 0,
            "flags": 0
        }

class HeuristicBot(BaseBot):
    def select_action(self, state: dict, legal_actions: dict) -> Optional[dict]:
        valid_ids = legal_actions.get("valid_ids", [])
        d_type = legal_actions.get("decision_type", 0)
        allow_early_stop = legal_actions.get("allow_early_stop", False)

        if not valid_ids:
            if allow_early_stop:
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0x80}
            return None

        # 1. SETUP: Prefer key battlegrounds
        if state.get("current_phase") == 0: # SETUP
            if self.role == "USSR":
                # East Germany (14), Poland (15)
                for preferred in [14, 15]:
                    if preferred in valid_ids:
                        return {"decision_type": d_type, "primary_id": preferred, "secondary_id": 0, "flags": 0}
            elif self.role == "US":
                # West Germany (6), Italy (10)
                for preferred in [6, 10]:
                    if preferred in valid_ids:
                        return {"decision_type": d_type, "primary_id": preferred, "secondary_id": 0, "flags": 0}

        # 2. SELECT_CARD: Play scoring cards if holding them or highest Ops card
        if d_type == 1: # SELECT_CARD
            # Check for scoring cards
            for cid in valid_ids:
                if 1 <= cid <= 110:
                    info = ts_engine.CardData.get_card_info(cid)
                    if info.get("is_scoring"):
                        return {"decision_type": d_type, "primary_id": cid, "secondary_id": 0, "flags": 0}

            # Else pick highest Ops card
            best_cid = valid_ids[0]
            best_ops = -1
            for cid in valid_ids:
                if 1 <= cid <= 110:
                    info = ts_engine.CardData.get_card_info(cid)
                    if info.get("ops", 0) > best_ops:
                        best_ops = info.get("ops", 0)
                        best_cid = cid
            return {"decision_type": d_type, "primary_id": best_cid, "secondary_id": 0, "flags": 0}

        # 3. SELECT_PLAY_MODE: If friendly event, play as Event (0), else Ops (1)
        if d_type == 2: # SELECT_PLAY_MODE
            ctx = state.get("decision_context", {})
            card_id = ctx.get("pending_op_card", 0)
            if 1 <= card_id <= 110:
                info = ts_engine.CardData.get_card_info(card_id)
                if info.get("side") == self.role and 0 in valid_ids:
                    return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}
            if 1 in valid_ids: # OPS
                return {"decision_type": d_type, "primary_id": 1, "secondary_id": 0, "flags": 0}

        # 4. CHOOSE_TIMING_BRANCH: OPS_FIRST (0)
        if d_type == 3: # CHOOSE_TIMING_BRANCH
            return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}

        # 5. SELECT_OP_MODE: Prefer Influence (0) unless DEFCON allows safe Coup (1)
        if d_type == 4: # SELECT_OP_MODE
            if 0 in valid_ids: # INFLUENCE
                return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": 0}

        # 6. POINT_NODE: Prefer battlegrounds
        if d_type == 5: # POINT_NODE
            for node_id in valid_ids:
                if 0 <= node_id < 84:
                    c_info = ts_engine.MapData.get_country_info(node_id)
                    if c_info.get("battleground"):
                        return {"decision_type": d_type, "primary_id": node_id, "secondary_id": 0, "flags": 0}

        # Fallback to random
        return {
            "decision_type": d_type,
            "primary_id": random.choice(valid_ids),
            "secondary_id": 0,
            "flags": 0
        }

async def run_bot_client(server_url: str, game_id: str, role: str, bot_type: str = "heuristic", delay: float = 0.3):
    bot = HeuristicBot(role) if bot_type == "heuristic" else RandomBot(role)
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
                            await websocket.send(json.dumps({
                                "type": "PLAY_ACTION",
                                "action": action
                            }))
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twilight Struggle Bot Client")
    parser.add_argument("--server", type=str, default="ws://localhost:8000", help="WebSocket server URI")
    parser.add_argument("--game-id", type=str, default="game-1", help="Game ID to join")
    parser.add_argument("--role", type=str, default="USSR", choices=["US", "USSR"], help="Player side")
    parser.add_argument("--type", type=str, default="heuristic", choices=["random", "heuristic"], help="Bot strategy")
    parser.add_argument("--delay", type=float, default=0.2, help="Artificial delay in seconds between moves")

    args = parser.parse_args()
    asyncio.run(run_bot_client(args.server, args.game_id, args.role, args.type, args.delay))
