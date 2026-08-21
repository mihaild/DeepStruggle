import asyncio
import json
import random
from typing import Dict, List, Optional, Set
from fastapi import WebSocket

import ts_engine
from server.replay import ReplayLogger

def get_action_description(state_before: dict, action: ts_engine.MicroAction) -> str:
    ctx = state_before.get("decision_context", {})
    p = ctx.get("decision_player", "NONE")
    d_type = action.decision_type
    primary = action.primary_id
    secondary = action.secondary_id

    if d_type == ts_engine.DecisionType.POINT_NODE:
        if action.is_confirm_done():
            return f"{p} finishes multi-placement / passes."
        c_name = ts_engine.MapData.get_country_name(primary)
        return f"{p} targets country: {c_name}"

    elif d_type == ts_engine.DecisionType.SELECT_CARD:
        c_info = ts_engine.CardData.get_card_info(primary) if 1 <= primary <= 110 else {}
        card_name = c_info.get("name", f"Card #{primary}")
        return f"{p} selects card #{primary} ({card_name})"

    elif d_type == ts_engine.DecisionType.SELECT_PLAY_MODE:
        modes = {0: "EVENT", 1: "OPERATIONS", 2: "SPACE RACE", 3: "PASS"}
        return f"{p} chooses mode: {modes.get(primary, str(primary))}"

    elif d_type == ts_engine.DecisionType.CHOOSE_TIMING_BRANCH:
        branches = {0: "OPS FIRST (Event Second)", 1: "EVENT FIRST (Ops Second)"}
        return f"{p} chooses timing: {branches.get(primary, str(primary))}"

    elif d_type == ts_engine.DecisionType.SELECT_OP_MODE:
        op_modes = {0: "INFLUENCE PLACEMENT", 1: "COUP ATTEMPT", 2: "REALIGNMENT"}
        return f"{p} chooses Op mode: {op_modes.get(primary, str(primary))}"

    elif d_type == ts_engine.DecisionType.CHOOSE_BRANCH:
        if action.is_confirm_done():
            return f"{p} confirms / passes."
        return f"{p} chooses option: {primary}"

    return f"{p} action: type={int(d_type)} primary={primary}"

class GameSession:
    def __init__(self, game_id: str, seed: Optional[int] = None, us_player: str = "US", ussr_player: str = "USSR"):
        self.game_id = game_id
        self.seed = seed if seed is not None else random.randint(1, 1_000_000_000)
        self.us_player = us_player
        self.ussr_player = ussr_player

        self.state = ts_engine.GameState()
        ts_engine.Engine.init_game(self.state, self.seed)

        self.step_index = 0
        self.action_logs: List[Dict[str, Any]] = []
        self.replay_logger = ReplayLogger(game_id, self.seed, us_player, ussr_player)

        self.connections: Set[WebSocket] = set()
        self.role_connections: Dict[str, Set[WebSocket]] = {
            "US": set(),
            "USSR": set(),
            "OBSERVER": set()
        }

        # Log initial state
        initial_dict = self.state.to_dict()
        self.action_logs.append({
            "step_index": 0,
            "turn": self.state.turn,
            "ar": self.state.action_round,
            "phase": "SETUP",
            "player": "SYSTEM",
            "text": f"Game started (Seed: {self.seed}). USSR setup: Place 6 Influence in Eastern Europe."
        })

    def get_state_dict(self) -> dict:
        d = self.state.to_dict()
        d["game_id"] = self.game_id
        d["seed"] = self.seed
        d["step_index"] = self.step_index
        d["action_logs"] = self.action_logs[-30:] # Last 30 logs for UI stream
        d["players"] = {
            "US": self.us_player,
            "USSR": self.ussr_player
        }
        return d

    async def connect(self, websocket: WebSocket, role: str):
        await websocket.accept()
        self.connections.add(websocket)
        role = role.upper()
        if role in self.role_connections:
            self.role_connections[role].add(websocket)
        else:
            self.role_connections["OBSERVER"].add(websocket)

        # Send full initial state immediately
        await websocket.send_json({
            "type": "STATE_UPDATE",
            "state": self.get_state_dict()
        })

    def disconnect(self, websocket: WebSocket):
        self.connections.discard(websocket)
        for r_set in self.role_connections.values():
            r_set.discard(websocket)

    async def broadcast_state(self):
        if not self.connections:
            return
        payload = {
            "type": "STATE_UPDATE",
            "state": self.get_state_dict()
        }
        dead_sockets = set()
        for ws in self.connections:
            try:
                await ws.send_json(payload)
            except Exception:
                dead_sockets.add(ws)
        for ws in dead_sockets:
            self.disconnect(ws)

    async def handle_action(self, action_dict: dict, sender_role: str = "OBSERVER") -> bool:
        """Executes a micro-action on the simulation engine and notifies all connected clients."""
        if ts_engine.Engine.is_terminal(self.state):
            return False

        try:
            d_type = ts_engine.DecisionType(action_dict["decision_type"])
            primary = int(action_dict.get("primary_id", 0))
            secondary = int(action_dict.get("secondary_id", 0))
            flags = int(action_dict.get("flags", 0))
        except (KeyError, ValueError) as e:
            return False

        # Validate decision type against active context
        ctx = self.state.ctx()
        if ctx.decision_type != d_type:
            return False

        state_before = self.state.to_dict()
        action = ts_engine.MicroAction(d_type, primary, secondary, flags)

        success = ts_engine.Engine.step(self.state, action)
        if not success:
            return False

        self.step_index += 1
        description = get_action_description(state_before, action)
        log_entry = {
            "step_index": self.step_index,
            "turn": self.state.turn,
            "ar": self.state.action_round,
            "phase": str(self.state.current_phase),
            "player": state_before.get("decision_context", {}).get("decision_player", "NONE"),
            "text": description
        }
        self.action_logs.append(log_entry)

        # Log to replay
        self.replay_logger.log_step(
            step_index=self.step_index,
            turn=self.state.turn,
            ar=self.state.action_round,
            phase=str(self.state.current_phase),
            player=log_entry["player"],
            action={"decision_type": int(d_type), "primary_id": primary, "secondary_id": secondary, "flags": flags},
            description=description,
            state_snapshot=self.state.to_dict()
        )

        # Check terminal state
        if ts_engine.Engine.is_terminal(self.state):
            util = ts_engine.Engine.get_terminal_utility(self.state)
            winner = "US" if util > 0 else ("USSR" if util < 0 else "DRAW")
            reason = "Victory Point Threshold (±20)" if abs(self.state.victory_points) >= 20 else "Game Over"
            self.replay_logger.set_result(winner, int(self.state.victory_points), int(self.state.turn), reason)
            self.replay_logger.save()

            self.action_logs.append({
                "step_index": self.step_index + 1,
                "turn": self.state.turn,
                "ar": self.state.action_round,
                "phase": "GAME_OVER",
                "player": "SYSTEM",
                "text": f"Game Over! Winner: {winner} (VP: {self.state.victory_points}, Turn: {self.state.turn})"
            })

        await self.broadcast_state()
        return True

    async def handle_debug_override(self, override_dict: dict):
        """Allows direct tweaking of influence, DEFCON, VP, etc. for engine testing."""
        op = override_dict.get("op")
        if op == "set_country":
            cid = int(override_dict["country_id"])
            us = int(override_dict["us"])
            ussr = int(override_dict["ussr"])
            self.state.set_country(cid, us, ussr)
            c_name = ts_engine.MapData.get_country_name(cid)
            self.action_logs.append({
                "step_index": self.step_index,
                "turn": self.state.turn,
                "ar": self.state.action_round,
                "phase": "DEBUG",
                "player": "DEBUG",
                "text": f"[DEBUG] Set {c_name} Influence -> US: {us}, USSR: {ussr}"
            })
        elif op == "set_defcon":
            self.state.defcon = int(override_dict["defcon"])
            self.action_logs.append({
                "step_index": self.step_index,
                "turn": self.state.turn,
                "ar": self.state.action_round,
                "phase": "DEBUG",
                "player": "DEBUG",
                "text": f"[DEBUG] Set DEFCON -> {self.state.defcon}"
            })
        elif op == "set_vp":
            self.state.victory_points = int(override_dict["vp"])
            self.action_logs.append({
                "step_index": self.step_index,
                "turn": self.state.turn,
                "ar": self.state.action_round,
                "phase": "DEBUG",
                "player": "DEBUG",
                "text": f"[DEBUG] Set VP -> {self.state.victory_points}"
            })

        await self.broadcast_state()
