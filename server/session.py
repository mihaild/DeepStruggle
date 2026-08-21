import os
import sys
import json
import random
import logging
from typing import Dict, List, Optional, Set, Any
from fastapi import WebSocket
import ts_engine
from server.replay import ReplayLogger

logger = logging.getLogger("ts_server.session")

def get_action_description(state_before: dict, action: ts_engine.MicroAction) -> str:
    """Generates a human-friendly narrative for a MicroAction."""
    d_type = action.decision_type
    primary = action.primary_id
    secondary = action.secondary_id
    p = state_before.get("decision_context", {}).get("decision_player", "NONE")

    if d_type == ts_engine.DecisionType.SELECT_CARD:
        card_name = ts_engine.CardData.get_card_name(primary)
        phase_name = state_before.get("current_phase_name", "")
        if phase_name == "HEADLINE":
            return f"{p} commits Headline Card: {card_name} (#{primary})"
        return f"{p} plays Card: {card_name} (#{primary})"

    elif d_type == ts_engine.DecisionType.POINT_NODE:
        if action.is_confirm_done():
            return f"{p} confirms / finishes point node selections."
        c_name = ts_engine.MapData.get_country_name(primary)
        pending_card = state_before.get("decision_context", {}).get("pending_op_card", 0)
        resolving_card = state_before.get("decision_context", {}).get("resolving_card", 0)
        phase_name = state_before.get("current_phase_name", "")

        if phase_name == "SETUP":
            return f"{p} places 1 Influence in {c_name} during Setup."
        if resolving_card > 0:
            res_name = ts_engine.CardData.get_card_name(resolving_card)
            return f"{p} targets {c_name} for Event: {res_name} (#{resolving_card})."
        return f"{p} targets {c_name} for Operations."

    elif d_type == ts_engine.DecisionType.SELECT_PLAY_MODE:
        modes = {0: "EVENT", 1: "OPERATIONS", 2: "SPACE RACE", 3: "PASS"}
        return f"{p} selects play mode: {modes.get(primary, str(primary))}"

    elif d_type == ts_engine.DecisionType.CHOOSE_TIMING_BRANCH:
        branches = {0: "OPS FIRST (Opponent Event Second)", 1: "OPPONENT EVENT FIRST (Ops Second)"}
        return f"{p} chooses timing branch: {branches.get(primary, str(primary))}"

    elif d_type == ts_engine.DecisionType.SELECT_OP_MODE:
        op_modes = {0: "INFLUENCE PLACEMENT", 1: "COUP ATTEMPT", 2: "REALIGNMENT"}
        return f"{p} chooses Op mode: {op_modes.get(primary, str(primary))}"

    elif d_type == ts_engine.DecisionType.CHOOSE_BRANCH:
        if action.is_confirm_done():
            return f"{p} confirms / passes option."
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
        self.history_snapshots: List[ts_engine.GameState] = [] # Snapshot history for undo / cancellation
        self.replay_logger = ReplayLogger(game_id, self.seed, us_player, ussr_player)

        self.connections: Set[WebSocket] = set()
        self.role_connections: Dict[str, Set[WebSocket]] = {
            "US": set(),
            "USSR": set(),
            "OBSERVER": set()
        }

        logger.info(f"[{self.game_id}] Game session initialized with seed {self.seed}. US: '{us_player}', USSR: '{ussr_player}'")

        # Log initial state
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
        d["action_logs"] = self.action_logs[-40:] # Last 40 logs for UI stream
        d["can_undo"] = len(self.history_snapshots) > 0
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

        logger.info(f"[{self.game_id}] Client connected as role '{role}'. Active connections: {len(self.connections)}")

        # Send full initial state immediately
        await websocket.send_json({
            "type": "STATE_UPDATE",
            "state": self.get_state_dict()
        })

    def disconnect(self, websocket: WebSocket):
        self.connections.discard(websocket)
        for r_set in self.role_connections.values():
            r_set.discard(websocket)
        logger.info(f"[{self.game_id}] Client disconnected. Remaining connections: {len(self.connections)}")

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
            except Exception as e:
                logger.debug(f"[{self.game_id}] Error broadcasting to client: {e}")
                dead_sockets.add(ws)
        for ws in dead_sockets:
            self.disconnect(ws)

    async def handle_action(self, action_dict: dict, sender_role: str = "OBSERVER") -> bool:
        """Executes a micro-action on the simulation engine, saves state snapshot for undo, and broadcasts."""
        if ts_engine.Engine.is_terminal(self.state):
            logger.warning(f"[{self.game_id}] Ignored action on terminal game state.")
            return False

        try:
            d_type = ts_engine.DecisionType(action_dict["decision_type"])
            primary = int(action_dict.get("primary_id", 0))
            secondary = int(action_dict.get("secondary_id", 0))
            flags = int(action_dict.get("flags", 0))
        except (KeyError, ValueError) as e:
            logger.warning(f"[{self.game_id}] Invalid action payload {action_dict}: {e}")
            return False

        ctx = self.state.ctx()
        logger.debug(f"[{self.game_id}] Incoming action: type={int(d_type)} primary={primary} secondary={secondary} flags={flags} (Current ctx: type={int(ctx.decision_type)} player={ctx.decision_player})")

        # Validate decision type against active context
        if ctx.decision_type != d_type:
            logger.warning(f"[{self.game_id}] Decision type mismatch: expected {int(ctx.decision_type)}, got {int(d_type)}")
            return False

        # Save snapshot for undo / cancel
        snapshot = self.state.clone()
        if len(self.history_snapshots) >= 300:
            self.history_snapshots.pop(0)
        self.history_snapshots.append(snapshot)

        state_before = self.state.to_dict()
        action = ts_engine.MicroAction(d_type, primary, secondary, flags)

        success = ts_engine.Engine.step(self.state, action)
        if not success:
            logger.warning(f"[{self.game_id}] Engine rejected action step: {action_dict}")
            self.history_snapshots.pop() # Remove snapshot on failed step
            return False

        self.step_index += 1
        description = get_action_description(state_before, action)
        logger.info(f"[{self.game_id}] Step {self.step_index} [Turn {self.state.turn} AR {self.state.action_round}]: {description}")

        log_entry = {
            "step_index": self.step_index,
            "turn": self.state.turn,
            "ar": self.state.action_round,
            "phase": str(state_before.get("current_phase_name", "ACTION")),
            "player": state_before.get("decision_context", {}).get("decision_player", "NONE"),
            "text": description
        }
        self.action_logs.append(log_entry)

        # Log to replay
        self.replay_logger.log_step(
            step_index=self.step_index,
            turn=self.state.turn,
            ar=self.state.action_round,
            phase=str(state_before.get("current_phase_name", "ACTION")),
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
            logger.info(f"[{self.game_id}] Game reached terminal state! Winner: {winner}, VP: {self.state.victory_points}, Reason: {reason}")
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

    async def handle_cancel_action(self) -> bool:
        """Cancels/undoes the last action and rolls back to previous state snapshot."""
        if not self.history_snapshots:
            logger.warning(f"[{self.game_id}] Cannot cancel action: snapshot history is empty.")
            return False

        self.state = self.history_snapshots.pop()
        if self.step_index > 0:
            self.step_index -= 1

        last_desc = ""
        if self.action_logs:
            popped = self.action_logs.pop()
            last_desc = popped.get("text", "")

        logger.info(f"[{self.game_id}] Action cancelled: '{last_desc}'. Rolled back to step {self.step_index}.")

        self.action_logs.append({
            "step_index": self.step_index,
            "turn": self.state.turn,
            "ar": self.state.action_round,
            "phase": str(self.state.to_dict().get("current_phase_name", "ACTION")),
            "player": "SYSTEM",
            "text": f"↺ Action cancelled (Rolled back to Step {self.step_index})"
        })

        await self.broadcast_state()
        return True

    async def handle_debug_override(self, override_dict: dict):
        """Allows direct tweaking of influence, DEFCON, VP, etc. for engine testing."""
        op = override_dict.get("op")
        logger.info(f"[{self.game_id}] Applying debug override: {override_dict}")

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
