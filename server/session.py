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

def describe_action_and_deltas(state_before: dict, state_after: dict, action: ts_engine.MicroAction) -> List[str]:
    """Generates detailed human-readable log messages and state deltas for an executed action."""
    d_type = action.decision_type
    primary = action.primary_id
    secondary = action.secondary_id
    flags = action.flags
    p = state_before.get("decision_context", {}).get("decision_player", "NONE")
    logs = []

    # 1. Primary Action Narrative
    if d_type == ts_engine.DecisionType.SELECT_CARD:
        card_name = ts_engine.CardData.get_card_name(primary)
        phase_name = state_before.get("current_phase_name", "")
        if phase_name == "HEADLINE":
            logs.append(f"{p} commits Headline Card: {card_name} (#{primary})")
        else:
            logs.append(f"{p} plays Card: {card_name} (#{primary})")

    elif d_type == ts_engine.DecisionType.POINT_NODE:
        if action.is_confirm_done():
            logs.append(f"{p} confirms / finishes point node selections.")
        else:
            c_name = ts_engine.MapData.get_country_name(primary)
            resolving_card = state_before.get("decision_context", {}).get("resolving_card", 0)
            phase_name = state_before.get("current_phase_name", "")

            if phase_name == "SETUP":
                logs.append(f"{p} places 1 Influence in {c_name} during Setup.")
            elif resolving_card > 0:
                res_name = ts_engine.CardData.get_card_name(resolving_card)
                logs.append(f"{p} targets {c_name} for Event: {res_name} (#{resolving_card})")
            else:
                # Standard Operations: Influence placement, Coup, or Realignment
                op_mode = state_before.get("decision_context", {}).get("op_mode", 0)
                if op_mode == 1:
                    logs.append(f"{p} attempts Coup in {c_name}")
                elif op_mode == 2:
                    logs.append(f"{p} conducts Realignment in {c_name}")
                else:
                    logs.append(f"{p} places Influence in {c_name}")

    elif d_type == ts_engine.DecisionType.SELECT_PLAY_MODE:
        modes = {0: "EVENT", 1: "OPERATIONS", 2: "SPACE RACE", 3: "PASS"}
        mode_str = modes.get(primary, str(primary))
        if primary == 2: # Space Race
            roll_info = f" (Input Roll: {secondary})" if secondary > 0 else ""
            logs.append(f"{p} attempts Space Race with pending card{roll_info}")
        else:
            logs.append(f"{p} selects play mode: {mode_str}")

    elif d_type == ts_engine.DecisionType.CHOOSE_TIMING_BRANCH:
        branches = {0: "OPS FIRST (Opponent Event Second)", 1: "OPPONENT EVENT FIRST (Ops Second)"}
        logs.append(f"{p} chooses timing branch: {branches.get(primary, str(primary))}")

    elif d_type == ts_engine.DecisionType.SELECT_OP_MODE:
        op_modes = {0: "INFLUENCE PLACEMENT", 1: "COUP ATTEMPT", 2: "REALIGNMENT"}
        logs.append(f"{p} chooses Op mode: {op_modes.get(primary, str(primary))}")

    elif d_type == ts_engine.DecisionType.CHOOSE_BRANCH:
        if action.is_confirm_done():
            logs.append(f"{p} confirms / passes option.")
        else:
            logs.append(f"{p} chooses option: {primary}")
    else:
        logs.append(f"{p} action: type={int(d_type)} primary={primary} secondary={secondary} flags={flags}")

    # 2. Influence deltas
    old_countries = state_before.get("countries", {})
    new_countries = state_after.get("countries", {})
    for c_name, new_data in new_countries.items():
        if c_name in old_countries:
            old_data = old_countries[c_name]
            us_diff = new_data["us_influence"] - old_data["us_influence"]
            ussr_diff = new_data["ussr_influence"] - old_data["ussr_influence"]
            if us_diff != 0 or ussr_diff != 0:
                parts = []
                if us_diff != 0:
                    parts.append(f"US: {old_data['us_influence']} -> {new_data['us_influence']} ({'+' if us_diff > 0 else ''}{us_diff})")
                if ussr_diff != 0:
                    parts.append(f"USSR: {old_data['ussr_influence']} -> {new_data['ussr_influence']} ({'+' if ussr_diff > 0 else ''}{ussr_diff})")
                logs.append(f"  • {c_name} Influence: " + ", ".join(parts))

    # 3. Track deltas (DEFCON, VP, MilOps)
    if state_before.get("defcon") != state_after.get("defcon"):
        logs.append(f"  • DEFCON: {state_before.get('defcon')} -> {state_after.get('defcon')}")
    if state_before.get("victory_points") != state_after.get("victory_points"):
        vp_before = state_before.get("victory_points", 0)
        vp_after = state_after.get("victory_points", 0)
        vp_str = f"+{vp_after} (US)" if vp_after > 0 else (f"{vp_after} (USSR)" if vp_after < 0 else "0 (Tie)")
        logs.append(f"  • Victory Points: {vp_before} -> {vp_str}")
    if state_before.get("us_mil_ops") != state_after.get("us_mil_ops"):
        logs.append(f"  • US MilOps: {state_before.get('us_mil_ops')} -> {state_after.get('us_mil_ops')}")
    if state_before.get("ussr_mil_ops") != state_after.get("ussr_mil_ops"):
        logs.append(f"  • USSR MilOps: {state_before.get('ussr_mil_ops')} -> {state_after.get('ussr_mil_ops')}")

    # 4. Card movement deltas
    old_locs = state_before.get("card_locations", {})
    new_locs = state_after.get("card_locations", {})
    for cid_str, new_loc in new_locs.items():
        old_loc = old_locs.get(cid_str)
        if old_loc and old_loc != new_loc:
            cid = int(cid_str)
            card_name = ts_engine.CardData.get_card_name(cid)
            logs.append(f"  • Card #{cid} ({card_name}) moved: {old_loc} -> {new_loc}")

    # 5. Die roll logging
    last_roll = state_after.get("last_die_roll", 0)
    last_opp_roll = state_after.get("last_opp_die_roll", 0)

    WAR_CARDS = {
        9: "Korean War",
        11: "Arab-Israeli War",
        23: "Indo-Pakistani War",
        36: "Brush War",
        45: "Summit",
        84: "Reagan Bombs Libya",
        102: "Iran-Iraq War",
        107: "Che",
        91: "Ortega Elected in Nicaragua"
    }

    resolving_card = state_before.get("decision_context", {}).get("resolving_card", 0)
    pending_card = state_before.get("decision_context", {}).get("pending_op_card", 0)
    op_mode = state_before.get("decision_context", {}).get("op_mode", 0)

    # Coup attempt
    if d_type == ts_engine.DecisionType.POINT_NODE and op_mode == 1 and not action.is_confirm_done():
        c_name = ts_engine.MapData.get_country_name(primary)
        c_stab = state_before.get("countries", {}).get(c_name, {}).get("stability", 1)
        ops_val = state_before.get("decision_context", {}).get("pending_ops_value", 0)
        total = last_roll + ops_val
        def_target = c_stab * 2
        net = total - def_target
        status_str = f"Net +{net} (Coup Succeeded)" if net > 0 else "Coup Failed (Roll + Ops <= 2x Stability)"
        logs.append(f"  🎲 Coup Roll in {c_name}: {last_roll} (+{ops_val} Ops = {total}) vs {def_target} Defense -> {status_str}")

    # Realignment
    elif d_type == ts_engine.DecisionType.POINT_NODE and op_mode == 2 and not action.is_confirm_done():
        c_name = ts_engine.MapData.get_country_name(primary)
        logs.append(f"  🎲 Realignment Rolls in {c_name}: US rolled {last_roll}, USSR rolled {last_opp_roll}")

    # War event play
    elif (d_type == ts_engine.DecisionType.SELECT_PLAY_MODE and primary == 0 and pending_card in WAR_CARDS) or          (d_type == ts_engine.DecisionType.POINT_NODE and resolving_card in WAR_CARDS and not action.is_confirm_done()):
        card_id = resolving_card if resolving_card in WAR_CARDS else pending_card
        war_name = WAR_CARDS.get(card_id, f"Card #{card_id}")
        if card_id == 45: # Summit
            logs.append(f"  🎲 Summit Die Rolls: US rolled {last_roll}, USSR rolled {last_opp_roll}")
        elif last_roll > 0:
            vp_diff = state_after.get("victory_points", 0) - state_before.get("victory_points", 0)
            status = "Success (Victory Points & Influence awarded)" if vp_diff != 0 else "Roll Failed"
            logs.append(f"  🎲 {war_name} Die Roll: {last_roll} -> {status}")

    # Space Race
    elif d_type == ts_engine.DecisionType.SELECT_PLAY_MODE and primary == 2:
        us_sp = state_after.get("space", {}).get("US", 0) - state_before.get("space", {}).get("US", 0)
        ussr_sp = state_after.get("space", {}).get("USSR", 0) - state_before.get("space", {}).get("USSR", 0)
        sp_success = (us_sp > 0 or ussr_sp > 0)
        new_step = state_after.get("space", {}).get("US" if p == "US" else "USSR", 0)
        status = f"Success! Advanced to Step {new_step}" if sp_success else "Failed (Roll exceeded required threshold)"
        logs.append(f"  🎲 Space Race Die Roll: {last_roll} -> {status}")

    # Bear Trap / Quagmire
    elif d_type == ts_engine.DecisionType.SELECT_CARD and (
        (p == "US" and "QUAGMIRE_ACTIVE" in state_before.get("persistent_effects_list", [])) or
        (p == "USSR" and "BEAR_TRAP_ACTIVE" in state_before.get("persistent_effects_list", []))
    ):
        if last_roll > 0:
            escaped = (last_roll <= 4)
            status = "Success! Discard canceled effect." if escaped else "Failed (Roll > 4, remains trapped)"
            logs.append(f"  🎲 Escape Die Roll: {last_roll} (Need 1-4) -> {status}")

    return logs

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
        d["action_logs"] = self.action_logs[-60:] # Last 60 logs for UI stream
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
        logger.info(f"[{self.game_id}] Action from {sender_role}: decision_type={int(ctx.decision_type)} ({int(d_type)}), primary={primary}, secondary={secondary}, flags={flags}")

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
        state_after = self.state.to_dict()
        delta_lines = describe_action_and_deltas(state_before, state_after, action)

        main_desc = delta_lines[0] if delta_lines else f"Action type={int(d_type)}"
        for line in delta_lines:
            logger.info(f"[{self.game_id}] Step {self.step_index} [Turn {self.state.turn} AR {self.state.action_round}]: {line}")

        log_entry = {
            "step_index": self.step_index,
            "turn": self.state.turn,
            "ar": self.state.action_round,
            "phase": str(state_before.get("current_phase_name", "ACTION")),
            "player": state_before.get("decision_context", {}).get("decision_player", "NONE"),
            "text": main_desc,
            "details": delta_lines[1:] if len(delta_lines) > 1 else []
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
            description=main_desc,
            state_snapshot=state_after
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
