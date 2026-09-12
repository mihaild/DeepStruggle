from web.server.replay_types import GameStateDict, ReplayActionDict
import os
import sys
import json
import random
import logging
import base64
from typing import Dict, List, Optional, Set, Any, cast

import numpy as np
from fastapi import WebSocket
try:
    import ts_engine
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _build = os.path.join(_root, "build")
    if os.path.exists(_build) and _build not in sys.path:
        sys.path.insert(0, _build)
    import ts_engine
from web.server.replay import ReplayLogger
from tools.lib.tournament_evaluator import classify_game_ending_reason

logger = logging.getLogger("ts_server.session")

def describe_action_and_deltas(state_before: Any, state_after: Any, action: ts_engine.MicroAction) -> List[str]:
    """Generates detailed human-readable log messages and state deltas for an executed action."""
    d_type = action.decision_type
    primary = action.primary_id
    secondary = action.secondary_id
    flags = action.flags
    p = state_before.get("decision_context", {}).get("decision_player", "NONE")
    logs = []

    # 1. Primary Action Narrative
    if d_type == ts_engine.DecisionType.SELECT_CARD:
        resolving_card = state_before.get("decision_context", {}).get("resolving_card", 0)
        card_name = ts_engine.CardData.get_card_name(primary) if 1 <= primary <= 110 else f"#{primary}"
        phase_name = state_before.get("current_phase_name", "")
        if resolving_card == 250:
            if action.is_confirm_done() or primary == 0:
                logs.append(f"{p} passes Space Walk (Box 6) turn-end discard opportunity.")
            else:
                logs.append(f"{p} discards {card_name} (#{primary}) via Space Walk (Box 6 privilege).")
        elif phase_name == "HEADLINE":
            us_h_before = state_before.get("headline_us_card", 0)
            ussr_h_before = state_before.get("headline_ussr_card", 0)
            is_second = (p == "US" and ussr_h_before > 0) or (p == "USSR" and us_h_before > 0)

            logs.append(f"{p} commits Headline Card: {card_name} (#{primary})")
            if is_second:
                us_card = primary if p == "US" else us_h_before
                ussr_card = primary if p == "USSR" else ussr_h_before
                us_name = ts_engine.CardData.get_card_name(us_card)
                ussr_name = ts_engine.CardData.get_card_name(ussr_card)
                us_ops = ts_engine.CardData.get_card_info(us_card).get("ops", 0)
                ussr_ops = ts_engine.CardData.get_card_info(ussr_card).get("ops", 0)

                first_p = "US" if us_ops >= ussr_ops else "USSR"
                second_p = "USSR" if first_p == "US" else "US"
                first_name = us_name if first_p == "US" else ussr_name
                second_name = ussr_name if first_p == "US" else us_name
                first_ops = max(us_ops, ussr_ops)
                second_ops = min(us_ops, ussr_ops)

                logs.append(f"  ★ Headlines Simultaneously Revealed: US plays '{us_name}' (#{us_card}, {us_ops} Ops) vs USSR plays '{ussr_name}' (#{ussr_card}, {ussr_ops} Ops)")
                logs.append(f"  ★ Resolution Order: 1st {first_p} '{first_name}' ({first_ops} Ops) -> 2nd {second_p} '{second_name}' ({second_ops} Ops)")
        else:
            logs.append(f"{p} plays Card: {card_name} (#{primary})")

    elif d_type == ts_engine.DecisionType.POINT_NODE:
        if action.is_confirm_done() or primary == 255 or (primary == 0 and (flags & 128)):
            logs.append(f"{p} confirms / finishes point node selections.")
        else:
            c_name = ts_engine.MapData.get_country_name(primary) if primary < 84 else f"Node #{primary}"
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
        vp_before = int(state_before.get("victory_points", 0))
        vp_after = int(state_after.get("victory_points", 0))
        vp_delta = vp_after - vp_before
        delta_str = f"+{vp_delta} VP" if vp_delta > 0 else f"{vp_delta} VP"
        vp_str_before = f"+{vp_before} (US)" if vp_before > 0 else (f"{vp_before} (USSR)" if vp_before < 0 else "0 (Tie)")
        vp_str_after = f"+{vp_after} (US)" if vp_after > 0 else (f"{vp_after} (USSR)" if vp_after < 0 else "0 (Tie)")
        logs.append(f"  • Victory Points: {vp_str_before} -> {vp_str_after} ({delta_str})")
    if state_before.get("us_mil_ops") != state_after.get("us_mil_ops"):
        logs.append(f"  • US MilOps: {state_before.get('us_mil_ops')} -> {state_after.get('us_mil_ops')}")
    if state_before.get("ussr_mil_ops") != state_after.get("ussr_mil_ops"):
        logs.append(f"  • USSR MilOps: {state_before.get('ussr_mil_ops')} -> {state_after.get('ussr_mil_ops')}")

    # 3.5 Effect Flag deltas
    old_flags = set(state_before.get("flags", []))
    new_flags = set(state_after.get("flags", []))
    for f in sorted(new_flags - old_flags):
        logs.append(f"  • Effect Activated: {f}")
    for f in sorted(old_flags - new_flags):
        logs.append(f"  • Effect Cancelled: {f}")

    # 4. Card movement deltas
    old_locs = state_before.get("card_locations", {})
    new_locs = state_after.get("card_locations", {})
    for cid_str, new_loc in new_locs.items():
        old_loc = old_locs.get(cid_str)
        if old_loc and old_loc != new_loc:
            cid = int(cid_str)
            card_name = ts_engine.CardData.get_card_name(cid)
            logs.append(f"  • Card #{cid} ({card_name}) moved: {old_loc} -> {new_loc}")

    # 5. Structured Die Roll Event Logging (Zero ghost / stale rolls)
    die_roll = state_after.get("die_roll", {})
    roll_type = die_roll.get("type", "NONE")

    if roll_type == "COUP":
        c_name = die_roll.get("country_name") or (ts_engine.MapData.get_country_name(die_roll.get("country_id", 0)) if die_roll.get("country_id", 255) < 84 else "")
        c_stab = state_before.get("countries", {}).get(c_name, {}).get("stability", 1)
        r_player = die_roll.get("roller") or p
        roll1 = die_roll.get("roll1", 0)
        mod1 = die_roll.get("mod1", 0)
        tot1 = die_roll.get("total1", roll1 + mod1)
        def_target = c_stab * 2
        success = die_roll.get("success", False)
        net = die_roll.get("net_delta", 0)
        card_name = die_roll.get("card_name", "")
        card_prefix = f" ({card_name})" if card_name else ""
        status_str = f"Net +{net} Influence (Coup Succeeded)" if success else "Coup Failed (Roll + Ops <= 2x Stability)"
        logs.append(f"  🎲 Coup in {c_name}{card_prefix}: {r_player} rolls {roll1} (+{mod1} Ops = {tot1}) vs {def_target} Defense (2x Stability {c_stab}) -> {status_str}")

    elif roll_type == "REALIGNMENT":
        c_name = die_roll.get("country_name") or (ts_engine.MapData.get_country_name(die_roll.get("country_id", 0)) if die_roll.get("country_id", 255) < 84 else "")
        roll_us = die_roll.get("roll1", 0)
        mod_us = die_roll.get("mod1", 0)
        tot_us = die_roll.get("total1", roll_us + mod_us)
        roll_ussr = die_roll.get("roll2", 0)
        mod_ussr = die_roll.get("mod2", 0)
        tot_ussr = die_roll.get("total2", roll_ussr + mod_ussr)
        net = die_roll.get("net_delta", 0)
        if tot_us > tot_ussr:
            res_str = f"US wins by +{tot_us - tot_ussr} (Removes {net} USSR Influence)"
        elif tot_ussr > tot_us:
            res_str = f"USSR wins by +{tot_ussr - tot_us} (Removes {net} US Influence)"
        else:
            res_str = "Tie (No Influence Removed)"
        logs.append(f"  🎲 Realignment in {c_name}: US rolls {roll_us} ({'+' if mod_us>=0 else ''}{mod_us} mod = {tot_us}), USSR rolls {roll_ussr} ({'+' if mod_ussr>=0 else ''}{mod_ussr} mod = {tot_ussr}) -> {res_str}")

    elif roll_type == "SPACE_RACE":
        target_step = die_roll.get("country_id", 0)
        roll1 = die_roll.get("roll1", 0)
        max_roll = die_roll.get("mod1", 0)
        success = die_roll.get("success", False)
        r_player = die_roll.get("roller") or p
        status = f"SUCCESS (Advanced to Box #{target_step})" if success else f"FAILED (Roll {roll1} > {max_roll} threshold)"
        logs.append(f"  🎲 Space Race Attempt (Target Box #{target_step}): {r_player} rolls {roll1} -> {status}")

    elif roll_type == "WAR_EVENT":
        card_name = die_roll.get("card_name") or f"Card #{die_roll.get('card_id')}"
        c_name = die_roll.get("country_name") or (ts_engine.MapData.get_country_name(die_roll.get("country_id", 0)) if die_roll.get("country_id", 255) < 84 else "")
        roll1 = die_roll.get("roll1", 0)
        mod1 = die_roll.get("mod1", 0)
        tot1 = die_roll.get("total1", roll1 + mod1)
        threshold = die_roll.get("mod2", 0)
        success = die_roll.get("success", False)
        r_player = die_roll.get("roller") or p
        target_str = f" targeting {c_name}" if c_name else ""
        status = f"Success ({tot1} >= {threshold} target: Target Captured & VP Awarded)" if success else f"Roll Failed ({tot1} < {threshold} target)"
        logs.append(f"  🎲 {card_name} Roll{target_str}: {r_player} rolls {roll1} ({'+' if mod1>=0 else ''}{mod1} mod = {tot1}) -> {status}")

    elif roll_type == "OLYMPIC_GAMES":
        sponsor = die_roll.get("roller") or p
        sp_roll = die_roll.get("roll1", 0)
        sp_tot = die_roll.get("total1", sp_roll + 2)
        opp_roll = die_roll.get("roll2", 0)
        opp_tot = die_roll.get("total2", opp_roll)
        opp_side = "US" if sponsor == "USSR" else "USSR"
        if sp_tot > opp_tot:
            res_str = f"Sponsor {sponsor} wins by +{sp_tot - opp_tot} (+2 VP)"
        elif opp_tot > sp_tot:
            res_str = f"Opponent {opp_side} wins"
        else:
            res_str = "Tie (No VP)"
        logs.append(f"  🎲 Olympic Games Competition: Sponsor ({sponsor}) rolls {sp_roll} (+2 = {sp_tot}), Opponent ({opp_side}) rolls {opp_roll} -> {res_str}")

    elif roll_type == "SUMMIT":
        us_roll = die_roll.get("roll1", 0)
        us_dom = die_roll.get("mod1", 0)
        us_tot = die_roll.get("total1", us_roll + us_dom)
        ussr_roll = die_roll.get("roll2", 0)
        ussr_dom = die_roll.get("mod2", 0)
        ussr_tot = die_roll.get("total2", ussr_roll + ussr_dom)
        diff = abs(us_tot - ussr_tot)
        if us_tot > ussr_tot:
            res_str = f"US wins Summit by +{diff}"
        elif ussr_tot > us_tot:
            res_str = f"USSR wins Summit by +{diff}"
        else:
            res_str = "Tied Summit (No Effect)"
        logs.append(f"  🎲 Summit Rolls: US rolls {us_roll} (+{us_dom} Dom = {us_tot}), USSR rolls {ussr_roll} (+{ussr_dom} Dom = {ussr_tot}) -> {res_str}")

    elif roll_type == "TRAP_ESCAPE":
        r_player = die_roll.get("roller") or p
        roll1 = die_roll.get("roll1", 0)
        success = die_roll.get("success", False)
        status = "Success! Discard canceled trap." if success else "Failed (Roll > 4, remains trapped)"
        logs.append(f"  🎲 Trap Escape Roll: {r_player} rolls {roll1} (Needed 1–4) -> {status}")

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

    def get_state_dict(self, for_role: Optional[str] = None) -> GameStateDict:
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
        obs = self._observation_for(for_role)
        if obs is not None:
            d["observation_b64"] = obs
        return cast(GameStateDict, d)

    def _observation_for(self, role: Optional[str]) -> Optional[str]:
        """The engine's own observation for `role`, base64 float32, when it is their move.

        A network client that rebuilds the observation itself will drift from
        Observation::extract, and a policy fed a slightly different encoding than it was
        trained on plays close to randomly. So the authoritative extractor -- the same one
        training uses -- runs here, where the real GameState lives.

        It is emitted only to the player whose decision it is, because the observation
        contains that player's own hand: broadcasting it to the opponent would leak hidden
        information that the encoding is specifically built to withhold.
        """
        if role not in ("US", "USSR"):
            return None
        ctx = self.state.ctx()
        decider = ctx.decision_player if ctx.decision_player != ts_engine.Player.NONE else self.state.phasing_player
        want = ts_engine.Player.US if role == "US" else ts_engine.Player.USSR
        if decider != want:
            return None
        try:
            arr = np.asarray(ts_engine.extract_observation(self.state, want), dtype=np.float32)
            return base64.b64encode(arr.tobytes()).decode("ascii")
        except Exception as exc:  # never let a diagnostic aid break the game loop
            logger.debug(f"[{self.game_id}] observation extraction failed: {exc}")
            return None

    def _role_of(self, websocket: WebSocket) -> Optional[str]:
        for role, sockets in self.role_connections.items():
            if websocket in sockets:
                return role
        return None

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
            "state": self.get_state_dict(for_role=role)
        })

    def disconnect(self, websocket: WebSocket):
        self.connections.discard(websocket)
        for r_set in self.role_connections.values():
            r_set.discard(websocket)
        logger.info(f"[{self.game_id}] Client disconnected. Remaining connections: {len(self.connections)}")

    async def broadcast_state(self):
        if not self.connections:
            return
        # Built per role, since the observation is perspective-specific and must not be
        # sent to the opponent.
        by_role: Dict[Optional[str], Dict[str, Any]] = {}
        dead_sockets = set()
        for ws in self.connections:
            role = self._role_of(ws)
            if role not in by_role:
                by_role[role] = {
                    "type": "STATE_UPDATE",
                    "state": self.get_state_dict(for_role=role),
                }
            try:
                await ws.send_json(by_role[role])
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

        # If the action produced a ROLL_DIE chance node (Coup, Realignment, Space Race, War Events), resolve it!
        while (not ts_engine.Engine.is_terminal(self.state)
               and self.state.ctx().decision_player == ts_engine.Player.NONE
               and self.state.ctx().decision_type == ts_engine.DecisionType.ROLL_DIE):
            ts_engine.Engine.step(self.state, ts_engine.MicroAction(ts_engine.DecisionType.ROLL_DIE, secondary, 0, 0))

        self.step_index += 1
        state_after = cast(GameStateDict, self.state.to_dict())
        delta_lines = describe_action_and_deltas(state_before, state_after, action)

        main_desc = delta_lines[0] if delta_lines else f"Action type={int(d_type)}"
        for line in delta_lines:
            logger.info(f"[{self.game_id}] Step {self.step_index} [Turn {self.state.turn} AR {self.state.action_round}]: {line}")

        step_turn = state_before.get("turn", self.state.turn)
        step_phase = str(state_before.get("current_phase_name", "ACTION"))
        step_ar = 0 if step_phase in ("HEADLINE", "SETUP") else state_before.get("action_round", self.state.action_round)

        vp_before_step = int(state_before.get("victory_points", 0))
        vp_after_step = int(state_after.get("victory_points", 0))
        step_vp_delta = vp_after_step - vp_before_step

        log_entry = {
            "step_index": self.step_index,
            "turn": step_turn,
            "ar": step_ar,
            "phase": step_phase,
            "player": state_before.get("decision_context", {}).get("decision_player", "NONE"),
            "text": main_desc,
            "details": delta_lines[1:] if len(delta_lines) > 1 else [],
            "vp_delta": step_vp_delta
        }
        self.action_logs.append(log_entry)

        # Log to replay
        self.replay_logger.log_step(
            step_index=self.step_index,
            turn=step_turn,
            ar=step_ar,
            phase=step_phase,
            player=log_entry["player"],
            action={"decision_type": int(d_type), "primary_id": primary, "secondary_id": secondary, "flags": flags},
            description=main_desc,
            state_snapshot=state_after
        )

        # Check terminal state
        if ts_engine.Engine.is_terminal(self.state):
            util = ts_engine.Engine.get_terminal_utility(self.state)
            winner = "US" if util > 0 else ("USSR" if util < 0 else "DRAW")
            reason = classify_game_ending_reason(self.state)
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
