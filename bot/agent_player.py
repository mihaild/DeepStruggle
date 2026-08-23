#!/usr/bin/env python3
"""
Twilight Struggle Agent Player & CLI Tool
Allows AI agents and users to inspect state, select actions by name or index, and play games on the simulation engine.
Works both with the live WebSocket/REST server and in standalone local engine mode.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Any, Union

import os
try:
    import ts_engine
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _build = os.path.join(_root, "build")
    if os.path.exists(_build) and _build not in sys.path:
        sys.path.insert(0, _build)
    import ts_engine

class LocalSessionManager:
    """Manages an in-memory or file-persisted game session using ts_engine directly."""
    SAVE_FILE = ".ts_local_session.json"

    def __init__(self, seed: int = 42):
        self.state = ts_engine.GameState()
        self.actions: List[dict] = []
        self.action_logs: List[dict] = []
        self.seed = seed
        if os.path.exists(self.SAVE_FILE):
            self.load()
        else:
            self.reset(seed)

    def reset(self, seed: int = 42):
        self.seed = seed
        self.state = ts_engine.GameState()
        ts_engine.Engine.init_game(self.state, seed)
        self.actions = []
        self.action_logs = []
        self.save()

    def get_state_dict(self) -> dict:
        d = self.state.to_dict()
        d["seed"] = self.seed
        d["can_undo"] = len(self.actions) > 0
        d["action_logs"] = self.action_logs[-60:]
        return d

    def step(self, action_dict: dict) -> dict:
        d_type = ts_engine.DecisionType(action_dict["decision_type"])
        primary = action_dict.get("primary_id", 0)
        secondary = action_dict.get("secondary_id", 0)
        flags = action_dict.get("flags", 0)
        action = ts_engine.MicroAction(d_type, primary, secondary, flags)

        self.actions.append(action_dict)
        ts_engine.Engine.step(self.state, action)

        self.action_logs.append({
            "step_index": len(self.actions),
            "turn": self.state.turn,
            "ar": self.state.action_round,
            "phase": str(self.state.current_phase),
            "player": str(self.state.phasing_player),
            "text": f"Action: type={int(d_type)} primary={primary} secondary={secondary} flags={flags}"
        })

        self.save()
        return self.get_state_dict()

    def undo(self) -> dict:
        if not self.actions:
            return self.get_state_dict()
        self.actions.pop()
        if self.action_logs:
            self.action_logs.pop()

        self.replay()
        self.save()
        return self.get_state_dict()

    def replay(self):
        temp_state = ts_engine.GameState()
        ts_engine.Engine.init_game(temp_state, self.seed)
        for act in self.actions:
            d_type = ts_engine.DecisionType(act["decision_type"])
            primary = act.get("primary_id", 0)
            secondary = act.get("secondary_id", 0)
            flags = act.get("flags", 0)
            m_act = ts_engine.MicroAction(d_type, primary, secondary, flags)
            ts_engine.Engine.step(temp_state, m_act)
        self.state = temp_state

    def save(self):
        try:
            with open(self.SAVE_FILE, "w") as f:
                json.dump({
                    "seed": self.seed,
                    "actions": self.actions,
                    "action_logs": self.action_logs
                }, f)
        except Exception:
            pass

    def load(self):
        try:
            with open(self.SAVE_FILE, "r") as f:
                data = json.load(f)
                self.seed = data.get("seed", 42)
                self.actions = data.get("actions", [])
                self.action_logs = data.get("action_logs", [])
                self.replay()
        except Exception:
            self.reset(42)


class AgentGameClient:
    """Client for interacting with the TS game engine via REST or local engine."""

    def __init__(self, server_url: Optional[str] = "http://127.0.0.1:8000", game_id: str = "game-1", role: str = "OBSERVER", use_local: bool = False):
        self.server_url = server_url.rstrip("/") if server_url else None
        self.game_id = game_id
        self.role = role.upper()
        self.use_local = use_local
        self.local_manager: Optional[LocalSessionManager] = None

        if self.use_local or not self.server_url:
            self.local_manager = LocalSessionManager()
        else:
            # Test connection; if unreachable, fallback to local
            try:
                import httpx
                resp = httpx.get(f"{self.server_url}/api/metadata/map", timeout=0.8)
                if resp.status_code != 200:
                    self.use_local = True
                    self.local_manager = LocalSessionManager()
            except Exception:
                self.use_local = True
                self.local_manager = LocalSessionManager()

    def get_state(self) -> Dict[str, Any]:
        """Fetches the full state dictionary."""
        if self.use_local and self.local_manager:
            return self.local_manager.get_state_dict()

        import httpx
        url = f"{self.server_url}/api/games/{self.game_id}"
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to fetch game state: {resp.status_code} {resp.text}")
            return resp.json()

    def new_game(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Creates or resets a game session."""
        s = seed if seed is not None else 42
        if self.use_local and self.local_manager:
            self.local_manager.reset(s)
            return self.local_manager.get_state_dict()

        import httpx
        url = f"{self.server_url}/api/games/new"
        payload = {"game_id": self.game_id, "seed": seed}
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to create game: {resp.status_code} {resp.text}")
            return resp.json()["state"]

    def cancel_action(self) -> Dict[str, Any]:
        """Cancels/undoes the last action."""
        if self.use_local and self.local_manager:
            return self.local_manager.undo()

        import httpx
        url = f"{self.server_url}/api/games/{self.game_id}/cancel_action"
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to cancel action: {resp.status_code} {resp.text}")
            return resp.json()["state"]

    def play_action(self, decision_type: int, primary_id: int, secondary_id: int = 0, flags: int = 0) -> Dict[str, Any]:
        """Sends a MicroAction to the game session."""
        action_dict = {
            "decision_type": decision_type,
            "primary_id": primary_id,
            "secondary_id": secondary_id,
            "flags": flags
        }
        if self.use_local and self.local_manager:
            return self.local_manager.step(action_dict)

        import websockets
        import asyncio

        async def _send_ws():
            ws_url = f"{self.server_url.replace('http', 'ws')}/ws/game/{self.game_id}?role={self.role}"
            async with websockets.connect(ws_url) as ws:
                # Receive initial state
                await ws.recv()
                # Send action
                payload = {
                    "type": "PLAY_ACTION",
                    "action": action_dict
                }
                await ws.send(json.dumps(payload))
                # Receive updated state
                resp_msg = json.loads(await ws.recv())
                return resp_msg.get("state", {})

        return asyncio.run(_send_ws())

    def get_legal_choices(self, state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Parses the current legal choices into human-friendly options."""
        if state is None:
            state = self.get_state()

        legal = state.get("legal_actions", {})
        d_type = legal.get("decision_type", 0)
        valid_ids = legal.get("valid_ids", [])
        allow_early_stop = legal.get("allow_early_stop", False)

        choices = []

        if d_type == 1: # SELECT_CARD
            for cid in valid_ids:
                card_name = ts_engine.CardData.get_card_name(cid) if 1 <= cid <= 110 else f"Card #{cid}"
                info = ts_engine.CardData.get_card_info(cid) if 1 <= cid <= 110 else {}
                ops = info.get("ops", 0)
                side = info.get("side", "neutral")
                era = info.get("era_name", "")
                one_time = "★" if info.get("one_time") else ""
                desc = f"#{cid} {card_name} {one_time} ({ops} Ops, {side.upper()}, {era.upper()})"
                choices.append({
                    "index": len(choices) + 1,
                    "id": cid,
                    "name": card_name,
                    "label": desc,
                    "decision_type": d_type,
                    "primary_id": cid,
                    "secondary_id": 0,
                    "flags": 0
                })

        elif d_type == 2: # SELECT_PLAY_MODE
            modes = {0: "EVENT", 1: "OPERATIONS", 2: "SPACE RACE", 3: "PASS"}
            for m in valid_ids:
                name = modes.get(m, f"MODE_{m}")
                choices.append({
                    "index": len(choices) + 1,
                    "id": m,
                    "name": name,
                    "label": f"{name} (Play mode {m})",
                    "decision_type": d_type,
                    "primary_id": m,
                    "secondary_id": 0,
                    "flags": 0
                })

        elif d_type == 3: # CHOOSE_TIMING_BRANCH
            branches = {0: "OPS_FIRST", 1: "EVENT_FIRST"}
            branch_labels = {0: "Operations First (Opponent Event Second)", 1: "Opponent Event First (Operations Second)"}
            for b in valid_ids:
                name = branches.get(b, f"BRANCH_{b}")
                choices.append({
                    "index": len(choices) + 1,
                    "id": b,
                    "name": name,
                    "label": branch_labels.get(b, name),
                    "decision_type": d_type,
                    "primary_id": b,
                    "secondary_id": 0,
                    "flags": 0
                })

        elif d_type == 4: # SELECT_OP_MODE
            op_modes = {0: "INFLUENCE", 1: "COUP", 2: "REALIGN"}
            op_labels = {0: "Place Influence", 1: "Conduct Coup Attempt", 2: "Conduct Realignment"}
            for m in valid_ids:
                name = op_modes.get(m, f"OP_{m}")
                choices.append({
                    "index": len(choices) + 1,
                    "id": m,
                    "name": name,
                    "label": op_labels.get(m, name),
                    "decision_type": d_type,
                    "primary_id": m,
                    "secondary_id": 0,
                    "flags": 0
                })

        elif d_type == 5: # POINT_NODE
            countries = state.get("countries", {})
            for nid in valid_ids:
                c_name = ts_engine.MapData.get_country_name(nid) if 0 <= nid < 84 else f"Country #{nid}"
                c_data = countries.get(c_name, {})
                us_inf = c_data.get("us_influence", 0)
                ussr_inf = c_data.get("ussr_influence", 0)
                stab = c_data.get("stability", 0)
                bg = "★BG" if c_data.get("battleground") else ""
                ctrl = c_data.get("controlled_by", "NONE")
                label = f"{c_name} (ID {nid}) [US:{us_inf} USSR:{ussr_inf} Stab:{stab} {bg} Ctrl:{ctrl}]"
                choices.append({
                    "index": len(choices) + 1,
                    "id": nid,
                    "name": c_name,
                    "label": label,
                    "decision_type": d_type,
                    "primary_id": nid,
                    "secondary_id": 0,
                    "flags": 0
                })

        elif d_type == 6: # CHOOSE_BRANCH
            labels = legal.get("valid_action_labels", {})
            for b in valid_ids:
                lbl = labels.get(str(b), f"Option {b + 1}")
                choices.append({
                    "index": len(choices) + 1,
                    "id": b,
                    "name": f"OPTION_{b}",
                    "label": lbl,
                    "decision_type": d_type,
                    "primary_id": b,
                    "secondary_id": 0,
                    "flags": 0
                })

        # Add Done / Pass option if allowed
        if allow_early_stop:
            choices.append({
                "index": len(choices) + 1,
                "id": "DONE",
                "name": "DONE",
                "label": "✓ Done / Confirm / Pass Remaining",
                "decision_type": d_type,
                "primary_id": 0,
                "secondary_id": 0,
                "flags": 0x80
            })

        return choices

    def resolve_choice(self, query: Union[str, int], roll: int = 0, flags: int = 0, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Matches a user/agent query string or index against current legal actions and executes it."""
        if state is None:
            state = self.get_state()

        choices = self.get_legal_choices(state)
        if not choices:
            raise ValueError("No legal actions available in current state.")

        query_str = str(query).strip().lower()

        # 1. Match by numeric index (e.g. 1, 2, 3)
        if query_str.isdigit():
            idx = int(query_str)
            for c in choices:
                if c["index"] == idx:
                    sec = roll if roll > 0 else c["secondary_id"]
                    flg = flags if flags > 0 else c["flags"]
                    return self.play_action(c["decision_type"], c["primary_id"], sec, flg)

        # 2. Match "done" / "confirm" / "pass"
        if query_str in ("done", "confirm", "pass", "stop"):
            for c in choices:
                if c["name"] == "DONE" or c["flags"] == 0x80:
                    return self.play_action(c["decision_type"], 0, roll, 0x80)

        # 3. Match by name or label substring
        for c in choices:
            if c["name"].lower() == query_str:
                sec = roll if roll > 0 else c["secondary_id"]
                flg = flags if flags > 0 else c["flags"]
                return self.play_action(c["decision_type"], c["primary_id"], sec, flg)

        for c in choices:
            if query_str in c["name"].lower() or query_str in c["label"].lower():
                sec = roll if roll > 0 else c["secondary_id"]
                flg = flags if flags > 0 else c["flags"]
                return self.play_action(c["decision_type"], c["primary_id"], sec, flg)

        # 4. If nothing matched, show valid choices
        choice_lines = [f"  [{c['index']}] {c['label']}" for c in choices]
        raise ValueError(f"Could not match '{query}' to legal actions:\n" + "\n".join(choice_lines))


def format_state_summary(state: Dict[str, Any], client: Optional[AgentGameClient] = None) -> str:
    """Formats a concise, structured markdown report for agents."""
    turn = state.get("turn", 1)
    phase = state.get("current_phase_name", "UNKNOWN")
    ar = state.get("action_round", 0)
    phasing = state.get("phasing_player", "NONE")
    vp = state.get("victory_points", 0)
    vp_str = f"+{vp} (US)" if vp > 0 else (f"{vp} (USSR)" if vp < 0 else "0 (Tie)")
    defcon = state.get("defcon", 5)

    mil_ops = state.get("mil_ops", {})
    us_mil = mil_ops.get("US", 0)
    ussr_mil = mil_ops.get("USSR", 0)

    space = state.get("space", {})
    us_sp = space.get("US", 0)
    ussr_sp = space.get("USSR", 0)

    ctx = state.get("decision_context", {})
    d_type_name = ctx.get("decision_type_name", "NONE")
    d_player = ctx.get("decision_player", "NONE")
    pending_card = ctx.get("pending_op_card_name", "")
    pending_ops = ctx.get("pending_ops_value", 0)
    remaining = ctx.get("remaining_steps", 0)
    resolving = ctx.get("resolving_card_name", "")

    lines = []
    lines.append("=" * 65)
    lines.append(f"★ TWILIGHT STRUGGLE STATE | Turn {turn} | Phase: {phase} | AR: {ar}")
    lines.append(f"DEFCON: {defcon} | VP: {vp_str} | Phasing: {phasing}")
    lines.append(f"MilOps: [US: {us_mil}/5 | USSR: {ussr_mil}/5] | Space: [US: {us_sp} | USSR: {ussr_sp}]")
    lines.append("-" * 65)

    # Active Decision Context
    ctx_info = f"ACTIVE DECISION: {d_type_name} for {d_player}"
    if pending_card:
        ctx_info += f" | Pending Card: {pending_card} ({pending_ops} Ops, {remaining} remaining)"
    if resolving:
        ctx_info += f" | Resolving Event: {resolving}"
    lines.append(ctx_info)

    # Hands
    hands = state.get("hands", {})
    us_cards = hands.get("US_cards", [])
    ussr_cards = hands.get("USSR_cards", [])

    us_hand_strs = [f"#{c['id']} {c['name']} ({c['ops']} Ops)" for c in us_cards]
    ussr_hand_strs = [f"#{c['id']} {c['name']} ({c['ops']} Ops)" for c in ussr_cards]

    lines.append(f"US Hand ({len(us_cards)}): " + (", ".join(us_hand_strs) if us_hand_strs else "Empty"))
    lines.append(f"USSR Hand ({len(ussr_cards)}): " + (", ".join(ussr_hand_strs) if ussr_hand_strs else "Empty"))

    # Deck summary
    lines.append(f"Draw Deck: {state.get('draw_deck_count', 0)} | Discard: {len(state.get('discard_pile', []))} | Removed: {len(state.get('removed_pile', []))}")
    lines.append("-" * 65)

    # Legal Actions
    if client is None:
        client = AgentGameClient(use_local=True)
    choices = client.get_legal_choices(state)

    lines.append(f"LEGAL CHOICES ({len(choices)} options):")
    for c in choices:
        lines.append(f"  [{c['index']}] {c['label']}")

    # Recent Action Stream (last 3)
    action_logs = state.get("action_logs", [])
    if action_logs:
        lines.append("-" * 65)
        lines.append("RECENT LOGS:")
        for log in action_logs[-3:]:
            lines.append(f"  • [T{log.get('turn')} AR{log.get('ar')}] {log.get('player')}: {log.get('text')}")
            for d in log.get("details", []):
                lines.append(f"      {d}")

    lines.append("=" * 65)
    return "\n".join(lines)


def format_board_overview(state: Dict[str, Any]) -> str:
    """Formats a breakdown of regional control and country influence."""
    countries = state.get("countries", {})
    region_names = {
        0: "EUROPE (Western & Eastern)",
        1: "ASIA",
        2: "MIDDLE EAST",
        3: "AFRICA",
        4: "CENTRAL AMERICA",
        5: "SOUTH AMERICA"
    }
    regions: Dict[str, List[str]] = {name: [] for name in region_names.values()}

    for c_name, c in countries.items():
        us = c.get("us_influence", 0)
        ussr = c.get("ussr_influence", 0)
        stab = c.get("stability", 0)
        bg = "★" if c.get("battleground") else " "
        ctrl = c.get("controlled_by", "NONE")
        r_id = c.get("region", 0)
        r_title = region_names.get(r_id, f"REGION_{r_id}")

        entry = f"{bg} {c_name:<20} [US:{us} USSR:{ussr}] (Stab:{stab}, Ctrl:{ctrl})"
        if r_title in regions:
            regions[r_title].append(entry)

    lines = ["=== GLOBAL BOARD CONTROL OVERVIEW ==="]
    for reg_name, c_list in regions.items():
        if c_list:
            lines.append(f"\n--- {reg_name} ({len(c_list)} countries) ---")
            lines.extend(c_list)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Twilight Struggle Agent Player Tool")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="Server base URL (or leave default)")
    parser.add_argument("--local", action="store_true", help="Force local direct in-memory engine mode")
    parser.add_argument("--game-id", default="game-1", help="Game session ID")
    parser.add_argument("--role", default="OBSERVER", help="Player role: US, USSR, or OBSERVER")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: status
    subparsers.add_parser("status", help="Print structured state summary and legal actions")

    # Command: board
    subparsers.add_parser("board", help="Print global country control & influence breakdown")

    # Command: act / play
    act_parser = subparsers.add_parser("act", aliases=["play"], help="Execute a move by choice index, country name, card name, or mode")
    act_parser.add_argument("choice", help="Index (e.g. 1, 2) or Name (e.g. 'West Germany', 'Duck and Cover', 'OPS', 'done')")
    act_parser.add_argument("--roll", type=int, default=0, help="Manual die roll (1..6) or 0 for auto PRNG")
    act_parser.add_argument("--flags", type=int, default=0, help="Flags or opponent die roll")

    # Command: undo / cancel
    subparsers.add_parser("undo", aliases=["cancel"], help="Undo / cancel the last action")

    # Command: new-game
    new_parser = subparsers.add_parser("new-game", help="Start a new game session")
    new_parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Command: step-bot (auto move using bot)
    bot_parser = subparsers.add_parser("auto", help="Auto-play one or more steps using baseline bot")
    bot_parser.add_argument("--steps", type=int, default=1, help="Number of steps to play")
    bot_parser.add_argument("--strategy", default="heuristic", choices=["heuristic", "random"], help="Bot strategy")

    args = parser.parse_args()

    client = AgentGameClient(server_url=args.server, game_id=args.game_id, role=args.role, use_local=args.local)

    if not args.command or args.command == "status":
        state = client.get_state()
        print(format_state_summary(state, client))

    elif args.command == "board":
        state = client.get_state()
        print(format_board_overview(state))

    elif args.command in ("act", "play"):
        state_after = client.resolve_choice(args.choice, roll=args.roll, flags=args.flags)
        print(format_state_summary(state_after, client))

    elif args.command in ("undo", "cancel"):
        state_after = client.cancel_action()
        print("↺ Action cancelled / rolled back.")
        print(format_state_summary(state_after, client))

    elif args.command == "new-game":
        state = client.new_game(seed=args.seed)
        print(f"★ New game started (Seed: {args.seed or 42}).")
        print(format_state_summary(state, client))

    elif args.command == "auto":
        from bot.bot_client import HeuristicBot, RandomBot
        bot_us = HeuristicBot("US") if args.strategy == "heuristic" else RandomBot("US")
        bot_ussr = HeuristicBot("USSR") if args.strategy == "heuristic" else RandomBot("USSR")

        for s in range(args.steps):
            state = client.get_state()
            if state.get("is_terminal"):
                print(f"Game over! Terminal utility: {state.get('terminal_utility')}")
                break

            legal = state.get("legal_actions", {})
            p = legal.get("decision_player", "")
            bot = bot_ussr if p == "USSR" else bot_us
            action = bot.select_action(state, legal)
            if not action:
                print("Bot has no available action.")
                break

            state = client.play_action(
                action["decision_type"],
                action["primary_id"],
                action.get("secondary_id", 0),
                action.get("flags", 0)
            )
            print(f"Step {s + 1}/{args.steps} executed for {p}: type={action['decision_type']} primary={action['primary_id']}")

        print(format_state_summary(state, client))

if __name__ == "__main__":
    main()
