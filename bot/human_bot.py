"""HumanBot: Interactive terminal player prompting user for decision selections via CLI."""

from typing import Dict, Any, Optional, List
import os
import sys

try:
    import ts_engine
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _build = os.path.join(_root, "build")
    if os.path.exists(_build) and _build not in sys.path:
        sys.path.insert(0, _build)
    import ts_engine

from bot.base_bot import BaseBot


class HumanBot(BaseBot):
    """Interactive human player selecting microactions via terminal stdin."""

    def select_action(self, state: Dict[str, Any], legal_actions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        choices = self.format_choices(state, legal_actions)
        if not choices:
            print("\n[HumanBot] No legal choices available (passing).")
            return None

        d_type_name = state.get("decision_context", {}).get("decision_type_name", "ACTION")
        print(f"\n★ YOUR TURN ({self.role}) — {d_type_name} ({len(choices)} options):")
        for c in choices:
            print(f"  [{c['index']}] {c['label']}")

        while True:
            try:
                raw = input(f"\nSelect action [1-{len(choices)}] (or 'q' to forfeit): ").strip()
                if raw.lower() == "q":
                    print("\n[HumanBot] Forfeiting game.")
                    return None
                idx = int(raw)
                if 1 <= idx <= len(choices):
                    chosen = choices[idx - 1]
                    return {
                        "decision_type": chosen["decision_type"],
                        "primary_id": chosen["primary_id"],
                        "secondary_id": chosen["secondary_id"],
                        "flags": chosen["flags"],
                    }
                print(f"Invalid index {idx}. Must be between 1 and {len(choices)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    @staticmethod
    def format_choices(state: Dict[str, Any], legal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Formats legal_actions into human-readable numbered choices."""
        d_type = legal.get("decision_type", 0)
        valid_ids = legal.get("valid_ids", [])
        allow_early_stop = legal.get("allow_early_stop", False)

        choices: List[Dict[str, Any]] = []

        if d_type == 1:  # SELECT_CARD
            for cid in valid_ids:
                c_name = ts_engine.CardData.get_card_name(cid) if 1 <= cid <= 110 else f"Card #{cid}"
                info = ts_engine.CardData.get_card_info(cid) if 1 <= cid <= 110 else {}
                ops = info.get("ops", 0)
                side = info.get("side", "neutral")
                desc = f"#{cid} {c_name} ({ops} Ops, {str(side).upper()})"
                choices.append({
                    "index": len(choices) + 1,
                    "label": desc,
                    "decision_type": d_type,
                    "primary_id": cid,
                    "secondary_id": 0,
                    "flags": 0,
                })

        elif d_type == 2:  # SELECT_PLAY_MODE
            # P17 merged the Ops mode and the event/ops timing into this node. On an opponent
            # card EVENT is the event-first branch -- the event resolves and the engine then
            # asks how to spend the Ops -- and each OPERATIONS entry is the ops-first branch.
            modes = {
                0: "EVENT",
                1: "SPACE RACE",
                2: "OPERATIONS - Place Influence",
                3: "OPERATIONS - Coup",
                4: "OPERATIONS - Realignment",
            }
            for m in valid_ids:
                name = modes.get(m, f"MODE_{m}")
                choices.append({
                    "index": len(choices) + 1,
                    "label": f"Play as {name}",
                    "decision_type": d_type,
                    "primary_id": m,
                    "secondary_id": 0,
                    "flags": 0,
                })

        elif d_type == 4:  # SELECT_OP_MODE
            op_modes = {0: "Place Influence", 1: "Conduct Coup Attempt", 2: "Conduct Realignment"}
            for m in valid_ids:
                choices.append({
                    "index": len(choices) + 1,
                    "label": op_modes.get(m, f"Mode {m}"),
                    "decision_type": d_type,
                    "primary_id": m,
                    "secondary_id": 0,
                    "flags": 0,
                })

        elif d_type == 5:  # POINT_NODE
            countries = state.get("countries", {})
            for nid in valid_ids:
                if 0 <= nid < 84:
                    c_name = ts_engine.MapData.get_country_name(nid)
                    c_data = countries.get(c_name, {}) if isinstance(countries, dict) else {}
                    us_inf = c_data.get("us_influence", 0)
                    ussr_inf = c_data.get("ussr_influence", 0)
                    desc = f"{c_name} (ID {nid}) [US: {us_inf} | USSR: {ussr_inf}]"
                else:
                    desc = f"Node #{nid}"
                choices.append({
                    "index": len(choices) + 1,
                    "label": desc,
                    "decision_type": d_type,
                    "primary_id": nid,
                    "secondary_id": 0,
                    "flags": 0,
                })

        else:
            for cid in valid_ids:
                choices.append({
                    "index": len(choices) + 1,
                    "label": f"Action ID {cid}",
                    "decision_type": d_type,
                    "primary_id": cid,
                    "secondary_id": 0,
                    "flags": 0,
                })

        if allow_early_stop:
            choices.append({
                "index": len(choices) + 1,
                "label": "Confirm Done / Pass Remaining",
                "decision_type": d_type,
                "primary_id": 0,
                "secondary_id": 0,
                "flags": 0x80,
            })

        return choices
