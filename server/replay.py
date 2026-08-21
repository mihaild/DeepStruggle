import json
import os
import time
from typing import List, Dict, Any, Optional

REPLAYS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "replays")

class ReplayLogger:
    def __init__(self, game_id: str, seed: int, us_player: str = "US Player", ussr_player: str = "USSR Player"):
        self.game_id = game_id
        self.seed = seed
        self.us_player = us_player
        self.ussr_player = ussr_player
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.steps: List[Dict[str, Any]] = []
        self.result: Optional[Dict[str, Any]] = None

    def log_step(self, step_index: int, turn: int, ar: int, phase: str, player: str,
                 action: Dict[str, Any], description: str, state_snapshot: Dict[str, Any]):
        self.steps.append({
            "step_index": step_index,
            "turn": turn,
            "ar": ar,
            "phase": phase,
            "player": player,
            "action": action,
            "description": description,
            "state_snapshot": state_snapshot
        })

    def set_result(self, winner: str, margin: int, end_turn: int, reason: str):
        self.result = {
            "winner": winner,
            "margin": margin,
            "end_turn": end_turn,
            "reason": reason
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "metadata": {
                "game_id": self.game_id,
                "created_at": self.created_at,
                "seed": self.seed,
                "players": {
                    "US": self.us_player,
                    "USSR": self.ussr_player
                },
                "total_steps": len(self.steps),
                "result": self.result
            },
            "initial_state": {
                "seed": self.seed
            },
            "steps": self.steps
        }

    def save(self, filepath: Optional[str] = None) -> str:
        os.makedirs(REPLAYS_DIR, exist_ok=True)
        if not filepath:
            filepath = os.path.join(REPLAYS_DIR, f"{self.game_id}.tslog.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath

class ReplayManager:
    @staticmethod
    def list_replays() -> List[Dict[str, Any]]:
        os.makedirs(REPLAYS_DIR, exist_ok=True)
        replays = []
        for fname in os.listdir(REPLAYS_DIR):
            if fname.endswith(".tslog.json") or fname.endswith(".json"):
                full_path = os.path.join(REPLAYS_DIR, fname)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        meta = data.get("metadata", {})
                        replays.append({
                            "filename": fname,
                            "game_id": meta.get("game_id", fname),
                            "created_at": meta.get("created_at", ""),
                            "players": meta.get("players", {}),
                            "total_steps": meta.get("total_steps", len(data.get("steps", []))),
                            "result": meta.get("result", None)
                        })
                except Exception:
                    continue
        replays.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return replays

    @staticmethod
    def load_replay(filename: str) -> Optional[Dict[str, Any]]:
        full_path = os.path.join(REPLAYS_DIR, filename)
        if not os.path.exists(full_path):
            # Try appending .tslog.json
            full_path = os.path.join(REPLAYS_DIR, f"{filename}.tslog.json")
        if not os.path.exists(full_path):
            return None
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
