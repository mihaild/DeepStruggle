import json
import os
import time
from typing import List, Dict, Any, Optional
from server.replay_types import (
    ReplayLogDict,
    ReplayStepDict,
    ReplaySummaryDict,
    ReplayMetadataDict,
    ReplayResultDict,
    GameStateDict,
    ReplayActionDict,
)

REPLAYS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "replays")


class ReplayLogger:
    def __init__(self, game_id: str, seed: int, us_player: str = "US Player", ussr_player: str = "USSR Player") -> None:
        self.game_id: str = game_id
        self.seed: int = seed
        self.us_player: str = us_player
        self.ussr_player: str = ussr_player
        self.created_at: str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.steps: List[ReplayStepDict] = []
        self.result: Optional[ReplayResultDict] = None

    def log_step(
        self,
        step_index: int,
        turn: int,
        ar: int,
        phase: str,
        player: str,
        action: ReplayActionDict,
        description: str,
        state_snapshot: GameStateDict,
    ) -> None:
        self.steps.append(
            {
                "step_index": step_index,
                "turn": turn,
                "ar": ar,
                "phase": phase,
                "player": player,
                "action": action,
                "description": description,
                "state_snapshot": state_snapshot,
            }
        )

    def set_result(self, winner: str, margin: int, end_turn: int, reason: str) -> None:
        self.result = {
            "winner": winner,
            "margin": margin,
            "end_turn": end_turn,
            "reason": reason,
        }

    def to_dict(self) -> ReplayLogDict:
        metadata: ReplayMetadataDict = {
            "game_id": self.game_id,
            "created_at": self.created_at,
            "seed": self.seed,
            "players": {
                "US": self.us_player,
                "USSR": self.ussr_player,
            },
            "total_steps": len(self.steps),
            "result": self.result,
        }
        return {
            "version": "1.0",
            "metadata": metadata,
            "initial_state": {
                "seed": self.seed,
            },
            "steps": self.steps,
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
    def list_replays() -> List[ReplaySummaryDict]:
        os.makedirs(REPLAYS_DIR, exist_ok=True)
        replays: List[ReplaySummaryDict] = []
        for fname in os.listdir(REPLAYS_DIR):
            if fname.endswith(".tslog.json") or fname.endswith(".json"):
                full_path = os.path.join(REPLAYS_DIR, fname)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        data: Dict[str, Any] = json.load(f)
                        meta = data.get("metadata")
                        if meta and isinstance(meta, dict):
                            total = meta.get("total_steps", len(data.get("steps", [])))
                            res = meta.get("result", None)
                            replays.append({
                                "filename": fname,
                                "game_id": meta.get("game_id", fname),
                                "created_at": meta.get("created_at", ""),
                                "players": meta.get("players", {}),
                                "total_steps": total,
                                "result": res,
                            })
                        else:
                            # Backward-compatible fallback for flat/legacy schemas (e.g. events, logs)
                            steps_list = data.get("steps") or data.get("events") or data.get("logs") or []
                            total_cnt = data.get("total_steps", len(steps_list))
                            fallback_result: Optional[ReplayResultDict] = None
                            if "winner" in data:
                                fallback_result = {
                                    "winner": str(data["winner"]),
                                    "margin": int(data.get("final_vp", 0)),
                                    "end_turn": int(data.get("final_turn", 1)),
                                    "reason": "Terminal Evaluation",
                                }
                            replays.append({
                                "filename": fname,
                                "game_id": str(data.get("game_id", fname)),
                                "created_at": str(data.get("created_at", "")),
                                "players": data.get("players", {}),
                                "total_steps": int(total_cnt),
                                "result": fallback_result,
                            })
                except Exception:
                    continue
        replays.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return replays

    @staticmethod
    def load_replay(filename: str) -> Optional[ReplayLogDict]:
        full_path = os.path.join(REPLAYS_DIR, filename)
        if not os.path.exists(full_path):
            full_path = os.path.join(REPLAYS_DIR, f"{filename}.tslog.json")
        if not os.path.exists(full_path):
            return None
        with open(full_path, "r", encoding="utf-8") as f:
            data: ReplayLogDict = json.load(f)
            return data
