from web.server.replay_types import ReplaySummaryDict, ReplayLogDict, GameStateDict
import os
import sys
import json
import argparse
import logging
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from web.server.session import GameSession
from web.server.replay import ReplayManager

def setup_server_logging(log_file: Optional[str] = None, log_level_name: str = "DEBUG"):
    """Configures detailed logging using Python's standard logging module."""
    log_level = getattr(logging, log_level_name.upper(), logging.DEBUG)
    log_format = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers: List[logging.Handler] = []
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(file_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(stream_handler)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True
    )
    # Suppress uvicorn's internal truncated frame dump
    logging.getLogger("uvicorn.protocols.websockets").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.protocols.http").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

logger = logging.getLogger("ts_server")

app = FastAPI(title="Twilight Struggle Web Engine & Debugger")

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active Game Sessions in memory
active_sessions: Dict[str, GameSession] = {}

class NewGameRequest(BaseModel):
    game_id: Optional[str] = None
    seed: Optional[int] = None
    us_player: str = "US Player"
    ussr_player: str = "USSR Player"

@app.post("/api/games/new")
async def create_game(req: NewGameRequest):
    game_id = req.game_id or f"game-{len(active_sessions) + 1}"
    logger.info(f"REST: Creating new game session '{game_id}' (seed: {req.seed})")
    session = GameSession(game_id, seed=req.seed, us_player=req.us_player, ussr_player=req.ussr_player)
    active_sessions[game_id] = session
    return {"game_id": game_id, "seed": session.seed, "state": session.get_state_dict()}

@app.get("/api/games/{game_id}")
async def get_game(game_id: str):
    if game_id not in active_sessions:
        logger.info(f"REST: Auto-initializing session '{game_id}' on request")
        session = GameSession(game_id)
        active_sessions[game_id] = session
    return active_sessions[game_id].get_state_dict()

@app.post("/api/games/{game_id}/cancel_action")
@app.post("/api/games/{game_id}/undo")
async def cancel_action(game_id: str):
    if game_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Game session not found")
    session = active_sessions[game_id]
    success = await session.handle_cancel_action()
    return {"success": success, "state": session.get_state_dict()}

@app.get("/api/replays", response_model=None)
async def list_replays() -> List[ReplaySummaryDict]:
    return ReplayManager.list_replays()

@app.get("/api/replays/{filename}", response_model=None)
async def get_replay(filename: str) -> ReplayLogDict:
    data = ReplayManager.load_replay(filename)
    if not data:
        raise HTTPException(status_code=404, detail="Replay not found")
    return data

@app.get("/api/metadata/map")
async def get_map_metadata():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    map_path = os.path.join(base_dir, "rules", "map.json")
    if not os.path.exists(map_path):
        map_path = os.path.join(base_dir, "map.json")
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/metadata/cards")
async def get_cards_metadata():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cards_path = os.path.join(base_dir, "rules", "cards.json")
    if not os.path.exists(cards_path):
        cards_path = os.path.join(base_dir, "cards.json")
    with open(cards_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/metadata/action_space", response_model=None)
async def get_action_space_metadata() -> Dict[str, Any]:
    """The 212-dim flat action space, so a client can map an index back to what it means.

    A replay's policy trace stores flat indices, and the workbench has to put each probability
    on the card, mode button or country it belongs to. Served from `ActionEncoder` and the
    engine's own enum rather than copied into the frontend: a second hand-kept table of these
    offsets would eventually disagree with the encoder, and every probability would then be
    attached to the wrong thing while still looking plausible.
    """
    import ts_engine
    from bindings.action_encoder import ActionEncoder

    return {
        "size": int(ActionEncoder.FLAT_ACTION_SIZE),
        "offsets": {
            "card": int(ActionEncoder.CARD_OFFSET),
            "play_mode": int(ActionEncoder.PLAY_MODE_OFFSET),
            "op_mode": int(ActionEncoder.OP_MODE_OFFSET),
            "roll_die": int(ActionEncoder.ROLL_DIE_INDEX),
            "node": int(ActionEncoder.NODE_OFFSET),
            "branch": int(ActionEncoder.BRANCH_OFFSET),
        },
        "confirm_done_index": int(ActionEncoder.CONFIRM_DONE_INDEX),
        "decision_types": {
            name: int(getattr(ts_engine.DecisionType, name))
            for name in ("SELECT_CARD", "SELECT_PLAY_MODE", "CHOOSE_TIMING_BRANCH",
                         "SELECT_OP_MODE", "POINT_NODE", "CHOOSE_BRANCH")
        },
    }


@app.websocket("/ws/game/{game_id}")
async def websocket_game(websocket: WebSocket, game_id: str, role: str = "OBSERVER"):
    if game_id not in active_sessions:
        logger.info(f"WebSocket: Initializing new session '{game_id}'")
        active_sessions[game_id] = GameSession(game_id)
    session = active_sessions[game_id]

    await session.connect(websocket, role)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "PLAY_ACTION":
                action_data = data.get("action", {})
                await session.handle_action(action_data, sender_role=role)
            elif msg_type in ("CANCEL_ACTION", "UNDO_ACTION"):
                logger.info(f"WebSocket: Received {msg_type} from role '{role}' for game '{game_id}'")
                await session.handle_cancel_action()
            elif msg_type == "DEBUG_OVERRIDE":
                await session.handle_debug_override(data.get("override", {}))
            elif msg_type == "PING":
                await websocket.send_json({"type": "PONG"})
    except WebSocketDisconnect:
        session.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error in game '{game_id}': {e}")
        session.disconnect(websocket)

# Mount frontend if dist exists
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dist_dir = os.path.join(_ROOT_DIR, "web", "ui", "dist")
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Twilight Struggle AI Game Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--log-file", default=None, help="File path to write detailed server logs (default: stdout)")
    parser.add_argument("--log-level", default="DEBUG", help="Logging level: DEBUG, INFO, WARNING, ERROR (default: DEBUG)")
    args = parser.parse_args()

    setup_server_logging(args.log_file, args.log_level)
    logger.info(f"Starting server on {args.host}:{args.port} (log_file={args.log_file}, log_level={args.log_level})")

    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
