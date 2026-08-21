import os
import json
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from server.session import GameSession
from server.replay import ReplayManager

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
    session = GameSession(game_id, seed=req.seed, us_player=req.us_player, ussr_player=req.ussr_player)
    active_sessions[game_id] = session
    return {"game_id": game_id, "seed": session.seed, "state": session.get_state_dict()}

@app.get("/api/games/{game_id}")
async def get_game(game_id: str):
    if game_id not in active_sessions:
        # Default auto-create session for quick debug
        session = GameSession(game_id)
        active_sessions[game_id] = session
    return active_sessions[game_id].get_state_dict()

@app.get("/api/replays")
async def list_replays():
    return ReplayManager.list_replays()

@app.get("/api/replays/{filename}")
async def get_replay(filename: str):
    data = ReplayManager.load_replay(filename)
    if not data:
        raise HTTPException(status_code=404, detail="Replay not found")
    return data

@app.get("/api/metadata/map")
async def get_map_metadata():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    map_path = os.path.join(base_dir, "rules", "map.json")
    if not os.path.exists(map_path):
        map_path = os.path.join(base_dir, "map.json")
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/metadata/cards")
async def get_cards_metadata():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cards_path = os.path.join(base_dir, "rules", "cards.json")
    if not os.path.exists(cards_path):
        cards_path = os.path.join(base_dir, "cards.json")
    with open(cards_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.websocket("/ws/game/{game_id}")
async def websocket_game(websocket: WebSocket, game_id: str, role: str = "OBSERVER"):
    if game_id not in active_sessions:
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
            elif msg_type == "DEBUG_OVERRIDE":
                await session.handle_debug_override(data.get("override", {}))
            elif msg_type == "PING":
                await websocket.send_json({"type": "PONG"})
    except WebSocketDisconnect:
        session.disconnect(websocket)
    except Exception:
        session.disconnect(websocket)

# Mount frontend if dist exists
dist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
