"""The local workbench server: the page, plus this machine's checkpoints and replays.

The workbench itself runs in the browser -- the engine as WebAssembly, models with onnxruntime-
web -- and the same page is published to GitHub Pages with no server at all. Locally, this adds
what a static page cannot reach:

    GET /api/local/info                     engine fingerprint of the sources, roots
    GET /api/local/models                   checkpoints, grouped by run
    GET /api/local/models/onnx?path=<rel>   a checkpoint as ONNX (exported + verified, cached)
    GET /api/local/replays                  saved .tslog.json replays
    GET /api/local/replays/<file>           one replay

and serves the built page (web/ui/dist, from tools/scripts/build_web.sh) at `/`.

    PYTHONPATH=.:build/release .venv/bin/python -m web.server.main --port 8000
"""
from web.server.replay_types import AnalysisModelListDict, ReplayLogDict, ReplaySummaryDict
import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from tools.lib.engine_fingerprint import fingerprint
from web.server.local_files import (LocalFileError, list_models, models_root, onnx_cache_root,
                                    onnx_for)
from web.server.replay import ReplayManager, replays_dir


def setup_server_logging(log_file: Optional[str] = None, log_level_name: str = "INFO") -> None:
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    log_format = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    handler: logging.Handler
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.basicConfig(level=log_level, handlers=[handler], force=True)
    logging.getLogger("uvicorn.protocols.http").setLevel(logging.WARNING)


logger = logging.getLogger("ts_server")

app = FastAPI(title="Twilight Struggle Workbench (local files)")

# The Vite dev server (npm run dev) runs on another port and proxies /api here; allow it.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


@app.get("/api/local/info")
async def local_info() -> Dict[str, Any]:
    """What the page compares its own engine against: a page built from other sources than
    these plays other rules, and says so."""
    return {
        "engine_fingerprint": fingerprint(),
        "checkpoints_root": models_root(),
        "replays_dir": replays_dir(),
        "onnx_cache": onnx_cache_root(),
    }


@app.get("/api/local/models", response_model=None)
async def local_models() -> AnalysisModelListDict:
    return list_models()


@app.get("/api/local/models/onnx", response_model=None)
async def local_model_onnx(path: str) -> FileResponse:
    """A checkpoint as ONNX for the page. The first request exports it (a few seconds: the
    export is verified against torch on real positions); later ones are served from the cache."""
    try:
        onnx_path = await run_in_threadpool(onnx_for, path)
    except LocalFileError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # an export refused, or a checkpoint that does not load
        logger.warning(f"ONNX export of {path} failed: {e}")
        raise HTTPException(status_code=422, detail=f"could not export {path}: {e}")
    return FileResponse(onnx_path, media_type="application/octet-stream",
                        filename=os.path.basename(path).replace(".pt", ".onnx"))


@app.get("/api/local/replays", response_model=None)
async def local_replays() -> List[ReplaySummaryDict]:
    return ReplayManager.list_replays()


@app.get("/api/local/replays/{filename}", response_model=None)
async def local_replay(filename: str) -> ReplayLogDict:
    data = ReplayManager.load_replay(filename)
    if not data:
        raise HTTPException(status_code=404, detail="Replay not found")
    return data


_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dist_dir = os.path.join(_ROOT_DIR, "web", "ui", "dist")
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Twilight Struggle workbench -- local files server")
    parser.add_argument("--host", default="127.0.0.1", help="host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_server_logging(args.log_file, args.log_level)
    if not os.path.exists(dist_dir):
        logger.warning("web/ui/dist is missing: build the page with tools/scripts/build_web.sh")
    logger.info(f"Workbench on http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
