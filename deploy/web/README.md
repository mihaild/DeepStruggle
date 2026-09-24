# Deploying the web workbench

`Dockerfile` builds the FastAPI server, the ts_engine extension and the UI **from the GitHub
repository** (`https://github.com/mihaild/DeepStruggle`, branch `main` by default), not from the
local checkout. What it deploys is therefore whatever that ref holds: publish first, then build.

```bash
docker build -t deepstruggle-web deploy/web                      # main
docker build -t deepstruggle-web --build-arg REF=<sha> deploy/web  # a specific commit or tag
docker run -d --name deepstruggle -p 8000:8000 \
  -v /path/to/checkpoints:/data/checkpoints:ro \
  -v deepstruggle-replays:/data/replays \
  deepstruggle-web
# or: CHECKPOINTS=/path/to/checkpoints docker compose -f deploy/web/compose.yaml up -d --build
```

Then open `http://<host>:8000/?game_id=game-1`.

Needs BuildKit (the default builder since Docker 23) for the git source. Pass the directory as
the context, as above: the Dockerfile uses nothing from it, and building from the repository
root would send the whole `data/` tree to the daemon.

## What is in the image

| stage | does |
|:---|:---|
| `source` | `ADD <repo>#<ref>`: BuildKit resolves the ref to a commit every build, so a new push rebuilds and an unchanged branch hits the cache |
| `ui` | `npm ci && npm run build` → `web/ui/dist` |
| `engine` | CMake builds the `ts_engine` target only |
| runtime | `python:3.14-slim`, CPU torch, fastapi/uvicorn; runs `python -m web.server.main` as uid 1000 |

**`-march`.** The repository compiles with `-march=native`, which suits a training box and not an
image: built on one CPU, it can die with SIGILL on another. The engine stage swaps it, in the
image's copy only, for `x86-64-v2` (`armv8-a` on ARM). Pass `--build-arg ENGINE_MARCH=native` when
the image runs on the machine that builds it.

**CPU torch.** Analysis reads one position at a time (a few ms), so the image ships the CPU wheel.
For a GPU you would need a CUDA base image and wheel, and `TS_ANALYSIS_DEVICE=cuda`.

## Runtime settings

| variable | default in the image | meaning |
|:---|:---|:---|
| `TS_CHECKPOINTS_DIR` | `/data/checkpoints` | checkpoints the analysis panel offers; mount yours here |
| `TS_REPLAYS_DIR` | `/data/replays` | where finished games are written (a volume) |
| `TS_ANALYSIS_DEVICE` | `cpu` | torch device for the analysis model |
| `TS_DATA_ROOT` | `/data` | the shared data tree, so nothing resolves it through git |

A checkpoint's action view (E4 or E4.1) is read from its run directory's `metadata.json`, so
mount run directories whole, not bare `.pt` files.

There is no authentication. The server is a workbench: anyone who can reach the port can play,
load positions and pick models. Put it behind a reverse proxy with auth before exposing it.
