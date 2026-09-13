# Deep Struggle

Welcome to Deep Struggle! It is a project dedicated to build an AI that plays the Deluxe Edition of **Twilight Struggle** (GMT Games, 2005).

Current status:

* engine in C++ that mostly reproduces rules - some bugs are probably still there, but supposedly nothing major
* interface to plumb different player implementations, record games between them, replay them in web viewer, and run tournamnets
* some baseline network architecture and training scripts, that in few hours on 4090 can learn to almost never propose Olympic Games at Defcon 2

## AI Slop warning

This repository is developed with heavy usage of AI. Most code is reviewed, but not all of it, neither documentation. Be even more skeptical code quality here than usually.

Known, reproducible, unfixed defects are tracked in [`BUGS.md`](BUGS.md).

---

## Highlights

* **C++20 Simulation Core (`engine/`)**: Zero-allocation, high-throughput simulation engine covering the Deluxe Edition—all 110 cards, persistent board effects, simultaneous headline resolution, space race tracks, and regional scoring. Every card has an event handler, but that is not the same as every rule being right: known rules defects are tracked in [`BUGS.md`](BUGS.md).
* **Neural Policies & Reinforcement Learning (`ai/`)**: Graph Neural Network + ResNet architecture (`ColdWarNet`) regularized under Nash Policy Gradient (NashPG) self-play, paired with behavioral cloning from demonstration datasets (self-play and a corpus of recorded human games) and a suite of evaluation probes in `ai/eval/`.
* **Interactive Web Workbench (`web/`)**: Full-featured web interface powered by FastAPI, WebSockets, and a Vite + TypeScript SVG Deluxe Map supporting human-vs-bot matches, bot-vs-bot exhibitions, and complete replay timelines (`.tslog.json`).
* **Evaluation & Tournament Suite (`tools/`)**: High-throughput tournament runner with Bradley-Terry Maximum Likelihood Elo estimation, automated loss-cause diagnostics, and match playback.

---

## Architecture Overview

```
.
├── engine/         # Core C++20 zero-allocation simulation engine
├── bindings/       # Native Python nanobind module (ts_engine) & environment bridge
├── ai/             # Neural networks (ColdWarNet), rewards, and NashPG RL pipelines
├── bot/            # Baseline bots (HeuristicBot, RandomBot, StrategicBot, NeuralBot)
├── web/            # Full-stack Web Workbench (FastAPI server + SVG map UI)
├── tools/          # Unified CLI suite for training, tournaments, matches, and replays
├── rules/          # Formal specifications (cards, map topology, effect flags)
└── tests/          # Python test suite across bindings, engine rules, and training
```

For in-depth architectural and developer documentation, see [`AGENTS.md`](AGENTS.md) and [`tools/README.md`](tools/README.md).

---

## Quickstart

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/mihaild/DeepStruggle.git
cd DeepStruggle

# Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA build of PyTorch, install it from [pytorch.org](https://pytorch.org) first; the rest
of `requirements.txt` installs on top of whatever `torch` is already present.

### 2. Build C++ Engine

```bash
cmake -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
cmake --build build/release -j
```

### 3. Build Web UI & Launch Workbench

```bash
# Build the frontend assets
cd web/ui && npm install && npm run build && cd ../..

# Start the game server
PYTHONPATH=.:build/release uvicorn web.server.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser to launch the Web Workbench.

---

## Running Tests

Build the engine first. `pytest` compares the `ts_engine` it actually imported against the
`engine/` and `bindings/` sources and aborts the run if it is stale, since a plausible answer
from yesterday's rules is worse than a failure. With no extension built at all it says nothing —
the tests that need one fail on their own import, with a clearer message.

```bash
# Engine rules, the nanobind surface, and the training stack. ~1 min with -n auto.
PYTHONPATH=.:build/release pytest -n auto tests/bindings tests/engine_logic tests/training
```

Two suites need something extra and are worth knowing about before you run a bare `pytest`:

```bash
# tests/replayer converts a corpus of real human games. Downloaded once per machine, cached
# outside the repository, ~5 MB.
PYTHONPATH=. python tools/download_ts_replayer.py
PYTHONPATH=.:build/release pytest -n auto tests/replayer

# tests/web needs the built frontend and a browser, neither of which is in the repository.
cd web/ui && npm install && npm run build && cd ../..
python -m playwright install chromium
PYTHONPATH=.:build/release pytest tests/web
```

`tests/differential/` cross-checks against an external reference engine. It is work in progress,
is excluded from collection, and is not part of the check a change is expected to pass.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Disclaimer

*Twilight Struggle* is a registered trademark of GMT Games LLC and was designed by Ananda Gupta and Jason Matthews. This repository is an independent research implementation developed strictly for academic, educational, and game-theoretic study. It is not affiliated with, authorized by, or endorsed by GMT Games LLC.
