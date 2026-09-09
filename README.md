# Twilight Struggle AI

A complete AI, simulation engine, web workbench, and reinforcement learning training infrastructure for the Deluxe Edition of **Twilight Struggle** (110 Cards).

---

## Highlights

* **C++20 Simulation Core (`engine/`)**: Zero-allocation, high-throughput simulation engine implementing the full rules of the Deluxe Edition—including all 110 cards, persistent board effects, simultaneous headline resolution, space race tracks, and regional scoring.
* **Neural Policies & Reinforcement Learning (`ai/`)**: Graph Neural Network + ResNet architecture (`ColdWarNet`) regularized under Nash Policy Gradient (NashPG) self-play, paired with behavioral cloning from expert demonstrations and game-theoretic credit assignment.
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
├── tests/          # Python test suite across bindings, engine rules, and training
└── docs/           # Strategic encyclopedias and engine design documentation
```

For in-depth architectural and developer documentation, see [`AGENTS.md`](AGENTS.md) and [`tools/README.md`](tools/README.md).

---

## Quickstart

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/mihaild/ts_ai.git
cd ts_ai

# Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or install nanobind fastapi uvicorn websockets pytest torch numpy
```

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

Run the test suite:

```bash
PYTHONPATH=.:build/release pytest
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Disclaimer

*Twilight Struggle* is a registered trademark of GMT Games LLC and was designed by Ananda Gupta and Jason Matthews. This repository is an independent research implementation developed strictly for academic, educational, and game-theoretic study. It is not affiliated with, authorized by, or endorsed by GMT Games LLC.
