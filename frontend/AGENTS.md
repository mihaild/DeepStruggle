# Frontend & Web Workbench Developer Guide

This directory contains the modern, interactive web client and development workbench for **Twilight Struggle**, built with **Vite + TypeScript + Vanilla CSS + SVG**.

---

## 1. Architecture & Component Structure

```
frontend/
├── index.html                  # Workbench layout (Top Bar, SVG Map, Decision HUD, Bottom Drawer)
├── package.json                # Dependencies and build scripts
├── vite.config.ts              # Vite server and WebSocket proxy config
├── src/
│   ├── main.ts                 # TSApp coordinator, global metadata loading, WebSocket client, Action Stream
│   ├── map_view.ts             # SVG Deluxe Map renderer (84 countries, lines, influence badges, pan/zoom)
│   ├── cards_view.ts           # Hand tabs, Card Explorer, and Active Continuous Effects panel (43 flags)
│   ├── tracks_view.ts          # Turn, Action Round, DEFCON, Mil Ops, Space Race, and Victory Points tracks
│   ├── action_hud.ts           # Decision HUD, human-readable branch catalog (CARD_BRANCHES), and dice selector
│   ├── replay_controls.ts      # Replay timeline scrubber, playback engine, and server replay picker
│   ├── debug_panel.ts          # State inspector and engine override tools
│   ├── style.css               # Theme tokens, dark mode palette, animations, layout styles
│   └── types.ts                # TypeScript interface mirrors of engine GameState and MicroAction
└── dist/                       # Production bundle (served directly by FastAPI at http://localhost:8000)
```

---

## 2. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Frontend Documentation Synchronized**:
> When adding new UI components, modifying SVG styling, expanding effect catalogs, adding branch mappings, or altering API metadata consumption, you **MUST** update this file and root [`AGENTS.md`](file:///home/mihaild/prog/ts_ai/AGENTS.md).

---

## 3. Key Frontend Features & Capabilities

1. **Interactive SVG Deluxe Map (`map_view.ts`)**:
   - Renders all 84 countries with regional color coding, battleground badges, stability indicators, and US/USSR influence counts.
   - Highlights valid country targets with dynamic glowing borders when `POINT_NODE` decision is active.
   - Smooth mouse pan and zoom with automatic metadata loading and immediate re-rendering.
2. **Active Continuous Effects Panel (`cards_view.ts`)**:
   - Displays all 47 persistent continuous effect & state bits (e.g. *Containment*, *Brezhnev Doctrine*, *Red Scare/Purge*, *Quagmire*, *Bear Trap*, *Flower Power*, *Willy Brandt*, *NORAD*, *NATO*, *Space Race Bonuses*) with side tags (`US`, `USSR`, `Both`, `Neutral`), duration badges (`Turn Only`, `Permanent`, `Conditional`, `Space Perk`), and rule summaries.
3. **Descriptive Branch Choice HUD (`action_hud.ts`)**:
   - `CARD_BRANCHES` catalog translates numeric branch IDs into clear, human-readable options (e.g. *Branch 0: Award 2 VP / Branch 1: Conduct 4 Ops* for Olympic Games; *Branch 0: Remove all US Influence / Branch 1: Add 5 USSR Influence* for Warsaw Pact).
4. **Deluxe Space Race Track & Modal Inspector (`tracks_view.ts`)**:
   - Renders 9-step ladder (0..8) in the top status bar with US and USSR marker tokens and turn attempt counters (`Attempts: used/max`).
   - Clickable Space Race widget opens the full Deluxe Space Race Modal displaying all 9 boxes, required Ops, success rolls, 1st/2nd VP awards, and special ongoing privileges.
5. **Interactive Action Stream & Replay Timeline (`replay_controls.ts` & `main.ts`)**:
   - Full timeline scrubber with play/pause, step forward/backward, and server replay loading.
   - Bottom Action Stream displays all recorded game events with active step highlighting (`.active-replay-step`) and click-to-scrub navigation.
5. **Robust Autonomous Metadata Pipeline**:
   - Views autonomously load `/api/metadata/map` and `/api/metadata/cards` and cache `lastState` so views refresh immediately as soon as metadata arrives.

---

## 4. Development & Build Commands

```bash
# Install dependencies
npm install

# Start Vite live development server (proxies WebSocket / REST to port 8000)
npm run dev

# Build production bundle into frontend/dist/ (served directly by FastAPI backend)
npm run build
```

---

## 5. Verification & Regression Testing

Frontend metadata delivery, index.html DOM structure, and full-fidelity replay snapshots are verified automatically via:
```bash
PYTHONPATH=. .venv/bin/pytest -v tests/test_web_workbench.py
```
