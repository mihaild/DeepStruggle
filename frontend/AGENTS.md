# Frontend Web Workbench Developer Guide

This directory contains the browser-based client and interactive debugging workbench for Twilight Struggle, built with **Vite, TypeScript, and SVG**.

---

## 1. Directory & Component Overview

```
frontend/
├── index.html              # Main single-page workbench layout
├── package.json            # Node dependencies (Vite, TypeScript)
├── tsconfig.json           # TypeScript configuration (strict mode)
├── vite.config.ts          # Vite build config & WebSocket proxy
├── public/                 # Static assets (map.json, cards.json)
└── src/
    ├── style.css           # Dark theme design system & SVG node styling
    ├── types.ts            # TypeScript interfaces matching engine state dict
    ├── main.ts             # Main orchestrator & WebSocket connection
    ├── map_view.ts         # Vector SVG map renderer with pan/zoom and click targets
    ├── tracks_view.ts      # DEFCON, VP meter, Turn/AR, MilOps, and Space tracks
    ├── cards_view.ts       # Full hands display, China Card, discard/removed modals
    ├── action_hud.ts       # Decision HUD translating context into interactive buttons
    ├── debug_panel.ts      # Context stack viewer, dice roll log, state tweaker
    └── replay_controls.ts  # Timeline scrubber, transport controls, export/import
```

---

## 2. Key Architecture Details

- **Interactive Vector Map (`map_view.ts`)**:
  - Renders countries from `map.json` coordinates.
  - Battleground banners (purple), stability shields, US/USSR influence counter boxes, and control outlines.
  - When in `POINT_NODE` mode (`decision_type === 5`), legal countries are marked with `.legal-target` and animated with a glowing gold pulse (`@keyframes pulse-legal-border`).
  - Supports mouse-wheel zoom and click-and-drag panning across the global map.

- **Full State Transparency**:
  - Both US and USSR hands are always visible (no secrecy required), facilitating immediate game inspection and AI policy debugging.

- **Replay & Timeline Scrubbing (`replay_controls.ts`)**:
  - Allows timeline scrubbing through all past game steps, variable speed playback, and JSON export/import.

---

## 3. How to Build & Run

```bash
# Install dependencies
npm install

# Run Vite dev server with live reload (proxying API to port 8000)
npm run dev

# Build production bundle into frontend/dist/ (served directly by FastAPI)
npm run build
```
