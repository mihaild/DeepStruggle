# Frontend & Web Workbench Developer Guide (`web/ui/`)

The interactive web client and development workbench for **Twilight Struggle**, built with
**Vite + TypeScript + vanilla CSS + SVG**. FastAPI serves the built bundle from `dist/`.

---

## 1. Architecture & Component Structure

```
web/ui/
├── index.html                  # Workbench layout (Top Bar, SVG Map, Decision HUD, Bottom Drawer)
├── package.json                # Dependencies and build scripts
├── vite.config.ts              # Vite server and WebSocket proxy config
├── tsconfig.json
├── src/
│   ├── main.ts                 # TSApp coordinator, global metadata loading, WebSocket client, Action Stream
│   ├── map_view.ts             # SVG Deluxe Map renderer (84 countries, lines, influence badges, pan/zoom)
│   ├── cards_view.ts           # Hand tabs, Card Explorer, and the Active Continuous Effects panel (EFFECT_INFO_MAP)
│   ├── tracks_view.ts          # Turn, Action Round, DEFCON, Mil Ops, Space Race, and Victory Points tracks
│   ├── action_hud.ts           # Decision HUD, human-readable branch catalog (CARD_BRANCHES), and dice selector
│   ├── replay_controls.ts      # Replay timeline scrubber, playback engine, and server replay picker
│   ├── debug_panel.ts          # State inspector and engine override tools
│   ├── style.css               # Theme tokens, dark mode palette, animations, layout styles
│   └── types.ts                # TypeScript interface mirrors of engine GameState and MicroAction
└── dist/                       # Production bundle, not committed (served by FastAPI at http://localhost:8000)
```

---

## 2. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Frontend Documentation Synchronized**:
> When adding new UI components, modifying SVG styling, expanding effect catalogs, adding branch mappings, or altering API metadata consumption, you **MUST** update this file and root [`AGENTS.md`](../../AGENTS.md).

---

## 3. Key Frontend Features & Capabilities

1. **Interactive SVG Deluxe Map (`map_view.ts`)**:
   - Renders all 84 countries with regional color coding, battleground badges, stability indicators, and US/USSR influence counts.
   - Highlights valid country targets with dynamic glowing borders when a `POINT_NODE` decision is active.
   - Mouse pan and zoom, with metadata loaded autonomously and re-rendered on arrival.
2. **Active Continuous Effects Panel (`cards_view.ts`)**:
   - `EFFECT_INFO_MAP` describes every persistent continuous-effect and state bit (e.g. *Containment*, *Brezhnev Doctrine*, *Red Scare/Purge*, *Quagmire*, *Bear Trap*, *Flower Power*, *Willy Brandt*, *NORAD*, *NATO*, Space Race bonuses) with side tags (`US`, `USSR`, `Both`, `Neutral`), duration badges (`Turn Only`, `Permanent`, `Conditional`, `Space Perk`), and rule summaries. Add an entry here whenever the engine gains an effect bit.
3. **Descriptive Branch Choice HUD (`action_hud.ts`)**:
   - `CARD_BRANCHES` translates numeric branch IDs into readable options (e.g. *Branch 0: Award 2 VP / Branch 1: Conduct 4 Ops* for Olympic Games; *Branch 0: Remove all US Influence / Branch 1: Add 5 USSR Influence* for Warsaw Pact).
4. **Deluxe Space Race Track & Modal Inspector (`tracks_view.ts`)**:
   - Renders the 9-step ladder (0..8) in the top status bar with US and USSR marker tokens and turn attempt counters (`Attempts: used/max`).
   - Clicking the widget opens the full modal: all 9 boxes, required Ops, success rolls, 1st/2nd VP awards, and special ongoing privileges.
5. **Interactive Action Stream & Replay Timeline (`replay_controls.ts`, `main.ts`)**:
   - Timeline scrubber with play/pause, step forward/backward, and server replay loading.
   - The bottom Action Stream lists recorded game events with active-step highlighting (`.active-replay-step`) and click-to-scrub navigation.
6. **Autonomous metadata pipeline**:
   - Views load `/api/metadata/map` and `/api/metadata/cards` themselves and cache `lastState`, so they refresh as soon as metadata arrives.

---

## 4. Development & Build Commands

```bash
npm install                     # install dependencies
npm run dev                     # Vite dev server, proxying WebSocket / REST to port 8000
npm run build                   # production bundle into dist/, served by the FastAPI backend
```

---

## 5. Verification & Regression Testing

`dist/` is not committed, so build it before running the web suite. Invoke pytest as a module,
never via `.venv/bin/pytest`: that console script carries an absolute shebang and breaks if the
venv is moved.

```bash
npm run build
PYTHONPATH=.:build/release .venv/bin/python -m pytest -q tests/web/test_web_workbench.py
```

Covers metadata delivery, the `index.html` DOM structure, and full-fidelity replay snapshots.

### End-to-End Headless Chrome Tests

Requires a Playwright browser (`.venv/bin/python -m playwright install chromium`).

```bash
PYTHONPATH=.:build/release .venv/bin/python -m pytest -q tests/web/test_e2e_space_race.py
```

Validates top header layout, Space Race widget rendering without clipping or overflow, modal
open/close (click and Escape), the milestone box metadata, and live replay timeline state
synchronization.
