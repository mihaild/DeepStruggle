# Frontend & Web Workbench Developer Guide (`web/ui/`)

The interactive web client and development workbench for **Twilight Struggle**, built with
**Vite + TypeScript + vanilla CSS + SVG**. It runs entirely in the browser -- the engine as
WebAssembly, models as ONNX -- so `dist/` works from GitHub Pages as well as from the local server
(`web/server/main.py`), which adds this machine's checkpoints and replays.

---

## 1. Architecture & Component Structure

```
web/ui/
├── index.html                  # Workbench layout (Top Bar; cards | SVG Map | HUD rail; Bottom Drawer)
├── leaderboard.html            # The Elo leaderboard page (src/leaderboard_page.ts, src/leaderboard.css)
├── package.json                # Dependencies and build scripts
├── vite.config.ts              # base path (TS_WEB_BASE, for Pages), /api proxy, the two pages,
│                               # the build's commit and the stats endpoint (TS_STATS_ENDPOINT)
├── tsconfig.json
├── src/
│   ├── main.ts                 # TSApp coordinator: boot, refresh(), links, auto-play, Action Stream
│   ├── engine/wasm_engine.ts   # the WebAssembly engine (public/engine/, built by build_web.sh)
│   ├── game/                   # session.ts (stepping, undo, log, export), describe.ts (log text),
│   │                           # names.ts (flat action names), position.ts (link tokens),
│   │                           # branches.ts (what an event's branch choice means: HUD and log),
│   │                           # view.ts (the US / USSR player views: what the other side hides),
│   │                           # last_turn.ts (what the other side's latest turn did to the map),
│   │                           # rolls.ts (the latest action round's dice, from the steps),
│   │                           # record.ts (a game for the stats database: log, tags, summary),
│   │                           # card_refs.ts (the cards a log line names, made hoverable),
│   │                           # saved_game.ts (the game kept in the browser, for a reload)
│   ├── result_dialog.ts        # Game over: the result, how it was played, submit to the stats database
│   ├── build_info.ts           # the commit the page was built from, the stats endpoint
│   ├── analysis/               # model.ts (ONNX sources + onnxruntime-web), hf_catalog.ts (the
│   │                           # Hugging Face repo's catalogue, default model and default search, no
│   │                           # onnxruntime), rollout_spec.ts (the search= grammar, no engine),
│   │                           # onnx_meta.ts,
│   │                           # model_cache.ts (downloaded models kept in the browser),
│   │                           # readout.ts (policy + critic for the position on screen),
│   │                           # rollout.ts / rollout_runner.ts / rollout_worker.ts (the rollout root
│   │                           # in a worker with its own engine and model; bindings/rollout_batch.hpp)
│   ├── metadata.ts             # rules/map.json and rules/cards.json, bundled
│   ├── map_view.ts             # SVG Deluxe Map renderer (84 countries, lines, influence markers, region
│   │                           # scoring badges, hover card, pan/pinch/zoom, region jump)
│   ├── region_summary.ts       # A region's scoring summary, the two sides side by side
│   ├── roll_tray.ts            # The map's roll tray: dice faces, modifiers, totals and outcomes
│   ├── cards_view.ts           # Hand tabs + Card Explorer (left column), Active Continuous Effects panel (EFFECT_INFO_MAP)
│   ├── tracks_view.ts          # Turn, Action Round, DEFCON, Mil Ops, Space Race, and Victory Points tracks
│   ├── action_hud.ts           # Decision HUD (branch choices worded by game/branches.ts) and dice selector
│   ├── replay_controls.ts      # Replay timeline scrubber, playback engine, and server replay picker
│   ├── trace_view.ts           # Replay trace: value ribbon, log chips, readout panel, probability badges;
│   │                           # the critic display (which reading is calibrated) shared with analysis_view
│   ├── analysis_view.ts        # Live model analysis panel: model picker, critic readings, top choices, badges
│   ├── debug_panel.ts          # State inspector and engine override tools
│   ├── style.css               # Theme tokens, dark mode palette, animations, layout styles
│   ├── leaderboard_page.ts     # The leaderboard: views (main / + lineages / all) filter one fit,
│   │                           # head-to-head on click, links to reports, weights and the workbench
│   │                           # (a player's own report; a search never shows its network's greedy one);
│   │                           # opens on the newest epoch with games played (?epoch= names another);
│   │                           # play links open each player's search; tags HF DEFAULT on the player
│   │                           # that is the repo's default model and search, read live from
│   │                           # models.json (hf_catalog.ts) after render
│   └── types.ts                # TypeScript interface mirrors of engine GameState and MicroAction
├── public/engine/              # ts_engine.mjs + .wasm, not committed (tools/scripts/build_web.sh)
├── public/leaderboard.json     # fitted from leaderboard/ by build_web.sh (tools/leaderboard.py fit), not committed
└── dist/                       # Production bundle, not committed (local server, or GitHub Pages)
```

---

## 2. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Frontend Documentation Synchronized**:
> When adding new UI components, modifying SVG styling, expanding effect catalogs, adding branch mappings, or altering API metadata consumption, you **MUST** update this file and root [`AGENTS.md`](../../AGENTS.md).

---

## 3. Key Frontend Features & Capabilities

1. **Interactive SVG Deluxe Map (`map_view.ts`)**:
   - Each country is a 38x22 box: a band in its region's colour (`REGION_STYLE`; Southeast Asia
     from `subregions`) with the name and the stability (purple for a battleground), over two
     **influence markers as on the physical board**: US left, USSR right; a number in the side's
     colour on white, *flipped* (white on the side's colour, `rect.inf-box` stroked white) when
     that influence controls the country; an empty space is a faint dashed outline with no
     number (`influenceMarker`). Long names are shortened (`SHORT_NAMES`) or squeezed.
   - **Control is the card's fill**: blue for a US-controlled country, red for a USSR one, the
     controller's influence box outlined in white (`data-control` on the node). Not the border
     alone, because a legal target's border is replaced by the pulsing highlight.
   - **Hovering a country** dims the rest of the map, lights its neighbours and the lines to them,
     and opens a card (`showCountryTooltip`): region, subregions, stability, influence, what each
     side needs (`controlLine`: control with +n at n Ops; against the other side's control, break
     with +b at 2b Ops, then control with +c at 2b+(c-b)), the coup target (2x stability) and
     whether DEFCON closes the region (`coupBlockedAt`), the last turn's change and the
     neighbours. Influence that changed since the last render flashes once (`.inf-changed`; not
     on the first render, nor when more than 12 countries changed -- a new position, not a move).
   - Highlights valid country targets with dynamic glowing borders when a `POINT_NODE` decision is active.
   - Each region's **scoring badge** (104x34 map units, in the region's accent colour, placed
     clear of every country box and of the turn-effects strip) gives the net VP the scoring card
     would award now, each side's standing (Pres / Dom / Ctrl and its VP; Southeast Asia has VP
     only), a `NO COUPS` chip when DEFCON closes the region (`.svg-region-defcon`; Southeast Asia
     closes with Asia), and a **scoring-card row** (`scoringCardWhereabouts`, from
     `state.card_locations`): `LIVE` while the card is in the draw deck or a hand (or headlined,
     or peeked), `OUT` once discarded or removed, `NOT YET` for a Mid War card before Mid War.
     Hovering a badge shows, and clicking or tapping it opens in the modal, the region's
     **scoring summary** (`region_summary.ts`, after the Playdek app's scoring screen): the US on
     the left, the USSR on the right, rows for standing, battlegrounds, other countries, countries
     next to the enemy superpower and the total, who controls what, the card's terms and where the
     card is.
   - A model's probability for a country (`trace_view.ts` `svgBadge`) is a small badge right of
     the rightmost influence marker, so it never covers a number; its font is sized to fit (not
     `textLength`, which leaves the measured box at the natural width). The HTML badge's CSS
     stays scoped to `span.trace-choice-badge`, or it reaches the SVG text.
   - **The roll tray** (`#map-rolls`, `roll_tray.ts`) shows the latest action round's dice at the
     top right of the map -- coup, realignment, Space Race, war, Olympics, Summit, trap escape --
     as die faces in the roller's colour with modifier, total, target and outcome, and the
     country a coup, realignment or war rolled in carries the die above its box (`.rolled`,
     `data-roll`; a Space Race roll's `country_id` is its box, not a country). The engine clears
     `die_roll` at every step, so the rolls come from the steps (`game/rolls.ts`, attached by
     `renderState` as `recent_rolls`, in replays too): a live step carries every die its click
     rolled (`ReplayStep.rolls`, from `clickRolls` -- one click can roll twice, and the cleanup
     after a turn's last action round clears the roll from the state it ends on), a replay file's
     step only its snapshot's roll. They show during the half action round they were made in and
     the next (the reply; the next turn's headline follows the last action round), after that
     they are the log's. The Olympic Games record holds the sponsor's die and its +2 apart
     (`roll1` = die, `mod1` = 2), like every other roll; until 2026-10-19 `roll1` was die + 2.
   - **This turn's effects** (`#map-turn-effects`, `CardsView.renderTurnEffects`) run across the
     top of the map: every flag whose `EFFECT_INFO_MAP` duration is `TURN`, the card's text on
     hover. Permanent and conditional effects, and the Space Race attempt flags, stay in the
     right rail's effects panel.
   - Pan and zoom through Pointer Events, one path for mouse and touch: drag to pan, two fingers
     to pinch (`zoomAt` keeps the point between them fixed), the wheel zooms at the cursor. A
     press that moves more than `DRAG_SLOP` (6px) is a pan and the click after it is swallowed in
     the capture phase, so a pan never chooses a country. `#map-region-jump` (Eu/As/SE/ME/Af/CA/SA,
     as the Playdek app's side buttons) frames a region (`zoomToRegion`); the reset button shows
     the whole map.
   - **Touch has no hover.** A tap on a country that is not a target shows its card (on a phone
     a bottom sheet) and lights its neighbours; a tap on open map puts it away; a target is still
     chosen by the tap. On touch, mouseenter/mouseleave are ignored (`lastPointer`): Chromium's
     compatibility mouse events after a tap fire a mouseleave elsewhere and would close the card.
2. **Layout: cards on the left, the board, the HUD rail on the right (`index.html`, `style.css`)**:
   - The top of the page is three blocks: the disclaimer; the technical bar (engine badge, play
     mode with New Game, the **model · mode label** (`#btn-model-choose`: the model's name and
     *Rollout root* / *Policy*, `renderModelModeLabels`) and Show analysis, then the developer's
     tools: Cancel Action, Debug Tools, Replay Mode, Leaderboard); and the game bar (the tracks and
     whose decision it is). A fresh open (a link naming no position, replay or view) and every New
     Game first ask for the play mode (`#mode-dialog`, `main.ts askMode`).
   - **The model & search section** (`#model-mode-section`): where the model comes from (the
     source, run / snapshot, Hugging Face repo or file -- `analysis_view.ts` binds them by id) and
     how it plays (`#mode-opponent`: rollout root or policy). It is one copy in the DOM, moved
     (`moveModelSection`) into whichever dialog shows it: the label opens it in
     `#model-mode-dialog` (`openModelMode`; a modal, closed by its button, the backdrop or Escape),
     and the play-mode question holds it folded in `#mode-model-details`, under a one-line summary
     of what it says, folded again on every opening. A model picked in either loads at once; how
     the search radios apply depends on the dialog (*How the model plays a side* below).
   - `.main-content` is a three-column grid: `#cards-sidebar` (the hands panel -- USSR / US /
     Piles / All tabs, the switch unchanged) the full height of the board on the left, the map in
     the middle, and the right rail with the Decision HUD, model analysis / replay readout and
     active effects. Below 1280px the side columns narrow; below 960px it becomes one scrolling
     column: map, HUD rail, cards.
   - **Phones (720px and narrower)**: the whole page scrolls, nothing is wider than the screen.
     The bars wrap, the map is `clamp(380px, 65vw + 150px, 72svh)` tall with the zoom buttons in
     a row at its top left and the region buttons along its bottom, the action log and replay bar
     stack below, and the modal fills the width. `(pointer: coarse)` enlarges the buttons on any
     touch screen.
   - The *All* tab lists each card's name and where it is -- no era/side line and no hover
     description (the hand tabs keep their tooltip); a click opens the card's text.
   - The *Piles* tab lists the Discard Pile and Removed from Game in the tab itself
     (`renderPile`): each card as the hand tabs draw it (`cardItem`: era bar, Ops in the side's
     colour, number and name, the text on hover), packed -- no gap between the cards
     (`.card-list-packed`) -- by side (USSR, US, Neutral, as the tabs run), then highest Ops first
     (`sortPile`). The tab scrolls as a whole, not each pile. A pile's cards carry no
     `data-card-id`, so the model's probabilities are painted on the hands only. It replaced the
     Discard / Removed buttons, whose modal had a column a side.
3. **Active Continuous Effects Panel (`cards_view.ts`)**:
   - `EFFECT_INFO_MAP` describes every persistent continuous-effect and state bit (e.g. *Containment*, *Brezhnev Doctrine*, *Red Scare/Purge*, *Quagmire*, *Bear Trap*, *Flower Power*, *Willy Brandt*, *NORAD*, *NATO*, Space Race bonuses) with side tags (`US`, `USSR`, `Both`, `Neutral`), duration badges (`Turn Only`, `Permanent`, `Conditional`, `Space Perk`), and rule summaries. Add an entry here whenever the engine gains an effect bit.
4. **Descriptive Branch Choice HUD (`action_hud.ts`)**:
   - `game/branches.ts` says what each branch of an event's CHOOSE_BRANCH means -- the engine asks
     for one in six events: Warsaw Pact Formed, Olympic Games, Wargames, Chernobyl (a region),
     Summit and How I Learned to Stop Worrying (the DEFCON level to set, marked unchanged / improve /
     degrade against the current one; Summit offers the current level and one either side). The
     HUD's buttons, the action log ("US designates Africa (Chernobyl)", "USSR sets DEFCON to 4
     (improve) (Summit)") and the analysis panel's choice names (`game/names.ts`) all read it, so
     no option shows as a bare number. It replaced `CARD_BRANCHES`, whose Summit entries described
     the pre-P17 +1/-1 branches and whose other entries were for cards that ask no branch.
   - An event's card choice (`SELECT_CARD` with a `resolving_card`) lists every offered card as a
     button in the HUD -- name, ops, side and where it is -- because those cards are mostly in no
     hand tab: Our Man in Tehran's five drawn cards (`PEEKED_TEMP`, "Drawn"), Star Wars' and SALT
     Negotiations' discard pile. The pass button appears whenever the mask offers card 0, labelled
     for what it means there (Done / Take no card / No card can be chosen).
   - Grain Sales to Soviets resolves the card it drew on a play-mode node (`resolving_card` 67,
     the drawn card in `pending_op_card`): the HUD names both, offers the play modes, and adds
     *Return … to the USSR* -- the engine's CONFIRM_DONE there (primary 255, flag 128; signalled by
     `allow_early_stop`, outside `valid_ids`), which hands the card back for Grain Sales' 2 Ops.
     Without it the page could only play the drawn card. Ops not yet granted read "the card's Ops",
     not "0 Ops".
   - A target decision whose eligible targets have run out offers only CONFIRM_DONE in the
     engine's flat mask, and the display's per-decision mask has no slot for it: with
     `allow_early_stop` off (a mandatory removal) the panel was empty and the game stuck -- Muslim
     Revolution with US Influence in one of its eight countries. The session sets
     `legal_actions.can_confirm_done` from the flat mask, and the panel then offers "✓ Done — no
     eligible target left" (`tests/web/test_e2e_muslim_revolution.py`).
   - `tests/web/test_e2e_card_choices.py` plays all of these from the US hand.
5. **Deluxe Space Race Track & Modal Inspector (`tracks_view.ts`)**:
   - Renders the 9-step ladder (0..8) in the top status bar with US and USSR marker tokens and turn attempt counters (`Attempts: used/max`).
     Each side's pill says what its next attempt needs (`.pill-next`, "next 2 Ops · 1–3", the
     Playdek app's *NEXT PROJECT*). `SPACE_RACE_BOXES` follows `engine/src/space_race.cpp`: the
     rolls alternate 1–3 / 1–4 up to box 7 (Man in Space is box 3, Man in Earth Orbit box 4).
   - Clicking the widget opens the full modal: all 9 boxes, required Ops, success rolls, 1st/2nd VP awards, and special ongoing privileges.
6. **Interactive Action Stream & Replay Timeline (`replay_controls.ts`, `main.ts`)**:
   - Timeline scrubber with play/pause, step forward/backward, and server replay loading.
   - The bottom Action Stream lists recorded game events with active-step highlighting (`.active-replay-step`) and click-to-scrub navigation.
7. **The game runs in the page (`engine/`, `game/`, `main.ts`)**:
   - `engine/wasm_engine.ts` loads the WebAssembly build of `engine/` + `bindings/`
     (`bindings/wasm/ts_engine_wasm.cpp`, built into `public/engine/` by
     `tools/scripts/build_web.sh`). Display state, saves, rules and observations are the same C++
     as the Python stack; `tests/web/test_wasm_engine.py` holds the two builds to bit-identical
     games. **After any engine change, rebuild the page** -- it runs its own copy of the engine.
     The header badge shows the page engine's fingerprint and turns **ENGINE STALE** when the
     local server's sources have moved on (`/api/local/info`).
   - `game/session.ts` is the game: apply a click (a MicroAction; `secondary_id` carries the HUD's
     manual die) or a model's flat action, resolve dice, undo (raw GameState snapshots; the log and
     the replay are restored with it), debug overrides (undoable), load a shared position, export
     the game as a `.tslog.json` (*Export Replay*). A composed E4.1 action is applied as its two E4
     steps. `game/describe.ts` writes the action log -- a line-for-line port of the old Python
     session, pinned by a golden (`tests/web/test_describe_golden.py`).
   - A click is described from the engine's settled state (`WasmEngine.settledDisplay`: after its
     action or last die roll), from cause to effect: the action (`describeAction`), a dice line for
     every roll it made (`clickRolls`, in order), what each Event did to VP and DEFCON, then the
     board's changes (`describeDeltas`, which also logs the Space Race track). When the click's Events moved VP or DEFCON in ways its before/after cannot show -- two
     headlines whose VP cancel, or more than one Event -- each gets a line of its own
     (`describeEventTracks`, from `clickEventTracks`; `ReplayStep.event_tracks` keeps them, and the
     replay log shows them too); a lone Event that is the whole change adds nothing.
     `tests/web/test_event_tracks.py` pins it on Nuclear Test Ban against Pershing II.
     When the click carried into the turn's end -- an action round that ends the turn stops on the
     turn's cleanup, resolved in the same click -- the turn's end (final VP, the military-ops reset,
     DEFCON, the next deal) is its own `TURN_END` entry, "End of turn N". Read from the end state,
     the turn's last realignment or coup used to lose its dice and wear the turn's end as its own.
   - **The log is shown one entry per play, not per click** (`game/log_view.ts`). The session's
     log stays per click -- undo, the game's record, the golden and the E2E tests read it -- and
     each click's entry carries what it is (`ActionLogItem.kind`, from `clickInfo`: setup, a
     headline commit, a card played, its mode, a point, any other decision) with the play's
     heading. `groupLog` groups clicks into plays -- a card played in an action round, the
     headline, one side's setup, every line the session writes itself (turn end, undo, debug) --
     and tells each once: the heading with its mode ("USSR plays Fidel (#8) — Ops: Influence",
     "Ops: Coup, then the US Event" for an opponent's card), the decisions within it (`▸`), dice
     and Events in order, the board's changes merged Event by Event and told where each Event's
     clicks end (a headline's Red Scare before the other headline's De-Stalinization, labelled
     "▸ De-Stalinization (#33) — USSR"), and where the cards went at the end, with card moves in words
     ("removed from the game", "now known to the USSR"), a deal as "US draws 7 cards: ..." and a
     reshuffle on one line. It runs on the log **after** `redactLogs`, so grouping only merges and
     rewords lines the view may read. A replay is grouped the same way (`replayItems` builds the
     items from its steps' snapshots, rolls and event records). `tests/web/test_log_view.py`
     holds the merged changes to the board's own over two turns and the USSR view to the redaction.
     A line that can be matched to one step shows its number (`LogEntry.steps`, the session's step
     index, which an exported replay keeps): a click's own lines, its dice and Events, and a merged
     change that one click made; a change merged over several clicks, and an Event's label, show
     none. An entry's heading shows the step that started it; in a replay a numbered line opens
     that step.
   - **A card named in the log shows its text on hover** (`game/card_refs.ts` `linkCardRefs`):
     every mention -- `Name (#id)`, the headline reveal's `'Name' (#id, n Ops)`, a card move's
     `Card #id (Name)`, and the bare names of a deal -- is wrapped in `span.log-card[data-card-id]`
     (by the engine's name or the rules'), and one delegated listener on `#action-log-stream`
     opens the hand tabs' tooltip (`CardsView.showCardTooltip`: Ops, side, era, text). Nothing
     else in the line changes, and the lines are the redacted log, so nothing hidden is named.
     The tooltip opens above the pointer where it would run off the bottom.
     `tests/web/test_card_refs.py` holds it to a two-turn game's log and to each form.
   - A card chosen while an event resolves (`resolving_card`) is said as what happened to it
     (`eventCardChoice`): discarded or removed, and from where ("USSR discards X from the US hand
     (Aldrich Ames Remix)"), taken back (SALT), played from the discard pile (Star Wars), played
     with UN Intervention, given away (Missile Envy) -- never as a play, nor, in the headline, as a
     headline commit with a "Headlines Simultaneously Revealed" pair, which is how the Python
     original logged them (the golden leaves those action lines out of its comparison). Five Year
     Plan's and Terrorism's random discards make no choice; each discarded card gets its own line
     ("✂ USSR discards Decolonization (#30) at random (Five Year Plan)"; for Five Year Plan, "— a US
     event, so its event fires" when it is one). Missile Envy's exchange is said in full (`⇄`
     lines): the card taken from whose hand and whether it resolves as its event or for its Ops,
     and that Missile Envy goes to the giver for its next round -- read from the engine marking
     Missile Envy the giver's forced card, not from a card reaching a hand, which a deal also does.
   - `refresh()` in `main.ts` is the one place a change of position lands: render, then (async,
     versioned so a stale result is dropped) the link token, the model's readout, and auto-play.
8. **Live model analysis (`analysis/`, `analysis_view.ts`)**:
   - Models are ONNX exports of checkpoints (`tools/export_onnx.py`) run by onnxruntime-web,
     single-threaded (GitHub Pages cannot send the cross-origin-isolation headers threads need),
     from three sources: **local** checkpoints (`/api/local/models`; the server exports on first
     use), a **Hugging Face** model repo (`owner/repo`, listed from the public tree API), or a
     dropped **file**. The file carries its description in ONNX metadata (`analysis/onnx_meta.ts`):
     a model whose observation width differs from the one the engine writes for the model's own
     `ts.obs_features` (`WasmEngine.obsSizeFor`), or that reads a feature bit this engine does not
     know, is refused; one exported next to another engine build is flagged.
   - **Every forward is built in the model's own view**: its action view (`ts.merged_influence`)
     and its observation features (`Model.obsFeatures` -> `engine.observation(player, features)`,
     the same `extract_observation_features` Python calls). `HEADLINE_REVEAL`, E8's default, has no
     width, so a base-layout observation handed to such a model would run and misread Space Race
     box 4's headline; `tests/web/test_wasm_engine.py` holds each feature set's view to Python's.
   - **The repo's catalogue (`hfCatalog`, `analysis/hf_catalog.ts`, re-exported by `model.ts`) is one
     request**: `models.json`
     at the repo root, written by `tools/publish_hf.py` on every publish (each `.onnx` with its
     commit date, sha256 and size, and the default model). A repo without one is listed by
     `listHfModels`: the tree *without* `expand` (1,000 entries a page, against 50 with it), then
     `paths-info` for the `.onnx` dates, and `default.json` for the default.
   - **The default model.** A link with no `model` loads the model `DEFAULT_HF_REPO`
     (`mihaild/deepstruggle@main`) names (`models.json`'s `default`, set by
     `tools/publish_hf.py default`; `default.json` in a repo without a manifest), or its newest
     `.onnx` when it names none. A default the repo does not hold is an error, not a fallback. The
     default is dropped if the user picks a model while it is listing. Turning analysis off
     writes `model=off`, so that link does not load the default.
   - **The repo mirrors `data/checkpoints/`** (`tools/publish_hf.py`): a run directory is
     published whole (an `.onnx` beside its final snapshot, final SWA and leaderboard
     networks, and beside every soup), so the picker groups files by directory
     (`groupHfModels`, one `<optgroup>` per run, groups ordered by their newest file).
     The E2E browsers launch with `HF_BLOCKED` (huggingface.co does not resolve); tests that need
     the repo serve it with `page.route`.
   - **Downloaded models are kept** (`analysis/model_cache.ts`), in the Cache API (`ts-models-v1`,
     the 8 newest), keyed by the file's LFS sha256, not its URL: `main` is mutable, so each load
     asks `paths-info` for the hash (one small request) and downloads only on a miss. Bytes are
     hashed before they are kept, so a file replaced mid-load is never stored under the old hash.
     The cache only saves a download, so when it is unavailable (an insecure origin, `paths-info`
     failing, quota) the model is downloaded as before. Local checkpoints are not cached in the
     page: the server already keeps their exports. The panel marks a cached load, and **Clear
     cache** empties it.
   - `analysis/readout.ts` is the port of the old server readout: one batched forward (the
     decider's observation with its real mask, and both sides' observations for the critic),
     softmax over the legal actions at temperature 1, the argmax as the favourite.
     `tests/web/test_e2e_workbench.py` checks it against `read_policy` / `read_critic` in Python.
   - **Which critic reading is believed.** The value head is trained only on the observation of
     the side to move, so of the two readings (`v_win_us` from the US observation, `v_win_ussr`
     from the USSR one) only the decider's is calibrated; the other is untrained extrapolation
     and can be far off. `trace_view.ts` `valueSidesHtml` shows, for each side, just its own
     P(victory) = (1 + v_win)/2 and expected final VP margin (`v_vp` × `VP_LIMIT` = 20), with the
     decider's card lit and the other dimmed -- no labels, bar or raw table. Live,
     the decider is `analysis.decision_player` (the readout is on the position itself); in a
     replay it is `replayDecider()`: the critic is read *after* the step's action, so the decider
     is the **next** step's `player` (the last step falls back to its snapshot's decision
     context). A terminal position has no decider -- its snapshot's stale decision context is
     ignored -- so both readings are dimmed and the result is shown below them. The replay's
     value ribbon and the log rows' `ΔP` chip plot the calibrated reading as P(US wins); the
     untrained one is only the ribbon's faint dashed line. `tests/web/test_value_readings.py`
     pins the rules on the TypeScript itself; the E2E tests check them on screen.
   - Every legal action's probability is painted on the card, HUD button or country that sends
     it, matched by the MicroAction each choice carries. For an E4.1 (merged-influence) model the
     composed placements go on the countries and the influence button carries their sum. The
     favourite is marked ★ (the replay's played move stays ◀).
   - *★ Play favourite* / key `F` / clicking a row in the panel plays a flat action in the model's
     own action view. **Auto-play** (none / USSR / US, `auto=` in the URL): whenever the chosen
     side is to move, `maybeAutoPlay()` plays the model's favourite after `AUTO_PLAY_DELAY_MS`,
     with the engine's own die, and drops the move if the position changed meanwhile. With
     auto-play on, *Cancel* keeps undoing until the decision is the other side's again.
     `ActionHud.onRerender` re-applies badges after the HUD redraws itself (the die selector).
   - **Search.** The section's search radios (`#mode-opponent`) choose the policy's argmax (`search=policy`) or the **rollout root**
     (`search=rollout:k:worlds:horizon:rule:stop:leaf[:keyed][:aligned][:box4][:beliefN]`, always written with every field --
     `analysis/rollout_spec.ts`, the grammar without the engine, which the leaderboard page reads too). Its spec is the
     default search -- the Hugging Face repo's `models.json` `default_search` (`tools/publish_hf.py default --player`),
     read once with the default model (`defaultHfCatalog`), else the built-in `DEFAULT_ROLLOUT`, the corrected keyed root
     `4:32:4:z2:earlier:own:keyed:aligned:box4` -- or one from the URL, or what the **search settings** (`#search-settings`,
     one field per spec field, a change the setting at once, Default back to the default search) say. A bare
     `search=rollout` from an older link is `DEFAULT_ROLLOUT`; a spelled-out spec turns on only the switches it names.
     The leaderboard's `play` links name each player's search (`player_spec.workbench_search`). The engine's own batch
     (`bindings/rollout_batch.hpp`, the same C++ as Python's searcher) driven by
     `analysis/rollout.ts` in a worker (`rollout_runner.ts`, inline where there are no workers).
     The spec takes the leaderboard's other modes too: `:keyed` (chance=keyed,
     `bindings/keyed_chance.hpp`) and `:belief[N]` (belief=N, 2,048 hands by default: the worlds dealt
     from the policy-likelihood belief over the opponent's hand, `bindings/belief.hpp`, the same C++
     as Python's). For the belief the session records every move (`GameSession.moves`, undone with
     undo) and `rollout_runner.ts` brings the searching engine's tracker up to them before each
     search -- new moves only, or reset and replayed after an undo, a loaded position or a new worker
     (the evidence about the opponent's style then restarts). The page builds each hypothesis row in
     full in WebAssembly and runs it through onnxruntime in chunks of 256: slow at 2,048 hands, but
     the leaderboard's player (`tests/web/test_wasm_engine.py` holds the tracker equal to Python's).
     The panel shows each candidate's mean value and paired lead and marks the pick; the favourite
     and auto-play take the pick and wait for it. A model in the merged (E4.1) view is refused for
     search.
   - **The address bar is the share link.** `syncUrl()` writes `pos` (the engine's save JSON,
     zlib, base64url -- `game/position.ts`, interchangeable with Python's zlib), `model`
     (`local:` / `hf:` source, or `off`; a dropped file has no address), `auto` and `view`, with
     `history.replaceState`, never `pushState`, so moves do not pile up in Back. A reload of the
     same link keeps the game's history; *New Game* starts afresh in the page (in the same view).
   - A generic `.hidden { display: none }` rule backs every mode-specific panel.
   - The header wraps rather than overflows: below ~1750px the tracks take a second row, so
     *New Game*, *Debug Tools*, *Replay Mode* and the view select stay on screen.
9. **Player views (`game/view.ts`, the header's view select, `view=us|ussr` in the URL)**:
   - *Developer view* shows everything. *Play as US / USSR* is for playing one side against the
     model without seeing the other side's private information. `main.ts` keeps the engine's full
     state in `state` and draws **every** panel from `shown = redactState(state, view)`; a panel
     must never read `state` directly, or it leaks.
   - What the other side hides: its cards whose location is `HAND_<opp>_UNKNOWN` (its `_KNOWN`
     cards stay named, as in the rules), the draw deck's contents, a card it peeks at
     (`PEEKED_TEMP` while it decides), and its headline until both are revealed (unless your Man
     in Space perk makes it reveal first). The hidden hand cards and the deck become one
     `UNSEEN` pile -- the *All* list says "Deck or <opp> hand", the deck count adds the hand --
     while the hand tab keeps its size as face-down cards (`hidden_cards`; hand sizes are public).
   - `redactLogs` drops the log's card-move lines *into* `HAND_<opp>_UNKNOWN` (the other side's
     deal, a card it draws) and its `PEEKED_TEMP` moves, and replaces its face-down headline
     commit with a line that does not name the card. A card *leaving* that hand -- played,
     discarded, removed, taken by an event -- is public and stays in the log. (A headline card
     stays in the hand until both are committed, `state_machine.cpp`, so its move is a reveal.)
   - **The other side's latest turn is marked on the map** (`game/last_turn.ts`, attached by
     `renderState` as `last_turn`): a dashed cyan ring on each country whose influence it changed
     or that it targeted (a failed coup changes nothing but matters), the signed change under each
     influence box, and a summary line at the bottom of the map (`#map-last-turn`). A turn is the
     latest run of `session.steps` sharing turn, phase, action round and owner -- the phasing
     player, so the other side's own event decisions inside it count; at setup, the side placing.
     Choosing a headline card is not a turn; a headline event is its owner's (the engine makes
     the resolving event's owner the phasing player). The developer's view marks nothing.
   - While the other side decides, the state carries `view_hidden_decision` and no legal actions:
     the HUD says who is to move, auto-play is locked to the other side, and `sendAction` /
     `sendFlatAction` refuse clicks for it (only auto-play moves for it).
   - **Show analysis** (`#analysis-visible-toggle` in the technical bar, `setReadoutShown`;
     `AnalysisPanel.readoutShown`): shows or hides the model's readout -- the critic's win
     chances, the policy, the search -- its probability badges on the board, and *★ Play
     favourite* / `F`. A change of view sets it to that view's default, on for Developer and off
     for a player (they would be hints), and the toggle overrides it until the view changes
     again. Whatever it says, a player's view never shows the readout of the other side's
     decisions (`AnalysisPanel.hiddenSide`: its policy would read the hidden hand off the
     probabilities). With the readout off, the panel says when the model is choosing, whether its
     search is running, and -- since auto-play then falls back to the policy -- that it failed.
     The model still reads every position: auto-play plays its favourite for its side.
   - **What is searched.** With the rollout root on, a decision is searched only when auto-play
     plays it or the readout shows it (`searchWanted`). A change of view, auto side, Show analysis
     or search setting never re-reads the position: `ensureSearch` searches the readout on screen
     if it now needs a search and has none, keeping one already made or running.
     `RolloutRunner` runs one search at a time, the worker one message at a time (they share one
     engine and one rollout batch), and a request still waiting when a newer one arrives is
     superseded without running.
     Inline fallback owns a separate, lazily loaded engine too, so user moves, undo and New Game
     during inference cannot be overwritten by a snapshot restore. Each worker loads one model
     per lifetime: replacement terminates it, rejects initialization and active requests, and
     invalidates queued searches. Late messages from an obsolete worker are ignored; worker
     errors also settle initialization so the search queue can continue. Invalid imported save
     context depths are refused before the live position changes.
   - **How the model plays a side.** The search radios (`#mode-opponent`, in the model & search
     section): the rollout root, the strongest player, or the policy, set through `setSearch`.
     The rollout option describes the spec it would keep (the one in force, else the measured
     one). In `#model-mode-dialog` a change is the setting at once. In the play-mode question
     (`#mode-dialog`, `askMode`) it waits for the choice of mode: a fresh open offers the rollout
     root unless the link named `search=`; New Game offers the setting in force. *Play as US /
     USSR* sets it, without a search of its own (the deal or the view reads the position once);
     *Developer* leaves it alone unless it was changed in the question (`searchChoiceTouched`),
     and cancelling sets nothing. Once either dialog closes the radios say the setting in force
     again. While either is open the page's keyboard shortcuts are off, so its Escape does not
     also take back a move.
   - Debug Tools and Replay Mode are developer-only; loading or dropping a replay switches to the
     developer view. Limitation: the `pos=` link token is the full save, so a player view hides
     the other side from the page, not from someone who decodes the link.
10. **Bundled metadata (`metadata.ts`)**:
   - `rules/map.json` and `rules/cards.json` are imported at build time, so the page needs no
     server to draw the board. The flat action layout comes from the engine
     (`WasmEngine.layout`), never from a hand-kept copy.
11. **Games against people: the record and the stats database (`game/record.ts`,
    `result_dialog.ts`, `game/saved_game.ts`, `web/stats_worker/`)**:
   - **The record.** `GameSession` tags every step (`tags`, aligned with `steps` and cut back with
     them on undo): who made it (`human` click, `suggested` -- the model's move played from the
     readout, F or a panel row -- or `auto`-play), whether the model's readout was on screen for
     the mover (`showsReadoutFor`), the mover, the forced die and whether the step rolled, and an
     index into `settings` -- the view, auto side, model (its file's sha256) and search in force,
     noted by `TSApp.noteSettings()` just before each move. Settings travel per step, not as a
     timeline, because undo would make a timeline name the wrong setting for steps played again.
     `events` keeps take-backs, debug overrides, a loaded position and reloads. Nothing is
     refused for being recorded; the reader filters.
   - **The log** (`encodeLog`): seed, then 7 bytes a step (MicroAction, forced die, tags), zlib +
     base64url -- about 2 KB a game where the replay is megabytes. `tools/lib/game_stats.py`
     decodes the same bytes, re-drives the game in the Python engine (`init_game` + `drain_chance`
     with each step's forced die) and checks `final_hash`, the sha256 of the save JSON at the end.
     `summarize()` (clean = one model and search on one side, the human the other in a player
     view, no moves for the model, no hints, no forced dice, no debug, not from a loaded position;
     take-backs counted, not disqualifying) has a Python twin that must agree --
     `tests/web/test_game_stats.py` holds the two to it.
   - **Game over** opens `#result-dialog` once per ending (`offerResult`, from `refresh`), and the
     *Result* button again: the result, the summary's notes in words, and the offer to add the
     game with a Playdek login and rating (remembered in localStorage). The submission
     (`buildSubmission`, record.ts `Submission`) goes to `${TS_STATS_ENDPOINT}/games` -- the
     Cloudflare Worker in `web/stats_worker/`, keeping it in D1. A build without the endpoint
     offers *Download record* only. While the dialog is open the keyboard shortcuts are off.
   - **Reloads keep the game.** After every move the record is stored in localStorage
     (`saved_game.ts`). On boot, a `pos=` link whose position the stored game plays to is that
     game, played back from its seed (`replayRecord`) -- id, history and undo intact -- rather
     than a new game from a loaded position, which could not be re-driven from the seed.
   - **The forced die is for one roll.** The HUD's die selector goes back to Auto once a move of
     yours has rolled with it (`afterOwnMove`): one click on a face no longer forces every roll
     after it. Auto-play always rolls.
   - **The build.** `vite.config.ts` writes the commit (`git rev-parse HEAD`) and whether tracked
     files differed from it (`dirty`) into `build_info.ts`, so a submitted game names what to
     rebuild it at; the engine fingerprint comes from the engine itself.

---

## 4. Development & Build Commands

```bash
tools/scripts/build_web.sh      # from the repo root: the WebAssembly engine, then npm run build
npm install                     # install dependencies
npm run dev                     # Vite dev server, proxying /api to the local server on port 8000
npm run build                   # production bundle into dist/ (needs public/engine/ built first)
```

---

## 5. Verification & Regression Testing

`public/engine/` and `dist/` are not committed, so build them before running the web suite
(`tools/scripts/build_web.sh`). Invoke pytest as a module, never via `.venv/bin/pytest`: that
console script carries an absolute shebang and breaks if the venv is moved. The browser tests need
a Playwright Chromium (`.venv/bin/python -m playwright install chromium`).

```bash
tools/scripts/build_web.sh
PYTHONPATH=.:build/release .venv/bin/python -m pytest -q tests/web
```

| test | holds |
|:---|:---|
| `test_wasm_engine.py` | the WebAssembly engine is the native one: 200 whole games, and display JSON / save JSON / observations / masks at every step of games driven by clicks |
| `test_e2e_action_log.py` | in Chromium: the turn's last realignment keeps its dice and the turn's end is its own entry; a click that rolls twice (a realignment with the opponent's war card) logs both dice; the card Aldrich Ames has the US discard in the headline is a discard, not a headline; Five Year Plan's and Terrorism's random discards are their own lines, with whether Five Year Plan's event fires; Missile Envy's exchange says the card taken and how it resolves; Chernobyl's options are region names in the HUD and the log |
| `test_describe_golden.py` | the TypeScript action log writes what the Python session wrote (1,098 steps); after an engine change refuses a recorded action, re-record with `fixtures/make_describe_golden.py` |
| `test_e2e_board_display.py` | in Chromium: this turn's effects on top of the map (not permanent ones, not Space Race attempts), control as the country's fill (a highlighted target included), an empty influence space blank with the US marker left and the controlling one flipped, the hover card's control arithmetic and neighbour highlight, the `NO COUPS` chip on the regions DEFCON closes, and the roll tray and the rolled country's die (no roll from an earlier turn; a Space Race roll marks no country), a redraw putting a tapped country's card away, a drag whose release was outside the window ending, the space perks named as their boxes |
| `test_e2e_mobile.py` | in Chromium at 390px with touch: one column with the map first and nothing wider than the screen, the region buttons clear of the zoom buttons and framing their region, a tap showing a non-target country's card and a tap on open map closing it, a badge opening the region's scoring summary with the right rows, and the space pills' next Ops and roll |
| `test_position_tokens.py` | a `pos=` token means the same position to the page and to Python's zlib |
| `test_local_server.py` | the local server lists and exports checkpoints, serves replays, and nothing outside its trees |
| `test_e2e_workbench.py` | in Chromium: a game played by the page, the model readout against Python's `read_policy`/`read_critic`, the live critic lighting the side to move, country probabilities clear of the influence, the cards column left of the map (and stacked when narrow), the Piles tab packed by side then Ops, a log card's tooltip on hover, auto-play + undo, links, a dropped `.onnx`, debug overrides, replay export, and the page on a static server with no API (GitHub Pages) |
| `test_e2e_player_view.py` | in Chromium, a player's view: the other hand and the deck are one unseen pile on every tab, the other side's deal and face-down headline are not logged (the cards leaving its hand are), its latest turn's countries are marked on the map, its decision is hidden and clicks for it refused, *New Game* and the view select fit a 1366px screen; the play-mode question holds the model & search section folded and sets the search for a side (rollout root by default) and leaves it alone for Developer unless changed there, the model · mode dialog sets it at once and its Escape takes back no move, describes the spec it would keep, and its Escape takes back no move; Show analysis follows the view and the toggle; with a model, the player's own decisions show no readout, no badges and no favourite and are not searched until Show analysis is on, a change of view keeps the search already made, the question starts one search at a time (the runner supersedes a waiting request), a failed search is reported in a player's view, and a change of view drops a move armed for the old auto side |
| `test_e2e_replay_trace.py`, `test_e2e_space_race.py` | replay trace views (the readout lights the next step's player); the header tracks and the Space Race widget |
| `test_card_refs.py` | `game/card_refs.ts` under node: on a two-turn game's grouped log and on one line per form of mention, every card named by number is wrapped (with its name where the name precedes it), a deal's names too, a Space Race box is not, and the line is otherwise unchanged |
| `test_recent_rolls.py` | `game/rolls.ts`, `roll_tray.ts` and the log's dice lines (`game/describe.ts`) under node: the turn's last coup shown in the next headline and gone a half round later, every die of one click, a replay file's snapshot roll, the Olympic sponsor's die without its +2 (tray and log) |
| `test_value_readings.py` | `trace_view.ts` under node: the decider of each replay position, the calibrated P(US wins), VP ×20, the terminal case, the `ΔP` chip |
| `test_web_workbench.py` | the bundled rules metadata, the page's DOM, replay snapshots |
| `test_game_stats.py` | `game/record.ts` and the stats Worker under node: two games played in the page's `GameSession` (a clean one, one with every mark) re-drive in the Python engine to the submitted final position, the page's summary is Python's, the log is under 7 bytes a step, a stored record plays back as a reload does, and the Worker keeps a game once, refuses another origin, a game without a login and the fourth an hour |
| `test_e2e_game_stats.py` | in Chromium: a finished game opens the result dialog once with its notes (download only without an endpoint), Escape closes it without a take-back, the Result button reopens it; a forced die goes back to Auto once it has rolled; a reload plays the stored game back (same id, steps and undo) |

The TypeScript under test in node (`tests/web/js/*.ts`) is bundled with the esbuild that ships
with Vite (`tests/web/js_runner.py`).
