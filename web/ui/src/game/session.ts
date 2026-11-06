/**
 * One game, in the page: stepping, dice, undo, the action log, debug overrides, loading a shared
 * position, a replay of the game so far for export, and its record for the stats database
 * (game/record.ts): every step tagged with who made it and the settings in force.
 *
 * The port of web/server/session.py's GameSession, which ran on the server until the workbench
 * moved into the browser. Two things it now does that the server did not: undo restores the log
 * and the replay to where they were (the server popped one log line, so undoing a game-ending
 * move left its line behind the removed "Game Over"), and debug overrides can be undone.
 */
import { ActionLogItem, GameState, MicroAction } from "../types";
import { WasmEngine } from "../engine/wasm_engine";
import { describeAction, describeDeltas, describeEventTracks, describeRoll } from "./describe";
import { clickInfo } from "./log_view";
import { ReplayStep } from "../replay_controls";
import { DebugOverride, GameRecordData, ModelInfo, MoveHow, RecordEvent, RecordEventKind, Settings, StepTag } from "./record";

const MAX_UNDO = 300;

interface UndoEntry {
  snapshot: Uint8Array;
  logs: number;
  steps: number;
  stepIndex: number;
  moves: number;
  /** A debug override's entry: its event in `events`, marked undone when this is taken back. */
  event?: number;
}

/** One move of the game as played: the position before it (raw state) and the click. The belief over
 *  the opponent's hand (bindings/belief.hpp) reads these -- the record both players saw. */
export interface Move {
  before: Uint8Array;
  action: MicroAction;
}

export interface GameResult {
  winner: string;
  margin: number;
  end_turn: number;
  reason: string;
}

const HUMAN: MoveHow = { source: "human", analysis: false };

function newGameId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `g${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

function sameSettings(a: Settings, b: Settings): boolean {
  return a.view === b.view && a.auto === b.auto && a.model === b.model && a.search === b.search;
}

export class GameSession {
  /** This game's id in the stats database; a new one per deal or loaded position. */
  id = newGameId();
  startedAt = new Date().toISOString();
  /** The game's record was added to the stats database. */
  submitted = false;
  seed = 0;
  stepIndex = 0;
  logs: ActionLogItem[] = [];
  steps: ReplayStep[] = [];
  result: GameResult | null = null;
  /** Set when the game began from a loaded position (a link) rather than from the seed. */
  startPosition: string | null = null;
  /** The display state before the first of `steps` (the "before" of step 0). */
  start: GameState | null = null;
  /** Per step of `steps`: who made it and how (game/record.ts). */
  tags: StepTag[] = [];
  /** Every distinct setting a step was made under; a tag names one by index. */
  settings: Settings[] = [];
  models: Record<string, ModelInfo> = {};
  events: RecordEvent[] = [];
  private setting = 0;
  /** Every move since the game began (or its position was loaded); undo takes them back. */
  moves: Move[] = [];
  private history: UndoEntry[] = [];

  constructor(readonly engine: WasmEngine) {}

  newGame(seed: number): void {
    this.engine.newGame(seed);
    this.seed = seed;
    this.stepIndex = 0;
    this.history = [];
    this.moves = [];
    this.steps = [];
    this.result = null;
    this.startPosition = null;
    this.resetRecord();
    const s = this.engine.display();
    this.start = s;
    this.logs = [{
      step_index: 0, turn: s.turn, ar: s.action_round, phase: "SETUP", player: "SYSTEM",
      text: `Game started (Seed: ${seed}). USSR setup: Place 6 Influence in Eastern Europe.`,
    }];
  }

  /** The display state plus what the page shows around it. */
  state(): GameState {
    const s = this.engine.display();
    s.seed = this.seed;
    s.step_index = this.stepIndex;
    s.action_logs = this.logs;
    s.can_undo = this.history.length > 0;
    // Muslim Revolution with US Influence in one of its eight countries takes that one and then
    // offers only CONFIRM_DONE: the removal is mandatory, so there is no early stop to say so.
    s.legal_actions.can_confirm_done = !!this.engine.mask(false)[this.engine.layout.confirm_done_index];
    s.players = { US: "US", USSR: "USSR" };
    return s;
  }

  /** A new game's record: a fresh id, no steps, the settings in force kept for its first move. */
  private resetRecord(): void {
    const current = this.settings[this.setting];
    this.id = newGameId();
    this.startedAt = new Date().toISOString();
    this.submitted = false;
    this.tags = [];
    this.events = [];
    this.settings = current ? [current] : [];
    this.setting = 0;
    const model = current?.model;
    this.models = model && this.models[model] ? { [model]: this.models[model] } : {};
  }

  /** The settings the page moves under from now on: the view, auto side, model and search. */
  setSettings(s: Settings, model?: ModelInfo): void {
    if (model && s.model) this.models[s.model] = model;
    let i = this.settings.findIndex(x => sameSettings(x, s));
    if (i < 0) {
      if (this.settings.length >= 255) return;   // a byte per step names it; past that, the last one stays
      i = this.settings.push({ ...s }) - 1;
    }
    this.setting = i;
  }

  private event(kind: RecordEventKind, text: string): void {
    this.events.push({ step: this.steps.length, kind, text });
  }

  /** The record of the game so far (game/record.ts). */
  record(): GameRecordData {
    return {
      id: this.id,
      started_at: this.startedAt,
      seed: this.seed,
      start_position: this.startPosition,
      steps: this.steps.map((s, i) => ({ action: s.action as Required<MicroAction>, tag: this.tags[i] })),
      settings: this.settings,
      models: this.models,
      events: this.events,
    };
  }

  /**
   * Play a saved record over again (a page reload): the same deal or position, then every step
   * with its forced die and tags. Its id, settings and events come back with it. Throws, leaving a
   * half-played game, if the engine refuses a step -- the caller then starts afresh.
   */
  replayRecord(rec: GameRecordData, saveJson: string | null, submitted: boolean): void {
    this.newGame(rec.seed);
    if (rec.start_position) {
      if (!saveJson) throw new Error("the record starts from a position that was not given");
      this.loadPosition(saveJson, rec.start_position);
    }
    this.settings = rec.settings.map(s => ({ ...s }));
    this.models = { ...rec.models };
    this.events = rec.events.map(e => ({ ...e }));
    // The debug overrides that stood, each where it was made: before the step it preceded.
    const overrides = this.events.flatMap((e, i) => (e.kind === "debug" && e.override && !e.undone ? [i] : []));
    const overridesAt = (step: number) => {
      for (const i of overrides) {
        const e = this.events[i];
        if (e.step !== step) continue;
        const why = this.override(e.text, e.override!, i);
        if (why) throw new Error(`a debug override of the saved game was refused: ${why}`);
      }
    };
    rec.steps.forEach((s, i) => {
      overridesAt(i);
      this.setting = s.tag.setting;
      const why = this.apply(s.action, s.tag.forcedDie, { source: s.tag.source, analysis: s.tag.analysis });
      if (why) throw new Error(`step ${i + 1} of the saved game was refused: ${why}`);
    });
    overridesAt(rec.steps.length);
    this.id = rec.id;
    this.startedAt = rec.started_at;
    this.submitted = submitted;
    this.event("resume", "page reloaded");
  }

  get canUndo(): boolean {
    return this.history.length > 0;
  }

  private remember(): void {
    this.history.push({
      snapshot: this.engine.snapshot(), logs: this.logs.length, steps: this.steps.length, stepIndex: this.stepIndex,
      moves: this.moves.length,
    });
    if (this.history.length > MAX_UNDO) this.history.shift();
  }

  /**
   * Apply one MicroAction (a click) and resolve its dice. Returns null, or why the engine
   * refused -- in which case nothing changed. `how` is who made the move, for the record.
   */
  apply(action: MicroAction, forcedDie = 0, how: MoveHow = HUMAN): string | null {
    if (this.engine.isTerminal()) return "the game is over";
    const a: Required<MicroAction> = {
      decision_type: action.decision_type, primary_id: action.primary_id,
      secondary_id: action.secondary_id ?? 0, flags: action.flags ?? 0,
    };
    const before = this.engine.display();
    const snapshot = this.engine.snapshot();
    const refused = this.engine.step(a, forcedDie);
    if (refused) return refused;
    // Only now is there something to undo.
    this.history.push({ snapshot, logs: this.logs.length, steps: this.steps.length, stepIndex: this.stepIndex,
                        moves: this.moves.length });
    if (this.history.length > MAX_UNDO) this.history.shift();
    this.moves.push({ before: snapshot, action: a });

    this.stepIndex += 1;
    const after = this.engine.display();
    // What the click itself did is the engine's settled state: after the action, or after its last
    // die roll. When an action round ends the turn, the engine stops on the turn's cleanup, and
    // resolving it -- final VP, the military-ops reset, DEFCON, the next deal -- clears the roll
    // record: read from the end state, the turn's last realignment or coup lost its dice, and the
    // turn's end read as part of it. So the click is described up to the settled state, and the
    // turn's end, when the click carried into it, gets its own entry.
    const settled = this.engine.settledDisplay();
    const crossed = settled.turn !== after.turn || settled.current_phase_name !== after.current_phase_name;
    const end = crossed ? settled : after;
    // From cause to effect: the action, every die the click rolled (in order, not only the last
    // one a state keeps), what each of its Events did to VP and DEFCON, then the board's changes.
    const mover = before.decision_context?.decision_player ?? "NONE";
    const rolls = this.engine.clickRolls();
    const eventTracks = this.engine.clickEventTracks();
    const lines = describeAction(this.engine, before, end, a);
    for (const r of rolls) lines.push(...describeRoll(this.engine, before, r, mover));
    lines.push(...describeEventTracks(before, end, eventTracks));
    lines.push(...describeDeltas(this.engine, before, { ...end, die_roll: undefined }, mover,
                                 a.decision_type === this.engine.layout.decision_types.SELECT_CARD ? a.primary_id : 0));
    this.tags.push({
      source: how.source, analysis: how.analysis, mover: mover === "US" || mover === "USSR" ? mover : "",
      forcedDie, rolled: rolls.length > 0, setting: this.setting,
    });
    const main = lines[0] ?? `Action type=${a.decision_type}`;
    const phase = before.current_phase_name ?? "ACTION";
    const ar = phase === "HEADLINE" || phase === "SETUP" ? 0 : before.action_round;
    const player = before.decision_context?.decision_player ?? "NONE";
    this.logs.push({
      step_index: this.stepIndex, turn: before.turn, ar, phase, player, text: main,
      details: lines.slice(1), vp_delta: (end.victory_points ?? 0) - (before.victory_points ?? 0),
      ...clickInfo(this.engine, before, a),
    });
    if (crossed) {
      const rest = describeDeltas(this.engine, settled, after, "SYSTEM");
      const text = after.turn !== settled.turn || after.current_phase_name === "GAME_OVER"
        ? `End of turn ${settled.turn}` : `${settled.current_phase_name ?? ""} ends`;
      this.logs.push({
        step_index: this.stepIndex, turn: settled.turn, ar: settled.action_round, phase: "TURN_END", player: "SYSTEM",
        text, details: rest, vp_delta: (after.victory_points ?? 0) - (settled.victory_points ?? 0),
      });
    }
    this.steps.push({
      step_index: this.stepIndex, turn: before.turn, ar, phase, player,
      action: { decision_type: a.decision_type, primary_id: a.primary_id, secondary_id: a.secondary_id, flags: a.flags },
      description: main, state_snapshot: after, rolls, event_tracks: eventTracks,
    });

    if (this.engine.isTerminal()) {
      const util = this.engine.terminalUtility();
      const winner = util > 0 ? "US" : util < 0 ? "USSR" : "DRAW";
      this.result = { winner, margin: after.victory_points, end_turn: after.turn, reason: this.engine.endingReason() };
      this.logs.push({
        step_index: this.stepIndex + 1, turn: after.turn, ar: after.action_round, phase: "GAME_OVER", player: "SYSTEM",
        text: `Game Over! Winner: ${winner} (VP: ${after.victory_points}, Turn: ${after.turn}) -- ${this.result.reason}`,
      });
    }
    return null;
  }

  /**
   * Apply a flat action in the given action view -- a model's move. A composed E4.1 action
   * ("Ops for influence, first point at X") is applied as the two E4 steps it is defined as,
   * exactly as Engine::step_flat composes it, each logged and undoable like a click; if the
   * second is refused the first is taken back too.
   */
  applyFlat(idx: number, merged: boolean, forcedDie = 0, how: MoveHow = HUMAN): string | null {
    const mask = this.engine.mask(merged);
    if (!(idx >= 0 && idx < mask.length && mask[idx])) {
      return `flat action ${idx} is not legal here (${merged ? "E4.1" : "E4"} view)`;
    }
    if (merged && this.engine.isMergedInfluenceAction(idx)) {
      const L = this.engine.layout;
      const first = this.engine.decodeFlat(L.ops_influence_index);
      const refused = this.apply(first, 0, how);
      if (refused) return refused;
      if (this.engine.isTerminal()) return null;
      const secondSlot = idx === L.ops_influence_index ? L.confirm_done_index : idx;
      const refused2 = this.apply(this.engine.decodeFlat(secondSlot), forcedDie, how);
      if (refused2) {
        this.undo(false);
        return refused2;
      }
      return null;
    }
    return this.apply(this.engine.decodeFlat(idx), forcedDie, how);
  }

  /** Take back the last step. `announce` adds the "cancelled" line to the log. */
  undo(announce = true): boolean {
    const e = this.history.pop();
    if (!e) return false;
    // A debug override has no step of its own: taking one back is not a move taken back, but a
    // rebuild of the game must not apply it.
    if (e.event !== undefined && this.events[e.event]) this.events[e.event].undone = true;
    if (announce && e.steps < this.steps.length) {
      const t = this.tags[e.steps];
      this.event("undo", `${t?.mover || "?"} ${t?.source ?? "?"} move taken back`);
    }
    this.engine.restore(e.snapshot);
    this.logs.length = e.logs;
    this.steps.length = e.steps;
    this.tags.length = e.steps;
    this.stepIndex = e.stepIndex;
    this.moves.length = e.moves;
    this.result = null;
    if (announce) {
      const s = this.engine.display();
      this.logs.push({
        step_index: this.stepIndex, turn: s.turn, ar: s.action_round, phase: s.current_phase_name ?? "ACTION",
        player: "SYSTEM", text: `↺ Action cancelled (Rolled back to Step ${this.stepIndex})`,
      });
    }
    return true;
  }

  // ---- debug overrides: engine testing, undoable, recorded -----------------------------------
  //
  // An override changes the position outside the rules, so the steps alone no longer rebuild the
  // game: each is recorded as a "debug" event carrying the exact change (record.ts DebugOverride),
  // which replayRecord applies again where it was made. Its summary is never `clean`.

  /** Apply `o` as one undoable change, logged; `event` is the record's event for it. */
  private override(text: string, o: DebugOverride, event: number): string | null {
    this.remember();
    const refused = o.kind === "country" ? this.engine.setCountry(o.id, o.us, o.ussr)
      : o.kind === "defcon" ? this.engine.setDefcon(o.value) : this.engine.setVp(o.value);
    if (refused) {
      this.history.pop();
      return refused;
    }
    this.history[this.history.length - 1].event = event;
    const s = this.engine.display();
    this.logs.push({ step_index: this.stepIndex, turn: s.turn, ar: s.action_round, phase: "DEBUG", player: "DEBUG", text });
    return null;
  }

  private debug(text: string, o: DebugOverride): string | null {
    const refused = this.override(text, o, this.events.length);
    if (!refused) this.events.push({ step: this.steps.length, kind: "debug", text, override: o });
    return refused;
  }

  setCountry(cid: number, us: number, ussr: number): string | null {
    return this.debug(`[DEBUG] Set ${this.engine.countryName(cid)} Influence -> US: ${us}, USSR: ${ussr}`,
      { kind: "country", id: cid, us, ussr });
  }
  setDefcon(v: number): string | null {
    return this.debug(`[DEBUG] Set DEFCON -> ${v}`, { kind: "defcon", value: v });
  }
  setVp(v: number): string | null {
    return this.debug(`[DEBUG] Set VP -> ${v}`, { kind: "vp", value: v });
  }

  // ---- positions -------------------------------------------------------------------------------

  /**
   * Put a saved position on the board (a shared link). False when it is already the position on
   * the board -- a page reload -- so the game keeps its history. Throws for a save the engine
   * refuses.
   */
  loadPosition(saveJson: string, token: string): boolean {
    if (saveJson === this.engine.saveJson()) return false;
    this.engine.loadSaveJson(saveJson);
    this.history = [];
    this.moves = [];
    this.steps = [];
    this.stepIndex = 0;
    this.result = null;
    this.resetRecord();
    this.startPosition = token;
    this.event("load", "position loaded from a link");
    const s = this.engine.display();
    this.start = s;
    this.logs = [{
      step_index: 0, turn: s.turn, ar: s.action_round, phase: s.current_phase_name ?? "", player: "SYSTEM",
      text: "Position loaded from a shared link (undo history starts here).",
    }];
    return true;
  }

  /** The game so far as a .tslog.json replay, for watching later or attaching to a bug. */
  replay(): object {
    const metadata: Record<string, unknown> = {
      game_id: "workbench",
      created_at: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
      seed: this.seed,
      players: { US: "US", USSR: "USSR" },
      total_steps: this.steps.length,
      result: this.result,
      engine_fingerprint: this.engine.fingerprint,
    };
    if (this.startPosition) metadata.start_position = this.startPosition;
    return { version: "1.0", metadata, initial_state: { seed: this.seed }, steps: this.steps };
  }
}
