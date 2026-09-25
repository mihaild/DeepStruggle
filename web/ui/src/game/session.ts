/**
 * One game, in the page: stepping, dice, undo, the action log, debug overrides, loading a shared
 * position, and a replay of the game so far for export.
 *
 * The port of web/server/session.py's GameSession, which ran on the server until the workbench
 * moved into the browser. Two things it now does that the server did not: undo restores the log
 * and the replay to where they were (the server popped one log line, so undoing a game-ending
 * move left its line behind the removed "Game Over"), and debug overrides can be undone.
 */
import { ActionLogItem, GameState, MicroAction } from "../types";
import { WasmEngine } from "../engine/wasm_engine";
import { describeActionAndDeltas } from "./describe";
import { ReplayStep } from "../replay_controls";

const MAX_UNDO = 300;

interface UndoEntry {
  snapshot: Uint8Array;
  logs: number;
  steps: number;
  stepIndex: number;
}

export interface GameResult {
  winner: string;
  margin: number;
  end_turn: number;
  reason: string;
}

export class GameSession {
  seed = 0;
  stepIndex = 0;
  logs: ActionLogItem[] = [];
  steps: ReplayStep[] = [];
  result: GameResult | null = null;
  /** Set when the game began from a loaded position (a link) rather than from the seed. */
  startPosition: string | null = null;
  private history: UndoEntry[] = [];

  constructor(readonly engine: WasmEngine) {}

  newGame(seed: number): void {
    this.engine.newGame(seed);
    this.seed = seed;
    this.stepIndex = 0;
    this.history = [];
    this.steps = [];
    this.result = null;
    this.startPosition = null;
    const s = this.engine.display();
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
    s.players = { US: "US", USSR: "USSR" };
    return s;
  }

  get canUndo(): boolean {
    return this.history.length > 0;
  }

  private remember(): void {
    this.history.push({
      snapshot: this.engine.snapshot(), logs: this.logs.length, steps: this.steps.length, stepIndex: this.stepIndex,
    });
    if (this.history.length > MAX_UNDO) this.history.shift();
  }

  /**
   * Apply one MicroAction (a click) and resolve its dice. Returns null, or why the engine
   * refused -- in which case nothing changed.
   */
  apply(action: MicroAction, forcedDie = 0): string | null {
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
    this.history.push({ snapshot, logs: this.logs.length, steps: this.steps.length, stepIndex: this.stepIndex });
    if (this.history.length > MAX_UNDO) this.history.shift();

    this.stepIndex += 1;
    const after = this.engine.display();
    const lines = describeActionAndDeltas(this.engine, before, after, a);
    const main = lines[0] ?? `Action type=${a.decision_type}`;
    const phase = before.current_phase_name ?? "ACTION";
    const ar = phase === "HEADLINE" || phase === "SETUP" ? 0 : before.action_round;
    const player = before.decision_context?.decision_player ?? "NONE";
    this.logs.push({
      step_index: this.stepIndex, turn: before.turn, ar, phase, player, text: main,
      details: lines.slice(1), vp_delta: (after.victory_points ?? 0) - (before.victory_points ?? 0),
    });
    this.steps.push({
      step_index: this.stepIndex, turn: before.turn, ar, phase, player,
      action: { decision_type: a.decision_type, primary_id: a.primary_id, secondary_id: a.secondary_id, flags: a.flags },
      description: main, state_snapshot: after,
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
  applyFlat(idx: number, merged: boolean, forcedDie = 0): string | null {
    const mask = this.engine.mask(merged);
    if (!(idx >= 0 && idx < mask.length && mask[idx])) {
      return `flat action ${idx} is not legal here (${merged ? "E4.1" : "E4"} view)`;
    }
    if (merged && this.engine.isMergedInfluenceAction(idx)) {
      const L = this.engine.layout;
      const first = this.engine.decodeFlat(L.ops_influence_index);
      const refused = this.apply(first, 0);
      if (refused) return refused;
      if (this.engine.isTerminal()) return null;
      const secondSlot = idx === L.ops_influence_index ? L.confirm_done_index : idx;
      const refused2 = this.apply(this.engine.decodeFlat(secondSlot), forcedDie);
      if (refused2) {
        this.undo(false);
        return refused2;
      }
      return null;
    }
    return this.apply(this.engine.decodeFlat(idx), forcedDie);
  }

  /** Take back the last step. `announce` adds the "cancelled" line to the log. */
  undo(announce = true): boolean {
    const e = this.history.pop();
    if (!e) return false;
    this.engine.restore(e.snapshot);
    this.logs.length = e.logs;
    this.steps.length = e.steps;
    this.stepIndex = e.stepIndex;
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

  // ---- debug overrides: engine testing, undoable ---------------------------------------------

  private debug(text: string, change: () => string | null): string | null {
    this.remember();
    const refused = change();
    if (refused) {
      this.history.pop();
      return refused;
    }
    const s = this.engine.display();
    this.logs.push({ step_index: this.stepIndex, turn: s.turn, ar: s.action_round, phase: "DEBUG", player: "DEBUG", text });
    return null;
  }

  setCountry(cid: number, us: number, ussr: number): string | null {
    return this.debug(`[DEBUG] Set ${this.engine.countryName(cid)} Influence -> US: ${us}, USSR: ${ussr}`,
      () => this.engine.setCountry(cid, us, ussr));
  }
  setDefcon(v: number): string | null {
    return this.debug(`[DEBUG] Set DEFCON -> ${v}`, () => this.engine.setDefcon(v));
  }
  setVp(v: number): string | null {
    return this.debug(`[DEBUG] Set VP -> ${v}`, () => this.engine.setVp(v));
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
    this.steps = [];
    this.stepIndex = 0;
    this.result = null;
    this.startPosition = token;
    const s = this.engine.display();
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
