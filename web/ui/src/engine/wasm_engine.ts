/**
 * The engine, in the page: a typed wrapper around the WebAssembly build of engine/ + bindings/
 * (bindings/wasm/ts_engine_wasm.cpp, built by tools/scripts/build_web.sh into public/engine/).
 *
 * The same C++ as the Python stack -- rules, observation, display state, saves -- so the page
 * cannot show or play a position differently from Python; tests/web/test_wasm_engine.py checks
 * the two builds bit for bit. Everything returned is a copy: the module's memory can move when it
 * grows, and a view into it would silently change under the caller.
 */
import { GameState, MicroAction } from "../types";

/** The flat action space's layout, from the engine itself (never a hand-kept copy). */
export interface ActionLayout {
  size: number;
  offsets: {
    card: number; play_mode: number; op_mode: number; roll_die: number;
    node: number; branch: number; defcon_value: number; region: number;
  };
  confirm_done_index: number;
  ops_influence_index: number;
  decision_types: Record<string, number>;
}

export const PLAYER_US = 1;
export const PLAYER_USSR = -1;

/* eslint-disable @typescript-eslint/no-explicit-any */
type Module = any;

export class WasmEngine {
  readonly fingerprint: string;
  readonly obsSize: number;
  readonly actionSize: number;
  readonly layout: ActionLayout;
  private readonly stateSize: number;

  private constructor(private readonly M: Module) {
    this.fingerprint = this.str(M._ts_fingerprint());
    this.obsSize = M._ts_obs_size();
    this.actionSize = M._ts_action_size();
    this.layout = JSON.parse(this.str(M._ts_action_layout_json()));
    this.stateSize = M._ts_state_size();
  }

  /** Load the module from `<base>engine/ts_engine.mjs`. */
  static async load(base: string = import.meta.env.BASE_URL): Promise<WasmEngine> {
    const url = `${base}engine/ts_engine.mjs`;
    let createModule: (opts?: object) => Promise<Module>;
    try {
      createModule = (await import(/* @vite-ignore */ url)).default;
    } catch (e) {
      throw new Error(`The WebAssembly engine is missing (${url}). Build it with tools/scripts/build_web.sh. (${e})`);
    }
    return new WasmEngine(await createModule());
  }

  private str(ptr: number): string {
    return this.M.UTF8ToString(ptr);
  }

  private lastError(): string {
    return this.str(this.M._ts_last_error());
  }

  // ---- the position -------------------------------------------------------------------------

  newGame(seed: number): void {
    const lo = seed >>> 0;
    const hi = Math.floor(seed / 4294967296) >>> 0;
    this.M._ts_new_game(lo, hi);
  }

  /** The raw GameState, for an undo stack (the state is trivially copyable). */
  snapshot(): Uint8Array {
    return this.M.HEAPU8.slice(this.M._ts_state_ptr(), this.M._ts_state_ptr() + this.stateSize);
  }

  restore(bytes: Uint8Array): void {
    if (bytes.length !== this.stateSize) throw new Error("snapshot from another engine build");
    this.M.HEAPU8.set(bytes, this.M._ts_state_ptr());
  }

  display(): GameState {
    return JSON.parse(this.str(this.M._ts_display_json()));
  }

  saveJson(): string {
    return this.str(this.M._ts_save_json());
  }

  /** Open a saved position. Throws with the engine's reason when it is not one. */
  loadSaveJson(text: string): void {
    const ptr = this.M.stringToNewUTF8(text);
    try {
      if (!this.M._ts_load_save_json(ptr)) throw new Error(this.lastError());
    } finally {
      this.M._free(ptr);
    }
  }

  isTerminal(): boolean { return this.M._ts_is_terminal() !== 0; }
  terminalUtility(): number { return this.M._ts_terminal_utility(); }
  endingReason(): string { return this.str(this.M._ts_ending_reason()); }
  /** 1 = US, -1 = USSR, 0 = nobody (a chance node, or the game is over). */
  decisionPlayer(): number { return this.M._ts_decision_player(); }
  decisionType(): number { return this.M._ts_decision_type(); }

  // ---- acting -------------------------------------------------------------------------------

  /**
   * One MicroAction, then the dice it lands on (`forcedDie` 0 = roll, 1..6 = force). All or
   * nothing: a refusal leaves the position unchanged and is returned as the reason.
   */
  step(a: MicroAction, forcedDie = 0): string | null {
    const ok = this.M._ts_step(a.decision_type, a.primary_id, a.secondary_id ?? 0, a.flags ?? 0, forcedDie);
    return ok ? null : this.lastError();
  }

  mask(merged: boolean): Uint8Array {
    const p = this.M._ts_mask(merged ? 1 : 0);
    return this.M.HEAPU8.slice(p, p + this.actionSize);
  }

  decodeFlat(idx: number): Required<MicroAction> {
    const p = this.M._ts_decode_flat(idx) >> 2;
    const h = this.M.HEAP32;
    return { decision_type: h[p], primary_id: h[p + 1], secondary_id: h[p + 2], flags: h[p + 3] };
  }

  isMergedInfluenceAction(idx: number): boolean {
    return this.M._ts_is_merged_influence_action(idx) !== 0;
  }

  /** The observation from one side's perspective (PLAYER_US / PLAYER_USSR). */
  observation(player: number): Float32Array {
    const p = this.M._ts_observation(player) >> 2;
    return this.M.HEAPF32.slice(p, p + this.obsSize);
  }

  // ---- names and debug ------------------------------------------------------------------------

  cardName(id: number): string { return this.str(this.M._ts_card_name(id)); }
  cardOps(id: number): number { return this.M._ts_card_ops(id); }
  countryName(id: number): string { return this.str(this.M._ts_country_name(id)); }

  setCountry(cid: number, us: number, ussr: number): string | null {
    return this.M._ts_set_country(cid, us, ussr) ? null : this.lastError();
  }
  setDefcon(v: number): string | null { return this.M._ts_set_defcon(v) ? null : this.lastError(); }
  setVp(v: number): string | null { return this.M._ts_set_vp(v) ? null : this.lastError(); }
}
