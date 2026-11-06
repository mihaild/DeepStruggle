/**
 * The engine, in the page: a typed wrapper around the WebAssembly build of engine/ + bindings/
 * (bindings/wasm/ts_engine_wasm.cpp, built by tools/scripts/build_web.sh into public/engine/).
 *
 * The same C++ as the Python stack -- rules, observation, display state, saves -- so the page
 * cannot show or play a position differently from Python; tests/web/test_wasm_engine.py checks
 * the two builds bit for bit. Everything returned is a copy: the module's memory can move when it
 * grows, and a view into it would silently change under the caller.
 */
import { DieRollState, EventTrackState, GameState, MicroAction, RegionScores } from "../types";
import { ActionLayout, setEngineLayout } from "./layout";

/** The flat action space's layout and the engine's enums, from the engine itself (engine/layout.ts). */
export type { ActionLayout } from "./layout";

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
    setEngineLayout(this.layout);
  }

  /**
   * Load the module from `<base>engine/ts_engine.mjs`. The URL is made absolute first: Vite's dev
   * server rewrites a dynamic import of a path ("/engine/...") to "...?import", which it refuses for
   * a file in public/ (a 500), so `npm run dev` could not load the engine. An absolute URL is
   * fetched as it is, in dev and in a build alike.
   */
  static async load(base: string = import.meta.env.BASE_URL): Promise<WasmEngine> {
    const here = globalThis.location?.href;
    const url = here ? new URL(`${base}engine/ts_engine.mjs`, here).href : `${base}engine/ts_engine.mjs`;
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

  /**
   * What the last `step` itself did: the display state after its action, or after its last die
   * roll -- before the roll-less chance steps that follow (the turn's cleanup, when an action round
   * ends the turn), which clear the roll record. `display()` is the state after all of them.
   */
  settledDisplay(): GameState {
    return JSON.parse(this.str(this.M._ts_settled_display_json()));
  }

  /** Every die the last `step` rolled, in order: one click can roll more than once (a realignment
   * with the opponent's war card rolls, then its event does), and each engine step keeps only its own. */
  clickRolls(): DieRollState[] {
    return JSON.parse(this.str(this.M._ts_click_rolls_json()));
  }

  /** What each Event the last `step` resolved did to VP and DEFCON, over the same engine steps as
   * clickRolls: the second headline commit resolves both headlines, and before/after shows their sum. */
  clickEventTracks(): EventTrackState[] {
    return JSON.parse(this.str(this.M._ts_click_event_tracks_json()));
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

  /**
   * The flat legal mask in an action view, for a model deciding with the obs_features bits
   * `features`: those seat the decider first (ts_seat_obs_features), so it plays the rule bits its
   * model was trained with -- a network from before OPP_EVENT_GATED keeps the opponent-card Event
   * that fizzles into Ops. `null` seats nobody and reads the seats' rules as they stand (the state's
   * own until a model is seated: feature set 0's, the rules from before every rule bit).
   */
  mask(merged: boolean, features: number | null): Uint8Array {
    const decider = this.decisionPlayer();
    // Loose on purpose: a page-side script calling mask(false) passes undefined, which seats nobody.
    if (features != null && decider !== 0 && !this.M._ts_seat_obs_features(decider, features >>> 0)) {
      throw new Error(this.lastError());
    }
    const p = this.M._ts_mask(merged ? 1 : 0);
    return this.M.HEAPU8.slice(p, p + this.actionSize);
  }

  /** Throws for an index outside the flat action space. */
  decodeFlat(idx: number): Required<MicroAction> {
    const ptr = this.M._ts_decode_flat(idx);
    if (!ptr) throw new Error(this.lastError());
    const p = ptr >> 2;
    const h = this.M.HEAP32;
    return { decision_type: h[p], primary_id: h[p + 1], secondary_id: h[p + 2], flags: h[p + 3] };
  }

  isMergedInfluenceAction(idx: number): boolean {
    return this.M._ts_is_merged_influence_action(idx) !== 0;
  }

  /**
   * The observation from one side's perspective (PLAYER_US / PLAYER_USSR), in the view of a model
   * reading the obs_features bits `features` (Model.obsFeatures; 0 is the base layout). A model must
   * be shown its own view: HEADLINE_REVEAL has no width, so the base layout fed to a model trained
   * with it runs without complaint and misreads Space Race box 4's headline.
   */
  observation(player: number, features = 0): Float32Array {
    const width = this.obsSizeFor(features);
    const p = this.M._ts_observation_features(player, features >>> 0) >> 2;
    return this.M.HEAPF32.slice(p, p + width);
  }

  /** The observation width for the obs_features bits `features`; throws for a bit this engine does
   *  not know (a model from a newer engine). */
  obsSizeFor(features: number): number {
    const w = this.M._ts_obs_size_for(features >>> 0);
    if (!w) throw new Error(this.lastError());
    return w;
  }

  // ---- the rollout root's playouts (bindings/rollout_batch.hpp, the same C++ as Python's) ------
  //
  // One batch at a time, over the position held. The caller runs the network: rolloutFillLive ->
  // logits -> rolloutStep until rolloutAllDone, then rolloutFillLeaves -> critic ->
  // rolloutSetLeafValues, then rolloutDecide. Arrays are copies out of the module's memory.

  /** Start a batch: a horizon in action-round boundaries, the model's obs_features bits, a seed, and
   *  where the playouts stop (0 after the boundaries, 1 at the turn's end, 2 the later of the two, 3 the earlier, 4 the turn-end park before
   *  the deal, 5 the earlier of the boundaries and that park); `keyed` ties every draw to its situation
   *  (bindings/keyed_chance.hpp): 1 the leaderboard's chance=keyed, 2 its chance=keyed2, 0 off; `aligned` counts every candidate's
   *  horizon from the end of the mover's own action round, the headline one boundary
   *  (rollout::Config::aligned); `box4` keeps the opponent's headline revealed by Space Race box 4 in
   *  every world (rollout::Config::box4). */
  rolloutReset(horizon: number, features: number, seed: number, sharedDice = true, guard = true, stop = 0,
               leaf = 0, keyed = 0, aligned = false, box4 = false): void {
    const ok = this.M._ts_rollout_reset(horizon, sharedDice ? 1 : 0, guard ? 1 : 0, features >>> 0,
      seed >>> 0, Math.floor(seed / 4294967296) >>> 0, stop, leaf, keyed, aligned ? 1 : 0, box4 ? 1 : 0);
    if (!ok) throw new Error(this.lastError());
  }

  /** The position held as one decision of the batch, its worlds' opposing hands resampled from the
   *  prepared belief; rows added. `deal` hands are dealt at once (a race's), the first `worlds` played
   *  and the rest kept for rolloutRace. */
  rolloutAddBelief(candidates: number[], worlds: number, deal = worlds): number {
    const p = this.M._malloc(4 * Math.max(1, candidates.length));
    try {
      this.M.HEAP32.set(candidates, p >> 2);
      const n = this.M._ts_rollout_add_belief(p, candidates.length, worlds, deal);
      if (n === 0 && candidates.length >= 2) throw new Error(this.lastError() || "no rows");
      return n;
    } finally {
      this.M._free(p);
    }
  }

  // ---- the belief over the opponent's hand (bindings/belief.hpp, the same C++ as Python's) ------
  //
  // One tracker, one game. beliefReset at a new game; beliefObserve before every move is applied;
  // at the AI's decision beliefPrepare, then per recorded decision and chunk of hands beliefRows ->
  // the network -> the move's log-probability at each temperature -> beliefSetLogLikelihoods; then
  // rolloutAddBelief.

  /** `reveal`: every hand holds the opponent's headline revealed by Space Race box 4 (the root's box4). */
  beliefReset(particles: number, seed: number, allDecisions = true, reveal = false): void {
    this.M._ts_belief_reset(particles >>> 0, allDecisions ? 1 : 0, seed >>> 0, Math.floor(seed / 4294967296) >>> 0,
      reveal ? 1 : 0);
  }
  beliefTemperatures(): number[] {
    const n = this.M._ts_belief_temperatures();
    return Array.from({ length: n }, (_, t) => this.M._ts_belief_temperature(t));
  }
  beliefParticles(): number { return this.M._ts_belief_particles(); }
  /** Record the move about to be applied at the position held; false when no flat action is that move. */
  beliefObserve(a: MicroAction): boolean {
    return this.M._ts_belief_observe(a.decision_type, a.primary_id, a.secondary_id ?? 0, a.flags ?? 0) !== 0;
  }
  /** Whether the position held has a belief; then beliefDecisions() blocks to evaluate (0: reused). */
  beliefPrepare(): boolean { return this.M._ts_belief_prepare() !== 0; }
  beliefDecisions(): number { return this.M._ts_belief_decisions(); }
  beliefAction(d: number): number { return this.M._ts_belief_action(d); }
  /** Recorded decision d under hands lo .. lo + n - 1: n observation rows and masks. */
  beliefRows(d: number, lo: number, n: number, features: number, width: number): { obs: Float32Array; masks: Uint8Array } {
    const rows = this.M._ts_belief_rows(d, lo, n, features >>> 0);
    if (rows !== n) throw new Error(this.lastError());
    const o = this.M._ts_rollout_buf_obs() >> 2;
    const m = this.M._ts_rollout_buf_masks();
    return { obs: this.M.HEAPF32.slice(o, o + n * width), masks: this.M.HEAPU8.slice(m, m + n * this.actionSize) };
  }
  /** temperatures x decisions x particles log-probabilities of the recorded moves. */
  beliefSetLogLikelihoods(lp: Float32Array): void {
    const p = this.M._malloc(4 * Math.max(1, lp.length));
    try {
      this.M.HEAPF32.set(lp, p >> 2);
      if (!this.M._ts_belief_set_log_likelihoods(p)) throw new Error(this.lastError());
    } finally {
      this.M._free(p);
    }
  }
  beliefTemperatureWeights(): number[] {
    const n = this.M._ts_belief_temperatures();
    const p = this.M._malloc(4 * n);
    try {
      this.M._ts_belief_temperature_weights(p);
      return Array.from(this.M.HEAPF32.subarray(p >> 2, (p >> 2) + n));
    } finally {
      this.M._free(p);
    }
  }

  /** The `k` most probable legal moves by the root's logits under `mask`, best first. */
  rolloutCandidates(logits: Float32Array, mask: Uint8Array, k: number): number[] {
    const lg = this.M._malloc(4 * this.actionSize), mk = this.M._malloc(this.actionSize), out = this.M._malloc(4 * k);
    try {
      this.M.HEAPF32.set(logits, lg >> 2);
      this.M.HEAPU8.set(mask, mk);
      const n = this.M._ts_rollout_candidates(lg, mk, k, out);
      if (n < 0) throw new Error(this.lastError());
      return Array.from(this.M.HEAP32.subarray(out >> 2, (out >> 2) + n));
    } finally {
      this.M._free(lg); this.M._free(mk); this.M._free(out);
    }
  }

  /** The position held as one decision of the batch, with `candidates`, in `worlds` worlds; rows added. */
  rolloutAdd(candidates: number[], worlds: number): number {
    const p = this.M._malloc(4 * Math.max(1, candidates.length));
    try {
      this.M.HEAP32.set(candidates, p >> 2);
      const n = this.M._ts_rollout_add(p, candidates.length, worlds);
      if (n === 0 && candidates.length >= 2) throw new Error(this.lastError() || "no rows");
      return n;
    } finally {
      this.M._free(p);
    }
  }

  /** Race `position` once its rows are valued (rollout::RolloutBatch::race): the challengers trailing the
   *  network's move leave, the rest play `worlds` more worlds, from the belief's kept hands with `belief`.
   *  Rows added; 0 when no challenger is left. */
  rolloutRace(position: number, worlds: number, belief = false): number {
    const n = this.M._ts_rollout_race(position, worlds, belief ? 1 : 0);
    if (n < 0) throw new Error(this.lastError());
    return n;
  }
  /** Whether candidate `c` of `position` is still in the race (every candidate of an unraced one). */
  rolloutInRace(position: number, c: number): boolean { return this.M._ts_rollout_in_race(position, c) !== 0; }

  rolloutAllDone(): boolean { return this.M._ts_rollout_all_done() !== 0; }
  rolloutLive(): number { return this.M._ts_rollout_live(); }
  rolloutRefused(): number { return this.M._ts_rollout_refused(); }
  rolloutObsWidth(): number { return this.M._ts_rollout_obs_width(); }

  /** The live rows' observations and masks, each row from its acting side. */
  rolloutFillLive(): { rows: number; obs: Float32Array; masks: Uint8Array } {
    const rows = this.M._ts_rollout_fill_live();
    return { rows, ...this.rolloutBuffers(rows) };
  }

  /** Advance the rows the last fill listed by the masked argmax of each logits row (rows x actions). */
  rolloutStep(logits: Float32Array, rows: number): void {
    const p = this.M._malloc(4 * Math.max(1, rows * this.actionSize));
    try {
      this.M.HEAPF32.set(logits.subarray(0, rows * this.actionSize), p >> 2);
      if (!this.M._ts_rollout_step(p)) throw new Error(this.lastError());
    } finally {
      this.M._free(p);
    }
  }

  /** The rows at their horizon and not yet valued, from the MOVER's side. */
  rolloutFillLeaves(): { rows: number; obs: Float32Array; masks: Uint8Array } {
    const rows = this.M._ts_rollout_fill_leaves();
    return { rows, ...this.rolloutBuffers(rows) };
  }

  rolloutSetLeafValues(values: Float32Array): void {
    const p = this.M._malloc(4 * Math.max(1, values.length));
    try {
      this.M.HEAPF32.set(values, p >> 2);
      if (!this.M._ts_rollout_set_leaf_values(p)) throw new Error(this.lastError());
    } finally {
      this.M._free(p);
    }
  }

  /** The decision for position `position` of the batch under `rule` (argmax, z<x>, kl<t>). */
  rolloutDecide(position: number, candidates: number[], logits: Float32Array, mask: Uint8Array, rule: string)
      : { pick: number; worlds: number; q: number[]; lead: number[]; lead_se: number[] } {
    const c = this.M._malloc(4 * Math.max(1, candidates.length));
    const lg = this.M._malloc(4 * this.actionSize), mk = this.M._malloc(this.actionSize);
    const r = this.M.stringToNewUTF8(rule);
    try {
      this.M.HEAP32.set(candidates, c >> 2);
      this.M.HEAPF32.set(logits, lg >> 2);
      this.M.HEAPU8.set(mask, mk);
      const text = this.str(this.M._ts_rollout_decide(position, c, candidates.length, lg, mk, r));
      if (!text) throw new Error(this.lastError());
      return JSON.parse(text);
    } finally {
      this.M._free(c); this.M._free(lg); this.M._free(mk); this.M._free(r);
    }
  }

  private rolloutBuffers(rows: number): { obs: Float32Array; masks: Uint8Array } {
    const w = this.M._ts_rollout_obs_width();
    const o = this.M._ts_rollout_buf_obs() >> 2;
    const m = this.M._ts_rollout_buf_masks();
    return { obs: this.M.HEAPF32.slice(o, o + rows * w), masks: this.M.HEAPU8.slice(m, m + rows * this.actionSize) };
  }

  // ---- what the scoring cards would score -----------------------------------------------------

  /**
   * What each scoring card would score in `state` -- any display state the page draws: the live
   * game, a replay's snapshot, a player's view -- by the engine's own scoring
   * (Scoring::evaluate_region, score_southeast_asia). Throws for a state the engine cannot read.
   */
  regionScores(state: Pick<GameState, "countries" | "flags">): RegionScores {
    const ptr = this.M.stringToNewUTF8(JSON.stringify({ countries: state.countries ?? {}, flags: state.flags ?? [] }));
    try {
      const text = this.str(this.M._ts_region_scores_json(ptr));
      if (!text) throw new Error(this.lastError());
      return JSON.parse(text);
    } finally {
      this.M._free(ptr);
    }
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
