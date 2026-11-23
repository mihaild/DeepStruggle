/**
 * A game as the stats database keeps it (stats.ts submits it, tools/game_stats.py reads it).
 *
 * Two parts. The moves: every step's MicroAction and forced die, from the seed (or a loaded
 * position) -- enough to re-drive the game in the engine of the build's commit, so the full
 * .tslog.json is rebuilt offline rather than sent (a few KB here against MBs). And how each move
 * was made: by whom (a click, the model's suggestion played by the human, or auto-play), whether
 * the model's readout was on screen for it, and the settings in force -- view, auto side, model,
 * search -- so a game in which the human moved for the model, saw its hand in the Developer view,
 * took hints, forced a die or switched models can be told from a straight game against one
 * player. Nothing is refused: everything is recorded, and the reader filters.
 *
 * The settings travel per step, not as a timeline of changes: undo cuts the steps back, and a
 * timeline would then say the wrong setting for the steps played again.
 *
 * The log's bytes (LOG_VERSION 1), little-endian, zlib-deflated, base64url without padding:
 *   u8 version, u32 seed, u32 step count, then per step 7 bytes:
 *   decision_type, primary_id, secondary_id, flags, forced die (0 = rolled), info, setting index
 *   info: bits 0-1 source (0 human, 1 suggested, 2 auto), bit 2 readout on screen,
 *         bits 3-4 mover (0 none, 1 US, 2 USSR), bit 5 the step rolled a die.
 * tools/lib/game_stats.py reads the same bytes; summarize() and its summarize() must agree.
 */
import { MicroAction } from "../types";
import { deflateBytes, fromBase64Url, inflateBytes, toBase64Url } from "./position";

export const LOG_VERSION = 1;
const STEP_BYTES = 7;
/** A log inflating past this is refused: a game is about 500 steps, 3.5 KB. */
const MAX_LOG_BYTES = 256 * 1024;

export type Side = "US" | "USSR";
/** Who made a move: a click, the model's suggestion played from the readout, or auto-play. */
export type StepSource = "human" | "suggested" | "auto";
const SOURCES: StepSource[] = ["human", "suggested", "auto"];

/** How a move was made, as the page knows it before the engine takes it. */
export interface MoveHow {
  source: StepSource;
  /** The model's readout was on screen for the mover's decision. */
  analysis: boolean;
}

export interface StepTag extends MoveHow {
  mover: Side | "";
  /** The die forced for the step's rolls (0 = the engine rolled). */
  forcedDie: number;
  /** The step rolled a die, so a forced die took effect. */
  rolled: boolean;
  /** Index into the record's settings. */
  setting: number;
}

export interface RecordStep {
  action: Required<MicroAction>;
  tag: StepTag;
}

/** What was in force for a move. `model` is the model file's sha256 ("" for none). */
export interface Settings {
  view: "DEV" | "US" | "USSR";
  auto: Side | "";
  model: string;
  /** "policy", or the rollout root's spec as the URL writes it (analysis/rollout.ts). */
  search: string;
}

export interface ModelInfo {
  label: string;
  /** Where it came from, as the URL writes it ("" for a dropped file). */
  source: string;
  /** The checkpoint it was exported from (ts.checkpoint in the ONNX metadata). */
  checkpoint: string;
  /** The obs_features bits the page read from it (ts.obs_features), whose rule bits its seat played:
   *  a reload seats it so again. Absent in records made before 2026-11-23. */
  obs_features?: number;
}

/** What happened besides moves. */
export type RecordEventKind = "undo" | "debug" | "load" | "resume";

/** A debug override's change to the position, exactly: what a rebuild of the game must apply. */
export type DebugOverride =
  | { kind: "country"; id: number; us: number; ussr: number }
  | { kind: "defcon"; value: number }
  | { kind: "vp"; value: number };

export interface RecordEvent {
  /** Steps on the board when it happened. */
  step: number;
  kind: RecordEventKind;
  text: string;
  /** A "debug" event's change. The steps alone do not rebuild a game that had one: it is applied
   *  before step `step` (game/session.ts replayRecord), and a game with any is never `clean`. */
  override?: DebugOverride;
  /** The override was taken back (undo): a rebuild skips it. It still counts in debug_overrides. */
  undone?: boolean;
}

export interface RecordSummary {
  /** The side auto-play moved; "mixed" when it moved both, null when it moved neither. */
  model_side: Side | "mixed" | null;
  human_side: Side | null;
  /** Moves the human made for the model's side. */
  human_moves_for_model: number;
  /** The human's moves made by playing the model's suggestion. */
  suggested_moves: number;
  /** The human's moves made with the model's readout on screen. */
  analysis_moves: number;
  /** The human's moves made in the Developer view, which shows the model's hand. */
  dev_view_moves: number;
  /** Steps whose die was forced (and rolled). */
  forced_dice: number;
  undos: number;
  debug_overrides: number;
  loaded_position: boolean;
  /** The models (sha256) and searches auto-play moved with. */
  models: string[];
  searches: string[];
  /** One model with one search played one side all game, the human the other with nothing of the
   *  above, and the game finished. Undos are counted, not disqualifying. */
  clean: boolean;
}

/** The game's moves and how they were made: what summarize() and the log are made from. */
export interface GameRecordData {
  id: string;
  started_at: string;
  seed: number;
  start_position: string | null;
  steps: RecordStep[];
  settings: Settings[];
  models: Record<string, ModelInfo>;
  events: RecordEvent[];
}

export interface GameOutcome {
  winner: string;
  margin: number;
  end_turn: number;
  reason: string;
}

/** The build the game was played on: the commit to rebuild it at, and the engine it ran. */
export interface BuildInfo {
  commit: string;
  /** Built from a tree with uncommitted changes: the commit alone may not rebuild it. */
  dirty: boolean;
  engine_fingerprint: string;
}

export const SUBMISSION_SCHEMA = 1;

/** One game for the stats database: POSTed to its Worker (web/stats_worker/), read back by
 *  tools/game_stats.py. The Worker checks this shape; keep the three in step. */
export interface Submission {
  schema: number;
  id: string;
  started_at: string;
  ended_at: string;
  build: BuildInfo;
  playdek: { login: string; rating: number };
  note: string;
  result: GameOutcome;
  summary: RecordSummary;
  settings: Settings[];
  models: Record<string, ModelInfo>;
  events: RecordEvent[];
  start_position: string | null;
  /** encodeLog(): the seed and every step. */
  log: string;
  n_steps: number;
  /** sha256 (hex) of the engine's save JSON at the end: a rebuilt game must land on it. */
  final_hash: string;
}

/** sha256, hex. Rejects where the browser has no crypto.subtle (a page not on https or localhost). */
export async function sha256Hex(data: string | Uint8Array): Promise<string> {
  const bytes = typeof data === "string" ? new TextEncoder().encode(data) : new Uint8Array(data);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("");
}

export async function buildSubmission(rec: GameRecordData, result: GameOutcome, finalSave: string, build: BuildInfo,
                                       playdek: { login: string; rating: number }, note: string): Promise<Submission> {
  return {
    schema: SUBMISSION_SCHEMA,
    id: rec.id,
    started_at: rec.started_at,
    ended_at: new Date().toISOString(),
    build,
    playdek,
    note,
    result,
    summary: summarize(rec, true),
    settings: rec.settings,
    models: rec.models,
    events: rec.events,
    start_position: rec.start_position,
    log: await encodeLog(rec.seed, rec.steps),
    n_steps: rec.steps.length,
    final_hash: await sha256Hex(finalSave),
  };
}

function sideCode(s: Side | ""): number {
  return s === "US" ? 1 : s === "USSR" ? 2 : 0;
}

function sideOf(code: number): Side | "" {
  return code === 1 ? "US" : code === 2 ? "USSR" : "";
}

function byte(v: number, what: string): number {
  if (!Number.isInteger(v) || v < 0 || v > 255) throw new Error(`${what} ${v} does not fit a byte`);
  return v;
}

export async function encodeLog(seed: number, steps: RecordStep[]): Promise<string> {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error(`seed ${seed} does not fit 32 bits`);
  const out = new Uint8Array(9 + STEP_BYTES * steps.length);
  const view = new DataView(out.buffer);
  out[0] = LOG_VERSION;
  view.setUint32(1, seed, true);
  view.setUint32(5, steps.length, true);
  steps.forEach((s, i) => {
    const o = 9 + STEP_BYTES * i;
    const t = s.tag;
    out[o] = byte(s.action.decision_type, "decision type");
    out[o + 1] = byte(s.action.primary_id, "primary id");
    out[o + 2] = byte(s.action.secondary_id, "secondary id");
    out[o + 3] = byte(s.action.flags, "flags");
    out[o + 4] = byte(t.forcedDie, "forced die");
    out[o + 5] = SOURCES.indexOf(t.source) | (t.analysis ? 4 : 0) | (sideCode(t.mover) << 3) | (t.rolled ? 32 : 0);
    out[o + 6] = byte(t.setting, "setting index");
  });
  return toBase64Url(await deflateBytes(out));
}

export async function decodeLog(token: string): Promise<{ seed: number; steps: RecordStep[] }> {
  const raw = await inflateBytes(fromBase64Url(token), MAX_LOG_BYTES);
  if (raw.length < 9 || raw[0] !== LOG_VERSION) throw new Error(`not a version-${LOG_VERSION} game log`);
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const seed = view.getUint32(1, true);
  const n = view.getUint32(5, true);
  if (raw.length !== 9 + STEP_BYTES * n) throw new Error(`game log of ${raw.length} bytes for ${n} steps`);
  const steps: RecordStep[] = [];
  for (let i = 0; i < n; i++) {
    const o = 9 + STEP_BYTES * i;
    const info = raw[o + 5];
    steps.push({
      action: { decision_type: raw[o], primary_id: raw[o + 1], secondary_id: raw[o + 2], flags: raw[o + 3] },
      tag: {
        source: SOURCES[info & 3] ?? "human", analysis: (info & 4) !== 0, mover: sideOf((info >> 3) & 3),
        forcedDie: raw[o + 4], rolled: (info & 32) !== 0, setting: raw[o + 6],
      },
    });
  }
  return { seed, steps };
}

/** What a game's record says about how it was played. `finished`: the game reached its end. */
export function summarize(rec: GameRecordData, finished: boolean): RecordSummary {
  const modelSides = new Set<Side>();
  const models = new Set<string>();
  const searches = new Set<string>();
  for (const { tag } of rec.steps) {
    if (tag.source !== "auto" || !tag.mover) continue;
    modelSides.add(tag.mover);
    const s = rec.settings[tag.setting];
    models.add(s?.model ?? "");
    searches.add(s?.search ?? "");
  }
  const modelSide: Side | "mixed" | null = modelSides.size === 0 ? null : modelSides.size > 1 ? "mixed" : [...modelSides][0];
  const humanSide: Side | null = modelSide === "US" ? "USSR" : modelSide === "USSR" ? "US" : null;
  let forModel = 0, suggested = 0, analysis = 0, devView = 0, forced = 0;
  for (const { tag } of rec.steps) {
    if (tag.forcedDie > 0 && tag.rolled) forced++;
    if (tag.source === "auto") continue;
    if (modelSide !== null && modelSide !== "mixed" && tag.mover === modelSide) {
      forModel++;
      continue;
    }
    if (tag.source === "suggested") suggested++;
    if (tag.analysis) analysis++;
    if ((rec.settings[tag.setting]?.view ?? "DEV") === "DEV") devView++;
  }
  const count = (k: RecordEventKind) => rec.events.filter(e => e.kind === k).length;
  const summary: RecordSummary = {
    model_side: modelSide,
    human_side: humanSide,
    human_moves_for_model: forModel,
    suggested_moves: suggested,
    analysis_moves: analysis,
    dev_view_moves: devView,
    forced_dice: forced,
    undos: count("undo"),
    debug_overrides: count("debug"),
    loaded_position: rec.start_position !== null,
    models: [...models].sort(),
    searches: [...searches].sort(),
    clean: false,
  };
  summary.clean = finished && humanSide !== null && forModel === 0 && suggested === 0 && analysis === 0
    && devView === 0 && forced === 0 && summary.debug_overrides === 0 && !summary.loaded_position
    && models.size === 1 && !models.has("") && searches.size === 1;
  return summary;
}
