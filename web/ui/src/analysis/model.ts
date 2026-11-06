/**
 * A policy/value network in the page: an ONNX export of a checkpoint (tools/export_onnx.py) run
 * by onnxruntime-web, from one of three sources --
 *
 *   local  the checkpoints tree the local server lists (`/api/local/models`); it exports a
 *          .pt to ONNX on first request and caches it, so any snapshot can be picked
 *   hf     a file in a Hugging Face model repo, fetched from huggingface.co/<repo>/resolve/...
 *          and kept in the browser under its sha256 (model_cache.ts), so reopening the page
 *          does not download it again
 *   file   an .onnx dropped onto the page
 *
 * The file carries its own description (metadata_props, see onnx_meta.ts). The observation
 * width is checked before anything runs: a model handed an observation of another width does not
 * fail, it reads fixed slices and misreads them, which is why the Python side refuses such a
 * checkpoint too (bindings.ts_env.check_obs_width). An engine fingerprint that differs from the
 * page's is shown, not refused: the model was exported next to another engine build, which is
 * worth knowing but not necessarily wrong.
 */
import * as ort from "onnxruntime-web/wasm";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import ortMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import { onnxMetadata } from "./onnx_meta";
import { cachedModel, hfFileSha256, storeModel } from "./model_cache";
import { hfFileUrl } from "./hf_catalog";

// The runtime is served from this build, not a CDN, and single-threaded: GitHub Pages cannot send
// the cross-origin-isolation headers threads need, and one position is well under a millisecond.
ort.env.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortMjsUrl };
ort.env.wasm.numThreads = 1;

export type ModelSource =
  | { kind: "local"; path: string }
  | { kind: "hf"; repo: string; revision: string; path: string }
  | { kind: "file"; name: string };

/** What a source is called in the address bar; `file` has no address and is not shareable. */
export function sourceToParam(s: ModelSource): string | null {
  if (s.kind === "local") return `local:${s.path}`;
  if (s.kind === "hf") return `hf:${s.repo}@${s.revision}:${s.path}`;
  return null;
}

export function sourceFromParam(v: string): ModelSource | null {
  if (v.startsWith("local:")) return { kind: "local", path: v.slice(6) };
  const hf = v.match(/^hf:([^@:]+)@([^:]+):(.+)$/);
  if (hf) return { kind: "hf", repo: hf[1], revision: hf[2], path: hf[3] };
  // A bare run/snapshot path, as links from the server-side workbench wrote them.
  if (v && !v.includes(":")) return { kind: "local", path: v };
  return null;
}

export function sourceLabel(s: ModelSource): string {
  if (s.kind === "local") return s.path;
  if (s.kind === "hf") return `${s.repo}/${s.path}`;
  return s.name;
}

// The repo's catalogue lives in hf_catalog.ts (no onnxruntime there, for the leaderboard page).
export {
  hfFileUrl, DEFAULT_HF_REPO, DEFAULT_HF_REVISION, listHfModels, hfCatalog, defaultHfCatalog, groupHfModels,
  fetchHfDefault,
} from "./hf_catalog";
export type { HfModelFile, HfCatalog, HfModelGroup } from "./hf_catalog";

/** The obs_features bits a model reads: ts.obs_features in the export; 0 for the base layout and for
 *  exports made before the key existed. */
function featureBits(raw: Record<string, string>): number {
  const v = Number(raw["ts.obs_features"] ?? "0");
  return Number.isFinite(v) ? v >>> 0 : 0;
}

export interface ModelMeta {
  label: string;
  mergedInfluence: boolean;
  /** P42: decides in the macro view; its logits are the engine's full width. */
  macroInfluence: boolean;
  obsSize: number;
  actionSize: number;
  engineFingerprint: string;
  checkpoint: string;
  raw: Record<string, string>;
}

async function download(source: ModelSource, url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`could not fetch ${sourceLabel(source)}: HTTP ${res.status} ${detail.slice(0, 300)}`);
  }
  return new Uint8Array(await res.arrayBuffer());
}

export interface Forward {
  logits: Float32Array;   // rows * actionSize
  vWin: Float32Array;     // rows
  vVp: Float32Array;      // rows
}

export class Model {
  private constructor(
    readonly source: ModelSource,
    readonly meta: ModelMeta,
    private readonly session: ort.InferenceSession,
    /** The bytes came from the browser's model cache rather than the network. */
    readonly fromCache: boolean,
    /** The file itself, kept so a worker can open its own session on the same model. */
    readonly bytes: Uint8Array,
    /** The engine's mask width: masks come in this wide and logits go out this wide. */
    readonly engineActionSize: number,
  ) {}

  /**
   * Build from the file's bytes. `obsSizeFor` is the engine's width for a feature set
   * (WasmEngine.obsSizeFor) and `actionSize` its action space; a model whose width is not the one
   * this engine writes for the model's own obs_features, or that reads a feature bit this engine does
   * not know, is refused here, with the reason, before it can misread a single observation. A base
   * network's logits are the base slots (`baseActionSize`); `run` pads them to the engine's width.
   */
  static async fromBytes(source: ModelSource, bytes: Uint8Array, obsSizeFor: (features: number) => number,
                         actionSize: number, fromCache = false, baseActionSize = 220): Promise<Model> {
    let raw: Record<string, string>;
    try {
      raw = onnxMetadata(bytes);
    } catch (e) {
      throw new Error(`${sourceLabel(source)} is not an ONNX model (${e instanceof Error ? e.message : e})`);
    }
    if (raw["ts.format"] !== "ts-onnx-v1") {
      throw new Error(`${sourceLabel(source)} was not exported by tools/export_onnx.py (no ts.format metadata)`);
    }
    const meta: ModelMeta = {
      label: raw["ts.label"] || sourceLabel(source),
      mergedInfluence: raw["ts.merged_influence"] === "true",
      macroInfluence: raw["ts.macro_influence"] === "true",
      obsSize: Number(raw["ts.obs_size"]),
      actionSize: Number(raw["ts.action_size"]),
      engineFingerprint: raw["ts.engine_fingerprint"] || "",
      checkpoint: raw["ts.checkpoint"] || "",
      raw,
    };
    const features = featureBits(raw);
    let obsSize: number;
    try {
      obsSize = obsSizeFor(features);
    } catch {
      throw new Error(`${meta.label} reads observation features 0x${features.toString(16)}, which this engine does not `
        + `know: it was exported next to a newer engine. Rebuild the page (tools/scripts/build_web.sh).`);
    }
    const width = meta.macroInfluence ? actionSize : baseActionSize;
    if (meta.obsSize !== obsSize || (meta.actionSize !== width && meta.actionSize !== actionSize)) {
      throw new Error(`${meta.label} reads ${meta.obsSize} observation floats and ${meta.actionSize} actions; `
        + `this engine has ${obsSize} and ${actionSize} for its features. It was trained on another layout and would `
        + `misread every position.`);
    }
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    return new Model(source, meta, session, fromCache, bytes, actionSize);
  }

  /** The obs_features bits the network reads (ts.obs_features in the export; 0 for the base layout
   *  and for exports made before the key existed). */
  get obsFeatures(): number {
    return featureBits(this.meta.raw);
  }

  static async fetch(source: ModelSource, obsSizeFor: (features: number) => number, actionSize: number,
                     fileBytes?: Uint8Array, baseActionSize = 220): Promise<Model> {
    if (source.kind === "file") {
      if (!fileBytes) throw new Error("a dropped model needs its bytes");
      return Model.fromBytes(source, fileBytes, obsSizeFor, actionSize, false, baseActionSize);
    }
    if (source.kind === "local") {
      // The local server exports and keeps the ONNX itself, and serves it from this machine.
      const url = `/api/local/models/onnx?path=${encodeURIComponent(source.path)}`;
      return Model.fromBytes(source, await download(source, url), obsSizeFor, actionSize, false, baseActionSize);
    }
    // The cache only saves a download: if Hugging Face will not say what the file is, or the
    // browser will not keep it, the model is downloaded and used all the same.
    const sha256 = await hfFileSha256(source.repo, source.revision, source.path).catch(() => null);
    const hit = sha256 ? await cachedModel(sha256).catch(() => null) : null;
    if (hit) return Model.fromBytes(source, hit, obsSizeFor, actionSize, true, baseActionSize);
    const bytes = await download(source, hfFileUrl(source.repo, source.revision, source.path));
    const model = await Model.fromBytes(source, bytes, obsSizeFor, actionSize, false, baseActionSize);
    // Only a model that loaded is kept: one refused here would be refused again from the cache.
    if (sha256) await storeModel(sha256, bytes).catch(() => false);
    return model;
  }

  /** One forward over `rows` positions: observations and masks row-major, masks at the engine's width. A base
   *  network sees the base slots of each mask and its logits are padded with the mask's fill (P42). */
  async run(obs: Float32Array, masks: Uint8Array, rows: number): Promise<Forward> {
    const own = this.meta.actionSize, full = this.engineActionSize;
    let m = masks;
    if (own < full) {
      m = new Uint8Array(rows * own);
      for (let r = 0; r < rows; r++) m.set(masks.subarray(r * full, r * full + own), r * own);
    }
    const out = await this.session.run({
      obs: new ort.Tensor("float32", obs, [rows, this.meta.obsSize]),
      mask: new ort.Tensor("uint8", m, [rows, own]),
    });
    let logits = out.logits.data as Float32Array;
    if (own < full) {
      const padded = new Float32Array(rows * full).fill(-1e9);
      for (let r = 0; r < rows; r++) padded.set(logits.subarray(r * own, r * own + own), r * full);
      logits = padded;
    }
    return {
      logits,
      vWin: out.v_win.data as Float32Array,
      vVp: out.v_vp.data as Float32Array,
    };
  }
}
