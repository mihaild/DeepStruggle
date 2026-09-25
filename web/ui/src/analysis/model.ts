/**
 * A policy/value network in the page: an ONNX export of a checkpoint (tools/export_onnx.py) run
 * by onnxruntime-web, from one of three sources --
 *
 *   local  the checkpoints tree the local server lists (`/api/local/models`); it exports a
 *          .pt to ONNX on first request and caches it, so any snapshot can be picked
 *   hf     a file in a Hugging Face model repo, fetched from huggingface.co/<repo>/resolve/...
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

export function hfFileUrl(repo: string, revision: string, path: string): string {
  return `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/${path.split("/").map(encodeURIComponent).join("/")}`;
}

/** Where the page takes its model from when a link names none: the newest upload here. */
export const DEFAULT_HF_REPO = "mihaild/deepstruggle";
export const DEFAULT_HF_REVISION = "main";

export interface HfModelFile {
  path: string;
  /** ISO-8601 time of the commit that last changed the file; "" if the API did not say. */
  date: string;
}

/**
 * The .onnx files of a public Hugging Face model repo, newest upload first. `expand=true` makes
 * the tree API report each file's last commit; an expanded listing is paged, and the next page
 * is named by the `Link` header, which the API exposes to cross-origin pages.
 */
export async function listHfModels(repo: string, revision: string): Promise<HfModelFile[]> {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) throw new Error("a Hugging Face repo must look like owner/name");
  const files: HfModelFile[] = [];
  let url: string | null =
    `https://huggingface.co/api/models/${repo}/tree/${encodeURIComponent(revision)}?recursive=true&expand=true`;
  while (url) {
    const res: Response = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}${res.status === 401 || res.status === 404 ? " (private or missing repo?)" : ""}`);
    const entries: Array<{ type: string; path: string; lastCommit?: { date?: string } }> = await res.json();
    for (const e of entries) {
      if (e.type === "file" && e.path.endsWith(".onnx")) files.push({ path: e.path, date: e.lastCommit?.date ?? "" });
    }
    url = res.headers.get("Link")?.match(/<([^>]+)>;\s*rel="next"/)?.[1] ?? null;
  }
  // ISO-8601 times order as strings; files uploaded in one commit fall back to their path.
  files.sort((a, b) => (a.date === b.date ? a.path.localeCompare(b.path) : a.date < b.date ? 1 : -1));
  return files;
}

export interface ModelMeta {
  label: string;
  mergedInfluence: boolean;
  obsSize: number;
  actionSize: number;
  engineFingerprint: string;
  checkpoint: string;
  raw: Record<string, string>;
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
  ) {}

  /**
   * Build from the file's bytes. `obsSize`/`actionSize` are the engine's; a model for another
   * width is refused here, with the reason, before it can misread a single observation.
   */
  static async fromBytes(source: ModelSource, bytes: Uint8Array, obsSize: number, actionSize: number): Promise<Model> {
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
      obsSize: Number(raw["ts.obs_size"]),
      actionSize: Number(raw["ts.action_size"]),
      engineFingerprint: raw["ts.engine_fingerprint"] || "",
      checkpoint: raw["ts.checkpoint"] || "",
      raw,
    };
    if (meta.obsSize !== obsSize || meta.actionSize !== actionSize) {
      throw new Error(`${meta.label} reads ${meta.obsSize} observation floats and ${meta.actionSize} actions; `
        + `this engine has ${obsSize} and ${actionSize}. It was trained on another layout and would misread every position.`);
    }
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    return new Model(source, meta, session);
  }

  static async fetch(source: ModelSource, obsSize: number, actionSize: number, fileBytes?: Uint8Array): Promise<Model> {
    if (source.kind === "file") {
      if (!fileBytes) throw new Error("a dropped model needs its bytes");
      return Model.fromBytes(source, fileBytes, obsSize, actionSize);
    }
    const url = source.kind === "local"
      ? `/api/local/models/onnx?path=${encodeURIComponent(source.path)}`
      : hfFileUrl(source.repo, source.revision, source.path);
    const res = await fetch(url);
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`could not fetch ${sourceLabel(source)}: HTTP ${res.status} ${detail.slice(0, 300)}`);
    }
    return Model.fromBytes(source, new Uint8Array(await res.arrayBuffer()), obsSize, actionSize);
  }

  /** One forward over `rows` positions: observations and masks row-major. */
  async run(obs: Float32Array, masks: Uint8Array, rows: number): Promise<Forward> {
    const out = await this.session.run({
      obs: new ort.Tensor("float32", obs, [rows, this.meta.obsSize]),
      mask: new ort.Tensor("uint8", masks, [rows, this.meta.actionSize]),
    });
    return {
      logits: out.logits.data as Float32Array,
      vWin: out.v_win.data as Float32Array,
      vVp: out.v_vp.data as Float32Array,
    };
  }
}
