/**
 * The rollout root off the main thread: this worker holds its own engine instance and its own
 * onnxruntime session on the same model, takes a position as raw state bytes, and answers with
 * the decision. The page stays responsive while a decision's forwards run (rollout_runner.ts).
 *
 *   { type: "load", base, bytes, source }  -> { type: "loaded" } | { type: "error" }
 *   { type: "search", id, state, spec, seed }                   -> { type: "result", id, result } | { type: "error", id }
 *   { type: "belief", reset, hands, reveal, moves }             -> (nothing) the belief tracker sees the game's moves
 */
import { WasmEngine } from "../engine/wasm_engine";
import { Model, ModelSource } from "./model";
import { RolloutSpec, SearchResult, rolloutRoot } from "./rollout";
import type { Move } from "../game/session";

interface LoadMsg { type: "load"; base: string; bytes: Uint8Array; source: ModelSource }
interface SearchMsg { type: "search"; id: number; state: Uint8Array; spec: RolloutSpec; seed: number }
interface BeliefMsg { type: "belief"; reset: boolean; hands: number; reveal: boolean; moves: Move[] }
type InMsg = LoadMsg | SearchMsg | BeliefMsg;
export type OutMsg =
  | { type: "loaded"; fingerprint: string }
  | { type: "result"; id: number; result: SearchResult | null }
  | { type: "error"; id?: number; message: string };

let engine: WasmEngine | null = null;
let model: Model | null = null;

const post = (m: OutMsg) => (self as unknown as Worker).postMessage(m);

/**
 * One message at a time. A search awaits the model between its batch steps, and the engine and its
 * rollout batch are this worker's only ones: a second search started meanwhile would restore its
 * own position into them under the first.
 */
let queue: Promise<void> = Promise.resolve();
self.onmessage = (ev: MessageEvent<InMsg>) => {
  queue = queue.then(() => handle(ev.data));
};

async function handle(msg: InMsg): Promise<void> {
  try {
    if (msg.type === "load") {
      engine = await WasmEngine.load(msg.base);
      const own = engine;
      // Both widths from this worker's own engine: its masks are actionSize wide, and a base model's logits are padded
      // to them (P42).
      model = await Model.fromBytes(msg.source, msg.bytes, f => own.obsSizeFor(f), own.actionSize, false,
                                    own.baseActionSize);
      post({ type: "loaded", fingerprint: engine.fingerprint });
      return;
    }
    if (!engine || !model) throw new Error("the search worker has no model loaded");
    if (msg.type === "belief") {
      if (msg.reset) engine.beliefReset(msg.hands, 0, true, msg.reveal);
      for (const m of msg.moves) {
        engine.restore(m.before);
        engine.beliefObserve(m.action);
      }
      return;
    }
    engine.restore(msg.state);
    const result = await rolloutRoot(engine, model, msg.spec, msg.seed);
    post({ type: "result", id: msg.id, result });
  } catch (e) {
    post({ type: "error", id: msg.type === "search" ? msg.id : undefined, message: e instanceof Error ? e.message : String(e) });
  }
}
