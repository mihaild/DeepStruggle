/**
 * The rollout root in the page: the network's top moves, each played out by the network in sampled
 * worlds, then a cautious choice (ai/search/rollout_root.py's docstring). Everything but the
 * network calls is the engine's C++ (bindings/rollout_batch.hpp, exported by the WebAssembly
 * build): the candidates, the worlds, the playouts with their boundary count and suicide guard,
 * the leaves, and the decision with its statistics. This file is the loop that feeds
 * onnxruntime's logits and values in -- the same loop the Python searcher runs with torch.
 *
 * The spec is the leaderboard's: `rollout:k:worlds:horizon:rule[:stop][:leaf][:keyed|keyed2][:aligned][:box4][:belief[N]][:race<n>]`
 * (`rollout` alone is the best 32-world player there, 4:32:4:z2:earlier:own:keyed:aligned:box4). `keyed` ties
 * every draw to its situation (the leaderboard's chance=keyed); `belief` deals the worlds from the
 * policy-likelihood belief over the opponent's hand (bindings/belief.hpp, N hands, 2,048 by default --
 * the leaderboard's belief=N), the same C++ as Python's; `race<n>` plays every candidate on the first n worlds,
 * then the rest only for the network's move and the challengers whose paired lead over it is not below 0
 * (rollout::RolloutBatch::race). The playouts are honest -- the candidates come
 * from the real position, the values from sampled worlds -- so the search never reads the other side's
 * hand; the belief reads only the moves both players saw.
 */
import { WasmEngine } from "../engine/wasm_engine";
import { Model } from "./model";
import { RolloutSpec } from "./rollout_spec";

export type { RolloutSpec } from "./rollout_spec";
export {
  STOPS, LEAVES, DEFAULT_ROLLOUT, DEFAULT_BELIEF_HANDS, parseRolloutParam, rolloutParam, specEqual, parseSearchParam,
} from "./rollout_spec";

/** What the root decided at one position, and the evidence (ts.rollout_decide's record). */
export interface SearchResult {
  kind: "rollout";
  spec: RolloutSpec;
  /** The candidates, the network's own move first, as flat actions. */
  candidates: number[];
  pick: number;
  worlds: number;
  q: number[];
  lead: number[];
  lead_se: number[];
  /** Racing: the candidates that left the race (never picked). */
  dropped?: number[];
  /** Network rows spent, the root included, and the wall time. */
  rows: number;
  ms: number;
  /** With a belief: the temperature weights it settled on for this opponent (bindings/belief.hpp). */
  temperature_weights?: number[];
}

/** Play the batch's live rows to their horizons and value its leaves; the network rows spent. */
async function playOut(engine: WasmEngine, model: Model): Promise<number> {
  let rows = 0;
  for (let ply = 0; !engine.rolloutAllDone(); ply++) {
    if (ply > 400) throw new Error(`${engine.rolloutLive()} playouts still running after 400 plies`);
    const live = engine.rolloutFillLive();
    const out = await model.run(live.obs, live.masks, live.rows);
    rows += live.rows;
    engine.rolloutStep(out.logits, live.rows);
  }
  if (engine.rolloutRefused()) throw new Error(`the engine refused a playout action in ${engine.rolloutRefused()} rows`);
  const leaves = engine.rolloutFillLeaves();
  if (leaves.rows > 0) {
    const out = await model.run(leaves.obs, leaves.masks, leaves.rows);
    rows += leaves.rows;
    engine.rolloutSetLeafValues(out.vWin.slice(0, leaves.rows));
  }
  return rows;
}

/** Hands evaluated per network call while the belief weighs them. */
const BELIEF_CHUNK = 256;

/**
 * Weigh the belief's hands for the position `engine` holds (prepared): every recorded decision of the
 * opponent's under every hand, through the network, the move's log-probability at each temperature
 * handed back to the C++. Returns the network rows spent.
 */
async function weighBelief(engine: WasmEngine, model: Model, width: number): Promise<number> {
  const D = engine.beliefDecisions();
  if (D === 0) return 0;
  const temps = engine.beliefTemperatures();
  const N = engine.beliefParticles();
  const act = engine.actionSize;
  const lp = new Float32Array(temps.length * D * N);
  for (let d = 0; d < D; d++) {
    const a = engine.beliefAction(d);
    for (let lo = 0; lo < N; lo += BELIEF_CHUNK) {
      const n = Math.min(BELIEF_CHUNK, N - lo);
      const rows = engine.beliefRows(d, lo, n, model.obsFeatures, width);
      const out = await model.run(rows.obs, rows.masks, n);
      for (let k = 0; k < n; k++) {
        const base = k * act;
        for (let t = 0; t < temps.length; t++) {
          // The move's masked log-softmax at temperature t, as ai/search/belief.device_log_probs.
          let mx = -Infinity;
          for (let i = 0; i < act; i++) if (rows.masks[base + i]) mx = Math.max(mx, out.logits[base + i] / temps[t]);
          let s = 0;
          for (let i = 0; i < act; i++) if (rows.masks[base + i]) s += Math.exp(out.logits[base + i] / temps[t] - mx);
          const v = rows.masks[base + a] ? out.logits[base + a] / temps[t] - mx - Math.log(s) : -Infinity;
          lp[(t * D + d) * N + lo + k] = Number.isFinite(v) ? v : -1e9;
        }
      }
    }
  }
  engine.beliefSetLogLikelihoods(lp);
  return D * N;
}

/**
 * Run the root on the position `engine` holds. `engine` must not be the one drawing the board
 * while this runs (the batch copies the position, but the forwards interleave with everything
 * else) -- both the worker and the inline fallback own a separate engine.
 * Returns null when no decision is open or the network has a single legal move.
 */
export async function rolloutRoot(engine: WasmEngine, model: Model, spec: RolloutSpec, seed: number): Promise<SearchResult | null> {
  if (model.meta.mergedInfluence) {
    throw new Error("the rollout root plays in the E4 action view; this model decides in the merged (E4.1) view");
  }
  if (engine.isTerminal()) return null;
  const decider = engine.decisionPlayer();
  if (decider === 0) return null;
  const t0 = performance.now();
  const act = engine.actionSize;
  // The root: the network's logits at the real position, from the decider's side.
  const mask = engine.mask(false);
  const root = await model.run(engine.observation(decider, model.obsFeatures), mask, 1);
  let rows = 1;
  const logits = root.logits.slice(0, act);
  const candidates = engine.rolloutCandidates(logits, mask, spec.k);
  if (candidates.length < 2) return null;

  // The engine's enum ordinals, looked up by name in its own export: the spec's names are the
  // search= grammar's, the ordinals the engine's.
  const stop = engine.layout.rollout_stops.indexOf(spec.stop);
  const leaf = engine.layout.rollout_leaves.indexOf(spec.leaf);
  if (stop < 0) throw new Error(`this engine has no rollout stop rule "${spec.stop}"`);
  if (leaf < 0) throw new Error(`this engine has no rollout leaf side "${spec.leaf}"`);
  engine.rolloutReset(spec.horizon, model.obsFeatures, seed, true, true, stop, leaf,
                      spec.keyed2 ? 2 : spec.keyed ? 1 : 0, spec.aligned, spec.box4);
  const width = engine.rolloutObsWidth();
  if (width !== model.meta.obsSize) {
    throw new Error(`the batch writes ${width} observation floats for this feature set; the model reads ${model.meta.obsSize}`);
  }
  let temperatureWeights: number[] | undefined;
  // Racing: the first `race` worlds for every candidate, the rest after the trailing ones leave.
  const first = spec.race || spec.worlds;
  let belief = false;
  if (spec.belief > 0 && engine.beliefParticles() === spec.belief && engine.beliefPrepare()) {
    rows += await weighBelief(engine, model, width);
    temperatureWeights = engine.beliefTemperatureWeights();
    engine.rolloutAddBelief(candidates, first, spec.worlds);
    belief = true;
  } else {
    if (spec.belief > 0 && engine.beliefParticles() !== spec.belief) {
      throw new Error(`the belief tracker holds ${engine.beliefParticles()} hands; the spec asks for ${spec.belief}`);
    }
    engine.rolloutAdd(candidates, first);
  }
  rows += await playOut(engine, model);
  if (spec.race) {
    engine.rolloutRace(0, spec.worlds - first, belief);
    rows += await playOut(engine, model);
  }
  const d = engine.rolloutDecide(0, candidates, logits, mask, spec.rule);
  const dropped = spec.race ? candidates.filter((_, c) => !engine.rolloutInRace(0, c)) : undefined;
  return { kind: "rollout", spec, candidates, pick: d.pick, worlds: d.worlds, q: d.q, lead: d.lead, lead_se: d.lead_se,
           dropped, rows, ms: Math.round(performance.now() - t0), temperature_weights: temperatureWeights };
}
