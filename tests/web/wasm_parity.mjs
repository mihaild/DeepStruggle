// Driven by tests/web/test_wasm_engine.py: runs the WebAssembly engine in Node and prints what
// the Python side compares against the native build.
//   node wasm_parity.mjs <ts_engine.mjs> selftest <games>
//   node wasm_parity.mjs <ts_engine.mjs> walk <actions.json>
//   node wasm_parity.mjs <ts_engine.mjs> macro <actions.json>
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [modulePath, mode, arg] = process.argv.slice(2);
const createModule = (await import(pathToFileURL(modulePath).href)).default;
const M = await createModule();

if (mode === "selftest") {
  const stepsPtr = M._malloc(4);
  const digest = M._ts_selftest(Number(arg), stepsPtr) >>> 0;
  console.log(JSON.stringify({ digest, steps: M.HEAPU32[stepsPtr >> 2], fingerprint: M.UTF8ToString(M._ts_fingerprint()) }));
  process.exit(0);
}

// The batch's live rows played to their horizons under a fixed synthetic policy, then their leaves
// valued by a fixed synthetic critic, every observation and mask into `hash`; `plies` and `leaves` count
// across calls (a raced position plays out twice), and the policy reads the running ply.
const playOuts = (hash, ACT) => {
  const W = M._ts_rollout_obs_width();
  const out = { plies: 0, leaves: 0 };
  out.run = () => {
    // The synthetic policy: logit[a] = ((a * 2654435761 + ply * 40503 + 7) mod 1000) / 1000.
    while (!M._ts_rollout_all_done()) {
      const n = M._ts_rollout_fill_live();
      hash.update(new Uint8Array(M.HEAPU8.buffer, M._ts_rollout_buf_obs(), n * W * 4).slice());
      hash.update(new Uint8Array(M.HEAPU8.buffer, M._ts_rollout_buf_masks(), n * ACT).slice());
      const lgPtr = M._malloc(4 * n * ACT);
      const lg = new Float32Array(M.HEAPF32.buffer, lgPtr, n * ACT);
      for (let j = 0; j < n; j++) {
        for (let a = 0; a < ACT; a++) lg[j * ACT + a] = (((a * 2654435761) % 4294967296 + out.plies * 40503 + 7) % 1000) / 1000;
      }
      M._ts_rollout_step(lgPtr);
      M._free(lgPtr);
      out.plies += 1;
      if (out.plies > 2000) break;
    }
    const n = M._ts_rollout_fill_leaves();
    hash.update(new Uint8Array(M.HEAPU8.buffer, M._ts_rollout_buf_obs(), n * W * 4).slice());
    const vPtr = M._malloc(4 * Math.max(1, n));
    const lv = new Float32Array(M.HEAPF32.buffer, vPtr, Math.max(1, n));
    for (let j = 0; j < n; j++) lv[j] = ((j * 31) % 17) / 17 - 0.5;
    M._ts_rollout_set_leaf_values(vPtr);
    M._free(vPtr);
    out.leaves += n;
  };
  return out;
};

// rollout: {seed, actions, k, worlds, horizon, batch_seed[, race]} -- walk to the position, then the
// rollout batch (bindings/rollout_batch.hpp) on its first k legal moves under a fixed synthetic
// policy and critic, so the Python module's batch can be compared row for row; with `race`, the
// position is then raced for that many more worlds and played again.
if (mode === "rollout") {
  const spec = JSON.parse(readFileSync(arg, "utf8"));
  M._ts_new_game(spec.seed >>> 0, 0);
  for (const [t, p, s, f] of spec.actions) {
    if (!M._ts_step(t, p, s, f, 0)) {
      console.log(JSON.stringify({ error: `refused: ${M.UTF8ToString(M._ts_last_error())}` }));
      process.exit(0);
    }
  }
  const ACT = M._ts_action_size();
  const mask = new Uint8Array(M.HEAPU8.buffer, M._ts_mask(0), ACT).slice();
  const cands = [];
  for (let a = 0; a < ACT && cands.length < spec.k; a++) if (mask[a]) cands.push(a);
  const candPtr = M._malloc(4 * cands.length);
  M.HEAP32.set(cands, candPtr >> 2);
  const stop = { boundaries: 0, turn: 1, later: 2, earlier: 3, predeal: 4, earlier_predeal: 5 }[spec.stop ?? "boundaries"];
  const leaf = { mover: 0, decider: 1, own: 2 }[spec.leaf ?? "mover"];
  if (!M._ts_rollout_reset(spec.horizon, 1, 1, 0, spec.batch_seed >>> 0, Math.floor(spec.batch_seed / 4294967296) >>> 0, stop, leaf,
                           Number(spec.keyed), spec.aligned ? 1 : 0, spec.box4 ? 1 : 0)) {
    console.log(JSON.stringify({ error: M.UTF8ToString(M._ts_last_error()) }));
    process.exit(0);
  }
  let rows = M._ts_rollout_add(candPtr, cands.length, spec.worlds);
  const hash = createHash("sha256");
  const play = playOuts(hash, ACT);
  play.run();
  let inRace = null;
  if (spec.race) {
    const added = M._ts_rollout_race(0, spec.race, 0);
    if (added < 0) {
      console.log(JSON.stringify({ error: M.UTF8ToString(M._ts_last_error()) }));
      process.exit(0);
    }
    rows += added;
    play.run();
    inRace = cands.map((_, c) => M._ts_rollout_in_race(0, c));
  }
  const values = Array.from(new Float32Array(M.HEAPF32.buffer, M._ts_rollout_values(), rows));
  const table = [];
  for (let r = 0; r < rows; r++) {
    table.push([M._ts_rollout_row_position(r), M._ts_rollout_row_world(r), M._ts_rollout_row_candidate(r),
                M._ts_rollout_row_done(r), M._ts_rollout_row_terminal(r)]);
  }
  // The decision under each rule, from synthetic root logits: logit[a] = ((a * 7919) mod 97) / 97.
  const rootPtr = M._malloc(4 * ACT), maskPtr = M._malloc(ACT);
  const root = new Float32Array(M.HEAPF32.buffer, rootPtr, ACT);
  for (let a = 0; a < ACT; a++) root[a] = ((a * 7919) % 97) / 97;
  M.HEAPU8.set(mask, maskPtr);
  const topPtr = M._malloc(4 * 3);
  const nTop = M._ts_rollout_candidates(rootPtr, maskPtr, 3, topPtr);
  const top = Array.from(M.HEAP32.subarray(topPtr >> 2, (topPtr >> 2) + nTop));
  const decisions = {};
  for (const rule of ["argmax", "z2", "kl0.05"]) {
    const rulePtr = M.stringToNewUTF8(rule);
    decisions[rule] = JSON.parse(M.UTF8ToString(M._ts_rollout_decide(0, candPtr, cands.length, rootPtr, maskPtr, rulePtr)));
    M._free(rulePtr);
  }
  console.log(JSON.stringify({ candidates: cands, rows, worlds_used: M._ts_rollout_worlds_used(0), plies: play.plies,
                               leaves: play.leaves,
                               refused: M._ts_rollout_refused(), digest: hash.digest("hex"), values, table,
                               top, decisions, in_race: inRace }));
  process.exit(0);
}

// belief: {seed, actions, particles, tracker_seed, worlds[, race, k, horizon, batch_seed]} -- the game's
// moves through the belief tracker (bindings/belief.hpp), each observed before it is applied, then at the
// last position: the recorded decisions' moves, a digest of every hypothesis row, the weights under
// synthetic log-likelihoods, the worlds' hands, and with `race` a raced rollout dealt from the belief.
if (mode === "belief") {
  const spec = JSON.parse(readFileSync(arg, "utf8"));
  M._ts_new_game(spec.seed >>> 0, 0);
  M._ts_belief_reset(spec.particles >>> 0, 1, spec.tracker_seed >>> 0, 0);
  let observed = 0;
  for (const [t, p, s, f] of spec.actions) {
    observed += M._ts_belief_observe(t, p, s, f);
    if (!M._ts_step(t, p, s, f, 0)) {
      console.log(JSON.stringify({ error: `refused: ${M.UTF8ToString(M._ts_last_error())}` }));
      process.exit(0);
    }
  }
  const has = M._ts_belief_prepare();
  const D = M._ts_belief_decisions();
  const P = spec.particles;
  const ACT = M._ts_action_size();
  const W = M._ts_obs_size();
  const actions = [];
  const hash = createHash("sha256");
  for (let d = 0; d < D; d++) {
    actions.push(M._ts_belief_action(d));
    if (M._ts_belief_rows(d, 0, P, 0) !== P) {
      console.log(JSON.stringify({ error: `rows: ${M.UTF8ToString(M._ts_last_error())}` }));
      process.exit(0);
    }
    hash.update(new Uint8Array(M.HEAPU8.buffer, M._ts_rollout_buf_obs(), P * W * 4).slice());
    hash.update(new Uint8Array(M.HEAPU8.buffer, M._ts_rollout_buf_masks(), P * ACT).slice());
  }
  const T = M._ts_belief_temperatures();
  let weights = [];
  let worlds = null;
  if (D > 0) {
    const lp = new Float32Array(T * D * P);
    for (let t = 0; t < T; t++)
      for (let d = 0; d < D; d++)
        for (let k = 0; k < P; k++)
          lp[(t * D + d) * P + k] = -(((k * 2654435761 + d * 40503 + t * 7) % 1000) / 250);
    const lpPtr = M._malloc(4 * lp.length);
    M.HEAPF32.set(lp, lpPtr >> 2);
    M._ts_belief_set_log_likelihoods(lpPtr);
    const wPtr = M._malloc(4 * T);
    M._ts_belief_temperature_weights(wPtr);
    weights = Array.from(new Float32Array(M.HEAPF32.buffer, wPtr, T));
    const h = M._ts_belief_worlds(spec.worlds);
    worlds = h ? Array.from(new Uint8Array(M.HEAPU8.buffer, h, spec.worlds * 110)) : null;
  }
  // With `race` and a belief here: the page's raced belief search on the first k legal moves -- every
  // candidate on the first `race` of the belief's worlds, then the rest of them for the ones left.
  let raced = null;
  if (spec.race && worlds) {
    const mask = new Uint8Array(M.HEAPU8.buffer, M._ts_mask(0), ACT).slice();
    const cands = [];
    for (let a = 0; a < ACT && cands.length < spec.k; a++) if (mask[a]) cands.push(a);
    const candPtr = M._malloc(4 * cands.length);
    M.HEAP32.set(cands, candPtr >> 2);
    M._ts_rollout_reset(spec.horizon, 1, 1, 0, spec.batch_seed >>> 0, 0, 0, 0, 0, 0, 0);
    let rows = M._ts_rollout_add_belief(candPtr, cands.length, spec.race, spec.worlds);
    const rhash = createHash("sha256");
    const play = playOuts(rhash, ACT);
    play.run();
    const added = M._ts_rollout_race(0, spec.worlds - spec.race, 1);
    if (added < 0) {
      console.log(JSON.stringify({ error: `race: ${M.UTF8ToString(M._ts_last_error())}` }));
      process.exit(0);
    }
    rows += added;
    play.run();
    raced = { candidates: cands, rows, worlds_used: M._ts_rollout_worlds_used(0), plies: play.plies,
              leaves: play.leaves, digest: rhash.digest("hex"),
              in_race: cands.map((_, c) => M._ts_rollout_in_race(0, c)) };
  }
  console.log(JSON.stringify({ observed, has, decisions: D, actions, digest: hash.digest("hex"), weights, worlds,
                               raced }));
  process.exit(0);
}

// The observation in a model's own view (ts_observation_features), ts_obs_size_for(f) floats.
const featureObs = (p, f) => {
  const width = M._ts_obs_size_for(f >>> 0);
  const ptr = M._ts_observation_features(p, f >>> 0);
  if (!width || !ptr) throw new Error(M.UTF8ToString(M._ts_last_error()));
  return new Uint8Array(M.HEAPU8.buffer, ptr, width * 4).slice();
};

// obs: {save, features} -- open a saved position and print the sha256 of each side's observation under
// each feature set, for the Python side to compare with extract_observation_features. (Digests, not the
// floats: process.exit does not wait for a long line on a pipe to drain.)
if (mode === "obs") {
  const spec = JSON.parse(readFileSync(arg, "utf8"));
  const ptr = M.stringToNewUTF8(spec.save);
  if (!M._ts_load_save_json(ptr)) {
    console.log(JSON.stringify({ error: M.UTF8ToString(M._ts_last_error()) }));
    process.exit(0);
  }
  const out = {};
  for (const f of spec.features) {
    for (const p of [1, -1]) out[`${f}:${p}`] = createHash("sha256").update(featureObs(p, f)).digest("hex");
  }
  console.log(JSON.stringify({ obs: out, unknown_width: M._ts_obs_size_for(1 << 30) }));
  process.exit(0);
}

// macro: {seed, actions} -- P42's view along a walk: the macro mask at every step, and for each until-control slot it
// offers, the save JSON after the page's expansion of it (ts_macro_points one-point placements in its country, as
// session.applyFlat applies it), the position restored after each.
if (mode === "macro") {
  const { seed, actions } = JSON.parse(readFileSync(arg, "utf8"));
  const ACT = M._ts_action_size(), BASE = M._ts_base_action_size();
  const NODE = JSON.parse(M.UTF8ToString(M._ts_action_layout_json())).offsets.node;
  const hash = createHash("sha256");
  const enc = new TextEncoder();
  let expansions = 0;
  M._ts_new_game(seed >>> 0, 0);
  const record = () => {
    const mask = new Uint8Array(M.HEAPU8.buffer, M._ts_mask(2), ACT).slice();
    hash.update(mask);
    for (let idx = BASE; idx < ACT; idx++) {
      if (!mask[idx]) continue;
      const before = M.stringToNewUTF8(M.UTF8ToString(M._ts_save_json()));
      const k = M._ts_macro_points(idx);
      const d = new Int32Array(M.HEAP32.buffer, M._ts_decode_flat(NODE + idx - BASE), 4).slice();
      for (let j = 0; j < k; j++) {
        if (!M._ts_step(d[0], d[1], d[2], d[3], 0)) throw new Error(`point ${j} of slot ${idx} refused`);
      }
      hash.update(enc.encode(`${idx}:${M.UTF8ToString(M._ts_save_json())}`));
      if (!M._ts_load_save_json(before)) throw new Error(M.UTF8ToString(M._ts_last_error()));
      M._free(before);
      expansions += 1;
    }
  };
  record();
  for (const [t, p, s, f] of actions) {
    if (!M._ts_step(t, p, s, f, 0)) {
      console.log(JSON.stringify({ error: `refused: ${M.UTF8ToString(M._ts_last_error())}` }));
      process.exit(0);
    }
    record();
  }
  console.log(JSON.stringify({ digest: hash.digest("hex"), expansions }));
  process.exit(0);
}

// walk: {seed, actions: [[type, primary, secondary, flags], ...], features: [obs_features sets]}
const { seed, actions, features = [] } = JSON.parse(readFileSync(arg, "utf8"));
const OBS = M._ts_obs_size();
const ACT = M._ts_action_size();
const hash = createHash("sha256");
const enc = new TextEncoder();
const record = () => {
  hash.update(enc.encode(M.UTF8ToString(M._ts_display_json())));
  hash.update(enc.encode(M.UTF8ToString(M._ts_save_json())));
  for (const p of [1, -1]) {
    hash.update(new Uint8Array(M.HEAPU8.buffer, M._ts_observation(p), OBS * 4).slice());
    for (const f of features) hash.update(featureObs(p, f));
  }
  hash.update(new Uint8Array(M.HEAPU8.buffer, M._ts_mask(0), ACT).slice());
};
M._ts_new_game(seed >>> 0, 0);
record();
let steps = 0;
for (const [t, p, s, f] of actions) {
  if (!M._ts_step(t, p, s, f, 0)) {
    console.log(JSON.stringify({ error: `refused at step ${steps}: ${M.UTF8ToString(M._ts_last_error())}` }));
    process.exit(0);
  }
  record();
  steps += 1;
}
console.log(JSON.stringify({ digest: hash.digest("hex"), steps, ending: M._ts_is_terminal() ? M.UTF8ToString(M._ts_ending_reason()) : null }));
