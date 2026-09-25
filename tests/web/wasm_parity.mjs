// Driven by tests/web/test_wasm_engine.py: runs the WebAssembly engine in Node and prints what
// the Python side compares against the native build.
//   node wasm_parity.mjs <ts_engine.mjs> selftest <games>
//   node wasm_parity.mjs <ts_engine.mjs> walk <actions.json>
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

// walk: {seed, actions: [[type, primary, secondary, flags], ...]}
const { seed, actions } = JSON.parse(readFileSync(arg, "utf8"));
const OBS = M._ts_obs_size();
const ACT = M._ts_action_size();
const hash = createHash("sha256");
const enc = new TextEncoder();
const record = () => {
  hash.update(enc.encode(M.UTF8ToString(M._ts_display_json())));
  hash.update(enc.encode(M.UTF8ToString(M._ts_save_json())));
  for (const p of [1, -1]) {
    hash.update(new Uint8Array(M.HEAPU8.buffer, M._ts_observation(p), OBS * 4).slice());
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
