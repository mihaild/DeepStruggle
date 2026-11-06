// Bundled by tests/web/test_event_tracks.py: the action log of a headline whose two events move VP,
// under node.
//   node event_tracks.mjs <public dir> <cases.json>
// Each case is a saved position at the headline, the two cards to commit, and the clicks after
// them (`then`: [decision_type, primary_id, forced die]; `finish`: then the first legal action
// until the headline is over). Each is opened in the page's GameSession on the WebAssembly
// engine, the clicks are made, and the last click's log entry, the steps' event records and the
// log as the page shows it (game/log_view.ts) come back for the Python side to check.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { WasmEngine } from "../../../web/ui/src/engine/wasm_engine";
import { GameSession } from "../../../web/ui/src/game/session";
import { groupLog } from "../../../web/ui/src/game/log_view";

interface Case { save: string; us: number; ussr: number; then?: [number, number, number][]; finish?: boolean }

const engine = await WasmEngine.load(pathToFileURL(process.argv[2]).href);
const cases: Record<string, Case> = JSON.parse(readFileSync(process.argv[3], "utf8"));
const out: Record<string, unknown> = {};
for (const [name, c] of Object.entries(cases)) {
  const s = new GameSession(engine);
  s.loadPosition(c.save, name);
  for (let k = 0; k < 2; k++) {
    const card = engine.decisionPlayer() > 0 ? c.us : c.ussr;
    const why = s.apply({ decision_type: 1, primary_id: card, secondary_id: 0, flags: 0 });
    if (why) throw new Error(`${name}: ${why}`);
  }
  for (const [type, primary, die] of c.then ?? []) {
    const why = s.apply({ decision_type: type, primary_id: primary, secondary_id: 0, flags: 0 }, die);
    if (why) throw new Error(`${name}: ${why}`);
  }
  const clicks = s.logs.length;
  while (c.finish && engine.display().current_phase_name === "HEADLINE") {
    const first = engine.mask(false).findIndex(v => v);
    const why = s.applyFlat(first, false);
    if (why) throw new Error(`${name}: ${why}`);
  }
  const last = s.logs[clicks - 1];
  out[name] = {
    text: last.text, details: last.details ?? [], event_tracks: s.steps.map(st => st.event_tracks ?? []),
    entries: groupLog(s.logs),
  };
}
console.log(JSON.stringify(out));
