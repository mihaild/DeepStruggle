// Bundled by tests/web/test_seat_rules.py: what the page's game offers a side at an opponent's card
// whose Event cannot fire, under the rules it is seated with.
//   node seat_rules.mjs <public dir> <save.json>
// The save is the USSR choosing a card with NATO in hand before Warsaw Pact Formed or the Marshall
// Plan. NATO is chosen, then both sides are seated as the page seats a person (every rule bit,
// WasmEngine.currentRules) and as an unseated pre-bit model (0), and the play-mode mask and the
// session's offered choices come back for each.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { WasmEngine } from "../../../web/ui/src/engine/wasm_engine";
import { GameSession } from "../../../web/ui/src/game/session";

const NATO = 21;
const engine = await WasmEngine.load(pathToFileURL(process.argv[2]).href);
const s = new GameSession(engine);
s.loadPosition(readFileSync(process.argv[3], "utf8"), "nato");
const why = s.apply({ decision_type: 1, primary_id: NATO, secondary_id: 0, flags: 0 });
if (why) throw new Error(why);
const out: Record<string, unknown> = { rules: engine.currentRules };
for (const [name, features] of [["person", engine.currentRules], ["legacy", 0]] as const) {
  const changed = [engine.seat(1, features), engine.seat(-1, features)];
  const offered = s.state().legal_actions;
  out[name] = { changed, mask: Array.from(engine.mask(false)), type: offered.decision_type, valid: offered.valid_ids };
}
console.log(JSON.stringify(out));
