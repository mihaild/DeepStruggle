// Bundled by tests/web/test_recent_rolls.py: which dice the map's roll tray shows, under node.
//   node recent_rolls.mjs <case.json>   with {start, steps, now, tray}
// Prints the rolls game/rolls.ts finds recent at `now`, and the tray's view (roll_tray.ts) and the
// log's dice lines (game/describe.ts) of each roll in `tray`.
import { readFileSync } from "node:fs";
import { describeRoll as logLines } from "../../../web/ui/src/game/describe";
import { lineEnglish } from "../../../web/ui/src/game/log";
import { recentRolls } from "../../../web/ui/src/game/rolls";
import { WasmEngine } from "../../../web/ui/src/engine/wasm_engine";
import { describeRoll } from "../../../web/ui/src/roll_tray";
import { ReplayStep } from "../../../web/ui/src/replay_controls";
import { GameState, RecentRoll } from "../../../web/ui/src/types";

const { start, steps, now, tray } = JSON.parse(readFileSync(process.argv[2], "utf8")) as {
  start: GameState | null; steps: ReplayStep[]; now: GameState; tray: RecentRoll[];
};

console.log(JSON.stringify({
  recent: recentRolls(start, steps, steps.length, now),
  tray: tray.map(describeRoll),
  // The rolls checked here name no card or country by id: the log needs no engine for them.
  log: tray.map(r => logLines({} as WasmEngine, now, r, r.roller ?? "NONE").map(lineEnglish)),
}));
