// Bundled by tests/web/test_log_view.py: the action log grouped into plays (game/log_view.ts),
// under node.
//   node log_view.mjs <public dir> <seed>
// Plays two turns of a game in the page's GameSession on the WebAssembly engine, by a fixed
// choice among the legal actions, and reports the grouped log next to what the board did, for
// the Python side to hold one to the other. Lines are rendered in English (game/log.ts).
import { pathToFileURL } from "node:url";
import { WasmEngine } from "../../../web/ui/src/engine/wasm_engine";
import { GameSession } from "../../../web/ui/src/game/session";
import { englishEntry, groupLog } from "../../../web/ui/src/game/log_view";
import { lineEnglish } from "../../../web/ui/src/game/log";
import { redactLogs } from "../../../web/ui/src/game/view";

const engine = await WasmEngine.load(pathToFileURL(process.argv[2]).href);
const s = new GameSession(engine);
s.newGame(Number(process.argv[3]));
let x = Number(process.argv[3]) >>> 0;
// A move that ends the game is taken back and another chosen, as game_stats.ts does: random
// play otherwise ends most games on DEFCON in the first turns. Taken back without a log line.
while (!engine.isTerminal() && engine.display().turn <= 2) {
  for (let tries = 0; tries < 50; tries++) {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    const legal: number[] = [];
    engine.mask(false).forEach((v, i) => { if (v) legal.push(i); });
    const why = s.applyFlat(legal[x % legal.length], false);
    if (why) throw new Error(why);
    if (!engine.isTerminal() || tries === 49) break;
    s.undo(false);
  }
}

const start = s.start!;
const end = engine.display();
const influence: Record<string, [number, number]> = {};
for (const c of Object.values(end.countries)) {
  const was = start.countries[c.name];
  influence[c.name] = [c.us_influence - was.us_influence, c.ussr_influence - was.ussr_influence];
}
const plays = s.logs.filter(l => l.kind === "play").length;
// The cards dealt into the US hand unseen by the USSR (a deal that empties the deck reveals both
// hands, so not every dealt card is one).
const unseenUs = s.logs.flatMap(l => l.details ?? []).map(lineEnglish)
  .map(d => /^  • Card #\d+ \((.*)\) moved: DRAW_DECK -> HAND_US_UNKNOWN$/.exec(d)?.[1]).filter(Boolean);
console.log(JSON.stringify({
  entries: groupLog(s.logs).map(englishEntry),
  ussr_view: groupLog(redactLogs(s.logs, "USSR")).map(englishEntry),
  influence, plays, unseen_us: unseenUs,
  vp: (end.victory_points ?? 0) - (start.victory_points ?? 0),
  terminal: engine.isTerminal(),
}));
