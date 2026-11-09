// Bundled by tests/web/test_log_html.py: the action log as the page puts it into HTML
// (game/log.ts lineHtml), under node.
//   node log_html.mjs <public dir> <seed>
// Plays two turns of a game in the page's GameSession on the WebAssembly engine, by a fixed
// choice among the legal actions, groups its log as the page does, and reports every heading and
// line: its English text, its HTML, and the cards its message names -- then the same lines' HTML in
// the pseudo-language, which marks every message it renders.
import { pathToFileURL } from "node:url";
import { WasmEngine } from "../../../web/ui/src/engine/wasm_engine";
import { GameSession } from "../../../web/ui/src/game/session";
import { groupLog } from "../../../web/ui/src/game/log_view";
import { LogLine, cardsIn, lineEnglish, lineHtml } from "../../../web/ui/src/game/log";
import { PSEUDO, setLanguage } from "../../../web/ui/src/i18n";

const engine = await WasmEngine.load(pathToFileURL(process.argv[2]).href);
const s = new GameSession(engine);
s.newGame(Number(process.argv[3]));
let x = Number(process.argv[3]) >>> 0;
// A move that ends the game is taken back and another chosen, as log_view.ts does.
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

// A heading is a line with no mark.
const lines: LogLine[] = groupLog(s.logs).flatMap(e => [{ msg: e.title }, ...e.lines]);
const game = lines.map(l => ({
  text: lineEnglish(l), html: lineHtml(l), cards: cardsIn(l.msg).map(c => [c.card, c.name]), mark: l.mark ?? "",
}));
await setLanguage(PSEUDO);
console.log(JSON.stringify({ game, pseudo: lines.map(lineHtml) }));
