// Bundled by tests/web/test_describe_golden.py (esbuild) and run with node: replays the golden
// games through the WebAssembly engine and the TypeScript action log, rendered in English
// (game/log.ts lineEnglish), and prints the mismatches.
//   node describe_golden.mjs <web/ui/public/> <describe_golden.json.gz>
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { pathToFileURL } from "node:url";
import { WasmEngine } from "../../../web/ui/src/engine/wasm_engine";
import { describeActionAndDeltas } from "../../../web/ui/src/game/describe";
import { lineEnglish } from "../../../web/ui/src/game/log";

const [publicDir, goldenPath] = process.argv.slice(2);
const engine = await WasmEngine.load(pathToFileURL(publicDir).href);
const golden = JSON.parse(gunzipSync(readFileSync(goldenPath)).toString("utf8"));

// Intended differences from the Python session, which the port corrects:
// - it never logged MilOps (it read keys the display state does not have), nor the Space Race
//   track;
// - a card chosen while an event resolves read as a play, or in the headline as a headline commit
//   with a "Headlines Simultaneously Revealed" pair for it; the port says what happened to the card,
//   so for those steps the action's own lines are left out of the comparison;
// - the port says Five Year Plan's and Terrorism's random discards on lines of their own ("  ✂ "),
//   and Missile Envy's exchange with how the taken card resolves ("  ⇄ ").
// The golden used to be corrected for the Olympic sponsor's +2 as well: the engine's roll1 held it
// (roll1 = die + 2) when the earlier golden was recorded, so the original counted it twice. The
// engine now records the die alone, and the golden re-recorded on it reads the Olympics as the port.
const comparable = (lines: string[], eventChoice: boolean) => {
  let out = lines.filter(l => !l.includes(" MilOps: ") && !l.startsWith("  • Space Race: ")
    && !l.startsWith("  ✂ ") && !l.startsWith("  ⇄ "));
  if (eventChoice) out = out.slice(1).filter(l => !l.startsWith("  ★ "));
  return out;
};

let steps = 0;
let mismatches = 0;
const examples: unknown[] = [];
for (const g of golden.games) {
  engine.newGame(g.seed);
  g.actions.forEach((a: number[], i: number) => {
    const action = { decision_type: a[0], primary_id: a[1], secondary_id: a[2], flags: a[3] };
    const before = engine.display();
    const refused = engine.step(action, 0);
    if (refused) throw new Error(`seed ${g.seed} step ${i}: engine refused (${refused})`);
    const ctx = before.decision_context;
    // ... and an event's branch choice, which it logged as "chooses option: N" (game/branches.ts).
    const eventChoice = (action.decision_type === 1 && (ctx?.resolving_card ?? 0) !== 0 && ctx?.resolving_card !== 250)
      || action.decision_type === 6;
    const got = comparable(describeActionAndDeltas(engine, before, engine.display(), action).map(lineEnglish), eventChoice);
    const want = comparable(g.lines[i] as string[], eventChoice);
    steps += 1;
    if (JSON.stringify(got) !== JSON.stringify(want)) {
      mismatches += 1;
      if (examples.length < 5) examples.push({ seed: g.seed, step: i, want, got });
    }
  });
}
console.log(JSON.stringify({ steps, mismatches, examples }));
