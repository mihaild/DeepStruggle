// Bundled by tests/web/test_describe_golden.py (esbuild) and run with node: replays the golden
// games through the WebAssembly engine and the TypeScript action log, and prints the mismatches.
//   node describe_golden.mjs <web/ui/public/> <describe_golden.json.gz>
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { pathToFileURL } from "node:url";
import { WasmEngine } from "../../../web/ui/src/engine/wasm_engine";
import { describeActionAndDeltas } from "../../../web/ui/src/game/describe";

const [publicDir, goldenPath] = process.argv.slice(2);
const engine = await WasmEngine.load(pathToFileURL(publicDir).href);
const golden = JSON.parse(gunzipSync(readFileSync(goldenPath)).toString("utf8"));

// The Python session never logged MilOps (it read keys the display state does not have); the
// port does. Those lines are the one intended difference.
const comparable = (lines: string[]) => lines.filter(l => !l.includes(" MilOps: "));

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
    const got = comparable(describeActionAndDeltas(engine, before, engine.display(), action));
    const want = g.lines[i] as string[];
    steps += 1;
    if (JSON.stringify(got) !== JSON.stringify(want)) {
      mismatches += 1;
      if (examples.length < 5) examples.push({ seed: g.seed, step: i, want, got });
    }
  });
}
console.log(JSON.stringify({ steps, mismatches, examples }));
