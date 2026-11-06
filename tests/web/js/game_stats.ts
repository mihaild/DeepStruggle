// Bundled by tests/web/test_game_stats.py: a game's record for the stats database, under node.
//   node game_stats.mjs <public dir>
// Plays two games in the page's GameSession on the WebAssembly engine -- a clean one (the model
// auto-plays the US, the human the USSR in the USSR view, with two take-backs) and a messy one
// (the human moves for the model, a forced die, the Developer view, a model switch, a debug
// override) -- builds each one's submission, replays the clean record into a fresh session (a
// page reload), and puts the submissions through the stats Worker's handler with a fake D1.
import { pathToFileURL } from "node:url";
import { WasmEngine } from "../../../web/ui/src/engine/wasm_engine";
import { GameSession } from "../../../web/ui/src/game/session";
import { MoveHow, Submission, buildSubmission, decodeLog } from "../../../web/ui/src/game/record";
import { D1Database, D1Statement, handle, invalid } from "../../../web/stats_worker/src/handler";

const engine = await WasmEngine.load(pathToFileURL(process.argv[2]).href);
const BUILD = { commit: "0123456789abcdef0123456789abcdef01234567", dirty: false, engine_fingerprint: engine.fingerprint };
const SHA_A = "a".repeat(64);
const SHA_B = "b".repeat(64);
const MODEL_A = { label: "model A", source: "hf:x/y@main:a.onnx", checkpoint: "ck-a" };
const MODEL_B = { label: "model B", source: "hf:x/y@main:b.onnx", checkpoint: "ck-b" };

/** A deterministic choice among the legal flat actions. */
function chooser(seed: number) {
  let x = seed >>> 0;
  return () => {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    const m = engine.mask(false);
    const legal: number[] = [];
    m.forEach((v, i) => { if (v) legal.push(i); });
    return legal[x % legal.length];
  };
}

function side(): "US" | "USSR" | "" {
  const p = engine.decisionPlayer();
  return p > 0 ? "US" : p < 0 ? "USSR" : "";
}

/**
 * A move of `pick`'s, but none that ends the game before turn 10 while another is legal: random
 * play otherwise ends most games on DEFCON in the first turns, and these must run long enough to
 * roll forced dice and switch models. A move taken back here is the test's, not the player's.
 */
function move(s: GameSession, pick: () => number, die: number, how: MoveHow): void {
  for (let tries = 0; tries < 50; tries++) {
    const why = s.applyFlat(pick(), false, die, how);
    if (why) throw new Error(why);
    if (!engine.isTerminal() || engine.display().turn >= 10 || tries === 49) return;
    s.undo(false);
  }
}

function play(s: GameSession, seed: number, messy: boolean): void {
  const pick = chooser(seed);
  s.newGame(seed);
  let human = 0;
  let model = 0;
  for (let n = 0; n < 20000 && !engine.isTerminal(); n++) {
    const mover = side();
    // The messy game switches model a while in, as the page does when another is loaded.
    const b = messy && n >= 200;
    const devView = messy && mover === "USSR" && human >= 9 && human < 13;
    s.setSettings({ view: devView ? "DEV" : "USSR", auto: "US", model: b ? SHA_B : SHA_A, search: "policy" },
                  b ? MODEL_B : MODEL_A);
    if (mover === "US") {
      // The model's side. In the messy game the human makes three of its moves.
      model++;
      const how: MoveHow = messy && model >= 30 && model < 33 ? { source: "human", analysis: false } : { source: "auto", analysis: false };
      move(s, pick, 0, how);
      continue;
    }
    human++;
    // Forced sixes in the messy game: whichever of these moves rolls, rolls a six.
    const die = messy && human >= 20 && human < 60 ? 6 : 0;
    const how: MoveHow = { source: messy && human === 5 ? "suggested" : "human", analysis: messy && human === 6 };
    move(s, pick, die, how);
    if ((human === 25 || human === 40) && !engine.isTerminal()) {
      // The player takes the move back and plays another.
      s.undo();
      move(s, pick, 0, { source: "human", analysis: false });
    }
    if (messy && human === 70) {
      const vp = engine.display().victory_points;
      const refused = s.setVp(vp);
      if (refused) throw new Error(refused);
    }
  }
  if (!s.result) throw new Error("the game did not end");
}

async function submission(s: GameSession, login: string): Promise<Submission> {
  return buildSubmission(s.record(), s.result!, engine.saveJson(), BUILD, { login, rating: 1650 }, "a note");
}

const session = new GameSession(engine);
play(session, 7771, false);
const clean = await submission(session, "Alice");
const cleanSave = engine.saveJson();
const cleanRecord = session.record();

// A reload: the stored record played over again lands on the same position, with its id and tags.
const reloaded = new GameSession(engine);
reloaded.replayRecord(cleanRecord, null, false);
const resumed = {
  same_position: engine.saveJson() === cleanSave,
  same_id: reloaded.id === cleanRecord.id,
  same_tags: JSON.stringify(reloaded.tags) === JSON.stringify(cleanRecord.steps.map(x => x.tag)),
  resume_event: reloaded.events[reloaded.events.length - 1]?.kind,
};

play(session, 5150, true);
const messy = await submission(session, "Bob");

const decoded = await decodeLog(clean.log);
const roundtrip = decoded.seed === 7771 && decoded.steps.length === clean.n_steps
  && JSON.stringify(decoded.steps) === JSON.stringify(cleanRecord.steps);

// The Worker, with a D1 that keeps rows in memory.
const rows = new Map<string, unknown[]>();
const db: D1Database = {
  prepare(sql: string): D1Statement {
    let vals: unknown[] = [];
    const st: D1Statement = {
      bind(...v: unknown[]) { vals = v; return st; },
      async run() {
        if (!sql.includes("INSERT OR IGNORE")) return { meta: {} };
        if (rows.has(vals[0] as string)) return { meta: { changes: 0 } };
        rows.set(vals[0] as string, vals);
        return { meta: { changes: 1 } };
      },
      async first<T>() {
        return { n: [...rows.values()].filter(r => r[18] === vals[0]).length } as T;
      },
    };
    return st;
  },
};
const env = { DB: db, ALLOWED_ORIGINS: "https://mihaild.github.io,http://localhost:5173", RATE_PER_HOUR: "3" };
async function post(body: unknown, origin = "https://mihaild.github.io") {
  const res = await handle(new Request("https://stats.example/games", {
    method: "POST", headers: { Origin: origin, "Content-Type": "application/json" }, body: JSON.stringify(body),
  }), env);
  return { status: res.status, body: await res.json().catch(() => null), cors: res.headers.get("Access-Control-Allow-Origin") };
}
const noLogin = { ...clean, playdek: { login: "", rating: 1650 } };
const responses = {
  first: await post(clean),
  again: await post(clean),
  other_origin: await post({ ...messy, id: "f0000000-0000-0000-0000-000000000000" }, "https://evil.example"),
  no_login: await post(noLogin),
  messy: await post(messy),
  // Two stored from this address, plus this one: the third of three an hour, then refused.
  third: await post({ ...messy, id: "c0000000-0000-0000-0000-000000000001" }),
  fourth: await post({ ...messy, id: "c0000000-0000-0000-0000-000000000002" }),
};
const stored = rows.get(clean.id)!;

console.log(JSON.stringify({
  fingerprint: engine.fingerprint, clean, messy, resumed, roundtrip,
  invalid_clean: invalid(clean), invalid_no_login: invalid(noLogin),
  responses, stored_body_is_submission: stored[19] === JSON.stringify(clean),
}));
