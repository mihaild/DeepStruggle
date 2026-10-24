import assert from "node:assert/strict";
import { WasmEngine } from "../../../web/ui/src/engine/wasm_engine";
import { RolloutRunner } from "../../../web/ui/src/analysis/rollout_runner";
import type { Model } from "../../../web/ui/src/analysis/model";
import type { OutMsg } from "../../../web/ui/src/analysis/rollout_worker";
import { DEFAULT_ROLLOUT } from "../../../web/ui/src/analysis/rollout_spec";

const base = process.argv[2];
const spec = { ...DEFAULT_ROLLOUT, k: 2, worlds: 2, horizon: 0, belief: 0 };

function gate() {
  let release!: () => void;
  const wait = new Promise<void>(resolve => { release = resolve; });
  return { wait, release };
}

async function within<T>(p: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  try {
    return await Promise.race([p, new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error("search did not settle")), 2000);
    })]);
  } finally { clearTimeout(timer!); }
}

// A real live engine changes while inline inference is suspended, on both success and error.
const live = await WasmEngine.load(base);
for (const fail of [false, true]) {
  for (const newGame of [false, true]) {
    live.newGame(1);
    const entered = gate(), resume = gate();
    let calls = 0;
    const model = {
      meta: { mergedInfluence: false, obsSize: live.obsSize }, obsFeatures: 0,
      async run(_obs: Float32Array, _masks: Uint8Array, rows: number) {
        if (++calls === 1) {
          entered.release();
          await resume.wait;
          if (fail) throw new Error("inference failed");
        }
        return { logits: new Float32Array(rows * live.actionSize), vWin: new Float32Array(rows) };
      },
    } as unknown as Model;
    const runner = new RolloutRunner(base);
    const searched = runner.search(live.snapshot(), model, "inline", spec, 1)
      .then(() => "ok", e => e.message);
    await within(entered.wait);
    if (newGame) live.newGame(27);
    else {
      const action = live.mask(false).findIndex(x => x > 0);
      assert.equal(live.step(live.decodeFlat(action), 0), null);
    }
    const afterUserAction = live.saveJson();
    resume.release();
    assert.equal(await within(searched), fail ? "inference failed" : "ok");
    assert.equal(live.saveJson(), afterUserAction);
  }
}

// Invalid imported positions are refused by the shared native/Wasm loader before publication.
const beforeLoad = live.saveJson();
for (const depth of [6, 255, 256, -1, "0", null]) {
  assert.throws(() => live.loadSaveJson(JSON.stringify({ ...JSON.parse(beforeLoad), ctx_stack_depth: depth })),
                /ctx_stack_depth/);
  assert.equal(live.saveJson(), beforeLoad);
}

class FakeWorker {
  static instances: FakeWorker[] = [];
  onmessage: ((event: { data: OutMsg }) => void) | null = null;
  onerror: ((event: { message: string }) => void) | null = null;
  sent: { type: string; id?: number }[] = [];
  terminated = false;
  constructor(..._args: unknown[]) { FakeWorker.instances.push(this); }
  postMessage(message: { type: string; id?: number }): void { this.sent.push(message); }
  terminate(): void { this.terminated = true; }
  emit(message: OutMsg): void { this.onmessage?.({ data: message }); }
  loaded(): void { this.emit({ type: "loaded", fingerprint: "test" }); }
}
Object.defineProperty(globalThis, "Worker", { value: FakeWorker, configurable: true });
const model = { bytes: new Uint8Array([1]), source: { kind: "file", name: "test" },
                meta: { actionSize: 220 } } as unknown as Model;
const runner = new RolloutRunner(base);
const search = (key: string) => runner.search(new Uint8Array(), model, key, spec, 1);
const tick = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };
const latest = () => FakeWorker.instances.at(-1)!;

runner.setModel(model, "A");
const a = latest();
const first = search("A").then(() => "ok", e => e.message);
await tick();
runner.setModel(model, "B");
const b = latest();
assert.notEqual(a, b);
assert.equal(a.terminated, true);
a.loaded();  // A late response must not initialize B.
assert.equal(await within(first), "the model changed");
const second = search("B");
await tick();
assert.equal(b.sent.filter(m => m.type === "search").length, 0);
b.loaded();
await tick();
const request = b.sent.find(m => m.type === "search")!;
assert.ok(request);
b.emit({ type: "result", id: request.id!, result: null });
assert.equal(await within(second), null);

// Errors during initialization settle the load waiter and release the serialized queue.
runner.setModel(model, "C");
const c = latest();
const failed = search("C").then(() => "ok", e => e.message);
await tick();
c.onerror?.({ message: "worker crashed" });
assert.equal(await within(failed), "worker crashed");
assert.equal(c.terminated, true);

// Disabling a model settles both initialization and work queued behind it.
runner.setModel(model, "D");
const loading = search("D").then(() => "ok", e => e.message);
await tick();
const queued = search("D").then(() => "ok", e => e.message);
runner.setModel(null, "");
assert.equal(await within(loading), "the model changed");
assert.match(await within(queued), /superseded/);

// Once loaded, a model replacement rejects requests already sent to the worker as well.
runner.setModel(model, "E");
latest().loaded();
const active = search("E").then(() => "ok", e => e.message);
await tick();
assert.ok(latest().sent.some(m => m.type === "search"));
runner.setModel(null, "");
assert.equal(await within(active), "the model changed");

console.log(JSON.stringify({ inline: true, invalidSave: true, lifecycle: true }));
