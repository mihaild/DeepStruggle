/**
 * Readable names for flat action indices -- the port of `ActionEncoder.get_action_name`
 * (bindings/action_encoder.py), with the offsets taken from the engine's own layout rather than
 * copied, and names for the two P17 heads (DEFCON value, region) that the Python version leaves
 * as "UnknownAction".
 */
import { ActionLayout, WasmEngine } from "../engine/wasm_engine";

const MODES = ["EVENT", "SPACE", "OPS_INFLUENCE", "OPS_COUP", "OPS_REALIGN"];
const OP_MODES = ["INFLUENCE", "COUP", "REALIGN"];

export function actionName(engine: WasmEngine, idx: number): string {
  const L: ActionLayout = engine.layout;
  const o = L.offsets;
  if (idx === L.confirm_done_index) return "CONFIRM_DONE / PASS";
  if (idx < o.play_mode) return `SelectCard #${idx + 1} (${engine.cardName(idx + 1)})`;
  if (idx < o.roll_die) {
    const mode = idx - o.play_mode;
    const name = MODES[mode] ?? String(mode);
    // The last three slots are also the deferred Ops choice; without a node to ask, say both.
    if (idx >= o.op_mode) return `Resolution: ${name} / OpMode: ${OP_MODES[idx - o.op_mode]}`;
    return `Resolution: ${name}`;
  }
  if (idx === o.roll_die) return "RollDie";
  if (idx < o.node) return `Unassigned #${idx}`;
  if (idx < o.branch) return `PointNode #${idx - o.node} (${engine.countryName(idx - o.node)})`;
  if (idx < L.confirm_done_index) return `Branch #${idx - o.branch}`;
  if (idx >= o.defcon_value && idx < o.region) return `DEFCON value #${idx - o.defcon_value + 1}`;
  if (idx >= o.region && idx < L.size) return `Region #${idx - o.region}`;
  return `UnknownAction #${idx}`;
}
