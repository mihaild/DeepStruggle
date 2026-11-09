/**
 * Readable names for flat action indices -- the port of `ActionEncoder.get_action_name`
 * (bindings/action_encoder.py), with the offsets and enum names taken from the engine's own layout rather than
 * copied, and names for the two P17 heads (DEFCON value, region) that the Python version leaves
 * as "UnknownAction".
 *
 * `actionName` is that port, English and kept as data; `actionLabel` is the same name as a Message
 * (the `actions` namespace), for the page to show in its language.
 */
import { ActionLayout, WasmEngine } from "../engine/wasm_engine";
import { GameState } from "../types";
import { namesByValue } from "../engine/layout";
import { Message, message, renderEnglish } from "../i18n";
import { enumArg } from "../i18n/enums";
import { branchMeta, regionEnglishName } from "./branches";

/** `state`, the position the action is taken in, names an event's branch by what it means
 * (game/branches.ts); without it a branch is a number, a region and a DEFCON level still named. */
export function actionName(engine: WasmEngine, idx: number, state?: GameState): string {
  const L: ActionLayout = engine.layout;
  const o = L.offsets;
  if (idx === L.confirm_done_index) return "CONFIRM_DONE / PASS";
  if (idx < o.play_mode) return `SelectCard #${idx + 1} (${engine.cardName(idx + 1)})`;
  if (idx < o.roll_die) {
    const mode = idx - o.play_mode;
    const name = namesByValue(L.resolutions)[mode] ?? String(mode);
    // The last three slots are also the deferred Ops choice; without a node to ask, say both.
    if (idx >= o.op_mode) return `Resolution: ${name} / OpMode: ${namesByValue(L.op_modes)[idx - o.op_mode]}`;
    return `Resolution: ${name}`;
  }
  if (idx === o.roll_die) return "RollDie";
  if (idx < o.node) return `Unassigned #${idx}`;
  if (idx < o.branch) return `PointNode #${idx - o.node} (${engine.countryName(idx - o.node)})`;
  const card = state?.decision_context?.resolving_card ?? 0;
  if (idx < L.confirm_done_index) {
    const meta = state ? branchMeta(card, idx - o.branch, state) : null;
    // The branch's title in English: this name is data (`actionLabel` is the one shown).
    return meta ? renderEnglish(meta.title) : `Branch #${idx - o.branch}`;
  }
  if (idx >= o.defcon_value && idx < o.region) {
    const level = idx - o.defcon_value + 1;
    const meta = state ? branchMeta(card, level, state) : null;
    return meta ? renderEnglish(meta.title) : `Set DEFCON to ${level}`;
  }
  if (idx >= o.region && idx < L.size) return regionEnglishName(idx - o.region) ?? `Region #${idx - o.region}`;
  return `UnknownAction #${idx}`;
}

/** `actionName` as a Message: the same words in English, the page's language where it is shown. */
export function actionLabel(engine: WasmEngine, idx: number, state?: GameState): Message {
  const L: ActionLayout = engine.layout;
  const o = L.offsets;
  if (idx === L.confirm_done_index) return message("actions:confirmDone");
  if (idx < o.play_mode) return message("actions:selectCard", { n: idx + 1, card: { card: idx + 1, name: engine.cardName(idx + 1) } });
  if (idx < o.roll_die) {
    const mode = idx - o.play_mode;
    const resolution = enumArg("resolution", namesByValue(L.resolutions)[mode] ?? String(mode));
    if (idx >= o.op_mode) {
      const opMode = enumArg("opMode", String(namesByValue(L.op_modes)[idx - o.op_mode]));
      return message("actions:resolutionOrOpMode", { resolution, opMode });
    }
    return message("actions:resolution", { resolution });
  }
  if (idx === o.roll_die) return message("actions:rollDie");
  if (idx < o.node) return message("actions:unassigned", { idx });
  if (idx < o.branch) {
    const n = idx - o.node;
    return message("actions:pointNode", { n, country: { country: n, name: engine.countryName(n) } });
  }
  const card = state?.decision_context?.resolving_card ?? 0;
  if (idx < L.confirm_done_index) {
    const meta = state ? branchMeta(card, idx - o.branch, state) : null;
    return meta ? meta.title : message("actions:branch", { n: idx - o.branch });
  }
  if (idx >= o.defcon_value && idx < o.region) {
    const level = idx - o.defcon_value + 1;
    const meta = state ? branchMeta(card, level, state) : null;
    return meta ? meta.title : message("actions:setDefcon", { level });
  }
  if (idx >= o.region && idx < L.size) {
    const english = regionEnglishName(idx - o.region);
    return english !== undefined ? message("actions:region", { region: { region: english } })
      : message("actions:regionNumber", { n: idx - o.region });
  }
  return message("actions:unknown", { idx });
}
