/**
 * The engine's names shown to a person: its enums (`enums:phase.HEADLINE`, `enums:decision.SELECT_CARD`,
 * `enums:resolution.OPS_COUP`, `enums:opMode.COUP`), its regions, and why a game ended
 * (`endings:<id>`). The engine's name stays the value everywhere -- in the state, a save, the stats
 * record -- and is worded only where it is shown.
 *
 * English shows each name as the page showed it before there was a catalog: a decision type or a
 * play mode as the engine spells it, a phase in lower case ("action round"). A name the catalog
 * lacks (an engine newer than the page) is shown that way too.
 */
import { titleCase } from "../engine/layout";
import { Message, MessageArg, messageIfKnown, regionName, render } from "./index";

export type EnumKind = "phase" | "decision" | "resolution" | "opMode" | "region";

/** How the page showed `name` before the catalog, and still shows a name the catalog lacks. */
function asShown(kind: EnumKind, name: string): string {
  if (kind === "phase") return name.toLowerCase().replace("_", " ");
  if (kind === "region") return titleCase(name);
  return name;
}

function enumMessage(kind: Exclude<EnumKind, "region">, name: string): Message | null {
  return name ? messageIfKnown(`enums:${kind}.${name}`) : null;
}

/**
 * An enum name as a Message parameter. A region is named as everywhere else, by its English name
 * (`game:regions.<English>`), which is the engine's name title-cased: "MIDDLE_EAST" is "Middle East".
 */
export function enumArg(kind: EnumKind, name: string): MessageArg {
  if (kind === "region") return { region: asShown(kind, name) };
  return enumMessage(kind, name) ?? asShown(kind, name);
}

/** An enum name in the current language: `enumLabel("decision", "SELECT_CARD")`. */
export function enumLabel(kind: EnumKind, name: string): string {
  if (kind === "region") return regionName(asShown(kind, name));
  const m = enumMessage(kind, name);
  return m ? render(m) : asShown(kind, name);
}

/** The engine's `ending_reason()` (bindings/state_json.cpp), the value a result keeps, by catalog id. */
const ENDINGS: Record<string, string> = {
  "20 VP": "vp20",
  "Europe Control": "europeControl",
  "wargames": "wargames",
  "final scoring": "finalScoring",
  "DEFCON 1 (own decision)": "defconOwn",
  "DEFCON 1 (opponent decision)": "defconOpponent",
};

function endingMessage(reason: string): Message | null {
  const id = ENDINGS[reason];
  return id ? messageIfKnown(`endings:${id}`) : null;
}

/** Why a game ended as a Message parameter; the engine's text itself for a reason the catalog lacks. */
export function endingArg(reason: string): MessageArg {
  return endingMessage(reason) ?? reason;
}

/** Why a game ended, in the current language: `endingLabel(result.reason)`. */
export function endingLabel(reason: string): string {
  const m = endingMessage(reason);
  return m ? render(m) : reason;
}
