/**
 * What an event's branch choice (CHOOSE_BRANCH) means, for the HUD's buttons and the action log.
 *
 * The engine asks for a branch in six events (engine/src/events/*.cpp, card_dispatcher.cpp), and the
 * branch is a number whose meaning is the event's: Chernobyl's is a region, Summit's and How I
 * Learned to Stop Worrying's the DEFCON level to set (Summit offers the current level and one either
 * side -- choosing the current one leaves DEFCON alone). The page used to show "Option 3" for those.
 *
 * The words are Messages (the `branches` namespace), rendered where they are shown, in the language
 * of that moment.
 */
import { GameState } from "../types";
import { engineLayout, namesByValue, titleCase } from "../engine/layout";
import { Message, MessageArg, message } from "../i18n";

export interface BranchMeta {
  /** The button's title. */
  title: Message;
  /** The button's description. */
  desc: Message;
  /** The log's verb phrase: "<player> <verb> (<event>)". */
  verb: Message;
}

/**
 * A region by the engine's `Region` value (Chernobyl's branch, the region head) under its English
 * name, the rules' ("Middle East"), which is what `regionName` (src/i18n) translates.
 */
export function regionEnglishName(region: number): string | undefined {
  const L = engineLayout();
  const name = L ? namesByValue(L.regions)[region] : undefined;
  return name === undefined ? undefined : titleCase(name);
}

/** A branch's three messages, all under one id: `branches:<id>.title`, `.desc`, `.verb`. */
function branch(id: string, params?: Record<string, MessageArg>): BranchMeta {
  const m = (part: string) => message(`branches:${id}.${part}` as Message["key"], params);
  return { title: m("title"), desc: m("desc"), verb: m("verb") };
}

/** By the event's ts::card_ids name: each branch's id in the `branches` namespace. */
const FIXED: Record<string, Record<number, string>> = {
  WARSAW_PACT_FORMED: { 0: "warsawPact.remove", 1: "warsawPact.add" },
  OLYMPIC_GAMES: { 0: "olympics.participate", 1: "olympics.boycott" },
  WARGAMES: { 0: "wargames.end", 1: "wargames.continue" },
};

function defconLevel(level: number, current: number): BranchMeta {
  const how = level === current ? "unchanged" : level > current ? "improve" : "degrade";
  const meta = branch(`defcon.${how}`, { level, current });
  return level === 1 ? { ...meta, desc: message("branches:defcon.endsGame") } : meta;
}

/** The meaning of branch `branchIndex` of `card`'s event in `state`, or null for an event it does not know. */
export function branchMeta(card: number, branchIndex: number, state: GameState): BranchMeta | null {
  const L = engineLayout();
  if (!L || !card) return null;
  const key = Object.keys(L.card_ids).find(k => L.card_ids[k] === card);
  if (key === "CHERNOBYL") {
    const english = regionEnglishName(branchIndex);
    const region: MessageArg = english !== undefined ? { region: english }
      : message("branches:chernobyl.regionNumber", { n: branchIndex });
    return branch("chernobyl", { region });
  }
  if (key === "SUMMIT" || key === "HOW_I_LEARNED_TO_STOP_WORRYING") return defconLevel(branchIndex, state.defcon);
  const id = key ? FIXED[key]?.[branchIndex] : undefined;
  return id ? branch(id) : null;
}
