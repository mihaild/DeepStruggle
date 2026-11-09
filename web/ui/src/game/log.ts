/**
 * The action log's lines, kept as what they say rather than as text: each is a Message (the `log`
 * namespace) with its cards, countries and sides as ids, built where the words are decided
 * (game/describe.ts, game/log_view.ts, the session) and rendered only where the log is shown, in the
 * language of that moment. Code that needs to know what a line says -- grouping the clicks into
 * plays (log_view.ts), a player's view hiding the other side's cards (view.ts), the page's styling
 * (main.ts) -- reads its message id and parameters, never its words.
 *
 * A line's English is the text the log was before it kept Messages, byte for byte:
 * tests/web/test_describe_golden.py holds it to the Python session's, and what is kept as English --
 * a replay's `description`, the stats record's event texts -- is rendered with `lineEnglish`.
 *
 * Shown in the page, a card argument is the hoverable `span.log-card[data-card-id]` that main.ts
 * gives the card's tooltip (`lineHtml`), and every other argument is escaped: a line can come from a
 * replay file (its descriptions), which is anyone's text.
 */
import { escapeHtml } from "../html";
import { Message, MessageArg, MessageFormat, MessageKey, message, messageIfKnown, render, renderEnglish, sideName } from "../i18n";
import type { ActionLogItem } from "../types";

/**
 * The mark a line is indented under ("  • "): a change to the board, a die, the headline reveal, a
 * random discard, Missile Envy's exchange, a decision or an Event within a play. An action's own
 * line, and an entry's heading, have none.
 */
export type LogMark = "•" | "🎲" | "★" | "✂" | "⇄" | "▸";

/** One line of the log: what it says, and the mark it is indented under. */
export interface LogLine {
  msg: Message;
  mark?: LogMark;
}

export function logLine(mark: LogMark | undefined, key: MessageKey, params?: Record<string, MessageArg>): LogLine {
  return mark ? { mark, msg: message(key, params) } : { msg: message(key, params) };
}

/** Text the page did not word -- a replay file's description -- shown as it is. */
export function verbatim(text: string): Message {
  return message("log:verbatim", { text });
}

export type CardArg = { card: number; name: string };
export type CountryArg = { country: number; name: string };

export const cardArg = (card: number, name: string): CardArg => ({ card, name });
export const countryArg = (country: number, name: string): CountryArg => ({ country, name });
export const sideArg = (side: string): MessageArg => ({ side });

export function signed(v: number): string {
  return v > 0 ? `+${v}` : `${v}`;
}

// ---- reading a line's parameters -------------------------------------------------------------

export function param(m: Message, k: string): MessageArg | undefined {
  return m.params?.[k];
}
export function numParam(m: Message, k: string): number {
  const v = param(m, k);
  return typeof v === "number" ? v : Number(v);
}
export function strParam(m: Message, k: string): string {
  const v = param(m, k);
  if (typeof v === "string" || typeof v === "number") return String(v);
  if (v && typeof v === "object" && "side" in v) return v.side;
  if (v && typeof v === "object" && "region" in v) return v.region;
  return "";
}
export function isCardArg(a: MessageArg | undefined): a is CardArg {
  return typeof a === "object" && a !== null && "card" in a;
}
export function isCountryArg(a: MessageArg | undefined): a is CountryArg {
  return typeof a === "object" && a !== null && "country" in a;
}
function isMessage(a: MessageArg | undefined): a is Message {
  return typeof a === "object" && a !== null && "key" in a;
}

/** The cards a message names, in the order its parameters list them, nested messages included. */
export function cardsIn(m: Message): CardArg[] {
  const out: CardArg[] = [];
  for (const a of Object.values(m.params ?? {})) {
    if (isCardArg(a)) out.push(a);
    else if (isMessage(a)) out.push(...cardsIn(a));
  }
  return out;
}

// ---- rendering -------------------------------------------------------------------------------

function withMark(l: LogLine, words: string): string {
  return l.mark ? `  ${l.mark} ${words}` : words;
}

/** A line in English, as the log's text was: for what is kept as English and for the tests. */
export function lineEnglish(l: LogLine): string {
  return withMark(l, renderEnglish(l.msg));
}

/** A line in the current language, as plain text. */
export function lineText(l: LogLine): string {
  return withMark(l, render(l.msg));
}

/** `inner` is text, escaped here. */
function cardSpan(id: number, inner: string): string {
  return `<span class="log-card" data-card-id="${id}">${escapeHtml(inner)}</span>`;
}

const HTML: MessageFormat = { text: escapeHtml, card: cardSpan };

/** A message as HTML in the current language: its cards hoverable, every other argument escaped. */
export function messageHtml(m: Message): string {
  return render(m, HTML);
}

/** A line as HTML in the current language (`messageHtml`), under its mark. */
export function lineHtml(l: LogLine): string {
  return withMark(l, messageHtml(l.msg));
}

/** Who a log entry is by, as the log shows it: a side in the current language, or the system's. */
export function playerLabel(player: string): string {
  if (player === "US" || player === "USSR") return sideName(player);
  const m = messageIfKnown(`log:player.${player}`);
  return m ? render(m) : player;
}

/** A log item with its words in English, as the log's items read before they kept Messages: for
 *  the tests and for debugging (`window.__wb.englishLog()`). */
export type EnglishLogItem = Omit<ActionLogItem, "text" | "details" | "title" | "mode"> & {
  text: string;
  details?: string[];
  title?: string;
  mode?: string;
};

export function englishItem(item: ActionLogItem): EnglishLogItem {
  const { text, details, title, mode, ...rest } = item;
  return {
    ...rest, text: renderEnglish(text),
    ...(details ? { details: details.map(lineEnglish) } : {}),
    ...(title ? { title: renderEnglish(title) } : {}),
    ...(mode ? { mode: renderEnglish(mode) } : {}),
  };
}
