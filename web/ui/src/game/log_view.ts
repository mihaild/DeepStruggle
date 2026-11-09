/**
 * The action log as the page shows it: one entry per play, not one per click.
 *
 * The session logs every click (game/session.ts), and undo, the game's record and the describe
 * golden (tests/web/test_describe_golden.py) all work on those. But a play is several clicks --
 * the card, its mode, then each Influence point or target -- and read click by click it came
 * apart: a card's discard showed on its last placement, each point was an entry of its own, and
 * a deal was fifteen lines. Here the clicks are grouped into plays -- a card played in an action
 * round, the headline, one side's setup, the turn's end -- and each play is told once: the
 * decisions within it and their dice and Events in the order they happened, then what the whole
 * play changed on the board, with cards' moves in words and a deal or a reshuffle on one line.
 *
 * Everything here reads the lines' message ids and parameters (game/log.ts) -- a country's
 * Influence summed by its id and side, a card's move by its `from` and `to` -- and writes Messages
 * in turn, which the page words in its language when it shows them. It runs on the log as the view
 * may read it (game/view.ts redactLogs): grouping only merges and rewords lines that survived, so
 * nothing a view hides comes back.
 */
import { ActionLogItem, GameState, LogKind, MicroAction } from "../types";
import type { ReplayStep } from "../replay_controls";
import { CARDS_METADATA } from "../metadata";
import { effectNameMessage } from "../cards_view";
import { WasmEngine } from "../engine/wasm_engine";
import { defconLine, describeAction, describeDeltas, describeEventTracks, describeRoll, influenceLine,
         milOpsLine, spaceLine, vpLine, vpOf } from "./describe";
import { namesByValue } from "../engine/layout";
import { Message, MessageArg, message, messageIfKnown, renderEnglish } from "../i18n";
import { CardArg, LogLine, cardArg, isCardArg, isCountryArg, lineEnglish, logLine, numParam, param, sideArg,
         strParam, verbatim } from "./log";
import { actionLabel } from "./names";

const SIDE = new Map(CARDS_METADATA.map(c => [c.id, c.side]));

/** What a click is, for grouping: set on each click's log entry by the session. */
export interface ClickInfo {
  kind: LogKind;
  /** The entry's heading when the click starts one. Never set for a headline still face down. */
  title?: Message;
  /** A play's mode ("Ops: Influence"), for the heading of the play it belongs to. */
  mode?: Message;
  /** Cards whose discard the heading already implies: the card played, the two headlines. */
  cards?: number[];
  /** The Event the click answered, 0 for none: a play's changes are told Event by Event. */
  resolving: number;
  /** That Event's card by the engine's name. */
  resolving_name?: string;
}

function other(p: string): string {
  return p === "US" ? "USSR" : "US";
}

/** An Ops spend in words (`log:ops.<OpMode>`), by the engine's OpMode names and the Resolution
 *  names that spend Ops; the number for a name the catalog lacks. */
function opsWords(name: string | undefined, value: number): MessageArg {
  return (name ? messageIfKnown(`log:ops.${name.replace(/^OPS_/, "")}`) : null) ?? value;
}

export function clickInfo(engine: WasmEngine, before: GameState | undefined,
                          action: Required<MicroAction>, player = "NONE"): ClickInfo {
  const resolving = before?.decision_context?.resolving_card ?? 0;
  const named = resolving >= 1 && resolving <= 110;
  return { ...classify(engine, before, action, player), resolving,
           ...(named ? { resolving_name: engine.cardName(resolving) } : {}) };
}

function classify(engine: WasmEngine, before: GameState | undefined,
                  action: Required<MicroAction>, player: string): Omit<ClickInfo, "resolving"> {
  const ctx = before?.decision_context;
  const p = ctx?.decision_player ?? player;
  const resolving = ctx?.resolving_card ?? 0;
  const phase = before?.current_phase_name ?? "SETUP";
  const d = action.decision_type;
  const card = action.primary_id;
  const name = (c: number) => cardArg(c, engine.cardName(c));
  const L = engine.layout;
  const DT = L.decision_types;
  if (phase === "SETUP") return { kind: "setup", title: message("log:entry.setup", { side: sideArg(p) }) };
  if (d === DT.SELECT_CARD && resolving === 0 && phase === "HEADLINE") {
    const theirs = (p === "US" ? before?.headline_ussr_card : before?.headline_us_card) ?? 0;
    // The first commit is face down to the other side: its words are the view's (redactLogs).
    if (!theirs) return { kind: "headline" };
    const us = p === "US" ? card : theirs;
    const ussr = p === "US" ? theirs : card;
    return { kind: "headline", title: message("log:entry.headlines", { us: name(us), usId: us, ussr: name(ussr), ussrId: ussr }),
             cards: [us, ussr] };
  }
  if (d === DT.SELECT_CARD && resolving === 0 && phase === "ACTION_ROUND") {
    return { kind: "play", title: message("log:entry.plays", { side: sideArg(p), card: name(card), id: card }), cards: [card] };
  }
  const confirm = (action.flags & 0x80) !== 0 || card === 255;
  if ((d === DT.SELECT_PLAY_MODE || d === DT.SELECT_OP_MODE) && resolving === 0 && !confirm) {
    if (d === DT.SELECT_OP_MODE) {
      return { kind: "mode", mode: message("log:mode.ops", { ops: opsWords(namesByValue(L.op_modes)[card], card) }) };
    }
    // An opponent's card played for Operations still has its Event, before or after the Ops.
    const side = SIDE.get(ctx?.pending_op_card ?? 0);
    const theirs = side === other(p).toLowerCase();
    const opp = sideArg(other(p));
    if (card === L.resolutions.EVENT) {
      return { kind: "mode", mode: theirs ? message("log:mode.eventThenOps", { side: opp }) : message("log:mode.event") };
    }
    if (card === L.resolutions.SPACE) return { kind: "mode", mode: message("log:mode.space") };
    const ops = opsWords(namesByValue(L.resolutions)[card], card);
    return { kind: "mode", mode: theirs ? message("log:mode.opsThenEvent", { ops, side: opp }) : message("log:mode.ops", { ops }) };
  }
  if (d === DT.POINT_NODE) return { kind: "point" };
  return { kind: "step" };
}

/** One entry of the log as shown. `items` are the clicks (and system lines) it was made from. */
export interface LogEntry {
  turn: number;
  ar: number;
  phase: string;
  player: string;
  title: Message;
  lines: LogLine[];
  /** Per line, the step it can be matched to: the click it came from, or for a change merged
   * over several clicks the one click that made it -- null when more than one did. */
  steps: (number | null)[];
  /** The step that started the entry (its card, its headline commit, the turn's end). */
  step: number | null;
  vp_delta: number;
  /** Index into the item list of the entry's first and last item. */
  first: number;
  last: number;
}

/** An entry with its words in English, as the log's entries read before they kept Messages: for
 *  the tests. */
export function englishEntry(e: LogEntry): Omit<LogEntry, "title" | "lines"> & { title: string; lines: string[] } {
  return { ...e, title: renderEnglish(e.title), lines: e.lines.map(lineEnglish) };
}

const INFLUENCE = new Set(["log:change.influenceUs", "log:change.influenceUssr", "log:change.influenceBoth"]);
const HAND = /^HAND_(US|USSR)_(KNOWN|UNKNOWN)$/;
/** The Space Race's attempts-this-turn counters: bookkeeping, the dice line says the attempt. */
const SPACE_ATTEMPT = /^SPACE_(US|USSR)_ATTEMPT_\d$/;

export function isInfluenceLine(l: LogLine): boolean {
  return INFLUENCE.has(l.msg.key);
}
export function isCardMove(l: LogLine): boolean {
  return l.msg.key === "log:change.cardMoved";
}

/** A line that says the score moved: the VP track's change, or an Event's VP. */
export function movesVp(l: LogLine): boolean {
  const k = l.msg.key;
  return k === "log:change.vp" || k === "log:eventTrack.vp" || k === "log:eventTrack.both";
}

/** A line that speaks of VP: `movesVp`, and a die whose outcome is put in VP (the Olympics' sponsor
 *  winning or a tie, a war won). */
export function mentionsVp(l: LogLine): boolean {
  if (movesVp(l)) return true;
  const result = param(l.msg, "result");
  const outcome = typeof result === "object" && "key" in result ? result.key : "";
  return outcome === "log:roll.olympicsSponsor" || outcome === "log:roll.olympicsTie" || outcome === "log:roll.warSucceeded";
}

type Span = [number, number];
/** A line and the one step it can be matched to, if there is one. */
type Line = [LogLine, number | null];

/** A country as a line names it, and the key it is summed under: its id. */
function countryKey(a: MessageArg | undefined): string {
  return isCountryArg(a) ? `#${a.country}` : JSON.stringify(a);
}

/** A play's changes to the board, each from its value before the play to its value after. */
class Board {
  private influence = new Map<string, { country: MessageArg; sides: Map<string, Span> }>();
  private defcon?: Span;
  private vp?: Span;
  private milOps = new Map<string, Span>();
  private space = new Map<string, Span>();
  private effects = new Map<string, number>();
  private cards = new Map<number, { card: CardArg; from: string; to: string }>();
  /** The steps that changed each thing, keyed as `lines` tells them. */
  private where = new Map<string, Set<number>>();

  private note(key: string, step: number): void {
    const s = this.where.get(key) ?? new Set<number>();
    s.add(step);
    this.where.set(key, s);
  }

  /** The one step that changed all of `keys`, or null. */
  private stepOf(...keys: string[]): number | null {
    const all = new Set<number>();
    for (const k of keys) for (const n of this.where.get(k) ?? []) all.add(n);
    return all.size === 1 ? [...all][0] : null;
  }

  private static widen(m: Map<string, Span>, key: string, a: number, b: number): void {
    const s = m.get(key);
    if (s) s[1] = b; else m.set(key, [a, b]);
  }

  /** Takes a board line of `step` into the play's changes; false for a line that is not one. */
  take(line: LogLine, step: number): boolean {
    const m = line.msg;
    const k = m.key;
    if (INFLUENCE.has(k)) {
      const country = param(m, "country") ?? "";
      const key = countryKey(country);
      this.note(`influence:${key}`, step);
      const c = this.influence.get(key) ?? { country, sides: new Map<string, Span>() };
      this.influence.set(key, c);
      if (k !== "log:change.influenceUssr") Board.widen(c.sides, "US", numParam(m, "usFrom"), numParam(m, "usTo"));
      if (k !== "log:change.influenceUs") Board.widen(c.sides, "USSR", numParam(m, "ussrFrom"), numParam(m, "ussrTo"));
    } else if (k === "log:change.defcon") {
      this.note("defcon", step);
      this.defcon = this.defcon ? [this.defcon[0], numParam(m, "to")] : [numParam(m, "from"), numParam(m, "to")];
    } else if (k === "log:change.vp") {
      this.note("vp", step);
      const to = vpOf(param(m, "to"));
      this.vp = this.vp ? [this.vp[0], to] : [vpOf(param(m, "from")), to];
    } else if (k === "log:change.milOps") {
      const side = strParam(m, "side");
      this.note(`milops:${side}`, step);
      Board.widen(this.milOps, side, numParam(m, "from"), numParam(m, "to"));
    } else if (k === "log:change.space") {
      const side = strParam(m, "side");
      this.note(`space:${side}`, step);
      Board.widen(this.space, side, numParam(m, "from"), numParam(m, "to"));
    } else if (k === "log:change.effectActivated" || k === "log:change.effectCancelled") {
      const flag = strParam(m, "flag");
      this.note(`effect:${flag}`, step);
      this.effects.set(flag, (this.effects.get(flag) ?? 0) + (k === "log:change.effectActivated" ? 1 : -1));
    } else if (k === "log:change.cardMoved") {
      const id = numParam(m, "id");
      const card = param(m, "card");
      this.note(`card:${id}`, step);
      const c = this.cards.get(id);
      if (c) c.to = strParam(m, "to");
      else this.cards.set(id, { card: isCardArg(card) ? card : cardArg(id, String(card)), from: strParam(m, "from"), to: strParam(m, "to") });
    } else {
      return false;
    }
    return true;
  }

  /** The play's changes, as lines; `implied` are cards whose discard the heading already says. */
  lines(implied: number[]): Line[] {
    const out: Line[] = [];
    for (const [key, { country, sides }] of this.influence) {
      const us = sides.get("US");
      const ussr = sides.get("USSR");
      const usMoved = us && us[0] !== us[1] ? us : null;
      const ussrMoved = ussr && ussr[0] !== ussr[1] ? ussr : null;
      if (usMoved || ussrMoved) out.push([influenceLine(country, usMoved, ussrMoved), this.stepOf(`influence:${key}`)]);
    }
    if (this.defcon && this.defcon[0] !== this.defcon[1]) {
      out.push([defconLine(this.defcon[0], this.defcon[1]), this.stepOf("defcon")]);
    }
    if (this.vp && this.vp[0] !== this.vp[1]) out.push([vpLine(this.vp[0], this.vp[1]), this.stepOf("vp")]);
    for (const [side, s] of this.milOps) if (s[0] !== s[1]) out.push([milOpsLine(side, s[0], s[1]), this.stepOf(`milops:${side}`)]);
    for (const [side, s] of this.space) if (s[0] !== s[1]) out.push([spaceLine(side, s[0], s[1]), this.stepOf(`space:${side}`)]);
    for (const [flag, net] of this.effects) {
      if (net === 0 || SPACE_ATTEMPT.test(flag)) continue;
      const effect = effectNameMessage(flag) ?? flag;
      out.push([logLine("•", net > 0 ? "log:change.inEffect" : "log:change.noLongerInEffect", { effect }), this.stepOf(`effect:${flag}`)]);
    }
    out.push(...this.cardLines(implied));
    return out;
  }

  private cardLines(implied: number[]): Line[] {
    const out: Line[] = [];
    const draws: Record<string, CardArg[]> = { US: [], USSR: [] };
    const drawKeys: Record<string, string[]> = { US: [], USSR: [] };
    const reshuffledKeys: string[] = [];
    const addedKeys: string[] = [];
    let reshuffled = 0;
    let added = 0;
    for (const [id, c] of this.cards) {
      const { from, to } = c;
      if (from === to || to === "HEADLINE_COMMITTED") continue;
      // Said already: the card played or headlined going to the discard, a random discard's card,
      // a card Missile Envy exchanged. A card removed from the game is still said.
      if (implied.includes(id) && (to === "DISCARD_PILE" || HAND.test(to))) continue;
      const card = { card: c.card, id };
      const hf = HAND.exec(from);
      const ht = HAND.exec(to);
      const key = `card:${id}`;
      if (from === "DRAW_DECK" && ht) { draws[ht[1]].push(c.card); drawKeys[ht[1]].push(key); continue; }
      if (to === "DRAW_DECK" && from === "DISCARD_PILE") { reshuffled++; reshuffledKeys.push(key); continue; }
      if (to === "DRAW_DECK" && from === "UNAVAILABLE") { added++; addedKeys.push(key); continue; }
      let says: Message;
      if (ht && hf && hf[1] === ht[1]) {
        const sides = { side: sideArg(ht[1]), other: sideArg(other(ht[1])) };
        says = message(ht[2] === "KNOWN" ? "log:card.nowKnown" : "log:card.noLongerKnown", { ...card, ...sides });
      } else if (ht) {
        says = message(hf ? "log:card.takenInto" : "log:card.toHand", { ...card, side: sideArg(ht[1]) });
      } else {
        says = messageIfKnown(`log:card.to.${to}`, card) ?? message("log:card.to.other", { ...card, to });
      }
      out.push([{ mark: "•", msg: says }, this.stepOf(key)]);
    }
    for (const side of ["US", "USSR"]) {
      const n = draws[side].length;
      if (n) {
        out.push([logLine("•", "log:card.draws", { side: sideArg(side), count: n, cards: list(draws[side]) }),
                  this.stepOf(...drawKeys[side])]);
      }
    }
    if (added) out.push([logLine("•", "log:card.newEra", { n: added }), this.stepOf(...addedKeys)]);
    if (reshuffled) out.push([logLine("•", "log:card.reshuffled", { n: reshuffled }), this.stepOf(...reshuffledKeys)]);
    return out;
  }
}

/** Several names as one parameter: "A, B, C". */
function list(items: MessageArg[]): MessageArg {
  return items.slice(0, -1).reduceRight<MessageArg>((rest, first) => message("log:list", { first, rest }), items[items.length - 1]);
}

/** A play's modes as one: "Event, then Ops: Coup". */
function modesOf(modes: Message[]): MessageArg {
  return modes.slice(0, -1).reduceRight<MessageArg>((next, first) => message("log:mode.then", { first, next }), modes[modes.length - 1]);
}

/** A click line that its own effects already say: a placement, a coup's or a target's. */
function toldByItsEffects(item: ActionLogItem): boolean {
  return (item.details ?? []).some(d => isInfluenceLine(d) || d.mark === "🎲" || d.msg.key.startsWith("log:eventTrack."));
}

/** Starts a new entry: a play, a headline commit, one side's setup, and every line the session
 * writes itself (the turn's end, game over, an undo, a debug override). */
function startsEntry(item: ActionLogItem, current: LogEntry | null, currentKind: LogKind | "system" | null): boolean {
  if (!current || currentKind === "system") return true;
  if (!item.kind || item.kind === "play" || item.kind === "headline") return true;
  if (item.kind === "setup") return !(currentKind === "setup" && current.player === item.player);
  return false;
}

/** The log as entries, one per play. */
export function groupLog(items: ActionLogItem[]): LogEntry[] {
  const entries: LogEntry[] = [];
  let entry: LogEntry | null = null;
  let kind: LogKind | "system" | null = null;
  let modes: Message[] = [];
  let anchor: ActionLogItem | null = null;
  let board = new Board();
  let implied: number[] = [];
  // The Event the clicks so far answer. The board's changes are told where it changes, so a play
  // reads in the order it happened: a headline's Red Scare before the other headline's
  // De-Stalinization, an opponent's card's Ops before its Event.
  let segment = 0;
  // Where the cards ended up is the play's conclusion, told once at its end: the engine moves an
  // Event's card when the Event fires, before the Event's own decisions.
  let cards = new Board();

  const add = (line: LogLine, step: number | null) => {
    if (!entry) return;
    entry.lines.push(line);
    entry.steps.push(step);
  };
  const addAll = (lines: Line[]) => {
    for (const [line, step] of lines) add(line, step);
  };

  const flush = () => {
    addAll(board.lines(implied));
    board = new Board();
  };

  const close = () => {
    if (!entry || !anchor) return;
    if (modes.length) entry.title = message("log:entry.withModes", { title: entry.title, modes: modesOf(modes) });
    addAll(board.lines(implied));
    addAll(cards.lines(implied));
  };

  items.forEach((item, i) => {
    if (startsEntry(item, entry, kind)) {
      close();
      kind = item.kind ?? "system";
      anchor = item;
      modes = [];
      board = new Board();
      cards = new Board();
      implied = [...(item.cards ?? [])];
      segment = item.resolving ?? 0;
      entry = {
        turn: item.turn, ar: item.ar, phase: item.phase, player: item.player,
        title: item.title ?? item.text, lines: [], steps: [], vp_delta: 0, first: i, last: i,
        // A click's, or the turn's end (the click that ran into it); not the session's own lines.
        step: item.kind || item.phase === "TURN_END" ? item.step_index : null,
      };
      // A replay's first step has no position before it to read its change from.
      if (kind === "setup" && !toldByItsEffects(item)) add({ mark: "▸", msg: item.text }, item.step_index);
      // Both headlines are one entry, neither side's.
      if (kind === "headline" && item.title) entry.player = "SYSTEM";
      // A click that does not start a play of its own heads one with its own words.
      if (kind === "point" || kind === "step" || kind === "mode") entry.title = item.text;
      entries.push(entry);
    } else if (entry) {
      entry.last = i;
      const resolving = item.resolving ?? 0;
      if (resolving !== segment) {
        flush();
        segment = resolving;
        // Which Event the lines below are, unless the play is that card's own Event.
        const ownEvent = kind === "play" && modes.length === 1 && modes[0].key === "log:mode.event";
        if (resolving && item.resolving_name && !ownEvent) {
          add(logLine("▸", "log:entry.eventOf", { card: cardArg(resolving, item.resolving_name), id: resolving, side: sideArg(item.player) }), null);
        }
      }
      if (item.kind === "mode" && kind === "play" && item.mode) {
        // An opponent's Event first, then the Ops: the mode chosen after names the Ops.
        const last = modes[modes.length - 1];
        if (last && last.key === "log:mode.eventThenOps" && item.mode.key === "log:mode.ops") {
          modes[modes.length - 1] = message("log:mode.eventThenOpsAs", { side: param(last, "side") ?? "", ops: param(item.mode, "ops") ?? "" });
        } else {
          modes.push(item.mode);
        }
      } else if ((item.kind === "setup" || item.kind === "point") && toldByItsEffects(item)) {
        // Said by the Influence or the dice below.
      } else {
        add({ mark: "▸", msg: item.text }, item.step_index);
      }
    }
    if (!entry) return;
    entry.vp_delta += item.vp_delta ?? 0;
    for (const d of item.details ?? []) {
      if (item.title && d.msg.key === "log:headline.revealed") continue;
      if (isCardMove(d) ? cards.take(d, item.step_index) : board.take(d, item.step_index)) continue;
      add(d, item.step_index);
      // A random discard and Missile Envy's exchange say where their cards went.
      if (d.mark === "✂" || d.mark === "⇄") {
        const card = param(d.msg, "card");
        if (isCardArg(card)) implied.push(card.card);
      }
    }
  });
  close();
  return entries;
}

/**
 * What a replay's step did, as its log line: the choice's name when the step carries its flat
 * action (`actionLabel`, the words Python's `get_action_name` wrote as its description), the
 * action's own line when it carries only the click and the position before it (a replay the page
 * exported, whose description is that line in English), and the file's description, as it is,
 * for anything else.
 */
function replayText(engine: WasmEngine, step: ReplayStep, before: GameState | undefined, after: GameState | undefined,
                    action: Required<MicroAction>): Message {
  const a = step.action ?? {};
  const flat = typeof a.flat_action_idx === "number" ? a.flat_action_idx : a.flat_action;
  if (typeof flat === "number" && flat >= 0) return actionLabel(engine, flat, before);
  if (before?.decision_context && after && typeof a.decision_type === "number") {
    const own = describeAction(engine, before, after, action)[0];
    if (own) return own.msg;
  }
  return verbatim(step.description || `Action type=${action.decision_type}`);
}

/**
 * A replay's steps as log items, for groupLog: the step's action, the dice it recorded, what its
 * Events did, and the board's changes between the step's snapshot and the one before. A replay
 * file holds only its snapshots, so a turn's end is part of the last step of the turn.
 */
export function replayItems(engine: WasmEngine, steps: ReplayStep[]): ActionLogItem[] {
  return steps.map((step, i) => {
    const before = i > 0 ? steps[i - 1].state_snapshot : undefined;
    const after = step.state_snapshot;
    const player = step.player || "SYSTEM";
    const a = step.action ?? {};
    const action: Required<MicroAction> = {
      decision_type: a.decision_type ?? 0, primary_id: a.primary_id ?? 0,
      secondary_id: a.secondary_id ?? 0, flags: a.flags ?? 0,
    };
    const details: LogLine[] = [];
    const rolls = step.rolls ?? (after?.die_roll && after.die_roll.type !== "NONE" ? [after.die_roll] : []);
    for (const r of rolls) details.push(...describeRoll(engine, before ?? after, r, player));
    if (before && after) {
      details.push(...describeEventTracks(before, after, step.event_tracks ?? after.event_tracks));
      details.push(...describeDeltas(engine, before, { ...after, die_roll: undefined }, player,
                                     action.decision_type === engine.layout.decision_types.SELECT_CARD ? action.primary_id : 0));
    }
    return {
      step_index: i + 1, turn: step.turn, ar: step.ar, phase: step.phase, player,
      text: replayText(engine, step, before, after, action), details,
      vp_delta: before && after ? (after.victory_points ?? 0) - (before.victory_points ?? 0) : 0,
      ...clickInfo(engine, before, action, player),
    };
  });
}
