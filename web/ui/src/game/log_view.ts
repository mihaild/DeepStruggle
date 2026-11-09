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
 * It runs on the log as the view may read it (game/view.ts redactLogs): grouping only merges and
 * rewords lines that survived, so nothing a view hides comes back.
 */
import { ActionLogItem, GameState, LogKind, MicroAction } from "../types";
import type { ReplayStep } from "../replay_controls";
import { CARDS_METADATA } from "../metadata";
import { effectEnglishName } from "../cards_view";
import { WasmEngine } from "../engine/wasm_engine";
import { describeDeltas, describeEventTracks, describeRoll, vpStr } from "./describe";
import { namesByValue } from "../engine/layout";

/** An Ops spend in words, by the engine's OpMode names and the Resolution names that spend Ops. */
const OPS_WORDS: Record<string, string> = {
  INFLUENCE: "Influence", COUP: "Coup", REALIGN: "Realignment",
  OPS_INFLUENCE: "Influence", OPS_COUP: "Coup", OPS_REALIGN: "Realignment",
};
const SIDE = new Map(CARDS_METADATA.map(c => [c.id, c.side]));

/** What a click is, for grouping: set on each click's log entry by the session. */
export interface ClickInfo {
  kind: LogKind;
  /** The entry's heading when the click starts one. Never set for a headline still face down. */
  title?: string;
  /** A play's mode ("Ops: Influence"), for the heading of the play it belongs to. */
  mode?: string;
  /** Cards whose discard the heading already implies: the card played, the two headlines. */
  cards?: number[];
  /** The Event the click answered, 0 for none: a play's changes are told Event by Event. */
  resolving: number;
  resolving_name?: string;
}

function other(p: string): string {
  return p === "US" ? "USSR" : "US";
}

export function clickInfo(engine: WasmEngine, before: GameState | undefined,
                          action: Required<MicroAction>, player = "NONE"): ClickInfo {
  const resolving = before?.decision_context?.resolving_card ?? 0;
  const named = resolving >= 1 && resolving <= 110;
  return { ...classify(engine, before, action, player), resolving,
           ...(named ? { resolving_name: `${engine.cardName(resolving)} (#${resolving})` } : {}) };
}

function classify(engine: WasmEngine, before: GameState | undefined,
                  action: Required<MicroAction>, player: string): Omit<ClickInfo, "resolving"> {
  const ctx = before?.decision_context;
  const p = ctx?.decision_player ?? player;
  const resolving = ctx?.resolving_card ?? 0;
  const phase = before?.current_phase_name ?? "SETUP";
  const d = action.decision_type;
  const card = action.primary_id;
  const name = (c: number) => `${engine.cardName(c)} (#${c})`;
  const L = engine.layout;
  const DT = L.decision_types;
  if (phase === "SETUP") return { kind: "setup", title: `${p} places its setup Influence` };
  if (d === DT.SELECT_CARD && resolving === 0 && phase === "HEADLINE") {
    const theirs = (p === "US" ? before?.headline_ussr_card : before?.headline_us_card) ?? 0;
    // The first commit is face down to the other side: its words are the view's (redactLogs).
    if (!theirs) return { kind: "headline" };
    const us = p === "US" ? card : theirs;
    const ussr = p === "US" ? theirs : card;
    return { kind: "headline", title: `Headlines: US ${name(us)} vs USSR ${name(ussr)}`, cards: [us, ussr] };
  }
  if (d === DT.SELECT_CARD && resolving === 0 && phase === "ACTION_ROUND") {
    return { kind: "play", title: `${p} plays ${name(card)}`, cards: [card] };
  }
  const confirm = (action.flags & 0x80) !== 0 || card === 255;
  if ((d === DT.SELECT_PLAY_MODE || d === DT.SELECT_OP_MODE) && resolving === 0 && !confirm) {
    if (d === DT.SELECT_OP_MODE) {
      return { kind: "mode", mode: `Ops: ${OPS_WORDS[namesByValue(L.op_modes)[card] ?? ""] ?? card}` };
    }
    // An opponent's card played for Operations still has its Event, before or after the Ops.
    const side = SIDE.get(ctx?.pending_op_card ?? 0);
    const theirs = side === other(p).toLowerCase();
    const opp = other(p);
    if (card === L.resolutions.EVENT) return { kind: "mode", mode: theirs ? `${opp} Event first, then Ops` : "Event" };
    if (card === L.resolutions.SPACE) return { kind: "mode", mode: "Space Race" };
    const mode = `Ops: ${OPS_WORDS[namesByValue(L.resolutions)[card] ?? ""] ?? card}`;
    return { kind: "mode", mode: theirs ? `${mode}, then the ${opp} Event` : mode };
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
  title: string;
  lines: string[];
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

const INFLUENCE = /^  • (.+) Influence: (.+)$/;
const SIDE_CHANGE = /^(US|USSR): (\d+) -> (\d+)/;
const DEFCON = /^  • DEFCON: (\d+) -> (\d+)$/;
const VP = /^  • Victory Points: (\S+) \(\w+\) -> (\S+) \(\w+\) \(.*\)$/;
const MIL_OPS = /^  • (US|USSR) MilOps: (\d+) -> (\d+)$/;
const SPACE = /^  • Space Race: (US|USSR) (\d+) -> (\d+)$/;
const EFFECT = /^  • Effect (Activated|Cancelled): (\S+)$/;
const MOVED = /^  • Card #(\d+) \((.*)\) moved: (\S+) -> (\S+)$/;
const HAND = /^HAND_(US|USSR)_(KNOWN|UNKNOWN)$/;
/** The Space Race's attempts-this-turn counters: bookkeeping, the dice line says the attempt. */
const SPACE_ATTEMPT = /^SPACE_(US|USSR)_ATTEMPT_\d$/;

type Span = [number, number];
/** A line and the one step it can be matched to, if there is one. */
type Line = [string, number | null];

/** A play's changes to the board, each from its value before the play to its value after. */
class Board {
  private influence = new Map<string, Map<string, Span>>();
  private defcon?: Span;
  private vp?: Span;
  private milOps = new Map<string, Span>();
  private space = new Map<string, Span>();
  private effects = new Map<string, number>();
  private cards = new Map<number, { name: string; from: string; to: string }>();
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
  take(line: string, step: number): boolean {
    let m: RegExpExecArray | null;
    if ((m = INFLUENCE.exec(line))) {
      this.note(`influence:${m[1]}`, step);
      const sides = this.influence.get(m[1]) ?? new Map<string, Span>();
      this.influence.set(m[1], sides);
      for (const part of m[2].split(", ")) {
        const s = SIDE_CHANGE.exec(part);
        if (s) Board.widen(sides, s[1], +s[2], +s[3]);
      }
    } else if ((m = DEFCON.exec(line))) {
      this.note("defcon", step);
      this.defcon = this.defcon ? [this.defcon[0], +m[2]] : [+m[1], +m[2]];
    } else if ((m = VP.exec(line))) {
      this.note("vp", step);
      this.vp = this.vp ? [this.vp[0], parseInt(m[2], 10)] : [parseInt(m[1], 10), parseInt(m[2], 10)];
    } else if ((m = MIL_OPS.exec(line))) {
      this.note(`milops:${m[1]}`, step);
      Board.widen(this.milOps, m[1], +m[2], +m[3]);
    } else if ((m = SPACE.exec(line))) {
      this.note(`space:${m[1]}`, step);
      Board.widen(this.space, m[1], +m[2], +m[3]);
    } else if ((m = EFFECT.exec(line))) {
      this.note(`effect:${m[2]}`, step);
      this.effects.set(m[2], (this.effects.get(m[2]) ?? 0) + (m[1] === "Activated" ? 1 : -1));
    } else if ((m = MOVED.exec(line))) {
      const id = +m[1];
      this.note(`card:${id}`, step);
      const c = this.cards.get(id);
      if (c) c.to = m[4]; else this.cards.set(id, { name: m[2], from: m[3], to: m[4] });
    } else {
      return false;
    }
    return true;
  }

  /** The play's changes, in words; `implied` are cards whose discard the heading already says. */
  lines(implied: number[]): Line[] {
    const out: Line[] = [];
    for (const [country, sides] of this.influence) {
      const parts: string[] = [];
      for (const side of ["US", "USSR"]) {
        const s = sides.get(side);
        if (s && s[0] !== s[1]) parts.push(`${side}: ${s[0]} -> ${s[1]} (${signed(s[1] - s[0])})`);
      }
      if (parts.length) out.push([`  • ${country} Influence: ${parts.join(", ")}`, this.stepOf(`influence:${country}`)]);
    }
    if (this.defcon && this.defcon[0] !== this.defcon[1]) {
      out.push([`  • DEFCON: ${this.defcon[0]} -> ${this.defcon[1]}`, this.stepOf("defcon")]);
    }
    if (this.vp && this.vp[0] !== this.vp[1]) {
      out.push([`  • Victory Points: ${vpStr(this.vp[0])} -> ${vpStr(this.vp[1])} (${signed(this.vp[1] - this.vp[0])} VP)`, this.stepOf("vp")]);
    }
    for (const [side, s] of this.milOps) if (s[0] !== s[1]) out.push([`  • ${side} MilOps: ${s[0]} -> ${s[1]}`, this.stepOf(`milops:${side}`)]);
    for (const [side, s] of this.space) if (s[0] !== s[1]) out.push([`  • Space Race: ${side} ${s[0]} -> ${s[1]}`, this.stepOf(`space:${side}`)]);
    for (const [flag, net] of this.effects) {
      if (net === 0 || SPACE_ATTEMPT.test(flag)) continue;
      // English, as the rest of the line (until i18n phase 2, which takes effectNameMessage).
      const name = effectEnglishName(flag);
      out.push([net > 0 ? `  • In effect: ${name}` : `  • No longer in effect: ${name}`, this.stepOf(`effect:${flag}`)]);
    }
    out.push(...this.cardLines(implied));
    return out;
  }

  private cardLines(implied: number[]): Line[] {
    const out: Line[] = [];
    const draws: Record<string, string[]> = { US: [], USSR: [] };
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
      const name = `${c.name} (#${id})`;
      const hf = HAND.exec(from);
      const ht = HAND.exec(to);
      const key = `card:${id}`;
      if (from === "DRAW_DECK" && ht) { draws[ht[1]].push(c.name); drawKeys[ht[1]].push(key); continue; }
      if (to === "DRAW_DECK" && from === "DISCARD_PILE") { reshuffled++; reshuffledKeys.push(key); continue; }
      if (to === "DRAW_DECK" && from === "UNAVAILABLE") { added++; addedKeys.push(key); continue; }
      let says: string;
      if (ht && hf && hf[1] === ht[1]) {
        says = ht[2] === "KNOWN" ? `in the ${ht[1]} hand, now known to the ${other(ht[1])}` : `in the ${ht[1]} hand, no longer known to the ${other(ht[1])}`;
      } else if (ht) {
        says = hf ? `taken into the ${ht[1]} hand` : `to the ${ht[1]} hand`;
      } else {
        says = {
          DISCARD_PILE: "discarded", REMOVED_FROM_GAME: "removed from the game", ONGOING_EVENT: "stays in play",
          DRAW_DECK: "returned to the draw deck", PEEKED_TEMP: "drawn to choose from", UNAVAILABLE: "set aside",
        }[to] ?? `moved to ${to}`;
      }
      out.push([`  • ${name} ${says}`, this.stepOf(key)]);
    }
    for (const side of ["US", "USSR"]) {
      const n = draws[side].length;
      if (n) out.push([`  • ${side} draws ${n} card${n === 1 ? "" : "s"}: ${draws[side].join(", ")}`, this.stepOf(...drawKeys[side])]);
    }
    if (added) out.push([`  • ${added} cards of the new era shuffled into the draw deck`, this.stepOf(...addedKeys)]);
    if (reshuffled) {
      out.push([`  • The discard pile (${reshuffled} cards) reshuffled into the draw deck`, this.stepOf(...reshuffledKeys)]);
    }
    return out;
  }
}

function signed(v: number): string {
  return v > 0 ? `+${v}` : `${v}`;
}

/** A click line that its own effects already say: a placement, a coup's or a target's. */
function toldByItsEffects(item: ActionLogItem): boolean {
  return (item.details ?? []).some(d => INFLUENCE.test(d) || d.startsWith("  🎲 ") || /^  • .+ \(#\d+\): /.test(d));
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
  let modes: string[] = [];
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

  const add = (line: string, step: number | null) => {
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
    if (modes.length) entry.title = `${entry.title} — ${modes.join(", then ")}`;
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
      if (kind === "setup" && !toldByItsEffects(item)) add(`  ▸ ${item.text}`, item.step_index);
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
        const ownEvent = kind === "play" && modes.length === 1 && modes[0] === "Event";
        if (resolving && item.resolving_name && !ownEvent) add(`  ▸ ${item.resolving_name} — ${item.player}`, null);
      }
      if (item.kind === "mode" && kind === "play" && item.mode) {
        // An opponent's Event first, then the Ops: the mode chosen after names the Ops.
        const last = modes[modes.length - 1];
        if (last && last.endsWith("then Ops") && item.mode.startsWith("Ops: ")) modes[modes.length - 1] = `${last.slice(0, -3)}${item.mode}`;
        else modes.push(item.mode);
      } else if ((item.kind === "setup" || item.kind === "point") && toldByItsEffects(item)) {
        // Said by the Influence or the dice below.
      } else {
        add(`  ▸ ${item.text}`, item.step_index);
      }
    }
    if (!entry) return;
    entry.vp_delta += item.vp_delta ?? 0;
    for (const d of item.details ?? []) {
      if (item.title && d.startsWith("  ★ Headlines Simultaneously Revealed")) continue;
      if (MOVED.test(d) ? cards.take(d, item.step_index) : board.take(d, item.step_index)) continue;
      add(d, item.step_index);
      // A random discard and Missile Envy's exchange say where their cards went.
      if (d.startsWith("  ✂ ") || d.startsWith("  ⇄ ")) {
        for (const m of d.matchAll(/\(#(\d+)\)/g)) implied.push(+m[1]);
      }
    }
  });
  close();
  return entries;
}

/**
 * A replay's steps as log items, for groupLog: the description and the dice the step recorded,
 * what its Events did, and the board's changes between the step's snapshot and the one before.
 * A replay file holds only its snapshots, so a turn's end is part of the last step of the turn.
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
    const details: string[] = [];
    const rolls = step.rolls ?? (after?.die_roll && after.die_roll.type !== "NONE" ? [after.die_roll] : []);
    for (const r of rolls) details.push(...describeRoll(engine, before ?? after, r, player));
    if (before && after) {
      details.push(...describeEventTracks(before, after, step.event_tracks ?? after.event_tracks));
      details.push(...describeDeltas(engine, before, { ...after, die_roll: undefined }, player,
                                     action.decision_type === engine.layout.decision_types.SELECT_CARD ? action.primary_id : 0));
    }
    return {
      step_index: i + 1, turn: step.turn, ar: step.ar, phase: step.phase, player,
      text: step.description || `Action type=${action.decision_type}`, details,
      vp_delta: before && after ? (after.victory_points ?? 0) - (before.victory_points ?? 0) : 0,
      ...clickInfo(engine, before, action, player),
    };
  });
}
