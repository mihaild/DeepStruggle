/**
 * The action log: what an applied action did, and what it changed, as log lines (game/log.ts) --
 * Messages with their cards, countries and sides by id, worded only where the log is shown.
 *
 * A line-for-line port of the Python session's `describe_action_and_deltas` (web/server/session.py
 * before the workbench moved into the browser): rendered in English (`lineEnglish`), each line is
 * that function's text, pinned by tests/web/test_describe_golden.py against lines it produced.
 * Names come from the engine, not from rules/*.json, so the English is the engine's own.
 *
 * Iteration order matters for a multi-line diff: countries are walked by id and cards 1..110, as
 * the Python dicts were built; the JSON the page reads has its keys sorted by name instead.
 */
import { DieRollState, EventTrackState, GameState, MicroAction } from "../types";
import { CARDS_METADATA } from "../metadata";
import { branchMeta } from "./branches";
import { WasmEngine } from "../engine/wasm_engine";
import { cardId, namesByValue } from "../engine/layout";
import { Message, MessageArg, MessageKey, message, messageIfKnown } from "../i18n";
import { LogLine, cardArg, countryArg, logLine, sideArg, signed } from "./log";

const SIDE = new Map(CARDS_METADATA.map(c => [c.id, c.side]));

/**
 * The cards the log words by name, by the engine's ids (WasmEngine.layout.card_ids): Grain Sales
 * to Soviets (its node offers the drawn card back, a CONFIRM_DONE, for its own 2 Ops), the cards
 * whose card choices eventCardChoice words, the random discards, and Eagle/Bear Has Landed's pseudo-card.
 */
function cardIds(engine: WasmEngine) {
  const L = engine.layout;
  return {
    GRAIN_SALES: cardId(L, "GRAIN_SALES_TO_SOVIETS"), STAR_WARS: cardId(L, "STAR_WARS"),
    SALT: cardId(L, "SALT_NEGOTIATIONS"), UN_INTERVENTION: cardId(L, "UN_INTERVENTION"),
    MISSILE_ENVY: cardId(L, "MISSILE_ENVY"), FIVE_YEAR_PLAN: cardId(L, "FIVE_YEAR_PLAN"),
    EAGLE_BEAR_LANDED: cardId(L, "EAGLE_BEAR_LANDED_DISCARD"), TERRORISM: cardId(L, "TERRORISM"),
  };
}

/** How the log says each play mode (`log:playMode.<Resolution>`) and the deferred Ops choice
 *  (`log:opMode.<OpMode>`), by the engine's names; the number for a name the catalog lacks. On an
 *  opponent card EVENT reads as event-first and any OPS_* as ops-first, which is what the retired
 *  CHOOSE_TIMING_BRANCH said. */
function modeWords(kind: "playMode" | "opMode", name: string | undefined, value: number): MessageArg {
  return (name ? messageIfKnown(`log:${kind}.${name}`) : null) ?? String(value);
}

function isConfirmDone(a: Required<MicroAction>): boolean {
  return (a.flags & 0x80) !== 0 || a.primary_id === 255;
}

/** A score as the log says it: "+3 (US)", "-2 (USSR)", "0 (Tie)". */
export function vpScore(vp: number): Message {
  return vp > 0 ? message("log:vpScore.us", { n: vp }) : vp < 0 ? message("log:vpScore.ussr", { n: -vp })
    : message("log:vpScore.tie");
}

/** The score a `vpScore` message says. */
export function vpOf(m: MessageArg | undefined): number {
  if (typeof m !== "object" || !("key" in m)) return Number.NaN;
  const n = Number(m.params?.n ?? 0);
  return m.key === "log:vpScore.us" ? n : m.key === "log:vpScore.ussr" ? -n : 0;
}

/** A card the log names, by the engine's name; a number that is no card stays "#n". */
function cardRef(engine: WasmEngine, id: number): MessageArg {
  return id >= 1 && id <= 110 ? cardArg(id, engine.cardName(id)) : `#${id}`;
}

function countriesById(s: GameState) {
  return Object.values(s.countries || {}).sort((a, b) => a.id - b.id);
}

export function describeActionAndDeltas(
  engine: WasmEngine, before: GameState, after: GameState, action: Required<MicroAction>,
): LogLine[] {
  const chosen = action.decision_type === engine.layout.decision_types.SELECT_CARD ? action.primary_id : 0;
  const p = before.decision_context?.decision_player ?? "NONE";
  return [...describeAction(engine, before, after, action), ...describeDeltas(engine, before, after, p, chosen)];
}

/**
 * What was done: the action's own line, and the headline reveal when it is the second commit. The
 * first half of `describeActionAndDeltas`; the session puts the click's dice and Events after it
 * and the board's changes last, so a log reads from cause to effect.
 */
export function describeAction(
  engine: WasmEngine, before: GameState, after: GameState, action: Required<MicroAction>,
): LogLine[] {
  const d = action.decision_type;
  const primary = action.primary_id;
  const secondary = action.secondary_id;
  const flags = action.flags;
  const ctx = before.decision_context;
  const p = ctx?.decision_player ?? "NONE";
  const side = sideArg(p);
  const logs: LogLine[] = [];
  const say = (key: MessageKey, params?: Record<string, MessageArg>) => logs.push(logLine(undefined, key, { side, ...params }));
  const L = engine.layout;
  const DT = L.decision_types;
  const C = cardIds(engine);

  // 1. What was done
  if (d === DT.SELECT_CARD) {
    const resolving = ctx?.resolving_card ?? 0;
    const card = { card: cardRef(engine, primary), id: primary };
    const phase = before.current_phase_name ?? "";
    if (resolving !== 0 && resolving !== C.EAGLE_BEAR_LANDED) {
      logs.push(eventCardChoice(engine, before, after, action, p, resolving));
    } else if (resolving === C.EAGLE_BEAR_LANDED) {
      if (isConfirmDone(action) || primary === 0) say("log:action.eagleBearLandedPass");
      else say("log:action.eagleBearLandedDiscard", card);
    } else if (phase === "HEADLINE") {
      const usBefore = before.headline_us_card ?? 0;
      const ussrBefore = before.headline_ussr_card ?? 0;
      const isSecond = (p === "US" && ussrBefore > 0) || (p === "USSR" && usBefore > 0);
      say("log:action.headlineCommit", card);
      if (isSecond) {
        const usCard = p === "US" ? primary : usBefore;
        const ussrCard = p === "USSR" ? primary : ussrBefore;
        const us = cardArg(usCard, engine.cardName(usCard));
        const ussr = cardArg(ussrCard, engine.cardName(ussrCard));
        const usOps = engine.cardOps(usCard);
        const ussrOps = engine.cardOps(ussrCard);
        // The order is the engine's: the card it resolves first (headline_first_card), set by the
        // commit that revealed both.
        const firstP = after.headline_first_card === ussrCard ? "USSR" : "US";
        const secondP = firstP === "US" ? "USSR" : "US";
        logs.push(logLine("★", "log:headline.revealed", {
          usCard: us, usId: usCard, usOps, ussrCard: ussr, ussrId: ussrCard, ussrOps,
        }));
        logs.push(logLine("★", "log:headline.order", {
          first: sideArg(firstP), firstCard: firstP === "US" ? us : ussr, firstOps: firstP === "US" ? usOps : ussrOps,
          second: sideArg(secondP), secondCard: firstP === "US" ? ussr : us, secondOps: firstP === "US" ? ussrOps : usOps,
        }));
      }
    } else {
      say("log:action.plays", card);
    }
  } else if (d === DT.POINT_NODE) {
    if (isConfirmDone(action) || primary === 255 || (primary === 0 && (flags & 128))) {
      say("log:action.confirmPoints");
    } else {
      const country = primary < 84 ? countryArg(primary, engine.countryName(primary)) : message("log:node", { n: primary });
      const resolving = ctx?.resolving_card ?? 0;
      const phase = before.current_phase_name ?? "";
      if (phase === "SETUP") {
        say("log:action.setupInfluence", { country });
      } else if (resolving > 0) {
        say("log:action.eventTarget", { country, card: cardArg(resolving, engine.cardName(resolving)), id: resolving });
      } else {
        const opMode = ctx?.op_mode ?? L.op_modes.INFLUENCE;
        if (opMode === L.op_modes.COUP) say("log:action.coup", { country });
        else if (opMode === L.op_modes.REALIGN) say("log:action.realign", { country });
        else say("log:action.influence", { country });
      }
    }
  } else if (d === DT.SELECT_PLAY_MODE) {
    // P17: one resolution node (`log:playMode`).
    if (isConfirmDone(action) && (ctx?.resolving_card ?? 0) === C.GRAIN_SALES) {
      // The decline at Grain Sales' node: the drawn card goes back, Grain Sales' own Ops are taken.
      const drawn = ctx?.pending_op_card ?? 0;
      say("log:action.grainSalesReturn", { card: cardArg(drawn, engine.cardName(drawn)), id: drawn });
    } else if (primary === L.resolutions.SPACE) {
      if (secondary > 0) say("log:action.spaceRoll", { roll: secondary });
      else say("log:action.space");
    } else {
      say("log:action.playMode", { mode: modeWords("playMode", namesByValue(L.resolutions)[primary], primary) });
    }
  } else if (d === DT.SELECT_OP_MODE) {
    say("log:action.opMode", { mode: modeWords("opMode", namesByValue(L.op_modes)[primary], primary) });
  } else if (d === DT.CHOOSE_BRANCH) {
    const card = ctx?.resolving_card ?? 0;
    const meta = isConfirmDone(action) ? null : branchMeta(card, primary, before);
    if (isConfirmDone(action)) say("log:action.confirmOption");
    else if (meta) say("log:action.branch", { verb: meta.verb, event: cardArg(card, engine.cardName(card)) });
    else say("log:action.option", { n: primary });
  } else {
    say("log:action.other", { type: d, primary, secondary, flags });
  }
  return logs;
}

/**
 * What changed from `before` to `after`: influence, the tracks, effects, cards that moved, and the
 * die roll `after` records. The deltas half of `describeActionAndDeltas`; on its own it is how the
 * session logs the changes a step made after the action itself (the turn's end).
 */
export function describeDeltas(engine: WasmEngine, before: GameState, after: GameState, p: string,
                               chosen = 0): LogLine[] {
  const logs: LogLine[] = [];
  // 2. Influence
  const oldCountries = before.countries || {};
  for (const c of countriesById(after)) {
    const old = oldCountries[c.name];
    if (!old) continue;
    const usDiff = c.us_influence - old.us_influence;
    const ussrDiff = c.ussr_influence - old.ussr_influence;
    if (usDiff !== 0 || ussrDiff !== 0) {
      logs.push(influenceLine(countryArg(c.id, c.name),
                              usDiff !== 0 ? [old.us_influence, c.us_influence] : null,
                              ussrDiff !== 0 ? [old.ussr_influence, c.ussr_influence] : null));
    }
  }

  // 3. Tracks
  if (before.defcon !== after.defcon) logs.push(defconLine(before.defcon, after.defcon));
  if (before.victory_points !== after.victory_points) {
    logs.push(vpLine(before.victory_points ?? 0, after.victory_points ?? 0));
  }
  // The Python session read `us_mil_ops`/`ussr_mil_ops`, keys the display state never had (it
  // carries `mil_ops: {US, USSR}`), so a MilOps change was never logged. Fixed here; the golden
  // test drops these lines when comparing with that function's output.
  for (const side of ["US", "USSR"] as const) {
    const b = before.mil_ops?.[side];
    const a = after.mil_ops?.[side];
    if (b !== a) logs.push(milOpsLine(side, b ?? 0, a ?? 0));
  }

  // The Space Race track: an advance pays VP, and Captured Nazi Scientist's +2 VP read as
  // unexplained without the move that paid it. The Python session never logged it.
  for (const side of ["US", "USSR"] as const) {
    const b = before.space?.[side];
    const a = after.space?.[side];
    if (b !== undefined && a !== undefined && b !== a) logs.push(spaceLine(side, b, a));
  }

  // 3.5 Effect flags
  const oldFlags = new Set(before.flags || []);
  const newFlags = new Set(after.flags || []);
  for (const f of [...newFlags].filter(x => !oldFlags.has(x)).sort()) logs.push(logLine("•", "log:change.effectActivated", { flag: f }));
  for (const f of [...oldFlags].filter(x => !newFlags.has(x)).sort()) logs.push(logLine("•", "log:change.effectCancelled", { flag: f }));

  // 4. Cards that moved
  const oldLocs = before.card_locations || {};
  const newLocs = after.card_locations || {};
  for (let cid = 1; cid <= 110; cid++) {
    const o = oldLocs[String(cid)];
    const n = newLocs[String(cid)];
    if (n === undefined) continue;
    if (o && o !== n) {
      logs.push(logLine("•", "log:change.cardMoved", { id: cid, card: cardArg(cid, engine.cardName(cid)), from: o, to: n }));
    }
  }

  // 4.5 The random discards events make, said as discards; Missile Envy's exchange, as one
  logs.push(...randomDiscards(engine, before, after, chosen));
  logs.push(...missileEnvyExchange(engine, before, after));

  // 5. The die roll, if one happened
  logs.push(...describeRoll(engine, before, after.die_roll, p));
  return logs;
}

/** A country's Influence change, by side: each `[from, to]`, or null where that side's did not move. */
export function influenceLine(country: MessageArg, us: [number, number] | null, ussr: [number, number] | null): LogLine {
  const u: Record<string, MessageArg> = us ? { usFrom: us[0], usTo: us[1], usDelta: signed(us[1] - us[0]) } : {};
  const r: Record<string, MessageArg> = ussr ? { ussrFrom: ussr[0], ussrTo: ussr[1], ussrDelta: signed(ussr[1] - ussr[0]) } : {};
  const key = us && ussr ? "log:change.influenceBoth" : us ? "log:change.influenceUs" : "log:change.influenceUssr";
  return logLine("•", key, { country, ...u, ...r });
}

export function defconLine(from: number, to: number): LogLine {
  return logLine("•", "log:change.defcon", { from, to });
}

/** The score's change, from `from` to `to`. */
export function vpLine(from: number, to: number): LogLine {
  return logLine("•", "log:change.vp", { from: vpScore(from), to: vpScore(to), delta: signed(to - from) });
}

export function milOpsLine(side: string, from: number, to: number): LogLine {
  return logLine("•", "log:change.milOps", { side: sideArg(side), from, to });
}

export function spaceLine(side: string, from: number, to: number): LogLine {
  return logLine("•", "log:change.space", { side: sideArg(side), from, to });
}

/**
 * What each Event of a step did to VP and DEFCON, when the step's before/after cannot say it. The
 * second headline commit resolves both headlines: Nuclear Test Ban's +1 VP for the US and Pershing
 * II's +1 VP for the USSR net to no change, and the Tracks lines then say nothing at all. A lone
 * Event that accounts for the whole change is already told by those lines, so it adds nothing.
 */
export function describeEventTracks(before: GameState, after: GameState,
                                    tracks: EventTrackState[] | undefined): LogLine[] {
  if (!tracks || tracks.length === 0) return [];
  const vp = tracks.reduce((sum, t) => sum + t.vp_delta, 0);
  const defcon = tracks.reduce((sum, t) => sum + t.defcon_delta, 0);
  const netVp = (after.victory_points ?? 0) - (before.victory_points ?? 0);
  const netDefcon = (after.defcon ?? 0) - (before.defcon ?? 0);
  if (tracks.length === 1 && vp === netVp && defcon === netDefcon) return [];
  return tracks.map(t => {
    const card = { card: cardArg(t.card_id, t.card_name || `Card #${t.card_id}`), id: t.card_id };
    const vpPart = t.vp_delta !== 0 ? { side: sideArg(t.vp_delta > 0 ? "US" : "USSR"), vp: Math.abs(t.vp_delta) } : null;
    const defconPart = t.defcon_delta !== 0 ? { defcon: signed(t.defcon_delta) } : null;
    const key = vpPart && defconPart ? "log:eventTrack.both" : vpPart ? "log:eventTrack.vp"
      : defconPart ? "log:eventTrack.defcon" : "log:eventTrack.none";
    return logLine("•", key, { ...card, ...vpPart, ...defconPart });
  });
}

/**
 * The dice line for one roll, as the display state writes a roll (`die_roll`); `before` is the
 * position the roll was made from (a coup's defence reads its stability). Nothing for no roll.
 */
export function describeRoll(engine: WasmEngine, before: GameState, roll: DieRollState | undefined, p: string): LogLine[] {
  const logs: LogLine[] = [];
  const r = roll || ({} as DieRollState);
  const type = r.type ?? "NONE";
  const named = (r.country_id ?? 255) < 84;
  const cname = () => r.country_name || (named ? engine.countryName(r.country_id ?? 0) : "");
  const country = (name: string): MessageArg => (named ? countryArg(r.country_id ?? 0, name) : name);
  const sgn = (v: number) => (v >= 0 ? `+${v}` : `${v}`);
  const roller = sideArg(r.roller || p);
  if (type === "COUP") {
    const c = cname();
    const stab = before.countries?.[c]?.stability ?? 1;
    const roll1 = r.roll1 ?? 0, mod1 = r.mod1 ?? 0, tot1 = r.total1 ?? roll1 + mod1;
    const net = r.net_delta ?? 0;
    const result = r.success ? message("log:roll.coupSucceeded", { net }) : message("log:roll.coupFailed");
    const values = { country: country(c), roller, roll: roll1, mod: mod1, total: tot1, defense: stab * 2, stability: stab, result };
    if (r.card_name) {
      const card: MessageArg = r.card_id ? cardArg(r.card_id, r.card_name) : r.card_name;
      logs.push(logLine("🎲", "log:roll.coupWithCard", { ...values, card }));
    } else {
      logs.push(logLine("🎲", "log:roll.coup", values));
    }
  } else if (type === "REALIGNMENT") {
    const c = cname();
    const rus = r.roll1 ?? 0, mus = r.mod1 ?? 0, tus = r.total1 ?? rus + mus;
    const rr = r.roll2 ?? 0, mr = r.mod2 ?? 0, tr = r.total2 ?? rr + mr;
    const net = r.net_delta ?? 0;
    const result = tus > tr ? message("log:roll.realignUs", { by: tus - tr, net })
      : tr > tus ? message("log:roll.realignUssr", { by: tr - tus, net }) : message("log:roll.realignTie");
    logs.push(logLine("🎲", "log:roll.realign", {
      country: country(c), usRoll: rus, usMod: sgn(mus), usTotal: tus, ussrRoll: rr, ussrMod: sgn(mr), ussrTotal: tr, result,
    }));
  } else if (type === "SPACE_RACE") {
    const box = r.country_id ?? 0;
    const roll1 = r.roll1 ?? 0;
    const result = r.success ? message("log:roll.spaceSucceeded", { box })
      : message("log:roll.spaceFailed", { roll: roll1, threshold: r.mod1 ?? 0 });
    logs.push(logLine("🎲", "log:roll.space", { box, roller, roll: roll1, result }));
  } else if (type === "WAR_EVENT") {
    const name = r.card_name || `Card #${r.card_id}`;
    const card: MessageArg = r.card_id !== undefined ? cardArg(r.card_id, name) : name;
    const c = cname();
    const roll1 = r.roll1 ?? 0, mod1 = r.mod1 ?? 0, tot1 = r.total1 ?? roll1 + mod1;
    const threshold = r.mod2 ?? 0;
    const result = r.success ? message("log:roll.warSucceeded", { total: tot1, threshold })
      : message("log:roll.warFailed", { total: tot1, threshold });
    const values = { card, roller, roll: roll1, mod: sgn(mod1), total: tot1, result };
    if (c) logs.push(logLine("🎲", "log:roll.warTargeting", { ...values, country: country(c) }));
    else logs.push(logLine("🎲", "log:roll.war", values));
  } else if (type === "OLYMPIC_GAMES") {
    const sponsor = r.roller || p;
    const sr = r.roll1 ?? 0, st = r.total1 ?? sr + (r.mod1 ?? 2);
    const or = r.roll2 ?? 0, ot = r.total2 ?? or;
    const opp = sponsor === "USSR" ? "US" : "USSR";
    const result = st > ot ? message("log:roll.olympicsSponsor", { side: sideArg(sponsor), by: st - ot })
      : ot > st ? message("log:roll.olympicsOpponent", { side: sideArg(opp) }) : message("log:roll.olympicsTie");
    logs.push(logLine("🎲", "log:roll.olympics", {
      sponsor: sideArg(sponsor), sponsorRoll: sr, sponsorTotal: st, opponent: sideArg(opp), opponentRoll: or, result,
    }));
  } else if (type === "SUMMIT") {
    const ur = r.roll1 ?? 0, ud = r.mod1 ?? 0, ut = r.total1 ?? ur + ud;
    const sr = r.roll2 ?? 0, sd = r.mod2 ?? 0, stt = r.total2 ?? sr + sd;
    const by = Math.abs(ut - stt);
    const result = ut > stt ? message("log:roll.summitUs", { by }) : stt > ut ? message("log:roll.summitUssr", { by })
      : message("log:roll.summitTie");
    logs.push(logLine("🎲", "log:roll.summit", {
      usRoll: ur, usMod: ud, usTotal: ut, ussrRoll: sr, ussrMod: sd, ussrTotal: stt, result,
    }));
  } else if (type === "TRAP_ESCAPE") {
    const result = r.success ? message("log:roll.trapSucceeded") : message("log:roll.trapFailed");
    logs.push(logLine("🎲", "log:roll.trap", { roller, roll: r.roll1 ?? 0, result }));
  }
  return logs;
}

/** Where a card was, for "from the US hand" and the like (null where it says nothing). */
function from(loc: string | undefined): Message | null {
  if (!loc) return null;
  if (loc.startsWith("HAND_US_")) return message("log:from.hand", { side: sideArg("US") });
  if (loc.startsWith("HAND_USSR_")) return message("log:from.hand", { side: sideArg("USSR") });
  if (loc === "DISCARD_PILE") return message("log:from.discardPile");
  if (loc === "PEEKED_TEMP") return message("log:from.drawn");
  return null;
}

/** What happened to a card chosen while an event resolves, by where it went: each without and with
 *  where it came from. */
const CHOICE = {
  discards: ["log:choice.discards", "log:choice.discardsFrom"],
  removes: ["log:choice.removes", "log:choice.removesFrom"],
  returns: ["log:choice.returns", "log:choice.returnsFrom"],
  takes: ["log:choice.takes", "log:choice.takesFrom"],
  gives: ["log:choice.gives", "log:choice.givesFrom"],
  chooses: ["log:choice.chooses", "log:choice.choosesFrom"],
} as const satisfies Record<string, readonly [MessageKey, MessageKey]>;

/**
 * A card chosen while an event resolves (`resolving_card`), said as what happened to it -- never as
 * a play or a headline commit, which is what these choices were logged as: the card the USSR has the
 * US discard with Aldrich Ames read as the USSR's headline, with a "Headlines Simultaneously
 * Revealed" line for it.
 */
function eventCardChoice(engine: WasmEngine, before: GameState, after: GameState, action: Required<MicroAction>,
                         p: string, resolving: number): LogLine {
  const C = cardIds(engine);
  const side = sideArg(p);
  const event = cardArg(resolving, engine.cardName(resolving));
  const id = action.primary_id;
  if (isConfirmDone(action) || id === 0) return logLine(undefined, "log:choice.none", { side, event });
  const card = { side, event, card: cardArg(id, engine.cardName(id)), id };
  const opp = p === "US" ? "USSR" : "US";
  const src = before.card_locations?.[String(id)];
  const dst = after.card_locations?.[String(id)];
  if (resolving === C.STAR_WARS) return logLine(undefined, "log:choice.starWars", card);
  if (resolving === C.SALT) return logLine(undefined, "log:choice.salt", card);
  if (resolving === C.UN_INTERVENTION) return logLine(undefined, "log:choice.unIntervention", card);
  if (resolving === C.MISSILE_ENVY) return logLine(undefined, "log:choice.missileEnvy", { ...card, opponent: sideArg(opp) });
  if (resolving === C.GRAIN_SALES && id === C.UN_INTERVENTION) {
    return logLine(undefined, "log:choice.grainSalesUn", { side, event, un: cardArg(id, engine.cardName(id)) });
  }
  const verb = dst === "DISCARD_PILE" ? CHOICE.discards : dst === "REMOVED_FROM_GAME" ? CHOICE.removes
    : dst === "DRAW_DECK" ? CHOICE.returns : dst?.startsWith(`HAND_${p}_`) ? CHOICE.takes
    : dst?.startsWith(`HAND_${opp}_`) ? CHOICE.gives : CHOICE.chooses;
  const where = from(src);
  return logLine(undefined, where ? verb[1] : verb[0], { ...card, opponent: sideArg(opp), ...(where ? { from: where } : {}) });
}

/**
 * The events that discard at random -- Five Year Plan (the USSR discards one; a US event discarded
 * so fires) and Terrorism (the player's opponent discards one, the US two once Iranian Hostage
 * Crisis is in play) -- make no choice, so their discards were only card-move lines. Each gets its
 * own line here: "✂ USSR discards Decolonization (#30) at random (Five Year Plan)". Read where the
 * event resolves in this step: played for the event (its play mode), or leaving play here (a
 * headline, or an opponent's card played for Ops); the step's own played and headline cards leave
 * a hand too and are not the event's discards.
 */
function randomDiscards(engine: WasmEngine, before: GameState, after: GameState, chosen: number): LogLine[] {
  const C = cardIds(engine);
  const lines: LogLine[] = [];
  const c = before.decision_context;
  const ownPlay = [chosen, c?.pending_op_card, before.headline_us_card, before.headline_ussr_card,
                   after.headline_us_card, after.headline_ussr_card];
  const events: Array<[number, string[]]> = [[C.FIVE_YEAR_PLAN, ["USSR"]], [C.TERRORISM, ["US", "USSR"]]];
  for (const [event, victims] of events) {
    const played = c?.decision_type_name === "SELECT_PLAY_MODE" && c.pending_op_card === event;
    const leaves = before.card_locations?.[String(event)] !== after.card_locations?.[String(event)];
    if (!played && !leaves) continue;
    for (let cid = 1; cid <= 110; cid++) {
      if (cid === event || ownPlay.includes(cid)) continue;
      const o = before.card_locations?.[String(cid)] ?? "";
      const n = after.card_locations?.[String(cid)] ?? "";
      const side = victims.find(v => o.startsWith(`HAND_${v}_`));
      if (!side || (n !== "DISCARD_PILE" && n !== "REMOVED_FROM_GAME")) continue;
      // A US event discarded by Five Year Plan fires.
      const fires = event === C.FIVE_YEAR_PLAN && SIDE.get(cid) === "us";
      lines.push(logLine("✂", fires ? "log:randomDiscardFires" : "log:randomDiscard", {
        side: sideArg(side), card: cardArg(cid, engine.cardName(cid)), id: cid, event: cardArg(event, engine.cardName(event)),
      }));
    }
  }
  return lines;
}

/**
 * Missile Envy's exchange: the player takes the opponent's highest-Ops card and hands Missile Envy
 * over. The taken card's event fires if it is the taker's own or both sides' event; an opponent's
 * event is played for its Ops instead, without the event. The giver must play Missile Envy for Ops in
 * its next action round. With one card of the highest Ops the exchange is no choice, so it was only
 * card-move lines (the taken card often straight to the discard pile, its event already played).
 * The engine marks the exchange: Missile Envy becomes the giver's forced card (`forced_card_id`,
 * `forced_card_player`) -- not merely Missile Envy reaching a hand, which a deal does too.
 */
function missileEnvyExchange(engine: WasmEngine, before: GameState, after: GameState): LogLine[] {
  const C = cardIds(engine);
  const forced = after.forced_card_id === C.MISSILE_ENVY && after.forced_card_player !== "NONE";
  const already = before.forced_card_id === C.MISSILE_ENVY && before.forced_card_player === after.forced_card_player;
  if (!forced || already) return [];
  const giver = after.forced_card_player, taker = giver === "US" ? "USSR" : "US";
  const sides = { giver: sideArg(giver), taker: sideArg(taker), event: cardArg(C.MISSILE_ENVY, engine.cardName(C.MISSILE_ENVY)) };
  const headlines = [before.headline_us_card, before.headline_ussr_card, after.headline_us_card, after.headline_ussr_card];
  const lines: LogLine[] = [];
  for (let cid = 1; cid <= 110; cid++) {
    if (cid === C.MISSILE_ENVY || headlines.includes(cid)) continue;
    const was = before.card_locations?.[String(cid)] ?? "";
    const now = after.card_locations?.[String(cid)] ?? "";
    if (!was.startsWith(`HAND_${giver}_`) || now.startsWith(`HAND_${giver}_`)) continue;
    const side = SIDE.get(cid);
    const card = { ...sides, card: cardArg(cid, engine.cardName(cid)), id: cid };
    if (side === giver.toLowerCase()) lines.push(logLine("⇄", "log:envy.takesGivers", { ...card, ops: engine.cardOps(cid) }));
    else if (side === "neutral") lines.push(logLine("⇄", "log:envy.takesBoth", card));
    else lines.push(logLine("⇄", "log:envy.takesTakers", card));
  }
  lines.push(logLine("⇄", "log:envy.goes", sides));
  return lines;
}
