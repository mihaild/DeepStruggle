/**
 * The action log: what an applied action did, in words, and what it changed.
 *
 * A line-for-line port of the Python session's `describe_action_and_deltas` (web/server/session.py
 * before the workbench moved into the browser), pinned by tests/web/test_describe_golden.py
 * against lines that function produced. Names come from the engine, not from rules/*.json, so
 * the text is the engine's own.
 *
 * Iteration order matters for a multi-line diff: countries are walked by id and cards 1..110, as
 * the Python dicts were built; the JSON the page reads has its keys sorted by name instead.
 */
import { GameState, MicroAction } from "../types";
import { WasmEngine } from "../engine/wasm_engine";

const DT = { SELECT_CARD: 1, SELECT_PLAY_MODE: 2, SELECT_OP_MODE: 4, POINT_NODE: 5, CHOOSE_BRANCH: 6 };

function isConfirmDone(a: Required<MicroAction>): boolean {
  return (a.flags & 0x80) !== 0 || a.primary_id === 255;
}

function signed(v: number): string {
  return v > 0 ? `+${v}` : `${v}`;
}

function vpStr(vp: number): string {
  return vp > 0 ? `+${vp} (US)` : vp < 0 ? `${vp} (USSR)` : "0 (Tie)";
}

function countriesById(s: GameState) {
  return Object.values(s.countries || {}).sort((a, b) => a.id - b.id);
}

export function describeActionAndDeltas(
  engine: WasmEngine, before: GameState, after: GameState, action: Required<MicroAction>,
): string[] {
  const d = action.decision_type;
  const primary = action.primary_id;
  const secondary = action.secondary_id;
  const flags = action.flags;
  const ctx = before.decision_context;
  const p = ctx?.decision_player ?? "NONE";
  const logs: string[] = [];

  // 1. What was done
  if (d === DT.SELECT_CARD) {
    const resolving = ctx?.resolving_card ?? 0;
    const cardName = primary >= 1 && primary <= 110 ? engine.cardName(primary) : `#${primary}`;
    const phase = before.current_phase_name ?? "";
    if (resolving === 250) {
      if (isConfirmDone(action) || primary === 0) {
        logs.push(`${p} passes Space Walk (Box 6) turn-end discard opportunity.`);
      } else {
        logs.push(`${p} discards ${cardName} (#${primary}) via Space Walk (Box 6 privilege).`);
      }
    } else if (phase === "HEADLINE") {
      const usBefore = before.headline_us_card ?? 0;
      const ussrBefore = before.headline_ussr_card ?? 0;
      const isSecond = (p === "US" && ussrBefore > 0) || (p === "USSR" && usBefore > 0);
      logs.push(`${p} commits Headline Card: ${cardName} (#${primary})`);
      if (isSecond) {
        const usCard = p === "US" ? primary : usBefore;
        const ussrCard = p === "USSR" ? primary : ussrBefore;
        const usName = engine.cardName(usCard);
        const ussrName = engine.cardName(ussrCard);
        const usOps = engine.cardOps(usCard);
        const ussrOps = engine.cardOps(ussrCard);
        const firstP = usOps >= ussrOps ? "US" : "USSR";
        const secondP = firstP === "US" ? "USSR" : "US";
        const firstName = firstP === "US" ? usName : ussrName;
        const secondName = firstP === "US" ? ussrName : usName;
        logs.push(`  ★ Headlines Simultaneously Revealed: US plays '${usName}' (#${usCard}, ${usOps} Ops) vs USSR plays '${ussrName}' (#${ussrCard}, ${ussrOps} Ops)`);
        logs.push(`  ★ Resolution Order: 1st ${firstP} '${firstName}' (${Math.max(usOps, ussrOps)} Ops) -> 2nd ${secondP} '${secondName}' (${Math.min(usOps, ussrOps)} Ops)`);
      }
    } else {
      logs.push(`${p} plays Card: ${cardName} (#${primary})`);
    }
  } else if (d === DT.POINT_NODE) {
    if (isConfirmDone(action) || primary === 255 || (primary === 0 && (flags & 128))) {
      logs.push(`${p} confirms / finishes point node selections.`);
    } else {
      const cName = primary < 84 ? engine.countryName(primary) : `Node #${primary}`;
      const resolving = ctx?.resolving_card ?? 0;
      const phase = before.current_phase_name ?? "";
      if (phase === "SETUP") {
        logs.push(`${p} places 1 Influence in ${cName} during Setup.`);
      } else if (resolving > 0) {
        logs.push(`${p} targets ${cName} for Event: ${engine.cardName(resolving)} (#${resolving})`);
      } else {
        const opMode = (ctx as { op_mode?: number } | undefined)?.op_mode ?? 0;
        if (opMode === 1) logs.push(`${p} attempts Coup in ${cName}`);
        else if (opMode === 2) logs.push(`${p} conducts Realignment in ${cName}`);
        else logs.push(`${p} places Influence in ${cName}`);
      }
    }
  } else if (d === DT.SELECT_PLAY_MODE) {
    // P17: one resolution node. On an opponent card EVENT reads as event-first and any OPS_* as
    // ops-first, which is what the retired CHOOSE_TIMING_BRANCH used to say.
    const modes: Record<number, string> = {
      0: "EVENT", 1: "SPACE RACE", 2: "OPERATIONS - INFLUENCE PLACEMENT",
      3: "OPERATIONS - COUP ATTEMPT", 4: "OPERATIONS - REALIGNMENT",
    };
    if (primary === 1) {
      const rollInfo = secondary > 0 ? ` (Input Roll: ${secondary})` : "";
      logs.push(`${p} attempts Space Race with pending card${rollInfo}`);
    } else {
      logs.push(`${p} selects play mode: ${modes[primary] ?? String(primary)}`);
    }
  } else if (d === DT.SELECT_OP_MODE) {
    const opModes: Record<number, string> = { 0: "INFLUENCE PLACEMENT", 1: "COUP ATTEMPT", 2: "REALIGNMENT" };
    logs.push(`${p} chooses Op mode: ${opModes[primary] ?? String(primary)}`);
  } else if (d === DT.CHOOSE_BRANCH) {
    if (isConfirmDone(action)) logs.push(`${p} confirms / passes option.`);
    else logs.push(`${p} chooses option: ${primary}`);
  } else {
    logs.push(`${p} action: type=${d} primary=${primary} secondary=${secondary} flags=${flags}`);
  }

  // 2. Influence
  const oldCountries = before.countries || {};
  for (const c of countriesById(after)) {
    const old = oldCountries[c.name];
    if (!old) continue;
    const usDiff = c.us_influence - old.us_influence;
    const ussrDiff = c.ussr_influence - old.ussr_influence;
    if (usDiff !== 0 || ussrDiff !== 0) {
      const parts: string[] = [];
      if (usDiff !== 0) parts.push(`US: ${old.us_influence} -> ${c.us_influence} (${signed(usDiff)})`);
      if (ussrDiff !== 0) parts.push(`USSR: ${old.ussr_influence} -> ${c.ussr_influence} (${signed(ussrDiff)})`);
      logs.push(`  • ${c.name} Influence: ` + parts.join(", "));
    }
  }

  // 3. Tracks
  if (before.defcon !== after.defcon) logs.push(`  • DEFCON: ${before.defcon} -> ${after.defcon}`);
  if (before.victory_points !== after.victory_points) {
    const vb = before.victory_points ?? 0;
    const va = after.victory_points ?? 0;
    const delta = va - vb;
    logs.push(`  • Victory Points: ${vpStr(vb)} -> ${vpStr(va)} (${delta > 0 ? `+${delta} VP` : `${delta} VP`})`);
  }
  // The Python session read `us_mil_ops`/`ussr_mil_ops`, keys the display state never had (it
  // carries `mil_ops: {US, USSR}`), so a MilOps change was never logged. Fixed here; the golden
  // test drops these lines when comparing with that function's output.
  for (const side of ["US", "USSR"] as const) {
    const b = before.mil_ops?.[side];
    const a = after.mil_ops?.[side];
    if (b !== a) logs.push(`  • ${side} MilOps: ${b} -> ${a}`);
  }

  // 3.5 Effect flags
  const oldFlags = new Set(before.flags || []);
  const newFlags = new Set(after.flags || []);
  for (const f of [...newFlags].filter(x => !oldFlags.has(x)).sort()) logs.push(`  • Effect Activated: ${f}`);
  for (const f of [...oldFlags].filter(x => !newFlags.has(x)).sort()) logs.push(`  • Effect Cancelled: ${f}`);

  // 4. Cards that moved
  const oldLocs = before.card_locations || {};
  const newLocs = after.card_locations || {};
  for (let cid = 1; cid <= 110; cid++) {
    const o = oldLocs[String(cid)];
    const n = newLocs[String(cid)];
    if (n === undefined) continue;
    if (o && o !== n) logs.push(`  • Card #${cid} (${engine.cardName(cid)}) moved: ${o} -> ${n}`);
  }

  // 5. The die roll, if one happened
  const r = after.die_roll || ({} as NonNullable<GameState["die_roll"]>);
  const type = r.type ?? "NONE";
  const cname = () => r.country_name || ((r.country_id ?? 255) < 84 ? engine.countryName(r.country_id ?? 0) : "");
  const sgn = (v: number) => (v >= 0 ? "+" : "");
  if (type === "COUP") {
    const c = cname();
    const stab = before.countries?.[c]?.stability ?? 1;
    const roller = r.roller || p;
    const roll1 = r.roll1 ?? 0, mod1 = r.mod1 ?? 0, tot1 = r.total1 ?? roll1 + mod1;
    const def = stab * 2;
    const net = r.net_delta ?? 0;
    const cardPrefix = r.card_name ? ` (${r.card_name})` : "";
    const status = r.success ? `Net +${net} Influence (Coup Succeeded)` : "Coup Failed (Roll + Ops <= 2x Stability)";
    logs.push(`  🎲 Coup in ${c}${cardPrefix}: ${roller} rolls ${roll1} (+${mod1} Ops = ${tot1}) vs ${def} Defense (2x Stability ${stab}) -> ${status}`);
  } else if (type === "REALIGNMENT") {
    const c = cname();
    const rus = r.roll1 ?? 0, mus = r.mod1 ?? 0, tus = r.total1 ?? rus + mus;
    const rr = r.roll2 ?? 0, mr = r.mod2 ?? 0, tr = r.total2 ?? rr + mr;
    const net = r.net_delta ?? 0;
    const res = tus > tr ? `US wins by +${tus - tr} (Removes ${net} USSR Influence)`
      : tr > tus ? `USSR wins by +${tr - tus} (Removes ${net} US Influence)` : "Tie (No Influence Removed)";
    logs.push(`  🎲 Realignment in ${c}: US rolls ${rus} (${sgn(mus)}${mus} mod = ${tus}), USSR rolls ${rr} (${sgn(mr)}${mr} mod = ${tr}) -> ${res}`);
  } else if (type === "SPACE_RACE") {
    const target = r.country_id ?? 0;
    const roll1 = r.roll1 ?? 0;
    const status = r.success ? `SUCCESS (Advanced to Box #${target})` : `FAILED (Roll ${roll1} > ${r.mod1 ?? 0} threshold)`;
    logs.push(`  🎲 Space Race Attempt (Target Box #${target}): ${r.roller || p} rolls ${roll1} -> ${status}`);
  } else if (type === "WAR_EVENT") {
    const card = r.card_name || `Card #${r.card_id}`;
    const c = cname();
    const roll1 = r.roll1 ?? 0, mod1 = r.mod1 ?? 0, tot1 = r.total1 ?? roll1 + mod1;
    const threshold = r.mod2 ?? 0;
    const target = c ? ` targeting ${c}` : "";
    const status = r.success ? `Success (${tot1} >= ${threshold} target: Target Captured & VP Awarded)` : `Roll Failed (${tot1} < ${threshold} target)`;
    logs.push(`  🎲 ${card} Roll${target}: ${r.roller || p} rolls ${roll1} (${sgn(mod1)}${mod1} mod = ${tot1}) -> ${status}`);
  } else if (type === "OLYMPIC_GAMES") {
    const sponsor = r.roller || p;
    const sr = r.roll1 ?? 0, st = r.total1 ?? sr + 2;
    const or = r.roll2 ?? 0, ot = r.total2 ?? or;
    const opp = sponsor === "USSR" ? "US" : "USSR";
    const res = st > ot ? `Sponsor ${sponsor} wins by +${st - ot} (+2 VP)` : ot > st ? `Opponent ${opp} wins` : "Tie (No VP)";
    logs.push(`  🎲 Olympic Games Competition: Sponsor (${sponsor}) rolls ${sr} (+2 = ${st}), Opponent (${opp}) rolls ${or} -> ${res}`);
  } else if (type === "SUMMIT") {
    const ur = r.roll1 ?? 0, ud = r.mod1 ?? 0, ut = r.total1 ?? ur + ud;
    const sr = r.roll2 ?? 0, sd = r.mod2 ?? 0, stt = r.total2 ?? sr + sd;
    const diff = Math.abs(ut - stt);
    const res = ut > stt ? `US wins Summit by +${diff}` : stt > ut ? `USSR wins Summit by +${diff}` : "Tied Summit (No Effect)";
    logs.push(`  🎲 Summit Rolls: US rolls ${ur} (+${ud} Dom = ${ut}), USSR rolls ${sr} (+${sd} Dom = ${stt}) -> ${res}`);
  } else if (type === "TRAP_ESCAPE") {
    const status = r.success ? "Success! Discard canceled trap." : "Failed (Roll > 4, remains trapped)";
    logs.push(`  🎲 Trap Escape Roll: ${r.roller || p} rolls ${r.roll1 ?? 0} (Needed 1–4) -> ${status}`);
  }
  return logs;
}
