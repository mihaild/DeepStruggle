/**
 * What the other side's latest turn did to the map, for a player's view: the countries whose
 * influence it changed, and the ones it targeted (a failed coup or a realignment that moved
 * nothing still matters).
 *
 * A "turn" is a run of steps that share a turn, phase, action round and owner. The owner is the
 * phasing player -- whose action round it is, which includes the other side's decisions inside it
 * (an opponent's event triggered by the card) -- except at setup, where each side places its own.
 * During the headline the engine makes each event's owner the phasing player while it resolves,
 * so a headline event is its owner's turn; choosing the headline card changes nothing on the map
 * and is not a turn.
 *
 * Everything here is public: the map, and which country a move named.
 */
import { GameState, LastTurn, LastTurnCountry } from "../types";
import { ReplayStep } from "../replay_controls";
import { engineLayout } from "../engine/layout";
import { Message, message, renderEnglish } from "../i18n";

const N_COUNTRIES = 84;

interface Owned {
  owner: string;
  key: string;
  headlineChoice: boolean;
}

function owned(before: GameState, step: ReplayStep): Owned {
  const phase = before.current_phase_name ?? step.phase;
  const owner = phase === "SETUP" ? step.player : before.phasing_player;
  const after = step.state_snapshot;
  const hl = step.player === "US" ? "headline_us_card" : "headline_ussr_card";
  const headlineChoice = phase === "HEADLINE" && step.player !== "NONE" && !before[hl] && !!after[hl];
  return { owner, key: `${before.turn}|${phase}|${before.action_round}|${owner}`, headlineChoice };
}

/** Which turn of a side's it was, by the state it began in: "Setup", "T3 Headline", "T3 AR4". */
export function turnLabel(before: GameState): Message {
  const phase = before.current_phase_name ?? "";
  if (phase === "SETUP") return message("common:round.setup");
  if (phase === "HEADLINE") return message("common:round.headline", { turn: before.turn });
  return message("common:round.actionRound", { turn: before.turn, ar: before.action_round });
}

/** The latest turn of `side` in a game's steps (`start` is the state before the first), or null. */
export function lastTurnOf(start: GameState | null, steps: ReplayStep[], side: "US" | "USSR"): LastTurn | null {
  if (!start || steps.length === 0) return null;
  const before = (k: number) => (k === 0 ? start : steps[k - 1].state_snapshot);
  let end = -1;
  for (let k = steps.length - 1; k >= 0; k--) {
    const o = owned(before(k), steps[k]);
    if (o.owner === side && !o.headlineChoice) { end = k; break; }
  }
  if (end < 0) return null;
  const key = owned(before(end), steps[end]).key;
  let first = end;
  while (first > 0 && owned(before(first - 1), steps[first - 1]).key === key) first--;

  const was = before(first).countries || {};
  const now = steps[end].state_snapshot.countries || {};
  const targeted = new Set<number>();
  const POINT_NODE = engineLayout()?.decision_types.POINT_NODE;
  for (let k = first; k <= end; k++) {
    const a = steps[k].action || {};
    if (a.decision_type === POINT_NODE && a.primary_id < N_COUNTRIES) targeted.add(a.primary_id);
  }
  const countries: LastTurnCountry[] = [];
  for (const [name, c] of Object.entries(now)) {
    const p = was[name];
    const us = c.us_influence - (p?.us_influence ?? 0);
    const ussr = c.ussr_influence - (p?.ussr_influence ?? 0);
    if (us || ussr || targeted.has(c.id)) countries.push({ id: c.id, name, us, ussr, targeted: targeted.has(c.id) });
  }
  countries.sort((a, b) => a.name.localeCompare(b.name));
  const labelMessage = turnLabel(before(first));
  return { player: side, label: renderEnglish(labelMessage), labelMessage, countries };
}
