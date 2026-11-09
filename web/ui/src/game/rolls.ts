/**
 * The die rolls of the latest action round that rolled, for the map's roll tray: a coup, a
 * realignment, a Space Race attempt, a war, the Olympics, the Summit, a trap escape.
 *
 * The engine keeps only the roll made by the step just taken (`die_roll` is cleared at every
 * step), so a roll followed by more steps -- the other side's whole turn played at once -- is gone
 * from the current state. The rolls are read from the steps instead, and shown only while they are
 * recent: from the half action round they were made in and the next one (the other side's reply),
 * after which they are history and belong to the log.
 *
 * A step of the live session carries every die its click rolled (`ReplayStep.rolls`, the engine's
 * `clickRolls`): one click can roll more than once, and the turn's cleanup after the turn's last
 * action round clears the roll record from the state the click ends on. A step read from a replay
 * file has only its snapshot, which holds the click's last roll and not one cleared by cleanup.
 *
 * Everything here is public: dice are rolled in the open.
 */
import { DieRollState, GameState, RecentRoll } from "../types";
import { ReplayStep } from "../replay_controls";
import { Message, message, renderEnglish } from "../i18n";

const MAX_ROLLS = 4;

/**
 * A half action round as an order key: the headline is turn*100+1, then USSR AR1 = +2, US AR1 =
 * +3, ... The phases between them (a discard, the turn's end) are no half round of their own: null.
 */
function halfRound(turn: number, phase: string, ar: number, phasing: string): number | null {
  if (phase === "HEADLINE" || phase === "SETUP") return turn * 100 + 1;
  if (phase === "ACTION_ROUND" || phase === "INTERRUPT") return turn * 100 + ar * 2 + (phasing === "US" ? 1 : 0);
  return null;
}

/** When a step's roll was made, by the step's turn, phase and action round: "T3 Headline", "T3 AR2". */
export function rollWhen(turn: number, phase: string, ar: number): Message {
  return phase === "HEADLINE" ? message("common:round.headline", { turn }) : message("common:round.actionRound", { turn, ar });
}

/** The dice a step rolled, in order. */
function stepRolls(step: ReplayStep): DieRollState[] {
  if (step.rolls) return step.rolls;
  const r = step.state_snapshot?.die_roll;
  return r && r.type !== "NONE" ? [r] : [];
}

/**
 * The rolls of the latest half action round with any, oldest first (the latest four), when that
 * round is `now`'s or the one before it; otherwise none. `steps[0..end)` are the steps taken so
 * far, `start` the state before the first. Half rounds are counted as they begin, step by step, so
 * the turn's last action round and the next turn's headline are as adjacent as any two.
 */
export function recentRolls(start: GameState | null, steps: ReplayStep[], end: number, now: GameState): RecentRoll[] {
  const before = (k: number): GameState | null => (k === 0 ? start : steps[k - 1].state_snapshot);
  let half = halfRound(now.turn, now.current_phase_name ?? "", now.action_round, now.phasing_player);
  let begun = 0;   // half rounds begun after the one being read
  let found = false;
  const out: RecentRoll[] = [];
  for (let k = Math.min(end, steps.length) - 1; k >= 0; k--) {
    const s = steps[k];
    const rolls = stepRolls(s);
    const phasing = before(k)?.phasing_player ?? rolls[0]?.roller ?? "NONE";
    const h = halfRound(s.turn, s.phase, s.ar, phasing) ?? half;
    if (half !== null && h !== half) {
      if (found) break;              // every roll of that half round is read
      if (++begun > 1) break;        // older than the half round before now's
    }
    half = h;
    if (!rolls.length) continue;
    found = true;
    const whenMessage = rollWhen(s.turn, s.phase, s.ar);
    const when = renderEnglish(whenMessage);
    for (let i = rolls.length - 1; i >= 0 && out.length < MAX_ROLLS; i--) out.push({ ...rolls[i], when, whenMessage });
    if (out.length >= MAX_ROLLS) break;
  }
  return out.reverse();
}
