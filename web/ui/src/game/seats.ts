/**
 * The rule bits a side plays (GameState::seat_rules, engine/AGENTS.md on rule bits): the auto-play side
 * its model's obs_features, a person's side -- and an auto side with no model -- the person rules. One
 * rule for the live game (main.ts `seatSides`) and for a game played back from its record
 * (GameSession.replayRecord), so a reload re-drives every step under the seats it was played under.
 */
import { Side } from "./record";

export function sideRules(side: Side, auto: Side | "", modelFeatures: number | null, personRules: number): number {
  return side === auto && modelFeatures !== null ? modelFeatures : personRules;
}

export const SEATS: ReadonlyArray<readonly [Side, number]> = [["US", 1], ["USSR", -1]];
