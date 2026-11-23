/**
 * A reload's playback of the game this browser stored (game/saved_game.ts), seated as it was played.
 *
 * GameSession.replayRecord seats each step from the record: the step's setting, its model's
 * obs_features, the person rules. A record stored before ModelInfo held obs_features (2026-11-23)
 * gets them from the model itself -- the file the record names by sha256, fetched from its source
 * and required to hash to it -- and the person rules are tried as the Python verifier tries a build
 * it cannot place (tools/lib/game_stats.py PERSON_RULES_HISTORY), newest first: the record does not
 * say which deploy it was played on, and the one that lands on the link's position is the game.
 */
import { GameSession } from "./session";
import { GameRecordData, ModelInfo } from "./record";

/** The rule bits a person's side has played on this site, newest first after the page's current ones:
 *  opp_event_gated (f383abd, 2026-11-06), then none. Mirrors tools/lib/game_stats.py. */
const OPP_EVENT_GATED = 1 << 3;
export function personRuleCandidates(current: number): number[] {
  return [...new Set([current, OPP_EVENT_GATED, 0])];
}

/** A model file the record names: its sha256 and the obs_features it reads. Rejects when it cannot be had. */
export type ObtainModel = (info: ModelInfo, sha256: string) => Promise<{ sha256: string; features: number }>;

/** `rec` with every model an auto-played step moved under holding its obs_features. Rejects, naming the
 *  model, when one cannot be obtained or is now another file. */
export async function withModelBits(rec: GameRecordData, obtain: ObtainModel): Promise<GameRecordData> {
  const models: Record<string, ModelInfo> = { ...rec.models };
  const needed = new Set(rec.steps.map(s => rec.settings[s.tag.setting])
    .filter(st => st && st.auto && st.model).map(st => st!.model));
  for (const sha of needed) {
    const info = models[sha];
    if (!info) throw new Error(`model ${sha.slice(0, 12)} is not among the saved game's models`);
    if (info.obs_features !== undefined) continue;
    const got = await obtain(info, sha);
    if (got.sha256 !== sha) {
      throw new Error(`${info.label}: ${info.source || "its file"} is now another file (sha256 ${got.sha256.slice(0, 12)}, `
        + `the game was played with ${sha.slice(0, 12)})`);
    }
    models[sha] = { ...info, obs_features: got.features };
  }
  return { ...rec, models };
}

/** Play `rec` back into `session` under each candidate person rule set until one lands on `target`
 *  (the link's position): true if one did. A refused playback is a candidate that is not the game. */
export function replayToPosition(session: GameSession, rec: GameRecordData, start: string | null, submitted: boolean,
                                 target: string, current: number, saveJson: () => string): boolean {
  for (const person of personRuleCandidates(current)) {
    try {
      session.replayRecord(rec, start, submitted, person);
      if (saveJson() === target) return true;
    } catch {
      // not this rule set
    }
  }
  return false;
}
