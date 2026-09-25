/**
 * What a model thinks of the position on the board: its policy over the open decision -- in its
 * own action view -- and both value heads from both perspectives.
 *
 * The port of the server-side readout (web/server/analysis.py `analyze`, over ai/eval/
 * policy_readout.py's read_policy/read_critic): the same softmax over the legal actions at
 * temperature 1, the same argmax as "the favourite", the same critic table and residuals. One
 * batched forward of three rows: the decider's observation with its real mask (policy), and the
 * US and USSR observations for the critic. Python asks the critic with no mask at all; the ONNX
 * graph takes one, so the critic rows get all-ones -- which tools/export_onnx.py proves changes
 * no value head before it writes a model.
 */
import { WasmEngine, PLAYER_US, PLAYER_USSR } from "../engine/wasm_engine";
import { actionName } from "../game/names";
import { Model } from "./model";
import { CriticTrace, PolicyTrace } from "../replay_controls";
import { AnalysisChoice, LiveAnalysis } from "../analysis_view";

export async function analyze(engine: WasmEngine, model: Model, modelKey: string): Promise<LiveAnalysis> {
  const obsSize = engine.obsSize;
  const act = engine.actionSize;
  const merged = model.meta.mergedInfluence;
  const decider = engine.isTerminal() ? 0 : engine.decisionPlayer();

  const obs = new Float32Array(3 * obsSize);
  const masks = new Uint8Array(3 * act).fill(1);
  const legalMask = decider !== 0 ? engine.mask(merged) : null;
  if (decider !== 0 && legalMask) {
    obs.set(engine.observation(decider), 0);
    masks.set(legalMask, 0);
  } else {
    obs.set(engine.observation(PLAYER_US), 0);
  }
  obs.set(engine.observation(PLAYER_US), obsSize);
  obs.set(engine.observation(PLAYER_USSR), 2 * obsSize);
  const out = await model.run(obs, masks, 3);

  const r6 = (v: number) => Math.round(v * 1e6) / 1e6;
  const critic: CriticTrace = {
    v_win_us: r6(out.vWin[1]), v_win_ussr: r6(out.vWin[2]),
    v_vp_us: r6(out.vVp[1]), v_vp_ussr: r6(out.vVp[2]),
    win_residual: r6(out.vWin[1] + out.vWin[2]), vp_residual: r6(out.vVp[1] + out.vVp[2]),
    at: "position",
  };
  const analysis: LiveAnalysis = {
    model: modelKey, label: model.meta.label, merged_influence: merged, critic, choices: [],
  };
  if (decider === 0 || !legalMask) return analysis;

  // Softmax over the legal actions only, at temperature 1.
  const legal: number[] = [];
  for (let i = 0; i < act; i++) if (legalMask[i]) legal.push(i);
  let maxLogit = -Infinity;
  for (const i of legal) maxLogit = Math.max(maxLogit, out.logits[i]);
  let z = 0;
  const p = new Map<number, number>();
  for (const i of legal) { const e = Math.exp(out.logits[i] - maxLogit); p.set(i, e); z += e; }
  let entropy = 0;
  let argmax = legal[0];
  for (const i of legal) {
    const q = p.get(i)! / z;
    p.set(i, q);
    if (q > 0) entropy -= q * Math.log(q);
    if (q > p.get(argmax)!) argmax = i;
  }

  const policy: PolicyTrace = {
    source: "analysis", temperature: 1, n_legal: legal.length, chosen_idx: argmax,
    p_chosen: r6(p.get(argmax)!), p_chosen_sampled: 1, argmax_idx: argmax, p_max: r6(p.get(argmax)!),
    entropy: r6(entropy), p_tail: 0, v_win: r6(out.vWin[0]), v_vp: r6(out.vVp[0]),
  };
  analysis.policy = policy;
  analysis.decision_player = decider > 0 ? "US" : "USSR";
  analysis.decision_type = engine.decisionType();

  const L = engine.layout;
  const choices: AnalysisChoice[] = legal.map(idx => {
    const entry: AnalysisChoice = { idx, p: r6(p.get(idx)!) };
    if (merged && engine.isMergedInfluenceAction(idx)) {
      // A composed E4.1 action: "Ops for influence" plus its first placement -- one action for
      // the model, two clicks for a person. Its MicroAction fields name the commit half.
      const commit = engine.decodeFlat(L.ops_influence_index);
      entry.composed = true;
      entry.decision_type = commit.decision_type;
      entry.primary_id = commit.primary_id;
      entry.flags = 0;
      if (idx !== L.ops_influence_index) {
        const cid = idx - L.offsets.node;
        entry.country_id = cid;
        entry.name = `Ops → Influence, first point in ${engine.countryName(cid)}`;
      } else {
        entry.name = "Ops → Influence, place nothing";
      }
      return entry;
    }
    const ma = engine.decodeFlat(idx);
    entry.decision_type = ma.decision_type;
    entry.primary_id = ma.primary_id;
    entry.flags = ma.flags;
    entry.name = actionName(engine, idx);
    return entry;
  });
  choices.sort((a, b) => b.p - a.p);
  analysis.choices = choices;
  return analysis;
}
