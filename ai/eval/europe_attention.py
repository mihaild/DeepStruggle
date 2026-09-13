"""Why one network attacks Europe as the USSR but not as the US.

Both sides are the same weights reading a perspective-aligned observation, so a behavioural
asymmetry has to come from somewhere nameable. Three candidates, and this separates them:

1. **Legality.** Influence placement needs presence or adjacency-to-control. A US with 0 in West
   Germany and no controlled neighbour *cannot* place there at all. "Never opens a new front"
   would then be a constraint, not a preference, and no amount of training would change it.
2. **Preference.** The action is legal and the policy puts little mass on it.
3. **Opportunity.** The situation simply arises less often for one side.

Measured over states sampled from real self-play, at INFLUENCE placement decisions only, split by
acting side. `legal_rate` separates (1); `mass` conditioned on legality separates (2) from (3).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import ts_engine as ts

#: The five European battlegrounds -- the countries Europe control actually turns on.
EUROPE_BG = [7, 8, 10, 14, 15]
NAMES = {c: str(ts.MapData.get_country_info(c)["name"]) for c in EUROPE_BG}
NODE_OFFSET = 119


def _controls(state: ts.GameState, cid: int, player: ts.Player) -> bool:
    c = state.get_country(cid)
    u, s = int(c.us_influence), int(c.ussr_influence)
    stab = int(ts.MapData.get_country_info(cid)["stability"])
    return (u - s >= stab) if player == ts.Player.US else (s - u >= stab)


def measure(model: Any, num_games: int = 128, temperature: float = 1.0,
            sample_every: int = 5, max_iters: int = 4000) -> Dict[str, Any]:
    import torch

    from bindings.action_encoder import ActionEncoder
    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    model.eval()

    env = TsVectorizedEnv(num_envs=num_games, base_seed=909_000)
    obs, masks, _ = env.reset_all()

    # per side: decisions seen, times the country was a legal target, summed probability mass
    stats: Dict[str, Dict[str, Any]] = {
        side: {"decisions": 0,
               "legal": {c: 0 for c in EUROPE_BG},
               "mass": {c: 0.0 for c in EUROPE_BG},
               "mass_when_legal": {c: 0.0 for c in EUROPE_BG},
               "opp_holds_legal": {c: 0 for c in EUROPE_BG},
               "opp_holds_mass": {c: 0.0 for c in EUROPE_BG},
               # Bucketed by turn, because the raw conditional is confounded: a side only
               # faces an opponent-held France in the part of the game where that happened,
               # and those are not the same positions for the two sides.
               "by_turn_n": {c: {} for c in EUROPE_BG},
               "by_turn_mass": {c: {} for c in EUROPE_BG}}
        for side in ("US", "USSR")}

    for it in range(max_iters):
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
        mask_t = torch.from_numpy(np.asarray(masks)).to(device)
        with torch.no_grad():
            acts, *_ = model.sample_action(obs_t, mask_t, temperature=temperature)

        if it % sample_every == 0:
            with torch.no_grad():
                logits, _, _ = model(obs_t, mask_t)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for i in range(num_games):
                st = env.runner.get_state(i)
                if ts.Engine.is_terminal(st):
                    continue
                ctx = st.ctx()
                if ctx.decision_type != ts.DecisionType.POINT_NODE:
                    continue
                p = ctx.decision_player
                if p == ts.Player.NONE:
                    continue
                side = "US" if p == ts.Player.US else "USSR"
                opp = ts.Player.USSR if p == ts.Player.US else ts.Player.US
                m = np.asarray(masks[i])
                s = stats[side]
                s["decisions"] += 1
                for c in EUROPE_BG:
                    idx = NODE_OFFSET + c
                    is_legal = bool(m[idx])
                    s["legal"][c] += int(is_legal)
                    s["mass"][c] += float(probs[i][idx])
                    if is_legal:
                        s["mass_when_legal"][c] += float(probs[i][idx])
                        if _controls(st, c, opp):
                            s["opp_holds_legal"][c] += 1
                            s["opp_holds_mass"][c] += float(probs[i][idx])
                            t = int(st.turn)
                            s["by_turn_n"][c][t] = s["by_turn_n"][c].get(t, 0) + 1
                            s["by_turn_mass"][c][t] = (
                                s["by_turn_mass"][c].get(t, 0.0) + float(probs[i][idx]))

        obs, masks, _, dones, _ = env.step(acts.cpu().numpy().astype(np.int64))
        if it > 400 and all(bool(d) for d in dones):
            break

    rows: List[Dict[str, Any]] = []
    for side in ("US", "USSR"):
        s = stats[side]
        n = max(s["decisions"], 1)
        for c in EUROPE_BG:
            legal_n = max(s["legal"][c], 1)
            held_n = max(s["opp_holds_legal"][c], 1)
            rows.append({
                "side": side,
                "country": NAMES[c],
                "decisions": s["decisions"],
                "legal_rate": s["legal"][c] / n,
                "mass_when_legal": s["mass_when_legal"][c] / legal_n,
                "opp_holds_n": s["opp_holds_legal"][c],
                "mass_when_opp_holds": s["opp_holds_mass"][c] / held_n,
                "by_turn": {t: (s["by_turn_mass"][c][t] / s["by_turn_n"][c][t],
                                s["by_turn_n"][c][t])
                            for t in sorted(s["by_turn_n"][c])},
            })
    return {"rows": rows, "decisions": {k: stats[k]["decisions"] for k in stats}}
