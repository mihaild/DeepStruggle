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
from bindings.action_encoder import ActionEncoder

#: The five European battlegrounds -- the countries Europe control actually turns on.
EUROPE_BG = [7, 8, 10, 14, 15]
NAMES = {c: str(ts.MapData.get_country_info(c)["name"]) for c in EUROPE_BG}
NODE_OFFSET = ActionEncoder.NODE_OFFSET   # one source; see bindings/action_encoder.py


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


#: The three cost regimes a placement decision can be in, for a given country and mover.
#: The distinction is the whole point: `get_influence_cost` charges 2 Ops per point only in a
#: country the *opponent controls*. A country nobody controls costs the normal 1 per point.
#:
#:   i_control   -- I control it. Reinforcing, 1 Op/point.
#:   contested   -- nobody controls it. Taking control costs 1 Op/point.
#:   opp_control -- the opponent controls it. Breaking costs 2 Ops/point.
#:
#: The comparison that matters for "the USSR pays double to break, the US will not pay single to
#: restore" is USSR mass in `opp_control` against US mass in `contested` -- not the two sides'
#: `opp_control` numbers, which is a different question and the one measured first.
#: `contested` is split further, because lumping the two together answers the wrong question:
#: placing into an empty country is opening a front, while placing into one where both sides
#: already sit is *restoring* a control the opponent just broke -- the second is what "the US
#: will not pay the normal price to take it back" is about, and it is a small subset of the first.
REGIMES = ("i_control", "contested_empty", "contested_both", "opp_control")


def regime_of(state: ts.GameState, cid: int, me: Any, opp: Any) -> str:
    if _controls(state, cid, me):
        return "i_control"
    if _controls(state, cid, opp):
        return "opp_control"
    c = state.get_country(cid)
    both = int(c.us_influence) > 0 and int(c.ussr_influence) > 0
    return "contested_both" if both else "contested_empty"


def measure_by_regime(model: Any, num_games: int = 256, temperature: float = 1.0,
                      sample_every: int = 5, max_iters: int = 4000) -> Dict[str, Any]:
    """Probability mass on each European battleground, split by who controls it."""
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    model.eval()

    env = TsVectorizedEnv(num_envs=num_games, base_seed=515_000)
    obs, masks, _ = env.reset_all()

    n: Dict[Any, int] = {}
    mass: Dict[Any, float] = {}

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
                for c in EUROPE_BG:
                    idx = NODE_OFFSET + c
                    if not m[idx]:
                        continue  # illegal: not a choice the policy declined
                    key = (side, NAMES[c], regime_of(st, c, p, opp))
                    n[key] = n.get(key, 0) + 1
                    mass[key] = mass.get(key, 0.0) + float(probs[i][idx])

        obs, masks, _, dones, _ = env.step(acts.cpu().numpy().astype(np.int64))
        if it > 400 and all(bool(d) for d in dones):
            break

    return {"n": n, "mass": mass}
