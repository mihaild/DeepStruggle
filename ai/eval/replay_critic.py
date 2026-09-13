"""Ask a checkpoint's critic what it thinks of positions taken from a recorded game.

The critic is perspective-aligned: `extract_observation(state, p)` produces the board as `p` sees
it, and the value heads answer for `p`. Evaluating the *same* state from both perspectives is
therefore a consistency check with a known answer -- a zero-sum critic must satisfy
`v(US) = -v(USSR)`, and the residual `v(US) + v(USSR)` is pure model error. Anything systematic
in that residual is an asymmetry between how the network reads the two sides, which no
single-perspective number can reveal.

Two heads, both perspective-aligned: `v_win` in [-1, 1] and `v_vp` in [-20, 20].

Read the numbers as policy-conditional. V^pi is the value *under this policy's own continuation*,
so "the critic likes the position where the US controls Italy" and "the policy would take Italy"
are different claims, and the first does not imply the second.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts


def load_actions(path: str) -> Tuple[int, List[Dict[str, Any]]]:
    """(seed, per-step records) from a .tslog.json replay, in engine order."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    seed = int(d["metadata"]["seed"])
    steps = [{
        "step_index": int(s["step_index"]),
        "flat": int(s["action"]["flat_action_idx"]),
        "description": str(s.get("description", "")),
        "player": str(s.get("player", "")),
        "turn": s.get("turn"),
        "ar": s.get("ar"),
        "snapshot": s.get("state_snapshot") or {},
    } for s in d["steps"]]
    return seed, steps


def _resolve_chance(state: ts.GameState) -> None:
    """Resolve chance nodes exactly as the self-play generator and batch runner do."""
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_player == ts.Player.NONE
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


def replay_to(seed: int, steps: Sequence[Dict[str, Any]], upto: int) -> ts.GameState:
    """The state *after* the action logged at step index `upto`.

    Step indices are 1-based and the logged snapshot is the state after the action, so this
    matches what the replay shows at that row.
    """
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    for rec in steps:
        if rec["step_index"] > upto:
            break
        ts.Engine.step_flat(state, rec["flat"])
        _resolve_chance(state)
    return state


def verify(state: ts.GameState, rec: Dict[str, Any]) -> Optional[str]:
    """Compare a reconstructed state against the snapshot the replay recorded.

    A reconstruction that has silently diverged would still return values, so this is checked
    rather than assumed. Returns None when they agree, else a description of the mismatch.
    """
    snap = rec["snapshot"]
    if not snap:
        return None
    problems = []
    if "victory_points" in snap and int(snap["victory_points"]) != int(state.victory_points):
        problems.append(f"vp {state.victory_points} != {snap['victory_points']}")
    if "turn" in snap and int(snap["turn"]) != int(state.turn):
        problems.append(f"turn {state.turn} != {snap['turn']}")
    if "defcon" in snap and int(snap["defcon"]) != int(state.defcon):
        problems.append(f"defcon {state.defcon} != {snap['defcon']}")
    for cid in range(84):
        name = str(ts.MapData.get_country_info(cid)["name"])
        c = (snap.get("countries") or {}).get(name)
        if not c:
            continue
        got = state.get_country(cid)
        if (int(c.get("us_influence", 0)) != int(got.us_influence)
                or int(c.get("ussr_influence", 0)) != int(got.ussr_influence)):
            problems.append(
                f"{name} {got.us_influence}/{got.ussr_influence} != "
                f"{c.get('us_influence')}/{c.get('ussr_influence')}")
            break
    return "; ".join(problems) if problems else None


def evaluate(model: Any, state: ts.GameState) -> Dict[str, float]:
    """Both value heads, from both perspectives, plus the zero-sum residuals."""
    import torch

    device = next(model.parameters()).device
    out: Dict[str, float] = {}
    for tag, p in (("us", ts.Player.US), ("ussr", ts.Player.USSR)):
        obs = np.asarray(ts.extract_observation(state, p), dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            _, v_win, v_vp = model(torch.from_numpy(obs).to(device), None)
        out[f"v_win_{tag}"] = float(v_win.item())
        out[f"v_vp_{tag}"] = float(v_vp.item())
    # Zero-sum: these must be 0 for a consistent critic. They are the asymmetry.
    out["win_residual"] = out["v_win_us"] + out["v_win_ussr"]
    out["vp_residual"] = out["v_vp_us"] + out["v_vp_ussr"]
    return out


def control(state: ts.GameState, cid: int) -> str:
    c = state.get_country(cid)
    u, s = int(c.us_influence), int(c.ussr_influence)
    stab = int(ts.MapData.get_country_info(cid)["stability"])
    who = "USSR" if s - u >= stab else "US" if u - s >= stab else "--"
    return f"{u}/{s} {who}"
