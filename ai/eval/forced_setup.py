"""Hand the policy a standard human opening and see whether it defends it.

Self-play arms since E3-17 end a growing share of games with a USSR Europe-control win, and the
generated replays all show the same cause: the US never places in West Germany, so the USSR needs
no combat there to complete the region. That leaves a question the replays cannot answer -- is the
US *unable* to hold West Germany, or has it simply never been in a position where it started with
one? A policy that only ever fights where it was placed would look identical either way.

So: overwrite the fifteen setup decisions with the standard human opening and let the policy play
from there. The control arm is the same games with the policy choosing its own setup, so the two
differ only in the opening.

    USSR  +1 East Germany, +4 Poland, +1 Yugoslavia
    US    +4 West Germany, +3 Italy, +2 Iran

Read the result off West Germany, not off the win rate alone. The win rate answers "is this
opening better", which is not the question; what is being asked is whether the US contests a
battleground it was handed, which is `wg_us_end` and `wg_us_control_*` below.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import ts_engine as ts

# The opening itself lives in `tools.lib.openings` so that this probe and the replay
# generators cannot drift apart -- a replay labelled "the human opening" has to be the opening
# these numbers were measured on.
from tools.lib.openings import (  # noqa: E402
    NODE_OFFSET, SETUP_DECISIONS, US_OPENING, USSR_OPENING, WEST_GERMANY, acting_side, expand)
from tools.lib.openings import EAST_GERMANY, IRAN, ITALY, POLAND, YUGOSLAVIA  # noqa: E402,F401

_expand = expand


def _influence(state: ts.GameState, cid: int) -> Tuple[int, int]:
    c = state.get_country(cid)
    return int(c.us_influence), int(c.ussr_influence)


def measure(model: Any, num_games: int = 512, forced: bool = True,
            temperature: float = 1.0, base_seed: int = 414_000,
            max_iters: int = 20_000) -> Dict[str, Any]:
    """Play `num_games` from a forced (or free) opening and report what became of Europe."""
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    model.eval()

    env = TsVectorizedEnv(num_envs=num_games, base_seed=base_seed)
    obs, masks, _ = env.reset_all()

    ussr_script, us_script = _expand(USSR_OPENING), _expand(US_OPENING)
    # Per-env cursors: the two sides place in a fixed order, but which side is acting on a given
    # setup step is read from the engine rather than assumed.
    cursor = {"US": [0] * num_games, "USSR": [0] * num_games}
    off_script = 0

    def scripted(step_masks: np.ndarray) -> np.ndarray:
        nonlocal off_script
        acts = np.zeros(num_games, dtype=np.int64)
        # Whose placement this is comes from the decision player, not `phasing_player`: the
        # USSR is the phasing player for the whole of setup, so keying off it sent all nine US
        # placements down the USSR script, which then ran out -- exactly 9 off-script rows per
        # game, with the US silently placing whatever was first in the mask.
        players = env.runner.get_decision_players()
        for i in range(num_games):
            side = "US" if int(players[i]) == int(ts.Player.US) else "USSR"  # see acting_side
            script = us_script if side == "US" else ussr_script
            k = cursor[side][i]
            chosen = -1
            if k < len(script):
                cand = NODE_OFFSET + script[k]
                if step_masks[i][cand]:
                    chosen = cand
                    cursor[side][i] = k + 1
            if chosen < 0:
                # Never silently substitute a different opening: count it and take the first
                # legal action so the run continues, then fail the whole measurement if any
                # env needed this. A partly-forced setup measures neither opening.
                off_script += 1
                chosen = int(np.flatnonzero(step_masks[i])[0])
            acts[i] = chosen
        return acts

    if forced:
        for _ in range(SETUP_DECISIONS):
            m = np.asarray(masks)
            obs, masks, _, _, _ = env.step(scripted(m))
        if off_script:
            raise RuntimeError(
                f"{off_script} setup placements could not follow the script -- the measurement "
                "would be of neither opening. Check the country ids and the placement order.")

    finished = [False] * num_games
    winners: List[str] = []
    reasons: List[str] = []
    wg_us_end: List[int] = []
    wg_ussr_end: List[int] = []
    europe_control = 0
    games = 0

    for _ in range(max_iters):
        if all(finished):
            break
        with torch.no_grad():
            acts, *_ = model.sample_action(
                torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(masks)).to(device), temperature=temperature)
        a = acts.cpu().numpy().astype(np.int64)
        # Cloned for the same reason as in held_scoring: get_state is a live view onto the
        # runner slot, and auto-reset rewinds it before step() returns.
        prev: List[Optional[ts.GameState]] = [
            env.runner.get_state(i).clone() if not finished[i] else None
            for i in range(num_games)]
        obs, masks, _, dones, info = env.step(a)

        for i, reason in enumerate(info["ending_reasons"]):
            if not reason or finished[i]:
                continue
            finished[i] = True
            games += 1
            st = prev[i]
            if st is None:
                continue
            reasons.append(reason)
            vp = int(info["victory_points"][i])
            winners.append("US" if vp > 0 else "USSR" if vp < 0 else "draw")
            u, s = _influence(st, WEST_GERMANY)
            wg_us_end.append(u)
            wg_ussr_end.append(s)
            if reason == "europe_control":
                europe_control += 1

    wg_u = np.asarray(wg_us_end, dtype=float)
    wg_s = np.asarray(wg_ussr_end, dtype=float)
    n = max(len(wg_u), 1)
    return {
        "forced": forced,
        "games": games,
        "off_script": off_script,
        "us_win_rate": winners.count("US") / max(len(winners), 1),
        "europe_control_rate": europe_control / max(games, 1),
        "wg_us_mean": float(wg_u.mean()) if len(wg_u) else float("nan"),
        "wg_ussr_mean": float(wg_s.mean()) if len(wg_s) else float("nan"),
        "wg_us_zero": float((wg_u == 0).sum()) / n,
        # West Germany is stability 4, so control needs a 4-point lead.
        "wg_us_control": float((wg_u - wg_s >= 4).sum()) / n,
        "wg_ussr_control": float((wg_s - wg_u >= 4).sum()) / n,
        "reason_mix": {r: reasons.count(r) / max(len(reasons), 1) for r in sorted(set(reasons))},
    }
