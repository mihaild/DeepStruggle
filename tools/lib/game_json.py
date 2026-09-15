"""The JSON boundary: one place that converts between the engine and everything outside it.

Inside a game loop there is exactly one action encoding -- the flat 212-index. JSON exists only at
the edge, for the browser, for replays and for bot protocols. Today those conversions are scattered
across `play_match`, `self_play`, `GameSession` and the replay logger, each doing part of it, and
one of them -- JSON back to an action -- does not exist at all. That missing direction is why the
viewer cannot re-drive a replay through the engine: it can render a recorded game but not check it.

With `json_to_action`, a replay becomes just another action source, and replay viewing and live
play become the same code path (see `research/plans/P13_one_game_driver.md`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

import numpy as np

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import IllegalActionError
from web.server.replay_types import GameStateDict, ReplayActionDict


def state_to_json(state: ts.GameState, for_role: Optional[str] = None) -> GameStateDict:
    """The state as the browser and the replay format see it.

    `for_role` attaches that seat's observation, which is what a client needs to show what the
    network would have been given. It is omitted when the role is not to move, since an
    observation for a player who cannot act is meaningless and invites being read as one.
    """
    d = cast(Dict[str, Any], state.to_dict())
    if for_role is not None:
        player = ts.Player.US if for_role.upper() == "US" else ts.Player.USSR
        ctx = state.ctx()
        mover = ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player
        if mover == player:
            obs = np.asarray(ts.extract_observation(state, player), dtype=np.float32)
            d["observation_f32"] = obs.tolist()
    return cast(GameStateDict, d)


def action_to_json(state: ts.GameState, flat: int) -> ReplayActionDict:
    """A flat action as the four fields a replay and the wire protocol carry.

    Decoded against `state`, because the flat space is state-dependent: the same index means
    different things at different decisions. Passing the wrong state here is the mistake that let a
    searcher hand back an action for a decision the caller was not at.
    """
    ma = ts.decode_flat_action(state, int(flat))
    return cast(ReplayActionDict, {
        "decision_type": int(ma.decision_type),
        "primary_id": int(ma.primary_id),
        "secondary_id": int(ma.secondary_id),
        "flags": int(ma.flags),
    })


def json_to_action(state: ts.GameState, obj: Dict[str, Any]) -> ts.MicroAction:
    """The inverse of `action_to_json`: the direction that did not exist.

    Returns a MicroAction rather than a flat index because not every legal action has one -- a
    forced die carries its value in `primary_id`, and the flat space has a single slot for "roll"
    (211) rather than one per value. Going through flat would silently turn a recorded roll of 3
    into a random one.
    """
    try:
        dt = ts.DecisionType(int(obj["decision_type"]))
    except (KeyError, ValueError) as exc:
        raise IllegalActionError(f"action json has no usable decision_type: {obj!r}") from exc
    return ts.MicroAction(dt, int(obj.get("primary_id", 0)),
                          int(obj.get("secondary_id", 0)), int(obj.get("flags", 0)))


def legal_actions_json(state: ts.GameState) -> Dict[str, Any]:
    """What a client needs to offer a choice: the decision being asked and the flat indices for it."""
    mask = np.asarray(ActionEncoder.get_legal_mask(state))
    ctx = state.ctx()
    return {
        "decision_type": int(ctx.decision_type),
        "decision_type_name": str(ts.DecisionType(int(ctx.decision_type))).split(".")[-1],
        "decision_player": str(ctx.decision_player).split(".")[-1],
        "flat_actions": [int(i) for i in np.flatnonzero(mask)],
    }
