"""Synthetic stats-database submissions for tools/lib/game_stats.py's tests: games really played in this
build, each side seated as the page seats it (web/ui/src/main.ts `seatSides`), and recorded as
web/ui/src/game/record.ts records them. No real game, login or id: the database is private."""
from __future__ import annotations

import base64
import os
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import ts_engine as ts

from ai.models.ladder_net import create_ladder_net
from bindings.action_encoder import ActionEncoder
from tools.lib.game_stats import SettingsDict, Step, final_hash, outcome, play_step, start_state, summarize

SHA = "c" * 64
CHECKPOINT = "m.pt"
PERSON = int(ts.OBS_FEATURES_PERSON_RULES)
#: A model of the E8.1 kind: the headline reveal and the opponent's-card Event gate.
MODEL = int(ts.OBS_FEATURE_HEADLINE_REVEAL) | int(ts.OBS_FEATURE_OPP_EVENT_GATED)
_TINY: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
    card_self_attention=False, cross_attention=False, per_entity_heads=16, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def tiny_checkpoint(directory: str, name: str, features: int) -> str:
    """A small LadderNet reading `features`, saved as a checkpoint at directory/name."""
    path = os.path.join(directory, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(create_ladder_net("cpu", **_TINY, obs_features=features).state_dict(), path)
    return path


def encode_log(seed: int, steps: List[Step]) -> str:
    """record.ts encodeLog."""
    raw = bytearray(struct.pack("<BII", 1, seed, len(steps)))
    for st in steps:
        info = (("human", "suggested", "auto").index(st.source) | (4 if st.analysis else 0)
                | ({"US": 1, "USSR": 2}.get(st.mover, 0) << 3) | (32 if st.rolled else 0))
        raw += bytes([st.decision_type, st.primary_id, st.secondary_id, st.flags, st.forced_die, info, st.setting])
    return base64.urlsafe_b64encode(zlib.compress(bytes(raw))).decode().rstrip("=")


def _mover(state: ts.GameState) -> str:
    return {ts.Player.US: "US", ts.Player.USSR: "USSR"}.get(state.ctx().decision_player, "")


def play_seated(seed: int, auto_sides: List[str], model: int, person: int, *,
                switch_at: Optional[int] = None) -> Tuple[List[Step], ts.GameState]:
    """A whole game of random legal moves, seated as the page seats it: the auto-play side (setting 0's,
    or setting 1's from step `switch_at`) with `model`, a person's side with `person`."""
    rng = np.random.default_rng(seed)
    state = start_state(seed, None)
    steps: List[Step] = []
    while not ts.Engine.is_terminal(state):
        setting = 1 if switch_at is not None and len(steps) >= switch_at else 0
        auto = auto_sides[setting]
        state.seat_obs_features(ts.Player.US, model if auto == "US" else person)
        state.seat_obs_features(ts.Player.USSR, model if auto == "USSR" else person)
        mover = _mover(state)
        flat = int(rng.choice(np.flatnonzero(ActionEncoder.get_legal_mask(state))))
        action = ActionEncoder.decode(state, flat)
        st = Step(int(action.decision_type), action.primary_id, action.secondary_id, action.flags, 0,
                  "auto" if mover == auto else "human", False, mover, False, setting)
        play_step(state, st)
        steps.append(st)
    return steps, state


def submission(seed: int, steps: List[Step], state: ts.GameState, auto_sides: List[str], fingerprint: str,
               *, game: str = "game-0000") -> Dict[str, Any]:
    """The page's submission of a game played by `play_seated`."""
    human = "USSR" if auto_sides[0] == "US" else "US"
    settings: List[SettingsDict] = [{"view": human, "auto": side, "model": SHA, "search": "policy"}
                                    for side in auto_sides]
    return {
        "schema": 1, "id": game, "started_at": "2026-11-20T10:00:00Z", "ended_at": "2026-11-20T11:00:00Z",
        "build": {"commit": "0123456789abcdef", "dirty": False, "engine_fingerprint": fingerprint},
        "playdek": {"login": "someone", "rating": 1700}, "note": "",
        "result": outcome(state), "settings": settings,
        "models": {SHA: {"label": "a test model", "source": "", "checkpoint": CHECKPOINT}},
        "events": [], "start_position": None, "log": encode_log(seed, steps), "n_steps": len(steps),
        "final_hash": final_hash(state),
        "summary": summarize(steps, settings, [], None, finished=True),
    }
