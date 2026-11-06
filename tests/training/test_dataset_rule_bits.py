"""A (seed, actions) game replays under the rules its seats played (obs_features rule bits).

The generators record each seat's feature set (`us_obs_features` / `ussr_obs_features`), and the
replay seats them before the first action. A game without them predates the first rule bit
(OPP_EVENT_GATED, ENG-9), so it is replayed under the rule from before it -- which offers everything
the current rule does, so a game played under either replays whole.
"""

import gzip
import json
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np
import ts_engine as ts

from ai.training.warmup_dataset_loader import WarmupDataset
from bindings.ts_env import legal_mask
from tools.lib.game_step import drain_chance

EVENT = 110
GATED = ts.OBS_FEATURE_OPP_EVENT_GATED


def _fizzles(state: ts.GameState) -> bool:
    """At this node an opponent's card offers its Event under the pre-fix rule, and it cannot fire."""
    ctx = state.ctx()
    if ctx.decision_type != ts.DecisionType.SELECT_PLAY_MODE or ctx.decision_player == ts.Player.NONE:
        return False
    card = int(ctx.pending_op_card)
    if not 1 <= card <= 110:
        return False
    side = ts.CardData.get_card_info(card)["side"]
    p = ctx.decision_player
    owner = ts.Player.US if side == "US" else (ts.Player.USSR if side == "USSR" else ts.Player.NONE)
    if owner == ts.Player.NONE or owner == p:
        return False
    return not ts.CardHandlers.can_trigger_event(state, card, owner)


def _legacy_game_with_a_fizzle() -> Tuple[Dict[str, Any], int]:
    """A game played under the pre-fix rule (feature set 0) by a policy that takes the fizzle where
    it is offered and the lowest legal action elsewhere, recorded the way the generators record it,
    up to a few decisions past its first fizzle. Returns the record and that fizzle's index."""
    for seed in range(1, 200):
        st = ts.GameState()
        ts.Engine.init_game(st, seed)
        actions: List[Dict[str, int]] = []
        fizzle_at = -1
        for _ in range(3000):
            drain_chance(st)
            if ts.Engine.is_terminal(st) or (fizzle_at >= 0 and len(actions) > fizzle_at + 5):
                break
            mask = legal_mask(st, False, 0)
            if fizzle_at < 0 and _fizzles(st) and mask[EVENT]:
                a, fizzle_at = EVENT, len(actions)
            else:
                a = int(np.flatnonzero(mask)[0])
            actions.append({"flat_action": a})
            ts.Engine.step_flat(st, a, False, False)
        if fizzle_at >= 0:
            return {"seed": seed, "actions": actions}, fizzle_at
    raise AssertionError("no game reached an opponent card whose Event cannot fire")


def _replayed(tmp_path: pathlib.Path, game: Dict[str, Any]) -> int:
    path = tmp_path / "games.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(game) + "\n")
    return sum(1 for _ in WarmupDataset(str(path))._replay(None, None))


def test_a_game_replays_under_the_rules_its_seats_played(tmp_path: pathlib.Path) -> None:
    game, fizzle_at = _legacy_game_with_a_fizzle()
    n = len(game["actions"])
    # Unrecorded: from before the first rule bit, so the pre-fix rule, and the fizzle replays.
    assert _replayed(tmp_path, game) == n
    assert _replayed(tmp_path, {**game, "us_obs_features": 0, "ussr_obs_features": 0}) == n
    # Recorded as the current rule, the same actions stop at the fizzle, which that rule refuses.
    assert _replayed(tmp_path, {**game, "us_obs_features": GATED, "ussr_obs_features": GATED}) == fizzle_at
