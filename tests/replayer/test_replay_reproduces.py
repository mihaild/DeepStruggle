"""A saved replay must reproduce its game when its actions are re-applied.

test_replay_matches_training.py pins the *stepping contract* -- that the generator drains
chance nodes the way the vectorized runner does. It never re-applies a stored log, so the
property the log exists for ("we can generate the same games for replays as in training")
had no coverage: an engine change could silently make every archived replay unreplayable
and nothing would fail.

The ordering matters and is easy to get wrong. The generator records its snapshot *after*
draining the ROLL_DIE nodes that follow an action, so a reader that compares before
draining sees spurious mismatches -- a coup's die roll has not been applied yet. Drain
after stepping, then compare.

The first test generates its own replay so it works in a fresh checkout, where
data/replays is empty (it is gitignored). The second checks whatever replays are on disk,
and skips rather than fails when there are none.
"""

import glob
import json
import os
from typing import Any, Dict, List

import pytest
import ts_engine as ts

REPLAY_DIRS = ("data/replays", "replays")


def _drain(state: ts.GameState) -> None:
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_player == ts.Player.NONE
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


def _hand(cards: List[Any]) -> List[int]:
    return sorted(c["id"] if isinstance(c, dict) else c for c in cards)


def _digest(d: Dict[str, Any]) -> Dict[str, Any]:
    """The parts of a state a divergence would show up in first."""
    return {
        "vp": d["victory_points"], "defcon": d["defcon"], "turn": d["turn"],
        "ar": d["action_round"], "space": d["space"],
        "us_hand": _hand(d["hands"]["US_cards"]),
        "ussr_hand": _hand(d["hands"]["USSR_cards"]),
    }


def _assert_reproduces(log: Any, label: str) -> None:
    steps = log.get("steps") or []
    assert steps, f"{label}: replay has no steps"
    if "flat_action_idx" not in steps[0].get("action", {}):
        pytest.skip(
            f"{label}: pre-dates flat_action_idx in the log format, so its actions cannot "
            f"be re-applied at all"
        )

    state = ts.GameState()
    ts.Engine.init_game(state, log["metadata"]["seed"])

    for step in steps:
        _drain(state)
        assert not ts.Engine.is_terminal(state), (
            f"{label}: game ended at step {step['step_index']} but the log continues"
        )
        idx = step["action"]["flat_action_idx"]
        assert ts.Engine.try_step_flat(state, idx), (
            f"{label}: engine rejected logged action {idx} ({step['description']}) "
            f"at step {step['step_index']}"
        )
        _drain(state)

        got, want = _digest(ts.state_to_dict(state)), _digest(step["state_snapshot"])
        assert got == want, (
            f"{label}: diverged at step {step['step_index']} ({step['description']}); "
            f"logged {want}, replayed {got}"
        )


def test_a_freshly_generated_replay_reproduces(tmp_path) -> None:
    """Generate through the canonical path, then retrace it from the seed alone."""
    from ai.models.coldwar_net import create_coldwar_net
    from tools.lib.self_play import generate_self_play_replay

    out = str(tmp_path / "roundtrip.tslog.json")
    log, _ = generate_self_play_replay(
        model=create_coldwar_net("cpu"), seed=31337, temperature=1.0,
        game_id="roundtrip", output_path=out, device="cpu", verbose=False,
    )
    assert log["steps"], "generator produced an empty replay"
    _assert_reproduces(log, "freshly generated replay")


def _replay_files() -> List[str]:
    for directory in REPLAY_DIRS:
        found = sorted(glob.glob(os.path.join(directory, "*.tslog.json")))
        if found:
            return found
    return []


@pytest.mark.parametrize("path", _replay_files() or [pytest.param("", marks=pytest.mark.skip(
    reason="no .tslog.json replays on disk (data/replays is gitignored)"))])
def test_saved_replay_reproduces_exactly(path: str) -> None:
    """Every replay kept on disk must still retrace under the current engine."""
    with open(path) as fh:
        log = json.load(fh)
    _assert_reproduces(log, os.path.basename(path))
