"""The observation, pinned exactly, so a change to it has to be deliberate.

Layout v2.3 is the only layout, and every live checkpoint was trained against it. A change to
any float here silently invalidates all of them: a network reads fixed slices, so it keeps
loading and simply misreads. "Byte for byte" is not something that can be argued; it has to be
pinned and compared.

This file is what proved the extractor was unchanged when the three-layout chain was flattened
into one function -- 4,288 observations across 40 games, not one float moved.

The golden file records a deterministic walk: fixed seed, and at every decision the lowest legal
flat action. That fixes the trajectory too, so the file also catches an engine change that alters
the decision stream -- which is the failure mode `tools/scripts/check_engine_fresh.sh` exists for,
caught here at the level of the actual numbers.

Regenerate deliberately, never to make a red test green:

    PYTHONPATH=.:build/release .venv/bin/python tests/engine_logic/test_observation_golden.py
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pytest

import ts_engine as ts

GOLDEN = os.path.join(os.path.dirname(__file__), "observation.golden.npz")

#: The one observation width, and a checkpoint contract.
OBS_SIZE = int(ts.OBS_SIZE)

_SEED = 20260907
_GAMES = 4
_STEPS_PER_GAME = 120


def _walk() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic games, capturing the observation from both sides at every decision.

    Both perspectives, because the card-location canonicalisation is per-observer and a change
    that got `MY_HAND` right and the opponent's hand wrong would pass a one-sided check.
    """
    obs_rows: List[np.ndarray] = []
    action_rows: List[int] = []
    marker_rows: List[Tuple[int, int, int, int]] = []

    for game in range(_GAMES):
        state = ts.GameState()
        ts.Engine.init_game(state, _SEED + game)
        for _ in range(_STEPS_PER_GAME):
            if ts.Engine.is_terminal(state):
                break
            mask = np.asarray(ts.get_flat_action_mask(state))
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            action = int(legal[0])
            for perspective in (ts.Player.US, ts.Player.USSR):
                obs_rows.append(
                    np.asarray(ts.extract_observation(state, perspective), dtype=np.float32))
            action_rows.append(action)
            marker_rows.append((int(state.turn), int(state.action_round),
                                int(state.victory_points), int(state.defcon)))
            ts.Engine.step_flat(state, action)

    return (np.stack(obs_rows), np.asarray(action_rows, dtype=np.int32),
            np.asarray(marker_rows, dtype=np.int32))


def _write_golden() -> None:
    obs, actions, markers = _walk()
    np.savez_compressed(GOLDEN, obs=obs, actions=actions, markers=markers)
    print(f"wrote {GOLDEN}")
    print(f"  {obs.shape[0]} observations of width {obs.shape[1]}, "
          f"{len(actions)} decisions over {_GAMES} games")


@pytest.mark.skipif(not os.path.exists(GOLDEN),
                    reason="golden not yet generated; run this file as a script")
def test_observation_is_unchanged() -> None:
    """Every legacy observation must match the golden file exactly."""
    golden = np.load(GOLDEN)
    obs, actions, markers = _walk()

    assert actions.tolist() == golden["actions"].tolist(), (
        "the decision stream changed: the engine no longer plays the same games from the same "
        "seed, so the observations below are not comparable and the difference is an engine "
        "change, not an observation change")
    assert markers.tolist() == golden["markers"].tolist(), (
        "turn/action-round/VP/DEFCON diverged from the golden walk")

    expected = golden["obs"]
    assert obs.shape == expected.shape, (
        f"observation width changed: {obs.shape} against {expected.shape}. The legacy "
        f"layout must stay {OBS_SIZE} wide or existing checkpoints stop loading.")

    if not np.array_equal(obs, expected):
        diff = np.flatnonzero((obs != expected).any(axis=0))
        raise AssertionError(
            f"{len(diff)} observation columns changed, first at index {int(diff[0])}. "
            f"Legacy observations must be bit-identical.")


@pytest.mark.skipif(not os.path.exists(GOLDEN), reason="golden not yet generated")
def test_golden_covers_both_perspectives_and_real_hands() -> None:
    """A golden of empty positions would pass any change; check it has content to protect."""
    golden = np.load(GOLDEN)
    obs = golden["obs"]
    assert obs.shape[0] >= 200, "too few positions to be a meaningful pin"
    assert obs.shape[1] == OBS_SIZE

    # I_AM_US is global feature 61 and must take both values, or only one side was walked.
    # (The old check read `active_player`, the last float of the retired layouts; v2.3 has no
    # such tail -- which side is to move is already in I_AM_US and in every perspective-relative
    # feature around it.)
    board, card_features = 84 * 26, 14
    globals_at = board + 110 * card_features
    values = set(obs[:, globals_at + 61].tolist())
    assert values == {0.0, 1.0}, f"expected both perspectives, saw {values}"

    # MY_HAND is card-feature slot 1 of 14, so cards 0..109 sit at board + 14*i + 1.
    my_hand = obs[:, board + 1: board + 110 * card_features: card_features]
    assert my_hand.sum() > 0, "no card was ever in the observer's hand"


if __name__ == "__main__":
    _write_golden()
