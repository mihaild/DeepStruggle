"""The X4a pipeline: a searcher's visit distribution has to survive the round trip to disk.

The dataset format stores `(seed, actions)` and replays through the engine, so nothing about a
position is stored directly — which makes the round trip the thing to test. A target that lands on
the wrong position, or keeps mass on an action the engine calls illegal, trains the policy toward
a move it cannot play.

Built on a synthetic game rather than a checkpoint: `data/` is git-ignored, so no test may assume
it holds anything.
"""

from __future__ import annotations

import gzip
import json
from typing import Any, Dict, List

import numpy as np
import pytest
import ts_engine as ts

from ai.training.warmup_dataset_loader import WarmupDataset
from bindings.action_encoder import ActionEncoder


def _legal(st: ts.GameState) -> List[int]:
    return np.flatnonzero(np.asarray(ts.get_flat_action_mask(st))).tolist()


def _drain(st: ts.GameState) -> None:
    while (not ts.Engine.is_terminal(st)
           and st.ctx().decision_player == ts.Player.NONE
           and st.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


def _make_game(seed: int, n_actions: int = 40, every: int = 3) -> Dict[str, Any]:
    """A replayable game whose every `every`-th decision carries a search target.

    The target is deliberately NOT the played action: the point of expert iteration is that the
    teacher disagrees with the actor, and a test where they coincide would pass with the target
    silently ignored.
    """
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    actions: List[Dict[str, Any]] = []
    for i in range(n_actions):
        if ts.Engine.is_terminal(st):
            break
        legal = _legal(st)
        if not legal:
            break
        played = legal[0]
        rec: Dict[str, Any] = {"flat_action": played}
        if i % every == 0 and len(legal) >= 2:
            # weight the LAST legal action highest, which is not the one played
            rec["search_pi"] = {"a": legal[:3], "v": [1.0, 2.0, 7.0][:len(legal[:3])]}
        actions.append(rec)
        ts.Engine.step(st, ts.decode_flat_action(st, played))
        _drain(st)
    return {"seed": seed, "actions": actions, "winner": "DRAW", "final_vp": 0}


def _write(tmp_path, games: List[Dict[str, Any]]) -> str:
    p = tmp_path / "targets.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")
    return str(p)


def test_only_searched_decisions_yield_a_target(tmp_path) -> None:
    game = _make_game(1234, n_actions=30, every=3)
    path = _write(tmp_path, [game])
    got = list(WarmupDataset(path).stream_policy_transitions())
    expected = sum(1 for a in game["actions"] if "search_pi" in a)
    assert expected > 0, "the fixture produced no targets; it is not testing anything"
    assert len(got) == expected, (
        "a decision without search_pi produced a target -- behaviour cloning of the acting "
        "policy has crept back in")


def test_targets_are_normalised_and_inside_the_mask(tmp_path) -> None:
    path = _write(tmp_path, [_make_game(99), _make_game(100)])
    n = 0
    for obs, mask, pi, dt in WarmupDataset(path).stream_policy_transitions():
        n += 1
        assert obs.shape[0] == ts.OBS_SIZE
        assert pi.shape == mask.shape
        assert pi.sum() == pytest.approx(1.0, abs=1e-5), "target is not a distribution"
        assert float(pi[mask == 0].sum()) == 0.0, "target has mass on an illegal action"
    assert n > 0


def test_mass_on_an_illegal_action_is_dropped_not_trusted(tmp_path) -> None:
    """The mask is the authority. A stale or mis-sampled visit must not reach the loss."""
    game = _make_game(7)
    first = next(a for a in game["actions"] if "search_pi" in a)
    st = ts.GameState()
    ts.Engine.init_game(st, game["seed"])
    legal = set(_legal(st))
    bogus = next(i for i in range(ActionEncoder.FLAT_ACTION_SIZE) if i not in legal)
    first["search_pi"]["a"] = list(first["search_pi"]["a"]) + [bogus]
    first["search_pi"]["v"] = list(first["search_pi"]["v"]) + [1000.0]

    path = _write(tmp_path, [game])
    obs, mask, pi, dt = next(iter(WarmupDataset(path).stream_policy_transitions()))
    assert mask[bogus] == 0
    assert float(pi[bogus]) == 0.0, "an illegal action kept its visits"
    assert pi.sum() == pytest.approx(1.0, abs=1e-5), "dropping it must renormalise"


def test_the_target_is_the_searcher_not_the_played_action(tmp_path) -> None:
    """If these coincided the loader could ignore search_pi and every other test would pass."""
    game = _make_game(31337)
    path = _write(tmp_path, [game])
    obs, mask, pi, dt = next(iter(WarmupDataset(path).stream_policy_transitions()))
    played = game["actions"][0]["flat_action"]
    assert int(np.argmax(pi)) != played, (
        "the fixture's target agrees with the played action, so it cannot detect the loader "
        "falling back to behaviour cloning")


def test_batches_are_shaped_and_normalised(tmp_path) -> None:
    path = _write(tmp_path, [_make_game(s) for s in range(200, 216)])
    import torch

    seen = 0
    for b_obs, b_mask, b_pi, b_dt in WarmupDataset(path).stream_policy_batches(
            batch_size=8, device=torch.device("cpu"), shuffle_buffer_size=16):
        assert b_obs.shape[0] == b_mask.shape[0] == b_pi.shape[0] == b_dt.shape[0]
        assert b_obs.shape[1] == ts.OBS_SIZE
        assert b_pi.shape[1] == b_mask.shape[1]
        sums = b_pi.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        seen += b_obs.shape[0]
    assert seen > 0


def test_the_decision_type_travels_with_the_target(tmp_path) -> None:
    """Without it the headline agreement number cannot be split, and X4a's 93.1% over nodes
    averaging 4.1 legal actions would stay indistinguishable from 93.1% over real choices."""
    path = _write(tmp_path, [_make_game(4242)])
    seen = set()
    for _obs, _mask, _pi, dt in WarmupDataset(path).stream_policy_transitions():
        assert isinstance(dt, int)
        assert 0 <= dt <= 7, f"decision type {dt} is outside the enum"
        seen.add(dt)
    assert seen, "no targets produced"


def test_a_desynchronised_game_stops_rather_than_mislabelling(tmp_path) -> None:
    """An action the replay cannot play means every later position is a different game."""
    game = _make_game(555)
    st = ts.GameState()
    ts.Engine.init_game(st, game["seed"])
    legal = set(_legal(st))
    game["actions"][0]["flat_action"] = next(i for i in range(ActionEncoder.FLAT_ACTION_SIZE) if i not in legal)
    path = _write(tmp_path, [game])
    assert list(WarmupDataset(path).stream_policy_transitions()) == []
