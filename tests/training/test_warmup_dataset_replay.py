"""Regression tests for WarmupDataset trace replay fidelity.

Demonstration datasets store only player decisions, because they are generated through
``VectorizedBatchRunner::step_flat_all``, which resolves ROLL_DIE chance nodes internally.
A replay that does not drain those same chance nodes stalls at the first die roll,
misapplies every subsequent recorded action and diverges the RNG stream, which silently
desynchronises the hands. The symptom is an empty action mask or a demonstrated action
that is illegal in the replayed state; behavioural cloning then regresses cross-entropy
against ``-1e9`` masked logits and the warmup checkpoint is destroyed rather than trained.
"""

import gzip
import json
import os
import tempfile
from typing import Iterator, List

import numpy as np
import pytest

import ts_engine as ts
from ai.training.warmup_dataset_loader import WarmupDataset


def _write_dataset(path: str, num_games: int = 6, seed_base: int = 90210) -> None:
    """Generates traces the way the real generator does: player decisions only.

    ``VectorizedBatchRunner::step_flat_all`` applies the player action and then drains any
    ROLL_DIE chance nodes it lands on, so the recorded trace contains no chance entries.
    Reproduced here with step_flat + an explicit drain so the fixture matches that contract
    while keeping the per-game seed known.
    """
    with gzip.open(path, "wt", encoding="utf-8") as out:
        for i in range(num_games):
            seed = seed_base + i
            st = ts.GameState()
            ts.Engine.init_game(st, seed)
            actions: List[int] = []
            for _ in range(4000):
                if ts.Engine.is_terminal(st):
                    break
                mask = np.array(ts.get_flat_action_mask(st), copy=True)
                legal = np.flatnonzero(mask)
                if not len(legal):
                    break
                act = int(legal[0])
                actions.append(act)
                ts.Engine.step_flat(st, act)
                while (
                    not ts.Engine.is_terminal(st)
                    and st.ctx().decision_player == ts.Player.NONE
                    and st.ctx().decision_type == ts.DecisionType.ROLL_DIE
                ):
                    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))

            util = float(ts.Engine.get_terminal_utility(st))
            out.write(json.dumps({
                "seed": seed,
                "winner": "US" if util > 0 else ("USSR" if util < 0 else "DRAW"),
                "final_vp": int(st.victory_points),
                "actions": [{"flat_action": a} for a in actions],
            }) + "\n")


@pytest.fixture(scope="module")
def dataset_path() -> Iterator[str]:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "replay_probe.jsonl.gz")
        _write_dataset(path)
        yield path


def test_replay_never_yields_an_illegal_demonstrated_action(dataset_path: str) -> None:
    """Every yielded transition must have a non-empty mask that permits the stored action."""
    total = 0
    for _obs, mask, act, _win, _vp in WarmupDataset(dataset_path).stream_transitions():
        total += 1
        assert mask.sum() > 0, "replay reached a state with no legal action (desynchronised)"
        assert 0 <= act < mask.shape[0], f"stored action {act} out of range"
        assert mask[act] == 1, "stored action is illegal in the replayed state (desynchronised)"
    assert total > 0, "fixture produced no transitions"


def test_replay_reaches_the_end_of_each_recorded_trace(dataset_path: str) -> None:
    """A correct replay consumes whole traces; a desynchronised one is cut short."""
    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        recorded = sum(len(json.loads(line)["actions"]) for line in f)
    replayed = sum(1 for _ in WarmupDataset(dataset_path).stream_transitions())
    assert replayed == recorded, (
        f"replayed {replayed} of {recorded} recorded transitions - "
        "the trace desynchronised and was truncated"
    )


def test_targets_stay_within_the_value_head_range(dataset_path: str) -> None:
    """win/vp targets feed Tanh and linear value heads; both must stay in [-1, 1]."""
    for _obs, _mask, _act, win_ret, vp_ret in WarmupDataset(dataset_path).stream_transitions():
        assert -1.0 <= win_ret <= 1.0
        assert -1.0 <= vp_ret <= 1.0
