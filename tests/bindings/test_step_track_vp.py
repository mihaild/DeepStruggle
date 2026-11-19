"""The batch runner's per-step Event VP record (set_capture_tracks / get_step_track_vp), which the per-region exploiters
read: it must not change the games, and it must hold every Event's VP from every micro-step of the runner's step."""
from __future__ import annotations

import collections

import numpy as np
import pytest
import ts_engine as ts

N = 32
STEPS = 3000


def _runner(capture: bool) -> "ts.VectorizedBatchRunner":
    r = ts.VectorizedBatchRunner(N, 9091)
    if capture:
        r.set_capture_tracks(True)
    return r


@pytest.mark.parametrize("auto_advance", [True, False])
def test_capture_leaves_the_games_unchanged_and_records_each_micro_step(auto_advance: bool) -> None:
    on, off = _runner(True), _runner(False)
    rng = np.random.default_rng(5)
    seen = collections.Counter()
    for _ in range(STEPS):
        masks = np.asarray(on.get_action_masks())
        assert np.array_equal(masks, np.asarray(off.get_action_masks()))
        live = [i for i in range(N) if not ts.Engine.is_terminal(on.get_state(i))]
        if not live:
            break
        acts = [int(rng.choice(np.flatnonzero(masks[i]))) if i in live else 0 for i in range(N)]
        # The same step by hand, one micro-step at a time, reading the engine's own record after each.
        expect = np.zeros((N, 111), dtype=np.int64)
        for i in live:
            st = on.get_state(i)
            ts.Engine.step_flat(st, acts[i], False)
            steps = 0
            while True:
                for t in st.to_dict().get("event_tracks", []) or []:
                    expect[i, int(t["card_id"])] += int(t["vp_delta"])
                if ts.Engine.is_terminal(st) or steps >= 128:
                    break
                if auto_advance:
                    if ts.Engine.auto_advance_step(st, 1) == 0:
                        break
                elif not (st.ctx().decision_player == ts.Player.NONE
                          and st.ctx().decision_type == ts.DecisionType.ROLL_DIE):
                    break
                elif ts.Engine.auto_advance_step(st, 1) == 0:   # at a chance node: exactly that roll
                    break
                steps += 1
        on.step_flat_all(acts, auto_advance)
        off.step_flat_all(acts, auto_advance)
        got = np.asarray(on.get_step_track_vp()).astype(np.int64)
        assert np.array_equal(np.asarray(on.get_victory_points()), np.asarray(off.get_victory_points()))
        for i in live:
            assert np.array_equal(got[i], expect[i]), (i, np.flatnonzero(got[i] != expect[i]))
        for c in np.flatnonzero(got.any(axis=0)):
            seen[int(c)] += 1
    scoring = {1, 2, 3, 37, 38, 79, 81}
    assert scoring & set(seen), f"no scoring card's VP recorded in {STEPS} random steps: {dict(seen)}"


def test_reading_it_without_capture_is_an_error() -> None:
    with pytest.raises(RuntimeError):
        _runner(False).get_step_track_vp()
