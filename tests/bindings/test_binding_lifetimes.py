"""Views and borrowed buffers outlive what used to free them (audit S-10, S-12): a BatchedSearch keeps the
arrays it writes leaves into, and a view of its own buffers taken before set_buffers stays readable; a
runner's observation, mask, pre-deal and track views own their storage, so dropping the runner or widening
its rows leaves them readable; ctx_stack_depth, which ctx() indexes the 6-frame stack with, is read-only."""
from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import ts_engine as ts


def test_set_buffers_keeps_the_arrays_alive() -> None:
    cs = ts.BatchedSearch(4)
    obs = np.zeros((4, int(cs.obs_width)), dtype=np.float32)
    masks = np.zeros((4, ts.FLAT_ACTION_SPACE_SIZE), dtype=np.uint8)
    obs_ref, masks_ref = weakref.ref(obs), weakref.ref(masks)
    cs.set_buffers(obs, masks)
    del obs, masks
    gc.collect()
    assert obs_ref() is not None and masks_ref() is not None
    st = ts.GameState()
    ts.Engine.init_game(st, 3)
    cs.reset([st], 2, [0] * 625)
    assert cs.select_leaves(0) == 1
    assert np.asarray(cs.masks())[0].any() and np.asarray(obs_ref()).any()


def test_a_view_taken_before_set_buffers_stays_readable() -> None:
    cs = ts.BatchedSearch(4)
    before = cs.observations()
    cs.set_buffers(np.zeros((4, int(cs.obs_width)), dtype=np.float32),
                   np.zeros((4, ts.FLAT_ACTION_SPACE_SIZE), dtype=np.uint8))
    assert float(np.asarray(before).sum()) == 0.0      # its own buffer, kept: not freed memory


def test_runner_views_own_their_storage() -> None:
    runner = ts.VectorizedBatchRunner(3, 5)
    runner.set_capture_predeal(True)
    runner.set_capture_tracks(True)
    obs = runner.get_observations()
    masks = runner.get_action_masks()
    want_obs, want_masks = np.array(obs), np.array(masks)
    flags, pre_us, pre_ussr = runner.get_predeal()
    tracks = runner.get_step_track_vp()
    # Widen the rows (an observation block) and reallocate every capture: the old views keep their storage.
    runner.set_obs_features([int(ts.OBS_FEATURE_OPS_BUDGET)] * 3, [int(ts.OBS_FEATURE_OPS_BUDGET)] * 3)
    runner.set_capture_predeal(True)
    runner.set_capture_tracks(True)
    assert runner.get_observations().shape[1] > obs.shape[1]
    del runner
    gc.collect()
    assert np.array_equal(np.asarray(obs), want_obs) and np.array_equal(np.asarray(masks), want_masks)
    assert np.asarray(flags).shape == (3,) and np.asarray(pre_us).shape == np.asarray(pre_ussr).shape
    assert np.asarray(tracks).shape[0] == 3


def test_ctx_stack_depth_is_read_only() -> None:
    st = ts.GameState()
    assert int(st.ctx_stack_depth) == 0
    with pytest.raises(AttributeError):
        st.ctx_stack_depth = 6   # type: ignore[misc]


def test_a_decision_context_keeps_its_state_alive() -> None:
    """BIND-1: `GameState.ctx()` returns a reference into the state; until db5f79e it had no keep-alive, so a
    context outliving a temporary state let nanobind hand the freed address to a new object (an instance
    collision). The context must keep its state alive."""
    def context() -> "ts.DecisionContext":
        st = ts.GameState()
        ts.Engine.init_game(st, 11)
        return st.ctx()

    ctxs = [context() for _ in range(64)]
    gc.collect()
    fresh = [ts.GameState() for _ in range(64)]      # would reuse the freed addresses
    assert all(int(c.decision_player) == int(ctxs[0].decision_player) for c in ctxs)
    assert len(fresh) == 64
