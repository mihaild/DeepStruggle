"""The coverage knob must restrict what it says it restricts.

A coverage sweep is only meaningful if the knob works: a filter that silently searched everything
would produce a flat strength curve and we would conclude that coverage does not matter, which is
exactly the wrong conclusion. These tests pin the filter, the fallback, and the legality of what
comes back.

See `research/log/search_cost_and_coverage.md`.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSAgent, BatchedMCTSConfig
from bindings.action_encoder import ActionEncoder

CARD_NODES = (int(ts.DecisionType.SELECT_CARD), int(ts.DecisionType.SELECT_PLAY_MODE))


def _model():
    m = create_coldwar_net_v2("cpu")
    m.eval()
    return m


def _mixed_batch(n: int = 64, advance: int = 40):
    """A batch holding a mix of decision types, not just setup placements."""
    runner = ts.VectorizedBatchRunner(n, 31337)
    for _ in range(advance):
        masks = np.array(runner.get_action_masks(), copy=False)
        acts = [int(np.flatnonzero(masks[i])[0]) if masks[i].any() else 211 for i in range(n)]
        runner.step_flat_all(acts, True)      # settle like the searcher does
    return runner, [runner.get_state(i) for i in range(n)]


def _agent(node_filter: str, subsample: float) -> BatchedMCTSAgent:
    cfg = BatchedMCTSConfig(simulations=2, temperature=0.0, auto_advance=True,
                            node_filter=node_filter, subsample=subsample)
    return BatchedMCTSAgent(_model(), device="cpu", config=cfg, featurise_capacity=256)


def test_the_card_filter_searches_only_card_and_play_mode_nodes() -> None:
    _r, states = _mixed_batch()
    mcts = BatchedMCTS(_model(), device="cpu",
                       config=BatchedMCTSConfig(simulations=2, node_filter="card_playmode"))
    for st in states:
        dt = int(st.ctx().decision_type)
        assert mcts.should_search(st) == (dt in CARD_NODES), (
            f"filter disagreed at decision_type {dt}")


def test_the_unfiltered_searcher_searches_everything() -> None:
    _r, states = _mixed_batch()
    mcts = BatchedMCTS(_model(), device="cpu", config=BatchedMCTSConfig(simulations=2))
    assert all(mcts.should_search(st) for st in states)


def test_the_subsample_reduces_coverage_roughly_as_asked() -> None:
    """Statistical, so the bound is loose -- it is here to catch 'ignored entirely', not drift."""
    _r, states = _mixed_batch()
    mcts = BatchedMCTS(_model(), device="cpu",
                       config=BatchedMCTSConfig(simulations=2, node_filter="all", subsample=0.125))
    hits = sum(mcts.should_search(st) for _ in range(40) for st in states)
    rate = hits / (40 * len(states))
    assert 0.06 < rate < 0.22, f"subsample 0.125 produced {rate:.3f}"


def test_an_unknown_filter_is_rejected_rather_than_ignored() -> None:
    mcts = BatchedMCTS(_model(), device="cpu",
                       config=BatchedMCTSConfig(simulations=2, node_filter="nonsense"))
    _r, states = _mixed_batch(n=8)
    with pytest.raises(ValueError):
        mcts.should_search(states[0])


def test_every_returned_action_is_legal_at_every_coverage() -> None:
    """Searched or fallen back to the policy, the action must be legal in the caller's state.

    The caller settles the same way the searcher does (`_mixed_batch` steps with auto_advance),
    which is the condition `advance_root=True` requires: without it, 7 of 128 came back illegal.
    """
    _r, states = _mixed_batch()
    for node_filter, subsample in (("all", 1.0), ("card_playmode", 1.0), ("card_playmode", 0.125)):
        picks = _agent(node_filter, subsample).select_actions_batch(states)
        assert len(picks) == len(states)
        for st, a in zip(states, picks):
            mask = np.asarray(ActionEncoder.get_legal_mask(st))
            assert 0 <= a < len(mask) and mask[a], (
                f"{node_filter}/{subsample}: action {a} illegal at "
                f"decision_type {int(st.ctx().decision_type)}")


def test_a_skipped_decision_falls_back_to_this_agent_s_own_policy() -> None:
    """The unsearched half must be the agent's own greedy policy, or a sweep varies two things."""
    _r, states = _mixed_batch()
    model = _model()
    cfg = BatchedMCTSConfig(simulations=2, temperature=0.0, auto_advance=True,
                            node_filter="card_playmode", subsample=0.0)
    agent = BatchedMCTSAgent(model, device="cpu", config=cfg, featurise_capacity=256)
    picks = agent.select_actions_batch(states)      # subsample 0 -> nothing is searched
    assert getattr(agent, "searched_count", 0) == 0

    obs = np.stack([np.asarray(ts.extract_observation(
        st, st.ctx().decision_player if st.ctx().decision_player != ts.Player.NONE
        else st.phasing_player), dtype=np.float32) for st in states])
    masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
                      for st in states])
    with torch.no_grad():
        logits, _, _ = model(torch.from_numpy(obs), torch.from_numpy(masks))
    expect = [int(a) for a in torch.argmax(logits, dim=-1).numpy()]
    assert picks == expect


def test_the_decision_mix_is_what_the_cost_model_assumed() -> None:
    """The 42% figure the arm costs were computed from, re-derived from live states.

    If the engine's decision mix shifts, `research/log/search_cost_and_coverage.md` is stale.
    """
    _r, states = _mixed_batch(n=256, advance=60)
    mix = Counter(int(st.ctx().decision_type) for st in states)
    card_share = sum(v for k, v in mix.items() if k in CARD_NODES) / len(states)
    assert 0.25 < card_share < 0.60, (
        f"card/play-mode share is {card_share:.1%}; the cost model in "
        f"research/log/search_cost_and_coverage.md assumed 42%")
