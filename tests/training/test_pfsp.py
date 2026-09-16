"""Prioritised fictitious self-play for the opponent pool, and the bookkeeping it needs.

Two things are being tested and they matter for different reasons.

The **bookkeeping** -- per-opponent win rates -- is instrumentation, and it is valuable whether or
not PFSP is. Until it existed `OpponentPool.stats()` reported that the pool exists and is used,
never how the learner was faring against it, so there was no in-run signal for whether pooling
works at all. The only live external opponents saturate above 89% by 120M.

The **prioritiser** is a hypothesis, and `opponent_pool.py`'s own docstring predicts it will lose:
the pool's documented mechanism is that weak old snapshots give the learner's trailing side
winnable games, and prioritisation strips exactly those out. Hence the uniform floor, and hence
`opp_pfsp_entropy` -- if the draw collapses onto one opponent the arm is no longer testing
prioritisation, it is testing a smaller pool.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from ai.training.opponent_pool import OpponentPool


class _Net:
    """Stand-in for a frozen policy: the pool only ever calls .eval() and .parameters()."""

    def __init__(self, tag: int) -> None:
        self.tag = tag

    def eval(self) -> "_Net":
        return self

    def parameters(self) -> List[Any]:
        return []


def _pool(n_nets: int = 4, num_envs: int = 8, **kw: Any) -> OpponentPool:
    p = OpponentPool([_Net(i) for i in range(n_nets)], num_envs=num_envs, frac=1.0, seed=7, **kw)
    return p


def test_uniform_is_unchanged_when_pfsp_is_off() -> None:
    """Every existing pooled run must be reproducible, or the arm is not one factor."""
    a, b = _pool(), _pool()
    seq_a = [(a.start_iteration(), a.current.tag)[1] for _ in range(50)]
    seq_b = [(b.start_iteration(), b.current.tag)[1] for _ in range(50)]
    assert seq_a == seq_b
    assert a.sampling_probs() == pytest.approx([0.25] * 4)


def test_recording_an_outcome_does_not_change_the_uniform_draw() -> None:
    """The bookkeeping is instrumentation: with pfsp off it must not touch behaviour."""
    a, b = _pool(), _pool()
    for _ in range(20):
        a.start_iteration()
        b.start_iteration()
        a.on_episode_end(0, victory_points=20.0)   # b records nothing
    assert a.current.tag == b.current.tag
    assert a.sampling_probs() == pytest.approx(b.sampling_probs())


def test_the_learner_side_decides_whether_a_result_is_a_win() -> None:
    p = _pool(n_nets=1)
    p.start_iteration()
    p.learner_side[0] = 1                     # learner is US
    p.on_episode_end(0, victory_points=20.0)  # US won
    oid = p.ids[0]
    assert p.wins[oid] == pytest.approx(p.games[oid])

    p2 = _pool(n_nets=1)
    p2.start_iteration()
    p2.learner_side[0] = -1                   # learner is USSR
    p2.on_episode_end(0, victory_points=20.0)  # US won, so the learner lost
    oid2 = p2.ids[0]
    assert p2.wins[oid2] == pytest.approx(0.0)
    assert p2.games[oid2] > 0.0


def test_a_draw_counts_as_half() -> None:
    p = _pool(n_nets=1)
    p.start_iteration()
    p.on_episode_end(0, victory_points=0.0)
    oid = p.ids[0]
    assert p.wins[oid] == pytest.approx(0.5 * p.games[oid])


def test_an_episode_is_split_across_the_opponents_that_played_it() -> None:
    """An episode outlives an iteration, so a game is played against several snapshots.

    Crediting the whole result to whichever was current at the end would be wrong most of the
    time; the result is divided by how many iterations each opponent was current for.
    """
    p = _pool(n_nets=4, num_envs=1)
    seen: List[int] = []
    for _ in range(4):
        p.start_iteration()
        seen.append(p.current_id)
    p.learner_side[0] = 1
    p.on_episode_end(0, victory_points=20.0)   # a win, spread over `seen`
    total = sum(p.games.values())
    assert total == pytest.approx(1.0), "one episode must contribute exactly one game of mass"
    for oid in set(seen):
        assert p.games[oid] == pytest.approx(seen.count(oid) / len(seen))


def test_exposure_resets_between_episodes() -> None:
    """Otherwise a result is credited to opponents from a game that already finished."""
    p = _pool(n_nets=2, num_envs=1)
    p.start_iteration()
    first = p.current_id
    p.on_episode_end(0, victory_points=20.0)
    # force the other opponent, then end a second episode
    p.current_id = [i for i in p.ids if i != first][0]
    p._exposure[0] = {p.current_id: 1.0}
    p.on_episode_end(0, victory_points=20.0)
    assert p.games[first] == pytest.approx(1.0), "the first episode leaked into the second"


def test_pfsp_var_prefers_the_evenly_matched_opponent() -> None:
    p = _pool(n_nets=3, pfsp=True, pfsp_weighting="var", pfsp_uniform_mix=0.0, pfsp_prior=0.5)
    easy, even, hard = p.ids
    p.wins[easy], p.games[easy] = 100.0, 100.0   # learner always wins -> uninformative
    p.wins[even], p.games[even] = 50.0, 100.0    # evenly matched -> most informative
    p.wins[hard], p.games[hard] = 0.0, 100.0     # learner always loses -> also uninformative
    probs = p.sampling_probs()
    assert probs[1] > probs[0] and probs[1] > probs[2], probs


def test_pfsp_hard_prefers_the_opponent_that_beats_the_learner() -> None:
    p = _pool(n_nets=3, pfsp=True, pfsp_weighting="hard", pfsp_uniform_mix=0.0, pfsp_prior=0.5)
    easy, even, hard = p.ids
    p.wins[easy], p.games[easy] = 100.0, 100.0
    p.wins[even], p.games[even] = 50.0, 100.0
    p.wins[hard], p.games[hard] = 0.0, 100.0
    probs = p.sampling_probs()
    assert probs[2] == max(probs), probs


def test_the_uniform_floor_keeps_every_opponent_reachable() -> None:
    """A prioritiser that empties the pool has defeated the mechanism it is meant to improve."""
    p = _pool(n_nets=4, pfsp=True, pfsp_uniform_mix=0.25, pfsp_prior=0.5)
    for oid in p.ids[1:]:
        p.wins[oid], p.games[oid] = 500.0, 500.0   # beaten every time, weight -> 0
    probs = p.sampling_probs()
    assert min(probs) >= 0.25 / 4 - 1e-9, probs
    assert sum(probs) == pytest.approx(1.0)


def test_a_thin_record_does_not_capture_the_draw() -> None:
    """One lucky game must not concentrate the pool; that is what the Beta prior is for."""
    p = _pool(n_nets=4, pfsp=True, pfsp_uniform_mix=0.0, pfsp_prior=4.0)
    p.wins[p.ids[0]], p.games[p.ids[0]] = 1.0, 1.0
    probs = p.sampling_probs()
    assert max(probs) < 0.40, f"a single game moved the draw to {max(probs):.3f}"


def test_stats_expose_the_missing_instrument() -> None:
    p = _pool(n_nets=3, pfsp=True)
    p.start_iteration()
    p.on_episode_end(0, victory_points=20.0)
    s = p.stats()
    for k in ("opp_win_rate_mean", "opp_win_rate_min", "opp_win_rate_max",
              "opp_games_recorded", "opp_pfsp_entropy", "opp_pfsp_max_prob"):
        assert k in s, f"{k} missing from stats(); the pool is unobservable without it"
    assert 0.0 <= s["opp_pfsp_entropy"] <= 1.0
    assert s["opp_games_recorded"] == pytest.approx(1.0)


def test_uniform_draw_has_maximal_entropy() -> None:
    assert _pool(n_nets=8).stats()["opp_pfsp_entropy"] == pytest.approx(1.0)


def test_eviction_drops_the_evicted_opponents_statistics() -> None:
    """Stale stats keyed by a shifting index would silently rate the wrong snapshot."""
    p = _pool(n_nets=3, num_envs=2)
    p.capacity = 3
    for oid in p.ids:
        p.wins[oid], p.games[oid] = 1.0, 2.0
    before = set(p.ids)
    p.add(_Net(99), steps=1_000_000)
    assert len(p.nets) == 3
    assert len(p.ids) == 3
    assert set(p.wins) == set(p.ids), "statistics outlived their net"
    assert set(p.ids) - before, "the new net was not registered"
    assert len(p.sampling_probs()) == 3


def test_bad_configuration_is_refused() -> None:
    with pytest.raises(ValueError, match="pfsp_weighting"):
        _pool(pfsp_weighting="nonsense")
    with pytest.raises(ValueError, match="pfsp_uniform_mix"):
        _pool(pfsp_uniform_mix=1.0)
    with pytest.raises(ValueError, match="pfsp_prior"):
        _pool(pfsp_prior=0.0)


def test_every_pool_metric_reaches_the_log() -> None:
    """The metrics `stats()` returns must not be dropped on the way to training_metrics.jsonl.

    They were. `generic_trainer` forwarded a hard-coded list of three pool keys, so the
    per-opponent win rates were computed every iteration and silently discarded. `stats()` was
    correct and its unit test passed; only the forwarding was wrong, which is why a smoke run
    caught it and the suite did not. Forwarding is now by prefix, and this pins that.
    """
    import inspect

    from ai.training import generic_trainer

    src = inspect.getsource(generic_trainer.train_pipeline)
    assert 'startswith("opp_")' in src, (
        "pool metrics are no longer forwarded by prefix; a named list goes stale silently")

    # And the names stats() actually produces must all be forwardable by that prefix.
    p = _pool(n_nets=2, pfsp=True)
    assert all(k.startswith("opp_") for k in p.stats()), p.stats().keys()
