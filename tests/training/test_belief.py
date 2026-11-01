"""ai/search/belief.py (P35 3b): the C++ rows are the probe's Python rebuild, the log-likelihoods are
the slow computation's, the tracker deals hands of the right size from the hidden pool and reuses them
until the opponent moves, and a belief root plays a tournament through the game hooks."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.eval.belief_history_probe import collect, world_at
from ai.models.ladder_net import create_ladder_net
from ai.search.belief import (DECK, UNKNOWN, BeliefTracker, Decision, Now, _hypothesis_obs, card_locations,
                              device_log_probs, log_weights, player, record, sample_hands, side)
from ai.search.rollout_root import RolloutConfig, RolloutRootAgent, rollout_spec_config
from bindings.ts_env import model_obs_features
from tests.training.test_ladder_tokens import SHALLOW
from tools.lib.batch_tournament import BatchMatchRunner


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return create_ladder_net("cpu", **SHALLOW).eval()


def _jobs(model: torch.nn.Module) -> List[Tuple[Now, np.ndarray, list]]:
    samples = collect(model, torch.device("cpu"), 6, 2, 1.0, 11, set(range(6)))
    rng = np.random.default_rng(3)
    jobs = []
    for s in samples:
        now = Now(s.decider, s.locs, s.pool > 0.5, int(s.label.sum()))
        rec = record(s.turn_log, now, True)
        if rec:
            jobs.append((now, sample_hands(now, 8, rng), rec))
        if len(jobs) == 6:
            break
    assert jobs
    return jobs


def test_the_c_rows_are_the_python_rebuild() -> None:
    model = _model()
    for now, hands, rec in _jobs(model)[:3]:
        opp = 3 - now.me
        for d, left in rec[:3]:
            held = np.ascontiguousarray(hands | left.astype(np.uint8)[None, :])
            obs, masks = ts.hypothesis_rows(d.state, player(now.me), held, 0)
            for k in range(len(held)):
                w = world_at(d, opp, held[k] > 0)
                assert np.array_equal(obs[k], ts.extract_observation_features(w, player(opp), 0))
                assert np.array_equal(masks[k], np.asarray(ts.ActionMask.generate_flat_mask(w)))
                assert np.array_equal(np.asarray([int(w.get_card_location(c)) for c in range(1, 111)]),
                                      np.asarray([int(x.get_card_location(c)) for c in range(1, 111)
                                                  for x in [ts.place_hand(d.state, player(now.me), held[k])]]))


def test_the_device_rows_are_the_engines_rows_exactly() -> None:
    model = _model()
    for now, hands, rec in _jobs(model)[:3]:
        opp = 3 - now.me
        for d, left in rec[:4]:
            held = np.ascontiguousarray(hands | left.astype(np.uint8)[None, :])
            want_obs, want_masks = ts.hypothesis_rows(d.state, player(now.me), held, 0)
            pool = torch.from_numpy((d.locs == DECK) | (d.locs == UNKNOWN[opp]))[None]
            base = torch.from_numpy(np.asarray(ts.extract_observation_features(d.state, player(opp), 0)))[None]
            got = _hypothesis_obs(base, torch.from_numpy(held)[None], pool).numpy()
            assert np.array_equal(got, want_obs)
            assert np.array_equal(ts.hypothesis_masks(d.state, player(now.me), held), want_masks)


def test_log_weights_are_the_slow_computation() -> None:
    model = _model()
    jobs = _jobs(model)
    got = log_weights(model, torch.device("cpu"), jobs)
    feats = model_obs_features(model)
    for (now, hands, rec), lw in zip(jobs, got):
        want = np.zeros(len(hands))
        for d, left in rec:
            for k in range(len(hands)):
                w = world_at(d, 3 - now.me, (hands[k] > 0) | left)
                obs = torch.from_numpy(ts.extract_observation_features(w, player(3 - now.me), feats))[None]
                m = torch.from_numpy(np.asarray(ts.ActionMask.generate_flat_mask(w)))[None]
                with torch.no_grad():
                    lg = model(obs, m)[0].float().masked_fill(m == 0, float("-inf"))
                    v = float(torch.log_softmax(lg, -1)[0, d.action])
                want[k] += v if np.isfinite(v) else -1e9
        assert np.allclose(lw, want, atol=1e-4)


def _play(n_games: int, seed: int, steps: int, on_step: Any) -> None:
    """Random legal play of `n_games` games; `on_step(live, states, actions, players)` before each step."""
    runner = ts.VectorizedBatchRunner(n_games, seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        terms = runner.get_terminals()
        if all(terms):
            break
        masks = np.asarray(runner.get_action_masks())
        players = list(runner.get_decision_players())
        states = [runner.get_state(i) for i in range(n_games)]
        acts = [int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks]
        live = [i for i in range(n_games) if players[i] != 0 and not terms[i]]
        on_step(live, [states[i] for i in live], [acts[i] for i in live], [players[i] for i in live])
        runner.step_flat_all(acts, True)


def test_the_tracker_deals_hidden_hands_and_reuses_them_until_the_opponent_moves() -> None:
    tr = BeliefTracker(_model(), torch.device("cpu"), particles=32, seed=1)
    dealt = [0]

    def step(live: List[int], states: List[ts.GameState], acts: List[int], players: List[int]) -> None:
        hands = tr.worlds(live, states, 4)
        for st, p, h in zip(states, players, hands):
            if h is None:
                continue
            now = Now.of(st, side(p))
            assert h.shape == (4, 110)
            assert np.all(h.sum(1) == now.hand) and not np.any(h[:, ~now.pool])
            dealt[0] += 1
        jobs = tr.stats["jobs"]
        again = tr.worlds(live, states, 4)
        assert tr.stats["jobs"] == jobs                              # the second call reuses the weights
        assert all((a is None) == (b is None) for a, b in zip(hands, again))
        assert all(a is None or (b is not None and np.array_equal(a, b)) for a, b in zip(hands, again))   # keyed by the position
        tr.observe(live, states, acts, players)

    _play(2, 77, 400, step)
    assert dealt[0] > 50 and tr.stats["jobs"] > 10


def _logged(logs: Dict[int, List[Decision]], live: List[int], states: List[ts.GameState], acts: List[int],
            players: List[int]) -> None:
    """The Python record's log: each game's decisions this turn."""
    for g, st, a, p in zip(live, states, acts, players):
        if g in logs and logs[g] and int(logs[g][-1].state.turn) != int(st.turn):
            logs[g] = []
        logs.setdefault(g, []).append(Decision(st, side(p), a, card_locations(st)))


def test_the_c_record_is_the_python_record() -> None:
    tr = ts.BeliefTracker(16, True, 3)
    logs: Dict[int, List[Decision]] = {}
    checked = [0]

    def step(live: List[int], states: List[ts.GameState], acts: List[int], players: List[int]) -> None:
        tr.prepare(live, states)
        if tr.job_count():
            rows = tr.rows_patch(0)
            at = 0
            for j in range(tr.job_count()):
                k = tr.job_position(j)
                rec = record(logs.get(live[k], []), Now.of(states[k], side(players[k])), True)
                assert tr.job_decisions(j) == len(rec)
                for d, (dec, left) in enumerate(rec):
                    assert np.array_equal(rows[2][at + d], left.astype(np.uint8))
                    assert int(rows[4][at + d]) == dec.action
                at += len(rec)
                checked[0] += 1
            tr.set_log_likelihoods(np.zeros((6, len(rows[0]), 16), dtype=np.float32))
        for g, st, a, p in zip(live, states, acts, players):
            tr.observe(g, st, player(side(p)), a)
        _logged(logs, live, states, acts, players)

    _play(2, 91, 300, step)
    assert checked[0] > 10


def test_the_ussr_does_not_read_the_us_headline_from_its_choice() -> None:
    """The US chooses first and its card stays in its hand until the USSR has chosen. The record must leave
    that choice out, or every hand that could have made it -- those holding the card -- takes all the
    weight, and the USSR's worlds always hold the US's unrevealed headline."""
    tr = BeliefTracker(_model(), torch.device("cpu"), particles=128, seed=9)
    held: List[float] = []
    uniform: List[float] = []
    after_reveal = [0]

    def step(live: List[int], states: List[ts.GameState], acts: List[int], players: List[int]) -> None:
        hands = tr.worlds(live, states, 128)
        for st, p, h in zip(states, players, hands):
            hidden = st.current_phase == ts.Phase.HEADLINE and int(st.headline_stage) == 0
            if h is None or side(p) != 2:
                continue
            if hidden and int(st.headline_us_card):
                now = Now.of(st, 2)
                held.append(float(h[:, int(st.headline_us_card) - 1].mean()))
                uniform.append(now.hand / float(now.pool.sum()))
            elif not hidden and st.current_phase == ts.Phase.ACTION_ROUND and int(st.turn) >= 1:
                after_reveal[0] += 1
        tr.observe(live, states, acts, players)

    _play(8, 123, 1500, step)
    assert len(held) >= 12
    # With the leak this is exactly 1.0 at every position; without it, the prior's rate.
    assert np.mean(held) < 0.75 and np.mean(held) == pytest.approx(np.mean(uniform), abs=0.15)
    assert sum(x > 0.99 for x in held) <= len(held) // 4
    assert after_reveal[0] > 0


def test_the_temperature_weights_are_the_evidence_softmax() -> None:
    tr = ts.BeliefTracker(8, True, 5)
    temps = ts.BeliefTracker.temperatures()
    seen = [0]

    def step(live: List[int], states: List[ts.GameState], acts: List[int], players: List[int]) -> None:
        has = tr.prepare(live, states)
        if tr.job_count() and seen[0] == 0:
            R = sum(tr.job_decisions(j) for j in range(tr.job_count()))
            rng = np.random.default_rng(0)
            lp = np.log(rng.uniform(0.05, 1.0, size=(len(temps), R, 8))).astype(np.float32)
            tr.set_log_likelihoods(lp)
            # The first job's game has no finished turn yet only if this is its first belief; take job 0's rows.
            d0 = tr.job_decisions(0)
            logw = lp[:, :d0, :].astype(np.float64).sum(1)
            ev = np.log(np.exp(logw - logw.max(1, keepdims=True)).mean(1)) + logw.max(1)
            want = np.exp(ev - ev.max()); want /= want.sum()
            p0 = [k for k, h in enumerate(has) if h][0]
            got = np.asarray(tr.temperature_weights(p0))
            if np.isfinite(got).all() and got.sum() > 0:
                assert np.allclose(got, want, atol=1e-5)
                seen[0] += 1
        elif tr.job_count():
            R = sum(tr.job_decisions(j) for j in range(tr.job_count()))
            tr.set_log_likelihoods(np.zeros((len(temps), R, 8), dtype=np.float32))
        for g, st, a, p in zip(live, states, acts, players):
            tr.observe(g, st, player(side(p)), a)

    _play(1, 5, 200, step)
    assert seen[0] == 1


def test_a_games_worlds_do_not_depend_on_the_batch() -> None:
    model = _model()
    together = BeliefTracker(model, torch.device("cpu"), particles=16, seed=9)
    alone = BeliefTracker(model, torch.device("cpu"), particles=16, seed=9)
    compared = [0]

    def step(live: List[int], states: List[ts.GameState], acts: List[int], players: List[int]) -> None:
        a = together.worlds(live, states, 4)
        if 0 in live:
            k = live.index(0)
            b = alone.worlds([0], [states[k]], 4)[0]
            assert (a[k] is None) == (b is None)
            ak = a[k]
            if b is not None and ak is not None:
                assert np.array_equal(ak, b)
                compared[0] += 1
            alone.observe([0], [states[k]], [acts[k]], [players[k]])
        together.observe(live, states, acts, players)

    _play(3, 33, 300, step)
    assert compared[0] > 20


def test_the_trackers_likelihoods_are_the_probes() -> None:
    model = _model()
    tr = ts.BeliefTracker(8, True, 4)
    logs: Dict[int, List[Decision]] = {}
    checked = [0]

    def step(live: List[int], states: List[ts.GameState], acts: List[int], players: List[int]) -> None:
        tr.prepare(live, states)
        if tr.job_count():
            rows = tr.rows_patch(0)
            lp = device_log_probs(model, torch.device("cpu"), rows, ts.BeliefTracker.temperatures())
            at = 0
            for j in range(tr.job_count()):
                k = tr.job_position(j)
                now = Now.of(states[k], side(players[k]))
                rec = record(logs.get(live[k], []), now, True)
                lw_c = lp[0, at:at + len(rec)].double().sum(0).numpy()
                lw_py = log_weights(model, torch.device("cpu"), [(now, tr.job_hands(j), rec)])[0]
                assert np.allclose(lw_py, lw_c, atol=1e-3)
                at += len(rec)
                checked[0] += 1
            tr.set_log_likelihoods(np.ascontiguousarray(lp.numpy(), dtype=np.float32))
        for g, st, a, p in zip(live, states, acts, players):
            tr.observe(g, st, player(side(p)), a)
        _logged(logs, live, states, acts, players)

    _play(1, 12, 300, step)
    assert checked[0] >= 3



def test_the_spec_names_the_belief() -> None:
    _, cfg, label = rollout_spec_config("rollout:m.pt:4:16:4:z2:earlier:own:keyed:belief")
    assert cfg.belief == 2048 and label.endswith("-keyed-belief2048-z2")
    assert rollout_spec_config("rollout:m.pt:4:16:4:z2:earlier:own:keyed:belief512")[1].belief == 512
    assert rollout_spec_config("rollout:m.pt:4:16:4:z2:earlier:own:keyed")[1].belief == 0


def test_a_belief_root_plays_a_tournament_through_the_game_hooks() -> None:
    model = _model()
    belief = RolloutRootAgent(model, "belief", RolloutConfig(k=2, worlds=2, horizon=0, belief=16))
    plain = RolloutRootAgent(model, "plain", RolloutConfig(k=2, worlds=2, horizon=0))
    res = BatchMatchRunner.play_parallel_matchup(belief, plain, games_per_side=1, device="cpu", temperature=0.0)
    assert res["total_games"] == 2
    assert belief.belief is not None
    assert belief.belief.stats["positions"] > 20 and belief.belief.stats["jobs"] > 0
