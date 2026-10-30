"""P40 Arm S's searched opponent (ai/training/searched_opponent.py) and the two mechanisms it rests on: the
env's hold (a held game does not move, earns nothing and is not counted), and the buffer's held rows (GAE passes
through them and no loss reads them). Then the whole loop: the searched envs' opponent rows are the environment's,
the root decides queued games in batches, and every held row repeats its env's next state."""

import copy
import json

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models.ladder_net import create_ladder_net
from ai.training import NashPGTrainer
from ai.training.opponent_pool import OpponentPool
from ai.training.rollout_buffer import RolloutBuffer
from ai.training.searched_opponent import SearchedOpponent
from bindings.ts_env import TsVectorizedEnv
from tools.lib.game_step import IllegalActionError

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False)
SPEC = "2:2:1:z2:earlier:own:keyed:aligned:box4"


#: What every engine step clears before validating its action, a refused one included (StateMachine::step):
#: the display-only record of the last roll, read by no observation.
EPHEMERAL = ("last_die_roll", "last_opp_die_roll", "last_roll", "last_event_tracks")


def _game(st: ts.GameState) -> dict:
    d = json.loads(st.to_save_json())
    for k in EPHEMERAL:
        d.pop(k, None)
    return d


def _random_actions(env: TsVectorizedEnv, rng: np.random.Generator) -> np.ndarray:
    masks = np.asarray(env.runner.get_action_masks())
    return np.array([int(rng.choice(np.flatnonzero(m))) for m in masks])


def test_a_held_game_does_not_move_at_any_kind_of_decision() -> None:
    env = TsVectorizedEnv(num_envs=4, base_seed=5)
    env.reset_all()
    rng = np.random.default_rng(5)
    kinds = set()
    for _ in range(300):
        hold = rng.random(4) < 0.4
        before = {int(e): (_game(env.runner.get_state(int(e))), int(env.runner.get_state(int(e)).rng_state),
                           np.array(env.runner.get_observations()[int(e)]), np.array(env.runner.get_action_masks()[int(e)]),
                           int(env.ep_lengths[int(e)]))
                  for e in np.flatnonzero(hold)}
        for e in before:
            kinds.add(int(env.runner.get_state(e).ctx().decision_type))
        _, _, rewards, dones, _ = env.step(_random_actions(env, rng), hold=hold)
        for e, (js, rs, obs, mask, length) in before.items():
            st = env.runner.get_state(e)
            assert _game(st) == js and int(st.rng_state) == rs
            assert np.array_equal(np.asarray(env.runner.get_observations()[e]), obs)
            assert np.array_equal(np.asarray(env.runner.get_action_masks()[e]), mask)
            assert rewards[e] == 0.0 and not dones[e] and int(env.ep_lengths[e]) == length
    assert len(kinds) >= 3                                       # card choices and other decisions held


def test_a_hold_the_engine_accepts_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    env = TsVectorizedEnv(num_envs=2, base_seed=7)
    env.reset_all()
    acts = _random_actions(env, np.random.default_rng(7))
    # Send a LEGAL action for the held env: the game would move, which the hold must refuse to hide.
    monkeypatch.setattr(env, "_refused_card_slot", lambda i, m: int(acts[i]))
    with pytest.raises(IllegalActionError, match="hold moved the game"):
        env.step(acts, hold=np.array([True, False]))


def _fill(buf: RolloutBuffer, rows: list) -> None:
    for obs, val, player, reward, done, held in rows:
        buf.add(obs=torch.full((1, buf.obs_dim), obs), masks=torch.ones((1, buf.action_dim), dtype=torch.uint8),
                actions=torch.zeros(1, dtype=torch.long), log_probs=torch.zeros(1),
                rewards=torch.tensor([reward]), dones=torch.tensor([done]), values_win=torch.tensor([val]),
                values_vp=torch.tensor([0.0]), players=torch.tensor([player], dtype=torch.int8),
                learner=torch.tensor([1.0]), turns=torch.tensor([1], dtype=torch.int8),
                held=torch.tensor([held]))


def test_gae_passes_through_held_rows_and_no_batch_reads_them() -> None:
    # One env: US, USSR, USSR, US, USSR, terminal. The held copy is inserted before row 2 (the USSR's).
    rows = [(0.1, 0.20, 1, 0.0, False), (0.2, -0.10, -1, 0.0, False), (0.3, 0.05, -1, 0.0, False),
            (0.4, 0.30, 1, 0.0, False), (0.5, -0.40, -1, 1.0, True)]
    plain = RolloutBuffer(len(rows), 1, obs_dim=4, action_dim=8, device="cpu")
    _fill(plain, [r + (False,) for r in rows])
    held = RolloutBuffer(len(rows) + 1, 1, obs_dim=4, action_dim=8, device="cpu")
    held.uses_hold = True
    _fill(held, [r + (False,) for r in rows[:2]] + [rows[2][:3] + (0.0, False, True)] + [r + (False,) for r in rows[2:]])
    zero = torch.zeros(1)
    for b in (plain, held):
        b.compute_gae(zero, zero, torch.ones(1, dtype=torch.int8))
    keep = [0, 1, 3, 4, 5]
    assert torch.allclose(held.advantages[keep, 0], plain.advantages[:, 0], atol=1e-6)
    assert torch.allclose(held.returns_win[keep, 0], plain.returns_win[:, 0], atol=1e-6)
    assert torch.allclose(held.returns_vp[keep, 0], plain.returns_vp[:, 0], atol=1e-6)
    assert float(held.advantages[2, 0]) == pytest.approx(float(held.advantages[3, 0]))
    seen = torch.cat([b[-1] for b in held.get_batches(2)])
    assert sorted(seen.tolist()) == keep                          # the held row is in no minibatch


def _trainer(envs: int = 8, buffer: int = 96) -> NashPGTrainer:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    return NashPGTrainer(active_net=create_ladder_net(dev, **CFG), env=TsVectorizedEnv(num_envs=envs, base_seed=11),
                         num_envs=envs, buffer_size=buffer, lr=3e-4, eta=0.1, ref_update_freq=500,
                         cuda_graphs=False, device=dev)


def test_the_searched_envs_play_the_learner_against_the_root_with_held_games() -> None:
    t = _trainer()
    t.opponent_pool = OpponentPool([copy.deepcopy(t.active_net)], num_envs=8, frac=0.25, seed=0,
                                   obs_features=t.obs_features)
    so = SearchedOpponent(t.active_net, SPEC, device=t.device, num_envs=8, share=0.25, pool_mixed=2,
                          refresh_every=50_000_000, batch=2, max_wait=3)
    t.searched_opp = so
    t.buffer.uses_hold = True
    assert so.envs.tolist() == [6, 7] and not t.opponent_pool.is_mixed[6:].any()
    m = t.collect_rollouts()
    assert m["searched_opp_calls"] > 0 and m["searched_opp_positions"] >= m["searched_opp_calls"]
    held = t.buffer.held
    assert int(held.sum()) == int(m["searched_opp_held_rows"]) > 0
    assert not bool(held[:, :6].any())                              # only the searched envs wait
    assert not bool((t.buffer.learner[held] > 0.5).any())           # a held row is the opponent's
    for step, env in held.nonzero().tolist():
        if step + 1 < t.buffer.buffer_size:                         # it repeats its env's next state
            assert torch.equal(t.buffer.obs[step, env], t.buffer.obs[step + 1, env])
            assert int(t.buffer.players[step, env]) == int(t.buffer.players[step + 1, env])
    # In the searched envs the learner moves on one side only per game; the opponent's rows are not its.
    opp = (t.buffer.learner[:, 6:] < 0.5)
    assert bool(opp.any()) and bool((t.buffer.learner[:, 6:] > 0.5).any())
    t.train_step()


def test_the_searched_block_must_not_overlap_the_pool() -> None:
    net = create_ladder_net(torch.device("cpu"), **CFG)
    with pytest.raises(ValueError, match="overlaps"):
        SearchedOpponent(net, SPEC, device=torch.device("cpu"), num_envs=8, share=0.5, pool_mixed=5,
                         refresh_every=50_000_000, batch=2, max_wait=3)


def test_the_snapshot_follows_the_learner_on_its_schedule() -> None:
    net = create_ladder_net(torch.device("cpu"), **CFG)
    so = SearchedOpponent(net, SPEC, device=torch.device("cpu"), num_envs=8, share=0.25, pool_mixed=2,
                          refresh_every=50_000_000, batch=2, max_wait=3)
    assert so.maybe_refresh(net, 0)                                 # the first call always copies
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    assert not so.maybe_refresh(net, 49_999_999)
    assert so.maybe_refresh(net, 50_000_000)
    for a, b in zip(net.parameters(), so.snapshot.parameters()):
        assert torch.equal(a, b)
    assert ts.Player.US is not None                                 # the engine is imported for the stubs
