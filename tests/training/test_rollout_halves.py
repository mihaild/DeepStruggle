"""P51 (--rollout-halves): the rollout in two halves of the envs, one half's network pass beside the other's engine
step. Stepping half A then half B leaves every env exactly where one full step leaves it; a halves rollout files
rows whose stored log-prob and value are the network's own on the stored observation; unsupported features refuse."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ai.models.ladder_net import create_ladder_net
from ai.training.nash_pg import NashPGTrainer
from bindings.ts_env import TsVectorizedEnv


def _net() -> torch.nn.Module:
    torch.manual_seed(0)
    return create_ladder_net(torch.device("cpu"), input_mode="grouped", aggregation="flatten", entity_dim=16,
                             entity_proj_dim=64, card_self_attention=False, cross_attention=False,
                             per_entity_heads=16, head_context=True, head_static=True, head_entities="country",
                             head_center=True, identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0,
                             num_attn_heads=4, card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
                             card_lookup_identity_dim=0, categorical_value=False)


def test_two_half_steps_are_one_full_step() -> None:
    """The env stepped as halves (step(only=A), then step(only=B)) against one stepped whole: the same observations,
    masks, rewards, dones and finished games, every step, games ending and restarting included."""
    whole = TsVectorizedEnv(num_envs=12, base_seed=21, game_seed=5)
    halves = TsVectorizedEnv(num_envs=12, base_seed=21, game_seed=5)
    obs_w, masks_w, _ = whole.reset_all()
    halves.reset_all()
    in_a = np.arange(12) < 6
    rng = np.random.default_rng(3)
    ended = 0
    for _ in range(1500):
        acts = np.array([rng.choice(np.flatnonzero(m)) for m in np.asarray(masks_w)])
        obs_w, masks_w, rew_w, done_w, info_w = whole.step(acts)
        _, _, rew_a, done_a, info_a = halves.step(acts, only=in_a)
        obs_h, masks_h, rew_b, done_b, info_b = halves.step(acts, only=~in_a)
        assert np.array_equal(obs_w, obs_h) and np.array_equal(masks_w, masks_h)
        assert np.array_equal(rew_w, np.where(in_a, rew_a, rew_b)) and np.array_equal(done_w, np.where(in_a, done_a, done_b))
        assert not (done_a & ~in_a).any() and not (done_b & in_a).any(), "a half's step finished the other half's game"
        assert np.array_equal(info_w["acting_players"], np.where(in_a, info_a["acting_players"], info_b["acting_players"]))
        assert [e["env_idx"] for e in info_w["completed_episodes"]] == sorted(
            e["env_idx"] for e in info_a["completed_episodes"] + info_b["completed_episodes"])
        ended += len(info_w["completed_episodes"])
    assert ended > 0
    assert np.array_equal(whole.ep_lengths, halves.ep_lengths)


def test_a_halves_rollout_files_the_networks_own_rows() -> None:
    net = _net()
    t = NashPGTrainer(active_net=net, env=TsVectorizedEnv(num_envs=8, base_seed=7), num_envs=8, buffer_size=96,
                      lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False, device=torch.device("cpu"),
                      rollout_halves=True, afterstate_value_coef=0.5, afterstate_frac=1.0, predeal_value_coef=0.5)
    t.collect_rollouts()
    b = t.buffer
    obs, masks = b.obs.reshape(-1, b.obs_dim), b.masks.reshape(-1, b.action_dim)
    with torch.no_grad():
        net.eval()
        logits, v_win, _ = net(obs, masks)
    lp = torch.log_softmax(logits, dim=-1).gather(1, b.actions.reshape(-1, 1).long()).squeeze(1)
    assert torch.allclose(lp, b.log_probs.reshape(-1), atol=1e-4), "a stored log-prob is not of its own row"
    assert torch.allclose(v_win.squeeze(-1), b.values_win.reshape(-1), atol=1e-4), "a stored value is not of its row"
    assert bool(masks.gather(1, b.actions.reshape(-1, 1).long()).all()), "an action outside its row's mask"
    assert t._as_rows and t._predeal_rows
    m = t.train_step()
    assert math.isfinite(m["afterstate_value_loss_after_decision"]) and math.isfinite(m["predeal_value_loss"])


def test_unsupported_features_refuse() -> None:
    t = NashPGTrainer(active_net=_net(), env=TsVectorizedEnv(num_envs=8, base_seed=7), num_envs=8, buffer_size=4,
                      lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False, device=torch.device("cpu"),
                      rollout_halves=True, critic_prior=True)
    with pytest.raises(ValueError, match="critic prior"):
        t.collect_rollouts()
