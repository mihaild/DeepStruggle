"""The view spec's observation half: optional blocks appended to the base layout (owner, 2026-10-01).

The base layout must stay bit-identical whatever is appended, each block must hold what it says,
a batch mixing two feature sets must give each decider its own, and a model of one set must never
be handed another's floats.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

import ts_engine as ts

from ai.models.ladder_net import OBS_FEATURE_WIDTHS, create_ladder_net
from ai.training.card_event_targets import BREZHNEV, CONTAINMENT, PURGE_US, PURGE_USSR, effective_ops
from bindings.ts_env import TsVectorizedEnv, check_obs_width, model_obs_features
from tools.lib.batch_tournament import BatchMatchRunner
from tools.lib.game_step import drain_chance
from tools.lib.player_agent import NeuralAgent

F = int(ts.OBS_FEATURE_OPS_BUDGET)
BASE = int(ts.OBS_SIZE)

M2D: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
    card_self_attention=False, cross_attention=False, per_entity_heads=64, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def _positions(n: int, seed: int = 3) -> List[Tuple["ts.GameState", "ts.Player"]]:
    env = TsVectorizedEnv(num_envs=16, base_seed=seed)
    _, masks, _ = env.reset_all()
    rng = np.random.default_rng(seed)
    out: List[Tuple["ts.GameState", "ts.Player"]] = []
    for _ in range(4000):
        for i in range(16):
            st = env.runner.get_state(i)
            p = st.ctx().decision_player
            if int(p) != 0:
                out.append((st.clone(), p))
        if len(out) >= n:
            break
        _, masks, *_ = env.step(np.array([int(rng.choice(np.flatnonzero(m))) for m in np.asarray(masks)]))
    return out[:n]


def test_widths_agree_between_engine_and_model() -> None:
    assert ts.obs_size_for(0) == BASE
    for bit, width in OBS_FEATURE_WIDTHS.items():
        assert ts.obs_size_for(bit) == BASE + width
    assert ts.OBS_FEATURES_ALL == sum(OBS_FEATURE_WIDTHS)


def test_unknown_bits_are_refused() -> None:
    st, p = _positions(1)[0]
    unknown = 1 << int(ts.OBS_FEATURES_ALL).bit_length()       # the first bit no block or rule has taken
    with pytest.raises(ValueError):
        ts.obs_size_for(unknown)
    with pytest.raises(ValueError):
        ts.extract_observation_features(st, p, unknown)
    with pytest.raises(ValueError):
        ts.VectorizedBatchRunner(2, 1).set_obs_features([unknown, 0], [0, 0])


def test_the_base_prefix_is_bit_identical() -> None:
    for st, p in _positions(600):
        base = np.asarray(ts.extract_observation(st, p))
        assert np.array_equal(base, np.asarray(ts.extract_observation_features(st, p, 0)))
        assert np.array_equal(base, np.asarray(ts.extract_observation_features(st, p, F))[:BASE])


def test_ops_budget_is_the_engines_grant_and_modifiers() -> None:
    plays = 0
    for st, p in _positions(3000, seed=8):
        c = st.ctx()
        if int(c.decision_type) != int(ts.DecisionType.SELECT_PLAY_MODE) or int(c.pending_op_card) == 0:
            continue
        card = int(c.pending_op_card)
        for bits in (0, CONTAINMENT, BREZHNEV, PURGE_US, PURGE_USSR, CONTAINMENT | PURGE_USSR):
            x = st.clone()
            x.persistent_effects = (int(x.persistent_effects) & ~(CONTAINMENT | BREZHNEV | PURGE_US | PURGE_USSR)) | bits
            o = np.asarray(ts.extract_observation_features(x, p, F))[BASE:]
            assert o[0] * 5 == pytest.approx(effective_ops(x, card, p, True, True))
            y = x.clone()
            if ts.Engine.try_step_flat(y, 112, True, False) and int(y.ctx().pending_op_card) == card:
                assert o[0] * 5 == pytest.approx(int(y.ctx().pending_ops_value))   # what influence grants
            mod = {ts.Player.US: int(bool(bits & CONTAINMENT)) - int(bool(bits & PURGE_US)),
                   ts.Player.USSR: int(bool(bits & BREZHNEV)) - int(bool(bits & PURGE_USSR))}
            opp = ts.Player.USSR if p == ts.Player.US else ts.Player.US
            assert (o[1], o[2]) == (mod[p], mod[opp])
        plays += 1
    assert plays >= 50
    # away from a play-mode decision the value is 0
    for st, p in _positions(200, seed=9):
        if int(st.ctx().decision_type) != int(ts.DecisionType.SELECT_PLAY_MODE):
            assert np.asarray(ts.extract_observation_features(st, p, F))[BASE] == 0.0


def test_a_mixed_batch_gives_each_decider_its_own_view() -> None:
    r = ts.VectorizedBatchRunner(6, 11)
    us, ussr = [F, 0, 0, F, F, 0], [0, F, 0, F, 0, F]
    r.set_obs_features(us, ussr)
    assert r.obs_width == BASE + OBS_FEATURE_WIDTHS[F]
    obs = np.asarray(r.get_observations())
    for i in range(6):
        st = r.get_state(i)
        p = st.ctx().decision_player if int(st.ctx().decision_player) != 0 else st.phasing_player
        want = np.asarray(ts.extract_observation_features(st, p, us[i] if p == ts.Player.US else ussr[i]))
        assert np.array_equal(obs[i, :len(want)], want) and not obs[i, len(want):].any()
    r.set_obs_features([0] * 6, [0] * 6)
    assert r.obs_width == BASE


def test_the_env_keeps_its_view_across_a_reseeded_reset() -> None:
    env = TsVectorizedEnv(num_envs=4, base_seed=2)
    env.set_obs_features(F, F)
    assert env.observation_size == BASE + OBS_FEATURE_WIDTHS[F]
    obs, _, _ = env.reset_all(base_seed=99)
    assert np.asarray(obs).shape[1] == BASE + OBS_FEATURE_WIDTHS[F]


def test_a_model_names_its_view_and_refuses_another() -> None:
    plain = create_ladder_net("cpu", **M2D)
    feat = create_ladder_net("cpu", **{**M2D, "obs_features": F})
    assert model_obs_features(plain) == 0 and model_obs_features(feat) == F
    assert check_obs_width(plain) == BASE and check_obs_width(feat) == BASE + OBS_FEATURE_WIDTHS[F]
    mask = torch.ones(1, 220, dtype=torch.bool)
    with pytest.raises(ValueError):
        feat(torch.zeros(1, BASE), mask)
    with pytest.raises(ValueError):
        plain(torch.zeros(1, BASE + OBS_FEATURE_WIDTHS[F]), mask)


def test_a_match_between_two_views_plays_out() -> None:
    from tools.lib.batch_tournament import BatchMatchRunner
    from tools.lib.player_agent import NeuralAgent
    torch.manual_seed(0)
    a = NeuralAgent(model=create_ladder_net("cpu", **M2D), name="base", device="cpu")
    b = NeuralAgent(model=create_ladder_net("cpu", **{**M2D, "obs_features": F}), name="ops", device="cpu")
    res = BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=4, batch_chunk_size=8, device="cpu")
    assert res is not None


def test_the_training_probes_run_in_the_models_view() -> None:
    from ai.eval.blunders import measure_blunders_batched
    from ai.eval.decisive_probe import measure_decisive_batched
    from ai.eval.position_diagnostics import profile_self_play_batched
    m = create_ladder_net("cpu", **{**M2D, "obs_features": F}).eval()
    measure_decisive_batched(m, num_envs=2)
    measure_blunders_batched(m, num_games=2, temperature=1.0)
    profile_self_play_batched(m, num_envs=2)


R = int(ts.OBS_FEATURE_HEADLINE_REVEAL)


def _box4_headline(seed: int) -> "ts.GameState":
    """The US's headline choice with Man in Earth Orbit its own: the USSR chose first."""
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    drain_chance(st)
    st.us_space_track = 4
    rng = np.random.default_rng(seed)
    while True:
        assert not ts.Engine.is_terminal(st)
        if (st.current_phase == ts.Phase.HEADLINE and st.headline_stage == 0 and st.headline_ussr_card
                and st.ctx().decision_player == ts.Player.US):
            return st
        m = np.asarray(ts.get_flat_action_mask(st, False))
        ts.Engine.step_flat(st, int(rng.choice(np.flatnonzero(m))))
        drain_chance(st)


def test_headline_reveal_shows_the_box4_holder_the_opponents_headline_at_no_width() -> None:
    """Space Race box 4: the opponent's headline, chosen first and revealed, shows in its card row as a
    known opposing card committed to resolve -- for the box-4 holder only, only while it chooses, and
    nothing else in the observation moves."""
    assert ts.obs_size_for(R) == BASE and ts.obs_size_for(R | F) == BASE + OBS_FEATURE_WIDTHS[F]
    st = _box4_headline(5)
    card = int(st.headline_ussr_card)
    assert ts.revealed_headline(st, ts.Player.US) == card
    base = np.asarray(ts.extract_observation_features(st, ts.Player.US, 0))
    shown = np.asarray(ts.extract_observation_features(st, ts.Player.US, R))
    row = 84 * 26 + (card - 1) * 14
    assert base[row + 0] == 1.0                              # DECK_OR_HIDDEN in the base layout
    assert shown[row + 0] == 0.0 and shown[row + 2] == 1.0   # KNOWN_OPPONENT_HAND
    assert shown[row + 13] == pytest.approx(0.3)             # ACTIVE_NEXT
    differ = np.flatnonzero(base != shown)
    assert set(differ) <= {row + 0, row + 2, row + 13}
    # The opponent's own view, and every position without a revealed headline, are the base layout.
    assert np.array_equal(np.asarray(ts.extract_observation_features(st, ts.Player.USSR, R)),
                          np.asarray(ts.extract_observation_features(st, ts.Player.USSR, 0)))
    for s, p in _positions(300):
        if ts.revealed_headline(s, p) == 0:
            assert np.array_equal(np.asarray(ts.extract_observation_features(s, p, R)),
                                  np.asarray(ts.extract_observation_features(s, p, 0)))


def test_a_reveal_model_plays_a_model_without_it_each_in_its_own_view() -> None:
    """The flagships read HEADLINE_REVEAL; every earlier model does not. In one batch each side gets its own
    set: at a box-4 headline the revealing side sees the opponent's headline and the other side's view is the
    base layout -- and a whole match between the two plays out."""
    st = _box4_headline(5)
    r = ts.VectorizedBatchRunner(2, 11)
    r.set_state(0, st)
    r.set_state(1, st)
    r.set_obs_features([R, 0], [0, R])
    r.refresh_all()
    assert r.obs_width == BASE
    obs = np.asarray(r.get_observations())
    assert np.array_equal(obs[0], np.asarray(ts.extract_observation_features(st, ts.Player.US, R)))
    assert np.array_equal(obs[1], np.asarray(ts.extract_observation_features(st, ts.Player.US, 0)))
    assert not np.array_equal(obs[0], obs[1])
    torch.manual_seed(0)
    a = NeuralAgent(model=create_ladder_net("cpu", **{**M2D, "obs_features": R}), name="reveal", device="cpu")
    b = NeuralAgent(model=create_ladder_net("cpu", **M2D), name="base", device="cpu")
    assert model_obs_features(a.model) == R and check_obs_width(a.model) == BASE
    res = BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=4, batch_chunk_size=8, device="cpu")
    assert res is not None
