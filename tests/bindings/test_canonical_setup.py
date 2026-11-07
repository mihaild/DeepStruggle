"""P43 phase A as a rule bit: obs_features::CANONICAL_SETUP, played per seat.

A seat playing the bit places each setup run (USSR 6; US 7; US 2) in non-decreasing
`MapData.placement_rank`, so every setup board has exactly one action sequence (the engine's own
exhaustive checks are engine/tests/test_canonical_setup.cpp). Here: the bit is a rule bit of no width
that a person does not play, the Python mask seats it for the decider, the runner plays it per side,
and a save keeps the run's floor.
"""

from typing import Any, Dict

import numpy as np
import ts_engine as ts

from ai.models.ladder_net import create_ladder_net
from ai.training.generic_trainer import add_rule_bits_to
from bindings.ts_env import TsVectorizedEnv, legal_mask, model_obs_features

CANON = ts.OBS_FEATURE_CANONICAL_SETUP
REVEAL = ts.OBS_FEATURE_HEADLINE_REVEAL
NODE = ts.FLAT_NODE
RANK = [ts.MapData.placement_rank(c) for c in range(84)]
POLAND = ts.MapData.get_country_by_name("Poland")
EAST_GERMANY = ts.MapData.get_country_by_name("East Germany")
AUSTRIA = ts.MapData.get_country_by_name("Austria")
#: A small LadderNet, as tests/bindings/test_obs_features.py builds it.
M2D: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
    card_self_attention=False, cross_attention=False, per_entity_heads=64, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def _ussr_setup() -> ts.GameState:
    s = ts.GameState()
    ts.Engine.init_game(s, 5)
    ts.Engine.auto_advance_step(s)
    assert s.current_phase == ts.Phase.SETUP and s.ctx().decision_player == ts.Player.USSR
    return s


def test_the_bit_is_a_rule_bit_of_no_width() -> None:
    assert CANON & ts.OBS_FEATURES_RULES
    assert CANON & ts.OBS_FEATURES_ALL
    assert ts.obs_size_for(CANON) == ts.obs_size_for(0)
    assert sorted(RANK) == list(range(84))
    assert RANK[POLAND] == 83 and RANK[EAST_GERMANY] == 82


def test_legal_mask_seats_the_decider() -> None:
    s = _ussr_setup()
    assert legal_mask(s, False, REVEAL | CANON)[NODE + AUSTRIA] == 1
    ts.Engine.step_flat(s, NODE + EAST_GERMANY)
    assert legal_mask(s, False, REVEAL | CANON)[NODE + AUSTRIA] == 0    # below East Germany in the order
    assert legal_mask(s, False, REVEAL | CANON)[NODE + EAST_GERMANY] == 1
    t = _ussr_setup()
    ts.Engine.step_flat(t, NODE + EAST_GERMANY)
    assert legal_mask(t, False, REVEAL)[NODE + AUSTRIA] == 1            # a seat without the bit: free order


def test_the_runner_plays_the_bit_per_side() -> None:
    """USSR seated with the bit, the US without: every USSR setup run is non-decreasing in rank, and the
    US's setups (random legal play) are not all so."""
    env = TsVectorizedEnv(num_envs=64, base_seed=900, game_seed=3)
    env.set_obs_features(np.zeros(64, dtype=np.int64), np.full(64, CANON))   # (us, ussr)
    _, masks, _ = env.reset_all()
    rng = np.random.default_rng(0)
    placed = {"USSR": [[] for _ in range(64)], "US": [[] for _ in range(64)]}
    past_setup = np.zeros(64, dtype=bool)   # only each env's first game: a random game can end and restart
    for _ in range(40):
        acts = []
        for i in range(64):
            st = env.runner.get_state(i)
            legal = np.flatnonzero(np.asarray(masks[i])[:ts.FLAT_BASE_SIZE])
            a = int(rng.choice(legal))
            past_setup[i] |= st.current_phase != ts.Phase.SETUP
            if (not past_setup[i] and st.current_phase == ts.Phase.SETUP
                    and st.ctx().decision_type == ts.DecisionType.POINT_NODE):
                side = "USSR" if st.ctx().decision_player == ts.Player.USSR else "US"
                if side == "USSR" or st.ctx().pending_ops_value == 0:
                    placed[side][i].append(a - NODE)
            acts.append(a)
        _, masks, _, _, _ = env.step(acts)
    # The decisions the USSR made (auto-advance plays the forced rest, e.g. everything after Poland).
    for run in placed["USSR"]:
        assert 1 <= len(run) <= 6
        assert all(RANK[a] <= RANK[b] for a, b in zip(run, run[1:]))
    assert any(any(RANK[a] > RANK[b] for a, b in zip(run, run[1:])) for run in placed["US"])


def test_a_save_keeps_the_runs_floor() -> None:
    s = _ussr_setup()
    s.seat_obs_features(ts.Player.USSR, CANON)
    ts.Engine.step_flat(s, NODE + EAST_GERMANY)
    back = ts.state_from_save_dict(s.to_save_dict())
    back.seat_obs_features(ts.Player.USSR, CANON)
    assert legal_mask(back, False, None)[NODE + AUSTRIA] == 0
    assert legal_mask(back, False, None)[NODE + EAST_GERMANY] == 1
    # A game without the bit never writes the field, so its save is what it always was.
    plain = _ussr_setup()
    ts.Engine.step_flat(plain, NODE + EAST_GERMANY)
    assert "canon_floor" not in str(plain.to_save_dict())


def test_rule_bits_are_added_to_a_trained_network_without_touching_its_weights() -> None:
    net = create_ladder_net("cpu", **{**M2D, "obs_features": REVEAL})
    before = {k: v.clone() for k, v in net.state_dict().items() if k != "obs_features"}
    add_rule_bits_to(net, CANON)
    assert model_obs_features(net) == REVEAL | CANON
    assert int(net.state_dict()["obs_features"]) == REVEAL | CANON
    for k, v in net.state_dict().items():
        if k != "obs_features":
            assert (v == before[k]).all()
