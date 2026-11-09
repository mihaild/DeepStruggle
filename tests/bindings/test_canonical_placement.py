"""P43 as rule bits: obs_features::CANONICAL_FIRST / CANONICAL_LAST, played per seat.

A seat playing one places every placement run -- the setup's, an Ops influence play, the verified event
runs -- in non-decreasing rank of its order (most important first, or last); a run that would strand
points releases its floor for that choice (the engine's exhaustive checks are
engine/tests/test_canonical_placement.cpp). Here: the bits are rule bits of no width that a person does
not play, the Python mask seats them for the decider, the runner plays them per side, a save keeps a
run's floor, and a trained network can be given one without touching its weights.
"""

from typing import Any, Dict

import numpy as np
import pytest
import ts_engine as ts

from ai.models.ladder_net import create_ladder_net
from ai.training.generic_trainer import add_rule_bits_to, obs_feature_bits
from bindings.ts_env import (CANONICAL_LEARNED, TsVectorizedEnv, give_run_priorities, give_state_priority,
                             legal_mask, model_obs_features, priority_ranks)
from tools.lib.batch_tournament import BatchMatchRunner
from tools.lib.player_agent import NeuralAgent

FIRST = ts.OBS_FEATURE_CANONICAL_FIRST
LAST = ts.OBS_FEATURE_CANONICAL_LAST
REVEAL = ts.OBS_FEATURE_HEADLINE_REVEAL
NODE = ts.FLAT_NODE
RANK_LAST = [ts.MapData.placement_rank(c) for c in range(84)]
RANKS = {FIRST: [83 - r for r in RANK_LAST], LAST: RANK_LAST}
EAST_GERMANY = ts.MapData.get_country_by_name("East Germany")
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


def _eastern_europe() -> list:
    return [c for c in range(84) if ts.MapData.get_country_info(c)["in_eastern_europe"]]


def test_the_bits_are_rule_bits_of_no_width_a_person_does_not_play() -> None:
    for bit in (FIRST, LAST):
        assert bit & ts.OBS_FEATURES_RULES and bit & ts.OBS_FEATURES_ALL
        assert not bit & ts.OBS_FEATURES_PERSON_RULES
        assert ts.obs_size_for(bit) == ts.obs_size_for(0)
    assert sorted(RANK_LAST) == list(range(84))
    with pytest.raises(ValueError, match="at most one"):
        obs_feature_bits(["canonical_first", "canonical_last"])
    with pytest.raises(ValueError, match="at most one"):
        obs_feature_bits(["canonical_first", "canonical_learned"])
    assert CANONICAL_LEARNED & ts.OBS_FEATURES_RULES and not CANONICAL_LEARNED & ts.OBS_FEATURES_PERSON_RULES


@pytest.mark.parametrize("order", [FIRST, LAST])
def test_legal_mask_seats_the_decider(order: int) -> None:
    s = _ussr_setup()
    legal_mask(s, False, REVEAL | order)                    # seats the decider before it places
    ts.Engine.step_flat(s, NODE + EAST_GERMANY)
    m = legal_mask(s, False, REVEAL | order)
    for c in _eastern_europe():
        assert m[NODE + c] == int(RANKS[order][c] >= RANKS[order][EAST_GERMANY])
    t = _ussr_setup()
    ts.Engine.step_flat(t, NODE + EAST_GERMANY)
    free = legal_mask(t, False, REVEAL)                      # a seat without a bit: free order
    assert all(free[NODE + c] == 1 for c in _eastern_europe())


def test_the_runner_plays_the_bits_per_side() -> None:
    """USSR seated with CANONICAL_LAST, the US without: the USSR's setup decisions are non-decreasing in its
    order, the US's (random legal play) are not all so."""
    env = TsVectorizedEnv(num_envs=64, base_seed=900, game_seed=3)
    env.set_obs_features(np.zeros(64, dtype=np.int64), np.full(64, LAST))   # (us, ussr)
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
    for run in placed["USSR"]:
        assert 1 <= len(run) <= 6
        assert all(RANK_LAST[a] <= RANK_LAST[b] for a, b in zip(run, run[1:]))
    assert any(any(RANK_LAST[a] > RANK_LAST[b] for a, b in zip(run, run[1:])) for run in placed["US"])


def test_a_save_keeps_the_runs_floor() -> None:
    s = _ussr_setup()
    s.seat_obs_features(ts.Player.USSR, LAST)
    ts.Engine.step_flat(s, NODE + EAST_GERMANY)
    back = ts.state_from_save_dict(s.to_save_dict())
    back.seat_obs_features(ts.Player.USSR, LAST)
    below = [c for c in _eastern_europe() if RANK_LAST[c] < RANK_LAST[EAST_GERMANY]]
    assert below and all(legal_mask(back, False, None)[NODE + c] == 0 for c in below)
    # A game without a bit never writes the fields, so its save is what it always was.
    plain = _ussr_setup()
    ts.Engine.step_flat(plain, NODE + EAST_GERMANY)
    assert "canon_floor" not in str(plain.to_save_dict())


def test_rule_bits_are_added_to_a_trained_network_without_touching_its_weights() -> None:
    net = create_ladder_net("cpu", **{**M2D, "obs_features": REVEAL})
    before = {k: v.clone() for k, v in net.state_dict().items() if k != "obs_features"}
    add_rule_bits_to(net, FIRST)
    assert model_obs_features(net) == REVEAL | FIRST
    assert int(net.state_dict()["obs_features"]) == REVEAL | FIRST
    for k, v in net.state_dict().items():
        if k != "obs_features":
            assert (v == before[k]).all()


def test_priority_ranks_rank_countries_by_logit() -> None:
    logits = np.zeros((1, 84))
    logits[0, 10], logits[0, 3] = 5.0, 4.0
    r = priority_ranks(logits)[0]
    assert r[10] == 0 and r[3] == 1 and sorted(r.tolist()) == list(range(84))
    assert r[0] == 2                                            # ties by country id


def test_a_learned_run_follows_the_given_priority() -> None:
    s = _ussr_setup()
    legal_mask(s, False, REVEAL | CANONICAL_LEARNED)             # seats the decider
    assert s.run_priority_wanted()
    with pytest.raises(RuntimeError):
        ts.Engine.step_flat(s, NODE + EAST_GERMANY)              # no order given: refused, loudly
    logits = np.zeros(ts.FLAT_ACTION_SPACE_SIZE, dtype=np.float32)
    prio = list(range(84))[::-1]                                 # country 83 first, then 82, ...
    for c, p in enumerate(prio):
        logits[NODE + c] = -p
    assert give_state_priority(s, logits)
    assert not s.run_priority_wanted()
    ts.Engine.step_flat(s, NODE + EAST_GERMANY)
    m = legal_mask(s, False, None)
    for c in _eastern_europe():
        assert m[NODE + c] == int(c <= EAST_GERMANY)             # priority falls with the id here


def test_the_runner_takes_run_orders_from_logits() -> None:
    env = TsVectorizedEnv(num_envs=8, base_seed=901, game_seed=4)
    env.set_obs_features(np.full(8, CANONICAL_LEARNED), np.full(8, CANONICAL_LEARNED))
    assert env.any_learned_order
    _, masks, _ = env.reset_all()
    wanted = np.asarray(env.runner.priority_wanted(), dtype=bool)
    assert wanted.all()                                          # every env opens the USSR setup run
    logits = np.random.default_rng(0).normal(size=(8, np.asarray(masks).shape[1])).astype(np.float32)
    assert give_run_priorities(env.runner, logits) == 8
    assert not np.asarray(env.runner.priority_wanted(), dtype=bool).any()
    acts = [int(np.flatnonzero(np.asarray(masks[i])[:ts.FLAT_BASE_SIZE])[0]) for i in range(8)]
    env.step(acts)                                               # accepted: each run had its order


@pytest.mark.parametrize("packed", [False, True])
def test_a_learned_order_network_plays_through_the_match_runner(packed: bool) -> None:
    """Both match paths give a V3 network its run orders: neither may hit a refused placement."""
    a = NeuralAgent(model=create_ladder_net("cpu", **{**M2D, "obs_features": REVEAL | CANONICAL_LEARNED}), name="v3",
                    device="cpu")
    b = NeuralAgent(model=create_ladder_net("cpu", **{**M2D, "obs_features": REVEAL}), name="free", device="cpu")
    if packed:
        res = BatchMatchRunner.play_packed_matchups([(a, b), (b, a)], games_per_side=2, device="cpu", batch_chunk_size=8)
    else:
        res = BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=2, batch_chunk_size=8, device="cpu")
    assert res is not None
