"""P30: the card-event auxiliary target (--aux-card-events)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, cast

import numpy as np
import torch

import ts_engine as ts

from ai.models.coldwar_net_v2 import create_like
from ai.models.ladder_net import CARD_AUX_DIM, create_ladder_net, ladder_config_from_state_dict
from ai.training.card_event_targets import (AUX_DIM, AUX_MEAN, AUX_SCALE, BREZHNEV, CARD_OFFSET, CONTAINMENT,
                                            GLOBAL_OFFSET, N_T2, OPS, PURGE_US, PURGE_USSR, SCORING, VIETNAM,
                                            aux_label, effective_ops, held_cards, label)
from ai.training.nash_pg import CardAuxRow, NashPGTrainer
from bindings.ts_env import TsVectorizedEnv

_update = cast(Any, NashPGTrainer._card_aux_update)

M2D: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
    card_self_attention=False, cross_attention=False, per_entity_heads=64, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def test_head_is_absent_by_default_and_recovered_from_the_weights() -> None:
    assert CARD_AUX_DIM == AUX_DIM == N_T2 + 12
    plain = create_ladder_net("cpu", **M2D)
    assert not plain.card_aux and not any(k.startswith("card_aux_") for k in plain.state_dict())
    net = create_ladder_net("cpu", **{**M2D, "card_aux": True})
    sd = net.state_dict()
    cfg = ladder_config_from_state_dict(sd)
    assert cfg is not None and cfg["card_aux"] is True and net.ladder_config()["card_aux"] is True
    create_like(net).load_state_dict(sd, strict=True)
    assert net.forward_card_aux(torch.zeros(3, ts.OBS_SIZE)).shape == (3, 110, CARD_AUX_DIM)
    assert len(net(torch.zeros(2, ts.OBS_SIZE), torch.ones(2, 220, dtype=torch.bool))) == 3


def _positions(n: int) -> List[tuple]:
    env = TsVectorizedEnv(num_envs=8, base_seed=5)
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(0)
    out: List[tuple] = []
    for t in range(4000):
        dp = np.asarray(env.runner.get_decision_players())
        if t % 25 == 0:
            for i in np.flatnonzero(dp != 0):
                # get_state was a view of the live environment (it is a copy now); the clone is kept
                st = env.runner.get_state(int(i)).clone()
                if int(st.turn) >= 2:
                    out.append((st, ts.Player(int(dp[i])), np.asarray(obs)[i].astype(np.float32)))
        if len(out) >= n:
            break
        acts = np.array([int(rng.choice(np.flatnonzero(m))) for m in np.asarray(masks)])
        obs, masks, *_ = env.step(acts)
    return out


def test_labels_cover_the_hand_and_the_scoring_vp_follows_the_regional_margin() -> None:
    region_of = {2: 0, 1: 1, 3: 2, 79: 3, 37: 4, 81: 5}
    xs, ys = [], []
    for st, mover, obs in _positions(60):
        lab = label(st, mover, obs, (11, 12))
        held = held_cards(obs)
        assert list(lab["cards"][:len(held)]) == held and not lab["cards"][len(held):].any()
        for j, c in enumerate(lab["cards"]):
            if lab["m4"][j] and int(c) in region_of:
                margin = float(obs[GLOBAL_OFFSET + 64 + region_of[int(c)]]) * 20.0
                # The score is capped where the game ends (20 VP), so a scoring that would pass the cap
                # pays less than the margin: from -16 a margin of -6 pays -4. The position is then not
                # one the margin predicts. The positions are sampled from random play, which sometimes
                # sits near 20 VP, and the correlation fell to 0.81 on CI once the few points here
                # included a couple of them.
                mover_vp = float(int(st.victory_points)) * (1.0 if mover == ts.Player.US else -1.0)
                if abs(mover_vp + margin) > 20.0:
                    continue
                xs.append(margin)
                ys.append(float(lab["y4"][j, 0]))
        a = aux_label(st, mover, obs, 7)
        if a is not None:
            cards, y, m = a
            assert y.shape == (12, AUX_DIM) and m.shape == (12, AUX_DIM)
            assert not y[~m].any()                    # unlabelled entries carry no target
    assert len(xs) >= 5 and set(region_of) <= set(SCORING)
    assert np.corrcoef(xs, ys)[0, 1] > 0.9           # scoring now pays the margin, from the mover's side


def test_effective_ops_is_the_engines_grant_under_every_modifier() -> None:
    """At real card plays, every combination of the five Ops flags: the replica's Ops (with the
    regional bonuses granted up front, as `Operations::grant_ops_for_card` does) must be the budget
    the engine hands the player on choosing influence."""
    env = TsVectorizedEnv(num_envs=32, base_seed=4)
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(1)
    cases: List[tuple] = []
    for _ in range(3000):
        dp = np.asarray(env.runner.get_decision_players())
        mk = np.asarray(masks)
        for i in np.flatnonzero(dp != 0):
            st = env.runner.get_state(int(i))
            c = st.ctx()
            if (int(c.decision_type) == 2 and mk[i][112] and int(c.pending_op_card) > 0
                    and int(st.current_phase) == int(ts.Phase.ACTION_ROUND)):
                cases.append((st.clone(), ts.Player(int(dp[i]))))
        if len(cases) >= 60:
            break
        obs, masks, *_ = env.step(np.array([int(rng.choice(np.flatnonzero(m))) for m in mk]))
    assert len(cases) >= 60
    bits = [CONTAINMENT, BREZHNEV, PURGE_US, PURGE_USSR, VIETNAM]
    compared, changed = 0, 0
    for st, p in cases:
        card = int(st.ctx().pending_op_card)
        for combo in range(32):
            x = st.clone()
            pe = int(x.persistent_effects)
            for k, b in enumerate(bits):
                pe = (pe | b) if combo >> k & 1 else (pe & ~b)
            x.persistent_effects = pe
            if not ts.Engine.try_step_flat(x, 112, True, False) or int(x.ctx().pending_op_card) != card:
                continue
            want = effective_ops(x, card, p, True, True)
            assert int(x.ctx().pending_ops_value) == want, (card, combo)
            compared += 1
            changed += want != int(OPS[card])
    assert compared >= 1000 and changed >= 100          # the modifiers were exercised, not just the base


def test_the_update_lowers_its_loss_on_a_fixed_batch() -> None:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **{**M2D, "card_aux": True})
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = 64
    rng = np.random.default_rng(0)
    obs = torch.randn(n, ts.OBS_SIZE).half()
    cards = np.zeros((n, 12), dtype=np.int16)
    cards[:, :5] = rng.integers(1, 111, size=(n, 5))
    y = rng.normal(size=(n, 12, AUX_DIM)).astype(np.float32)
    m = np.zeros((n, 12, AUX_DIM), dtype=bool)
    m[:, :5] = True
    y[~m] = 0.0

    def fake() -> SimpleNamespace:
        return SimpleNamespace(active_net=net, optimizer=opt, max_grad_norm=10.0, aux_card_coef=1.0,
                               aux_card_min_batch=n, aux_card_steps=2, aux_card_batch=32,
                               _card_label_seconds=0.0, _card_labelled=n,
                               _card_ready=[CardAuxRow(obs[i], cards[i], y[i], m[i]) for i in range(n)])

    first = _update(fake())
    for _ in range(40):
        last = _update(fake())
    assert first["card_aux_n"] == 2 * 32 and last["card_aux_loss"] < first["card_aux_loss"]
    small = fake()
    small._card_ready = small._card_ready[:10]
    assert "card_aux_n" not in _update(small)
    assert np.all(AUX_SCALE > 0) and AUX_MEAN.shape == (AUX_DIM,)
    assert CARD_OFFSET == 84 * 26
