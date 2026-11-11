"""P31 (show, then let training decide). 1a, the behaviour floor: it samples the mixture it claims,
stores log pi and corrects with the weight pi / mu outside PPO's clip, so a rare action it draws can
be pushed down as well as up; off, it touches nothing. seeding forces the precursor plays the engine
expects, once per card per game, and stores them as the environment's (learner = 0)."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.eval.positions import PositionBuilder, step_to_play_mode
from ai.models.ladder_net import create_ladder_net
from ai.training import NashPGTrainer
from ai.training.show_and_decide import (EVENT_SLOT, PLAY_MODE_LO, REGION_LO, SCENARIOS,
                                         ScenarioSeeder, apply_floor, event_choice_rows,
                                         floor_eps, floor_rows)
from bindings.action_encoder import ActionEncoder as A
from bindings.ts_env import TsVectorizedEnv

SHALLOW: dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
    card_self_attention=False, cross_attention=False, per_entity_heads=16, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def _trainer(buffer: int = 16, envs: int = 8, opp_legal: bool = False, **kw: Any) -> NashPGTrainer:
    torch.manual_seed(0)
    # the env draws new games' seeds from numpy's global RNG (as in _random_play_with below): a
    # rollout that resets games would otherwise play whatever deals earlier tests left
    np.random.seed(0)
    dev = torch.device("cpu")
    model = create_ladder_net(dev, **SHALLOW, opp_legal_aux=opp_legal)
    env = TsVectorizedEnv(num_envs=envs, base_seed=123)
    return NashPGTrainer(active_net=model, env=env, num_envs=envs, buffer_size=buffer, lr=3e-4,
                         eta=0.1, ref_update_freq=500, cuda_graphs=False, device=dev, **kw)


# ------------------------------------------------------------------------------- 1a: the floor

def test_the_floor_covers_play_mode_and_in_event_choices_only() -> None:
    m = torch.zeros(7, 220, dtype=torch.bool)
    m[0, PLAY_MODE_LO:PLAY_MODE_LO + 5] = True            # play mode
    m[1, PLAY_MODE_LO + 2:PLAY_MODE_LO + 5] = True        # the deferred Ops choice: not a play mode
    m[2, REGION_LO:REGION_LO + 6] = True                  # Chernobyl's region
    m[3, A.DEFCON_VALUE_OFFSET:A.DEFCON_VALUE_OFFSET + 3] = True   # a DEFCON value
    m[4, A.BRANCH_OFFSET] = True                          # a one-option branch: nothing to floor
    m[5, A.NODE_OFFSET:A.NODE_OFFSET + 10] = True         # country targets
    m[6, 0:7] = True                                      # card choice
    assert floor_rows(m).tolist() == [True, False, True, True, False, False, False]
    assert event_choice_rows(m).tolist() == [False, False, True, True, False, False, False]


def test_the_floor_samples_the_mixture_and_returns_its_log_prob() -> None:
    g = torch.Generator().manual_seed(0)
    n, eps = 200_000, 0.3
    mask = torch.zeros(n, 220, dtype=torch.bool)
    mask[:, PLAY_MODE_LO:PLAY_MODE_LO + 5] = True
    logits = torch.full((n, 220), -1e9)
    logits[:, PLAY_MODE_LO:PLAY_MODE_LO + 5] = torch.tensor([3.0, 0.0, 1.0, -2.0, 0.5])
    lp = torch.log_softmax(logits, -1)
    pi_a = torch.multinomial(lp.exp(), 1, generator=g).squeeze(1)
    rows = torch.ones(n, dtype=torch.bool)
    rows[: n // 2] = False                                # half the rows off the floor
    acts, out_lp, weight, took = apply_floor(pi_a, lp, mask, rows, eps, generator=g)

    pi = lp[0, PLAY_MODE_LO:PLAY_MODE_LO + 5].exp()
    mu = (1 - eps) * pi + eps / 5
    on = acts[n // 2:] - PLAY_MODE_LO
    freq = torch.bincount(on, minlength=5).double() / on.numel()
    se = (mu.double() * (1 - mu.double()) / on.numel()).sqrt()
    assert bool(((freq - mu.double()).abs() < 5 * se).all()), (freq, mu)
    # off the floor: the pi sample and its log pi, untouched
    assert torch.equal(acts[: n // 2], pi_a[: n // 2])
    assert torch.allclose(out_lp[: n // 2], lp.gather(1, pi_a[:, None]).squeeze(1)[: n // 2])
    assert not bool(took[: n // 2].any())
    assert torch.all(weight[: n // 2] == 1.0)
    # on the floor: log pi of the action actually taken, and the weight pi / mu for it
    assert torch.allclose(out_lp[n // 2:], pi.log()[on], atol=1e-5)
    assert torch.allclose(weight[n // 2:], (pi / mu)[on], atol=1e-5)
    assert float(weight.max()) <= 1.0 / (1.0 - eps) + 1e-6
    assert abs(float(took[n // 2:].double().mean()) - eps) < 0.01


def test_the_floor_schedule() -> None:
    assert floor_eps(0.0, 10**9, 0, None, 0) == 0.0
    assert floor_eps(0.03, 99, 100, None, 0) == 0.0
    assert floor_eps(0.03, 100, 100, None, 0) == 0.03
    assert floor_eps(0.03, 150, 100, 200, 100) == 0.03
    assert floor_eps(0.03, 250, 100, 200, 100) == pytest.approx(0.015)
    assert floor_eps(0.03, 400, 100, 200, 100) == 0.0


def test_a_rollout_stores_log_pi_and_the_floors_behaviour_weight() -> None:
    eps = 0.5
    t = _trainer(buffer=64, play_mode_floor=eps)
    m = t.collect_rollouts()
    buf = t.buffer
    obs = buf.obs.reshape(-1, buf.obs.shape[-1]).float()
    masks = buf.masks.reshape(-1, buf.masks.shape[-1])
    acts = buf.actions.reshape(-1).long()
    with torch.no_grad():
        lp_all = torch.log_softmax(t.active_net(obs, masks)[0].float(), -1)
    lp = lp_all.gather(1, acts[:, None]).squeeze(1)
    fr = floor_rows(masks)
    assert int(fr.sum()) > 10, "too few floor rows to test anything"
    n_legal = masks.float().sum(1)
    # the stored log-prob is the policy's own everywhere; the floor shows in the weight pi / mu
    assert torch.allclose(buf.log_probs.reshape(-1).float(), lp, atol=1e-4)
    mu = (1 - eps) * lp.exp() + eps / n_legal
    want_w = torch.where(fr, lp.exp() / mu, torch.ones_like(lp))
    assert t._behaviour_w is not None
    assert torch.allclose(t._behaviour_w, want_w, atol=1e-4)
    t.train_step()                                        # and it trains
    assert True


def test_the_floor_metrics_and_validation() -> None:
    t = _trainer(play_mode_floor=0.2)
    t.train_iteration()
    m = t.train_iteration()
    assert m["floor_eps"] == pytest.approx(0.2)
    assert 0.0 < m["floor_row_frac"] <= 1.0
    with pytest.raises(ValueError):
        _trainer(play_mode_floor=1.0)
    with pytest.raises(ValueError, match="seed-scenarios"):
        _trainer(play_mode_floor=0.1, floor_scope="seeded")


def test_off_the_floor_and_seeding_leave_a_rollout_bit_identical() -> None:
    # each trainer runs straight after it is built: _trainer reseeds torch, and a rollout draws
    # from torch's global RNG, so building both first would start the second from another state
    a = _trainer()
    a.collect_rollouts()
    b = _trainer(play_mode_floor=0.0, floor_from=5, floor_anneal_from=7, floor_anneal_steps=3)
    b.collect_rollouts()
    assert torch.equal(a.buffer.actions, b.buffer.actions)
    assert torch.equal(a.buffer.log_probs, b.buffer.log_probs)


# ----------------------------------------------------------------------------- 1b: seeding

def _random_play_with(seeder: ScenarioSeeder, envs: int, steps: int, seed: int) -> List[tuple]:
    """Random legal play with the seeder applied, as the trainer applies it; returns every forced
    (env, action, decision_type) in order."""
    # the env draws new games' seeds from numpy's global RNG, so seed it: otherwise how far games
    # get -- and whether a late-war card like Chernobyl is ever held -- depends on test order
    np.random.seed(seed)
    env = TsVectorizedEnv(num_envs=envs, base_seed=seed)
    rng = np.random.default_rng(seed)
    _o, mk, _ = env.reset_all()
    seeder.draw(np.arange(envs), 0)
    out: List[tuple] = []
    for _ in range(steps):
        dp = np.asarray(env.runner.get_decision_players(), dtype=np.int8)
        acts = torch.tensor([int(rng.choice(np.flatnonzero(r))) for r in mk])
        types = [env.runner.get_state(e).ctx().decision_type for e in range(envs)]
        forced = seeder.apply(acts, np.asarray(mk), dp, env.runner, bool(env._auto_advance))
        for e in np.flatnonzero(forced):
            assert dp[e] == int(ts.Player.US)
            out.append((int(e), int(acts[e]), types[e]))
        _o, mk, _r, dones, _i = env.step(acts.numpy())
        seeder.draw(np.flatnonzero(np.asarray(dones) > 0.5), 0)
    return out


def test_seeding_forces_the_precursor_plays_the_engine_expects() -> None:
    seeder = ScenarioSeeder(["subs", "chernobyl"], 1.0, num_envs=64)
    forced = _random_play_with(seeder, envs=64, steps=3000, seed=7)
    assert seeder.forced["subs"] > 0 and seeder.forced["chernobyl"] > 0
    by_env: dict[int, List[tuple]] = {}
    for e, a, d in forced:
        by_env.setdefault(e, []).append((a, d))
    for seq in by_env.values():
        i = 0
        while i < len(seq):
            a, d = seq[i]
            assert d == ts.DecisionType.SELECT_CARD and a in (SCENARIOS["subs"].card - 1,
                                                               SCENARIOS["chernobyl"].card - 1)
            assert seq[i + 1] == (EVENT_SLOT, ts.DecisionType.SELECT_PLAY_MODE)
            if a == SCENARIOS["chernobyl"].card - 1:
                region, dt = seq[i + 2]
                assert dt == ts.DecisionType.CHOOSE_BRANCH and region >= REGION_LO
                i += 3
            else:
                i += 2


def test_an_unseeded_game_is_never_forced() -> None:
    seeder = ScenarioSeeder(["subs", "chernobyl"], 1.0, num_envs=16, start_step=10**12)
    assert _random_play_with(seeder, envs=16, steps=600, seed=3) == []
    assert seeder.games[0] == 0


def test_seeded_plays_are_stored_as_the_environments() -> None:
    t = _trainer(buffer=64, envs=16, seed_scenarios=["subs", "chernobyl"], seed_frac=1.0)
    learner_zero: List[torch.Tensor] = []
    for _ in range(12):
        t.collect_rollouts()
        z = t.buffer.learner.reshape(-1) < 0.5
        if bool(z.any()):
            learner_zero.append(t.buffer.actions.reshape(-1)[z].long())
        if sum(t.seeder.forced.values()) >= 2 and learner_zero:  # type: ignore[union-attr]
            break
    assert learner_zero, "no forced play in 12 rollouts"
    acts = torch.cat(learner_zero)
    allowed = {SCENARIOS["subs"].card - 1, SCENARIOS["chernobyl"].card - 1, EVENT_SLOT,
               *range(REGION_LO, REGION_LO + 6)}
    # without a pool every other row is the learner's, so learner = 0 is exactly the forced rows
    assert set(acts.tolist()) <= allowed
    t.train_step()


def test_seeding_validation() -> None:
    with pytest.raises(ValueError):
        ScenarioSeeder(["nope"], 0.1, 4)
    with pytest.raises(ValueError):
        _trainer(seed_frac=0.1)
    with pytest.raises(ValueError):
        _trainer(seed_scenarios=["subs"], seed_frac=0.0)


# ---------------------------------------------------------------- 1d: the opponent-legality head

def _board_obs(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    obs = torch.zeros(n, int(ts.OBS_SIZE))
    obs[:, :84 * 26] = (torch.rand(n, 84 * 26, generator=g) > 0.5).float()
    return obs


def _flags(obs: torch.Tensor, place: int, coup: int) -> torch.Tensor:
    b = obs[:84 * 26].reshape(84, 26)
    return torch.stack([b[:, place], b[:, coup]], -1) > 0.5


def test_a_position_is_labelled_with_its_opponents_legality_at_the_opponents_next_decision() -> None:
    t = _trainer(envs=2, opp_legal=True, aux_opp_legality=0.1, aux_opp_legality_frac=1.0)
    us, ussr = int(ts.Player.US), int(ts.Player.USSR)
    a = _board_obs(2, 1)                                  # step 1: env 0 US (A), env 1 USSR (B)
    t._opp_legal_record(a, np.array([us, ussr], dtype=np.int8))
    c = _board_obs(2, 2)                                  # step 2: env 0 US again (C), env 1 US (D)
    t._opp_legal_record(c, np.array([us, us], dtype=np.int8))
    e = _board_obs(2, 3)                                  # step 3: env 0 USSR (E), env 1 no decision
    t._opp_legal_record(e, np.array([ussr, 0], dtype=np.int8))
    ready = t._ol_ready
    assert len(ready) == 3
    # step 2 answers env 1's USSR position B with D's own flags (D is the US, B's opponent)
    assert torch.equal(ready[0][0].float(), a[1]) and torch.equal(ready[0][1], _flags(c[1], 19, 21))
    assert torch.equal(ready[0][2], _flags(a[1], 20, 22))     # B's own view of its opponent
    # step 3 answers env 0's two US positions A and C with E's own flags
    assert torch.equal(ready[1][0].float(), a[0]) and torch.equal(ready[1][1], _flags(e[0], 19, 21))
    assert torch.equal(ready[2][0].float(), c[0]) and torch.equal(ready[2][1], _flags(e[0], 19, 21))
    # still pending: env 0's USSR position E, env 1's US position D
    assert t._ol_npend.tolist() == [[0, 1], [1, 0]]
    t._opp_legal_game_over(np.array([1]))               # env 1's game ends: D is never answered
    assert t._ol_npend.tolist() == [[0, 1], [0, 0]]


def test_the_label_slots_are_the_movers_own_placement_legality() -> None:
    """Slots 19 / 20 are the mover's and its opponent's can_place_influence, as the engine has it."""
    env = TsVectorizedEnv(num_envs=8, base_seed=4)
    rng = np.random.default_rng(4)
    obs, mk, _ = env.reset_all()
    checked = 0
    for step in range(200):
        dp = np.asarray(env.runner.get_decision_players(), dtype=np.int8)
        if step > 40:
            for e in range(8):
                st = env.runner.get_state(e)
                if dp[e] == 0 or st.current_phase == ts.Phase.SETUP:
                    continue
                me = ts.Player(int(dp[e]))
                opp = ts.Player.USSR if me == ts.Player.US else ts.Player.US
                b = np.asarray(obs[e][:84 * 26]).reshape(84, 26)
                for i in range(84):
                    assert bool(b[i, 19] > 0.5) == ts.Operations.can_place_influence(st, me, i)
                    assert bool(b[i, 20] > 0.5) == ts.Operations.can_place_influence(st, opp, i)
                checked += 1
        obs, mk, _r, _d, _i = env.step(np.asarray([int(rng.choice(np.flatnonzero(r))) for r in mk]))
    assert checked > 100


def test_the_head_trains_and_reports(tmp_path: Any) -> None:
    from ai.training.generic_trainer import load_resume_state, save_resume_state
    a = _trainer(buffer=32)
    path = str(tmp_path / "resume.pt")
    save_resume_state(path, a.active_net, a, iteration=1, total_env_steps=128, elapsed_seconds=1.0, seed=0)
    torch.manual_seed(0)
    model = create_ladder_net(torch.device("cpu"), **SHALLOW, opp_legal_aux=True)
    b = NashPGTrainer(active_net=model, env=TsVectorizedEnv(num_envs=8, base_seed=123), num_envs=8,
                      buffer_size=32, lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False,
                      device=torch.device("cpu"), aux_opp_legality=0.5, aux_opp_legality_frac=1.0,
                      aux_opp_legality_min_batch=16)
    load_resume_state(path, b.active_net, b, seed=0)      # a branch adds the head to a saved state
    m = b.train_iteration()
    assert m["opp_legal_n"] >= 16 and np.isfinite(m["opp_legal_loss"])
    assert 0.0 <= m["opp_legal_changed_frac"] <= 1.0
    head = b.active_net.opp_legal_head[0].weight.detach().clone()  # type: ignore[union-attr]
    b.train_iteration()
    assert not torch.equal(head, b.active_net.opp_legal_head[0].weight)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="opponent-legality head"):
        _trainer(aux_opp_legality=0.1)


# ------------------------------------------------------------- 1c: counterfactual mode credit

def _queue_play_mode_decisions(cf: Any, n_decisions: int, seed: int) -> TsVectorizedEnv:
    """Random play until `n_decisions` play-mode decisions are queued (one per env at most per step)."""
    env = TsVectorizedEnv(num_envs=16, base_seed=seed)
    rng = np.random.default_rng(seed)
    _o, mk, _ = env.reset_all()
    for step in range(2000):
        dp = np.asarray(env.runner.get_decision_players(), dtype=np.int8)
        acts = np.asarray([int(rng.choice(np.flatnonzero(r))) for r in mk])
        rows = np.flatnonzero(floor_rows(torch.from_numpy(np.asarray(mk))).numpy())
        if rows.size and len(cf.queue) < n_decisions:
            cf.enqueue(step, rows[:n_decisions - len(cf.queue)], env.runner, np.asarray(mk), acts, dp)
        if len(cf.queue) >= n_decisions:
            return env
        _o, mk, _r, _d, _i = env.step(acts)
    raise AssertionError("not enough play-mode decisions")


def test_counterfactual_playouts_price_every_option_without_touching_the_game() -> None:
    from ai.training.mode_cf import ModeCounterfactual
    torch.manual_seed(0)
    net = create_ladder_net(torch.device("cpu"), **SHALLOW).eval()
    cf = ModeCounterfactual(subsample=1, playouts=1, auto_advance=True)
    env = _queue_play_mode_decisions(cf, 6, seed=11)
    before = [(int(env.runner.get_state(e).rng_state), int(env.runner.get_state(e).turn)) for e in range(16)]
    queued = list(cf.queue)
    results = cf.flush(net, torch.device("cpu"))
    assert [q for q, _ in results] == queued and cf.queue == []
    for q, qv in results:
        assert qv.shape == q.options.shape
        assert set(np.unique(qv)) <= {-1.0, 0.0, 1.0}
    assert cf.plies_played > 0
    after = [(int(env.runner.get_state(e).rng_state), int(env.runner.get_state(e).turn)) for e in range(16)]
    assert before == after                                # the live games were not stepped


def test_options_of_one_decision_share_their_dice() -> None:
    """With a (near-)deterministic network, two playouts of the same option from the same clone end
    the same way: every die and redeal comes from the cloned state's RNG."""
    from ai.training.mode_cf import ModeCounterfactual

    class Sharp(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, obs: torch.Tensor, masks: torch.Tensor) -> Any:
            out = self.inner(obs, masks)
            return (out[0] * 1e4,) + tuple(out[1:])

    torch.manual_seed(1)
    net = Sharp(create_ladder_net(torch.device("cpu"), **SHALLOW)).eval()
    cf = ModeCounterfactual(subsample=1, playouts=4, auto_advance=True)
    _queue_play_mode_decisions(cf, 4, seed=5)
    games = []
    orig = cf._play

    def spy(n: Any, d: Any, part: Any) -> np.ndarray:
        v = orig(n, d, part)
        games.append((list(part), v))
        return v

    cf._play = spy  # type: ignore[method-assign]
    cf.flush(net, torch.device("cpu"))
    per_job: dict = {}
    for part, v in games:
        for job, val in zip(part, v):
            per_job.setdefault(job, []).append(float(val))
    assert per_job and all(len(set(vs)) == 1 for vs in per_job.values())


def test_counterfactual_credit_is_filed_against_its_rows_and_trains() -> None:
    t = _trainer(buffer=48, mode_cf_coef=0.5, mode_cf_subsample=2)
    m = t.collect_rollouts()
    assert t._cf_has is not None and t._cf_adv is not None
    has = t._cf_has.nonzero().squeeze(1)
    assert has.numel() > 0
    acts = t.buffer.actions.reshape(-1).long()
    masks = t.buffer.masks.reshape(-1, t.buffer.masks.shape[-1]).bool()
    adv = t._cf_adv.float()
    assert torch.all(adv[has, acts[has]] == 0)          # centred on the option taken
    assert torch.all(adv[has][~masks[has]] == 0)        # nothing outside the legal set
    assert torch.all(t.buffer.learner.reshape(-1)[has] > 0.5)
    out = t.train_step()
    assert "mode_cf_loss" in out and np.isfinite(out["mode_cf_loss"])


def test_off_counterfactual_credit_leaves_a_rollout_bit_identical() -> None:
    a = _trainer()
    a.collect_rollouts()
    b = _trainer(mode_cf_coef=0.0, mode_cf_subsample=3, mode_cf_playouts=2)
    b.collect_rollouts()
    assert torch.equal(a.buffer.actions, b.buffer.actions)
    assert b._cf_has is None


def test_the_counterfactual_term_moves_probability_toward_the_better_option() -> None:
    from ai.training.mode_cf import mode_cf_loss
    logits = torch.zeros(3, 220)
    logits[:, :4] = torch.tensor([2.0, 0.0, -1.0, 0.0])
    logits[:, 4:] = -1e9
    logits.requires_grad_(True)
    adv = torch.zeros(3, 220)
    adv[0, 2] = 1.0                       # row 0: option 2 beat the option taken (0)
    adv[1, 1] = -1.0                      # row 1: option 1 lost to it
    has = torch.tensor([True, True, False])
    adv[2, 3] = 5.0                       # row 2 is not priced: must contribute nothing
    loss = mode_cf_loss(logits, adv, has)
    loss.backward()
    g = logits.grad
    assert g is not None
    assert float(g[0, 2]) < 0 and float(g[1, 1]) > 0     # descent raises option 2, lowers option 1
    assert torch.all(g[2] == 0)
    p = torch.softmax(logits.detach(), -1)
    want = -((p[0] * adv[0]).sum() + (p[1] * adv[1]).sum()) / 2
    assert torch.allclose(loss.detach(), want)


def test_a_floor_drawn_rare_action_gets_gradient_whatever_the_sign_of_its_advantage() -> None:
    """The bug this design replaced: with log mu stored, a rare action the floor drew sat at ratio
    pi / mu ~ 0.25, below PPO's clip, so a negative advantage was clipped to a constant and only a
    positive one moved it -- every floor draw could only raise a rare action. With the ratio on
    pi / pi_old and the weight outside the clip, both signs reach it."""
    eps, clip = 0.03, 0.2
    logits = torch.tensor([4.0, 0.0, 0.0, 0.0, -2.0], requires_grad=True)   # action 4 is rare
    a = 4
    pi_old = torch.softmax(logits.detach(), -1)
    mu_a = (1 - eps) * pi_old[a] + eps / 5

    def grad(adv: float, ratio_against_mu: bool) -> float:
        logits.grad = None
        lp = torch.log_softmax(logits, -1)[a]
        if ratio_against_mu:                       # the replaced design
            r = torch.exp(lp - mu_a.log())
            surr = -torch.min(r * adv, torch.clamp(r, 1 - clip, 1 + clip) * adv)
        else:                                      # this one
            r = torch.exp(lp - pi_old[a].log())
            w = pi_old[a] / mu_a
            surr = -torch.min(r * adv, torch.clamp(r, 1 - clip, 1 + clip) * adv) * w
        surr.backward()
        assert logits.grad is not None
        return float(logits.grad[a])

    assert grad(+1.0, True) < 0 and grad(-1.0, True) == 0.0     # one-sided: the bias
    assert grad(+1.0, False) < 0 and grad(-1.0, False) > 0      # both signs reach the action


def test_the_update_applies_the_behaviour_weight() -> None:
    t = _trainer(buffer=32, play_mode_floor=0.2)
    t.collect_rollouts()
    assert t._behaviour_w is not None
    t._behaviour_w.zero_()                 # every sample's surrogate weighted to nothing
    out = t.train_step()
    assert out["policy_loss"] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------- applicable-event forcing (owner, 2026-10-06)

def test_the_applicability_conditions() -> None:
    from ai.training.show_and_decide import event_applicable
    env = TsVectorizedEnv(num_envs=1, base_seed=1)
    env.reset_all()
    st = env.runner.get_state(0).clone()
    us, ussr = int(ts.Player.US), int(ts.Player.USSR)
    st.defcon, st.victory_points = 2, 7
    assert event_applicable("wargames", st, us) and not event_applicable("wargames", st, ussr)
    st.victory_points = 6
    assert not event_applicable("wargames", st, us)
    st.victory_points, st.defcon = -9, 2
    assert event_applicable("wargames", st, ussr)
    st.defcon = 3
    assert not event_applicable("wargames", st, ussr)
    st.us_mil_ops, st.ussr_mil_ops = 3, 2
    assert event_applicable("arms_race", st, us) and not event_applicable("arms_race", st, ussr)
    st.ussr_mil_ops = 3
    assert not event_applicable("arms_race", st, us) and not event_applicable("arms_race", st, ussr)
    st.us_space_track, st.ussr_space_track = 2, 4
    assert event_applicable("one_small_step", st, us) and not event_applicable("one_small_step", st, ussr)
    st.ussr_space_track = 2
    assert not event_applicable("one_small_step", st, us)


def test_the_forcer_plays_the_event_only_where_it_is_applicable() -> None:
    from ai.training.show_and_decide import APPLICABLE_EVENTS, ApplicableEventForcer, event_applicable
    np.random.seed(2)
    forcer = ApplicableEventForcer(["wargames", "arms_race", "one_small_step"], 1.0)
    env = TsVectorizedEnv(num_envs=64, base_seed=2)
    rng = np.random.default_rng(2)
    _o, mk, _ = env.reset_all()
    by_card = {v: k for k, v in APPLICABLE_EVENTS.items()}
    seen = 0
    for _ in range(1500):
        acts = torch.tensor([int(rng.choice(np.flatnonzero(r))) for r in mk])
        states = {e: env.runner.get_state(e).clone() for e in range(64)}
        forced = forcer.apply(acts, torch.from_numpy(np.asarray(mk)), np.ones(64, dtype=bool),
                              env.runner, 0)
        for e in np.flatnonzero(forced):
            st = states[int(e)]
            ctx = st.ctx()
            assert ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE
            name = by_card[int(ctx.pending_op_card)]
            assert event_applicable(name, st, int(ctx.decision_player))
            assert int(acts[e]) == EVENT_SLOT
            seen += 1
        _o, mk, _r, _d, _i = env.step(acts.numpy())            # the engine accepts every forced event
    assert seen > 0 and forcer.forced["arms_race"] > 0 and forcer.forced["one_small_step"] > 0
    assert sum(forcer.forced.values()) == sum(forcer.applicable.values()) == seen   # frac 1: all forced


def test_a_forced_event_is_trained_as_the_policys_own() -> None:
    # 32 envs: these are mid-war cards, and 16 short-rollout envs of an untrained net rarely reach one
    t = _trainer(buffer=64, envs=32, force_applicable_events=["arms_race", "one_small_step", "wargames"],
                 force_event_frac=1.0)
    assert t.event_forcer is not None
    for _ in range(20):
        t.collect_rollouts()
        if sum(t.event_forcer.forced.values()) > 0:
            break
    assert sum(t.event_forcer.forced.values()) > 0
    buf = t.buffer
    assert torch.all(buf.learner > 0.5)                       # still the learner's own rows
    obs = buf.obs.reshape(-1, buf.obs.shape[-1]).float()
    masks = buf.masks.reshape(-1, buf.masks.shape[-1])
    acts = buf.actions.reshape(-1).long()
    with torch.no_grad():
        lp = torch.log_softmax(t.active_net(obs, masks)[0].float(), -1).gather(1, acts[:, None]).squeeze(1)
    assert torch.allclose(buf.log_probs.reshape(-1).float(), lp, atol=1e-4)   # log pi of the forced event
    m = t.train_iteration()
    assert "force_forced_arms_race" in m


# --------------------------------------------- the per-branch head (owner, 2026-10-06)

def _branch_obs(n: int, cards: List[int], seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    obs = torch.rand(n, int(ts.OBS_SIZE), generator=g)
    co = 84 * 26
    act = obs[:, co:co + 110 * 14].view(n, 110, 14)
    act[:, :, 13] = 0.0
    for i, c in enumerate(cards):
        act[i, c - 1, 13] = 1.0                         # the card this decision is about
    return obs


def test_the_branch_head_starts_as_the_network_without_it_and_only_touches_the_branch_block() -> None:
    torch.manual_seed(0)
    plain = create_ladder_net(torch.device("cpu"), **SHALLOW).eval()
    torch.manual_seed(0)
    headed = create_ladder_net(torch.device("cpu"), **SHALLOW, branch_head=True).eval()
    missing, unexpected = headed.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("branch_head_net.") for k in missing)
    obs = _branch_obs(4, [100, 100, 46, 94])
    mask = torch.ones(4, 220, dtype=torch.bool)
    with torch.no_grad():
        assert torch.equal(plain(obs, mask)[0], headed(obs, mask)[0])     # zero-initialised
        last = headed.branch_head_net[-1]
        assert isinstance(last, torch.nn.Linear)
        torch.nn.init.normal_(last.weight, std=0.5)
        a, b = plain(obs, mask)[0], headed(obs, mask)[0]
    lo = A.BRANCH_OFFSET
    assert torch.equal(a[:, :lo], b[:, :lo])                              # nothing else moves
    assert not torch.allclose(a[:, lo:], b[:, lo:])
    d = (b - a)[:, lo:]
    assert torch.allclose(d[0], d[0]) and not torch.allclose(d[1], d[2])  # the card changes it
    obs2 = obs.clone()
    obs2[1] = obs[0]
    with torch.no_grad():
        b2 = headed(obs2, mask)[0]
    assert torch.allclose(b2[0], b2[1])                                    # same board, same card


def test_the_branch_head_is_recovered_from_the_weights_and_added_on_resume(tmp_path: Any) -> None:
    from ai.models.ladder_net import ladder_config_from_state_dict
    from ai.training.generic_trainer import load_resume_state, save_resume_state
    headed = create_ladder_net(torch.device("cpu"), **SHALLOW, branch_head=True)
    cfg = ladder_config_from_state_dict(headed.state_dict())
    assert cfg is not None and cfg["branch_head"] is True
    a = _trainer()
    path = str(tmp_path / "resume.pt")
    save_resume_state(path, a.active_net, a, iteration=1, total_env_steps=128, elapsed_seconds=1.0, seed=0)
    torch.manual_seed(0)
    model = create_ladder_net(torch.device("cpu"), **SHALLOW, branch_head=True)
    b = NashPGTrainer(active_net=model, env=TsVectorizedEnv(num_envs=8, base_seed=123), num_envs=8,
                      buffer_size=16, lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False,
                      device=torch.device("cpu"))
    load_resume_state(path, b.active_net, b, seed=0)
    b.train_iteration()


def test_the_floor_can_cover_event_choices_only() -> None:
    m = torch.zeros(3, 220, dtype=torch.bool)
    m[0, PLAY_MODE_LO:PLAY_MODE_LO + 5] = True
    m[1, A.BRANCH_OFFSET:A.BRANCH_OFFSET + 2] = True
    m[2, REGION_LO:REGION_LO + 6] = True
    assert floor_rows(m, "event_choices").tolist() == [False, True, True]
    assert floor_rows(m).tolist() == [True, True, True]


def test_environment_credit_stores_forced_events_as_the_environments() -> None:
    """--force-event-credit environment: the forced plays are learner 0 (no policy gradient), as
    P31 1b. Every env starts at the US's play-mode decision for One Small Step while behind in space,
    so the first step forces the event in all of them -- a built position, not one random play has to
    reach (that version waited on an untrained net to get into the Mid War, and CI's build missed it)."""
    t = _trainer(buffer=4, envs=4, force_applicable_events=["one_small_step"], force_event_frac=1.0,
                 force_event_credit="environment")
    assert t.event_forcer is not None
    st = step_to_play_mode(PositionBuilder(hand=[80, 26], side=ts.Player.US, turn=4, us_space=0,
                                           ussr_space=2).build(), 80)
    for i in range(4):
        t.env.runner.set_state(i, st)
    t.env.runner.refresh_all()
    t._obs_np = np.array(t.env.runner.get_observations(), dtype=np.float32)
    t._masks_np = np.array(t.env.runner.get_action_masks())
    t.collect_rollouts()
    assert t.event_forcer.forced["one_small_step"] == 4
    learner = t.buffer.learner.reshape(-1) > 0.5
    acts = t.buffer.actions.reshape(-1)
    # the first step's four rows are the forced events, stored as the environment's
    assert not bool((t.buffer.learner[0] > 0.5).any()) and bool(torch.all(t.buffer.actions[0] == EVENT_SLOT))
    # without a pool, learner = 0 marks exactly the forced plays: all of them the event at play mode
    assert torch.all(acts[~learner] == EVENT_SLOT)
    t.train_step()


def test_wargames_branch_forcing_reaches_the_branch_at_any_lead() -> None:
    from ai.training.show_and_decide import ApplicableEventForcer
    np.random.seed(5)
    forcer = ApplicableEventForcer(["wargames_branch"], 1.0)
    env = TsVectorizedEnv(num_envs=64, base_seed=5)
    rng = np.random.default_rng(5)
    _o, mk, _ = env.reset_all()
    reached, leads = 0, set()
    for _ in range(3000):
        acts = torch.tensor([int(rng.choice(np.flatnonzero(r))) for r in mk])
        forced = forcer.apply(acts, torch.from_numpy(np.asarray(mk)), np.ones(64, dtype=bool), env.runner, 0)
        for e in np.flatnonzero(forced):
            st = env.runner.get_state(int(e))
            assert int(st.defcon) == 2 and int(st.ctx().pending_op_card) == 100
            side = int(st.ctx().decision_player)
            leads.add(side * int(st.victory_points) >= 7)
            br = st.clone()
            assert ts.Engine.try_step_flat(br, EVENT_SLOT, True)
            assert br.ctx().decision_type == ts.DecisionType.CHOOSE_BRANCH and int(br.ctx().resolving_card) == 100
            assert int(br.ctx().decision_player) == side                  # the same side decides the branch
            reached += 1
        _o, mk, _r, _d, _i = env.step(acts.numpy())
        if reached >= 20 and len(leads) == 2:
            break
    assert reached > 0


def test_conditions_on_one_card_are_refused() -> None:
    from ai.training.show_and_decide import ApplicableEventForcer
    with pytest.raises(ValueError, match="twice"):
        ApplicableEventForcer(["wargames", "wargames_branch"], 0.1)


# ------------------------------------------- the play-mode head (owner, 2026-10-07)

def test_the_play_mode_head_starts_as_the_network_without_it_and_only_touches_its_block() -> None:
    torch.manual_seed(0)
    plain = create_ladder_net(torch.device("cpu"), **SHALLOW).eval()
    both = create_ladder_net(torch.device("cpu"), **SHALLOW, branch_head=True, play_mode_head=True).eval()
    missing, unexpected = both.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(("branch_head_net.", "play_mode_head_net.")) for k in missing)
    obs = _branch_obs(3, [100, 39, 80])
    mask = torch.ones(3, 220, dtype=torch.bool)
    with torch.no_grad():
        assert torch.equal(plain(obs, mask)[0], both(obs, mask)[0])        # both zero-initialised
        pm = both.play_mode_head_net[-1]
        assert isinstance(pm, torch.nn.Linear)
        torch.nn.init.normal_(pm.weight, std=0.5)
        a, b = plain(obs, mask)[0], both(obs, mask)[0]
    lo, hi = A.PLAY_MODE_OFFSET, A.PLAY_MODE_OFFSET + 5
    assert torch.equal(a[:, :lo], b[:, :lo]) and torch.equal(a[:, hi:], b[:, hi:])   # only 110..114
    d = (b - a)[:, lo:hi]
    assert not torch.allclose(d[0], d[1]) and not torch.allclose(d[1], d[2])         # the card changes it


def test_the_play_mode_head_is_recovered_from_the_weights_and_added_on_resume(tmp_path: Any) -> None:
    from ai.models.ladder_net import ladder_config_from_state_dict
    from ai.training.generic_trainer import load_resume_state, save_resume_state
    m = create_ladder_net(torch.device("cpu"), **SHALLOW, play_mode_head=True)
    cfg = ladder_config_from_state_dict(m.state_dict())
    assert cfg is not None and cfg["play_mode_head"] is True and cfg["branch_head"] is False
    a = _trainer()
    path = str(tmp_path / "resume.pt")
    save_resume_state(path, a.active_net, a, iteration=1, total_env_steps=128, elapsed_seconds=1.0, seed=0)
    torch.manual_seed(0)
    model = create_ladder_net(torch.device("cpu"), **SHALLOW, play_mode_head=True, branch_head=True)
    b = NashPGTrainer(active_net=model, env=TsVectorizedEnv(num_envs=8, base_seed=123), num_envs=8,
                      buffer_size=16, lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False,
                      device=torch.device("cpu"))
    load_resume_state(path, b.active_net, b, seed=0)
    b.train_iteration()


def test_a_scripted_scenario_forces_what_its_script_returns_and_nothing_else() -> None:
    """cmc_combo is a scripted line: wherever the script names a move for the US in a seeded game,
    that move is forced (as environment); where it returns None the row is left alone."""
    seeder = ScenarioSeeder(["cmc_combo"], 1.0, num_envs=16)
    assert [n for n, _ in seeder.scripts] == ["cmc_combo"] and seeder.scenarios == []
    calls: List[int] = []

    def every_other(st: Any, mask: Any) -> Any:              # a stand-in script
        calls.append(1)
        legal = np.flatnonzero(mask)
        return int(legal[0]) if len(calls) % 2 else None

    seeder.scripts = [("cmc_combo", every_other)]
    forced = _random_play_with(seeder, envs=16, steps=200, seed=5)
    assert forced and seeder.forced["cmc_combo"] == len(forced)
    assert all(d is not None for _, _, d in forced)
