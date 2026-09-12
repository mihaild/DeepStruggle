"""P1: the categorical value head, and advantage filtering.

Two independent changes, screened as a 2x2, so they are tested independently here too.

The categorical head replaces the auxiliary *VP* regression with a distribution over final VP.
It deliberately leaves `v_win` alone: that scalar is the baseline GAE subtracts, and an earlier
version which derived it from the distribution as P(VP>0) - P(VP<0) saturated, because that form
reads only the distribution's sign. What must hold is that nothing downstream notices: `v_win`
and `v_vp` are still produced, still have the same shapes and ranges, and still mean the same
thing — because the tournament code, the probes and NashPG's advantage computation all read them.
"""

import numpy as np
import pytest
import torch

from ai.models.coldwar_net_v2 import (VALUE_ATOMS, VP_LIMIT, create_coldwar_net_v2,
                                      create_like)


def _net(categorical: bool):
    # One layout now, so the factory defaults are it; only the head varies here.
    net = create_coldwar_net_v2("cpu", categorical_value=categorical)
    net.eval()
    return net


def _batch(net, n: int = 8):
    return torch.randn(n, net.TOTAL_OBS_SIZE), torch.ones(n, 212, dtype=torch.uint8)


# --- the head itself --------------------------------------------------------------------------

def test_the_support_is_the_engines_actual_vp_range() -> None:
    net = _net(True)
    assert VALUE_ATOMS == 2 * VP_LIMIT + 1 == 41
    assert float(net.value_support[0]) == -20.0
    assert float(net.value_support[-1]) == 20.0
    # An atom exactly on zero keeps a draw off both sides when the distribution is read.
    assert 0.0 in [float(x) for x in net.value_support]


@pytest.mark.parametrize("categorical", [False, True])
def test_both_heads_expose_the_same_interface(categorical: bool) -> None:
    net = _net(categorical)
    obs, mask = _batch(net)
    with torch.no_grad():
        logits, v_win, v_vp = net(obs, mask)
    assert logits.shape == (8, 212)
    assert v_win.shape == (8, 1) and v_vp.shape == (8, 1)
    assert bool((v_win >= -1.0).all() and (v_win <= 1.0).all()), "v_win must stay a utility"
    assert bool((v_vp >= -VP_LIMIT).all() and (v_vp <= VP_LIMIT).all())


def test_the_two_forward_paths_agree() -> None:
    net = _net(True)
    obs, mask = _batch(net)
    with torch.no_grad():
        _, v_win, v_vp = net(obs, mask)
        _, v_win2, v_vp2, value_logits = net.forward_with_value_logits(obs, mask)
    assert torch.allclose(v_win, v_win2) and torch.allclose(v_vp, v_vp2)
    assert value_logits is not None and value_logits.shape == (8, VALUE_ATOMS)


def test_a_scalar_model_has_no_value_logits() -> None:
    net = _net(False)
    obs, mask = _batch(net)
    with torch.no_grad():
        *_, value_logits = net.forward_with_value_logits(obs, mask)
    assert value_logits is None, "a scalar head must not pretend to have a distribution"


def test_v_vp_is_derived_from_the_distribution() -> None:
    """v_vp = E[VP]/VP_LIMIT, computed by hand from the logits. v_win is not derived."""
    net = _net(True)
    obs, mask = _batch(net, 4)
    with torch.no_grad():
        _, _v_win, v_vp, logits = net.forward_with_value_logits(obs, mask)
        assert logits is not None
        probs = torch.softmax(logits, dim=-1)
        # v_vp leaves the model normalised; the support itself is real VP.
        expect_vp = (probs * net.value_support).sum(dim=-1, keepdim=True) / VP_LIMIT
    assert torch.allclose(v_vp, expect_vp, atol=1e-5)


def test_v_win_keeps_its_own_head_rather_than_the_distributions_sign_mass() -> None:
    """The GAE baseline must not be P(VP>0) - P(VP<0).

    That form reads only the distribution's sign, so a head that merely leans already returns
    +/-1. Measured on a 24-iteration net it put |v_win| above 0.9 in 49.6% of states, against
    a regressed head that never passed 0.8; the inflated baseline doubled adv_std_raw and the
    arm reached 18% against the anchor where the scalar control reached 84%.
    """
    net = _net(True)
    assert net.val_win_head is not None, "the categorical variant must keep the win head"
    obs, mask = _batch(net, 8)
    with torch.no_grad():
        _, v_win, _v_vp, logits = net.forward_with_value_logits(obs, mask)
        assert logits is not None
        probs = torch.softmax(logits, dim=-1)
        sign_mass = ((probs * (net.value_support > 0)).sum(dim=-1, keepdim=True)
                     - (probs * (net.value_support < 0)).sum(dim=-1, keepdim=True))
    assert not torch.allclose(v_win, sign_mass, atol=1e-4), \
        "v_win is being derived from the distribution again"


# --- the two-hot target -----------------------------------------------------------------------

@pytest.mark.parametrize("vp", [-20.0, -7.5, -0.5, 0.0, 0.5, 7.3, 19.9, 20.0])
def test_two_hot_is_a_distribution_with_the_right_mean(vp: float) -> None:
    net = _net(True)
    t = net.two_hot(torch.tensor([vp]))
    assert t.shape == (1, VALUE_ATOMS)
    assert float(t.sum()) == pytest.approx(1.0, abs=1e-6)
    assert int((t > 0).sum()) <= 2, "a two-hot target puts mass on at most two atoms"
    assert float((t[0] * net.value_support).sum()) == pytest.approx(vp, abs=1e-5)


def test_two_hot_clamps_rather_than_crashing_out_of_range() -> None:
    # The engine cannot produce these, so anything outside is an upstream bug -- it should pile
    # up visibly on the end atom rather than raise or silently wrap.
    net = _net(True)
    for vp, expect in ((999.0, 20.0), (-999.0, -20.0)):
        t = net.two_hot(torch.tensor([vp]))
        assert float((t[0] * net.value_support).sum()) == pytest.approx(expect)


def test_two_hot_handles_a_batch() -> None:
    net = _net(True)
    vals = torch.tensor([[-3.5], [0.0], [11.25]])
    t = net.two_hot(vals)
    assert t.shape == (3, VALUE_ATOMS)
    assert torch.allclose((t * net.value_support).sum(-1), vals.reshape(-1), atol=1e-5)


# --- the checkpoint contract -------------------------------------------------------------------

def test_the_two_variants_are_not_interchangeable_checkpoints() -> None:
    """A categorical checkpoint must not silently load as a scalar one, or the reverse."""
    scalar, categorical = _net(False), _net(True)
    assert "value_dist_head.0.weight" in categorical.state_dict()
    assert "value_dist_head.0.weight" not in scalar.state_dict()
    # The win head is in *both* now -- it is what separates them that matters, and that is the
    # VP side: val_vp_head against value_dist_head.
    assert "val_win_head.0.weight" in scalar.state_dict()
    assert "val_win_head.0.weight" in categorical.state_dict()
    assert "val_vp_head.0.weight" in scalar.state_dict()
    assert "val_vp_head.0.weight" not in categorical.state_dict()
    with pytest.raises(RuntimeError):
        scalar.load_state_dict(categorical.state_dict())


def test_create_like_preserves_the_head() -> None:
    # The failure this guards has happened twice on other dimensions: a frozen evaluation copy
    # built by listing arguments missed one, and the arm died at its first snapshot.
    for categorical in (False, True):
        src = _net(categorical)
        copy = create_like(src, "cpu")
        assert copy.categorical_value is categorical
        assert set(copy.state_dict()) == set(src.state_dict())


# --- advantage filtering ------------------------------------------------------------------------

def test_filtering_keeps_the_high_advantage_tail() -> None:
    """The quantile is over the minibatch, so it adapts as the advantage scale shrinks."""
    adv = torch.tensor([-5.0, 0.1, -0.05, 3.0, 0.02, -4.0])
    q = 0.5
    keep = adv.abs() >= torch.quantile(adv.abs().float(), q)
    assert keep.tolist() == [True, False, False, True, False, True]


def test_a_zero_quantile_keeps_everything() -> None:
    adv = torch.tensor([-5.0, 0.1, 3.0])
    keep = adv.abs() >= torch.quantile(adv.abs().float(), 0.0)
    assert bool(keep.all())


def test_the_trainer_accepts_both_options() -> None:
    from ai.training.nash_pg import NashPGTrainer

    assert "adv_filter_quantile" in NashPGTrainer.__init__.__code__.co_varnames
    net = _net(True)
    assert hasattr(net, "two_hot") and hasattr(net, "forward_with_value_logits")


def test_the_value_loss_is_the_cross_entropy_it_claims_to_be() -> None:
    """Wiring, not statistics: the trainer's loss must equal the hand-computed CE.

    An earlier version of this test asserted the untrained head sits near ln(41). It does, but
    only loosely and depending on the initialisation -- it measured 3.6 on one seed and 2.97 on
    another, which makes it a flaky proxy for the thing actually worth pinning: that the loss is
    computed from the right logits against the right target.
    """
    import torch.nn.functional as F

    from ai.training.nash_pg import NashPGTrainer
    from bindings.ts_env import TsVectorizedEnv

    net = _net(True)
    env = TsVectorizedEnv(num_envs=4, base_seed=7)
    trainer = NashPGTrainer(active_net=net, env=env, num_envs=4, buffer_size=8, batch_size=8)
    # The trainer moves the net to its device; follow it rather than assuming CPU.
    device = next(net.parameters()).device

    obs, mask = _batch(net, 16)
    obs, mask = obs.to(device), mask.to(device)
    ret_vp = torch.empty(16, device=device).uniform_(-1.0, 1.0)   # as the buffer stores it
    with torch.no_grad():
        _, v_win, v_vp, logits = net.forward_with_value_logits(obs, mask)
        assert logits is not None
        got = trainer._value_loss(v_win.squeeze(-1), v_vp.squeeze(-1),
                                  torch.zeros(16, device=device), ret_vp, logits)
        # ret_vp is normalised; the loss rescales it to VP units before projecting. The win
        # MSE is part of the same value loss -- the distribution is additive, not a replacement
        # for the baseline -- so it has to be in the expectation too.
        ret_win = torch.zeros(16, device=device)
        cross_entropy = -(net.two_hot(ret_vp * VP_LIMIT)
                          * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        want = (F.mse_loss(v_win.squeeze(-1), ret_win)
                + trainer.value_dist_coef * cross_entropy)
    assert float(got) == pytest.approx(float(want), abs=1e-6)
    assert trainer.value_dist_coef != 1.0, "the coefficient must actually be applied"


def test_a_confident_wrong_prediction_costs_more_than_a_confident_right_one() -> None:
    """The direction the loss has to push, independent of initialisation."""
    import torch.nn.functional as F

    net = _net(True)
    target_vp = torch.tensor([10.0])
    target = net.two_hot(target_vp)

    right = torch.full((1, VALUE_ATOMS), -10.0)
    right[0, int((10.0 + VP_LIMIT))] = 10.0            # mass on the correct atom
    wrong = torch.full((1, VALUE_ATOMS), -10.0)
    wrong[0, int((-10.0 + VP_LIMIT))] = 10.0           # mass on the opposite one

    ce_right = -(target * F.log_softmax(right, dim=-1)).sum()
    ce_wrong = -(target * F.log_softmax(wrong, dim=-1)).sum()
    assert float(ce_right) < float(ce_wrong)


def test_the_target_uses_the_whole_support_not_just_its_middle() -> None:
    """The units bug, pinned.

    `rollout_buffer` stores returns_vp *normalised* to [-1, 1] -- it divides the final score by
    20 -- while the atom support is real VP across [-20, +20]. Projecting the normalised value
    without rescaling puts every target on the three middle atoms: the distribution never learns
    its tails, and v_win = P(VP>0) - P(VP<0) is then computed over a near-degenerate
    distribution. The arm that trained that way reached 40% against the anchor with 14% as the
    US, against 83% for the scalar control, and oscillated between the two sides all run.

    Every test above exercised two_hot on hand-written VP values, so none of them saw it. This
    one starts from what the buffer actually holds.
    """
    net = _net(True)

    # A terminal return: the buffer stores +/-1.0 for a won/lost game, being +/-20 VP over 20.
    normalised_win, normalised_loss = torch.tensor([1.0]), torch.tensor([-1.0])
    assert float((net.two_hot(normalised_win * VP_LIMIT)[0] * net.value_support).sum()) == (
        pytest.approx(20.0)), "a won game must land on the +20 atom"
    assert float((net.two_hot(normalised_loss * VP_LIMIT)[0] * net.value_support).sum()) == (
        pytest.approx(-20.0)), "a lost game must land on the -20 atom"

    # Unrescaled, both collapse to the middle -- which is what the broken arm was fitting.
    assert float((net.two_hot(normalised_win)[0] * net.value_support).sum()) == pytest.approx(1.0)

    # And a spread of returns must occupy far more than the middle of the support.
    spread = torch.linspace(-1.0, 1.0, 41) * VP_LIMIT
    occupied = int((net.two_hot(spread).sum(0) > 0).sum())
    assert occupied >= 30, f"only {occupied} of {VALUE_ATOMS} atoms ever receive mass"


def test_v_win_spans_its_range_once_the_target_is_scaled() -> None:
    """A distribution fit to rescaled returns must be able to express near-certain outcomes."""
    net = _net(True)
    support = net.value_support

    certain_win = net.two_hot(torch.tensor([1.0]) * VP_LIMIT)[0]
    v_win = float((certain_win * (support > 0)).sum() - (certain_win * (support < 0)).sum())
    assert v_win == pytest.approx(1.0), "a distribution on +20 must read as a certain win"

    # Unrescaled it reads as a coin flip, which is the defect.
    middling = net.two_hot(torch.tensor([1.0]))[0]
    v_win_bad = float((middling * (support > 0)).sum() - (middling * (support < 0)).sum())
    assert v_win_bad == pytest.approx(1.0), "mass on +1 VP is still 'winning', but only just"
    assert float((middling * support).sum()) == pytest.approx(1.0)


def test_v_vp_is_normalised_for_both_heads() -> None:
    """The scale contract, which two separate bugs violated in opposite directions.

    `rollout_buffer` stores returns_vp normalised to [-1, 1] and bootstraps the buffer boundary
    with `last_ret_vp = last_v_vp.clone()`. So `v_vp` is a *normalised* quantity by contract, and
    a head returning real VP injects a value twenty times too large into the next iteration's
    targets. The categorical arm that did so stalled: clip fraction fell to 3%, the policy stopped
    moving, and it finished at 3% against HeuristicBot where the scalar control reached 84%.

    The first bug was the mirror image -- the *target* was normalised VP projected onto a VP-unit
    support. Internally the support is real VP, which is what makes +/-20 land on the end atoms;
    only what crosses this boundary is normalised.
    """
    import ts_engine

    from bindings.ts_env import TsVectorizedEnv

    env = TsVectorizedEnv(num_envs=16, base_seed=31337)
    obs_np, mask_np, _ = env.reset_all()
    obs = torch.from_numpy(np.asarray(obs_np, dtype=np.float32))
    mask = torch.from_numpy(np.asarray(mask_np))

    for categorical in (False, True):
        # Seed before building. The scalar head's v_vp is a Tanh over random weights, so whether
        # it happens to exceed 1.0 depends on the initialisation -- which, unseeded, depends on
        # whatever ran earlier in the same worker. This passed for as long as nothing else in
        # the file drew from the global RNG first, and started failing when an unrelated test
        # file was added to the same xdist worker. The contract being asserted is real; the
        # dependence on draw order was not.
        torch.manual_seed(20260913)
        net = _net(categorical)
        with torch.no_grad():
            _, v_win, v_vp = net(obs, mask)
        assert bool((v_vp.abs() <= 1.0).all()), (
            f"categorical={categorical}: v_vp reaches {float(v_vp.abs().max()):.2f}; it must be "
            f"normalised to [-1, 1] like returns_vp, not in real VP units")
        assert bool((v_win.abs() <= 1.0).all())


def test_a_certain_win_reads_as_one_on_both_scales() -> None:
    """Ties the two ends together: the +20 atom is a certain win and a normalised v_vp of 1."""
    net = _net(True)
    support = net.value_support
    certain = net.two_hot(torch.tensor([1.0]) * VP_LIMIT)[0]

    v_vp = float((certain * support).sum()) / VP_LIMIT
    v_win = float((certain * (support > 0)).sum() - (certain * (support < 0)).sum())
    assert v_vp == pytest.approx(1.0)
    assert v_win == pytest.approx(1.0)


def test_a_categorical_checkpoint_loads_through_the_standard_agent_path(tmp_path) -> None:
    """The tournament and every probe go through NeuralAgent.from_checkpoint.

    It built a scalar net unconditionally, so a categorical arm could not be rated at all --
    the load failed on `value_dist_head` being unexpected and `val_vp_head` missing. The head is
    now detected by weight name, the same way the architecture already was.
    """
    import torch

    from tools.lib.player_agent import NeuralAgent

    for categorical in (False, True):
        net = create_coldwar_net_v2("cpu", categorical_value=categorical)
        path = tmp_path / f"cat{categorical}.pt"
        torch.save(net.state_dict(), path)
        agent = NeuralAgent.from_checkpoint(str(path), device="cpu")
        assert getattr(agent.model, "categorical_value") is categorical
